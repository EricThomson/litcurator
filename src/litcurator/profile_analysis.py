"""
profile_analysis.py -- synthesize flag patterns into the pattern memory.

The offline learning path. It reads the numeric flags (the residuals between the
judge and the user) and turns them into tracked PATTERNS -- recurring taste-gaps the
human curates in the workbench and hand-authors into the profile. It SURFACES and
REMEMBERS; the human authors every word of the actual edit.

The design separates RECORD (permissive: track everything real, into the append-only
pattern memory) from SELECT (restrictive: which patterns to act on this round, decided
as a ranking + the human, NOT by discarding). The old middle stage fused the two and
dumped everything it did not act on into a free-text "Considered and cut" line that no
code read -- so real-but-not-now patterns, and the whole judge-not-applying (prompt-fix)
signal, were silently lost and recurrence could never accumulate. Now nothing is
dumped; every candidate gets a choice and a home.

Two LLM stages (the Tao of litcurator: generate cheap-and-broad, then decide):
  Step 1 (cluster, Sonnet): RECALL -- surface every candidate preference pattern from
    the UNASSIGNED (not-yet-patterned) flags.
  Step 2 (consolidate, Sonnet, forced tool-use): for EACH candidate emit a structured
    choice -- new / merge into an open pattern / recurs against a closed pattern /
    hold -- plus direction, an act-now-vs-defer priority HINT (where the
    false-negative bias lives, governing the hint only), and its supporting papers.
Then RECORD (pure code) writes the choices into patterns / pattern_flags /
pattern_events. The consolidate step is shown the open patterns + closed patterns WITH ids, so
it captures the cross-round match the old pipeline already made and threw away: a
recurring candidate merges into its existing pattern (a 'carried' event, so recurrence
accumulates) instead of minting a duplicate.

Goodhart guard: this NEVER re-runs the judge on the flags to "validate" an edit --
that is exactly how v1 taught the LLM to game the score. It surfaces evidence only;
if you ever validate an edit, do it on a held-out month, not the flag set.

Reads UNASSIGNED flags from db_interface.get_flags(exclude_assigned=True); reads the active
profile from profile_interface. Streams the recall to console and saves a dated
markdown report to ~/.litcurator/suggestions/. See starry-brewing-horizon.md.
"""

import os
import sys

import anthropic
from dotenv import load_dotenv

from litcurator import db_interface, profile_interface
from litcurator.config import DATA_DIR, USER_JOURNAL_RATINGS

load_dotenv()

# Both stages are Sonnet. Recall is Sonnet's strength (crisp, broad generation).
# Consolidate is structured tagging with NO prose authorship, so Opus's documented
# prose-padding liability does not apply and Sonnet is ~5x cheaper; keep Opus as a
# drop-in fallback only if the duplicate rate on real data proves poor.
DEFAULT_CLUSTER_MODEL = "claude-sonnet-4-6"
DEFAULT_CONSOLIDATE_MODEL = "claude-sonnet-4-6"

# Approximate API prices, ($/M input, $/M output). Update if pricing changes.
MODEL_COSTS = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

SUGGESTIONS_DIR = DATA_DIR / "suggestions"

MIN_FLAGS = 10
DELTA_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Prompts (validated in v1/v3 -- kept verbatim)
# ---------------------------------------------------------------------------

CLUSTER_PROMPT = """
You are analyzing a researcher's numeric flags on a neuroscience literature curation system.

The system scores papers 0-1 against the researcher's seed profile. The researcher has reviewed
papers and given each their own score (your_score). delta = your_score - judge_score:
  - Negative delta: judge scored too high -- the seed is over-triggering on something
  - Positive delta: judge scored too low -- the seed is missing coverage for something
  - Near zero: agreement

The JUDGE RATIONALE shows what the seed caused the judge to say -- the primary diagnostic.
Over-scored papers show the judge citing interests that do not really apply; under-scored papers
show it missing interests absent from the seed.

YOUR JOB HERE IS RECALL. Surface every distinct candidate preference pattern the flags reveal.
A later stage records and ranks these, so do not self-censor -- but a "pattern" is a regularity
across MULTIPLE papers, not one paper's quirk. Merge exact duplicates; otherwise be thorough.

For each candidate pattern:
- A short name (3-6 words)
- The underlying preference signal (1-2 sentences), read from the judge rationales, not keyword overlap
- Supporting papers by number
- How large and pervasive the cluster is: how many papers, and roughly what share of the flags it
  spans. Frequency matters in its own right -- a pattern that recurs across many papers is important
  even when each delta is small. (Elapsed time / how many months it spans is NOT a factor; reason
  about the cluster, not the calendar.)
- Whether any supporting papers carry an explicit USER NOTE, and what it says. The user writes few
  notes and is selective, so a note is a deliberate, high-confidence taste signal -- stronger than
  the delta number alone. Flag note-backed patterns clearly.
- Direction: seed MISSING coverage (positive deltas), OVER-TRIGGERING (negative deltas), or existing
  language needs SHARPENING -- plus the rough delta magnitude (how wrong, and which way)

Articles are referenced by number; all references are stripped downstream so the researcher is never
anchored to specific papers. Plain text.
""".strip()


# ---------------------------------------------------------------------------
# Formatting helpers for the prompt
# ---------------------------------------------------------------------------

def _format_journal_ratings():
    """Group USER_JOURNAL_RATINGS by value for the cluster prompt.
    The LLM should use these, not its own priors about journal prestige."""
    from collections import defaultdict
    groups = defaultdict(list)
    for journal, rating in USER_JOURNAL_RATINGS.items():
        groups[rating].append(journal)
    lines = ["## User journal quality ratings (use these, not your own priors about journal prestige)"]
    for rating in sorted(groups, reverse=True):
        lines.append(f"  {rating:+.2f}: {', '.join(groups[rating])}")
    return "\n".join(lines)

def _format_papers(flags):
    """Render the flags as the numbered papers block, AND return the flags in the
    SAME order the numbers follow -- so paper number N in the LLM output maps back to
    ordered[N-1] (and thus its flag id) for provenance. Numbering is deterministic:
    over-scored (delta asc), then under-scored (delta desc), then roughly-agreed."""
    neg = sorted([f for f in flags if f["delta"] < -DELTA_THRESHOLD], key=lambda f: f["delta"])
    pos = sorted([f for f in flags if f["delta"] > DELTA_THRESHOLD], key=lambda f: f["delta"],
                 reverse=True)
    near = [f for f in flags if abs(f["delta"]) <= DELTA_THRESHOLD]

    ordered = []
    sections = []

    def render(group, header):
        if not group:
            return
        lines = [header]
        for f in group:
            ordered.append(f)
            num = len(ordered)
            note_line = f"\n   YOUR NOTE: {f['note']}" if f.get("note") else ""
            mismatch_line = (f"\n   POSSIBLE MISMATCH: {f['possible_mismatch']}"
                             if f.get("possible_mismatch") else "")
            abstract = (f.get("abstract") or "")[:500]
            lines.append(
                f"[{num}] delta {f['delta']:+.2f}  "
                f"(judge {f['judge_score']:.2f} -> you {f['user_score']:.2f})\n"
                f"   Title: {f.get('title') or '(no title)'}\n"
                f"   Journal: {f.get('journal') or ''}  |  {f.get('pub_date_iso') or ''}\n"
                f"   Abstract: {abstract}\n"
                f"   JUDGE RATIONALE: {f.get('rationale') or ''}"
                f"{mismatch_line}"
                f"{note_line}"
            )
        sections.append("\n\n".join(lines))

    render(neg, "## JUDGE SCORED TOO HIGH (you scored lower -- seed over-triggering)")
    render(pos, "## JUDGE SCORED TOO LOW (you scored higher -- seed missing coverage)")
    render(near, f"## ROUGHLY AGREED (|delta| <= {DELTA_THRESHOLD}) -- provided for context")

    return "\n\n---\n\n".join(sections), ordered


def _format_existing_patterns(active, closed_patterns):
    """The pattern memory, shown to the consolidate step WITH ids so it can name the exact
    pattern a candidate merges into (open) or recurs against (closed). Empty string when
    there is no history yet."""
    if not active and not closed_patterns:
        return ""
    lines = ["## Existing pattern memory (match candidates against these by MEANING, using the id)"]
    if active:
        lines.append("\nOPEN patterns (still awaiting a decision) -- a candidate that is the same "
                     "gap is merge_into_open with that id:")
        for p in active:
            lines.append(f"  - id={p['id']}  [{p['direction']}] {p['name']}: "
                         f"{p.get('description') or ''}")
    if closed_patterns:
        lines.append("\nCLOSED patterns (already INCORPORATED or REJECTED) -- a candidate that matches "
                     "is recurs_closed with that id (logs the recurrence, does NOT reopen):")
        for p in closed_patterns:
            why = p["status"] + (f": {p['status_note']}" if p.get("status_note") else "")
            lines.append(f"  - id={p['id']}  [{why}] {p['name']}: {p.get('description') or ''}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _cost(model, usage):
    cin, cout = MODEL_COSTS.get(model, (3.0, 15.0))
    return (usage.input_tokens * cin + usage.output_tokens * cout) / 1_000_000


def _echo(text):
    """Print a streamed chunk to the console without ever crashing on a character the
    console encoding cannot represent (Windows cp1252 vs the model's unicode minus /
    em-dash / smart quotes). The returned text keeps the real characters; only the live
    echo is degraded, and only for the rare unencodable char."""
    try:
        print(text, end="", flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc), end="", flush=True)


def _stream(client, model, system, user_msg, max_tokens):
    parts = []
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            _echo(text)
            parts.append(text)
        final = stream.get_final_message()
    print()
    return "".join(parts), _cost(model, final.usage)


def run_cluster_step(client, papers_block, n_flags, seed_text, model):
    journal_block = _format_journal_ratings()
    user_msg = (
        f"## Current profile\n\n{seed_text}\n\n"
        f"---\n\n"
        f"{journal_block}\n\n"
        f"---\n\n"
        f"## Flagged papers ({n_flags} total)\n\n{papers_block}"
    )
    # Recall scales with flag count; give it room so it is never truncated mid-pattern.
    return _stream(client, model, CLUSTER_PROMPT, user_msg, max_tokens=6000)


# ---------------------------------------------------------------------------
# Consolidate: assign every candidate a choice, then RECORD (structured)
# ---------------------------------------------------------------------------

_CONSOLIDATE_SYSTEM = """
You are consolidating candidate preference patterns (distilled from a researcher's flags) against the
researcher's profile and their EXISTING pattern memory. You do NOT author profile prose and you do
NOT discard real signal. You assign EVERY candidate a choice and record it via the tool.

This is a MEMORY step, not a selection step. The bar for recording is low and objective: a candidate
is real if it is a regularity the flags actually show. The ONLY candidate not recorded is a HOLD -- a
lone one-paper correction too early to act on, kept unassigned so it returns and can accumulate. Everything
else is recorded; whether to ACT on it this round is a separate ranking the human does later, carried
by the `priority` hint, never by dropping.

MERGE FIRST. Before assigning choices, collapse candidates that a single profile edit would
satisfy, or that are facets of ONE underlying taste, into ONE pattern (union their paper_numbers).
Several sub-themes of the same taste -- distinct topics that all express one interest ("I value
theoretical/computational work"), or distinct methods that all express one disinterest ("scalp EEG
is uninteresting") -- are ONE pattern, not several. This is CONSOLIDATION, not dropping: every
supporting paper stays assigned to the merged pattern, so no signal is lost. Aim for the FEWEST
patterns that capture the genuinely DISTINCT tastes; a proliferation of narrow near-duplicates is the
failure mode. Recording everything real means not losing a distinct taste -- it does NOT mean
recording every fine-grained slice of one taste as its own pattern.

For each candidate choose a choice:
- new: a real taste-gap not already tracked. Give name, direction, description, suggested_edit,
  priority, paper_numbers.
- merge_into_open: essentially one of the OPEN patterns shown below (the same taste). Give its
  existing_pattern_id and the paper_numbers of the NEW supporting flags -- this is how recurrence
  accumulates on a pattern instead of spawning a duplicate.
- recurs_closed: it matches a pattern already INCORPORATED or REJECTED (a closed pattern). Give the
  existing_pattern_id. This LOGS that the taste came back; it does NOT reopen the decision. Say so in
  `rationale` ONLY if the new flags are a materially stronger case than when it was decided -- the
  human decides whether to reopen.
- hold: a real correction too lonely to act on yet -- NOT dropped. Record nothing to a pattern; the
  flag stays unassigned and returns next round until enough copies accumulate. Give a one-line rationale.

MATCHING: match against the shown patterns by MEANING, using their ids. Bias toward `new` when
identity is UNCERTAIN -- a duplicate is cheap for the human to reject, but an over-merge is sticky and
hard to undo. Only merge/recurs when it is clearly the same taste.

DIRECTION:
- under: the profile is MISSING coverage the flags show (judge scored too low).
- over: the profile OVER-triggers on something (judge scored too high).
- sharpen: the profile is genuinely vague/ambiguous on a boundary that needs resolution.
- judge-not-applying: the profile ALREADY states this preference clearly, yet the judge is not
  applying it. This is NOT a drop and NOT "already covered so ignore" -- it is a first-class signal to
  fix the PROMPT (the other tuning knob), so RECORD it as its own pattern. Use it whenever a flagged
  mismatch is the judge failing to honor clear existing profile text, rather than a profile gap.

PRIORITY (governs the act-now HINT only, never record-vs-drop):
- act_now: reserve for the FEW candidates that clearly beat the do-nothing default -- a real,
  generalizable taste worth a profile edit this round.
- defer: everything else real. Deferral is cheap here: the pattern is recorded and accumulates
  recurrence until it earns action. Bias toward defer.

THE NOTE WALL: in suggested_edit, state a GENERAL principle in the researcher's own voice -- never
transcribe a user's private note verbatim, and do not adopt the judge's framing or vocabulary (its
reasoning may be the error). Name the taste; do not draft the final profile line.

paper_numbers are the [N] references from the candidate clusters (union across any candidates you
merge). Record EVERY candidate exactly once. Output only via the record_consolidation tool.
""".strip()

_CONSOLIDATE_TOOL = {
    "name": "record_consolidation",
    "description": "Record a choice for EVERY candidate pattern (new / merge / recurs / hold).",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "choice": {"type": "string",
                            "enum": ["new", "merge_into_open", "recurs_closed", "hold"]},
                        "existing_pattern_id": {"type": "string",
                            "description": "id of the open pattern (merge_into_open) or closed pattern "
                                           "(recurs_closed) this matches; omit for new / hold"},
                        "name": {"type": "string", "description": "short label, 3-6 words (for new)"},
                        "direction": {"type": "string",
                            "enum": ["over", "under", "sharpen", "judge-not-applying"]},
                        "description": {"type": "string", "description": "one sentence (for new)"},
                        "suggested_edit": {"type": "string",
                            "description": "the directive as the researcher would author it (for new)"},
                        "priority": {"type": "string", "enum": ["act_now", "defer"],
                            "description": "act_now only if it clearly beats do-nothing; else defer"},
                        "paper_numbers": {"type": "array", "items": {"type": "integer"},
                            "description": "supporting [N] paper numbers from the clusters"},
                        "rationale": {"type": "string",
                            "description": "one line: why this choice/priority"},
                    },
                    "required": ["choice", "paper_numbers", "rationale"],
                },
            }
        },
        "required": ["candidates"],
    },
}


def run_consolidate_step(client, clusters_text, seed_text, existing_block, model):
    """Assign every candidate a choice via forced tool-use (so the JSON is always
    valid). Shown the clusters, the profile (to tell a real gap from the judge ignoring
    clear text -> judge-not-applying), and the existing patterns + closed patterns WITH ids
    (to capture the cross-round match). Returns (candidates, cost)."""
    memory = f"{existing_block}\n\n---\n\n" if existing_block else ""
    user_msg = (
        f"## Candidate patterns (with [N] paper numbers)\n\n{clusters_text}\n\n---\n\n"
        f"{memory}"
        f"## CURRENT PROFILE (source of truth -- a preference already clear here that the judge "
        f"still gets wrong is judge-not-applying, not a gap)\n\n{seed_text}"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        system=_CONSOLIDATE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        tools=[_CONSOLIDATE_TOOL],
        tool_choice={"type": "tool", "name": "record_consolidation"},
    )
    candidates = []
    for block in resp.content:
        if block.type == "tool_use":
            candidates = block.input.get("candidates", [])
            break
    return candidates, _cost(model, resp.usage)


def _record_consolidation(conn, candidates, ordered_flags):
    """Write each candidate's choice into the pattern memory. Provenance: paper
    number N -> ordered_flags[N-1] -> flag id. The event attached to a merge/recurs is
    driven by the TARGET pattern's REAL status, not the LLM's label -- so a mislabeled id
    can never resurrect a closed pattern (open target -> 'carried', closed pattern target ->
    'recurred'), and the event is skipped when no NEW flags were actually assigned, so
    re-running an overlapping window never inflates recurrence.

    LOSSLESS by construction: the ONLY candidate that is not recorded is an explicit
    hold (or one with no content and no papers at all). A malformed candidate --
    a merge naming a pattern id that does not exist, a missing name, an unrecognized
    choice -- is recovered as a new pattern rather than discarded, because a
    silently dropped candidate is exactly the signal-into-the-void failure this redesign
    exists to prevent. Returns a summary dict."""
    n = len(ordered_flags)
    summary = {"new": [], "merged": [], "recurred": [], "held": [], "skipped": []}

    def flag_ids_for(c):
        nums = c.get("paper_numbers") or []
        return [ordered_flags[i - 1]["id"] for i in nums
                if isinstance(i, int) and 1 <= i <= n]

    def status_of(pid):
        row = conn.execute(
            "SELECT event FROM pattern_events WHERE pattern_id = ? "
            "AND event IN ('created','carried','incorporated','rejected') "
            "ORDER BY created_at DESC, id DESC LIMIT 1", (pid,)).fetchone()
        return row["event"] if row else None

    def _fallback_name(c):
        """A usable name for a candidate the model left unnamed -- walk the other prose
        fields so a missing `name` never costs us the candidate. None only if the
        candidate carries no prose at all."""
        for key in ("name", "description", "suggested_edit", "rationale"):
            val = (c.get(key) or "").strip()
            if val:
                return val[:60]
        return None

    def _create(c, flag_ids, extra_note=""):
        note = f"{c.get('priority', 'defer')}: {c.get('rationale', '')}{extra_note}".strip()
        return db_interface.create_pattern(
            conn, name=(_fallback_name(c) or "(unnamed pattern)"),
            direction=(c.get("direction") or "under"),
            description=c.get("description"), suggested_edit=c.get("suggested_edit"),
            flag_ids=flag_ids, note=note or None)

    def _record_new(c, flag_ids, extra_note="", recovered=False):
        entry = {"id": _create(c, flag_ids, extra_note), "name": _fallback_name(c),
                 "direction": c.get("direction"), "priority": c.get("priority"),
                 "n_flags": len(flag_ids)}
        if recovered:
            entry["recovered"] = True
        summary["new"].append(entry)

    for c in candidates:
        choice = c.get("choice")
        flag_ids = flag_ids_for(c)
        if choice == "new":
            _record_new(c, flag_ids)
        elif choice in ("merge_into_open", "recurs_closed"):
            eid = c.get("existing_pattern_id")
            st = status_of(eid) if eid else None
            if st is None:
                # Missing / hallucinated target id (e.g. the model puts a direction in the
                # id field, which happens when the memory is empty and nothing can be
                # merged). Recover as a new pattern; only a wholly empty candidate is skipped.
                if _fallback_name(c) or flag_ids:
                    _record_new(c, flag_ids, recovered=True,
                                extra_note=" (merge target not found; recorded as new)")
                else:
                    summary["skipped"].append({"why": "empty candidate -- nothing to record"})
                continue
            added = db_interface.assign_flags_to_pattern(conn, eid, flag_ids)
            if not added:
                # Target already holds every one of these flags: nothing new, so no event.
                # This is the re-run idempotency guard, not a lost candidate.
                summary["skipped"].append({"id": eid, "why": "no new flags to link"})
                continue
            if st in ("incorporated", "rejected"):
                db_interface.add_pattern_event(conn, eid, "recurred", note=c.get("rationale"))
                summary["recurred"].append({"id": eid, "name": _fallback_name(c), "added": added})
            else:
                db_interface.add_pattern_event(conn, eid, "carried",
                                               note=f"recurred: {c.get('rationale', '')}")
                summary["merged"].append({"id": eid, "name": _fallback_name(c), "added": added})
        elif choice == "hold":
            summary["held"].append({"name": _fallback_name(c), "rationale": c.get("rationale")})
        else:
            # Unrecognized choice -- record rather than lose it; the human can reject.
            _record_new(c, flag_ids, recovered=True,
                        extra_note=f" (unrecognized choice {choice!r}; recorded as new)")
    return summary


def _format_consolidation_md(candidates):
    """Render the consolidate decisions as a readable markdown list -- ALL choices,
    holds included (transparency, not a discard sink)."""
    if not candidates:
        return "(no candidates)"
    lines = []
    for c in candidates:
        head = f"- **{c.get('choice', '?')}**"
        if c.get("name"):
            head += f" -- {c['name']}"
        if c.get("direction"):
            head += f" ({c['direction']})"
        if c.get("priority"):
            head += f" [{c['priority']}]"
        if c.get("existing_pattern_id"):
            head += f"  -> {c['existing_pattern_id'][:12]}"
        lines.append(head)
        if c.get("suggested_edit"):
            lines.append(f"    - edit: {c['suggested_edit']}")
        if c.get("rationale"):
            lines.append(f"    - why: {c['rationale']}")
        if c.get("paper_numbers"):
            lines.append(f"    - papers: {c['paper_numbers']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _summary_line(summary):
    if not summary:
        return "recorded nothing"
    return (f"recorded {len(summary['new'])} new, {len(summary['merged'])} merged, "
            f"{len(summary['recurred'])} recurred; {len(summary['held'])} held")


def suggest_edits(start=None, end=None,
                  cluster_model=DEFAULT_CLUSTER_MODEL, consolidate_model=DEFAULT_CONSOLIDATE_MODEL,
                  persist=True):
    """Cluster the UNASSIGNED (not-yet-patterned) flags in [start, end], consolidate each
    candidate against the pattern memory, and RECORD every real one (new / merge into an
    open pattern / recurs against a closed pattern); a genuine one-paper hold is kept unassigned, not recorded.
    Streams the recall to console and saves a dated markdown report. Returns the output
    path (or None if too few flags). Never re-validates on the flag set. persist=False is
    a dry run (writes the markdown, records nothing)."""
    seed_text = profile_interface.load_active()

    conn = db_interface.get_connection()
    try:
        # UNASSIGNED flags only: a flag already assigned into a pattern is "handled" and must
        # not re-cluster into a duplicate candidate. This is the bloat/idempotency bound.
        flags = db_interface.get_flags(conn, start=start, end=end, exclude_assigned=True)
        n = len(flags)
        if n < MIN_FLAGS:
            print(f"Only {n} unassigned (not-yet-patterned) flags in range -- need at least "
                  f"{MIN_FLAGS} to run.")
            return None

        # The pattern memory, shown to consolidate WITH ids so it captures cross-round
        # matches (merge into an open pattern / recurs against a closed pattern).
        active_patterns = db_interface.get_active_patterns(conn)
        closed_patterns = db_interface.get_patterns(conn, statuses=("incorporated", "rejected"))
        existing_block = _format_existing_patterns(active_patterns, closed_patterns)

        rng = f"{start or 'all'} to {end or 'all'}"
        print(f"{n} unassigned flags ({rng})  |  memory: {len(active_patterns)} open + "
              f"{len(closed_patterns)} decided")
        print(f"Models: cluster={cluster_model}  consolidate={consolidate_model}\n")

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        papers_block, ordered_flags = _format_papers(flags)

        print("=== Step 1: cluster (recall) ===\n")
        clusters, cost1 = run_cluster_step(client, papers_block, n, seed_text, cluster_model)
        print(f"\n[step 1 cost: ${cost1:.4f}]\n")

        print("=== Step 2: consolidate (choice) ===")
        candidates, cost2 = run_consolidate_step(client, clusters, seed_text, existing_block,
                                               consolidate_model)
        total = cost1 + cost2

        summary = None
        if persist:
            summary = _record_consolidation(conn, candidates, ordered_flags)
            for c in summary["new"]:
                tag = " [act_now]" if c.get("priority") == "act_now" else ""
                rec = " (recovered)" if c.get("recovered") else ""
                print(f"  + new [{c['direction']}] {c['name']}{tag}{rec}  ({c['n_flags']} flags)")
            for c in summary["merged"]:
                print(f"  ~ merged into {c['id'][:12]} (+{c['added']} flags -> carried)")
            for c in summary["recurred"]:
                print(f"  ! closed pattern {c['id'][:12]} recurred (+{c['added']} flags)")
            for c in summary["held"]:
                print(f"  . held (unassigned): {c.get('name') or c.get('rationale')}")
            for c in summary["skipped"]:
                print(f"  x skipped: {c.get('why')}")
            print(f"[{_summary_line(summary)}  |  total cost: ${total:.4f}]")
        else:
            print(f"[dry run: {len(candidates)} candidates consolidated, nothing recorded  "
                  f"|  total cost: ${total:.4f}]")
    finally:
        conn.close()

    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{start or 'all'}_{end or 'all'}"

    def _short(model_id):
        return model_id.replace("claude-", "").replace("/", "-")

    tail = "DRY RUN (nothing recorded)" if not persist else _summary_line(summary)
    # Both models in the name: swapping only the consolidate model must not clobber the
    # previous report, or a model A/B is unreadable.
    out = (SUGGESTIONS_DIR /
           f"pattern_suggestions_{slug}_{_short(cluster_model)}__{_short(consolidate_model)}.md")
    out.write_text(
        f"# Pattern suggestions\n\n"
        f"Unassigned flags: {n}  |  range: {rng}  |  cluster: {cluster_model}  "
        f"consolidate: {consolidate_model}  |  cost: ${total:.4f}  |  {tail}\n\n"
        f"---\n\n## Raw clusters (recall)\n\n{clusters}\n\n"
        f"---\n\n## Consolidation (choices)\n\n{_format_consolidation_md(candidates)}\n",
        encoding="utf-8",
    )
    print(f"\nSaved to {out}")
    return out
