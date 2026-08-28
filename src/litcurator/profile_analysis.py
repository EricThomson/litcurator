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
code read -- so real-but-not-now patterns were silently lost and recurrence could never
accumulate. Now nothing is dumped; every candidate gets a choice and a home.

Two LLM stages (the Tao of litcurator: generate cheap-and-broad, then decide):
  Step 1 (cluster, Sonnet): RECALL -- surface every candidate preference pattern from
    the UNATTACHED (not-yet-patterned) flags.
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

Reads UNATTACHED flags from db_interface.get_flags(exclude_attached=True); reads the active
profile from profile_interface. Streams the recall to console and saves a dated
markdown report to ~/.litcurator/suggestions/. See starry-brewing-horizon.md.
"""

import copy
import os
import sys
from datetime import datetime

import anthropic
from dotenv import load_dotenv

from litcurator import analysis_prompt_interface, db_interface, profile_interface
from litcurator.config import DATA_DIR, MAX_ACT_NOW

load_dotenv()

# Both stages are Sonnet. Recall is Sonnet's strength (crisp, broad generation).
# Consolidate is structured tagging with NO prose authorship, so Opus's documented
# prose-padding liability does not apply and Sonnet is ~5x cheaper; keep Opus as a
# drop-in fallback only if the duplicate rate on real data proves poor.
DEFAULT_CLUSTER_MODEL = "claude-sonnet-4-6"
DEFAULT_CONSOLIDATE_MODEL = "claude-sonnet-4-6"

# Re-exported, NOT redefined. db_interface owns the vocabulary; this name stays because it
# reads better at the clamp below and because callers already import it. Until 2026-08-16 it
# was a fourth hand-maintained copy, and a value missing from it was silently rewritten to
# `under` -- see the note beside db_interface.DIRECTIONS.
VALID_DIRECTIONS = db_interface.DIRECTIONS

# Rendered once for the status subquery below; db_interface owns the vocabulary.
_DECISIONS = ", ".join(f"'{e}'" for e in db_interface.DECISION_EVENTS)

# Approximate API prices, ($/M input, $/M output). Update if pricing changes.
MODEL_COSTS = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

SUGGESTIONS_DIR = DATA_DIR / "suggestions"

# Below this many unattached flags, suggest_edits refuses to run: clustering three flags buys
# a pattern you could have seen by eye. Inherited from the same v1-era script as the delta
# threshold below and never re-derived, but kept deliberately (2026-08-07) -- unlike that one
# it is a cadence guard, it is not shaping the evidence, and it fails loudly.
MIN_FLAGS = 10

# NO MAGNITUDE THRESHOLD ON DELTA. There used to be one -- DELTA_THRESHOLD = 0.15, which split
# the papers block into over-scored / under-scored / "roughly agreed, provided for context".
# It was inherited verbatim from the v1-era sandbox/suggest_seed_edits.py, never chosen, never
# justified, never pinned by a test, and it turned out to be doing real damage. Removed
# 2026-08-07. Do not add another one; the reasoning is worth keeping because the constant
# looked harmless for months:
#
#   The label was false. On the January flags every single paper it filed under "roughly
#   agreed" was a complaint carrying an explicit instruction ("should be bumped down",
#   "how is this close to threshold", "the mismatch reasoning is bad"). Zero agreements.
#   A flag is a correction by definition -- the user does not spend one to say "correct".
#
#   It cut tastes in half. Two flags asking for nearly the same profile edit in nearly the
#   same words (delta -0.20 and -0.15, both "topics of disinterest section") landed on
#   opposite sides, one as evidence and one as background. Same for venue calibration and
#   for high-resolution imaging in nonhumans. The cluster prompt asks the model to weigh how
#   pervasive a pattern is; the renderer was hiding half the evidence for three of them.
#
#   The line sat inside the noise. Judge test-retest sigma is about 0.05 where these flags
#   live, and one flag sat 0.001 from the boundary while a re-score moved it 0.046.
#
# The sign of delta is categorical and free -- it is on every paper's own line. The MAGNITUDE
# is continuous and belongs to the model's judgment, which is where the analysis prompt now
# puts it ("a steady ~0.1 bias across many papers is a real, systematic error"). If a
# near-zero concept is ever wanted again it goes in prompt/analysis_prompt.md, which is
# hand-authored, content-addressed and stamped on every analysis_run -- not here, where a
# number silently reshapes the evidence for every future run and leaves no record at all.


# ---------------------------------------------------------------------------
# SHIPPED DEFAULTS -- deliberately minimal, and NOT what runs for you.
#
# The two steps below run the CLUSTER and CONSOLIDATE sections of
# prompt/analysis_prompt.md in your data directory, which is versioned and content-addressed
# exactly like your profile and your judge prompt. These constants exist only to seed that file
# the first time. Editing them changes nothing for an existing install -- edit the active prompt
# in the analysis prompt lab (it has a Set ACTIVE button) or the file itself.
#
# They are MINIMAL on purpose, and the split is deliberate: the prose carries only what the
# machinery cannot enforce for itself. The four choices, the two directions and the required
# fields are already pinned by _CONSOLIDATE_TOOL below, so the prompt does not restate them --
# it says what they MEAN. Everything else is tuning (how eagerly to merge, how to weigh a note,
# what counts as enough evidence), and tuning belongs in your versioned copy where it can be
# A/B'd and rolled back, not in the package.
#
# If either of these grows past a screen, someone has tuned the default instead of their own.
# ---------------------------------------------------------------------------

# litcurator ships NO prompt content -- see the same note in judge.py. The two steps below run
# the CLUSTER and CONSOLIDATE sections of prompt/analysis_prompt.md in your data directory,
# which you author and which is versioned and content-addressed like your profile. Writing a
# good minimal default for a new user is a real task, and it is on the long-term list rather
# than something to approximate here: a mediocre default would silently shape every pattern
# this machinery ever produces.
PROMPT_NOT_AUTHORED = "<no analysis prompt authored>"


def _require_prompt(prompt, which):
    """Resolve a step's prompt, or fail loudly. Every caller in the package passes one
    explicitly (suggest_edits loads the active analysis prompt and splits it); this catches a
    caller that assumed a default exists."""
    if prompt and prompt != PROMPT_NOT_AUTHORED:
        return prompt
    raise ValueError(
        f"No {which} prompt. litcurator ships no default -- author one and set it active "
        f"(the analysis prompt lab has a Set ACTIVE button), or pass prompt= explicitly. "
        f"The active prompt lives at prompt/analysis_prompt.md in your data directory.")

# ---------------------------------------------------------------------------
# Formatting helpers for the prompt
# ---------------------------------------------------------------------------

# _format_journal_ratings lived here and rendered config.USER_JOURNAL_RATINGS into the cluster
# message. Both are gone (2026-08-07) -- the reasoning is in config.py where the table was.
# Short version: it handed the auditor the auditee's rubric.

def _format_papers(flags):
    """Render the flags as ONE numbered papers block, AND return the flags in the SAME order
    the numbers follow -- so paper number N in the LLM output maps back to ordered[N-1] (and
    thus its flag id) for provenance.

    Order is strongest disagreement first: |delta| descending, flag id ascending to break
    ties. The tiebreak is load-bearing rather than tidy -- judge scores are lumpy (the model
    reaches for favored anchors), so equal |delta| happens, and without a stable second key
    the block would reshuffle between runs for no reason and a re-run would be unattributable.
    db_interface.get_pattern_examples documents the same hazard for the memory block.

    NO SECTIONS, and no magnitude cut -- see the note beside MIN_FLAGS. Every paper carries its
    own signed delta on its own line, which is all the sections ever asserted, minus the part
    they asserted falsely."""
    ordered = sorted(flags, key=lambda f: (-abs(f["delta"]), f["id"]))

    # No header here: run_cluster_step already writes one over this block. The old three
    # headers were SECTIONS underneath it, so a wrapper plus sections made sense; with one
    # list it would just be the same heading twice.
    lines = []
    for num, f in enumerate(ordered, start=1):
        note_line = f"\n   YOUR NOTE: {f['note']}" if f.get("note") else ""
        mismatch_line = (f"\n   POSSIBLE MISMATCH: {f['possible_mismatch']}"
                         if f.get("possible_mismatch") else "")
        lines.append(
            f"[{num}] delta {f['delta']:+.2f}  "
            f"(judge {f['judge_score']:.2f} -> you {f['user_score']:.2f})\n"
            f"   Title: {f.get('title') or '(no title)'}\n"
            f"   Journal: {f.get('journal') or ''}  |  {f.get('pub_date_iso') or ''}\n"
            f"   Abstract: {f.get('abstract') or ''}\n"
            f"   JUDGE RATIONALE: {f.get('rationale') or ''}"
            f"{mismatch_line}"
            f"{note_line}"
        )

    return "\n\n".join(lines), ordered


def _format_existing_patterns(active, closed_patterns, examples=None, held=()):
    """The pattern memory, shown to the consolidate step WITH ids so it can name the exact
    pattern a candidate merges into (open) or recurs against (closed). Empty string when
    there is no history yet.

    `examples` is {pattern_id: [{title, ...}]} from db_interface.get_pattern_examples, and it
    is the most useful thing in the block. Everything else here -- the name, the description --
    is the model's own paraphrase of a gap, so matching a returning gap against it means
    comparing two model-authored noun phrases, which is a coin flip: shown "Formal/Normative
    Theory as Mechanistic" the model minted "Theoretical / Computational Neuroscience Accounts"
    for the same returning gap. The PAPERS are different in kind. The candidate the model is
    holding is described by its papers too, so like compares with like.

    Also shown: how many flags a pattern holds, and how many times it has already come back --
    a gap on its third return should be logged, not minted afresh. Closed patterns keep their
    direction (the bracket used to be reused for status, which dropped it for exactly the
    patterns hardest to match).

    NOT shown: suggested_edit. It is the description again in imperative mood, it nearly
    doubles the block, and it is first-person profile prose arriving in a message that ends
    with the real profile -- which invites the model to read it as profile text rather than as
    a description of a pattern already decided.

    Passing examples=None renders exactly as before, with no papers."""
    if not active and not closed_patterns and not held:
        return ""
    examples = examples or {}

    def rows(p, bracket):
        seen = f", returned {p['recurred_count']}x" if p.get("recurred_count") else ""
        # HOW BADLY, not just how many. Without it the model weighing whether a returning gap
        # has become worth showing sees the head count of what it already holds and nothing
        # about the size of it -- so it judges accumulation with half the evidence. A fresh
        # candidate arrives with its magnitude described in cluster's prose; until now the
        # remembered side had none at all.
        size = (f", |delta| avg {p['mean_abs_delta']:.2f} max {p['max_abs_delta']:.2f}"
                if p.get("mean_abs_delta") is not None else "")
        out = [f"  - id={p['id']}  [{bracket}]  {p['name']}  "
               f"({p.get('flag_count', 0)} flags{size}{seen})"]
        if p.get("description"):
            out.append(f"      {p['description']}")
        papers = examples.get(p["id"]) or []
        if papers:
            titles = " | ".join((e["title"] or "")[:120] for e in papers)
            out.append(f"      papers: {titles}")
        return out

    lines = ["## Existing pattern memory (match candidates against these by MEANING, using the id)"]
    if active:
        lines.append("\nOPEN patterns (still awaiting a decision) -- a candidate that is the same "
                     "gap is merge_into_open with that id:")
        for p in active:
            lines += rows(p, p["direction"])
    if held:
        # Between open and closed, which is where they sit: undecided, but not yet shown. The
        # model needs them for the same reason it needs the open ones -- so a returning gap is
        # recognised rather than minted again -- and merging into one is what promotes it.
        lines.append("\nHELD patterns (recorded from earlier flags, NOT yet shown to the user) -- "
                     "a candidate that is the same gap is merge_into_open with that id:")
        for p in held:
            lines += rows(p, p["direction"])
    if closed_patterns:
        lines.append("\nCLOSED patterns (already INCORPORATED or REJECTED) -- a candidate that matches "
                     "is merge_into_closed with that id (logs the recurrence, does NOT reopen):")
        for p in closed_patterns:
            why = p["status"] + (f": {p['status_note']}" if p.get("status_note") else "")
            lines += rows(p, f"{p['direction']} | {why}")
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


def _consolidate_tool(has_memory):
    """The output contract, narrowed to what is POSSIBLE this round.

    With no pattern memory there is nothing to merge into, so `merge_into_open` and
    `merge_into_closed` come out of the enum and forced tool-use makes them unemittable. That
    is not a preference: with zero recorded patterns, every existing_pattern_id is necessarily
    invented.

    STRUCTURAL RATHER THAN INSTRUCTED, deliberately. Both January dry runs produced exactly one
    spurious merge against an empty memory -- in the second, the target was the NAME of a
    pattern minted in the same round. The prompt already said there was nothing to merge into,
    so the instruction was present and ignored; the record step recovered them, so nothing was
    lost and nothing failed loudly. Narrowing the enum removes the possibility instead of
    restating the rule.

    Invisible from the second round onward, when a memory exists."""
    if has_memory:
        return _CONSOLIDATE_TOOL
    tool = copy.deepcopy(_CONSOLIDATE_TOOL)
    props = tool["input_schema"]["properties"]["candidates"]["items"]["properties"]
    props["choice"]["enum"] = [c for c in props["choice"]["enum"]
                               if not c.startswith("merge_into")]
    props["existing_pattern_id"]["description"] = (
        "unused this round -- there is no pattern memory yet, so there is nothing to merge into")
    return tool


def run_cluster_step(client, papers_block, n_flags, seed_text, model, prompt=None):
    """`prompt` is the CLUSTER section of the active analysis prompt, passed in by the caller.
    Taking it as an argument rather than reading a module global is what lets the harness and
    the lab test a draft without mutating shared state -- the judge has always worked this way."""
    prompt = _require_prompt(prompt, "cluster")
    # The message is the profile plus the flagged papers, and nothing else. A journal-ratings
    # block used to sit between them; see config.py for why it is gone.
    user_msg = (
        f"## Current profile\n\n{seed_text}\n\n"
        f"---\n\n"
        f"## Flagged papers ({n_flags} total, strongest disagreement first)\n\n{papers_block}"
    )
    # Recall scales with flag count; give it room so it is never truncated mid-pattern.
    return _stream(client, model, prompt, user_msg, max_tokens=6000)


# ---------------------------------------------------------------------------
# Consolidate: assign every candidate a choice, then RECORD (structured)
# ---------------------------------------------------------------------------

_CONSOLIDATE_TOOL = {
    "name": "record_consolidation",
    "description": "Record a choice for every REAL pattern (new / merge / recurs / discard).",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "choice": {"type": "string",
                            "enum": ["new", "merge_into_open", "merge_into_closed", "discard"]},
                        "existing_pattern_id": {"type": "string",
                            "description": "id of the open pattern (merge_into_open) or closed pattern "
                                           "(merge_into_closed) this matches; omit for new / hold"},
                        "name": {"type": "string", "description": "short label, 3-6 words (for new)"},
                        # From db_interface.DIRECTIONS, so the set the model is offered
                        # cannot drift from the set the database accepts.
                        "direction": {"type": "string", "enum": list(VALID_DIRECTIONS)},
                        "description": {"type": "string", "description": "one sentence (for new)"},
                        "suggested_edit": {"type": "string",
                            "description": "the directive as the researcher would author it (for new)"},
                        # SURFACING, not urgency (2026-08-26). This is the only field that
                        # decides whether the user is SHOWN a pattern; everything real is
                        # recorded either way. It replaced an act_now/defer pair where both
                        # values produced an identical screen -- nothing in the workbench read
                        # the field at all, so it was a label with no consumer.
                        "rank": {"type": "integer",
                            "description": "1 = the pattern whose fix would improve the judge "
                                           "most. Rank ALL candidates against each other, no "
                                           "ties; only the top few are shown to the user"},
                        "priority": {"type": "string", "enum": ["act_now", "hold"],
                            "description": "act_now shows it in the user's queue this round; "
                                           "hold records it and keeps it out until later "
                                           "evidence makes it actionable"},
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


def build_memory_block(conn):
    """THE pattern memory block, assembled in ONE place.

    Both the live path (suggest_edits) and the harness (machinery.run_round) show the model the
    same three lists, and until 2026-08-26 each built it by hand. When held patterns were added
    only the live path was updated, so every paid gate ran with held patterns invisible -- and a
    held pattern the model cannot see is one it cannot merge into, so it mints a duplicate and
    the gap fragments. That is not a cosmetic omission, it is the mechanism the held design
    depends on, and it was silently absent for a whole $0.73 sweep.

    So the assembly lives here and the callers pass a connection. Two hand-built copies of the
    same block is the DRY violation that caused it; one function is the fix."""
    active = db_interface.get_active_patterns(conn)
    held = db_interface.get_held_patterns(conn)
    closed = db_interface.get_patterns(conn, statuses=db_interface.CLOSED_STATUSES)
    examples = db_interface.get_pattern_examples(
        conn, [p["id"] for p in active + held + closed])
    return _format_existing_patterns(active, closed, examples, held=held), active, held, closed


def run_consolidate_step(client, clusters_text, seed_text, existing_block, model,
                         prompt=None):
    """Assign every candidate a choice via forced tool-use (so the JSON is always
    valid). Shown the clusters, the profile (context for judging what a candidate is really
    about), and the existing patterns + closed patterns WITH ids
    (to capture the cross-round match). Returns (candidates, cost).

    `prompt` defaults to the in-code seed; the live path passes the active consolidate section
    from disk, the harness passes a draft. See run_cluster_step."""
    prompt = _require_prompt(prompt, "consolidate")
    memory = f"{existing_block}\n\n---\n\n" if existing_block else ""
    tool = _consolidate_tool(has_memory=bool(memory))
    user_msg = (
        f"## Candidate patterns (with [N] paper numbers)\n\n{clusters_text}\n\n---\n\n"
        f"{memory}"
        # A LABEL, NOT AN INSTRUCTION. This used to append "a preference already clear here
        # that the judge still gets wrong is judge-not-applying, not a gap" -- prompt text
        # living in code, outside the hand-authored content-addressed prompt file, and the
        # strongest single push toward the value removed on 2026-08-25.
        f"## CURRENT PROFILE (the user's own prose about what they want to read)"
        f"\n\n{seed_text}"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        system=prompt,
        messages=[{"role": "user", "content": user_msg}],
        tools=[tool],
        tool_choice={"type": "tool", "name": "record_consolidation"},
    )
    candidates = []
    for block in resp.content:
        if block.type == "tool_use":
            candidates = block.input.get("candidates", [])
            break
    return candidates, _cost(model, resp.usage)


def _record_consolidation(conn, candidates, ordered_flags, analysis_run_id=None):
    """Write each candidate's choice into the pattern memory. Provenance: paper
    number N -> ordered_flags[N-1] -> flag id. The event attached to a merge/recurs is
    driven by the TARGET pattern's REAL status, not the LLM's label -- so a mislabeled id
    can never resurrect a closed pattern (open target -> 'carried', closed pattern target ->
    'recurred'), and the event is skipped when no NEW flags were actually attached, so
    re-running an overlapping window never inflates recurrence.

    LOSSLESS by construction: the ONLY candidate that is not recorded is an explicit
    hold (or one with no content and no papers at all). A malformed candidate --
    a merge naming a pattern id that does not exist, a missing name, an unrecognized
    choice -- is recovered as a new pattern rather than discarded, because a
    silently dropped candidate is exactly the signal-into-the-void failure this redesign
    exists to prevent. Returns a summary dict."""
    n = len(ordered_flags)
    # `held` now holds RECORDED patterns (rows, provenance, a 'held' event) rather than the
    # name-only ghosts it held until 2026-08-26, when a hold wrote nothing at all. `surfaced`
    # is a held pattern promoted this round; `discarded` is a candidate judged not to be a real
    # pattern, which is the only outcome that records nothing.
    summary = {"new": [], "merged": [], "recurred": [], "held": [], "surfaced": [],
               "discarded": [], "skipped": []}

    # THE QUEUE CAP, applied here rather than asked for. The model ranks; code cuts.
    #
    # Instructing it did not work: two January dry runs put TEN patterns in the queue against a
    # stated cap of eight, at exactly ten both times. Asking one forced tool call to hold a
    # running total across output it has not finished emitting is a poor shape for a generator,
    # and the instruction sat where positioning is worst. Ranking is something it CAN do while
    # emitting, so the tool asks for a rank and the arithmetic happens here -- the same division
    # as computed_sign, where the model judges and code owns what is computable.
    #
    # Only `new` candidates are capped, because only they add a card to the queue: a merge into
    # an open pattern lands on one already there. Overflow is DEMOTED to hold, never dropped --
    # it keeps its row, its provenance and its place in the Held tab, and returns next round.
    # The model's own word is preserved beside it so a demotion is legible in the report and a
    # model that ranks badly is visible rather than silently corrected.
    wants_queue = [c for c in candidates
                   if c.get("choice") == "new" and (c.get("priority") or "act_now") != "hold"]
    if len(wants_queue) > MAX_ACT_NOW:
        ranked = sorted(
            enumerate(wants_queue),
            key=lambda ic: (ic[1]["rank"] if isinstance(ic[1].get("rank"), int) else 10 ** 6,
                            ic[0]))
        for _pos, c in ranked[MAX_ACT_NOW:]:
            c["priority_asked"] = c.get("priority") or "act_now"
            c["priority"] = "hold"
            c["rationale"] = ((c.get("rationale") or "").rstrip()
                              + f" [demoted: over the {MAX_ACT_NOW}-pattern queue cap]").strip()

    def surface_of(c):
        """act_now | hold -- the ONLY field deciding whether the user is shown this pattern.
        Anything unrecognized shows it: failing toward visible is the safe direction, since a
        pattern the user can see is one they can reject in a second."""
        return "hold" if (c.get("priority") or "").strip() == "hold" else "act_now"

    def flag_ids_for(c):
        nums = c.get("paper_numbers") or []
        return [ordered_flags[i - 1]["id"] for i in nums
                if isinstance(i, int) and 1 <= i <= n]

    def deltas_for(c):
        """The deltas of the flags this candidate actually cites, in render order."""
        nums = c.get("paper_numbers") or []
        return [ordered_flags[i - 1]["delta"] for i in nums
                if isinstance(i, int) and 1 <= i <= n]

    def computed_sign(c):
        """over / under / mixed / None, from ARITHMETIC on the cited flags.

        WHY THIS EXISTS AND WHY IT OVERRIDES THE MODEL. `direction` was asking one field to
        answer two different questions: WHICH WAY the judge erred (the sign of the delta --
        deterministic, and already in this function's hands) and WHAT IS WRONG at the other end
        (profile silent / vague / stated-but-ignored -- a real judgment about the profile).
        Only the second needs a model.

        The first was being routed through two LLM stages as English and arriving flipped.
        Measured over three different fixtures on 2026-08-16: the cluster step wrote "Judge
        scoring too HIGH. Deltas of -0.13 on both papers" -- correct, explicit, unambiguous --
        and consolidate recorded `under`, then `judge-not-applying`, then `judge-not-applying`.
        It never once produced the declared sign. Not a data problem: the answer was in the
        input in plain words. It matched the DEFINITION it was given, where `under` reads as
        "the profile is MISSING coverage", which was true of a profile silent on the topic.

        No wording fixes that, because the two questions genuinely have different answers. So
        code answers the one it can answer exactly. Returns None when the candidate cites no
        usable papers, in which case the model's word stands -- there is nothing to compute.

        RESOLVED PROPERLY 2026-08-25: the second question was DELETED rather than re-worded.
        `sharpen` and `judge-not-applying` asked the model for an etiology its inputs cannot
        support (see the note beside db_interface.DIRECTIONS), so direction is now only the
        sign, and this function owns the field outright. The history above is kept because it
        is the evidence that produced the deletion."""
        ds = [d for d in deltas_for(c) if d is not None]
        if not ds:
            return None
        if all(d > 0 for d in ds):
            return "under"
        if all(d < 0 for d in ds):
            return "over"
        return "mixed"

    def status_of(pid):
        row = conn.execute(
            "SELECT event FROM pattern_events WHERE pattern_id = ? "
            f"AND event IN ({_DECISIONS}) "
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

    def _direction(c):
        """The candidate's direction, clamped to the four the patterns table allows.

        The tool schema enums it, so the model is strongly steered, but nothing HARD-validates
        a tool argument on the way back -- and `patterns` carries a CHECK constraint, so one
        out-of-vocabulary value raises IntegrityError partway through recording. Since
        create_pattern commits per candidate, that leaves a HALF-WRITTEN consolidation. Every
        other field the model controls already has a recovery path (a missing name, a
        hallucinated pattern id, an unrecognized choice); this was the one that could still
        abort the run, and lossless-by-construction has to mean it too. Marked so the coercion
        is visible in the summary and the report rather than passing as the model's own word."""
        val = (c.get("direction") or "").strip()
        if val in VALID_DIRECTIONS:
            return val, False
        return "under", True

    def _resolved_direction(c):
        """The direction actually recorded, plus a note fragment when code overrode the model.

        RULE: the computed sign wins, because direction is arithmetic on the cited flags'
        deltas. The model's answer survives only where the arithmetic has nothing to say --
        no flags cited (`sign is None`), or flags pointing both ways (`mixed`).

        Since the diagnosis values went (2026-08-25), this is the whole of it. There used to be
        a branch keeping `sharpen` / `judge-not-applying` untouched, because those were claims
        about the PROFILE that no arithmetic could check. With direction reduced to a sign,
        code owns the field outright.

        DISAGREEMENTS ARE RECORDED, NOT SILENTLY CORRECTED. "The model said under, the flags say
        over" is the signal that the prompt's definitions are off, and hiding it would turn a
        measurable prompt defect into a mystery. It also keeps the harness honest: a check
        asserting the RECORDED direction is now tautologically green, so the gate asserts
        model-vs-computed AGREEMENT instead, which is the thing that can still be wrong.

        `mixed` is recorded in the note but falls back to the model's word, because
        `patterns.direction` has no value for it. That is the deferred "wrong in BOTH directions
        at once" question, and this gives it a free measurement: if mixed shows up often on real
        flags, a third value has earned itself."""
        model_dir, coerced = _direction(c)
        sign = computed_sign(c)
        if sign is None:
            return model_dir, coerced, ""
        if sign == "mixed":
            return model_dir, coerced, f" [flags point BOTH ways; recorded {model_dir}]"
        if sign == model_dir:
            return sign, coerced, ""
        return sign, coerced, (f" [direction from flag deltas: {sign}; the model proposed "
                               f"{(c.get('direction') or model_dir)!r}]")

    def _create(c, flag_ids, extra_note=""):
        direction, coerced, disagreement = _resolved_direction(c)
        extra_note += disagreement
        if coerced and c.get("direction"):
            extra_note += f" [direction {c['direction']!r} not recognized, recorded as under]"
        note = f"{c.get('priority', 'defer')}: {c.get('rationale', '')}{extra_note}".strip()
        return db_interface.create_pattern(
            conn, name=(_fallback_name(c) or "(unnamed pattern)"),
            direction=direction,
            description=c.get("description"), suggested_edit=c.get("suggested_edit"),
            flag_ids=flag_ids, note=note or None, analysis_run_id=analysis_run_id)

    def _record_new(c, flag_ids, extra_note="", recovered=False):
        direction, coerced, disagreement = _resolved_direction(c)
        pid = _create(c, flag_ids, extra_note)
        surface = surface_of(c)
        if surface == "hold":
            # A REAL pattern, recorded with full provenance, that the user is simply not shown.
            # 'created' stays as the minting fact and 'held' is the decision on top of it, so the
            # log reads created -> held -> carried -> incorporated for a gap noticed early,
            # promoted when it recurred, and folded in. Collapsing the two would save a row and
            # lose "when was this first recognized".
            db_interface.add_pattern_event(conn, pid, "held", note=c.get("rationale"))
        entry = {"id": pid, "name": _fallback_name(c),
                 "direction": direction, "priority": c.get("priority"),
                 "n_flags": len(flag_ids)}
        if recovered:
            entry["recovered"] = True
        if coerced:
            entry["direction_coerced"] = c.get("direction")
        if disagreement:
            # What the model WANTED, kept beside what was recorded. The harness grades on this
            # -- asserting the recorded direction would be grading arithmetic, not the prompt.
            # The RAW word, not the clamped one: an off-schema value clamps to `under`, and
            # recording that would report an inversion the model never proposed.
            entry["direction_proposed"] = c.get("direction") or _direction(c)[0]
            entry["direction_note"] = disagreement.strip()
        # Both are minted rows; the bucket says whether the user is shown it this round.
        summary["held" if surface == "hold" else "new"].append(entry)

    for c in candidates:
        choice = c.get("choice")
        flag_ids = flag_ids_for(c)
        if choice == "new":
            _record_new(c, flag_ids)
        elif choice in ("merge_into_open", "merge_into_closed"):
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
            added = db_interface.attach_flags_to_pattern(conn, eid, flag_ids)
            if not added:
                # Target already covers every one of these PAPERS: nothing new arrived, so no
                # event. This is the re-run idempotency guard, not a lost candidate -- and
                # since 2026-08-26 `added` counts papers rather than pattern_flags rows, so a
                # re-flag of a paper the pattern already holds no longer fires one either.
                summary["skipped"].append({"id": eid, "why": "no new papers to attach"})
                continue
            if st in db_interface.CLOSED_STATUSES:
                db_interface.add_pattern_event(conn, eid, "recurred", note=c.get("rationale"))
                summary["recurred"].append({"id": eid, "name": _fallback_name(c), "added": added})
            elif st in db_interface.HELD_STATUSES and surface_of(c) == "hold":
                # Came back, still not actionable. Logged rather than silent, so the history
                # reads held -> held -> carried and "this has been sitting for three rounds"
                # is answerable.
                db_interface.add_pattern_event(conn, eid, "held",
                                               note=f"returned: {c.get('rationale', '')}")
                summary["held"].append({"id": eid, "name": _fallback_name(c), "added": added,
                                        "returned": True})
            else:
                # PROMOTION happens here, and needs no threshold: 'carried' is an active status,
                # so a held pattern the model now wants shown simply becomes visible. An
                # already-open pattern was visible anyway and stays so.
                db_interface.add_pattern_event(conn, eid, "carried",
                                               note=f"recurred: {c.get('rationale', '')}")
                bucket = "surfaced" if st in db_interface.HELD_STATUSES else "merged"
                summary[bucket].append({"id": eid, "name": _fallback_name(c), "added": added})
        elif choice == "discard":
            # The ONLY outcome that records nothing, and the only one where that is right: the
            # model judged this candidate not to be a real pattern at all. Its flags were never
            # attached, so they stay in the pool and cluster reads the papers again next round.
            summary["discarded"].append({"name": _fallback_name(c),
                                         "rationale": c.get("rationale")})
        else:
            # Unrecognized choice -- record rather than lose it; the human can reject.
            _record_new(c, flag_ids, recovered=True,
                        extra_note=f" (unrecognized choice {choice!r}; recorded as new)")
    return summary


def _looks_like_pattern_id(value):
    """A real pattern id is a 32-char hex uuid. Anything else the model put in that field is a
    hallucination, and the report must show it WHOLE."""
    v = (value or "").strip()
    return len(v) == 32 and all(ch in "0123456789abcdef" for ch in v.lower())


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
            # A REAL id is truncated for readability; anything else is shown WHOLE and marked.
            # It used to truncate unconditionally, which is fine for a 32-char uuid and actively
            # misleading for an invented target: the 2026-08-27 dry run rendered a hallucinated
            # value as "human-only-s", which reads like a real id prefix and hid what the model
            # actually emitted. The report is the only record of a dry run, so it has to show
            # the evidence rather than a tidy-looking slice of it.
            eid = c["existing_pattern_id"]
            head += (f"  -> {eid[:12]}" if _looks_like_pattern_id(eid)
                     else f"  -> NOT A PATTERN ID: {eid!r}")
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
            f"{len(summary['recurred'])} recurred, {len(summary['held'])} held, "
            f"{len(summary['surfaced'])} surfaced; {len(summary['discarded'])} discarded")


def suggest_edits(start=None, end=None,
                  cluster_model=DEFAULT_CLUSTER_MODEL, consolidate_model=DEFAULT_CONSOLIDATE_MODEL,
                  persist=True):
    """Cluster the UNATTACHED (not-yet-patterned) flags in [start, end], consolidate each
    candidate against the pattern memory, and RECORD every real one (new / merge into an
    open pattern / recurs against a closed pattern); a genuine one-paper hold is kept unattached, not recorded.
    Streams the recall to console and saves a dated markdown report. Returns the output
    path (or None if too few flags). Never re-validates on the flag set. persist=False is
    a dry run (writes the markdown, records nothing).

    The analysis prompt is loaded from disk and REGISTERED, exactly as the pipeline does with
    the judge prompt: every pattern this produces points at an analysis_run, which points at
    the prompt content that made it. Without that, "which prompt produced this pattern" is
    unanswerable, which was the one provenance gap in the system."""
    seed_text = profile_interface.load_active()
    analysis_prompt = analysis_prompt_interface.load_active()
    cluster_prompt, consolidate_prompt = analysis_prompt_interface.split(analysis_prompt)

    conn = db_interface.get_connection()
    try:
        # UNATTACHED flags only: a flag already attached to a pattern is "handled" and must
        # not re-cluster into a duplicate candidate. This is the bloat/idempotency bound.
        flags = db_interface.get_flags(conn, start=start, end=end, exclude_attached=True)
        n = len(flags)
        if n < MIN_FLAGS:
            print(f"Only {n} unattached (not-yet-patterned) flags in range -- need at least "
                  f"{MIN_FLAGS} to run.")
            return None

        # The pattern memory, shown to consolidate WITH ids so it captures cross-round
        # matches (merge into an open pattern / recurs against a closed pattern).
        existing_block, active_patterns, held_patterns, closed_patterns =             build_memory_block(conn)

        rng = f"{start or 'all'} to {end or 'all'}"
        print(f"{n} unattached flags ({rng})  |  memory: {len(active_patterns)} open + "
              f"{len(held_patterns)} held + {len(closed_patterns)} decided")
        print(f"Models: cluster={cluster_model}  consolidate={consolidate_model}\n")

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        papers_block, ordered_flags = _format_papers(flags)

        print("=== Step 1: cluster (recall) ===\n")
        clusters, cost1 = run_cluster_step(client, papers_block, n, seed_text, cluster_model,
                                           prompt=cluster_prompt)
        print(f"\n[step 1 cost: ${cost1:.4f}]\n")

        print("=== Step 2: consolidate (choice) ===")
        candidates, cost2 = run_consolidate_step(client, clusters, seed_text, existing_block,
                                               consolidate_model, prompt=consolidate_prompt)
        total = cost1 + cost2

        # Everything below has already been PAID FOR, and the report is the only place some
        # of it ever lands: the raw cluster text, and every `hold` (a held candidate touches
        # no table by design). Recording commits per candidate, so an exception between here
        # and the file write would leave a half-written consolidation AND no record of what
        # was proposed -- and a re-run cannot reproduce it, because the flag pool has moved.
        # So the write happens in a finally, and says so when recording did not finish.
        summary, recorded = None, False
        try:
            if persist:
                # The run row first: patterns reference it, so it has to exist before any is
                # created. Registering the prompt is idempotent (content-addressed), so an
                # unchanged prompt adds no row.
                analysis_prompt_id = db_interface.get_or_create_prompt(
                    conn, analysis_prompt, kind="analysis")
                run_id = db_interface.create_analysis_run(
                    conn, analysis_prompt_id, cluster_model, consolidate_model,
                    profile_id=db_interface.get_or_create_profile(conn, seed_text),
                    date_start=start, date_end=end, n_flags=n, cost_usd=total)
                summary = _record_consolidation(conn, candidates, ordered_flags,
                                                analysis_run_id=run_id)
                for c in summary["new"]:
                    tag = " [act_now]" if c.get("priority") == "act_now" else ""
                    rec = " (recovered)" if c.get("recovered") else ""
                    coerced = (f" (direction {c['direction_coerced']!r} not recognized)"
                               if c.get("direction_coerced") else "")
                    _echo(f"  + new [{c['direction']}] {c['name']}{tag}{rec}{coerced}"
                          f"  ({c['n_flags']} flags)\n")
                for c in summary["merged"]:
                    _echo(f"  ~ merged into {c['id'][:12]} (+{c['added']} flags -> carried)\n")
                for c in summary["recurred"]:
                    _echo(f"  ! closed pattern {c['id'][:12]} recurred (+{c['added']} flags)\n")
                for c in summary["surfaced"]:
                    _echo(f"  ^ held pattern {c['id'][:12]} SURFACED (+{c['added']} papers)\n")
                for c in summary["held"]:
                    back = " (returned)" if c.get("returned") else ""
                    _echo(f"  . held{back}: {c.get('name') or c.get('rationale')}\n")
                for c in summary["discarded"]:
                    _echo(f"  x discarded (not a real pattern): "
                          f"{c.get('name') or c.get('rationale')}\n")
                for c in summary["skipped"]:
                    _echo(f"  x skipped: {c.get('why')}\n")
                _echo(f"[{_summary_line(summary)}  |  total cost: ${total:.4f}]\n")
                recorded = True
            else:
                print(f"[dry run: {len(candidates)} candidates consolidated, nothing recorded  "
                      f"|  total cost: ${total:.4f}]")
                recorded = True
        finally:
            out = _save_report(start, end, n, rng, total, clusters, candidates,
                               cluster_model, consolidate_model, persist, summary, recorded)
    finally:
        conn.close()

    print(f"\nSaved to {out}")
    return out


def _save_report(start, end, n, rng, total, clusters, candidates,
                 cluster_model, consolidate_model, persist, summary, recorded):
    """Write the suggestions markdown and return its path. Called from a finally, so it must
    tolerate a run that died partway: `summary` is None and `recorded` False in that case."""
    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{start or 'all'}_{end or 'all'}"

    def _short(model_id):
        return model_id.replace("claude-", "").replace("/", "-")

    if not recorded:
        tail = ("INCOMPLETE -- the run raised during recording. The database may hold a "
                "PARTIAL consolidation; this file is the only record of what was proposed.")
    elif persist:
        tail = _summary_line(summary)
    else:
        tail = "DRY RUN (nothing recorded)"

    # Both models in the name: swapping only the consolidate model must not clobber the
    # previous report, or a model A/B is unreadable.
    #
    # And a TIMESTAMP, because the window plus the models did not make the name unique: two
    # runs over the same flags with the same models -- a dry run and the real one, or the same
    # window before and after a prompt edit -- wrote the same path and the second silently
    # replaced the first. Timestamped and never overwritten, like the harness reports.
    # `dryrun` in the name so the previewing runs are distinguishable from the one that wrote.
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    kind = "pattern_suggestions" if persist else "pattern_suggestions_dryrun"
    out = (SUGGESTIONS_DIR /
           f"{kind}_{slug}_{_short(cluster_model)}__{_short(consolidate_model)}_{stamp}.md")
    out.write_text(
        f"# Pattern suggestions\n\n"
        f"Unattached flags: {n}  |  range: {rng}  |  cluster: {cluster_model}  "
        f"consolidate: {consolidate_model}  |  cost: ${total:.4f}  |  {tail}\n\n"
        f"---\n\n## Raw clusters (recall)\n\n{clusters}\n\n"
        f"---\n\n## Consolidation (choices)\n\n{_format_consolidation_md(candidates)}\n",
        encoding="utf-8",
    )
    return out
