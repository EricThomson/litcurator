"""
error_analysis.py -- synthesize flag patterns into the pattern memory.

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
import pathlib
import random
import re
import sys
from datetime import datetime

import anthropic
from dotenv import load_dotenv

from litcurator import (analysis_prompt_interface, db_interface, pick_prompt_interface,
                        profile_interface, prompt_interface)
from litcurator.config import BEST_OF_RUNS, DATA_DIR, MAX_ACT_NOW

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
#
# CORRECTED 2026-08-30: claude-opus-4-8 was listed at (15.0, 75.0), three times its actual
# price. Nothing used Opus, so the error was invisible -- until the first Opus consolidate run
# would have reported a cost near triple the truth, on the one number you would use to decide
# whether the upgrade is affordable. Verified against the Anthropic pricing table.
# NOT LISTED, DELIBERATELY: claude-fable-5. It refuses biology content -- `refusal` is a real
# stop reason with a `bio` category -- and every paper this system reads is neuroscience. Working
# around it means server-side fallbacks, which is machinery added to accommodate a model that
# should not be in this loop. Opus is the ceiling here.
MODEL_COSTS = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
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

    # `#` not `##`: one level ABOVE the headings inside the profile and the judge prompt, which
    # are embedded whole in the same message. At the same level their "## Output" and
    # "## Topic interests" read as siblings of the message's own sections.
    lines = ["# Existing pattern memory (match candidates against these by MEANING, using the id)"]
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
    # An unlisted model is priced as Sonnet 4.6, which UNDER-reports every Opus-tier model.
    # Add the row rather than trusting the fallback when trying a new one.
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


def _stream(client, model, system, user_msg, max_tokens, echo=True):
    """`echo=False` still streams (which is what keeps a long generation off the request
    timeout) but does not print. Used when several rounds run back to back: three cluster
    dumps scrolling past is not something anyone reads, and the prose is in each report."""
    parts = []
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            if echo:
                _echo(text)
            parts.append(text)
        final = stream.get_final_message()
    if echo:
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


_CANDIDATE_START = re.compile(r"^[ 	]*(?:\*\*)?NAME[ 	]*:", re.M)


def shuffle_cluster_candidates(text, seed):
    """Reorder the candidate blocks cluster produced. Returns (text, order), where order[i] is
    the ORIGINAL 1-based position of the candidate now presented i+1th. order is [] if the text
    could not be split, in which case the text comes back untouched.

    A DIAGNOSTIC, not machinery, and off unless --shuffle-candidates is passed.

    WHAT IT IS FOR. The consolidate step supplies a `rank`, and on the first real run that rank
    correlated with the order cluster happened to present its candidates in at rho = 0.79 --
    seven of cluster's first eight got ranks 1-6 and 8, seven of its last eight got 9-14. The
    pattern with the MOST evidence in the set (3 papers) came out 13th, because cluster
    mentioned it last. Nothing asks cluster to order by importance, so that sequence is
    arbitrary, and a rank anchored to it is not a judgment.

    Shuffling separates two explanations that look identical in one run: consolidate ANCHORS on
    presentation order (rank will follow the shuffle), or it ranks genuinely and the first run
    was coincidence (rank will hold roughly steady against the shuffle). One dry run either way,
    and it decides whether ranking has to move to a step of its own."""
    marks = [m.start() for m in _CANDIDATE_START.finditer(text)]
    if len(marks) < 2:
        return text, []
    preamble = text[:marks[0]]
    blocks = [text[a:b] for a, b in zip(marks, marks[1:] + [len(text)])]
    order = list(range(len(blocks)))
    random.Random(seed).shuffle(order)
    return preamble + "".join(blocks[i] for i in order), [i + 1 for i in order]


def clusters_from_report(path):
    """Pull the raw cluster text back out of a saved suggestions report.

    THE REPORT IS THE CACHE. Re-running cluster to compare two consolidate models wastes the
    expensive half of the run -- and worse, confounds the comparison: cluster is stochastic, so
    each run hands the models a DIFFERENT candidate set. The 2026-08-30 Sonnet pair both produced
    14 candidates, but not the same 14, so even that A/B was measuring two things at once.

    Returns (clusters_text, n_flags_at_the_time). The flag count comes from the report header and
    is checked by the caller: the [N] paper numbers inside the cluster text index into the ordered
    flag list, so reusing them against a different flag set would silently mis-map every pattern
    to the wrong papers."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if "## Raw clusters (recall)" not in text or "## Consolidation" not in text:
        raise ValueError(f"{path} does not look like a suggestions report "
                         f"(no '## Raw clusters (recall)' / '## Consolidation' sections)")
    body = text.split("## Raw clusters (recall)", 1)[1].split("## Consolidation", 1)[0]
    body = body.rstrip().removesuffix("---").rstrip()
    m = re.search(r"Unattached flags:\s*(\d+)", text)
    return body.strip(), (int(m.group(1)) if m else None)


def run_cluster_step(client, papers_block, n_flags, seed_text, model, prompt=None, echo=True):
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
    return _stream(client, model, prompt, user_msg, max_tokens=6000, echo=echo)


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
                        # WHICH ARTIFACT TO FIX. Decidable from what this step already
                        # holds -- the profile is in its context and the note usually says
                        # it outright -- so it is a lookup, not an etiology judgment, and
                        # the user overrides it in the workbench with a click.
                        "blame": {"type": "string", "enum": list(db_interface.BLAMES),
                            "description": "'profile' if the profile does not state this "
                                           "taste or states it too vaguely -- the fix is "
                                           "profile prose. 'prompt' if the profile ALREADY "
                                           "states it clearly and the judge scored against "
                                           "it anyway -- then more profile prose will not "
                                           "help and the scoring procedure is at fault. The "
                                           "user's note is the best evidence: 'my profile "
                                           "literally says X' means prompt; 'I wasn't clear "
                                           "in my profile' means profile"},
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
                    # rank is REQUIRED. Left optional it would be silently omissible, and the cap
                    # would quietly degrade from "cut by the model's judgment" to "cut in
                    # cluster's arrival order" -- a fallback that looks like it works.
                    "required": ["choice", "rank", "paper_numbers", "rationale"],
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
                         prompt=None, judge_prompt_text=None):
    """Assign every candidate a choice via forced tool-use (so the JSON is always
    valid). Shown the clusters, the profile, the judge prompt, and the existing patterns +
    closed patterns WITH ids (to capture the cross-round match). Returns (candidates, cost).

    `prompt` defaults to the in-code seed; the live path passes the active consolidate section
    from disk, the harness passes a draft. See run_cluster_step.

    BOTH JUDGE ARTIFACTS, added 2026-09-18, because `blame` cannot be answered from one.
    The field asks whether a taste is already written down, and four of the five rule families
    (journal tier, article type, level of organization, score bands) live in the judge prompt
    as well as the profile -- some, like the tier table, ONLY there. Shown the profile alone,
    this step cannot tell "stated nowhere" (a real profile gap) from "stated in the prompt and
    ignored" (a prompt job), which is precisely the distinction blame exists to draw. The
    Annual Review pattern is the worked example: its rule lives in the judge prompt, the user
    fixed it in the judge prompt, and a consolidate step that had never read that file would
    have found no journal rules in the profile and called it a profile gap.

    `judge_prompt_text=None` omits the block, which is what the harness wants -- its scenarios
    are synthetic and have no meaningful judge prompt, and its profile is deliberately silent
    on every planted gap.

    NB the FULL authored file is sent, including the tail below `## Output` that
    judge._batch_prompt drops before the judge ever sees it (1198 of 7757 chars today). That
    is the right choice here -- the file is the artifact the user edits, so "is this written
    down" means written down in it -- but it is a live discrepancy worth closing at the judge
    end rather than papering over at this one."""
    prompt = _require_prompt(prompt, "consolidate")
    memory = f"{existing_block}\n\n---\n\n" if existing_block else ""
    tool = _consolidate_tool(has_memory=bool(memory))
    # LABELS, NOT INSTRUCTIONS. The header used to append "a preference already clear here
    # that the judge still gets wrong is judge-not-applying, not a gap" -- prompt text living
    # in code, outside the hand-authored content-addressed prompt file, and the strongest
    # single push toward the value removed on 2026-08-25. What to DO with these two blocks
    # belongs in the authored consolidate prompt, not here.
    judge_block = (
        f"# THE JUDGE PROMPT (the scoring procedure the judge follows)\n\n"
        f"This is what is currently written, not a claim that it is right.\n\n"
        f"{judge_prompt_text}\n\n---\n\n"
    ) if judge_prompt_text else ""
    user_msg = (
        f"# Candidate patterns (with [N] paper numbers)\n\n{clusters_text}\n\n---\n\n"
        f"{memory}"
        f"# CURRENT PROFILE (the user's own prose about what they want to read)"
        f"\n\n{seed_text}\n\n---\n\n"
        f"{judge_block}".rstrip("-\n ")
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

    _apply_queue_cap(candidates)

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
            flag_ids=flag_ids, note=note or None, analysis_run_id=analysis_run_id,
            blame=(c.get("blame") if c.get("blame") in db_interface.BLAMES else "profile"),
            # Kept so the workbench can show the queue in the order the model ranked it.
            rank=c["rank"] if isinstance(c.get("rank"), int) else None)

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
            db_interface.add_pattern_event(conn, pid, "held", note=c.get("rationale"),
                                           analysis_run_id=analysis_run_id)
        entry = {"id": pid, "name": _fallback_name(c),
                 "direction": direction, "priority": c.get("priority"),
                 "blame": c.get("blame") or "profile",
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
            added = db_interface.attach_flags_to_pattern(conn, eid, flag_ids,
                                                         analysis_run_id=analysis_run_id)
            if not added:
                # Target already covers every one of these PAPERS: nothing new arrived, so no
                # event. This is the re-run idempotency guard, not a lost candidate -- and
                # since 2026-08-26 `added` counts papers rather than pattern_flags rows, so a
                # re-flag of a paper the pattern already holds no longer fires one either.
                summary["skipped"].append({"id": eid, "why": "no new papers to attach"})
                continue
            if st in db_interface.CLOSED_STATUSES:
                db_interface.add_pattern_event(conn, eid, "recurred", note=c.get("rationale"),
                                               analysis_run_id=analysis_run_id)
                summary["recurred"].append({"id": eid, "name": _fallback_name(c), "added": added})
            elif st in db_interface.HELD_STATUSES and surface_of(c) == "hold":
                # Came back, still not actionable. Logged rather than silent, so the history
                # reads held -> held -> carried and "this has been sitting for three rounds"
                # is answerable.
                db_interface.add_pattern_event(conn, eid, "held", analysis_run_id=analysis_run_id,
                                               note=f"returned: {c.get('rationale', '')}")
                summary["held"].append({"id": eid, "name": _fallback_name(c), "added": added,
                                        "returned": True})
            else:
                # PROMOTION happens here, and needs no threshold: 'carried' is an active status,
                # so a held pattern the model now wants shown simply becomes visible. An
                # already-open pattern was visible anyway and stays so.
                db_interface.add_pattern_event(conn, eid, "carried", analysis_run_id=analysis_run_id,
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


def _apply_queue_cap(candidates):
    """Demote the act_now overflow to hold, by the model's own RANK. Mutates in place.

    THE MODEL RANKS, CODE CUTS. Instructing the limit did not work: two January dry runs put ten
    patterns in the queue against a stated cap of eight, at exactly ten both times. Asking one
    forced tool call to hold a running total across output it has not finished emitting is a
    poor shape for a generator. Ranking is something it CAN do while emitting, so the tool asks
    for a rank and the arithmetic happens here -- the same division as computed_sign, where the
    model judges and code owns what is computable.

    CALLED FROM BOTH PATHS, which is the point of it living out here. It used to sit inside
    _record_consolidation, so a DRY RUN -- the mode whose entire purpose is previewing what
    would happen -- was the one mode the cap never applied in, and its report showed twelve
    act_now patterns that a real run would have cut to eight. A preview that does not preview
    is worse than none.

    Only `new` candidates are capped: only they add a card to the queue, since a merge lands on
    one already there. A model `hold` is never overridden -- this is a ceiling, not a quota, so
    it demotes and never promotes. Overflow keeps its row, its provenance and its place in the
    Held tab, and returns next round. The model's own word is preserved beside it so a demotion
    is legible rather than a silent correction, and idempotent: re-running finds the demoted
    ones already held and does nothing."""
    wants_queue = [c for c in candidates
                   if c.get("choice") == "new" and (c.get("priority") or "act_now") != "hold"]
    if len(wants_queue) <= MAX_ACT_NOW:
        return
    ranked = sorted(
        enumerate(wants_queue),
        key=lambda ic: (ic[1]["rank"] if isinstance(ic[1].get("rank"), int) else 10 ** 6, ic[0]))
    for _pos, c in ranked[MAX_ACT_NOW:]:
        c["priority_asked"] = c.get("priority") or "act_now"
        c["priority"] = "hold"
        c["rationale"] = ((c.get("rationale") or "").rstrip()
                          + f" [demoted: over the {MAX_ACT_NOW}-pattern queue cap]").strip()


_HEAD = re.compile(
    r"^- \*\*(?P<choice>\w+)\*\*"
    r"(?: -- (?P<name>.*?))?"
    r"(?: \((?P<direction>over|under)\))?"
    r"(?P<blame> \{PROMPT\})?"
    r"(?: #(?P<rank>\d+))?"
    r"(?: \[(?P<priority>\w+)\])?"
    r"(?: \(asked (?P<asked>\w+)\))?"
    r"(?:  -> (?P<target>.*))?$")

_FIELD_LABELS = {"desc": "description", "edit": "suggested_edit", "why": "rationale"}


def parse_consolidation_md(path):
    """Read a suggestions report back into the candidate list that produced it.

    THE REPORT IS THE RECORD. A dry run makes both LLM calls and then throws the parsed
    candidates away, so a run you LIKED could only be recovered by paying again -- and
    a round is unstable enough that you would get a different one. On 2026-08-30 a run
    the user called a shit show was followed immediately by one he called amazing, with
    NOTHING changed between them. At that variance best-of-N is the sensible workflow, and it
    only works if a good run can be kept.

    Parsing our own prose is safe only because record_stage round-trips it -- it renders
    candidates, parses them back and asserts they match -- so a renderer change fails loudly
    instead of silently mis-recording a round.

    Returns (candidates, n_flags_at_the_time)."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if "## Consolidation (choices)" not in text:
        raise ValueError(f"{path} has no '## Consolidation (choices)' section -- "
                         f"not a suggestions report?")
    m = re.search(r"Unattached flags:\s*(\d+)", text)
    n_then = int(m.group(1)) if m else None

    out, cur = [], None
    for line in text.split("## Consolidation (choices)", 1)[1].splitlines():
        head = _HEAD.match(line)
        if head:
            if cur:
                out.append(cur)
            g = head.groupdict()
            cur = {"choice": g["choice"]}
            if g["name"]:
                cur["name"] = g["name"]
            if g["direction"]:
                cur["direction"] = g["direction"]
            if g["blame"]:
                cur["blame"] = "prompt"
            if g["rank"]:
                cur["rank"] = int(g["rank"])
            if g["priority"]:
                # The RENDERED priority is post-cap; `asked` carries what the model wanted.
                # Restoring the model's own word means re-recording reproduces the demotion
                # from the rank, rather than baking in a cap that may since have changed.
                cur["priority"] = g["asked"] or g["priority"]
            if g["target"]:
                t = g["target"].strip()
                if t.startswith("NOT A PATTERN ID: "):
                    t = t[len("NOT A PATTERN ID: "):].strip().strip("'\"")
                cur["existing_pattern_id"] = t
        elif cur is not None and line.startswith("    - "):
            label, _, value = line[6:].partition(": ")
            if label == "papers":
                cur["paper_numbers"] = [int(x) for x in re.findall(r"\d+", value)]
            elif label in _FIELD_LABELS:
                cur[_FIELD_LABELS[label]] = value
    if cur:
        out.append(cur)
    if not out:
        raise ValueError(f"{path}: found the Consolidation section but no candidates in it")
    return out, n_then


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
        # One token, so a prompt-blamed pattern survives the report round-trip that
        # promote_suggestions depends on.
        if c.get("blame") == "prompt":
            head += " {PROMPT}"
        if isinstance(c.get("rank"), int):
            head += f" #{c['rank']}"
        if c.get("priority"):
            head += f" [{c['priority']}]"
            if c.get("priority_asked") and c["priority_asked"] != c["priority"]:
                head += f" (asked {c['priority_asked']})"
        if c.get("existing_pattern_id"):
            # A REAL id is truncated for readability; anything else is shown WHOLE and marked.
            # It used to truncate unconditionally, which is fine for a 32-char uuid and actively
            # misleading for an invented target: the 2026-08-27 dry run rendered a hallucinated
            # value as "human-only-s", which reads like a real id prefix and hid what the model
            # actually emitted. The report is the only record of a dry run, so it has to show
            # the evidence rather than a tidy-looking slice of it.
            # IN FULL, not truncated. It used to show a 12-char prefix, which is fine for
            # reading and wrong for a file that `promote_suggestions` parses back: a prefix
            # cannot be re-recorded without a database lookup, and the report is meant to be
            # self-contained. An invalid target is still marked, which is why truncation was
            # removed in the first place -- a hallucinated value rendered as "human-only-s"
            # reads exactly like a real id prefix.
            eid = c["existing_pattern_id"]
            head += (f"  -> {eid}" if _looks_like_pattern_id(eid)
                     else f"  -> NOT A PATTERN ID: {eid!r}")
        lines.append(head)
        # ONE LINE EACH, whitespace collapsed. The report is not only for reading: it is the
        # input to `promote_suggestions`, which parses it back to re-record a run you liked
        # rather than paying to re-roll one. A model-written field containing a newline would
        # split into two lines and silently truncate on the way back.
        for label, key in (("desc", "description"), ("edit", "suggested_edit"),
                           ("why", "rationale")):
            if c.get(key):
                lines.append(f"    - {label}: {' '.join(str(c[key]).split())}")
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
                  persist=True, shuffle_seed=None, include_attached=False,
                  reuse_clusters=None, best_of=None, pick_model=None):
    """Cluster the UNATTACHED (not-yet-patterned) flags in [start, end], consolidate each
    candidate against the pattern memory, and RECORD every real one (new / merge into an
    open pattern / recurs against a closed pattern).

    RUNS THE WHOLE THING `best_of` TIMES AND PICKS ONE (config.BEST_OF_RUNS, default 3).
    A round is unreliable enough that a single one is a lottery -- two runs on identical
    input, minutes apart, produced one the user called a shit show and one he called amazing.
    The instability is mostly CLUSTER's, not consolidate's, which is why all three rounds run
    the whole thing rather than sharing one cluster draw. So the round is run several times, a picker reads them anonymised and
    chooses, and the winner is what gets recorded. Every round is written to disk first, so a
    failure in the pick or the record costs money but never evidence. best_of=1 skips the
    picker entirely and needs no pick prompt.

    Returns the round DIRECTORY (or None if too few flags), holding run_1.md ... run_N.md and
    verdict.md. Never re-validates on the flag set. persist=False is a dry run (writes
    everything, records nothing).

    The analysis prompt is loaded from disk and REGISTERED, exactly as the pipeline does with
    the judge prompt: every pattern this produces points at an analysis_run, which points at
    the prompt content that made it. Without that, "which prompt produced this pattern" is
    unanswerable, which was the one provenance gap in the system."""
    best_of = BEST_OF_RUNS if best_of is None else best_of
    if best_of < 1:
        raise ValueError(f"best_of must be at least 1, got {best_of}")
    if include_attached and persist:
        raise ValueError(
            "include_attached is a comparison mode and cannot be recorded: clustering flags that "
            "are already attached would mint duplicate patterns over the same papers. Re-run with "
            "--dry-run.")
    seed_text = profile_interface.load_active()
    analysis_prompt = analysis_prompt_interface.load_active()
    cluster_prompt, consolidate_prompt = analysis_prompt_interface.split(analysis_prompt)
    # The OTHER judge artifact. Consolidate needs both to answer `blame` -- see
    # run_consolidate_step. Cluster deliberately does NOT get it: its job is recall from the
    # papers and the user's notes, and handing the step that hunts for miscalibration the
    # current calibration is the mistake the journal-ratings table made in 2026-08.
    judge_prompt_text = prompt_interface.load_active()
    # Stamped into every run report. promote_suggestions records a report by hand, possibly
    # days later, and without these it registered whatever was active THAT day -- attaching a
    # round to artifacts it never ran under. The report is meant to be self-contained; this is
    # the same argument that stopped merge targets being truncated to 12 chars.
    report_stamps = {
        "analysis prompt": analysis_prompt_interface.content_hash(analysis_prompt),
        "profile": profile_interface.content_hash(seed_text),
        "judge prompt": prompt_interface.content_hash(judge_prompt_text),
    }

    conn = db_interface.get_connection()
    try:
        # UNATTACHED flags only: a flag already attached to a pattern is "handled" and must
        # not re-cluster into a duplicate candidate. This is the bloat/idempotency bound.
        # INCLUDE_ATTACHED is a COMPARISON mode, not a round. It clusters every flag in the
        # window rather than only the unattached ones, so the same evidence can be re-run under a
        # different prompt or model without disturbing state -- the need that kept arising as
        # resetting the database, which is heavier and costs whatever else happened since.
        #
        # It CANNOT persist, and that is enforced below rather than documented: recording a
        # round over papers already attached to patterns would mint duplicates of them, which is
        # the one thing the unattached pool exists to prevent.
        flags = db_interface.get_flags(conn, start=start, end=end,
                                       exclude_attached=not include_attached)
        n = len(flags)
        if n < MIN_FLAGS:
            print(f"Only {n} unattached (not-yet-patterned) flags in range -- need at least "
                  f"{MIN_FLAGS} to run.")
            return None

        # The pattern memory, shown to consolidate WITH ids so it captures cross-round
        # matches (merge into an open pattern / recurs against a closed pattern).
        existing_block, active_patterns, held_patterns, closed_patterns = (
            build_memory_block(conn))
        if include_attached:
            # FRESH EYES, not just the old flags. The point of the mode is to re-run past
            # evidence under a different prompt or model and compare against what a previous
            # round produced -- and that round saw an EMPTY memory. Leaving the memory in place
            # made the first Opus comparison (2026-08-30) meaningless: it was shown twelve
            # patterns built from these very flags, correctly merged all twelve, and produced
            # nothing comparable to the Sonnet runs it was meant to be measured against.
            existing_block = ""
            active_patterns = held_patterns = closed_patterns = []

        rng = f"{start or 'all'} to {end or 'all'}"
        if include_attached:
            print("[COMPARISON RUN: all flags, attached included, and NO pattern memory -- "
                  "same evidence, fresh eyes. Nothing will be recorded.]")
        print(f"{n} flags ({rng})  |  memory: {len(active_patterns)} open + "
              f"{len(held_patterns)} held + {len(closed_patterns)} decided")
        print(f"Models: cluster={cluster_model}  consolidate={consolidate_model}\n")

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        papers_block, ordered_flags = _format_papers(flags)

        # ONE DIRECTORY PER INVOCATION, holding every round it ran plus the verdict. Which
        # reports belong to the same round used to be answerable only by comparing timestamps
        # in a flat directory and hoping; now it is structural. The round is the unit.
        round_dir = SUGGESTIONS_DIR / f"round_{datetime.now():%Y%m%d_%H%M%S}"
        round_dir.mkdir(parents=True, exist_ok=True)
        rounds, round_cost = [], 0.0

        for run_number in range(1, best_of + 1):
            if best_of > 1:
                print(f"\n=== ROUND {run_number} of {best_of} ===")
            clusters, cost1 = _cluster_for_round(
                client, papers_block, n, seed_text, cluster_model, cluster_prompt,
                reuse_clusters, echo=(best_of == 1))

            # DIAGNOSTIC. Reorders the candidate blocks so `rank` can be correlated against the
            # order they were PRESENTED in rather than the order cluster wrote them. See
            # shuffle_cluster_candidates.
            shown, shuffle_order = ((clusters, []) if shuffle_seed is None
                                    else shuffle_cluster_candidates(clusters, shuffle_seed))
            if shuffle_order:
                print(f"[candidates shuffled, seed {shuffle_seed}: presentation order = "
                      f"{shuffle_order}]")
            if best_of > 1:
                print("  consolidating ...", end="", flush=True)
            candidates, cost2 = run_consolidate_step(
                client, shown, seed_text, existing_block, consolidate_model,
                prompt=consolidate_prompt, judge_prompt_text=judge_prompt_text)
            if best_of > 1:
                print(f" {len(candidates)} patterns  ${cost2:.4f}")
            # BEFORE anything is written, so every report previews the real outcome. Recording
            # calls it again; it is idempotent.
            _apply_queue_cap(candidates)
            total = cost1 + cost2
            round_cost += total

            # SAVED IMMEDIATELY, before any picking or recording. The raw cluster text and every
            # `hold` land nowhere else, the flag pool moves once anything is recorded, and a
            # re-run cannot reproduce what was just paid for. Writing here means a later failure
            # in the pick or the record costs money but never evidence.
            path = _save_report(
                round_dir / f"run_{run_number}.md", n, rng, total, shown, candidates,
                cluster_model, consolidate_model,
                tail=(f"run {run_number} of {best_of} -- see verdict.md" if best_of > 1
                      else ("DRY RUN (nothing recorded)" if not persist else "single round")),
                shuffle_seed=shuffle_seed, shuffle_order=shuffle_order,
                stamps=report_stamps)
            rounds.append((path, candidates))
            if best_of > 1:
                print(f"  -> {path.name}  (${total:.4f})")

        winner, verdict, pick_cost, by_label = _pick_winner(rounds, start, end, pick_model)
        round_cost += pick_cost

        # A round the picker calls bad is the one case where recording it anyway would ignore
        # the signal this whole step exists to produce. Nothing is lost: the reports are on
        # disk and `promote_suggestions` records any of them by hand.
        blocked = bool(verdict and verdict.get("none_are_good"))
        summary = None
        if persist and not blocked:
            winner_candidates = next(c for p, c in rounds if p == winner)
            # The run row first: patterns reference it, so it has to exist before any is
            # created. Registering the prompt is idempotent (content-addressed), so an
            # unchanged prompt adds no row.
            run_id = db_interface.create_analysis_run(
                conn, db_interface.get_or_create_prompt(conn, analysis_prompt, kind="analysis"),
                cluster_model, consolidate_model,
                profile_id=db_interface.get_or_create_profile(conn, seed_text),
                date_start=start, date_end=end, n_flags=n, cost_usd=round_cost,
                # The other two artifacts this round read. The judge prompt is what makes
                # `blame` answerable at all, so a round that does not name its version has
                # made a claim about a file nobody can identify later.
                judge_prompt_id=db_interface.get_or_create_prompt(
                    conn, judge_prompt_text, kind="judge"),
                # NULL when no picker ran (a single round has nothing to pick), because a
                # stamped pick prompt would assert a choice that never happened. `by_label`
                # is the discriminator: _pick_winner returns it only on the paid path.
                pick_prompt_id=(db_interface.get_or_create_prompt(
                    conn, pick_prompt_interface.load_active(), kind="pick")
                    if by_label else None))
            summary = _record_consolidation(conn, winner_candidates, ordered_flags,
                                            analysis_run_id=run_id)
            _echo_record_summary(summary)
        elif blocked:
            print("\nNOT RECORDED: the picker judged every round weak. Read them and, if you "
                  "disagree, record one with `litcurator promote_suggestions`.")
        else:
            print(f"\n[dry run: nothing recorded]")

        out = _write_round_verdict(round_dir, rounds, winner, verdict, summary,
                                   round_cost, persist and not blocked, by_label,
                                   stamps=report_stamps,
                                   pick_stamp=(pick_prompt_interface.content_hash(
                                       pick_prompt_interface.load_active())
                                       if by_label else None))
    finally:
        conn.close()

    print(f"\nRound saved to {round_dir}")
    print(f"Winner: {winner.name}  |  verdict: {out.name}  |  round cost: ${round_cost:.4f}")
    return round_dir


def _cluster_for_round(client, papers_block, n, seed_text, model, prompt, reuse_clusters, echo):
    """Step 1 for one round: cluster, or reuse a saved report's clusters. Returns (text, cost).

    Reusing pins cluster output so several rounds differ only in consolidate. That is a
    COMPARISON tool, not how best-of runs: cluster is where most of the variance lives, so
    holding it fixed would freeze one draw's faults into every round."""
    if reuse_clusters:
        clusters, n_then = clusters_from_report(reuse_clusters)
        if n_then is not None and n_then != n:
            raise ValueError(
                f"{reuse_clusters} clustered {n_then} flags but this run has {n}. The [N] "
                f"paper numbers in that report index into the ordered flag list, so reusing "
                f"them here would map every pattern to the wrong papers.")
        print(f"=== Step 1: SKIPPED, clusters reused from "
              f"{pathlib.Path(reuse_clusters).name} ===")
        return clusters, 0.0
    if echo:
        print("=== Step 1: cluster (recall) ===\n")
    else:
        # Not streaming the prose still means a minute of silence per round, which reads as a
        # hung process. One line that completes when the step does.
        print("  clustering ...", end="", flush=True)
    clusters, cost = run_cluster_step(client, papers_block, n, seed_text, model, prompt=prompt,
                                      echo=echo)
    if echo:
        print(f"\n[step 1 cost: ${cost:.4f}]\n")
    else:
        print(f" {len(_CANDIDATE_START.findall(clusters))} candidates  ${cost:.4f}")
    return clusters, cost


def _pick_winner(rounds, start, end, pick_model):
    """(winner_path, verdict, cost). With one round there is nothing to pick, so no pick prompt
    is needed and nothing is spent -- which is what keeps BEST_OF_RUNS=1 usable before anyone
    has authored one."""
    if len(rounds) == 1:
        return rounds[0][0], None, 0.0, None
    # Lazy: consolidation_picker imports this module.
    from litcurator import consolidation_picker
    print(f"\n=== Picking among {len(rounds)} rounds ===")
    kwargs = {"model": pick_model} if pick_model else {}
    verdict, presented, cost = consolidation_picker.pick_best(
        [p for p, _ in rounds], start=start, end=end, **kwargs)
    by_label = {label: path for label, path, _ in presented}
    winner = by_label[verdict["ranking"][0]]
    print("Presented as: " + ", ".join(f"{lb}={p.name}" for lb, p, _ in presented))
    print(f"Ranking: {' > '.join(verdict['ranking'])}  ->  {winner.name}")
    for entry in verdict["assessments"]:
        print(f"  {entry['run']}: {entry['worst_problem']}")
    print(f"[pick cost: ${cost:.4f}]")
    return winner, verdict, cost, by_label


def _echo_record_summary(summary):
    for c in summary["new"]:
        tag = " [act_now]" if c.get("priority") == "act_now" else ""
        rec = " (recovered)" if c.get("recovered") else ""
        coerced = (f" (direction {c['direction_coerced']!r} not recognized)"
                   if c.get("direction_coerced") else "")
        job = "   ** PROMPT JOB **" if c.get("blame") == "prompt" else ""
        _echo(f"  + new [{c['direction']}] {c['name']}{tag}{rec}{coerced}"
              f"  ({c['n_flags']} flags){job}\n")
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
        _echo(f"  x discarded (not a real pattern): {c.get('name') or c.get('rationale')}\n")
    for c in summary["skipped"]:
        _echo(f"  x skipped: {c.get('why')}\n")
    _echo(f"[{_summary_line(summary)}]\n")


def _write_round_verdict(round_dir, rounds, winner, verdict, summary, cost, recorded,
                         by_label=None, stamps=None, pick_stamp=None):
    """verdict.md -- what this round produced, which run won, and what was written.

    Always written, one round or several: the round directory should say what happened without
    anyone reconstructing it from timestamps. It is the outcome record, which is why the run
    reports no longer carry a recording summary in their own headers."""
    lines = [f"# Round verdict\n", f"{datetime.now():%Y-%m-%d %H:%M:%S}  |  "
             f"{len(rounds)} round(s)  |  total cost: ${cost:.4f}\n"]
    # Every artifact this round read, the PICK prompt included -- it is the only one not known
    # until the rounds are done, which is why it lands here rather than in the run headers.
    # consolidation_picker's standalone verdict already stamped its prompts; this one recorded
    # nothing, so "which pick prompt chose this" was answerable for the side path and not for
    # the one that actually writes patterns.
    artifacts = dict(stamps or {})
    if pick_stamp:
        artifacts["pick prompt"] = pick_stamp
    if artifacts:
        lines.append("artifacts -- " + "  |  ".join(
            f"{k}: {v}" for k, v in artifacts.items()) + "\n")
    if verdict:
        if verdict.get("none_are_good"):
            lines.append("**The picker judged every round weak.** Nothing was recorded. The "
                         "ranking below is a least-bad ordering.\n")
        lines.append(f"WINNER: {winner.name}\n")
        lines.append("## Ranking\n")
        # Filenames beside the shuffled letters: without the mapping, "round B was sloppy"
        # is unreadable a month later -- it only ever existed on the console.
        def _name(label):
            return (f"round {label} -- {by_label[label].name}" if by_label
                    else f"round {label}")
        lines.append("\n".join(f"{place}. {_name(label)}" for place, label in
                               enumerate(verdict["ranking"], start=1)))
        lines.append("\n## Why the winner\n")
        lines.append(verdict["why_the_winner"])
        lines.append("\n## The worst thing in each round\n")
        for entry in verdict["assessments"]:
            lines.append(f"- **{_name(entry['run'])}**: {entry['worst_problem']}")
    else:
        lines.append(f"Single round, nothing to pick: {winner.name}\n")
    lines.append("\n## Recorded\n")
    lines.append(_summary_line(summary) if recorded and summary
                 else "Nothing was recorded. To record one by hand:\n\n"
                      f"    litcurator promote_suggestions {winner}")
    out = round_dir / "verdict.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


ARTIFACT_STAMP_KEYS = ("analysis prompt", "profile", "judge prompt")


def _format_artifact_stamps(stamps):
    """The `artifacts --` header line. One line, fixed key order, so it is greppable by eye and
    by parse_report_stamps. Keys are spelled the way a person would say them; the values are
    12-char short ids, which are the same prefix the DB content-address uses."""
    if not stamps:
        return ""
    return ("artifacts -- " + "  |  ".join(
        f"{k}: {stamps[k]}" for k in ARTIFACT_STAMP_KEYS if stamps.get(k)) + "\n\n")


def parse_report_stamps(path):
    """The artifact short ids a report was produced under, as {key: short_id}.

    Missing keys mean the report predates the stamps (every report before 2026-09-19), which
    the caller must treat as unknown rather than as "the active one" -- that silent substitution
    is the whole defect this header exists to fix."""
    # HEADER ONLY -- everything above the first rule. Searching the whole file would let a
    # cluster paragraph or a model-written rationale supply a "profile: <12 hex>" and be read
    # as provenance, which is a silent wrong attribution rather than a loud failure.
    header = pathlib.Path(path).read_text(encoding="utf-8").split("\n---\n", 1)[0]
    out = {}
    for key in ARTIFACT_STAMP_KEYS:
        m = re.search(rf"{re.escape(key)}:\s*([0-9a-f]{{12}})\b", header)
        if m:
            out[key] = m.group(1)
    return out


def _save_report(out_path, n, rng, total, clusters, candidates,
                 cluster_model, consolidate_model, tail,
                 shuffle_seed=None, shuffle_order=(), stamps=None):
    """Write one round's suggestions markdown and return its path.

    The path is passed in: the round directory owns naming now, and run_1/run_2/run_3 inside it
    carry what the old flat timestamped filename did. What the round RECORDED lives in
    verdict.md rather than in this header, because it is not known until every round has run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# Pattern suggestions\n\n"
        f"Unattached flags: {n}  |  range: {rng}  |  cluster: {cluster_model}  "
        f"consolidate: {consolidate_model}  |  cost: ${total:.4f}  |  {tail}\n\n"
        + _format_artifact_stamps(stamps)
        + (f"CANDIDATES SHUFFLED (seed {shuffle_seed}). The clusters below are in the order "
           f"consolidate SAW them; their original positions in cluster's own output were "
           f"{list(shuffle_order)}. Correlate `rank` against BOTH orders to tell anchoring "
           f"from judgement.\n\n" if shuffle_order else "")
        + f"---\n\n## Raw clusters (recall)\n\n{clusters}\n\n"
        f"---\n\n## Consolidation (choices)\n\n{_format_consolidation_md(candidates)}\n",
        encoding="utf-8",
    )
    return out_path
