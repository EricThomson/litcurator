"""
profile_analysis.py -- synthesize flag patterns into user-profile edit suggestions.

The offline learning path. It reads the numeric flags (the residuals between the
judge and the user), clusters them, and surfaces a few high-impact, evidence-backed
suggestions. It SURFACES; the human authors every word of the actual edit.

Two steps -- the Tao of litcurator: generate cheap-and-broad, select expensive-and-sharp.
  Step 1 (cluster, Sonnet): RECALL -- find every candidate preference pattern.
  Step 2 (distill, Opus): SELECTION -- ruthlessly cut to the few worth acting on,
    each checked against the current profile, biased hard toward doing nothing.

Goodhart guard: this NEVER re-runs the judge on the flags to "validate" an edit --
that is exactly how v1 taught the LLM to game the score. It surfaces evidence only;
if you ever validate an edit, do it on a held-out month, not the flag set.

Reads flags from db_interface.get_flags; reads the active profile from
profile_interface. Output streams to console and saves to
~/.litcurator/suggestions/<range>.md for the human to author edits from.
"""

import math
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from litcurator import db_interface, profile_interface
from litcurator.config import DATA_DIR, USER_JOURNAL_RATINGS

load_dotenv()

# The Tao of litcurator: generate cheap-and-broad, select expensive-and-sharp.
# Step 1 (recall) is Sonnet's strength -- crisp, broad candidate generation.
# Step 2 (selection) is a judgment task where Opus is visibly better -- correct
# root-cause merging and holding the false-negative bias. Same shape as the
# pipeline itself: Haiku gate -> Sonnet judge.
DEFAULT_CLUSTER_MODEL = "claude-sonnet-4-6"
DEFAULT_DISTILL_MODEL = "claude-opus-4-8"

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
A later stage will ruthlessly select the few worth acting on, so do not self-censor -- but a
"pattern" is a regularity across MULTIPLE papers, not one paper's quirk. Merge exact duplicates;
otherwise be thorough.

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


DISTILL_PROMPT = """
Below are candidate seed-edit patterns distilled from a researcher's flags, each with its supporting
papers, how pervasive each is, and the direction/magnitude of the mismatch.

Your job is SELECTION, not generation. You may only CUT and MERGE candidates -- never invent new
ones. You are choosing the few edits actually worth making to the researcher's seed profile.

CHECK EVERY CANDIDATE AGAINST THE CURRENT SEED (appended below the candidates -- it is the source of
truth for what is already covered). Three cases, only one of which is a seed edit:
- Seed already clearly states this preference -> CUT it. Adding more prose for something already
  said is bloat (the v1 failure) and will not help. A preference the seed states CLEARLY but the
  judge keeps getting wrong is a JUDGE-APPLICATION problem (the judge is not reading its own seed),
  NOT a seed edit -- cut it and note in "Considered and cut" as "already clear in seed; judge not
  applying it." Do not recommend adding/broadening prose for it.
- Seed is genuinely SILENT on the point -> ADD is legitimate.
- Seed is genuinely VAGUE or internally ambiguous on the point -> SHARPEN is legitimate (but only if
  the wording is actually unclear, not merely because the judge ignored clear wording).

THE DEFAULT IS TO CHANGE NOTHING. This seed has a documented history of failing from over-editing:
every added line risks vocabulary bloat and drift that degraded earlier versions. An edit must
clearly beat the do-nothing default to survive.

PREFER FALSE NEGATIVES. When unsure whether a change earns its place, LEAVE IT OUT. A real pattern
you omit will resurface in later flags and be caught then -- omission is cheap and self-correcting.
A marginal one you include risks permanently bloating the seed -- inclusion is expensive and
compounding. Bias hard toward omission.

THIS IS A SLOW, PATIENT, RECURRING PROCESS -- not a one-time cleanup. You are not fixing everything
now; you are picking only this round's handful. Anything you pass over is not lost: it resurfaces in
future flags and gets fixed in a later pass. A steady trickle of a few well-chosen edits, compounding
over time, is the whole design. Trying to fix it all at once is the failure mode.

EXCEPTION -- cheap, clean specifics. The "it will recur, so omission is cheap" logic assumes the
pattern recurs at a useful rate AND that encoding it risks bloat. A narrow, unambiguous, NAMED
disinterest -- a specific method or specific bounded topic the user has clearly flagged (e.g. a
particular recording modality) -- breaks both assumptions: it is cheap and safe to encode (one
precise line, near-zero bloat risk) and it may be RARE, not reappearing for a long time. For these,
deferral is EXPENSIVE, not cheap -- you could lose it for many cycles. KEEP a clean, specific,
clearly-flagged disinterest even at low coverage or low frequency, ESPECIALLY when the user left an
explicit note stating it. The false-negative bias is for BROAD, AMBIGUOUS, or hard-to-phrase edits;
it does NOT apply to narrow named specifics. (This exception is only for genuinely narrow, named
items -- a broad topic area is still subject to normal ranking, not auto-kept.)

HOW MANY: most rounds warrant 2-4 edits. More than 5 means you have not merged or cut hard enough;
treat {max_s} as an almost-never-reached ceiling, not a target.

PRECEDENCE (these rules conflict; apply in this order):
1. ALREADY IN THE SEED -> CUT. If the seed already clearly states the preference, cut it. A note or
   recurrence does NOT make an existing seed line worth duplicating. If the seed says it and the
   judge ignores it, that is a judge-application problem, not an edit (note it in the cut block).
   This overrides every keep-pressure below -- note-weight and the cheap-specific exception NEVER
   resurrect an already-covered preference.
2. NOT A GENUINE GAP -> CUT.
3. Among genuine gaps, KEEP only the few highest-impact that clearly beat the do-nothing default.
   The note-weight and cheap-clean-specific exception are BOOSTS WITHIN this step -- they can lift a
   borderline, seed-SILENT gap over the bar (this is what rescues a rare, note-backed disinterest the
   seed does not yet name). They are NOT auto-keeps. In particular: a single data point does not
   overturn a deliberate existing seed rule -- defer it, even if note-backed.

Process:
1. MERGE: candidates a single seed edit would satisfy are one candidate. Collapse shared root causes
   (several topic/method complaints that are really one "X over Y" principle become one).
2. RANK BY REASONING, not a formula. Ask of each cluster: does it reveal a real, generalizable taste
   the seed gets wrong -- one that would correctly re-rank papers the user has not yet flagged? Weigh
   three signals TOGETHER, and let NONE of them trump the others:
     - PERVASIVENESS -- how many papers, and what share of the corpus the cluster spans. A small
       average delta is no reason to dismiss a pattern that shows up almost everywhere: a ~0.1 bias
       across most papers is a systematic profile error well worth surfacing, and outweighs a large
       delta on a lone idiosyncratic paper.
     - MAGNITUDE -- how badly the seed mishandles these papers (delta), and which way.
     - CLARITY -- how unambiguously the flags name a real taste. A single sharply-flagged case can
       justify an edit when it reveals a clean, nameable preference -- e.g. a specific method or
       topic that belongs in the active-disinterest list -- especially when a user note states it
       outright. Frequency is not required here; clarity carries it.
   Pervasive-but-mild, sharp-but-rare, and large-and-recurring are all legitimate ways to earn a
   place. Reason about which one the cluster is; do not collapse the signals into a single score.
   NOTE WEIGHT: an explicit user note is a deliberate, selective act -- the user writes few, so a
   note is a strong signal of real taste, stronger than the delta alone. Let it lift a clear,
   seed-silent pattern over the bar (subject to the precedence above), not as a trump card. Do NOT
   transcribe the note's wording -- state a general principle; the user authors the seed line from it.
3. CUT everything that does not clearly beat the do-nothing default, applying the false-negative bias.

Output:
- A single list, ranked by impact (1 = highest). For each survivor, give:
    - THE DIRECTIVE: one clear action, tagged ADD / SUPPRESS / SHARPEN. State the preference itself,
      in plain self-contained terms, the way the researcher would describe their own taste. Do NOT
      phrase it as an instruction to modify existing seed wording, and do NOT adopt the judge's
      framing or vocabulary (its reasoning may be the error). Name the taste that is mis-scored; do
      not draft the edit, pick the clause, or assume how the researcher categorizes things.
    - A SHORT EXPOSITION: 2-3 sentences on the reasoning -- what the flags reveal, why this taste
      holds, and how it generalizes beyond the specific papers. Enough for the researcher to think
      with, not just a verdict. Substantive, not filler: earn every sentence, no padding or restating.
      Cut first, expound second: being able to write a justification is NEVER a reason to keep an
      item. Only write exposition for things that already survived the precedence and ranking above.
    - SUPPORT: (N papers; note whether it is pervasive across the corpus or a sharp isolated case).
    - Strip ALL paper-specific references from BOTH the directive and the exposition (numbers, titles,
      journals, paradigms named only in those papers). State general principles.
- Then a final line "Considered and cut:" naming the strongest candidates you rejected and why in a
  few words each (redundant / idiosyncratic, does not generalize / low-impact / does not beat
  do-nothing / already in seed).

Output ONLY the final ranked list followed by the single "Considered and cut:" block. Do NOT show
drafts, working, or revisions; do NOT restate or echo these instructions; do NOT write any preamble
or transition lines. If you reconsider while composing, emit only the final result -- never both a
draft and a final. ASCII only.
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
    neg = [f for f in flags if f["delta"] < -DELTA_THRESHOLD]
    pos = [f for f in flags if f["delta"] > DELTA_THRESHOLD]
    near = [f for f in flags if abs(f["delta"]) <= DELTA_THRESHOLD]

    sections = []
    idx = 1

    def render(group, header):
        nonlocal idx
        if not group:
            return
        lines = [header]
        for f in group:
            note_line = f"\n   YOUR NOTE: {f['note']}" if f.get("note") else ""
            mismatch_line = (f"\n   POSSIBLE MISMATCH: {f['possible_mismatch']}"
                             if f.get("possible_mismatch") else "")
            abstract = (f.get("abstract") or "")[:500]
            lines.append(
                f"[{idx}] delta {f['delta']:+.2f}  "
                f"(judge {f['judge_score']:.2f} -> you {f['your_score']:.2f})\n"
                f"   Title: {f.get('title') or '(no title)'}\n"
                f"   Journal: {f.get('journal') or ''}  |  {f.get('pub_date_iso') or ''}\n"
                f"   Abstract: {abstract}\n"
                f"   JUDGE RATIONALE: {f.get('rationale') or ''}"
                f"{mismatch_line}"
                f"{note_line}"
            )
            idx += 1
        sections.append("\n\n".join(lines))

    render(sorted(neg, key=lambda f: f["delta"]),
           "## JUDGE SCORED TOO HIGH (you scored lower -- seed over-triggering)")
    render(sorted(pos, key=lambda f: f["delta"], reverse=True),
           "## JUDGE SCORED TOO LOW (you scored higher -- seed missing coverage)")
    render(near,
           f"## ROUGHLY AGREED (|delta| <= {DELTA_THRESHOLD}) -- provided for context")

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _cost(model, usage):
    cin, cout = MODEL_COSTS.get(model, (3.0, 15.0))
    return (usage.input_tokens * cin + usage.output_tokens * cout) / 1_000_000


def _stream(client, model, system, user_msg, max_tokens):
    parts = []
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            parts.append(text)
        final = stream.get_final_message()
    print()
    return "".join(parts), _cost(model, final.usage)


def run_cluster_step(client, flags, seed_text, max_s, model):
    papers_block = _format_papers(flags)
    journal_block = _format_journal_ratings()
    system = CLUSTER_PROMPT.format(max_s=max_s)
    user_msg = (
        f"## Current seed profile\n\n{seed_text}\n\n"
        f"---\n\n"
        f"{journal_block}\n\n"
        f"---\n\n"
        f"## Flagged papers ({len(flags)} total)\n\n{papers_block}"
    )
    # Step 1 is the generous recall stage and scales with flag count; give it
    # room so it is never truncated mid-pattern. Step 2 then ruthlessly selects.
    return _stream(client, model, system, user_msg, max_tokens=6000)


def run_distill_step(client, clusters_text, seed_text, max_s, model):
    # Step 2 must see the seed so it can cut suggestions already covered by it
    # (and tell a real seed gap from the judge failing to apply clear seed text).
    user_msg = (
        f"{clusters_text}\n\n---\n\n"
        f"## CURRENT SEED PROFILE (source of truth -- check candidates against this)\n\n{seed_text}"
    )
    return _stream(client, model, DISTILL_PROMPT.format(max_s=max_s), user_msg, max_tokens=4000)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def suggest_edits(start=None, end=None, max_patterns=None,
                  cluster_model=DEFAULT_CLUSTER_MODEL, distill_model=DEFAULT_DISTILL_MODEL,
                  only_uningested=True):
    """Cluster the flags in [start, end] and surface ranked seed-edit suggestions.
    Streams to console and saves a dated markdown report. Returns the output path
    (or None if there are too few flags). Never re-validates on the flag set."""
    seed_text = profile_interface.load_active()

    conn = db_interface.get_connection()
    try:
        flags = db_interface.get_flags(conn, only_uningested=only_uningested, start=start, end=end)
    finally:
        conn.close()

    n = len(flags)
    if n < MIN_FLAGS:
        print(f"Only {n} flags in range -- need at least {MIN_FLAGS} to run.")
        return None

    max_s = max_patterns if max_patterns is not None else math.floor(n / 2)
    rng = f"{start or 'all'} to {end or 'all'}"
    print(f"{n} flags ({rng})  |  pattern range 1-{max_s}")
    print(f"Models: cluster={cluster_model}  distill={distill_model}\n")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print("=== Step 1: cluster (recall) ===\n")
    clusters, cost1 = run_cluster_step(client, flags, seed_text, max_s, cluster_model)
    print(f"\n[step 1 cost: ${cost1:.4f}]\n")

    print("=== Step 2: distill (selection) ===\n")
    distilled, cost2 = run_distill_step(client, clusters, seed_text, max_s, distill_model)
    total = cost1 + cost2
    print(f"\n[step 2 cost: ${cost2:.4f}  |  total: ${total:.4f}]")

    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{start or 'all'}_{end or 'all'}"

    def _short(model_id):
        return model_id.replace("claude-", "").replace("/", "-")

    out = SUGGESTIONS_DIR / f"seed_suggestions_{slug}_{_short(cluster_model)}__{_short(distill_model)}.md"
    out.write_text(
        f"# Seed edit suggestions\n\n"
        f"Flags: {n}  |  range: {rng}  |  pattern range 1-{max_s}  |  "
        f"cluster: {cluster_model}  distill: {distill_model}  |  cost: ${total:.4f}\n\n"
        f"---\n\n## Raw clusters (step 1)\n\n{clusters}\n\n"
        f"---\n\n## Distilled suggestions (step 2)\n\n{distilled}\n",
        encoding="utf-8",
    )
    print(f"\nSaved to {out}")
    return out
