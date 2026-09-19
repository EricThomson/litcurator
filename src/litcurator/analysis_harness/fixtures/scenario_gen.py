"""
scenario_gen.py -- compile a declarative ScenarioSpec into the {name, papers, then, expect}
rounds the scenario harness runs.

WHY. The harness driver is already N-round generic; the only thing that gated length was the
hand-written literal in scenarios.py. This module replaces that literal with specs you
author programmatically -- per-session flag STREAMS by intended pattern -- so a 40-session history (a weak
trickle, a growing noise floor) is a few lines, not a typed-out list.

No judge runs here (as in the harness): a synthetic flag is just its content plus a fabricated
(judge_score, user_score) pair whose signed delta (user - judge) lands in the intended pattern's band.
Grading stays provenance-based, so the exact pmid string never matters -- only that it is unique.

Two ways to fill a session:
  - EXPLICIT: hand-authored papers emitted verbatim. Used to reproduce the original lifecycle
    fixture exactly (the parity gate) and for any case that must be pinned.
  - STREAM: draw N papers from an intended pattern's papers each firing session, with light variation. Used
    for the long-horizon behavior scenarios where volume is the point.

The delta sign follows the intended pattern direction, matching db_interface.insert_flag:
  under  -> judge scored too LOW  -> user - judge > 0
  over   -> judge scored too HIGH -> user - judge < 0
|delta| sets how WEAK the signal is, which is the regime the accumulation scenario probes. It
no longer changes how a flag is RENDERED: _format_papers emits one list ordered by |delta|
descending, with no magnitude buckets (2026-08-07).
"""

from dataclasses import dataclass, field
from itertools import count


# ---------------------------------------------------------------------------
# The atoms
# ---------------------------------------------------------------------------

@dataclass
class SyntheticPaper:
    """One hand-authored exemplar for an intended pattern -- the realism anchor the generator varies.
    The `rationale` (the judge's WRONG reason) is load-bearing: the cluster step groups by
    meaning of the rationales, not keyword overlap, so meaning-equivalent rationales are what
    make an intended pattern cohere."""
    title: str
    abstract: str
    journal: str
    rationale: str
    note: str = ""


@dataclass
class IntendedPatternPool:
    """The supply a scenario draws from for ONE intended pattern: the papers to emit, plus the
    shape their flags should take. `direction` fixes the delta sign; `delta_band` fixes the
    |delta| magnitude (the knob the weak-signal experiment sweeps).

    `rationale_templates` matters on long runs. `papers` is cycled, so a twelve-session scenario
    drawing on four papers emits each one three times -- identical title, abstract and rationale,
    differing only in pmid and a little score jitter. Handing the cluster step literal duplicates
    makes grouping easier than reality. These are extra meaning-equal restatements of the same
    complaint, rotated across emissions so the repeats are not verbatim. Mind the arithmetic: the
    paper index is `seq % len(papers)` and the rationale index `seq % (1 + len(templates))`, so if
    those periods share a factor a given paper always draws the same rationale and rotating buys
    nothing."""
    label: str
    direction: str                       # "under" | "over"
    delta_band: tuple                    # (lo, hi) magnitude of |delta|, both in (0, 1)
    papers: list                           # list[SyntheticPaper] -- cycled to emit flags
    rationale_templates: list = field(default_factory=list)   # extra meaning-equal rationales


@dataclass
class Stream:
    """Emit `count_per_session` flags of one intended pattern on each session in `sessions`
    (a range or a list of 0-based session indices)."""
    label: str
    sessions: object                     # range | list[int]
    count_per_session: int = 1


@dataclass
class ScenarioSpec:
    """A whole synthetic history. Sessions are 0-based, 0..n_sessions-1.
      explicit[i]        -> list of full paper dicts emitted verbatim in session i
      streams            -> Streams; each fires in the sessions it names
      then_rules[i]      -> scripted human actions AFTER session i, e.g. [("incorporate","A")]
      per_round_expect[i]-> the per-round expect dict check_round grades
      session_names[i]   -> optional display name for session i
      terminal_expect    -> cross-session assertions graded once at the end (Phase 3)
    A full paper dict has: intended, title, abstract, journal, judge_score, user_score,
    rationale, note (pmid is assigned at compile time)."""
    name: str
    n_sessions: int
    profile: str
    # The synthetic judge prompt consolidate is shown alongside the profile. Defaults to the
    # shared scenarios.JUDGE_PROMPT via behaviors, so a scenario only names it to say something
    # different; None means "run consolidate blind", which is the pre-2026-09-19 configuration
    # and is used deliberately by the blame gate's negative control.
    judge_prompt: str = None
    pools_by_intended_pattern: dict = field(default_factory=dict)          # label -> IntendedPatternPool
    explicit: dict = field(default_factory=dict)         # session_idx -> list[paper dict]
    streams: list = field(default_factory=list)          # list[Stream]
    then_rules: dict = field(default_factory=dict)       # session_idx -> list[(action, label)]
    per_round_expect: dict = field(default_factory=dict) # session_idx -> expect dict
    session_names: dict = field(default_factory=dict)    # session_idx -> str
    terminal_expect: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

PAPER_KEYS = ("intended", "title", "abstract", "journal",
              "judge_score", "user_score", "rationale", "note")


def _scores(direction, delta_mag, rng):
    """A (judge_score, user_score) pair whose signed delta (user - judge) matches the intended pattern:
    +delta_mag for `under`, -delta_mag for `over`. Both clamped into (0, 1)."""
    d = round(delta_mag, 3)
    if direction == "under":                       # judge too low: user sits d above judge
        judge = rng.uniform(0.25, min(0.55, 0.95 - d))
        user = judge + d
    elif direction == "over":                      # judge too high: user sits d below judge
        judge = rng.uniform(max(0.45, d + 0.05), 0.85)
        user = judge - d
    else:
        raise ValueError(f"direction must be 'under' or 'over', got {direction!r}")
    clamp = lambda x: round(max(0.02, min(0.98, x)), 2)
    return clamp(judge), clamp(user)


def _emit(pool, rng, seq):
    """Synthesize one flag dict from a pool. Cycles the papers and rotates the rationale, so a long
    run does not emit identical duplicates, while the intended pattern's meaning stays constant."""
    stub = pool.papers[seq % len(pool.papers)]
    rationales = [stub.rationale, *pool.rationale_templates]
    judge, user = _scores(pool.direction, rng.uniform(*pool.delta_band), rng)
    return {"intended": pool.label, "title": stub.title, "abstract": stub.abstract,
            "journal": stub.journal, "judge_score": judge, "user_score": user,
            "rationale": rationales[seq % len(rationales)], "note": stub.note}


def build_rounds(spec, rng):
    """Compile a ScenarioSpec into the list of {name, papers, then, expect} rounds the harness
    driver consumes. pmids come from one monotonic counter across the whole scenario, so they
    are unique regardless of per-session volume (no manual offsets, no INSERT-OR-IGNORE
    collisions). Deterministic given `rng`."""
    ids = count(1)
    seq = {}      # per-pattern running index, for cycling the papers across the whole history
    rounds = []
    for s in range(spec.n_sessions):
        papers = []
        for p in spec.explicit.get(s, []):
            _validate_paper(p)
            papers.append({**p, "pmid": f"SYN{next(ids):07d}"})
        for stream in spec.streams:
            if s in stream.sessions:
                pool = spec.pools_by_intended_pattern[stream.label]
                for _ in range(stream.count_per_session):
                    i = seq.get(stream.label, 0)
                    seq[stream.label] = i + 1
                    papers.append({**_emit(pool, rng, i), "pmid": f"SYN{next(ids):07d}"})
        rounds.append({
            "name": spec.session_names.get(s, f"session {s + 1}"),
            "papers": papers,
            "then": spec.then_rules.get(s, []),
            "expect": spec.per_round_expect.get(s, {}),
        })
    return rounds


def _validate_paper(p):
    missing = [k for k in PAPER_KEYS if k not in p]
    if missing:
        raise ValueError(f"explicit paper missing fields {missing}: {p.get('title', p)!r}")
