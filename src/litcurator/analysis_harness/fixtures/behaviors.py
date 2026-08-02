"""
behaviors.py -- the named scenarios the long-horizon gate can run.

Each is a function returning a ScenarioSpec built from the banks. SCENARIOS at the bottom
maps a name to its CONSTRUCTOR, so every run gets a fresh spec whose expectations match its
own session count.

  pattern_lifecycle -- a pattern's life with the human deciding in between (carry, incorporate,
                       reject) and gaps coming back afterwards
  accumulation      -- a weak same-direction trickle must coalesce into ONE pattern
  robustness        -- that trickle survives a growing pool of unrelated one-offs
  dual_nature       -- one paper instantiating TWO tastes lands in a pattern for each
  named_disinterest -- a gap the profile already states is recorded, not dropped as covered

Every case here should be an EASY call. These are unit tests: a borderline fixture produces a
coin flip rather than a verdict, and a gate that flickers gets ignored. When one goes red, the
first question is whether the papers are genuinely unambiguous -- see the note in robustness on
why its two strong signals now point in opposite directions.
"""

from . import scenarios as SC
from . import scenario_gen as GEN
from . import banks


def accumulation(delta_band=(0.16, 0.22), n_sessions=12):
    """A weak same-direction TRICKLE: one connectomics (intended pattern CONNECTOME) flag per session,
    nothing else. Each flag stays HELD (unattached) until enough accumulate for the cluster step to see
    the regularity, at which point they should coalesce into exactly ONE pattern.

    coalesces_to_one grades three things at once: it fires (a lone flag or two must not mint a
    pattern -> min_session), it fires by a deadline (must not be lost forever -> max_session),
    and the pattern is pure. delta_band is the sweep knob: above 0.15 the flags render in the
    'judge scored too low' bucket; below, in the 'roughly agreed / context' bucket the cluster
    prompt is told not to pattern -- so the sweep finds where stateless accumulation breaks.

    TWELVE SESSIONS IS THE LONG-HORIZON TEST BED, and it is meant to grow. With `robustness` cut
    this is the only scenario emulating a long history at all -- everything else runs one to four
    sessions -- and emulating many months is the thing litcurator is actually for. The intended
    direction is MORE checks riding on these sessions, and more sessions if a question needs them,
    not fewer.

    So do not shrink it to save model calls. That has been proposed twice, both times on the
    arithmetic that this is 12 of the suite's 24 calls for one check. The arithmetic is right and
    the inference is backwards twice over. It reads a test bed as a line item; and the later
    sessions are not idle repeats even though their summary lines are identical (`0 new, 1 merged`,
    over and over), because the INPUT grows every session -- by session 10 the model is shown a
    pattern carrying nine flags and a long paper list, not the two-flag pattern it saw at session 2.
    A failure like "once a pattern is big enough the model stops recognising it and mints a
    duplicate" can only appear late, and that is drift and bloat setting in over time: the exact v1
    failure this project exists to prevent.

    The obvious next thing to add here costs no model calls. This currently grades only the END
    STATE -- one pattern, right window, pure -- and asserts nothing about the ramp, such as the
    pattern having kept absorbing flags rather than spawning a sibling at session 9. The per-session
    history is already collected and handed to the terminal grader, so that is a new check over data
    already in hand."""
    return GEN.ScenarioSpec(
        name=f"accumulation(delta={delta_band[0]:.2f}-{delta_band[1]:.2f})",
        n_sessions=n_sessions,
        profile=SC.PROFILE,
        papers_by_intended_pattern={"CONNECTOME": banks.connectome_paper_set(delta_band)},
        streams=[GEN.Stream("CONNECTOME", range(0, n_sessions), count_per_session=1)],
        terminal_expect={
            "coalesces_to_one": [
                {"label": "CONNECTOME", "min_session": 1, "max_session": max(2, n_sessions - 4),
                 "min_purity": 0.8}],
            # The ramp, not just the end state. coalesces_to_one is satisfied by a run that mints
            # the pattern early and then ignores every later flag -- one pure pattern, right
            # window, and ten flags left on the floor. Measured over the last third of the run,
            # since the early sessions legitimately hold while the signal is still accumulating.
            "pool_drains": [{"over_last_sessions": max(2, n_sessions // 3), "max_growth": 0}],
        },
    )


def robustness(n_sessions=8, weak_band=(0.08, 0.12), one_offs_per_session=1):
    """Strong signal + a weak trickle must survive a growing pool of diffuse ONE-OFFS.

    Each session emits: 1 B (computational theory, strong under), 1 C (invertebrate, strong under),
    1 CONNECTOME (WEAK under -- the competition test the pure trickle couldn't do), and
    `one_offs_per_session` ONE_OFF flags (diverse single corrections). Only the one-offs never get
    attached, so the unattached pool grows monotonically -- the stressor.

    Grades: B, C and CONNECTOME all surface, and CONNECTOME still coalesces into one pure pattern
    despite competing signal AND a sub-threshold delta (the real DELTA_THRESHOLD test -- if a cliff
    exists it shows here). The one-offs are pure stressor: whether unrelated one-offs spuriously
    group together is a separate question no scenario currently tests.

    THIS SCENARIO NO LONGER GRADES "B AND C STAY SEPARATE", and the reason is worth keeping.
    It did, on the grounds that they are adjacent and same-sign. It flickered about one run in
    three, and the cause was structural rather than sloppy wording in the papers: any two
    UNDER-scored groups are joined by a statement that is actually TRUE ("the profile is too
    narrow"), so an honest pattern spanning them always exists and the check was asking the model
    not to notice something correct. The observed failure was a pattern named "Organism and Journal
    Agnosticism" holding two B flags and two C flags -- a real insight, graded as a chimera only
    because its direction came out `under` rather than `sharpen`.

    Separation is tested properly by dual_nature, which pairs B against C inside ONE session with a
    paper that is honestly both, and grades the split directly. One scenario carries that case
    deliberately; a stress test should not carry it by accident.

    Two rejected fixes, recorded because both looked obviously right and both made things worse.
    Grading separation on an opposite-direction pair instead (B under vs D over) does remove the
    ambiguity -- a pattern carries one direction, so no single taste pattern can hold both
    complaints -- but changing the pool to get that pair broke the weak-signal test: with the mix
    altered, CONNECTOME stopped having to compete, minted a pattern in session 0, and failed its
    min_session floor. Swapping C for D did it (coalesce 1/2), and so did adding D alongside C
    (coalesce 2/3), where the original mix had been 12/12. The pool composition is load-bearing,
    so the right move was to delete the bad check rather than reshape the scenario around keeping
    it.

    A gate is meant to be an easy call. When one is borderline, ask whether the check is asking the
    model to be wrong before reaching for the fixture."""
    papers_by_intended_pattern = {
        "B": banks.PAPERS_BY_INTENDED_PATTERN["B"],
        "C": banks.PAPERS_BY_INTENDED_PATTERN["C"],
        "CONNECTOME": banks.connectome_paper_set(weak_band),
        "ONE_OFF": banks.one_off_paper_set(),
    }
    streams = [
        GEN.Stream("B", range(0, n_sessions), 1),
        GEN.Stream("C", range(0, n_sessions), 1),
        GEN.Stream("CONNECTOME", range(0, n_sessions), 1),
        GEN.Stream("ONE_OFF", range(0, n_sessions), one_offs_per_session),
    ]
    return GEN.ScenarioSpec(
        name=f"robustness(weak={weak_band[0]:.2f}-{weak_band[1]:.2f}, "
             f"oneoffs={one_offs_per_session}/s)",
        n_sessions=n_sessions,
        profile=SC.PROFILE,
        papers_by_intended_pattern=papers_by_intended_pattern,
        streams=streams,
        terminal_expect={
            "intended_patterns_surface": ["B", "C", "CONNECTOME"],
            "coalesces_to_one": [{"label": "CONNECTOME", "min_session": 1,
                                  "max_session": n_sessions - 2, "min_purity": 0.7}],
        },
    )


# One paper that genuinely instantiates TWO tastes at once: invertebrate neuroethology (intended
# pattern C)
# AND computational/normative theory (intended pattern B). The judge rationale names both reasons
# it scored
# low, so the evidence for splitting is right there in the text the cluster step reads.
_DUAL_PAPER = {
    "intended": ["B", "C"],
    "title": "A normative model of evidence accumulation in cuttlefish hunting decisions",
    "abstract": ("Cuttlefish adjust the duration of prey stalking with the reliability of visual "
                 "cues. We recorded strike latencies across controlled cue-reliability conditions "
                 "and show the distribution is captured by a drift-diffusion model with a "
                 "collapsing bound. The fitted bound predicts individual strike timing on held-out "
                 "trials, indicating that an invertebrate predator implements bounded evidence "
                 "accumulation of the kind described in primate decision tasks."),
    "journal": "eLife",
    "judge_score": 0.35,
    "user_score": 0.80,
    # The wrong reason, naming BOTH tastes -- an LLM that reads this and still emits one blended
    # pattern is fusing two independent gaps, which is exactly what the check catches.
    "rationale": ("This is invertebrate work in a non-standard model organism, and its primary "
                  "contribution is a fitted theoretical model rather than new circuit-level data. "
                  "Neither invertebrate neuroethology nor normative modeling is emphasized in the "
                  "profile, so this sits below the surfacing threshold."),
    "note": "invertebrates AND decision theory -- I want both of these, separately",
}


def dual_nature():
    """One paper instantiating TWO tastes must land in a pattern for EACH, not fuse them into one.

    A single session with three computational-theory flags (B), three invertebrate flags (C), and
    ONE paper that is honestly both. The independent B and C flags are what make the two tastes
    demonstrably SEPARABLE -- the separability test the prompts now state -- so the correct output
    is two patterns with the dual paper attached to both, never a single blended
    "invertebrate decision-modeling" pattern.

    Why this matters beyond tidiness: a chimera yields a profile edit the human cannot write, since
    it asks for two unrelated things in one line. And it is a WITHIN-session question -- nothing to
    do with accumulation across sessions -- so one session is the whole test."""
    return GEN.ScenarioSpec(
        name="dual_nature(one paper, two tastes)",
        n_sessions=1,
        profile=SC.PROFILE,
        papers_by_intended_pattern={"B": banks.PAPERS_BY_INTENDED_PATTERN["B"], "C": banks.PAPERS_BY_INTENDED_PATTERN["C"]},
        streams=[GEN.Stream("B", [0], count_per_session=3),
                 GEN.Stream("C", [0], count_per_session=3)],
        explicit={0: [_DUAL_PAPER]},
        terminal_expect={
            "intended_patterns_surface": ["B", "C"],
            "stay_separate": [["B", "C"]],
            "shared_flag_in_both": [{"labels": ["B", "C"]}],
        },
    )


def pattern_lifecycle():
    """A pattern's whole life, with the human deciding in between: created, then carried or
    incorporated or rejected, then coming back.

    This is the only scenario that scripts human decisions, so it is the only cover for the
    memory behaviors -- a repeat merging into a still-open pattern instead of spawning a twin, a
    repeat of a CLOSED pattern being logged as a return without reopening it, a closed pattern
    staying closed, and the open pile staying bounded. Everything else here tests how flags turn
    into patterns; this tests what happens to a pattern afterwards.

    It runs the hand-authored fixture in scenarios.py, which is worth reading top to bottom: it
    states in plain language what correct behavior means, and its expectations are executable, so
    the spec cannot quietly drift out of date the way prose would.

    The session count comes from the fixture, so lengthening the arc is a fixture edit and
    nothing here changes. The last session deliberately contains NO new gaps: with everything
    already tracked, what is left to grade is purely memory."""
    return GEN.ScenarioSpec(
        name="pattern_lifecycle(human decides between sessions)",
        n_sessions=len(SC.SESSIONS),
        profile=SC.PROFILE,
        explicit={i: [{k: p[k] for k in GEN.PAPER_KEYS} for p in session["papers"]]
                  for i, session in enumerate(SC.SESSIONS)},
        then_rules={i: session["then"] for i, session in enumerate(SC.SESSIONS)},
        per_round_expect={i: session["expect"] for i, session in enumerate(SC.SESSIONS)},
        session_names={i: session["name"] for i, session in enumerate(SC.SESSIONS)},
        terminal_expect=SC.TERMINAL_EXPECT,
    )


# A disinterest stated outright, appended to the fixture profile. The plain fixture profile is
# deliberately silent on every planted gap, so each one reads as a genuine hole; this scenario
# needs the opposite, a profile that clearly SAYS the thing the judge keeps ignoring.
_DISINTEREST_LINE = (
    "\n\nI have no interest in non-invasive human studies. Neuroimaging, scalp recordings, "
    "psychophysics and eye tracking in human volunteers are off-topic for me no matter which "
    "cognitive function is being measured, because none of them reach the circuit."
)


def named_disinterest():
    """A gap the profile ALREADY states must still be RECORDED, not silently dropped.

    One session of clinical-EEG flags (intended pattern A, over-scored) against a profile that names that
    disinterest outright. The judge scoring them high anyway is a PROMPT problem, not a hole in
    the profile -- so the right outcome is a recorded pattern, ideally with direction
    judge-not-applying, and the wrong outcome is consolidate reasoning "the profile already
    covers this" and letting the flags fall on the floor.

    This is the one property the old test_suggest_patterns.py checked that nothing else did. It
    cannot run on the plain fixture profile, which is why it carries its own."""
    return GEN.ScenarioSpec(
        name="named_disinterest(profile says it, judge ignores it)",
        n_sessions=1,
        profile=SC.PROFILE + _DISINTEREST_LINE,
        papers_by_intended_pattern={"A": banks.PAPERS_BY_INTENDED_PATTERN["A"]},
        streams=[GEN.Stream("A", [0], count_per_session=5)],
        terminal_expect={"named_disinterest_not_dropped": [
            {"label": "A", "directions": ["judge-not-applying", "over"]}]},
    )


# The CONSTRUCTORS, not built specs. A spec carries expectations computed from its own
# session count (accumulation's deadline is n_sessions - 4), so a built spec cannot be
# resized after the fact -- the old registry stored instances and callers mutated
# spec.n_sessions, which moved the horizon but left the deadline behind. Building per run
# keeps a spec and its expectations consistent, and stops one gate's mutation leaking into
# the next gate in the same process.
SCENARIOS = {
    "pattern_lifecycle": pattern_lifecycle,
    "accumulation": accumulation,
    "robustness": robustness,
    "dual_nature": dual_nature,
    "named_disinterest": named_disinterest,
}
