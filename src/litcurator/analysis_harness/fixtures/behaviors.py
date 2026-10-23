"""
behaviors.py -- the named scenarios the long-horizon gate can run.

Each is a function returning a ScenarioSpec, drawing its papers from paper_pools. SCENARIOS at
the bottom maps a name to its CONSTRUCTOR, so every run gets a fresh spec whose expectations
match its own session count.

  pattern_lifecycle -- a pattern's life with the human deciding in between (carry, incorporate,
                       reject) and gaps coming back afterwards
  accumulation      -- a weak same-direction trickle must coalesce into ONE pattern, and keep
                       absorbing flags rather than being recognised once and ignored
  dual_nature       -- one paper instantiating TWO tastes lands in a pattern for each
  named_disinterest -- a gap the profile already states is recorded, not dropped as covered
  note_carry        -- the user's own words survive from the note into the pattern
  blame_routing     -- a fix is routed to the document that can absorb it: a wrong rule in the
                       judge prompt goes to the prompt, a topic neither document mentions goes
                       to the profile
  note_decides_blame -- when the note names the document, that wins, even against a scoring
                       rule found in the other one

Every case here should be an EASY call. These are unit tests: a borderline fixture produces a
coin flip rather than a verdict, and a gate that flickers gets ignored. When one goes red, the
first question is whether the papers are genuinely unambiguous.

Two traps when writing a new one, both learned by falling into them. An intended pattern's papers
must share exactly ONE salient property and vary on everything else -- sharing more makes it MORE
ambiguous, because every shared property is another defensible grouping. And if you want to check
that two tastes stay SEPARATE, give them opposite directions: two under-scored groups can always
be joined by a statement that is true ("the profile is too narrow"), so such a check asks the
model not to notice something real, and it will flicker forever. That is what removed the
`robustness` scenario -- see the note at the bottom of gates_llm/__init__.py.
"""

from . import scenarios as SC
from . import scenario_gen as GEN
from . import paper_pools


def accumulation(delta_band=(0.16, 0.22), n_sessions=12):
    """A weak same-direction TRICKLE: one connectomics (intended pattern CONNECTOME) flag per session,
    nothing else. Each flag stays HELD (unattached) until enough accumulate for the cluster step to see
    the regularity, at which point they should coalesce into exactly ONE pattern.

    coalesces_to_one grades three things at once: it fires (a lone flag or two must not be put
    in front of the user -> min_session), it fires by a deadline (must not be lost forever ->
    max_session), and the pattern is pure.

    WHAT "FIRES" MEANS CHANGED 2026-08-26, and the mechanism this scenario tests did not. A thin
    gap can now be recorded as a HELD pattern from the first flag -- a real row with real
    provenance that the user is simply not shown -- so min_session no longer grades when a ROW
    appears, it grades when the pattern becomes VISIBLE (long_horizon reads first_surfaced off
    the active list for exactly this reason). The trickle still piles up in the clustering pool
    while it is held, because a held pattern's flags deliberately stay there; that is what keeps
    the accumulation happening in CLUSTER, the only step that reads titles and abstracts. delta_band is the sweep knob -- it sets how WEAK the signal is, so
    the sweep finds the magnitude at which stateless accumulation stops working.

    This docstring used to say the band mattered because 0.15 was a rendering boundary, and
    that below it the flags landed in a bucket 'the cluster prompt is told not to pattern'.
    Both halves were wrong. No such instruction was ever in any cluster prompt -- it lived in a
    header string in _format_papers -- and the boundary is gone (2026-08-07; see the note beside
    error_analysis.MIN_FLAGS). The band is now a plain magnitude dose, which is the cleaner
    experiment: the old 0.13-0.17 band was half above the line and half below it, so it dosed
    two things at once.

    Two checks ride on it. `coalesces_to_one` grades the END STATE and `pool_drains` grades the
    RAMP, because the end state alone is satisfied by a run that mints the pattern early and then
    holds every later flag: one pure pattern in the right window, with ten flags left on the floor.
    The unattached count is the tell -- flat when flags are being absorbed, climbing when they are
    not -- and it costs no model calls, being the number already handed to each session.

    TWELVE SESSIONS IS THE LONG-HORIZON TEST BED, and it is meant to grow. This is the only
    scenario emulating a long history at all -- everything else runs one to four sessions -- and
    emulating many months is the thing litcurator is actually for. The intended direction is MORE
    checks riding on these sessions, and more sessions if a question needs them, not fewer.

    So do not shrink it to save model calls. That has been proposed twice, both times on the
    arithmetic that this is 12 of the suite's 24 calls for one check. The arithmetic is right and
    the inference is backwards twice over. It reads a test bed as a line item; and the later
    sessions are not idle repeats even though their summary lines are identical (`0 new, 1 merged`,
    over and over), because the INPUT grows every session -- by session 10 the model is shown a
    pattern carrying nine flags and a long paper list, not the two-flag pattern it saw at session 2.
    A failure like "once a pattern is big enough the model stops recognising it and mints a
    duplicate" can only appear late, and that is drift and bloat setting in over time: the exact v1
    failure this project exists to prevent.

    Anything else you want to know about a long history goes here, and the per-session record the
    terminal grader already receives (new patterns, open count, unattached count) is probably
    enough to grade it without spending anything more."""
    return GEN.ScenarioSpec(
        name=f"accumulation(delta={delta_band[0]:.2f}-{delta_band[1]:.2f})",
        n_sessions=n_sessions,
        profile=SC.PROFILE,
        judge_prompt=SC.JUDGE_PROMPT,
        pools_by_intended_pattern={"CONNECTOME": paper_pools.connectome_pool(delta_band)},
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
            # CONNECTOME flags are all positive deltas, so `over` on this pattern is never
            # defensible. Worth having here in particular: this is the pool with the smallest
            # deltas after CANCER, and the inversion that prompted this check turned up on the
            # smallest-delta pool in the suite.
            "direction_not_inverted": [{"label": "CONNECTOME", "taste": "under"}],
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
        judge_prompt=SC.JUDGE_PROMPT,
        pools_by_intended_pattern={"B": paper_pools.POOLS_BY_INTENDED_PATTERN["B"], "C": paper_pools.POOLS_BY_INTENDED_PATTERN["C"]},
        streams=[GEN.Stream("B", [0], count_per_session=3),
                 GEN.Stream("C", [0], count_per_session=3)],
        explicit={0: [_DUAL_PAPER]},
        terminal_expect={
            "intended_patterns_surface": ["B", "C"],
            "stay_separate": [["B", "C"]],
            "direction_not_inverted": [{"label": "B", "taste": "under"},
                                       {"label": "C", "taste": "under"}],
            "shared_flag_in_both": [{"labels": ["B", "C"]}],
        },
    )


def note_carry():
    """A strongly-worded user NOTE must reach the pattern intact, across several sessions.

    THE GAP THIS CLOSES. Every other check in this suite grades the provenance graph -- which
    flags landed on which pattern -- and never reads a word the model produced. That leaves a
    whole class invisible: the user writes "put this on my disinterest list", the
    cluster step paraphrases it into a pattern description, consolidate (which never sees the
    note, only cluster's prose) restates it again, and what reaches the workbench is a blurred
    version of a sentence the user had already written precisely. The graph is perfect
    throughout. Found 2026-08-07 by reading ONE real January flag, after twelve green gates.

    WHY IT RUNS THREE SESSIONS RATHER THAN ONE. The one-hop version would be a weaker test. The
    worry is DECAY: session 1 mints the pattern from the note, sessions 2 and 3 merge more flags
    into it, and each merge is another chance for the wording to be restated. So the checks run
    at the END, after the pattern has been carried twice. Three is enough to see drift; there is
    no reason to pay for twelve.

    The B stream is ballast with the OPPOSITE direction, for two reasons: the note-carrier has to
    compete for attention rather than being the only thing in the pool (pool composition is
    load-bearing -- see prompt_insights), and opposite directions are what make a separation
    check honest, since any two same-direction groups can be joined by something true.

    EXPECT THIS TO BE RED AT FIRST, and that is the point. The active consolidate prompt says
    "never transcribe a user's private note verbatim", so the directive check should FAIL until
    that paragraph is rewritten to distinguish "do not let the machine invent profile prose"
    (the v1 failure the rule exists for) from "when the user already stated the fix, carry their
    words". A test that goes green the day you write it has told you nothing."""
    return GEN.ScenarioSpec(
        name="note_carry(the user's own words reach the pattern)",
        n_sessions=3,
        profile=SC.PROFILE,
        judge_prompt=SC.JUDGE_PROMPT,
        pools_by_intended_pattern={"CANCER": paper_pools.cancer_pool(),
                                   "B": paper_pools.POOLS_BY_INTENDED_PATTERN["B"]},
        streams=[GEN.Stream("CANCER", [0, 1, 2], count_per_session=2),
                 GEN.Stream("B", [0, 1, 2], count_per_session=2)],
        terminal_expect={
            "intended_patterns_surface": ["CANCER"],
            "stay_separate": [["CANCER", "B"]],
            # B ONLY, and CANCER deliberately absent. Cancer biology sits outside the profile's
            # STATED domain, so when this was written the correct answer for it was
            # `judge-not-applying` and the check could only ever pass -- the vacuous shape this
            # suite has been bitten by twice. That value was deleted 2026-08-25, so CANCER now
            # has a real two-way answer and could be added. Left out until a run shows what the
            # model actually proposes for it; adding it on prediction is how vacuous checks get
            # written. Direction is covered where it discriminates -- A and D (over), B, C and
            # CONNECTOME (under).
            "direction_not_inverted": [{"label": "B", "taste": "under"}],
            # Three depths of the same question, kept separate so a red LOCALISES the loss.
            # term green + directive red  -> the note wall is eating the actionable half.
            # both red                    -> the wording never survived the cluster step.
            "note_wording_survives": [
                {"label": "CANCER", "kind": "directive", "markers": ["disinterest"],
                 "fields": ("suggested_edit",)},
            ],
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
        judge_prompt=SC.JUDGE_PROMPT,
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

    One session of clinical-EEG flags (intended pattern A, over-scored) against a profile that
    names that disinterest outright. The wrong outcome is consolidate reasoning "the profile
    already covers this" and letting the flags fall on the floor; the right outcome is a
    recorded pattern, whichever document it is blamed on.

    NO BLAME CHECK, since 2026-10-23. From its first version this scenario asserted the answer
    "prompt" (first as the direction `judge-not-applying`, then as `blame`), on the argument that
    a taste the profile already states cannot be fixed with more profile prose. That argument is
    where the live consolidate step got its habit of blaming the prompt for half of every round,
    and the case is genuinely arguable: "say it more strongly in the profile" is the fix that
    has actually been shown to work. A test whose right answer needs an argument is a coin flip
    wearing a verdict, so the check is gone; the unarguable blame cases live in blame_routing
    and note_decides_blame.

    It cannot run on the plain fixture profile, which is why it carries its own."""
    return GEN.ScenarioSpec(
        name="named_disinterest(profile says it, judge ignores it)",
        n_sessions=1,
        profile=SC.PROFILE + _DISINTEREST_LINE,
        judge_prompt=SC.JUDGE_PROMPT,
        pools_by_intended_pattern={"A": paper_pools.POOLS_BY_INTENDED_PATTERN["A"]},
        streams=[GEN.Stream("A", [0], count_per_session=5)],
        terminal_expect={
            "named_disinterest_not_dropped": [
                {"label": "A", "directions": ["over"]}],
            # Compatible with the check above rather than a duplicate of it: that one requires
            # a specific allowed set, this one only forbids the inversion. Both must hold.
            "direction_not_inverted": [{"label": "A", "taste": "over"}],
        },
    )


def blame_routing():
    """A fix must be routed to the artifact that can absorb it.

    ONE session, TWO pools, one of each answer, graded together so the check is a discrimination
    rather than a label a model could get right by always saying the same word:

      E  review articles, UNDER-scored. scenarios.JUDGE_PROMPT caps secondary literature below
         0.55, the judge OBEYS that cap, and the user wants these near the top. The written rule
         is WRONG, so the only place to fix it is the scoring procedure -> blame 'prompt'.
      D  translational disease models, OVER-scored. Neither document mentions them -> blame
         'profile'. Reused rather than purpose-built: already calibrated, note-complete, and
         pointing the opposite way.

    WHY THIS PROVES THE WIRING. E's notes say only that reviews are scored too low; they never
    say a cap exists or where it lives. So 'prompt' is unreachable without having read the judge
    prompt, and withholding it flips E to 'profile' while leaving D untouched. That is the
    negative control, and it is not a sabotage -- it is the exact configuration every paid gate
    ran in until 2026-09-19.

    THE FIRST VERSION OF THIS GATE WAS AMBIGUOUS AND FLICKERED. E used to be commentary pieces
    the judge scored ABOVE a stated cap, i.e. a rule being ignored -- which let consolidate
    reason "the scoring procedure already covers this" and DISCARD the candidate. It did, in 2
    of 3 runs, and took pool-calibration red with it. The model was not wrong; the fixture was,
    because "already covered" was defensible. A test may be HARD (accumulation is genuinely
    tricky) but its right answer must never be arguable.

    OPPOSITE DIRECTIONS on purpose. Two pools erring the same way can always be joined by a
    generalization that happens to be true, so a check that they stay separate would be asking
    the model not to notice something real -- the mistake that removed `robustness`."""
    return GEN.ScenarioSpec(
        name="blame_routing(prompt job vs profile job, in one session)",
        n_sessions=1,
        profile=SC.PROFILE,
        judge_prompt=SC.JUDGE_PROMPT,
        pools_by_intended_pattern={
            "E": paper_pools.POOLS_BY_INTENDED_PATTERN["E"],
            "D": paper_pools.POOLS_BY_INTENDED_PATTERN["D"],
        },
        streams=[GEN.Stream("E", [0], count_per_session=4),
                 GEN.Stream("D", [0], count_per_session=4)],
        terminal_expect={
            "blame_routing": [{"label": "E", "blame": "prompt"},
                              {"label": "D", "blame": "profile"}],
            # Both pools must surface, stay apart and keep their sign. Without these, a blame
            # check on a missing or fused pattern reads as a routing failure when it is really a
            # recall failure -- which is exactly how the first version of this gate failed.
            "intended_patterns_surface": ["E", "D"],
            "stay_separate": [("E", "D")],
            "direction_not_inverted": [{"label": "E", "taste": "under"},
                                       {"label": "D", "taste": "over"}],
        },
    )


# The one rule note_decides_blame adds to its own copy of the fixture judge prompt. The same
# shape as the review cap the shared prompt carries: a number the judge obeys, so the rule is
# unarguably what produced the scores. It lives here and not in scenarios.JUDGE_PROMPT because
# pool TOOLS is drawn by pool-calibration too, and there it must read as a plain gap.
_METHODS_CAP = """
## Methods papers

A paper whose primary contribution is a new tool, instrument or recording technique, rather than
a finding about the brain, is capped below 0.55 whatever the topic.
""".strip()


def note_decides_blame():
    """When the user's note says which document to edit, that is the answer.

    ONE session, TWO pools, one answer each, opposite directions, graded together so a constant
    answer fails:

      TOOLS   recording and imaging instruments, UNDER-scored. This scenario's judge prompt caps
              methods papers below 0.55 and the judge obeys it, so the written rule plainly
              produced the scores and the documents alone would send the fix to the PROMPT. The
              notes say PROFILE, four ways -> blame 'profile'. This is the override under test.
      PLANTS  plant physiology in neuroscience vocabulary, OVER-scored by word matching. Neither
              document mentions plants, and the notes say JUDGE PROMPT, four ways -> blame
              'prompt'. Here the note and the principle agree (the fix is topic-agnostic
              reasoning), so this pool is the discrimination partner, not a second override.

    WHY THIS IS THE RIGHT TEST. The live failure (2026-10-20) was consolidate finding a rule in
    the judge prompt and preferring it to the user's words: six runs out of six blamed the prompt
    for a pattern whose note said "we absolutely have to in profile". The user's decision is
    that an explicit note wins, even when it names the document that does not hold the rule,
    because overruling his note is the machine being clever. Both pools are easy calls under
    that rule; nothing here asks the model to weigh degrees.

    THE CONTROL (sandbox/blame_fixture/note_decides_blame_control.py) runs the same session with
    the document clause stripped from every note. TOOLS should then flip to 'prompt', because the
    cap is the only evidence left, and PLANTS should stay 'prompt'. That is what shows the TOOLS
    check is reading the note and not something else about the papers."""
    return GEN.ScenarioSpec(
        name="note_decides_blame(the note names the document, against a rule in the other one)",
        n_sessions=1,
        profile=SC.PROFILE,
        judge_prompt=SC.JUDGE_PROMPT + "\n\n" + _METHODS_CAP,
        pools_by_intended_pattern={
            "TOOLS": paper_pools.POOLS_BY_INTENDED_PATTERN["TOOLS"],
            "PLANTS": paper_pools.POOLS_BY_INTENDED_PATTERN["PLANTS"],
        },
        streams=[GEN.Stream("TOOLS", [0], count_per_session=4),
                 GEN.Stream("PLANTS", [0], count_per_session=4)],
        terminal_expect={
            "blame_routing": [{"label": "TOOLS", "blame": "profile"},
                              {"label": "PLANTS", "blame": "prompt"}],
            # As in blame_routing: a blame check on a missing or fused pattern reads as a
            # routing failure when it is really a recall failure, so recall is graded first.
            "intended_patterns_surface": ["TOOLS", "PLANTS"],
            "stay_separate": [("TOOLS", "PLANTS")],
            "direction_not_inverted": [{"label": "TOOLS", "taste": "under"},
                                       {"label": "PLANTS", "taste": "over"}],
        },
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
    "dual_nature": dual_nature,
    "named_disinterest": named_disinterest,
    "note_carry": note_carry,
    "blame_routing": blame_routing,
    "note_decides_blame": note_decides_blame,
}
