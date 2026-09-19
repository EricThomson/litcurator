"""Gates that drive the real cluster and consolidate steps, and therefore cost money.

Each one builds a throwaway database from nothing and deletes it at the end; the live database
is never opened, and no judge runs anywhere.

None of these has its own command line. Every knob a gate needs -- how many reps, what vote a
check must win, how many sessions -- lives here as a constant rather than a flag, on the
principle that a gate whose pass threshold you can turn down until it goes green is not a gate.
The practical effect is that money can only be spent through `litcurator analysis_harness`.
"""

from . import pool_calibration, long_horizon

# scenario name -> (reps, the share of reps a check must win).
#
# ONE rep. The rep vote exists for genuinely marginal signals, but leaning on it is a way of
# not fixing the scenario: if a check only holds two times in three, the scenario is weak, not
# under-sampled, and the fix is a clearer signal rather than more samples. Reps also multiply
# straight into wall time, which is what stops a gate from being run at all. run_reps still
# takes the count, so a specific investigation can raise it without changing anything here.
_SCENARIOS = {
    "pattern-lifecycle": ("pattern_lifecycle", 1, 1.0),
    "dual-nature": ("dual_nature", 1, 1.0),
    "named-disinterest": ("named_disinterest", 1, 1.0),
    "accumulation": ("accumulation", 1, 1.0),
    "note-carry": ("note_carry", 1, 1.0),
    "blame-routing": ("blame_routing", 1, 1.0),
}


def _scenario_runner(scenario, reps, pass_frac):
    def run(ctx):
        return long_horizon.run(ctx, scenario, reps, pass_frac)
    return run


# Every gate runs by default, including the long-horizon ones. They looked slow at first and
# are not: a cold cluster cache makes accumulation take three minutes, but a warm one takes 47
# seconds, and warm is the normal case since most iteration is on the consolidate prompt. The
# cost only returns when the CLUSTER prompt changes, which invalidates every key -- so that is
# a warning in the dry run, not a reason to demote the scenario that answers the central
# question of whether a weak trickle accumulates at all.
COLD_CACHE_GATES = {"accumulation"}


# name -> (layer, run(ctx) -> (checks, cost, transcript), when_red, model calls)
# `when_red` is printed ONLY when the gate fails. An alarm with no interpretation is noise,
# and these strings cost nothing until something is actually broken.
PAID_GATES = {
    "pool-calibration": (
        2, pool_calibration.run,
        "the synthetic PAPERS are miscalibrated, not the machinery. Two fixtures are graded, "
        "the generated pools and the hand-written lifecycle sessions -- the second was added "
        "2026-08-26 after a colliding unicorn cost a full paid sweep to find. "
        "A pool that shattered has "
        "papers too varied; two pools that fused have papers too alike. Fix the offending "
        "fixture -- paper_pools.py for [pools], scenarios.py for [lifecycle] -- and do not "
        "read the layer 3 and 4 results until this is green.",
        # 4, not 2: two fat sessions now, one per fixture, each a cluster + a consolidate call.
        4),
    "pattern-lifecycle": (
        3, _scenario_runner(*_SCENARIOS["pattern-lifecycle"]),
        "the memory behaviors: a repeat merged into a twin instead of into the pattern it already "
        "belongs to, a closed pattern was reopened or its return went unlogged, or the open pile "
        "is climbing. Read the saved transcript for the final provenance table and check what the "
        "pattern actually holds.",
        8),
    "dual-nature": (
        4, _scenario_runner(*_SCENARIOS["dual-nature"]),
        "a paper instantiating two tastes was fused into one blended pattern instead of attaching "
        "to a pattern for each. That yields a profile edit the human cannot write, since it asks "
        "for two unrelated things in one line.",
        1),
    "named-disinterest": (
        4, _scenario_runner(*_SCENARIOS["named-disinterest"]),
        "a gap the profile ALREADY states was dropped as 'already covered'. It should be "
        "recorded like any other -- the profile saying something is not a reason to discard "
        "flags that contradict the score.",
        1),
    "accumulation": (
        4, _scenario_runner(*_SCENARIOS["accumulation"]),
        "the weak trickle fragmented instead of gathering into one pattern. Expected if you "
        "lowered the delta band; a real regression otherwise. run_accumulation_sweep.py in the "
        "sandbox shows the whole curve.",
        12),
    "note-carry": (
        4, _scenario_runner(*_SCENARIOS["note-carry"]),
        "the user's own words did not survive into the pattern. Read WHICH of the three checks "
        "went red, they localise the loss: 'term' green + 'directive' red means the note wall in "
        "the consolidate prompt is eating the actionable half ('put this on my disinterest "
        "list'); both red means the wording never made it out of the cluster step, which is a "
        "cluster-prompt problem instead. This is the ONLY gate that reads produced TEXT rather "
        "than the provenance graph, so nothing else here can see this failure.",
        6),
    "blame-routing": (
        4, _scenario_runner(*_SCENARIOS["blame-routing"]),
        "a fix was routed to the artifact that cannot absorb it. Read WHICH label went red. "
        "E red (commentary pieces called a PROFILE job) means the judge prompt is not reaching "
        "consolidate, or the CONSOLIDATE section still does not tell the model what to do with "
        "it -- check that section mentions blame at all before suspecting the model, because "
        "until 2026-09-19 it did not. F red (motor-circuit work called a PROMPT job) is the opposite "
        "failure, the model answering 'prompt' by default, and it is what the negative control "
        "in sandbox/blame_fixture/ is written to catch. If both are red, look at the two "
        "companion checks first: a pattern that never surfaced or that fused with its partner "
        "makes this read as a routing failure when it is a recall failure.",
        2),
}

# ---------------------------------------------------------------------------
# DELETED SCENARIO: `robustness` (removed 2026-08-02; last present at commit 00affcd)
# ---------------------------------------------------------------------------
# Kept as a note because the QUESTION is still real and nothing else here tests it. The code is
# gone; `git show 00affcd:src/litcurator/analysis_harness/fixtures/behaviors.py` has it, along
# with the ONE_OFF clutter pool it needed, which was deleted from paper_pools.py at the same time.
#
# WHAT IT ASKED. Eight sessions, each emitting two strong signals, one weak trickle, and unrelated
# one-off flags that never get attached -- so the pile of unfiled flags grows every session. The
# question underneath: as unfiled flags accumulate over months, does the machinery degrade? Does it
# start sweeping unrelated flags into patterns, stop finding real ones, or mint ever more patterns?
# That last is the v1 bloat failure, so this is worth reviving IF a failure of that shape turns up
# in the 2025 rollout.
#
# WHY IT WAS DELETED, so the same thing is not rebuilt. A sabotage prompt written to wreck
# consolidation scored 4/4 on it while a good prompt scored 3/4 -- anti-correlated with quality,
# which is worse than having no gate. Its one substantive check rewarded HOLDING a weak signal
# back before gathering it, and the sabotage's headline instruction was to hold everything, so
# being bad at the job scored well. Its other three checks only asked whether a label produced a
# pattern anywhere across eight sessions, which is nearly free when a stream feeds that label every
# session. And the growing clutter pile it was named for was never graded at all.
#
# HOW TO REBUILD IT PROPERLY.
#   1. As a MEASUREMENT, not a gate. "Does quality decay as the pile grows" is a curve, and forcing
#      a curve into pass/fail produced both the flicker and the backwards incentive. Report
#      per-session purity, pattern count and coverage, and read the trend -- the shape of
#      sandbox/consolidate_prompt_ab/run_ab.py, not of a gate.
#   2. Grade the STRESSOR, which the original never did: do the unattached one-off flags start
#      contaminating the real patterns as the pool grows? That is the actual robustness question.
#   3. Require the strong signals to surface EARLY rather than eventually, which is what makes
#      over-holding fail instead of pass.
#   4. If you grade "two tastes stay separate", use pools with OPPOSITE directions. Two under-
#      scored pools can always be joined by something true ("the profile is too narrow"), so such
#      a check asks the model not to notice a real pattern.
#   5. Do not disturb accumulation's stream mix to get there. Two attempts to reshape the pool
#      composition knocked its weak-signal check from reliable to 1-in-2 and 2-in-3.
