"""Gates that drive the real cluster and consolidate steps, and therefore cost money.

Each one builds a throwaway database from nothing and deletes it at the end; the live database
is never opened, and no judge runs anywhere.

None of these has its own command line. Every knob a gate needs -- how many reps, what vote a
check must win, how many sessions -- lives here as a constant rather than a flag, on the
principle that a gate whose pass threshold you can turn down until it goes green is not a gate.
The practical effect is that money can only be spent through `litcurator analysis_harness`.
"""

from . import bank_calibration, long_horizon

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
    "bank-calibration": (
        2, bank_calibration.run,
        "the synthetic PAPERS are miscalibrated, not the machinery. A paper set that shattered has "
        "papers too varied; two paper sets that fused have papers too alike. Fix fixtures/banks.py, "
        "and do not read the layer 3 and 4 results until this is green.",
        2),
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
        "a gap the profile ALREADY states was dropped as 'already covered'. It should be recorded, "
        "ideally as judge-not-applying, because it points at the PROMPT rather than at a hole in "
        "the profile.",
        1),
    "accumulation": (
        4, _scenario_runner(*_SCENARIOS["accumulation"]),
        "the weak trickle fragmented instead of gathering into one pattern. Expected if you "
        "lowered the delta band; a real regression otherwise. run_accumulation_sweep.py in the "
        "sandbox shows the whole curve.",
        12),
}

# `robustness` was cut from the gate run on 2026-08-01. The scenario itself is still built and
# still runnable (fixtures/behaviors.py, and sandbox/consolidate_prompt_ab/run_ab.py drives it);
# only its GATE registration is gone.
#
# Why it was cut. A sabotage prompt -- deliberately written to wreck consolidation -- scored 4/4
# on it, while a good prompt scored 3/4. Anti-correlated with prompt quality, which makes it worse
# than no gate. The cause: its one substantive check rewards HOLDING a weak signal back before
# gathering it, and the sabotage's headline instruction was to hold everything, so being bad at the
# job scored well. Its other three checks only ask whether a label produced a pattern anywhere
# across eight sessions, which is nearly free when a stream feeds that label every session.
# Meanwhile the thing it is named for -- a growing pile of unfiled clutter -- was never graded at
# all. Full reasoning in sandbox/docs/analysis_harness.md.
#
# When it would be worth reviving. There IS a real question underneath, and nothing else here
# tests it: as unfiled flags pile up over months, does the machinery degrade -- sweeping unrelated
# flags into patterns, missing signal, or minting ever more patterns? That last one is the v1 bloat
# failure. Bring this back if a real failure of that shape shows up in the 2025 rollout.
#
# Bring it back as a MEASUREMENT, not a gate. The answer to "does quality decay as the pile grows"
# is a curve, and forcing a curve into pass/fail is what produced both the flicker and the backwards
# incentive. It should report per-session purity, pattern count and coverage, and you read the trend.
