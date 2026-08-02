"""
pool_calibration.py -- the sanity gate on the synthetic papers themselves, not the machinery.
Calibration in the dial-it-in sense: are the paper pools separable enough to draw a
conclusion from, and not so alike that everything passes for the wrong reason.

Before trusting any result about the machinery you have to know the fake papers are sane, and
the risk runs both ways. If the papers carrying one intended pattern are near-copies of each
other the test is too easy and everything passes for the wrong reason; if they are too varied
the intended pattern will not hold together, for reasons that have nothing to do with the code
under test.

So this throws every intended pattern into ONE fat session against an empty memory --
deliberately harder than any real round -- and asks whether each was recovered as a single
produced pattern, holding most of its papers and made mostly of them. One shattering across
several produced patterns means its papers are too varied; two of them fusing into one impure
pattern means they are too alike. Either way the fix is in the papers, not the machinery.

A fat single session also splits genuinely dual-nature papers across more than one pattern (the
cuttlefish study is both invertebrate work and computational-behavior work). Those extras are
reported separately and do not gate: every paper is still in its main pattern, so no signal is
lost, and they do not recur at the densities a real session runs at.
"""

import random
from collections import defaultdict

from ..fixtures import scenarios as SC
from ..fixtures import scenario_gen as GEN
from ..fixtures import paper_pools
from .. import machinery as H
from ..grading import coverage_by_intended_pattern

CALIB_DB = H.scratch_db_path("pool_calibration")

# What "recovered" means. These are constants rather than flags on purpose: a gate whose
# pass threshold you can turn down until it goes green is not a gate.
MIN_COVERAGE = 0.7      # of an intended pattern's papers, how many its produced pattern must hold
MIN_PURITY = 0.6        # of that produced pattern's flags, how many must be from the intended one


def run(ctx):
    """Run the one fat session and grade each intended pattern. Returns (checks, cost, transcript)."""
    pools = paper_pools.POOLS_BY_INTENDED_PATTERN
    streams = [GEN.Stream(label, [0], len(pool.papers)) for label, pool in pools.items()]
    spec = GEN.ScenarioSpec("calibration", n_sessions=1, profile=SC.PROFILE,
                            pools_by_intended_pattern=pools, streams=streams)
    papers = GEN.build_rounds(spec, random.Random(0))[0]["papers"]

    papers_emitted = defaultdict(int)
    for p in papers:
        papers_emitted[p["intended"]] += 1

    lines = [f"calibration: {len(papers)} flags across {len(pools)} intended patterns  ("
             + ", ".join(f"{label}x{papers_emitted[label]}" for label in pools) + ")"]

    conn, run_id, _ = H.build_db(CALIB_DB, spec.profile)
    flag_intended = {}
    try:
        H.add_round_flags(conn, run_id, papers, flag_intended)
        n_flags, _candidates, summary, cost, _ = H.run_round(
            conn, ctx.client, spec.profile, ctx.cluster_model, ctx.consolidate_model,
            use_cache=ctx.use_cache)
        pp = H.pattern_intended(conn, flag_intended)
    finally:
        conn.close()
        CALIB_DB.unlink(missing_ok=True)

    lines.append(f"\n{n_flags} flags -> {len(summary['new'])} new patterns   (${cost:.4f})")
    lines.append("\nwhat consolidate produced (dominant intended pattern, from provenance):")
    for pid, counter in pp.items():
        lines.append(f"  {pid[:10]} dominant={H.dominant_intended(counter)} "
                     f"purity={H.purity_of(counter):.2f} flags={dict(counter)}")

    stats = coverage_by_intended_pattern(pp, papers_emitted)
    checks, recovered = [], set()
    for label in pools:
        recovered_as, coverage, purity = stats.get(label, (None, 0.0, 0.0))
        if recovered_as is None:
            checks.append((False, f"calibration: intended pattern {label} recovered as one pattern",
                           "MISSING -- no pattern holds it"))
            continue
        recovered.add(recovered_as)
        checks.append((coverage >= MIN_COVERAGE and purity >= MIN_PURITY,
                       f"calibration: intended pattern {label} recovered as one pattern",
                       f"coverage {coverage:.2f} (>= {MIN_COVERAGE}), "
                       f"purity {purity:.2f} (>= {MIN_PURITY})"))

    extra = [pid for pid in pp if pid not in recovered]
    lines.append(f"\nintended patterns recovered: {len(recovered)}   "
                 f"extra patterns (one intended pattern split across more than one): {len(extra)}")
    return checks, cost, "\n".join(lines)
