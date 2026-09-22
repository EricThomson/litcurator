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


def _generated_papers():
    """Every paper from the generated POOLS, one of each, in a single session."""
    pools = paper_pools.POOLS_BY_INTENDED_PATTERN
    streams = [GEN.Stream(label, [0], len(pool.papers)) for label, pool in pools.items()]
    spec = GEN.ScenarioSpec("calibration", n_sessions=1, profile=SC.PROFILE,
                            pools_by_intended_pattern=pools, streams=streams)
    return GEN.build_rounds(spec, random.Random(0))[0]["papers"], list(pools)


def _lifecycle_papers():
    """Every paper from the HAND-WRITTEN lifecycle fixture, all four sessions flattened.

    Added 2026-08-26, and it should have existed from the start. This gate read only
    POOLS_BY_INTENDED_PATTERN, which feeds the generated scenarios -- so the fixture behind
    pattern-lifecycle, written by hand in scenarios.py, was never calibrated at all. That is
    how U2 shipped as an electric fish sharing four properties with intended pattern C: the
    one gate whose entire job is catching confusable fixtures had never looked at it, and the
    collision cost a full paid sweep to discover.

    Flattening four sessions into one is deliberately harder than any real round, which is the
    whole method here. It is especially sharp on the UNICORNS: a single-paper intended pattern
    absorbed into an eight-flag neighbour scores purity 0.11 and fails loudly, which is exactly
    the shape that got through."""
    papers, labels = [], []
    for session in SC.SESSIONS:
        for paper in session["papers"]:
            papers.append(paper)
            if paper["intended"] not in labels:
                labels.append(paper["intended"])
    return papers, labels


def _calibrate(ctx, name, papers, labels):
    """Run one fat session over `papers` and grade each intended pattern."""

    papers_emitted = defaultdict(int)
    for p in papers:
        papers_emitted[p["intended"]] += 1

    lines = [f"calibration [{name}]: {len(papers)} flags across {len(labels)} intended "
             f"patterns  (" + ", ".join(f"{l}x{papers_emitted[l]}" for l in labels) + ")"]

    conn, run_id, _ = H.build_db(CALIB_DB, SC.PROFILE)
    flag_intended = {}
    try:
        H.add_round_flags(conn, run_id, papers, flag_intended)
        n_flags, _candidates, summary, cost, _ = H.run_round(
            conn, ctx.client, SC.PROFILE, ctx.cluster_model, ctx.consolidate_model,
            ctx.cluster_prompt, ctx.consolidate_prompt, use_cache=ctx.use_cache,
            judge_prompt=SC.JUDGE_PROMPT)
        pp = H.pattern_intended(conn, flag_intended)
        # Pulled BEFORE the finally drops the database. A contaminated unicorn is reported as a
        # bare ratio otherwise, and the scratch DB holding the answer is deleted milliseconds
        # later -- see H.pattern_flag_details.
        details = H.pattern_flag_details(conn, flag_intended)
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
    for label in labels:
        recovered_as, coverage, purity = stats.get(label, (None, 0.0, 0.0))
        if recovered_as is None:
            checks.append((False, f"calibration [{name}]: {label} recovered as one pattern",
                           "MISSING -- no pattern holds it"))
            continue
        recovered.add(recovered_as)
        ok = coverage >= MIN_COVERAGE and purity >= MIN_PURITY
        detail = (f"coverage {coverage:.2f} (>= {MIN_COVERAGE}), "
                  f"purity {purity:.2f} (>= {MIN_PURITY})")
        if not ok:
            # The texture, not just the verdict: which papers actually landed together. A
            # purity number cannot distinguish a defensibly adjacent neighbour from a flag
            # pointing the opposite way, and those want different fixes.
            lines.append(f"\ncontamination detail for {label} "
                         f"(pattern {recovered_as[:10]}, purity {purity:.2f}):")
            lines += H.describe_contamination(details.get(recovered_as, []), label)
        checks.append((ok, f"calibration [{name}]: {label} recovered as one pattern", detail))

    extra = [pid for pid in pp if pid not in recovered]
    lines.append(f"\nintended patterns recovered: {len(recovered)}   "
                 f"extra patterns (one intended pattern split across more than one): {len(extra)}")
    return checks, cost, "\n".join(lines)


def run(ctx):
    """Calibrate BOTH bodies of synthetic papers: the generated pools and the hand-written
    lifecycle sessions. Two fat sessions rather than one, because the suite has two independent
    fixtures and only the first was ever checked. Returns (checks, cost, transcript)."""
    checks, cost, parts = [], 0.0, []
    for name, source in (("pools", _generated_papers), ("lifecycle", _lifecycle_papers)):
        papers, labels = source()
        c, k, text = _calibrate(ctx, name, papers, labels)
        checks += c
        cost += k
        parts.append(text)
    return checks, cost, "\n\n".join(parts)
