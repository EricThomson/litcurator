"""
No-LLM parity + smoke test for scenario_gen.build_rounds.

PARITY: reproduce the hand-authored lifecycle fixture (scenarios.SESSIONS) as a ScenarioSpec whose sessions
are EXPLICIT (verbatim) papers, compile it, and assert the compiled rounds match the literal on
every field the harness reads -- name, then, expect, and each paper's content -- ignoring only
the pmid string. Because _format_papers renders content (not pmid) and grading is
provenance-based, structural equality here guarantees the harness would grade the compiled
scenario identically to the literal. That is the "faithful generalization" gate, deterministic
and free.

SMOKE: a tiny stream-driven spec, to confirm the STREAM path emits the right count/intended pattern with
in-band, correctly-signed deltas and unique pmids.

    python -m litcurator.analysis_harness.gates_no_llm.scenario_compiler
"""

import random

from ..fixtures import scenarios as SC
from ..fixtures import scenario_gen as GEN
from ..fixtures import behaviors


def test_parity():
    """Compile the spec the harness ACTUALLY runs and assert it still matches the literal.

    Note it calls behaviors.pattern_lifecycle() rather than rebuilding an equivalent spec here.
    It used to construct its own copy of that conversion, which made the check hollow: change the
    conversion in behaviors.py and this still passed, because it was comparing the literal against
    a second implementation nothing runs. The point is to guard the production path."""
    spec = behaviors.pattern_lifecycle()
    compiled = GEN.build_rounds(spec, random.Random(0))

    assert len(compiled) == len(SC.SESSIONS), (len(compiled), len(SC.SESSIONS))
    all_pmids = []
    for i, (got, want) in enumerate(zip(compiled, SC.SESSIONS)):
        assert got["name"] == want["name"], (i, got["name"], want["name"])
        assert got["then"] == want["then"], (i, got["then"], want["then"])
        assert got["expect"] == want["expect"], (i, got["expect"], want["expect"])
        assert len(got["papers"]) == len(want["papers"]), (i, len(got["papers"]))
        for gp, wp in zip(got["papers"], want["papers"]):
            for k in GEN.PAPER_KEYS:
                assert gp[k] == wp[k], (i, k, gp[k], wp[k])
            assert gp["pmid"].startswith("SYN")
            all_pmids.append(gp["pmid"])

    assert len(all_pmids) == len(set(all_pmids)), "pmids must be unique across the scenario"
    print(f"parity: {len(compiled)} rounds, {len(all_pmids)} papers, "
          f"all content fields + then + expect identical to the literal (pmid aside)")


def test_stream_smoke():
    pool = GEN.IntendedPatternPool(
        label="X", direction="under", delta_band=(0.10, 0.12),
        papers=[GEN.SyntheticPaper("Weak title", "Weak abstract.", "J Neuro",
                            "the judge undersold this")],
    )
    spec = GEN.ScenarioSpec(
        name="smoke", n_sessions=3, profile="p",
        pools_by_intended_pattern={"X": pool}, streams=[GEN.Stream("X", range(0, 3), count_per_session=2)],
    )
    rounds = GEN.build_rounds(spec, random.Random(1))
    pmids = []
    for session in rounds:
        assert len(session["papers"]) == 2, session["papers"]
        for p in session["papers"]:
            assert p["intended"] == "X"
            delta = round(p["user_score"] - p["judge_score"], 3)
            assert delta > 0, ("under must be positive delta", delta)     # judge scored too low
            assert 0.10 - 0.015 <= delta <= 0.12 + 0.015, ("out of band", delta)
            assert 0.0 < p["judge_score"] < 1.0 and 0.0 < p["user_score"] < 1.0
            pmids.append(p["pmid"])
    assert len(pmids) == 6 and len(set(pmids)) == 6, pmids
    print(f"stream: 3 sessions x 2 flags of intended pattern X, deltas in-band + positive, "
          f"{len(set(pmids))} unique pmids")


# The checks this gate runs, in order. Named here once so the harness and a direct run
# cannot drift apart.
CHECKS = [test_parity, test_stream_smoke]


if __name__ == "__main__":
    for check in CHECKS:
        check()
    print("\nALL CHECKS PASSED")
