"""
No-LLM unit test for grading.check_terminal. Feeds fabricated provenance graphs +
pattern rows + first_surfaced + history straight into the pure grader and asserts every
terminal reduction fires correctly, pass AND fail. No DB, no model, deterministic.

    python -m litcurator.analysis_harness.gates_no_llm.terminal_grader
"""

from collections import Counter

from .. import grading as E


def _one(expect, pp, patterns, first_surfaced, history=None,
         flag_patterns=None, flag_intended=None):
    """Grade an expect carrying exactly one check; return its ok bool. The last two args are
    only needed by the dual-nature check, which reads flag-level attachment."""
    res = E.check_terminal(expect, pp, patterns, first_surfaced, history or [],
                           flag_patterns, flag_intended)
    assert len(res) == 1, f"expected 1 check, got {len(res)}: {res}"
    return res[0][0]


def _pat(pid, direction="under", status="created", recurred_count=0):
    return {"id": pid, "direction": direction, "status": status,
            "recurred_count": recurred_count}


def _hist(*active_counts):
    """A minimal history: one entry per session carrying only the open-pile size, which is
    all open_pile_settles reads."""
    return [{"session": i, "new": [], "active_count": n}
            for i, n in enumerate(active_counts)]


def _pool(*unattached_counts):
    """A history carrying only the unattached-flag count per session, which is all
    pool_drains reads."""
    return [{"session": i, "new": [], "active_count": 1, "unattached": n}
            for i, n in enumerate(unattached_counts)]


def test_coalesces_to_one():
    band = {"coalesces_to_one": [{"label": "W", "min_session": 3, "max_session": 10,
                                  "min_purity": 0.6}]}
    # PASS: exactly one pure pattern, first surfaced inside the band
    assert _one(band, {"pW": Counter(W=5)}, [_pat("pW")], {"W": 6}) is True
    # FAIL: fragmented into two patterns
    assert _one(band, {"pW": Counter(W=3), "pW2": Counter(W=2)},
                [_pat("pW"), _pat("pW2")], {"W": 6}) is False
    # FAIL: surfaced too early (before the band)
    assert _one(band, {"pW": Counter(W=5)}, [_pat("pW")], {"W": 1}) is False
    # FAIL: the pattern is only half one intended pattern (0.5 < 0.6)
    assert _one(band, {"pW": Counter(W=3, X=3)}, [_pat("pW")], {"W": 6}) is False
    # FAIL: never surfaced at all
    assert _one(band, {"pA": Counter(A=4)}, [_pat("pA")], {"A": 2}) is False
    print("coalesces_to_one: one-pure-in-band passes; fragment / early / impure / absent fail")


def test_intended_patterns_surface():
    assert _one({"intended_patterns_surface": ["A"]}, {"pA": Counter(A=3)}, [_pat("pA")], {"A": 0}) is True
    assert _one({"intended_patterns_surface": ["Z"]}, {"pA": Counter(A=3)}, [_pat("pA")], {"A": 0}) is False
    print("intended_patterns_surface: present passes, absent fails")


def test_stay_separate():
    # PASS: A and B live in disjoint, uncontaminated patterns
    assert _one({"stay_separate": [["A", "B"]]},
                {"pA": Counter(A=3), "pB": Counter(B=3)},
                [_pat("pA"), _pat("pB")], {"A": 0, "B": 1}) is True
    # FAIL: one TASTE pattern absorbed >=2 of each -> the two tastes got fused
    assert _one({"stay_separate": [["A", "B"]]},
                {"pM": Counter(A=2, B=2)}, [_pat("pM", "under")], {"A": 0}) is False
    # PASS: a pattern spanning both is fine when it is a cross-cutting observation about the
    # judge rather than a taste -- "the judge penalises specialist journals" HAS to span them
    assert _one({"stay_separate": [["A", "B"]]},
                {"pX": Counter(A=2, B=2)}, [_pat("pX", "judge-not-applying")], {"A": 0}) is True
    # PASS: same for a wording complaint
    assert _one({"stay_separate": [["A", "B"]]},
                {"pS": Counter(A=2, B=2)}, [_pat("pS", "sharpen")], {"A": 0}) is True
    print("stay_separate: disjoint passes; a fused TASTE pattern fails; "
          "a cross-cutting sharpen/judge-not-applying pattern passes")


def test_shared_flag_in_both():
    spec = {"shared_flag_in_both": [{"labels": ["B", "C"]}]}
    fi = {1: ["B", "C"], 2: "B", 3: "C"}          # flag 1 is the paper that is honestly both
    # A dual flag counts into BOTH labels of whichever pattern holds it, so its presence shows
    # up as a single cross-label count in each pattern -- not as contamination.
    split_pp = {"pB": Counter(B=4, C=1), "pC": Counter(C=4, B=1)}
    split_rows = [_pat("pB"), _pat("pC")]

    # PASS: two patterns, and the dual flag is attached to one of each
    assert _one(spec, split_pp, split_rows, {"B": 0, "C": 0},
                flag_patterns={1: {"pB", "pC"}, 2: {"pB"}, 3: {"pC"}},
                flag_intended=fi) is True
    # FAIL: the dual flag was dropped into the B pattern only -- the C evidence is lost
    assert _one(spec, split_pp, split_rows, {"B": 0, "C": 0},
                flag_patterns={1: {"pB"}, 2: {"pB"}, 3: {"pC"}},
                flag_intended=fi) is False
    # FAIL: chimera -- one blended TASTE pattern absorbed both tastes instead of splitting them
    assert _one(spec, {"pM": Counter(B=4, C=4)}, [_pat("pM")], {"B": 0},
                flag_patterns={1: {"pM"}, 2: {"pM"}, 3: {"pM"}},
                flag_intended=fi) is False
    # PASS: a pattern spanning both is fine when it is a cross-cutting observation rather than a
    # taste. It has to span them to be true, so counting it as a chimera fails a real finding --
    # the same carve-out stay_separate, fragmentation and min_purity already make.
    assert _one(spec, {"pB": Counter(B=4, C=1), "pC": Counter(C=4, B=1),
                       "pX": Counter(B=2, C=2)},
                [_pat("pB"), _pat("pC"), _pat("pX", "judge-not-applying")], {"B": 0, "C": 0},
                flag_patterns={1: {"pB", "pC"}, 2: {"pB"}, 3: {"pC"}},
                flag_intended=fi) is True
    # FAIL: no dual flag in the fixture at all -- nothing to verify, so this cannot pass silently
    assert _one(spec, split_pp, split_rows, {"B": 0, "C": 0},
                flag_patterns={2: {"pB"}, 3: {"pC"}},
                flag_intended={2: "B", 3: "C"}) is False
    print("shared_flag_in_both: split+shared passes; dropped / chimera / no-dual fail")


def test_named_disinterest_not_dropped():
    spec = {"named_disinterest_not_dropped": [
        {"label": "A", "directions": ["judge-not-applying", "over"]}]}
    # PASS: recorded as judge-not-applying -- the profile says it, the judge ignores it
    assert _one(spec, {"pA": Counter(A=5)},
                [_pat("pA", "judge-not-applying")], {"A": 0}) is True
    # PASS: recorded as a plain over-trigger is also acceptable -- it was not dropped
    assert _one(spec, {"pA": Counter(A=5)}, [_pat("pA", "over")], {"A": 0}) is True
    # FAIL: dropped entirely as "the profile already covers this"
    assert _one(spec, {"pB": Counter(B=3)}, [_pat("pB")], {"B": 0}) is False
    # FAIL: recorded, but as a profile GAP -- the wrong diagnosis, it points at the prompt
    assert _one(spec, {"pA": Counter(A=5)}, [_pat("pA", "under")], {"A": 0}) is False
    print("named_disinterest_not_dropped: JNA/over passes; dropped or mis-directed fails")


def test_recurrence_accumulates():
    spec = {"recurrence_accumulates": [{"label": "B", "min_recurrences": 2}]}
    # PASS: one closed pattern logged the return twice
    assert _one(spec, {"pB": Counter(B=6)},
                [_pat("pB", status="rejected", recurred_count=2)], {"B": 0}) is True
    # FAIL: it came back only once
    assert _one(spec, {"pB": Counter(B=6)},
                [_pat("pB", status="rejected", recurred_count=1)], {"B": 0}) is False
    # FAIL: two SIBLING patterns logged one return each. This is the reason the reduction is
    # MAX and not SUM -- summing reads 1+1 as "came back twice", when what actually happened is
    # the gap fragmented and each half was seen once. That is the failure the check exists to
    # catch, so it must not be the thing that makes it pass.
    assert _one(spec, {"pB": Counter(B=3), "pB2": Counter(B=3)},
                [_pat("pB", status="rejected", recurred_count=1),
                 _pat("pB2", status="rejected", recurred_count=1)], {"B": 0}) is False
    # FAIL: the gap never produced a pattern at all, so nothing could have recurred
    assert _one(spec, {"pA": Counter(A=4)}, [_pat("pA")], {"A": 0}) is False
    print("recurrence_accumulates: repeated returns on ONE pattern pass; "
          "a single return, split siblings, or no pattern fail")


def test_open_pile_settles():
    spec = {"open_pile_settles": [{"over_last_sessions": 2, "max_growth": 1}]}
    # PASS: the pile grew early and then flattened -- 4 -> 6 -> 7 -> 7 is +0 over the last two
    assert _one(spec, {"pA": Counter(A=3)}, [_pat("pA")], {"A": 0}, _hist(4, 6, 7, 7)) is True
    # PASS: shrinking counts as settling (the human closed more than the model opened)
    assert _one(spec, {"pA": Counter(A=3)}, [_pat("pA")], {"A": 0}, _hist(4, 6, 7, 5)) is True
    # FAIL: the treadmill -- a steady climb of one per session. Every per-round
    # max_open_pattern_growth check passes on this history (each step is only +1); only the
    # cumulative view over the tail sees it, which is the whole reason this check exists.
    assert _one(spec, {"pA": Counter(A=3)}, [_pat("pA")], {"A": 0}, _hist(4, 5, 6, 7)) is False
    # FAIL: no history at all -- there is nothing to measure, so it must not pass silently
    assert _one(spec, {"pA": Counter(A=3)}, [_pat("pA")], {"A": 0}, []) is False
    print("open_pile_settles: a flattening or shrinking tail passes; "
          "a +1-per-session climb and an empty history fail")


def test_pool_drains():
    """The failure this catches is the one coalesces_to_one CANNOT see. If the machinery mints a
    pattern early and then holds every later flag instead of attaching it, the end state is a
    single pure pattern surfaced in the right window -- a clean pass -- while every flag after
    the second one was left on the floor. The tell is the unattached pool, which stays flat when
    flags are being absorbed and climbs when they are not."""
    spec = {"pool_drains": [{"over_last_sessions": 3, "max_growth": 0}]}
    # PASS: one flag in, one attached, every session -- the pool never grows
    assert _one(spec, {"pW": Counter(W=9)}, [_pat("pW")], {"W": 1},
                _pool(1, 2, 1, 1, 1, 1)) is True
    # PASS: a backlog being worked off
    assert _one(spec, {"pW": Counter(W=9)}, [_pat("pW")], {"W": 1},
                _pool(1, 4, 3, 2, 1, 1)) is True
    # FAIL: recognised once, then ignored -- the pattern exists but stopped absorbing
    assert _one(spec, {"pW": Counter(W=2)}, [_pat("pW")], {"W": 1},
                _pool(1, 2, 3, 4, 5, 6)) is False
    # FAIL: nothing recorded, so nothing can be concluded
    assert _one(spec, {"pW": Counter(W=9)}, [_pat("pW")], {"W": 1}, []) is False
    print("pool_drains: a flat or draining pool passes; a climbing pool and an empty history fail")


def test_empty_expect():
    assert E.check_terminal({}, {"pA": Counter(A=3)}, [_pat("pA")], {"A": 0}, []) == []
    print("empty terminal_expect -> no checks")


# The checks this gate runs, in order. Named here once so the harness and a direct run
# cannot drift apart.
CHECKS = [
    test_coalesces_to_one,
    test_intended_patterns_surface,
    test_stay_separate,
    test_shared_flag_in_both,
    test_named_disinterest_not_dropped,
    test_recurrence_accumulates,
    test_open_pile_settles,
    test_pool_drains,
    test_empty_expect,
]


if __name__ == "__main__":
    for check in CHECKS:
        check()
    print("\nALL CHECKS PASSED")
