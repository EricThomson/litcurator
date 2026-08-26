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


def _pat(pid, direction="under", status="created", recurred_count=0,
         name="", description="", suggested_edit="", direction_proposed=None):
    """The text fields default to EMPTY so every pre-existing caller keeps grading exactly the
    provenance graph and nothing else. Only note_wording_survives fills them."""
    return {"id": pid, "direction": direction, "status": status,
            "recurred_count": recurred_count, "name": name,
            "description": description, "suggested_edit": suggested_edit,
            "direction_proposed": direction_proposed}


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
    # FAIL whatever the direction. The cross-cutting exemption went with the diagnosis values
    # on 2026-08-25 -- a pattern holding >=2 of each has fused them, full stop.
    assert _one({"stay_separate": [["A", "B"]]},
                {"pX": Counter(A=2, B=2)}, [_pat("pX", "over")], {"A": 0}) is False
    print("stay_separate: disjoint passes; a fused pattern fails whatever its direction")


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
    # FAIL: a third pattern holding >=2 of each is a chimera whatever its direction, even
    # alongside a correct split. The cross-cutting exemption went on 2026-08-25.
    assert _one(spec, {"pB": Counter(B=4, C=1), "pC": Counter(C=4, B=1),
                       "pX": Counter(B=2, C=2)},
                [_pat("pB"), _pat("pC"), _pat("pX", "over")], {"B": 0, "C": 0},
                flag_patterns={1: {"pB", "pC"}, 2: {"pB"}, 3: {"pC"}},
                flag_intended=fi) is False
    # FAIL: no dual flag in the fixture at all -- nothing to verify, so this cannot pass silently
    assert _one(spec, split_pp, split_rows, {"B": 0, "C": 0},
                flag_patterns={2: {"pB"}, 3: {"pC"}},
                flag_intended={2: "B", 3: "C"}) is False
    print("shared_flag_in_both: split+shared passes; dropped / chimera / no-dual fail")


def test_named_disinterest_not_dropped():
    spec = {"named_disinterest_not_dropped": [{"label": "A", "directions": ["over"]}]}
    # PASS: recorded as a plain over-trigger -- the point is that it was not dropped
    assert _one(spec, {"pA": Counter(A=5)}, [_pat("pA", "over")], {"A": 0}) is True
    # FAIL: dropped entirely as "the profile already covers this"
    assert _one(spec, {"pB": Counter(B=3)}, [_pat("pB")], {"B": 0}) is False
    # FAIL: recorded, but as a profile GAP -- the wrong diagnosis, it points at the prompt
    assert _one(spec, {"pA": Counter(A=5)}, [_pat("pA", "under")], {"A": 0}) is False
    print("named_disinterest_not_dropped: JNA/over passes; dropped or mis-directed fails")


def test_direction_not_inverted():
    """Grades what the MODEL proposed, not what got recorded. The recorded direction is computed
    from the flags in code, so asserting it would be asserting arithmetic -- green by
    construction, which is the vacuous pass this suite has been bitten by twice."""
    spec = {"direction_not_inverted": [{"label": "T", "taste": "over"}]}
    # PASS: the model agreed with the flags
    assert _one(spec, {"pT": Counter(T=6)},
                [_pat("pT", "over", direction_proposed="over")], {"T": 0}) is True
    # PASS: anything that is not the opposite sign is not an inversion. These two values were
    # legal directions until 2026-08-25; the model can no longer propose them, but an
    # off-schema word still reaches direction_proposed RAW (see _record_new), and it must not
    # be read as an inversion just because the clamp rewrote the recorded value to `under`.
    assert _one(spec, {"pT": Counter(T=6)},
                [_pat("pT", "over", direction_proposed="judge-not-applying")], {"T": 0}) is True
    assert _one(spec, {"pT": Counter(T=6)},
                [_pat("pT", "over", direction_proposed="sharpen")], {"T": 0}) is True
    # FAIL: the model asked for the OPPOSITE sign. THE RECORDED VALUE IS CORRECT HERE -- code
    # already fixed it -- so a check reading `direction` would score this green while the prompt
    # defect it exists to catch went unreported. This is the whole reason the check moved.
    assert _one(spec, {"pT": Counter(T=6)},
                [_pat("pT", "over", direction_proposed="under")], {"T": 0}) is False
    # FAIL: one good pattern does not excuse an inverted sibling
    assert _one(spec, {"pT": Counter(T=3), "pT2": Counter(T=3)},
                [_pat("pT", "over", direction_proposed="over"),
                 _pat("pT2", "over", direction_proposed="under")], {"T": 0}) is False
    # FAIL: no pattern at all must not pass vacuously
    assert _one(spec, {"pB": Counter(B=2)}, [_pat("pB")], {"B": 0}) is False
    # BACKWARD COMPATIBILITY: runs predating the computed-sign change carry no proposal, so the
    # recorded direction stands in
    assert _one(spec, {"pT": Counter(T=6)}, [_pat("pT", "under")], {"T": 0}) is False
    assert _one(spec, {"pT": Counter(T=6)}, [_pat("pT", "over")], {"T": 0}) is True
    # and the mirror sign, so the check is not hard-coded to one direction
    under = {"direction_not_inverted": [{"label": "T", "taste": "under"}]}
    assert _one(under, {"pT": Counter(T=6)},
                [_pat("pT", "under", direction_proposed="under")], {"T": 0}) is True
    assert _one(under, {"pT": Counter(T=6)},
                [_pat("pT", "under", direction_proposed="over")], {"T": 0}) is False
    print("direction_not_inverted: grades the model's PROPOSAL, not the computed record; "
          "cross-cutting passes, the opposite sign fails even when the record is right")


def test_note_wording_survives():
    """The only checks that read produced TEXT. Everything here is about them being DISCRETE --
    a substring either is or is not present -- because that is what lets the paid gate settle in
    ONE run instead of needing reps and an average."""
    term = {"note_wording_survives": [
        {"label": "T", "kind": "term", "markers": ["circatidal"],
         "fields": ("name", "description", "suggested_edit")}]}
    # PASS: the user's word survived into the description
    assert _one(term, {"pT": Counter(T=4)},
                [_pat("pT", description="judge over-scores circatidal rhythm work")],
                {"T": 0}) is True
    # PASS: case does not matter -- the model title-cases things constantly
    assert _one(term, {"pT": Counter(T=4)},
                [_pat("pT", name="Circatidal Rhythm Studies")], {"T": 0}) is True
    # FAIL: generalised away. This is the actual failure mode -- a true, useless paraphrase
    assert _one(term, {"pT": Counter(T=4)},
                [_pat("pT", description="reduced interest in chronobiology")], {"T": 0}) is False
    # FAIL: no pattern for the label at all -- must not pass vacuously by having nothing to read
    assert _one(term, {"pB": Counter(B=3)}, [_pat("pB")], {"B": 0}) is False

    # FIELD SCOPING is the point of the `directive` check and needs its own proof. The wording
    # can survive in the description and still never reach suggested_edit, which is the field
    # the human actually edits and the one the prompt's note-wall paragraph governs. If scoping
    # did not work, that failure would score green off the description.
    directive = {"note_wording_survives": [
        {"label": "T", "kind": "directive", "markers": ["disinterest"],
         "fields": ("suggested_edit",)}]}
    assert _one(directive, {"pT": Counter(T=4)},
                [_pat("pT", suggested_edit="add circatidal work to the disinterest list")],
                {"T": 0}) is True
    assert _one(directive, {"pT": Counter(T=4)},
                [_pat("pT", description="belongs on the disinterest list",
                      suggested_edit="de-prioritise tidal chronobiology")], {"T": 0}) is False

    # ANY-OF across markers and across patterns: one accepted synonym landing anywhere is a
    # pass. Deliberately tolerant -- the case is meant to be EASY, so a near-miss in wording
    # must not produce a red that sends you hunting for a bug that is not there.
    multi = {"note_wording_survives": [
        {"label": "T", "kind": "directive", "markers": ["disinterest list", "disinterest section"],
         "fields": ("suggested_edit",)}]}
    assert _one(multi, {"pT": Counter(T=2), "pT2": Counter(T=2)},
                [_pat("pT", suggested_edit="drop these"),
                 _pat("pT2", suggested_edit="goes in the disinterest section")],
                {"T": 0}) is True
    print("note_wording_survives: substring present passes; paraphrase, missing pattern and "
          "wrong-field all fail; markers and patterns are any-of")


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
    test_direction_not_inverted,
    test_note_wording_survives,
    test_recurrence_accumulates,
    test_open_pile_settles,
    test_pool_drains,
    test_empty_expect,
]


if __name__ == "__main__":
    for check in CHECKS:
        check()
    print("\nALL CHECKS PASSED")
