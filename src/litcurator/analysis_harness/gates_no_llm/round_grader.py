"""
No-LLM test for the per-session grader and the scripted human: grading.check_round and
machinery.apply_actions. Builds a small scratch database with a hand-made provenance graph,
so every check is deterministic and free.

This gate exists because those two functions had no test at all -- their only caller was a
paid gate, so every exercise of them cost an API call, and two real bugs lived there for
weeks. Each check below pins one of them:

  - the scripted human decided ONE pattern per gap and silently left its siblings open, so a
    returning gap could merge into an open sibling and the recurrence check went red for
    behavior that was correct;
  - when no pattern matched at all, the decision was skipped, and `no_new_pattern_for` then
    passed VACUOUSLY -- a genuine duplicate scored green.

Unlike record_stage and pattern_schema, this needs no copy of the live database:
machinery.build_db and add_round_flags fabricate their own articles, so there are no
preconditions and nothing to clean up but one scratch file.

    python -m litcurator.analysis_harness.gates_no_llm.round_grader
"""

from collections import Counter

from litcurator import db_interface as DB, profile_analysis as PA

from .. import grading as G
from .. import machinery as M

PROFILE = "I follow systems neuroscience: circuits, computation, and behavior."


def _paper(intended, n, over=False):
    """One synthetic flagged paper. `over` flips the delta sign so a set of papers can be an
    over-scored gap or an under-scored one."""
    judge, user = (0.70, 0.20) if over else (0.25, 0.75)
    return {"pmid": f"RG{n:05d}", "intended": intended,
            "title": f"Synthetic paper {n} for {intended}",
            "abstract": f"A synthetic abstract for {intended}, paper {n}.",
            "journal": "Journal of Synthetic Results",
            "judge_score": judge, "user_score": user,
            "rationale": f"The judge's wrong reason for {intended}.", "note": ""}


def _world(counts, over=()):
    """A scratch database holding `counts` = {intended_label: n_papers}.

    Returns (conn, flag_intended, flags_by_label) where flags_by_label maps a label to the
    ordered flag ids for its papers, so a test can hand-pick exactly which flags go into
    which pattern and the provenance graph is known rather than guessed."""
    path = M.scratch_db_path("round_grader")
    conn, run_id, _ = M.build_db(path, PROFILE)
    papers, n = [], 0
    for label, how_many in counts.items():
        for _ in range(how_many):
            n += 1
            papers.append(_paper(label, n, over=label in over))
    flag_intended = {}
    M.add_round_flags(conn, run_id, papers, flag_intended)
    flags_by_label = {}
    for flag_id, label in flag_intended.items():
        flags_by_label.setdefault(label, []).append(flag_id)
    return conn, flag_intended, flags_by_label


def _close(conn):
    conn.close()
    M.scratch_db_path("round_grader").unlink(missing_ok=True)


def _grade(conn, expect, flag_intended, pattern_for, new_ids=(), summary=None, open_before=0):
    """Run check_round and return {label_prefix: (ok, detail)} keyed on the part of each check
    label before the first colon, which is the property being checked."""
    summary = summary or {"new": [], "merged": [], "recurred": [], "held": [], "skipped": []}
    results = G.check_round(conn, expect, summary, [], flag_intended,
                            set(new_ids), open_before, pattern_for)
    return {label.split(":", 1)[0]: (ok, detail) for ok, label, detail in results}


# ---------------------------------------------------------------------------
# The scripted human
# ---------------------------------------------------------------------------

def test_every_matching_pattern_is_decided():
    """A gap can arrive as several patterns, and the human must decide ALL of them.

    Deciding only one leaves a sibling open, which is a legitimate home for the gap when it
    returns -- so the recurrence lands there as 'carried' and the check goes red for correct
    behavior. This is the exact shape of the run that failed: the human rejected an impure
    B pattern (B:4, C:2) because it had more TOTAL flags, leaving a pure B:8 pattern open."""
    conn, flag_intended, by_label = _world({"B": 6, "A": 3})
    pure = DB.create_pattern(conn, "pure B", "under", flag_ids=by_label["B"][:4])
    impure = DB.create_pattern(conn, "impure B", "under",
                               flag_ids=by_label["B"][4:] + by_label["A"][:1])
    other = DB.create_pattern(conn, "A's own", "over", flag_ids=by_label["A"][1:])
    pattern_for, logged = {}, []
    M.apply_actions(conn, PROFILE, [("reject", "B")], flag_intended, pattern_for, logged.append)

    status = {p["id"]: p["status"] for p in DB.get_patterns(conn)}
    assert status[pure] == "rejected", f"the pure B pattern was left open: {status[pure]}"
    assert status[impure] == "rejected", f"the impure B pattern was left open: {status[impure]}"
    assert status[other] != "rejected", "A's pattern must not be touched by a decision on B"
    assert pattern_for["B"]["ids"] == {pure, impure}, pattern_for["B"]
    assert pattern_for["B"]["action"] == "reject", pattern_for["B"]
    _close(conn)
    print("scripted human: decides EVERY pattern owning the gap, leaves other gaps alone")


def test_unresolved_decision_is_recorded():
    """When nothing covers the gap the decision must still be RECORDED, with an empty id set.

    It used to just log and continue, leaving pattern_for[label] unset -- which is what let
    the duplicate check pass vacuously (see the next test)."""
    conn, flag_intended, by_label = _world({"A": 3})
    DB.create_pattern(conn, "A's own", "over", flag_ids=by_label["A"])
    pattern_for, logged = {}, []
    M.apply_actions(conn, PROFILE, [("reject", "B")], flag_intended, pattern_for, logged.append)

    assert "B" in pattern_for, "an unresolved decision must be recorded, not omitted"
    assert pattern_for["B"]["ids"] == set(), pattern_for["B"]
    assert any("UNRESOLVED" in line for line in logged), logged
    _close(conn)
    print("scripted human: an unresolved decision is recorded with an empty id set, and said loudly")


# ---------------------------------------------------------------------------
# The cross-session assertions
# ---------------------------------------------------------------------------

def test_no_duplicate_fails_loudly_when_unresolved():
    """The silent false-green. With no decided pattern there is no direction to compare
    against, so the old check found zero duplicates and passed -- while a blatant duplicate
    sat in new_ids. An unverifiable check must fail, not pass."""
    conn, flag_intended, by_label = _world({"B": 4})
    dupe = DB.create_pattern(conn, "a duplicate of B", "under", flag_ids=by_label["B"])
    pattern_for = {"B": {"action": "reject", "ids": set()}}      # decision never resolved
    got = _grade(conn, {"no_new_pattern_for": ["B"]}, flag_intended, pattern_for, new_ids=[dupe])

    ok, detail = got["no-duplicate"]
    assert ok is False, "an unresolvable duplicate check must FAIL, not pass vacuously"
    assert "UNRESOLVED" in detail, detail
    _close(conn)
    print("no-duplicate: an unresolved decision fails loudly (was: passed vacuously)")


def test_no_duplicate_same_vs_other_diagnosis():
    """A new pattern repeating any DECIDED direction is a duplicate; a different diagnosis of
    the same papers (sharpen, judge-not-applying) is a second finding and stays allowed."""
    conn, flag_intended, by_label = _world({"B": 6})
    decided = DB.create_pattern(conn, "decided B", "under", flag_ids=by_label["B"][:3])
    DB.add_pattern_event(conn, decided, "rejected", note="test")
    same = DB.create_pattern(conn, "same diagnosis", "under", flag_ids=by_label["B"][3:5])
    other = DB.create_pattern(conn, "other diagnosis", "judge-not-applying",
                              flag_ids=by_label["B"][5:])
    pattern_for = {"B": {"action": "reject", "ids": {decided}}}

    got = _grade(conn, {"no_new_pattern_for": ["B"]}, flag_intended, pattern_for, new_ids=[same])
    assert got["no-duplicate"][0] is False, "a same-direction new pattern is a duplicate"

    got = _grade(conn, {"no_new_pattern_for": ["B"]}, flag_intended, pattern_for, new_ids=[other])
    assert got["no-duplicate"][0] is True, "a different diagnosis is a second finding, not a duplicate"
    assert "judge-not-applying" in got["no-duplicate"][1], got["no-duplicate"][1]
    _close(conn)
    print("no-duplicate: same direction fails, a different diagnosis passes and is named")


def test_purity_exempts_cross_cutting_patterns():
    """min_purity applies to TASTE patterns only. A sharpen or judge-not-applying pattern is an
    observation about the profile or the judge rather than about one taste, so spanning several
    tastes is what makes it true -- grading it as impure fails a real finding.

    This is not hypothetical. A live run produced "Circuit Access Filter -- Judge Not Applying"
    from 8 A flags, 4 B and 3 D, purity 0.53, and the gate went red on the most useful pattern
    in the run."""
    conn, flag_intended, by_label = _world({"A": 6, "B": 4})
    # A deliberately mixed pattern: half one taste, half another.
    mixed = by_label["A"][:3] + by_label["B"][:3]
    taste = DB.create_pattern(conn, "impure taste pattern", "under", flag_ids=mixed)
    meta = DB.create_pattern(conn, "cross-cutting observation", "judge-not-applying",
                             flag_ids=mixed)
    expect = {"min_purity": 0.8}

    got = _grade(conn, expect, flag_intended, {}, new_ids=[taste])
    assert got["purity"][0] is False, "an impure TASTE pattern must still fail"

    got = _grade(conn, expect, flag_intended, {}, new_ids=[meta])
    assert got["purity"][0] is True, "a cross-cutting pattern is meant to span tastes"
    assert "exempt" in got["purity"][1], got["purity"][1]

    # Mixed batch: the taste pattern still decides the verdict, the meta one is set aside.
    got = _grade(conn, expect, flag_intended, {}, new_ids=[taste, meta])
    assert got["purity"][0] is False, "one impure taste pattern must not be masked by an exemption"
    _close(conn)
    print("purity: impure taste pattern fails, cross-cutting pattern is exempt and said to be")


def test_recurrence_is_any_of():
    """The return is ONE event. Logging it against any of the gap's closed patterns is
    correct; demanding all of them would demand bookkeeping nobody wants."""
    conn, flag_intended, by_label = _world({"B": 6})
    one = DB.create_pattern(conn, "B one", "under", flag_ids=by_label["B"][:3])
    two = DB.create_pattern(conn, "B two", "under", flag_ids=by_label["B"][3:])
    for pid in (one, two):
        DB.add_pattern_event(conn, pid, "rejected", note="test")
    expect = {"recurrence_logged_for": ["B"]}

    summary = {"new": [], "merged": [], "recurred": [{"id": two}], "held": [], "skipped": []}
    got = _grade(conn, expect, flag_intended, {"B": {"action": "reject", "ids": {one, two}}},
                 summary=summary)
    assert got["recurrence"][0] is True, "a recurrence on EITHER closed pattern is correct"

    summary = {"new": [], "merged": [], "recurred": [], "held": [], "skipped": []}
    got = _grade(conn, expect, flag_intended, {"B": {"action": "reject", "ids": {one, two}}},
                 summary=summary)
    assert got["recurrence"][0] is False, "no recurrence anywhere must fail"

    got = _grade(conn, expect, flag_intended, {"B": {"action": "reject", "ids": set()}})
    assert got["recurrence"][0] is False and "UNRESOLVED" in got["recurrence"][1]
    _close(conn)
    print("recurrence: any-of passes, none fails, unresolved fails loudly")


def test_merged_into_existing_is_any_of():
    conn, flag_intended, by_label = _world({"C": 6})
    one = DB.create_pattern(conn, "C one", "under", flag_ids=by_label["C"][:3])
    two = DB.create_pattern(conn, "C two", "under", flag_ids=by_label["C"][3:])
    expect = {"merged_into_existing": ["C"]}
    decided = {"C": {"action": "carry", "ids": {one, two}}}

    summary = {"new": [], "merged": [{"id": one}], "recurred": [], "held": [], "skipped": []}
    assert _grade(conn, expect, flag_intended, decided, summary=summary)["merge"][0] is True
    summary = {"new": [], "merged": [], "recurred": [], "held": [], "skipped": []}
    assert _grade(conn, expect, flag_intended, decided, summary=summary)["merge"][0] is False
    _close(conn)
    print("merge: any-of passes, none fails")


def test_stay_closed_is_all_of():
    """One reopened pattern is a real defect and must not be masked by a sibling that stayed
    closed. And a label whose last action was `carry` closes nothing, so asking it to stay
    closed is a fixture error rather than a machinery failure."""
    conn, flag_intended, by_label = _world({"A": 6}, over=("A",))
    one = DB.create_pattern(conn, "A one", "over", flag_ids=by_label["A"][:3])
    two = DB.create_pattern(conn, "A two", "over", flag_ids=by_label["A"][3:])
    for pid in (one, two):
        DB.add_pattern_event(conn, pid, "rejected", note="test")
    expect = {"stay_closed": ["A"]}
    decided = {"A": {"action": "reject", "ids": {one, two}}}
    assert _grade(conn, expect, flag_intended, decided)["stay-closed"][0] is True

    DB.add_pattern_event(conn, two, "carried", note="reopened")
    ok, detail = _grade(conn, expect, flag_intended, decided)["stay-closed"]
    assert ok is False, "a reopened sibling must fail even though the other stayed closed"
    assert two[:10] in detail, detail

    ok, detail = _grade(conn, expect, flag_intended,
                        {"A": {"action": "carry", "ids": {one}}})["stay-closed"]
    assert ok is False and "FIXTURE ERROR" in detail, detail
    _close(conn)
    print("stay-closed: all-of; a reopened sibling fails and is named; carry is a fixture error")


def test_recovered_candidate_is_named():
    """A 'duplicate' that came from the lossless-recovery path (a malformed merge target) is
    not the model deliberately minting a twin. summary['new'] already carries recovered=True;
    the report must say so instead of blaming the model."""
    conn, flag_intended, by_label = _world({"D": 4}, over=("D",))
    decided = DB.create_pattern(conn, "decided D", "over", flag_ids=by_label["D"][:2])
    DB.add_pattern_event(conn, decided, "carried", note="test")
    recovered = DB.create_pattern(conn, "recovered D", "over", flag_ids=by_label["D"][2:])
    summary = {"new": [{"id": recovered, "recovered": True}], "merged": [], "recurred": [],
               "held": [], "skipped": []}
    got = _grade(conn, {"no_new_pattern_for": ["D"]}, flag_intended,
                 {"D": {"action": "carry", "ids": {decided}}},
                 new_ids=[recovered], summary=summary)
    ok, detail = got["no-duplicate"]
    assert ok is False, "it is still a duplicate"
    assert "RECOVERED" in detail.upper(), detail
    _close(conn)
    print("no-duplicate: a recovered malformed candidate is named as such, not blamed on the model")


# ---------------------------------------------------------------------------
# The memory block the model is shown
# ---------------------------------------------------------------------------

def test_memory_block_shows_evidence():
    """The block must carry enough to recognise a returning gap: the PAPERS behind a pattern,
    the direction of CLOSED patterns (the bracket used to be reused for status, losing it),
    and the flag count. It must NOT carry suggested_edit -- that is description again in
    imperative mood, it nearly doubles the block, and it is first-person profile prose landing
    in a message that ends with the real profile as source of truth."""
    assert hasattr(DB, "get_pattern_examples"), \
        "db_interface.get_pattern_examples(conn, ids, limit) does not exist yet"
    conn, flag_intended, by_label = _world({"B": 4})
    open_pid = DB.create_pattern(conn, "still open", "under", description="an open gap",
                                 suggested_edit="UNIQUESUGGESTEDEDIT open",
                                 flag_ids=by_label["B"][:2])
    closed_pid = DB.create_pattern(conn, "already decided", "under", description="a closed gap",
                                   suggested_edit="UNIQUESUGGESTEDEDIT closed",
                                   flag_ids=by_label["B"][2:])
    DB.add_pattern_event(conn, closed_pid, "rejected", note="test")

    ids = [open_pid, closed_pid]
    examples = DB.get_pattern_examples(conn, ids, limit=3)
    assert set(examples) == set(ids), examples
    for pid in ids:
        want = [r["title"] for r in DB.get_pattern_provenance(conn, pid)[:3]]
        got = [r["title"] for r in examples[pid]]
        assert got == want, f"batched examples drifted from get_pattern_provenance: {got} != {want}"

    block = PA._format_existing_patterns(
        DB.get_active_patterns(conn),
        DB.get_patterns(conn, statuses=("incorporated", "rejected")),
        examples=examples)

    assert "under" in block.split("CLOSED")[1], "closed patterns must still show their direction"
    assert "UNIQUESUGGESTEDEDIT" not in block, "suggested_edit must NOT be in the memory block"
    for pid in ids:
        for row in examples[pid]:
            assert row["title"][:40] in block, f"missing example paper: {row['title']}"
    assert "2 flag" in block, "flag_count must be shown"

    plain = PA._format_existing_patterns(DB.get_active_patterns(conn), [])
    assert "Synthetic paper" not in plain, "examples=None must render as before, with no papers"
    _close(conn)
    print("memory block: shows papers, closed-pattern direction and flag counts; "
          "omits suggested_edit; examples=None unchanged")


CHECKS = [
    test_every_matching_pattern_is_decided,
    test_unresolved_decision_is_recorded,
    test_no_duplicate_fails_loudly_when_unresolved,
    test_no_duplicate_same_vs_other_diagnosis,
    test_purity_exempts_cross_cutting_patterns,
    test_recurrence_is_any_of,
    test_merged_into_existing_is_any_of,
    test_stay_closed_is_all_of,
    test_recovered_candidate_is_named,
    test_memory_block_shows_evidence,
]


if __name__ == "__main__":
    for check in CHECKS:
        check()
    print("\nALL CHECKS PASSED")
