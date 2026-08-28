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


def _grade(conn, expect, flag_intended, pattern_for, new_ids=(), summary=None, open_before=0,
           candidates=(), memory_empty=False):
    """Run check_round and return {label_prefix: (ok, detail)} keyed on the part of each check
    label before the first colon, which is the property being checked."""
    summary = summary or {"new": [], "merged": [], "recurred": [], "held": [], "skipped": []}
    results = G.check_round(conn, expect, summary, list(candidates), flag_intended,
                            set(new_ids), open_before, pattern_for, memory_empty=memory_empty)
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


def test_no_duplicate_same_vs_opposite_direction():
    """A new pattern repeating a DECIDED direction is a duplicate; one carrying the opposite
    direction is a second finding and stays allowed.

    Until 2026-08-25 the "second finding" case was a `sharpen` or `judge-not-applying` pattern
    -- a different DIAGNOSIS of the same papers. Those values are gone, so the only surviving
    way to be a second finding is the opposite sign. That is a much narrower escape, which is
    the point: `computed_sign` forces two patterns citing the same flags to agree, so a real
    gap can no longer be split in two and hide behind different labels."""
    conn, flag_intended, by_label = _world({"B": 6})
    decided = DB.create_pattern(conn, "decided B", "under", flag_ids=by_label["B"][:3])
    DB.add_pattern_event(conn, decided, "rejected", note="test")
    same = DB.create_pattern(conn, "same direction", "under", flag_ids=by_label["B"][3:5])
    other = DB.create_pattern(conn, "opposite direction", "over", flag_ids=by_label["B"][5:])
    pattern_for = {"B": {"action": "reject", "ids": {decided}}}

    got = _grade(conn, {"no_new_pattern_for": ["B"]}, flag_intended, pattern_for, new_ids=[same])
    assert got["no-duplicate"][0] is False, "a same-direction new pattern is a duplicate"

    got = _grade(conn, {"no_new_pattern_for": ["B"]}, flag_intended, pattern_for, new_ids=[other])
    assert got["no-duplicate"][0] is True, "the opposite direction is a second finding"
    assert "over" in got["no-duplicate"][1], got["no-duplicate"][1]
    _close(conn)
    print("no-duplicate: same direction fails, the opposite direction passes and is named")


def test_purity_applies_to_every_new_pattern():
    """min_purity now covers EVERY new pattern, with no exemption.

    It used to exempt `sharpen` / `judge-not-applying` patterns, because a cross-cutting
    observation has to span tastes to be true -- a live run produced one from 8 A flags, 4 B
    and 3 D at purity 0.53, and the gate went red on the most useful pattern in the run. Those
    directions were deleted 2026-08-25, so the exemption has nothing to key on and every
    pattern is a claim about one taste.

    NOTE FOR WHOEVER WANTS THE EXEMPTION BACK: the old one keyed on a field the MODEL wrote, so
    a model could exempt itself from this check by picking a label. Put the next one in the
    fixture (an `allow_spanning` key), where the test author sets it."""
    conn, flag_intended, by_label = _world({"A": 6, "B": 4})
    mixed = by_label["A"][:3] + by_label["B"][:3]
    impure = DB.create_pattern(conn, "impure pattern", "under", flag_ids=mixed)
    pure = DB.create_pattern(conn, "pure pattern", "under", flag_ids=by_label["A"][3:])
    expect = {"min_purity": 0.8}

    got = _grade(conn, expect, flag_intended, {}, new_ids=[impure])
    assert got["purity"][0] is False, "an impure pattern must fail"

    got = _grade(conn, expect, flag_intended, {}, new_ids=[pure])
    assert got["purity"][0] is True, "a pure pattern must pass"

    # Mixed batch: worst-case wins, so one impure pattern still decides the verdict.
    got = _grade(conn, expect, flag_intended, {}, new_ids=[pure, impure])
    assert got["purity"][0] is False, "one impure pattern must not be masked by a pure sibling"
    _close(conn)
    print("purity: impure fails, pure passes, worst-case decides a mixed batch")


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


def test_scripted_human_can_decide_a_held_pattern():
    """apply_actions must see the SAME two lists the person does: Active and Held.

    The workbench has two tabs, and a held pattern is decided from the second one (Promote or
    Reject). While the scripted human scanned only get_active_patterns, a gap the model held in
    round one could never be decided: pattern_for[label] came back with an empty id set, and
    every cross-session check for that label then failed with UNRESOLVED two or three rounds
    later -- reading exactly like a model regression while being nothing of the kind. This pins
    both halves: the held pattern is found, and the decision lands on it."""
    conn, flag_intended, by_label = _world({"B": 4})
    held = DB.create_pattern(conn, "held gap B", "under", flag_ids=by_label["B"])
    DB.add_pattern_event(conn, held, "held", note="thin for now")
    assert held not in {p["id"] for p in DB.get_active_patterns(conn)}, "fixture: must be held"

    pattern_for = {}
    M.apply_actions(conn, PROFILE, [("reject", "B")], flag_intended, pattern_for, lambda *_: None)

    assert pattern_for["B"]["ids"] == {held},         f"the scripted human must reach a HELD pattern, got {pattern_for['B']}"
    assert DB.get_patterns(conn)[0]["status"] == "rejected", DB.get_patterns(conn)[0]["status"]
    _close(conn)
    print("apply_actions: a held pattern is visible to the scripted human and can be decided")


def test_memory_block_includes_held_patterns():
    """The block the model is shown must contain HELD patterns, with their papers.

    This is the check that would have caught the 2026-08-26 bug. Held patterns were added to
    the live path and not to machinery.run_round, so every paid gate ran with them invisible:
    accumulation held the connectome flag at session 0, saw "(empty -- no history yet)" at
    session 1, and minted a second pattern for the same gap. The fragmentation looked like
    model behaviour and was a missing argument.

    It tests build_memory_block, which is now the ONE assembler both callers use -- so this
    cannot go stale for one path while passing for the other."""
    conn, flag_intended, by_label = _world({"A": 4, "B": 2})
    shown = DB.create_pattern(conn, "shown gap", "under", flag_ids=by_label["A"][:2])
    held = DB.create_pattern(conn, "held gap", "under", description="thin but real",
                             flag_ids=by_label["B"])
    DB.add_pattern_event(conn, held, "held", note="not yet")

    block, active, held_list, closed = PA.build_memory_block(conn)
    assert [p["id"] for p in held_list] == [held], held_list
    assert [p["id"] for p in active] == [shown], active
    assert "HELD patterns" in block, block
    assert "held gap" in block and held[:12] in block, block
    # The papers are the identity signal that makes a returning gap recognisable at all.
    titles = [f["title"] for f in DB.get_pattern_provenance(conn, held)]
    assert any(t and t[:40] in block for t in titles), block

    # MAGNITUDE, on every remembered pattern. A fresh candidate arrives with its size described
    # in cluster's prose; without this the remembered side has a head count and nothing else, so
    # the model decides whether an accumulating gap is worth showing while blind to how badly
    # the judge was wrong on what it already holds.
    assert "|delta|" in block, block
    for row in DB.get_patterns(conn):
        assert row["mean_abs_delta"] is not None and row["total_delta"] is not None, dict(row)
        assert f"{row['mean_abs_delta']:.2f}" in block, (row["name"], block)
    _close(conn)
    print("memory block: every remembered pattern carries its |delta| size, not just a count")


def test_held_is_ranked_by_evidence_weight_not_head_count():
    """The Held tab is a ranked what-is-nearly-ready list, so its order has to mean something.

    Sorting on flag_count put a three-flag trivial pattern above a one-flag severe one, which is
    the volume criterion this project removed from the definition of a pattern, reintroduced in
    a sort. total_delta is the quantity the analysis prompt already describes: a steady small
    bias across many papers weighs the same as one big delta on a lone paper."""
    conn, flag_intended, by_label = _world({"A": 3, "B": 3}, over=("A",))
    many = DB.create_pattern(conn, "many small", "over", flag_ids=by_label["A"])
    one = DB.create_pattern(conn, "one large", "under", flag_ids=by_label["B"][:1])
    for pid in (many, one):
        DB.add_pattern_event(conn, pid, "held", note="thin")
    # Make the single-flag pattern carry the heavier evidence.
    conn.execute("UPDATE flags SET delta = 0.05 WHERE id IN (%s)"
                 % ",".join(str(i) for i in by_label["A"]))
    conn.execute("UPDATE flags SET delta = 0.60 WHERE id = ?", (by_label["B"][0],))
    conn.commit()

    order = [p["id"] for p in DB.get_held_patterns(conn)]
    assert order[0] == one, (
        "the heavier pattern must rank first; got head-count order "
        f"{[(p['name'], p['flag_count'], p['total_delta']) for p in DB.get_held_patterns(conn)]}")
    _close(conn)
    print("held ranking: ordered by evidence weight, so one big delta outranks three small ones")
    _close(conn)
    print("memory block: held patterns are shown, with their ids and papers")


def test_merge_on_an_empty_memory_is_caught():
    """With nothing recorded, every existing_pattern_id is invented -- so a merge is always
    wrong and needs no fixture judgement.

    The record step RECOVERS these (the bogus target becomes a new pattern rather than being
    lost, which record_stage pins), so nothing is dropped and nothing fails loudly. That is
    precisely why it needs its own check: the machinery handles it correctly and silently, and
    the only symptom is a duplicate card the model told you it was trying to avoid.

    Observed on real flags 2026-08-27, in the January dry run: empty memory, and the model
    emitted merge_into_open against an invented slug while explaining that it was folding two
    of THIS round's candidates together."""
    conn, flag_intended, _by = _world({"A": 2})
    clean = [{"choice": "new", "name": "a gap", "paper_numbers": [1], "rationale": "r"}]
    bogus = clean + [{"choice": "merge_into_open", "existing_pattern_id": "human-only-studies",
                      "paper_numbers": [2], "rationale": "same as the one above"}]

    got = _grade(conn, {}, flag_intended, {}, candidates=bogus, memory_empty=True)
    assert got["empty memory"][0] is False, got["empty memory"]
    assert "human-only-studies" in got["empty memory"][1], got["empty memory"][1]

    got = _grade(conn, {}, flag_intended, {}, candidates=clean, memory_empty=True)
    assert got["empty memory"][0] is True, got["empty memory"]

    # And it must NOT fire once there is a memory: merging is the correct move then.
    got = _grade(conn, {}, flag_intended, {}, candidates=bogus, memory_empty=False)
    assert "empty memory" not in got, got
    _close(conn)
    print("empty memory: a merge with nothing to merge into is caught, and only then")


CHECKS = [
    test_every_matching_pattern_is_decided,
    test_unresolved_decision_is_recorded,
    test_no_duplicate_fails_loudly_when_unresolved,
    test_no_duplicate_same_vs_opposite_direction,
    test_purity_applies_to_every_new_pattern,
    test_scripted_human_can_decide_a_held_pattern,
    test_memory_block_includes_held_patterns,
    test_merge_on_an_empty_memory_is_caught,
    test_held_is_ranked_by_evidence_weight_not_head_count,
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
