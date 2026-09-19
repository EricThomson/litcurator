"""
undo_stage.py -- unit tests for undo_error_analysis (free, no model).

The claim under test is strong and cheap to verify exactly: undoing the latest analysis run
leaves the four pattern-layer tables BYTE-IDENTICAL to their state before that run. Anything
weaker (row counts, spot checks) can pass while leaking a stamped event or eating a human one,
so every check here compares full table dumps.

Each check builds its own throwaway world and tears it down in a finally -- a failing check
that leaks its scratch DB takes every later check down with a Windows PermissionError, which
is the exact shape the workbench-actions negative control documented.
"""

import contextlib
import os

from litcurator import db_interface as DB
from litcurator import error_analysis as PA

from .. import machinery as H


def _dump(conn):
    """The pattern layer, byte-for-byte: every row of every table an analysis run can touch."""
    out = []
    for table in ("patterns", "pattern_flags", "pattern_events", "analysis_runs"):
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        rows = conn.execute(
            f"SELECT {', '.join(cols)} FROM {table} ORDER BY {cols[0]}").fetchall()
        out.append((table, [tuple(r) for r in rows]))
    return out


@contextlib.contextmanager
def _world():
    """A scratch database holding one recorded round: patterns A (open) and B (closed by an
    unstamped human 'incorporated', as the workbench writes them). Yields
    (conn, ordered_flags, ids) where ids maps names to pattern ids."""
    path = H.scratch_db_path("undo_stage")
    conn, scoring_run, profile_id = H.build_db(path, "synthetic profile")
    flag_intended = {}
    papers = [{"pmid": f"SYNUNDO{i}", "title": f"Paper {i}", "abstract": "a", "journal": "J",
               "judge_score": 0.5, "rationale": "r", "user_score": 0.9, "note": "n",
               "intended": "X"} for i in range(1, 6)]
    H.add_round_flags(conn, scoring_run, papers, flag_intended)
    ordered = sorted(DB.get_flags(conn, exclude_attached=True),
                     key=lambda f: (-abs(f["delta"]), f["id"]))
    try:
        run1 = _record(conn, profile_id, [
            _new("A", [1, 2]),
            _new("B", [3]),
        ], ordered)
        ids = {p["name"]: p["id"] for p in
               conn.execute("SELECT id, name FROM patterns").fetchall()}
        # The human closes B in the workbench: an UNSTAMPED event, like add_pattern_event
        # writes without a run id. Later rounds can then merge_into_closed against it.
        DB.add_pattern_event(conn, ids["B"], "incorporated", profile_id=profile_id)
        yield conn, profile_id, ordered, ids
    finally:
        conn.close()
        if path.exists():
            os.unlink(path)


def _new(name, papers, priority="act_now"):
    return {"choice": "new", "name": name, "direction": "under", "description": "d",
            "suggested_edit": "e", "rank": papers[0], "priority": priority,
            "paper_numbers": papers, "rationale": "why"}


def _record(conn, profile_id, candidates, ordered_flags):
    """One recorded round through the REAL persist path, returning its run id."""
    run_id = DB.create_analysis_run(
        conn, DB.get_or_create_prompt(conn, "synthetic analysis prompt", kind="analysis"),
        "m", "m", profile_id=profile_id, date_start=None, date_end=None,
        n_flags=len(ordered_flags), cost_usd=0.0)
    PA._record_consolidation(conn, candidates, ordered_flags, analysis_run_id=run_id)
    return run_id


def _second_round(conn, profile_id, ordered, ids):
    """A later round exercising every kind of write undo must reverse: a minted pattern, a
    merge into an earlier round's OPEN pattern, and a recurrence on its CLOSED one."""
    return _record(conn, profile_id, [
        _new("C", [4]),
        {"choice": "merge_into_open", "existing_pattern_id": ids["A"],
         "rank": 2, "paper_numbers": [5], "rationale": "same gap"},
        {"choice": "merge_into_closed", "existing_pattern_id": ids["B"],
         "rank": 3, "paper_numbers": [5], "rationale": "came back"},
    ], ordered)


def test_undo_restores_tables_byte_identical():
    with _world() as (conn, profile_id, ordered, ids):
        before = _dump(conn)
        pool_before = len(DB.get_flags(conn, exclude_attached=True))
        run2 = _second_round(conn, profile_id, ordered, ids)
        assert _dump(conn) != before, "round 2 wrote nothing -- the test is vacuous"

        manifest = DB.analysis_run_manifest(conn, run2)
        assert not manifest["blockers"], manifest["blockers"]
        assert {p["name"] for p in manifest["minted"]} == {"C"}
        assert {(r["name"], r["pmid"]) for r in manifest["foreign_attaches"]} == {
            ("A", "SYNUNDO5"), ("B", "SYNUNDO5")}
        assert {(r["name"], r["event"]) for r in manifest["foreign_events"]} == {
            ("A", "carried"), ("B", "recurred")}

        DB.delete_analysis_run(conn, run2)
        assert _dump(conn) == before, "undo did not restore the pattern layer exactly"
        assert len(DB.get_flags(conn, exclude_attached=True)) == pool_before
        assert DB.latest_analysis_run(conn)["id"] != run2, "run row survived"
        print(f"byte-identical restore: 4 tables, pool back to {pool_before} unattached")


def test_human_decision_blocks_undo():
    with _world() as (conn, profile_id, ordered, ids):
        run2 = _second_round(conn, profile_id, ordered, ids)
        c_id = conn.execute("SELECT id FROM patterns WHERE name='C'").fetchone()["id"]
        DB.add_pattern_event(conn, c_id, "rejected", note="human said no")
        manifest = DB.analysis_run_manifest(conn, run2)
        assert manifest["blockers"], "an unstamped human event on a minted pattern must block"
        try:
            DB.delete_analysis_run(conn, run2)
            raise AssertionError("delete_analysis_run ran despite a human decision")
        except ValueError as e:
            assert "rejected" in str(e)
        # And the human event on the EARLIER round's pattern B never blocked run 2's undo:
        # that is the whole reason blockers scan only the run's own minted patterns.
        print("blocked on the human event; earlier round's decisions did not interfere")


def test_hand_edited_wording_blocks_undo():
    with _world() as (conn, profile_id, ordered, ids):
        run2 = _second_round(conn, profile_id, ordered, ids)
        c_id = conn.execute("SELECT id FROM patterns WHERE name='C'").fetchone()["id"]
        DB.update_pattern_content(conn, c_id, name="C, but reworded by hand")
        manifest = DB.analysis_run_manifest(conn, run2)
        assert any("hand-edited" in b for b in manifest["blockers"]), manifest["blockers"]
        print("blocked on hand-edited wording")


def test_undo_peels_lifo():
    with _world() as (conn, profile_id, ordered, ids):
        after_round1 = _dump(conn)
        run2 = _second_round(conn, profile_id, ordered, ids)
        run3 = _record(conn, profile_id, [_new("D", [4], priority="hold")], ordered)
        assert DB.latest_analysis_run(conn)["id"] == run3
        DB.delete_analysis_run(conn, run3)
        assert DB.latest_analysis_run(conn)["id"] == run2, "peel order is not LIFO"
        DB.delete_analysis_run(conn, run2)
        assert _dump(conn) == after_round1, "two peels did not restore the round-1 state"
        print("peeled run3 then run2, back to round-1 state exactly")


CHECKS = [
    test_undo_restores_tables_byte_identical,
    test_human_decision_blocks_undo,
    test_hand_edited_wording_blocks_undo,
    test_undo_peels_lifo,
]
