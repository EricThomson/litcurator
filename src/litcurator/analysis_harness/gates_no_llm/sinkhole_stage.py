"""
sinkhole_stage.py -- unit tests for the review feed's accumulator button (free, no model).

attach_to_accumulator is a one-click permanent write from the review feed, so every status it
can return is exercised against a scratch database, plus the two properties the design leans
on: an attached paper leaves the unattached pool (the accumulator silences the discovery loop
for papers already diagnosed), and the attachment is invisible to undo_error_analysis (a
human write carries no run stamp).
"""

import contextlib
import os

from litcurator import db_interface as DB

from .. import machinery as H

NAME = "Test sinkhole"


@contextlib.contextmanager
def _world():
    path = H.scratch_db_path("sinkhole_stage")
    conn, scoring_run, profile_id = H.build_db(path, "synthetic profile")
    flag_intended = {}
    papers = [{"pmid": f"SYNSINK{i}", "title": f"Paper {i}", "abstract": "a", "journal": "J",
               "judge_score": 0.6, "rationale": "r", "user_score": 0.2, "note": "molecular",
               "intended": "X"} for i in (1, 2)]
    H.add_round_flags(conn, scoring_run, papers, flag_intended)
    # A third paper that is judged but NEVER flagged, for the no_flag path.
    H.add_round_flags(conn, scoring_run, [], flag_intended)
    DB.insert_articles(conn, [{"pubmed_id": "SYNSINK3", "title": "Paper 3", "abstract": "a",
                               "journal": "J", "pub_date": "2026-01-15", "epub_date": None,
                               "authors": [], "pub_types": [], "pages": None, "doi": None}])
    try:
        yield conn, profile_id
    finally:
        conn.close()
        if path.exists():
            os.unlink(path)


def test_first_click_creates_and_attaches():
    with _world() as (conn, _profile_id):
        pool_before = len(DB.get_flags(conn, exclude_attached=True))
        status, count = DB.attach_to_accumulator(conn, NAME, "SYNSINK1")
        assert (status, count) == ("attached", 1), (status, count)
        pattern = conn.execute("SELECT * FROM patterns WHERE name=?", (NAME,)).fetchone()
        assert pattern is not None and pattern["direction"] == "over"
        # Open status: the attached flag leaves the unattached pool, which is the point --
        # a diagnosed paper stops feeding the discovery loop.
        assert len(DB.get_flags(conn, exclude_attached=True)) == pool_before - 1
        print(f"created + attached; pool shrank by 1")


def test_second_click_is_idempotent_and_second_paper_counts():
    with _world() as (conn, _profile_id):
        DB.attach_to_accumulator(conn, NAME, "SYNSINK1")
        assert DB.attach_to_accumulator(conn, NAME, "SYNSINK1") == ("already", 1)
        assert DB.attach_to_accumulator(conn, NAME, "SYNSINK2") == ("attached", 2)
        rows = conn.execute("SELECT COUNT(*) FROM patterns WHERE name=?", (NAME,)).fetchone()[0]
        assert rows == 1, f"{rows} accumulator patterns minted, expected 1"
        print("idempotent; second paper counted; one pattern row")


def test_unflagged_paper_is_refused():
    with _world() as (conn, _profile_id):
        DB.attach_to_accumulator(conn, NAME, "SYNSINK1")
        status, count = DB.attach_to_accumulator(conn, NAME, "SYNSINK3")
        assert (status, count) == ("no_flag", 1), (status, count)
        print("unflagged paper refused; the button never invents a score")


def test_closed_accumulator_refuses():
    with _world() as (conn, profile_id):
        DB.attach_to_accumulator(conn, NAME, "SYNSINK1")
        pid = conn.execute("SELECT id FROM patterns WHERE name=?", (NAME,)).fetchone()["id"]
        DB.add_pattern_event(conn, pid, "incorporated", profile_id=profile_id)
        status, count = DB.attach_to_accumulator(conn, NAME, "SYNSINK2")
        assert (status, count) == ("closed", 1), (status, count)
        assert conn.execute("SELECT COUNT(*) FROM patterns WHERE name=?",
                            (NAME,)).fetchone()[0] == 1, "a duplicate accumulator was minted"
        print("closed accumulator refuses instead of resurrecting or duplicating")


def test_invisible_to_undo():
    with _world() as (conn, profile_id):
        DB.attach_to_accumulator(conn, NAME, "SYNSINK1")
        run_id = DB.create_analysis_run(
            conn, DB.get_or_create_prompt(conn, "p", kind="analysis"), "m", "m",
            profile_id=profile_id, date_start=None, date_end=None, n_flags=1, cost_usd=0.0)
        manifest = DB.analysis_run_manifest(conn, run_id)
        assert not manifest["minted"] and not manifest["foreign_attaches"], \
            "a human accumulator write leaked into a run's manifest"
        DB.delete_analysis_run(conn, run_id)
        assert conn.execute("SELECT COUNT(*) FROM pattern_flags").fetchone()[0] == 1, \
            "undo deleted a human attachment"
        print("human attachment survives an unrelated undo untouched")


CHECKS = [
    test_first_click_creates_and_attaches,
    test_second_click_is_idempotent_and_second_paper_counts,
    test_unflagged_paper_is_refused,
    test_closed_accumulator_refuses,
    test_invisible_to_undo,
]
