"""
sinkhole_stage.py -- unit tests for the Levels Bucket (free, no model).

The bucket's buttons write straight to the database from the review feed, so every path runs
against a scratch database. Adding saves the score and note straight into the bucket, and the
paper is never an ordinary flag in error_analysis's pool, not even if the add fails halfway.
Re-saving a bucketed paper keeps it in. Taking it out makes it an ordinary flag again and
loses nothing. The writes are invisible to undo_error_analysis. And the bucket stays off
error_analysis's list of existing patterns, so only the user ever puts papers in it.
"""

import contextlib
import os

from litcurator import config, db_interface as DB, error_analysis as PA

from .. import machinery as H


@contextlib.contextmanager
def _world():
    """SYNSINK1 and SYNSINK2 are judged AND flagged; SYNSINK3 is judged and never flagged."""
    path = H.scratch_db_path("sinkhole_stage")
    conn, scoring_run, profile_id = H.build_db(path, "synthetic profile")
    papers = [{"pmid": f"SYNSINK{i}", "title": f"Paper {i}", "abstract": "a", "journal": "J",
               "judge_score": 0.6, "rationale": "r", "user_score": 0.2, "note": "molecular",
               "intended": "X"} for i in (1, 2)]
    H.add_round_flags(conn, scoring_run, papers, {})
    DB.insert_articles(conn, [{"pubmed_id": "SYNSINK3", "title": "Paper 3", "abstract": "a",
                               "journal": "J", "pub_date": "2026-01-15", "epub_date": None,
                               "authors": [], "pub_types": [], "pages": None, "doi": None}])
    DB.insert_evaluation(conn, "SYNSINK3", scoring_run, 0.3, rationale="r")
    try:
        yield conn, profile_id
    finally:
        conn.close()
        if path.exists():
            os.unlink(path)


def _eid(conn, pmid):
    return conn.execute("SELECT id FROM evaluations WHERE pmid = ?", (pmid,)).fetchone()["id"]


def _pool(conn):
    """The papers error_analysis would cluster."""
    return {f["pmid"] for f in DB.get_flags(conn, exclude_attached=True)}


def _n_flags(conn):
    return conn.execute("SELECT COUNT(*) FROM flags").fetchone()[0]


def test_add_goes_straight_into_the_bucket():
    with _world() as (conn, _profile_id):
        flag_id, count = DB.add_to_levels_bucket(conn, _eid(conn, "SYNSINK3"), 0.8, "systems")
        assert count == 1 and DB.levels_bucket_pmids(conn) == {"SYNSINK3"}, \
            f"count {count}, bucket holds {DB.levels_bucket_pmids(conn)}"
        flag = DB.get_flag(conn, flag_id)
        assert (flag["user_score"], flag["note"]) == (0.8, "systems"), flag
        # The flag exists and is the paper's newest, so the ONLY way it stays out of the pool
        # is being in the bucket: a never-flagged paper went in without passing through.
        assert "SYNSINK3" not in _pool(conn), "a bucketed paper is in error_analysis's pool"
        print("never-flagged paper went straight in with its score and note")


def test_a_failed_add_leaves_no_ordinary_flag():
    """The flag and its bucket link commit together. If linking fails, the flag must not
    survive on its own, or it would sit in error_analysis's pool as an ordinary flag."""
    with _world() as (conn, _profile_id):
        DB.add_to_levels_bucket(conn, _eid(conn, "SYNSINK1"), 0.3, "creates the bucket")
        before = _n_flags(conn)
        real = DB.attach_flags_to_pattern

        def boom(*_args, **_kwargs):
            raise RuntimeError("simulated failure while linking")

        DB.attach_flags_to_pattern = boom
        try:
            DB.add_to_levels_bucket(conn, _eid(conn, "SYNSINK3"), 0.8, "should vanish")
            raise AssertionError("the simulated failure never fired")
        except RuntimeError:
            conn.rollback()
        finally:
            DB.attach_flags_to_pattern = real
        assert _n_flags(conn) == before, "a failed add left an ordinary flag behind"
        print("a failed add left nothing behind")


def test_adding_again_saves_a_newer_version():
    with _world() as (conn, _profile_id):
        DB.add_to_levels_bucket(conn, _eid(conn, "SYNSINK3"), 0.8, "first")
        _flag_id, count = DB.add_to_levels_bucket(conn, _eid(conn, "SYNSINK3"), 0.7, "second")
        assert count == 1, f"one paper counted {count} times"
        assert DB.get_latest_flag(conn, "SYNSINK3")["note"] == "second"
        assert conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0] == 1, \
            "a second bucket was minted"
        print("re-adding saved a newer version; one paper, one bucket")


def test_resaving_a_bucketed_paper_keeps_it_in():
    """SYNSINK3 is the control: it was never flagged, so an ordinary save must put it IN the
    pool. Without it, a save_flag that bucketed everything would pass."""
    with _world() as (conn, _profile_id):
        DB.add_to_levels_bucket(conn, _eid(conn, "SYNSINK1"), 0.3, "levels")
        _flag_id, in_bucket = DB.save_flag(conn, _eid(conn, "SYNSINK1"), 0.35, "edited later")
        assert in_bucket and "SYNSINK1" not in _pool(conn), \
            "re-saving leaked a bucketed paper back into error_analysis's pool"
        assert DB.get_latest_flag(conn, "SYNSINK1")["note"] == "edited later"
        _flag_id, in_bucket = DB.save_flag(conn, _eid(conn, "SYNSINK3"), 0.9, "ordinary")
        assert not in_bucket and "SYNSINK3" in _pool(conn), \
            "control failed: an ordinary save did not land in the pool"
        print("re-saving a bucketed paper stayed in the bucket; an ordinary save did not")


def test_take_out_restores_an_ordinary_flag():
    with _world() as (conn, _profile_id):
        DB.add_to_levels_bucket(conn, _eid(conn, "SYNSINK1"), 0.3, "mis-click")
        assert "SYNSINK1" not in _pool(conn), "precondition: a bucketed paper is out of the pool"
        before = _n_flags(conn)
        assert DB.remove_from_levels_bucket(conn, "SYNSINK1") == 0
        assert "SYNSINK1" in _pool(conn), "taken out, but still hidden from error_analysis"
        assert _n_flags(conn) == before, "taking a paper out deleted a score or note"
        assert DB.get_latest_flag(conn, "SYNSINK1")["note"] == "mis-click"
        print("taken out: back in the pool, every score and note kept")


def test_invisible_to_undo():
    with _world() as (conn, profile_id):
        DB.add_to_levels_bucket(conn, _eid(conn, "SYNSINK1"), 0.3, "levels")
        run_id = DB.create_analysis_run(
            conn, DB.get_or_create_prompt(conn, "p", kind="analysis"), "m", "m",
            profile_id=profile_id, date_start=None, date_end=None, n_flags=1, cost_usd=0.0)
        manifest = DB.analysis_run_manifest(conn, run_id)
        assert not manifest["minted"] and not manifest["foreign_attaches"], \
            "a bucket write leaked into a run's manifest"
        DB.delete_analysis_run(conn, run_id)
        assert DB.levels_bucket_pmids(conn) == {"SYNSINK1"}, "undo emptied the bucket"
        print("bucket survives an unrelated undo untouched")


def test_levels_bucket_is_not_a_pattern():
    """The ordinary pattern beside it is the control: without one, an empty list would pass
    this check for the wrong reason."""
    with _world() as (conn, _profile_id):
        DB.add_to_levels_bucket(conn, _eid(conn, "SYNSINK1"), 0.3, "levels")
        flag2 = conn.execute("SELECT id FROM flags WHERE pmid = 'SYNSINK2'").fetchone()["id"]
        DB.create_pattern(conn, name="Ordinary pattern", direction="under",
                          description="control", flag_ids=[flag2], note="control")
        block, active, _held, _closed = PA.build_memory_block(conn)
        assert "Ordinary pattern" in block, "control failed: the ordinary pattern is missing"
        assert config.LEVELS_BUCKET_NAME not in block, "the bucket leaked into error_analysis"
        assert [p["name"] for p in active] == ["Ordinary pattern"], [p["name"] for p in active]
        bucket = DB.get_levels_bucket(conn)
        assert bucket is not None and bucket["paper_count"] == 1, bucket
        print("bucket stays off error_analysis's list; the ordinary pattern beside it is shown")


CHECKS = [
    test_add_goes_straight_into_the_bucket,
    test_a_failed_add_leaves_no_ordinary_flag,
    test_adding_again_saves_a_newer_version,
    test_resaving_a_bucketed_paper_keeps_it_in,
    test_take_out_restores_an_ordinary_flag,
    test_invisible_to_undo,
    test_levels_bucket_is_not_a_pattern,
]
