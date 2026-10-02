"""
analysis_workbench_actions.py -- unit tests for the analysis workbench (free, no model).

The workbench edits the analysis prompt one section per tab through
analysis_prompt_interface.Section, and its buttons are the shared artifact_editor's, which until
2026-10-02 had no test at all. Everything runs against a scratch prompt file and a scratch
database, and that is ASSERTED before any check writes: the 2026-09-22 incident was a free gate
overwriting a live prompt while reporting green.

The promises checked: a tab's Set Active changes only its own section; a refusal writes nothing;
a saved version is a whole runnable prompt that loads back per tab; the two tabs' autosaves do
not collide; the shared editor's buttons route each tab to its own section; and the preview runs
the real error_analysis on the drafts on screen, never recording them.
"""

import contextlib
import shutil

from litcurator import analysis_prompt_interface as API
from litcurator import db_interface as DB, error_analysis as PA
from litcurator.apps import artifact_editor as AE

from .. import machinery as M

CLUSTER = "Cluster the flags. ORIGINAL cluster wording."
CONSOLIDATE = "Decide each candidate. ORIGINAL consolidate wording."


class _Ctx:
    """What dash.ctx gives a callback: the id of the component that fired."""

    def __init__(self, triggered_id):
        self.triggered_id = triggered_id


@contextlib.contextmanager
def _world():
    """A scratch active prompt (both sections ORIGINAL) and a scratch database, with every
    module-level path the Section methods read pointed at them, restored on the way out."""
    db_path = M.scratch_db_path("analysis_workbench")
    prompt_dir = db_path.with_suffix("")
    shutil.rmtree(prompt_dir, ignore_errors=True)
    prompt_dir.mkdir(parents=True)
    saved = (API.ANALYSIS_PROMPT_PATH, API.VERSIONS_DIR, DB.LITCURATOR_DB, AE.ctx)
    live_before = API.read_active_or_empty()
    conn = None
    try:
        scratch_conn, _run_id, _profile_id = M.build_db(db_path, "synthetic profile")
        scratch_conn.close()
        API.ANALYSIS_PROMPT_PATH = prompt_dir / "analysis_prompt.md"
        API.VERSIONS_DIR = prompt_dir / "versions"
        DB.LITCURATOR_DB = db_path
        import litcurator.apps.analysis_workbench as aw
        # THE GUARD: every tab must resolve to a file inside the scratch folder before anything
        # writes. If a future change routes a tab elsewhere, this fails before the damage.
        for artifact in aw.ARTIFACTS:
            path = aw._interface(artifact).active_path()
            assert prompt_dir in path.parents, f"{artifact!r} would write {path}, not scratch"
        API.set_active(API.compose(CLUSTER, CONSOLIDATE))
        conn = DB.get_connection(db_path)
        yield aw, conn
    finally:
        if conn is not None:
            conn.close()
        API.ANALYSIS_PROMPT_PATH, API.VERSIONS_DIR, DB.LITCURATOR_DB, AE.ctx = saved
        shutil.rmtree(prompt_dir, ignore_errors=True)
        db_path.unlink(missing_ok=True)
        assert API.read_active_or_empty() == live_before, "the LIVE analysis prompt changed"


def _active():
    return API.load_active_sections()


def test_a_tab_sets_only_its_own_section():
    with _world() as (aw, _conn):
        aw._interface("cluster").set_active("NEW cluster wording")
        assert _active() == ("NEW cluster wording", CONSOLIDATE), _active()
        aw._interface("consolidate").set_active("NEW consolidate wording")
        assert _active() == ("NEW cluster wording", "NEW consolidate wording"), _active()
        print("each tab changed only its own section")


def test_refusals_write_nothing():
    with _world() as (aw, _conn):
        before = API.read_active_or_empty()
        for bad in ("", "   ", f"pasted the whole file\n{API.CONSOLIDATE_MARKER}\noops"):
            try:
                aw._interface("cluster").set_active(bad)
                raise AssertionError(f"accepted {bad!r}")
            except ValueError:
                pass
        assert API.read_active_or_empty() == before, "a refused set_active changed the file"
        print("empty and marker-carrying tabs refused; the file did not move")


def test_a_saved_version_is_whole_and_loads_back_per_tab():
    with _world() as (aw, _conn):
        path = aw._interface("cluster").save_version("DRAFT cluster wording")
        assert API.split(path.read_text(encoding="utf-8")) == ("DRAFT cluster wording",
                                                                CONSOLIDATE)
        assert aw._interface("cluster").read_version(path) == "DRAFT cluster wording",             "loading a version filled the cluster tab with more than its own section"
        assert aw._interface("consolidate").read_version(path) == CONSOLIDATE,             "loading a version filled the consolidate tab with more than its own section"
        assert _active() == (CLUSTER, CONSOLIDATE), "saving a version changed the active file"
        print("a version is a whole prompt; each tab loads its own section of it")


def test_the_two_autosaves_do_not_collide():
    with _world() as (aw, _conn):
        assert aw._interface("cluster").save_autosave("cluster draft")
        assert aw._interface("consolidate").save_autosave("consolidate draft")
        assert aw._interface("cluster").load_autosave() == "cluster draft",             "the consolidate tab's autosave overwrote the cluster tab's"
        assert aw._interface("consolidate").load_autosave() == "consolidate draft",             "the tabs' autosaves collided"
        print("each tab restores its own autosave")


def test_the_shared_buttons_route_each_tab():
    """Drives the REAL artifact_editor callbacks, which had no test before this gate. The
    consolidate tab is the one clicked, so a callback that ignored the trigger and used the
    first registered artifact would hit cluster and fail."""
    with _world() as (aw, _conn):
        target = AE.aid("x", "consolidate")
        AE.ctx = _Ctx(target)
        msg, _open, color = AE.cb_set_active(1, "SET through the button")
        assert color == "success", msg
        assert _active() == (CLUSTER, "SET through the button"), _active()
        AE.ctx = _Ctx(target)
        msg, _open, color = AE.cb_set_active(1, "")
        assert color == "danger" and _active()[1] == "SET through the button", msg
        AE.ctx = _Ctx(target)
        AE.cb_save_version(1, "SAVED through the button")
        saved = API.latest_version()
        AE.ctx = _Ctx(target)
        text, _msg, _open, color = AE.cb_load_version(1, str(saved))
        assert (text, color) == ("SAVED through the button", "success"), (text, color)
        print("Set Active, a refusal, Save version and Load all routed to the clicked tab")


def test_the_preview_runs_the_real_pipeline_on_the_drafts():
    with _world() as (aw, _conn):
        calls = []
        real = aw.error_analysis.suggest_edits
        aw.error_analysis.suggest_edits = lambda **kw: calls.append(kw)   # returns None
        try:
            aw.cb_preview(1, "draft CLUSTER", "draft CONSOLIDATE", "2025-01-01", "2025-01-15")
            aw.cb_preview(1, f"bad\n{API.CLUSTER_MARKER}", "draft CONSOLIDATE", None, None)
        finally:
            aw.error_analysis.suggest_edits = real
        assert len(calls) == 1, f"{len(calls)} runs; the marker-carrying draft must not run"
        kw = calls[0]
        assert kw["analysis_prompt_text"] == API.compose("draft CLUSTER", "draft CONSOLIDATE")
        assert kw["persist"] is False and kw["best_of"] == aw.PREVIEW_ROUNDS, kw
        assert (kw["start"], kw["end"]) == ("2025-01-01", "2025-01-15"), kw
        print("the preview sends both tabs' drafts to suggest_edits as a dry run")


def test_recording_a_draft_is_refused():
    try:
        PA.suggest_edits(persist=True, analysis_prompt_text="a draft")
        raise AssertionError("suggest_edits recorded a round from a draft prompt")
    except ValueError:
        print("suggest_edits refuses to record a round run on a draft")


CHECKS = [
    test_a_tab_sets_only_its_own_section,
    test_refusals_write_nothing,
    test_a_saved_version_is_whole_and_loads_back_per_tab,
    test_the_two_autosaves_do_not_collide,
    test_the_shared_buttons_route_each_tab,
    test_the_preview_runs_the_real_pipeline_on_the_drafts,
    test_recording_a_draft_is_refused,
]
