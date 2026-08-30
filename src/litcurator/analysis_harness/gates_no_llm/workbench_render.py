"""
Import-and-render smoke test for the reworked profile_workbench, against a SCRATCH
copy of the live DB seeded with one pattern. Catches construction errors (bad ids,
missing components like dbc.Select, the provenance drill-down) without a browser and
without touching the live DB. Dash callback *behavior* still needs a manual click-through.

    python -m litcurator.analysis_harness.gates_no_llm.workbench_render
"""
import importlib
import shutil
from pathlib import Path

from litcurator import config, db_interface

SCRATCH = Path(config.DATA_DIR) / "_scratch_wb.db"


def _wipe_patterns(conn):
    """Empty the pattern tables in the SCRATCH copy. These gates copy the live database for its
    real articles and then assert on counts they create themselves, so they only worked while
    the live pattern tables happened to be empty. The first real profile_analysis run put 15
    patterns in them (2026-08-29) and both gates went red on their own fixtures. Predicted in
    CLAUDE.md: "record_stage, pattern_schema and workbench_render must be made hermetic first."
    record_stage already wiped; these two did not."""
    for t in ("pattern_flags", "pattern_events", "patterns"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()


def main():
    shutil.copy(config.LITCURATOR_DB, SCRATCH)
    db_interface.LITCURATOR_DB = SCRATCH   # reroute get_connection's default

    # seed a flag + pattern so the workbench renders a real card
    conn = db_interface.get_connection(SCRATCH)
    _wipe_patterns(conn)
    pmid = conn.execute("SELECT pmid FROM articles LIMIT 1").fetchone()[0]
    pid = db_interface.get_or_create_profile(conn, "wb test profile")
    run_id = db_interface.find_or_create_scoring_run(
        conn, "curation", "m", "benchmark", profile_id=pid, judge_prompt_hash="h")
    db_interface.insert_evaluation(conn, pmid, run_id, 0.3, rationale="r")
    ev = conn.execute("SELECT id FROM evaluations WHERE pmid=? AND run_id=?",
                      (pmid, run_id)).fetchone()["id"]
    fid = db_interface.insert_flag(conn, ev, 0.85, note="clearly interesting")
    pat = db_interface.create_pattern(
        conn, "invert neuroethology under-scored", "under",
        description="buried invertebrate behavior", suggested_edit="amplify it",
        flag_ids=[fid])
    conn.close()

    # importing builds app.layout via _initial_patterns() (reads SCRATCH)
    #
    # The reload is what keeps that true. `import` is a no-op once the module is in
    # sys.modules, and workbench-actions now imports it first, so this gate was asserting a
    # layout built earlier against a DIFFERENT database -- the construction it exists to check
    # never ran, and it passed by replay. Reloading re-executes the module body, which is the
    # whole point of the check, and makes the gate independent of what ran before it rather
    # than silently dependent on gate order.
    import litcurator.apps.profile_workbench as wb
    importlib.reload(wb)
    assert wb.app.layout is not None
    print("layout built OK; app.title =", wb.app.title)

    # render a card and confirm the pattern-matched ids are present
    conn = db_interface.get_connection(SCRATCH)
    try:
        children = wb._render_patterns(conn)
        assert len(children) == 1, len(children)
        # walk the component tree collecting dict ids
        ids = _collect_ids(children)
        for t in ["pat-name", "pat-dir", "pat-desc", "pat-sugg", "pat-reject-note",
                  "pat-save", "pat-discuss", "pat-hold", "pat-incorporate", "pat-reject",
                  "pat-card"]:
            assert any(i.get("type") == t and i.get("pid") == pat for i in ids), f"missing {t}"
        print(f"pattern card OK: {len([i for i in ids if i.get('pid')==pat])} pattern-matched ids for pid {pat[:8]}")
    finally:
        conn.close()

    # _state_value helper
    states = [{"id": {"type": "pat-name", "pid": "A"}, "value": "alpha"},
              {"id": {"type": "pat-name", "pid": "B"}, "value": "beta"}]
    assert wb._state_value(states, "B") == "beta"
    assert wb._state_value(states, "Z") is None
    print("_state_value OK")

    SCRATCH.unlink(missing_ok=True)
    print("\nALL CHECKS PASSED")


def _collect_ids(node, out=None):
    out = [] if out is None else out
    if isinstance(node, (list, tuple)):     # _render_patterns returns a list of cards
        for c in node:
            _collect_ids(c, out)
        return out
    comp_id = getattr(node, "id", None)
    if isinstance(comp_id, dict):
        out.append(comp_id)
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for c in children:
            _collect_ids(c, out)
    elif children is not None:
        _collect_ids(children, out)
    return out


if __name__ == "__main__":
    main()
