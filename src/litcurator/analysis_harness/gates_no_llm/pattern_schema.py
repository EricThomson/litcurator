"""
Validate the pattern-memory schema + helpers on a SCRATCH copy of the live DB.
Exercises the guarded migration (old patterns stub + consolidation_runs dropped,
new three tables created) and every helper. Read-only w.r.t. the live DB.

    python -m litcurator.analysis_harness.gates_no_llm.pattern_schema
"""
import shutil
import sqlite3
from pathlib import Path

from litcurator import config, db_interface

SCRATCH = Path(config.DATA_DIR) / "_scratch_pattern_test.db"


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
    ok = True

    conn = db_interface.get_connection(SCRATCH)   # triggers the guarded migration + new schema
    _wipe_patterns(conn)
    try:
        # 1. schema shape
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "patterns" in tables and "pattern_flags" in tables and "pattern_events" in tables
        assert "consolidation_runs" not in tables, "consolidation_runs should be dropped"
        pcols = [r[1] for r in conn.execute("PRAGMA table_info(patterns)").fetchall()]
        assert "rounds_seen" not in pcols, "old stub column should be gone"
        assert set(["id", "name", "direction", "description", "suggested_edit"]).issubset(pcols)
        print("schema OK:", sorted(tables & {"patterns", "pattern_flags", "pattern_events"}))
        print("patterns cols:", pcols)

        # CHECK constraints enforced
        for bad in [("INSERT INTO patterns (id,name,direction) VALUES ('x','n','bogus')", "direction"),
                    ("INSERT INTO pattern_events (pattern_id,event) VALUES ('x','bogus')", "event")]:
            try:
                conn.execute(bad[0]); conn.rollback(); ok = False
                print(f"  CHECK {bad[1]}: NOT enforced (BAD)")
            except sqlite3.IntegrityError:
                conn.rollback(); print(f"  CHECK {bad[1]} enforced: OK")

        # EVERY declared direction is actually accepted by the table. The vocabulary used to be
        # written out in five places (the CHECK, the clamp, the tool enum, the workbench
        # dropdown, the taste subset) and two of them failed silently when they drifted -- a
        # value missing from the clamp is rewritten to `under`, one missing from the enum is
        # never proposed. db_interface now owns the tuple and the CHECK is generated from it,
        # so this asserts the two cannot come apart: a value in DIRECTIONS that the table
        # rejects means the constraint was not regenerated, which on an EXISTING database is
        # the expected failure, because CREATE TABLE IF NOT EXISTS never alters a live table.
        # Adding a direction still needs a migration; this is what tells you so.
        for d in db_interface.DIRECTIONS:
            try:
                conn.execute("INSERT INTO patterns (id,name,direction) VALUES (?,?,?)",
                             (f"dirprobe_{d}", "probe", d))
                conn.rollback()
            except sqlite3.IntegrityError:
                conn.rollback(); ok = False
                print(f"  direction {d!r} is declared but the table REJECTS it (BAD -- the "
                      f"CHECK constraint predates it; needs a migration)")
        print(f"  all {len(db_interface.DIRECTIONS)} declared directions accepted: OK")

        # 2. fixtures: a run + evaluation + two flags on real articles
        pmids = [r[0] for r in conn.execute("SELECT pmid FROM articles LIMIT 2").fetchall()]
        pid = db_interface.get_or_create_profile(conn, "test profile content")
        run_id = db_interface.find_or_create_scoring_run(
            conn, "curation", "m", "benchmark", profile_id=pid, judge_prompt_hash="h",
            date_start="2025-01-01", date_end="2025-01-31")
        flag_ids = []
        for pm, your in zip(pmids, (0.9, 0.1)):
            db_interface.insert_evaluation(conn, pm, run_id, 0.4, rationale="r")
            ev = conn.execute("SELECT id FROM evaluations WHERE pmid=? AND run_id=?",
                              (pm, run_id)).fetchone()["id"]
            flag_ids.append(db_interface.insert_flag(conn, ev, your, note=f"note {pm}"))
        print("fixtures: flags", flag_ids)

        # 3. create_pattern -> pattern + pattern_flags + 'created' event
        pat = db_interface.create_pattern(
            conn, "invert neuroethology under-scored", "under",
            description="planaria/octopus buried", suggested_edit="amplify invertebrate behavior",
            flag_ids=flag_ids)
        links = conn.execute("SELECT COUNT(*) FROM pattern_flags WHERE pattern_id=?",
                             (pat,)).fetchone()[0]
        evs = db_interface.get_pattern_events(conn, pat)
        assert links == 2, links
        assert len(evs) == 1 and evs[0]["event"] == "created"
        print(f"create_pattern OK: id={pat[:8]} links={links} events={[e['event'] for e in evs]}")

        # 4. active list + provenance
        active = db_interface.get_active_patterns(conn)
        assert len(active) == 1 and active[0]["status"] == "created"
        assert active[0]["flag_count"] == 2
        prov = db_interface.get_pattern_provenance(conn, pat)
        assert len(prov) == 2 and {p["pmid"] for p in prov} == set(pmids)
        print(f"active OK: status={active[0]['status']} flag_count={active[0]['flag_count']}")
        print(f"provenance OK: papers={[p['title'][:30] for p in prov]}")

        # 5. lifecycle events: carry, then incorporate -> drops off active, closed pattern persists
        db_interface.add_pattern_event(conn, pat, "carried", note="not yet")
        assert db_interface.get_active_patterns(conn)[0]["status"] == "carried"
        db_interface.add_pattern_event(conn, pat, "incorporated", profile_id=pid)
        assert db_interface.get_active_patterns(conn) == [], "incorporated must leave active list"
        tomb = db_interface.get_patterns(conn, statuses=("incorporated", "rejected"))
        assert len(tomb) == 1 and tomb[0]["status"] == "incorporated"
        assert tomb[0]["status_profile_id"] == pid, "incorporated stamps the profile version"
        log = [e["event"] for e in db_interface.get_pattern_events(conn, pat)]
        assert log == ["created", "carried", "incorporated"], log
        print(f"lifecycle OK: full log = {log}, incorporated profile = {tomb[0]['status_profile_id'][:8]}")

        # 6. update content in place (fate untouched)
        db_interface.update_pattern_content(conn, pat, name="renamed")
        assert db_interface.get_pattern(conn, pat)["name"] == "renamed"
        assert [e["event"] for e in db_interface.get_pattern_events(conn, pat)] == log
        print("update_pattern_content OK (fate log unchanged)")

        # 7. full chain: profile -> pattern -> flags -> papers
        chain = conn.execute("""
            SELECT p.name, a.title FROM pattern_events ev
            JOIN patterns p ON p.id = ev.pattern_id
            JOIN pattern_flags pf ON pf.pattern_id = p.id
            JOIN flags f ON f.id = pf.flag_id
            JOIN articles a ON a.pmid = f.pmid
            WHERE ev.event='incorporated' AND ev.profile_id = ?
        """, (pid,)).fetchall()
        assert len(chain) == 2
        print(f"chain OK: profile {pid[:8]} -> pattern '{chain[0]['name']}' -> {len(chain)} papers")

        # 8. 'recurred' is a NON-decision annotation: accepted by the CHECK, but it does
        # NOT become status (this pattern stays 'incorporated'); recurred_count tracks it.
        # Deliberately overlaps record_stage, which asserts the same semantics end to end
        # through the real pipeline. Kept because that one covers a REJECTED pattern and
        # this one an INCORPORATED pattern -- a distinction the CHECK constraint cannot make.
        db_interface.add_pattern_event(conn, pat, "recurred", note="taste came back")
        again = db_interface.get_patterns(conn, statuses=("incorporated", "rejected"))
        assert len(again) == 1 and again[0]["status"] == "incorporated", again
        assert again[0]["recurred_count"] == 1, again[0]
        assert any(x["id"] == pat for x in db_interface.get_closed_recurrences(conn))
        print("recurred OK: accepted, status stays incorporated, surfaces as a closed pattern alert")

    finally:
        conn.close()
        SCRATCH.unlink(missing_ok=True)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
