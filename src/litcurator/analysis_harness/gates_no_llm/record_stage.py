"""
Test the suggester's RECORD stage WITHOUT the LLM: provenance mapping (paper number N
-> the right flag id via _format_papers ordering) and every _record_consolidation
choice (new / merge_into_open / merge_into_closed / hold + recovery), plus
the unattached-flag exclusion and the recurred-event semantics.

The recurred assertions here go through the real pipeline against a REJECTED pattern;
pattern_schema checks the same semantics at schema level against an INCORPORATED one.
The overlap is deliberate -- the two cover different closed states.

    python -m litcurator.analysis_harness.gates_no_llm.record_stage
"""
import shutil
from pathlib import Path

from litcurator import config, db_interface, profile_analysis

SCRATCH = Path(config.DATA_DIR) / "_scratch_sugg.db"


def _wipe_model_output(conn):
    """Clean model-output slate so the test controls every flag (the live DB now has
    real flags that would otherwise interleave). Children before parents (FK on).
    Ground truth (articles / human_labels / profiles / prompts) is left intact."""
    for t in ("pattern_flags", "pattern_events", "patterns", "flags",
              "evaluations", "scoring_runs"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()


def _make_flags(conn):
    """Six flags with distinct deltas so _format_papers numbering is deterministic:
    neg (delta asc) then pos (delta desc). paper i maps to flag F{i}."""
    pmids = [r[0] for r in conn.execute("SELECT pmid FROM articles LIMIT 6").fetchall()]
    pid = db_interface.get_or_create_profile(conn, "sugg test")
    run_id = db_interface.find_or_create_scoring_run(
        conn, "curation", "m", "benchmark", profile_id=pid, judge_prompt_hash="h")
    # (judge, your) -> delta: F1..F3 negative (asc), F4..F6 positive (desc)
    specs = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.2, 0.9), (0.3, 0.8), (0.4, 0.75)]
    flag_id = {}
    for i, (judge, your) in enumerate(specs, start=1):
        pm = pmids[i - 1]
        db_interface.insert_evaluation(conn, pm, run_id, judge, rationale="r")
        ev = conn.execute("SELECT id FROM evaluations WHERE pmid=? AND run_id=?",
                          (pm, run_id)).fetchone()["id"]
        flag_id[f"F{i}"] = db_interface.insert_flag(conn, ev, your, note=f"flag F{i}")
    return flag_id


def main():
    shutil.copy(config.LITCURATOR_DB, SCRATCH)
    conn = db_interface.get_connection(SCRATCH)
    _wipe_model_output(conn)
    flag_id = _make_flags(conn)

    # --- 1. numbering + provenance mapping (paper N -> flag id) ---
    flags = db_interface.get_flags(conn)
    papers_block, ordered = profile_analysis._format_papers(flags)
    order = [next(l for l, fid in flag_id.items() if fid == f["id"]) for f in ordered]
    assert order == ["F1", "F2", "F3", "F4", "F5", "F6"], order
    assert all(f"[{i}]" in papers_block for i in range(1, 7))
    print("paper numbering deterministic:", order)

    # --- 2. a `new` candidate: attaches the right flags + note rides the created event ---
    s = profile_analysis._record_consolidation(conn, [{
        "choice": "new", "name": "NewTaste", "direction": "under",
        "description": "d", "suggested_edit": "e", "priority": "act_now",
        "paper_numbers": [1, 3], "rationale": "real gap"}], ordered)
    newp = s["new"][0]
    attached = {r[0] for r in conn.execute(
        "SELECT flag_id FROM pattern_flags WHERE pattern_id=?", (newp["id"],)).fetchall()}
    assert attached == {flag_id["F1"], flag_id["F3"]}, attached
    note = conn.execute("SELECT note FROM pattern_events WHERE pattern_id=? AND event='created'",
                        (newp["id"],)).fetchone()[0]
    assert note == "act_now: real gap", note
    print("new: links F1,F3 and created-event note rides:", repr(note))

    # bad / out-of-range paper numbers are ignored, not crashing
    s_bad = profile_analysis._record_consolidation(conn, [{
        "choice": "new", "name": "x", "direction": "over",
        "paper_numbers": [99, "bad", 2], "rationale": "r"}], ordered)
    linked_bad = {r[0] for r in conn.execute(
        "SELECT flag_id FROM pattern_flags WHERE pattern_id=?", (s_bad["new"][0]["id"],)).fetchall()}
    assert linked_bad == {flag_id["F2"]}, linked_bad
    print("robust to bad paper_numbers: kept only F2")

    # --- 3. merge / recurs / noise / recovery in one batch ---
    p_open = db_interface.create_pattern(conn, "open", "under", flag_ids=[flag_id["F1"]])
    p_tomb = db_interface.create_pattern(conn, "tomb", "over", flag_ids=[flag_id["F4"]])
    db_interface.add_pattern_event(conn, p_tomb, "rejected", note="not a gap")

    s2 = profile_analysis._record_consolidation(conn, [
        {"choice": "merge_into_open", "existing_pattern_id": p_open,
         "paper_numbers": [2, 3], "rationale": "same taste"},           # +F2,F3
        {"choice": "merge_into_closed", "existing_pattern_id": p_tomb,
         "paper_numbers": [5], "rationale": "came back"},               # +F5 (new)
        {"choice": "hold", "name": "quirk", "paper_numbers": [], "rationale": "one-off"},
        {"choice": "merge_into_open", "existing_pattern_id": "deadbeefdeadbeef",
         "name": "Recovered", "direction": "over", "paper_numbers": [6], "rationale": "bad id"},
    ], ordered)
    assert len(s2["merged"]) == 1 and s2["merged"][0]["added"] == 2, s2["merged"]
    assert len(s2["recurred"]) == 1 and s2["recurred"][0]["added"] == 1, s2["recurred"]
    assert len(s2["held"]) == 1, s2["held"]
    assert any(c.get("recovered") for c in s2["new"]), s2["new"]

    def pat(pid):
        return [x for x in db_interface.get_patterns(conn) if x["id"] == pid][0]
    po, pt = pat(p_open), pat(p_tomb)
    assert po["status"] == "carried" and po["carried_count"] == 1 and po["flag_count"] == 3, po
    assert pt["status"] == "rejected" and pt["recurred_count"] == 1, pt
    assert any(x["id"] == p_tomb for x in db_interface.get_closed_recurrences(conn))
    print("merge -> carried (+2), recurs -> closed pattern stays rejected (+1) and alerts, held, "
          "hallucinated id recovered as new")

    # --- 3b. LOSSLESS recovery of malformed candidates (regression: the real Sonnet run
    # emitted merge_into_open with a DIRECTION in the id field and no name, which the
    # first version silently skipped -- losing a judge-not-applying signal) ---
    before = len(db_interface.get_patterns(conn))
    s_bad2 = profile_analysis._record_consolidation(conn, [
        # exactly the malformed shape seen in the live dry run: bogus id, no name
        {"choice": "merge_into_open", "existing_pattern_id": "judge-not-applying",
         "direction": "over", "priority": "act_now", "paper_numbers": [1],
         "rationale": "news-and-views hard cap not enforced"},
        # unrecognized choice must be recorded, not dropped
        {"choice": "banana", "name": "Weird One", "direction": "under",
         "paper_numbers": [2], "rationale": "model went off-script"},
        # genuinely empty candidate is the ONLY thing allowed to be skipped
        {"choice": "merge_into_open", "existing_pattern_id": "nope", "paper_numbers": []},
    ], ordered)
    assert len(s_bad2["new"]) == 2, s_bad2
    assert all(c.get("recovered") for c in s_bad2["new"]), s_bad2["new"]
    assert len(s_bad2["skipped"]) == 1, s_bad2["skipped"]
    assert len(db_interface.get_patterns(conn)) == before + 2
    # the unnamed one got a name from its rationale rather than being lost
    names = {c["name"] for c in s_bad2["new"]}
    assert any("news-and-views" in (nm or "") for nm in names), names
    print("lossless: malformed merge + unknown choice recovered as new "
          f"(names: {sorted(n[:32] for n in names)}); only the empty candidate skipped")

    # --- 4. idempotency: re-running the same merge adds nothing ---
    s3 = profile_analysis._record_consolidation(conn, [{
        "choice": "merge_into_open", "existing_pattern_id": p_open,
        "paper_numbers": [2, 3], "rationale": "again"}], ordered)
    assert len(s3["skipped"]) == 1 and pat(p_open)["carried_count"] == 1
    print("idempotent: re-merge adds 0 flags, no second carried event")

    # --- 5. exclude_attached drops papers whose latest flag is patterned ---
    all_pm = {f["pmid"] for f in db_interface.get_flags(conn)}
    unattached_pm = {f["pmid"] for f in db_interface.get_flags(conn, exclude_attached=True)}
    f1_pm = conn.execute("SELECT pmid FROM flags WHERE id=?", (flag_id["F1"],)).fetchone()[0]
    assert f1_pm in all_pm and f1_pm not in unattached_pm, (f1_pm, unattached_pm)
    print(f"exclude_attached: {len(all_pm)} all -> {len(unattached_pm)} unattached (F1's paper dropped)")

    # --- 6. attach_flags_to_pattern dedups and returns the new count ---
    assert db_interface.attach_flags_to_pattern(conn, p_open, [flag_id["F2"]]) == 0
    print("attach_flags_to_pattern dedups on the PK -> 0")

    conn.close()
    SCRATCH.unlink(missing_ok=True)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
