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
import re
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
    """Six flags with distinct |delta| so _format_papers numbering is deterministic.

    The deltas alternate in sign as |delta| falls, so the render order INTERLEAVES the
    fixture labels (F1, F4, F2, F5, F3, F6) rather than running F1..F6. That is deliberate:
    a lookup bug that ignored the paper number and took a fixed position would still be
    right by accident if the labels came out in order."""
    pmids = [r[0] for r in conn.execute("SELECT pmid FROM articles LIMIT 6").fetchall()]
    pid = db_interface.get_or_create_profile(conn, "sugg test")
    run_id = db_interface.find_or_create_scoring_run(
        conn, "curation", "m", "benchmark", profile_id=pid, judge_prompt_hash="h")
    # (judge, your) -> delta: F1 -0.80, F2 -0.60, F3 -0.40, F4 +0.70, F5 +0.50, F6 +0.35
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

    # --- 1a. THE PROVENANCE CONTRACT, asserted independently of the sort ---
    # What must hold is that the delta printed on the "[N]" line belongs to ordered[N-1]:
    # that is the whole basis of paper N -> flag id -> pattern_flags. Pinning a literal
    # order instead would make a DELIBERATE re-sort look exactly like a broken mapping,
    # which is what happened when the magnitude sections were removed.
    flags = db_interface.get_flags(conn)
    papers_block, ordered = profile_analysis._format_papers(flags)
    printed = dict(re.findall(r"\[(\d+)\] delta ([-+]\d\.\d\d)", papers_block))
    assert len(printed) == len(ordered) == 6, (len(printed), len(ordered))
    for n, f in enumerate(ordered, start=1):
        assert printed[str(n)] == f"{f['delta']:+.2f}", (n, printed[str(n)], f["delta"])
    print(f"provenance contract: [N] delta line matches ordered[N-1] for all {len(ordered)}")

    # --- 1b. and the documented order: strongest disagreement first ---
    # Separate assertion on purpose. If this one alone goes red, the sort changed; if 1a
    # goes red, the mapping broke. Those are different bugs and should not share a check.
    magnitudes = [abs(f["delta"]) for f in ordered]
    assert magnitudes == sorted(magnitudes, reverse=True), magnitudes
    label = {fid: l for l, fid in flag_id.items()}
    order = [label[f["id"]] for f in ordered]
    assert order == ["F1", "F4", "F2", "F5", "F3", "F6"], order
    print("order is |delta| descending, signs interleaved:", order)

    # --- 2. a `new` candidate: attaches the right flags + note rides the created event ---
    s = profile_analysis._record_consolidation(conn, [{
        "choice": "new", "name": "NewTaste", "direction": "under",
        "description": "d", "suggested_edit": "e", "priority": "act_now",
        # papers 2 and 4 -> F4 and F5. Both are MIDDLE positions: never the first or last
        # paper, so a lookup taking ordered[0] or ordered[-1] cannot pass by accident.
        "paper_numbers": [2, 4], "rationale": "real gap"}], ordered)
    newp = s["new"][0]
    attached = {r[0] for r in conn.execute(
        "SELECT flag_id FROM pattern_flags WHERE pattern_id=?", (newp["id"],)).fetchall()}
    assert attached == {flag_id["F4"], flag_id["F5"]}, attached
    note = conn.execute("SELECT note FROM pattern_events WHERE pattern_id=? AND event='created'",
                        (newp["id"],)).fetchone()[0]
    assert note == "act_now: real gap", note
    print("new: links F4,F5 and created-event note rides:", repr(note))

    # bad / out-of-range paper numbers are ignored, not crashing
    s_bad = profile_analysis._record_consolidation(conn, [{
        "choice": "new", "name": "x", "direction": "over",
        "paper_numbers": [99, "bad", 3], "rationale": "r"}], ordered)
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
         "paper_numbers": [3, 5], "rationale": "same taste"},           # +F2,F3
        {"choice": "merge_into_closed", "existing_pattern_id": p_tomb,
         "paper_numbers": [4], "rationale": "came back"},               # +F5 (new)
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
    # first version silently skipped -- losing the candidate entirely) ---
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
        "paper_numbers": [3, 5], "rationale": "again"}], ordered)
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

    # --- 7. DIRECTION COMES FROM THE FLAGS, NOT THE MODEL ---------------------------------
    # `direction` was asking one field two questions: which way the judge erred (arithmetic on
    # the deltas, which this function already has in hand) and what is wrong at the other end
    # (a judgment about the profile). Only the second needs a model, and the first was being
    # routed through two LLM stages as English and arriving flipped. So code now answers it.
    #
    # TESTED HERE, model-free, because it IS pure code -- the paid gate cannot cover it, since
    # it can only exercise whatever the model happens to propose that run. On 2026-08-16 the
    # model proposed a cross-cutting label and the override branch never fired at all, which is
    # exactly why this belongs offline.
    #
    # Fixture deltas: F1 -0.80, F2 -0.60, F3 -0.40 (negative), F4 +0.70, F5 +0.50, F6 +0.35.
    # Render order is |delta| descending: papers 1..6 = F1, F4, F2, F5, F3, F6.
    def _one_new(direction, paper_numbers):
        out = profile_analysis._record_consolidation(conn, [{
            "choice": "new", "name": "dir probe", "direction": direction,
            "paper_numbers": paper_numbers, "rationale": "r"}], ordered)
        rec = out["new"][0]
        note = conn.execute(
            "SELECT note FROM pattern_events WHERE pattern_id=? AND event='created'",
            (rec["id"],)).fetchone()[0] or ""
        return rec, note

    # (a) the model contradicts the flags -> the FLAGS win, and the disagreement is RECORDED.
    # Silently correcting would turn a measurable prompt defect into a mystery, and the harness
    # grades the PROPOSAL precisely because the recorded value is now arithmetic.
    rec, note = _one_new("under", [1, 3])            # papers 1,3 -> F1,F2, both negative
    assert rec["direction"] == "over", rec
    assert rec.get("direction_proposed") == "under", rec
    assert "from flag deltas: over" in note and "'under'" in note, note
    print("direction: model said under, flags say over -> recorded over, disagreement logged")

    # (b) an OFF-SCHEMA word clamps to `under`, but direction_proposed keeps the RAW word so
    # the grader does not read a clamp as an inversion the model never proposed. This branch
    # used to assert that `judge-not-applying` was kept untouched; that value was deleted
    # 2026-08-25, and what replaced it is the recovery path for anything outside the enum.
    rec, note = _one_new("judge-not-applying", [1, 3])
    assert rec["direction"] == "over", rec                    # flags win
    assert rec.get("direction_coerced") == "judge-not-applying", rec
    assert rec.get("direction_proposed") == "judge-not-applying", rec
    print("direction: an off-schema word is clamped, and reported raw, not as an inversion")

    # (c) agreement is silent -- no note fragment, nothing for the grader to flag
    rec, note = _one_new("over", [1, 3])
    assert rec["direction"] == "over" and "direction_proposed" not in rec, rec
    assert "flag deltas" not in note, note
    print("direction: model agreed with the flags -> recorded, nothing logged")

    # (d) flags pointing BOTH ways keep the model's word and say so. patterns.direction has no
    # value for it, and this is the free measurement for the deferred "wrong in both directions
    # at once" question -- if it shows up often on real flags, a fifth value has earned itself.
    rec, note = _one_new("under", [1, 2])            # F1 negative, F4 positive
    assert rec["direction"] == "under", rec
    assert "BOTH ways" in note, note
    print("direction: mixed-sign flags keep the model's word and are logged as mixed")

    # (e) nothing to compute from -> the model's word stands, unremarked
    rec, note = _one_new("under", [])
    assert rec["direction"] == "under" and "flag deltas" not in note, (rec, note)
    print("direction: no usable papers -> model's word stands")

    conn.close()
    SCRATCH.unlink(missing_ok=True)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
