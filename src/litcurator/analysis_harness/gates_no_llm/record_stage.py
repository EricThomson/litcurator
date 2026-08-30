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
        {"choice": "discard", "name": "quirk", "paper_numbers": [], "rationale": "not a pattern"},
        {"choice": "merge_into_open", "existing_pattern_id": "deadbeefdeadbeef",
         "name": "Recovered", "direction": "over", "paper_numbers": [6], "rationale": "bad id"},
    ], ordered)
    assert len(s2["merged"]) == 1 and s2["merged"][0]["added"] == 2, s2["merged"]
    assert len(s2["recurred"]) == 1 and s2["recurred"][0]["added"] == 1, s2["recurred"]
    assert len(s2["discarded"]) == 1, s2["discarded"]
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

    # --- 8. HELD PATTERNS: recorded, not shown, and their flags stay in the pool -----------
    # Added 2026-08-26 with the `held` status. Until then a `hold` wrote nothing at all -- the
    # exact "dumped it where no code reads" failure profile_analysis's own docstring says the
    # redesign killed -- so a recognized-but-thin pattern was lost every round and re-derived
    # from raw papers. These checks pin the four properties that make the fix worth having.
    for t in ("pattern_flags", "pattern_events", "patterns"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    ordered = profile_analysis._format_papers(db_interface.get_flags(conn))[1]
    by_pos = {f["id"]: i + 1 for i, f in enumerate(ordered)}      # flag id -> paper number

    def _consolidate(*candidates):
        return profile_analysis._record_consolidation(conn, list(candidates), ordered)

    # (a) new + priority=hold mints a REAL row that the workbench never sees.
    s8 = _consolidate({"choice": "new", "name": "thin gap", "direction": "under",
                       "priority": "hold", "paper_numbers": [by_pos[flag_id["F1"]]],
                       "rationale": "real but not actionable yet"})
    assert len(s8["held"]) == 1 and not s8["new"], s8
    held_id = s8["held"][0]["id"]
    assert db_interface.get_pattern(conn, held_id) is not None, "a hold must write a row"
    assert [p["id"] for p in db_interface.get_held_patterns(conn)] == [held_id]
    assert held_id not in {p["id"] for p in db_interface.get_active_patterns(conn)}, \
        "a held pattern must NOT reach the workbench queue"
    print("held: new+hold writes a real pattern that stays off the active list")

    # (b) THE CRUX. Its flags keep their provenance AND stay in the clustering pool. Cluster is
    # the only step that reads papers, and accumulation works because thin flags pile up until
    # it sees several together -- draining them here would move that job to consolidate, which
    # never sees a paper at all.
    assert db_interface.get_pattern_provenance(conn, held_id), "held pattern must own its flags"
    pool = {f["id"] for f in db_interface.get_flags(conn, exclude_attached=True)}
    assert flag_id["F1"] in pool, "a HELD pattern's flags must stay in the clustering pool"
    print("held: flags are attached for provenance and still returned by the unattached pool")

    # (c) Coming back and still being thin logs another `held` -- so "sitting for three rounds"
    # is answerable -- and does NOT surface it.
    s8b = _consolidate({"choice": "merge_into_open", "existing_pattern_id": held_id,
                        "priority": "hold", "paper_numbers": [by_pos[flag_id["F2"]]],
                        "rationale": "came back, still thin"})
    assert len(s8b["held"]) == 1 and s8b["held"][0].get("returned"), s8b
    assert [e["event"] for e in db_interface.get_pattern_events(conn, held_id)] \
        == ["created", "held", "held"], db_interface.get_pattern_events(conn, held_id)
    assert held_id not in {p["id"] for p in db_interface.get_active_patterns(conn)}
    print("held: a return that is still thin re-holds and is logged, not silently skipped")

    # (d) PROMOTION, and it needs no threshold: `carried` is an active status, so the pattern
    # simply becomes visible. This is why no count constant is required anywhere.
    s8c = _consolidate({"choice": "merge_into_open", "existing_pattern_id": held_id,
                        "priority": "act_now", "paper_numbers": [by_pos[flag_id["F3"]]],
                        "rationale": "enough to act on now"})
    assert len(s8c["surfaced"]) == 1 and s8c["surfaced"][0]["id"] == held_id, s8c
    assert held_id in {p["id"] for p in db_interface.get_active_patterns(conn)}
    assert not db_interface.get_held_patterns(conn), "it must leave the held list"
    pool = {f["id"] for f in db_interface.get_flags(conn, exclude_attached=True)}
    assert flag_id["F1"] not in pool, "once surfaced, its flags are handled and leave the pool"
    print("held: act_now on a return promotes it, and its flags then leave the pool")

    # (e) A RE-FLAG is not new evidence. The user re-flags routinely (11 of 20 live papers), and
    # until this was fixed the extra flag row counted as an arrival and fired a carried event
    # for a paper the pattern already held.
    f1_pmid = conn.execute("SELECT pmid FROM flags WHERE id=?", (flag_id["F1"],)).fetchone()[0]
    ev1 = conn.execute("SELECT evaluation_id FROM flags WHERE id=?",
                       (flag_id["F1"],)).fetchone()[0]
    refl = db_interface.insert_flag(conn, ev1, 0.11, note="same paper, longer note")
    before = len(db_interface.get_pattern_events(conn, held_id))
    assert db_interface.attach_flags_to_pattern(conn, held_id, [refl]) == 0, \
        "a re-flag of an already-covered PAPER is not a new arrival"
    assert len(db_interface.get_pattern_events(conn, held_id)) == before
    assert db_interface.get_patterns(conn, statuses=("carried",))[0]["flag_count"] == 3, \
        "flag_count must count papers, not pattern_flags rows"
    print(f"held: a re-flag of pmid {f1_pmid} adds provenance but is not an arrival")

    # (f) discard is the one outcome that records nothing -- and that is correct, because the
    # model judged the candidate not to be a pattern at all.
    n_before = len(db_interface.get_patterns(conn))
    s8d = _consolidate({"choice": "discard", "name": "not a pattern",
                        "paper_numbers": [by_pos[flag_id["F6"]]], "rationale": "cluster noise"})
    assert len(s8d["discarded"]) == 1 and len(db_interface.get_patterns(conn)) == n_before
    assert flag_id["F6"] in {f["id"] for f in
                             db_interface.get_flags(conn, exclude_attached=True)}
    print("discard: records nothing and leaves its flags in the pool")

    # --- 9. THE QUEUE CAP: the model ranks, code cuts, nothing is dropped ------------------
    # Instructing this failed twice on real flags -- ten patterns queued against a stated cap of
    # eight, at exactly ten both times -- so it moved into code. Overflow is DEMOTED, never lost.
    for t in ("pattern_flags", "pattern_events", "patterns"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    n_over = config.MAX_ACT_NOW + 4
    # Ranks handed over SHUFFLED (worst first), so a cap that merely truncated the list in
    # arrival order would look correct by accident.
    cands = [{"choice": "new", "name": f"p{i}", "direction": "under", "priority": "act_now",
              "rank": n_over - i, "paper_numbers": [], "rationale": "r"} for i in range(n_over)]
    s9 = profile_analysis._record_consolidation(conn, cands, [])
    assert len(s9["new"]) == config.MAX_ACT_NOW, len(s9["new"])
    assert len(s9["held"]) == n_over - config.MAX_ACT_NOW, len(s9["held"])
    kept = sorted(c["rank"] for c in cands if c["priority"] == "act_now")
    assert kept == list(range(1, config.MAX_ACT_NOW + 1)), kept
    assert all(c.get("priority_asked") == "act_now" for c in cands if c["priority"] == "hold"), \
        "a demotion must keep the model's own word beside it"
    assert len(db_interface.get_patterns(conn)) == n_over, "nothing may be DROPPED, only demoted"
    print(f"cap: {n_over} wanted, {config.MAX_ACT_NOW} surfaced by rank, "
          f"{n_over - config.MAX_ACT_NOW} demoted and still recorded")

    for t in ("pattern_flags", "pattern_events", "patterns"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    few = [{"choice": "new", "name": f"q{i}", "direction": "under", "priority": "act_now",
            "rank": i + 1, "paper_numbers": [], "rationale": "r"} for i in range(3)]
    s9b = profile_analysis._record_consolidation(conn, few, [])
    assert len(s9b["new"]) == 3 and not s9b["held"], s9b
    print("cap: inert when fewer patterns want the queue than the cap allows")

    # --- 10. AN EMPTY MEMORY CANNOT OFFER A MERGE -----------------------------------------
    # Structural, not instructed: with zero patterns every existing_pattern_id is invented, so
    # the values come out of the enum and forced tool-use makes them unemittable. Both January
    # dry runs produced exactly one spurious merge while the prompt already forbade it.
    def _enum(tool):
        return (tool["input_schema"]["properties"]["candidates"]["items"]
                ["properties"]["choice"]["enum"])

    assert _enum(profile_analysis._consolidate_tool(has_memory=False)) == ["new", "discard"]
    full = _enum(profile_analysis._consolidate_tool(has_memory=True))
    assert "merge_into_open" in full and "merge_into_closed" in full, full
    assert _enum(profile_analysis._CONSOLIDATE_TOOL) == full, "the constant must not be mutated"
    print("empty memory: merge values are removed from the enum, and the constant is untouched")

    # --- 11. THE REPORT ROUND-TRIPS ------------------------------------------------------
    # `promote_suggestions` re-records a run you liked by parsing its own markdown report, which
    # is only safe because of this check. Render candidates, parse them back, assert they match.
    # A change to _format_consolidation_md then fails HERE, loudly, instead of silently
    # mis-recording a round -- which is the difference between parsing your own prose being a
    # reasonable design and being a trap.
    #
    # Every shape the renderer can emit is covered, because the ones that broke it during
    # development were the unusual ones: a real 32-char merge target (which used to render
    # truncated and could not be parsed back), a hallucinated target (rendered with a marker),
    # and a demoted candidate (whose RENDERED priority is post-cap, so the parser has to restore
    # the model's own `asked` word or promoting would bake in a cap that may since have changed).
    shapes = [
        {"choice": "new", "name": "A Gap", "direction": "under", "rank": 1,
         "priority": "act_now", "description": "one sentence", "suggested_edit": "add this",
         "rationale": "because", "paper_numbers": [1, 2]},
        {"choice": "new", "name": "Demoted", "direction": "over", "rank": 9, "priority": "hold",
         "priority_asked": "act_now", "rationale": "r [demoted: over the cap]",
         "paper_numbers": [3]},
        {"choice": "merge_into_open", "existing_pattern_id": "66e72f0dba644749b17e977616b3089f",
         "rank": 2, "priority": "act_now", "rationale": "same taste", "paper_numbers": [4]},
        {"choice": "merge_into_open", "existing_pattern_id": "human-only-studies", "rank": 3,
         "priority": "act_now", "rationale": "invented target", "paper_numbers": [6]},
        {"choice": "discard", "name": "Not real", "rank": 13, "priority": "hold",
         "rationale": "noise", "paper_numbers": [5]},
    ]
    md = ("# Pattern suggestions\n\nUnattached flags: 20  |  x\n\n"
          "## Consolidation (choices)\n\n" + profile_analysis._format_consolidation_md(shapes))
    report = Path(config.DATA_DIR) / "_scratch_roundtrip.md"
    report.write_text(md, encoding="utf-8")
    try:
        back, n_then = profile_analysis.parse_consolidation_md(report)
        assert n_then == 20, n_then
        assert len(back) == len(shapes), (len(back), len(shapes))
        for orig, got in zip(shapes, back):
            want = {k: v for k, v in orig.items() if k != "priority_asked"}
            if orig.get("priority_asked"):
                want["priority"] = orig["priority_asked"]
            assert got == want, ("round trip lost or changed a field", got, want)
    finally:
        report.unlink(missing_ok=True)
    print(f"report round-trips: {len(shapes)} candidate shapes render and parse back identically")

    conn.close()
    SCRATCH.unlink(missing_ok=True)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
