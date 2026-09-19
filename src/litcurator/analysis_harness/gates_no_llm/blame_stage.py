"""
blame_stage.py -- unit tests for the blame field (free, no model).

blame answers one question per pattern: is this taste already written in the profile? If it is
and the judge scored against it anyway, more profile prose provably does not help -- measured
2026-09-07, where 0 of 4 such patterns moved after being incorporated, while 3 of 3 patterns
adding a NEW disinterest line landed cleanly.

The consequence that has to hold, and the only one worth a gate: a prompt-blamed pattern's
'incorporated' event stamps the JUDGE PROMPT, not a profile version that will never contain the
fix. That untruth is already in the live log (the Annual Review pattern, 2026-09-01), and it is
what makes "which version absorbed this pattern" unanswerable for exactly the patterns whose
answer matters most.
"""

import contextlib
import os

from litcurator import db_interface as DB
from litcurator import error_analysis as PA

from .. import machinery as H


@contextlib.contextmanager
def _world():
    path = H.scratch_db_path("blame_stage")
    conn, scoring_run, profile_id = H.build_db(path, "synthetic profile")
    flag_intended = {}
    papers = [{"pmid": f"SYNBLAME{i}", "title": f"Paper {i}", "abstract": "a", "journal": "J",
               "judge_score": 0.6, "rationale": "r", "user_score": 0.2, "note": "n",
               "intended": "X"} for i in (1, 2, 3)]
    H.add_round_flags(conn, scoring_run, papers, flag_intended)
    ordered = sorted(DB.get_flags(conn, exclude_attached=True),
                     key=lambda f: (-abs(f["delta"]), f["id"]))
    try:
        yield conn, profile_id, ordered
    finally:
        conn.close()
        if path.exists():
            os.unlink(path)


def _record(conn, profile_id, candidates, ordered):
    run_id = DB.create_analysis_run(
        conn, DB.get_or_create_prompt(conn, "synthetic analysis prompt", kind="analysis"),
        "m", "m", profile_id=profile_id, date_start=None, date_end=None,
        n_flags=len(ordered), cost_usd=0.0)
    return PA._record_consolidation(conn, candidates, ordered, analysis_run_id=run_id)


def _new(name, papers, blame=None):
    c = {"choice": "new", "name": name, "direction": "under", "description": "d",
         "suggested_edit": "e", "rank": papers[0], "priority": "act_now",
         "paper_numbers": papers, "rationale": "why"}
    if blame is not None:
        c["blame"] = blame
    return c


def test_blame_is_recorded_and_defaults_to_profile():
    with _world() as (conn, profile_id, ordered):
        _record(conn, profile_id, [_new("Stated", [1], blame="prompt"),
                                   _new("Missing", [2], blame="profile"),
                                   _new("Unset", [3])], ordered)
        got = {r["name"]: r["blame"] for r in conn.execute("SELECT name, blame FROM patterns")}
        assert got == {"Stated": "prompt", "Missing": "profile", "Unset": "profile"}, got
        print(f"recorded {got}; an omitted blame defaults to profile (today's behaviour)")


def test_garbage_blame_falls_back_rather_than_raising():
    with _world() as (conn, profile_id, ordered):
        _record(conn, profile_id, [_new("Junk", [1], blame="the judge prompt probably")],
                ordered)
        row = conn.execute("SELECT blame FROM patterns WHERE name='Junk'").fetchone()
        assert row["blame"] == "profile", row["blame"]
        # A model writing an off-schema value must never lose the pattern -- same rule the
        # unrecognized-choice path follows.
        assert conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0] == 1
        print("an off-schema blame clamps to profile and the pattern is still recorded")


def test_prompt_blamed_incorporation_stamps_the_prompt_not_the_profile():
    with _world() as (conn, profile_id, ordered):
        _record(conn, profile_id, [_new("Stated", [1], blame="prompt"),
                                   _new("Missing", [2], blame="profile")], ordered)
        ids = {r["name"]: r["id"] for r in conn.execute("SELECT id, name FROM patterns")}
        prompt_id = DB.get_or_create_prompt(conn, "a judge prompt", kind="judge")

        DB.add_pattern_event(conn, ids["Stated"], "incorporated", prompt_id=prompt_id)
        DB.add_pattern_event(conn, ids["Missing"], "incorporated", profile_id=profile_id)

        stated = conn.execute(
            "SELECT profile_id, prompt_id FROM pattern_events WHERE pattern_id=? "
            "AND event='incorporated'", (ids["Stated"],)).fetchone()
        missing = conn.execute(
            "SELECT profile_id, prompt_id FROM pattern_events WHERE pattern_id=? "
            "AND event='incorporated'", (ids["Missing"],)).fetchone()
        assert stated["prompt_id"] == prompt_id and stated["profile_id"] is None, dict(stated)
        assert missing["profile_id"] == profile_id and missing["prompt_id"] is None, dict(missing)
        print("prompt-blamed stamps prompt_id; profile-blamed stamps profile_id")


def test_blame_survives_the_report_round_trip():
    """promote_suggestions re-records a round from its markdown, so a blame that does not
    survive rendering would silently become 'profile' on the way back -- routing a prompt job
    into the profile queue, which is the whole failure this field exists to stop."""
    import pathlib
    import tempfile
    cands = [_new("Stated", [1], blame="prompt"), _new("Missing", [2], blame="profile"),
             _new("Unset", [3])]
    md = PA._format_consolidation_md(cands)
    assert "{PROMPT}" in md and md.count("{PROMPT}") == 1, md
    path = pathlib.Path(tempfile.mkdtemp()) / "r.md"
    path.write_text("# x\n\nUnattached flags: 3\n\n## Consolidation (choices)\n\n" + md,
                    encoding="utf-8")
    back, _ = PA.parse_consolidation_md(path)
    got = {c.get("name"): c.get("blame", "profile") for c in back}
    assert got == {"Stated": "prompt", "Missing": "profile", "Unset": "profile"}, got
    print(f"round-trip preserved {got}")


CHECKS = [
    test_blame_is_recorded_and_defaults_to_profile,
    test_garbage_blame_falls_back_rather_than_raising,
    test_prompt_blamed_incorporation_stamps_the_prompt_not_the_profile,
    test_blame_survives_the_report_round_trip,
]
