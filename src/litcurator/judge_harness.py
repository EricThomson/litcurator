"""
judge_harness.py -- the JUDGE's UNIT TESTS (fast floor-of-competence gate).

Distinct from the benchmark (the slow statistical eval): a small fixture of OBVIOUS
papers with loose bands (reject <= 0.25, keep >= 0.75, a dead middle), each tagged
with the boundary it covers. Every case is scored under the ACTIVE profile plus a
given prompt -- the active prompt on disk by default, or a DRAFT passed in from the
workbench -- so it tests the WHOLE JUDGE and guards BOTH of the judge's inputs (prompt AND
profile) at once. A red can be either knob.

A case is keyed by kind:
  - guard      -- correct now, MUST stay in band. A guard FAIL = your edit broke
                  something (over-correction / regression you introduced).
  - regression -- wrong now (a documented over/under-score), expected to FLIP into
                  band when the fix lands. A regression fail is expected until then;
                  a regression PASS is the proof the fix worked.

The fixture is scored in a single batched judge call. Bands live in
config.JUDGE_HARNESS_CASES_FILE; pin the profile you want to test against before
running (the runner reads whatever profile is active on disk). The judge's reasoning
per case is kept and appended to the saved report (format_rationales), so "what did
it say about X" is answerable without re-scoring.

Run:
    litcurator judge_harness                 # active prompt + active profile
    litcurator judge_harness --prompt draft.md
    litcurator judge_harness --dry-run       # verify the fixture resolves, no scoring
    litcurator judge_harness --quick         # free: re-grade the last saved run's scores
"""

import hashlib
import json
import re

from litcurator import config, db_interface, judge, profile_interface, prompt_interface


def _ascii(text):
    """Console-safe: the corpus is full of non-cp1252 glyphs (Greek, em-dashes)."""
    return (text or "").encode("ascii", "replace").decode()


def _fp(text):
    # 12-char prefix of SHA256, matching the DB content-address (profiles.id /
    # prompts.id = SHA256(content)); display only, for the harness report header.
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def _chunks(seq, n):
    """Yield successive n-sized chunks of seq (mirrors pipeline._chunks)."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _in_band(case, score):
    return case["low"] <= score <= case["high"]


def load_cases(path=None):
    path = path or config.JUDGE_HARNESS_CASES_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def run_tests(prompt_text=None, profile_text=None, cases=None, conn=None, progress=None):
    """Score every case under (profile, prompt) and check its band. Returns a list of
    result dicts (the case fields plus score/passed/title/journal/rationale/
    possible_mismatch/error). Does not print. prompt_text/profile_text default to
    whatever is active on disk. Scores in chunks of JUDGE_BATCH_SIZE, exactly like the
    live pipeline (so harness scores match production, not an all-in-one-batch that the
    pipeline never uses), and calls progress(done, total) after each chunk if given --
    the CLI passes a callback to print n/total; the workbench passes none (stays silent)."""
    close = False
    if conn is None:
        conn = db_interface.get_connection()
        close = True
    try:
        if prompt_text is None:
            prompt_text = prompt_interface.load_active()
        if profile_text is None:
            profile_text = profile_interface.load_active()
        if cases is None:
            cases = load_cases()

        articles = {c["pmid"]: db_interface.get_article(conn, c["pmid"]) for c in cases}
        scorable = [c for c in cases if articles[c["pmid"]] is not None]
        from litcurator.pipeline import JUDGE_BATCH_SIZE

        judged = {}
        chunk_errors = {}
        total = len(scorable)
        done = 0
        for chunk in _chunks(scorable, JUDGE_BATCH_SIZE):
            items = [{"title": articles[c["pmid"]]["title"],
                      "abstract": articles[c["pmid"]].get("abstract"),
                      "journal": articles[c["pmid"]].get("journal"),
                      "pages": articles[c["pmid"]].get("pages")} for c in chunk]
            try:
                judgments, _cost = judge.judge_articles_batch(items, profile_text, system_prompt=prompt_text)
                for c, j in zip(chunk, judgments):
                    judged[c["pmid"]] = j
            except Exception as e:
                # one flaky chunk (e.g. a transient empty API response) should not nuke
                # the whole run -- mark its cases and keep scoring the rest.
                for c in chunk:
                    chunk_errors[c["pmid"]] = str(e)
            done += len(chunk)
            if progress:
                progress(done, total)
        results = []
        for c in cases:
            art = articles[c["pmid"]]
            if art is None:
                results.append({**c, "title": c.get("title"), "journal": None,
                                "score": None, "passed": None, "rationale": None,
                                "possible_mismatch": None, "error": "pmid not in articles"})
                continue
            if c["pmid"] not in judged:
                results.append({**c, "title": art["title"], "journal": art["journal"],
                                "score": None, "passed": None, "rationale": None,
                                "possible_mismatch": None,
                                "error": f"judge failed: {chunk_errors.get(c['pmid'], 'unknown')}"})
                continue
            j = judged[c["pmid"]]
            score = j["estimated_score"]
            results.append({**c, "title": art["title"], "journal": art["journal"],
                            "score": score, "passed": _in_band(c, score),
                            "rationale": j.get("curation_rationale"),
                            "possible_mismatch": j.get("possible_mismatch"),
                            "error": None})
        return results, _fp(prompt_text), _fp(profile_text)
    finally:
        if close:
            conn.close()


def _band(c):
    if c["high"] <= 0.25:
        return f"<= {c['high']:.2f}"
    if c["low"] >= 0.75:
        return f">= {c['low']:.2f}"
    return f"{c['low']:.2f}-{c['high']:.2f}"


def verdict_line(results):
    """The one line that answers "did everything pass?". It heads the report and the CLI prints
    it again as the very last line, so the answer never needs scrolling for. Keeps the one split
    that matters: a failing GUARD means an edit broke something; a failing REGRESSION case is a
    known problem whose fix has not landed yet."""
    scored = [r for r in results if r["error"] is None]
    n_passed = sum(1 for r in scored if r["passed"])
    broke = sum(1 for r in scored if not r["passed"] and r["kind"] == "guard")
    known = sum(1 for r in scored if not r["passed"] and r["kind"] == "regression")
    unscored = len(results) - len(scored)
    notes = []
    if broke:
        notes.append(f"{broke} that should pass failed (something broke)")
    if known:
        notes.append(f"{known} known problem{'' if known == 1 else 's'} still unfixed, as expected")
    if unscored:
        notes.append(f"{unscored} could not be scored")
    return f"{n_passed}/{len(results)} TESTS PASSED" + (" -- " + "; ".join(notes) if notes else "")


def format_report(results, prompt_fp="", profile_fp=""):
    """Compact plain-text report: failures first, guard/regression split, coverage by
    tag. The judge's reasoning is NOT here -- see format_rationales (appended to the
    saved file)."""
    scored = [r for r in results if r["error"] is None]
    errs = [r for r in results if r["error"]]
    fails = [r for r in scored if not r["passed"]]
    guard_fails = [r for r in fails if r["kind"] == "guard"]
    regr_fails = [r for r in fails if r["kind"] == "regression"]
    passed = [r for r in scored if r["passed"]]

    lines = [f"JUDGE HARNESS -- prompt {prompt_fp} | profile {profile_fp}",
             verdict_line(results)]

    def row(r):
        mark = "PASS" if r["passed"] else "FAIL"
        return (f"  [{mark}] {r['score']:.2f} want {_band(r):<8} {r['kind']:<10} "
                f"{','.join(r['tests'])} | {r['pmid']} {_ascii(r['title'])[:70]}")

    if guard_fails:
        lines.append("\nGUARD FAILS (your edit broke these -- real regressions):")
        lines += [row(r) for r in guard_fails]
    if regr_fails:
        lines.append("\nREGRESSION FAILS (expected red until the fix lands):")
        lines += [row(r) for r in regr_fails]
    if passed:
        lines.append("\nPASSING:")
        lines += [row(r) for r in passed]
    if errs:
        lines.append("\nUNRESOLVED (missing from DB, or judge error):")
        lines += [f"  {r['pmid']} {_ascii(r.get('title'))[:55]}  -- {_ascii(r.get('error'))[:45]}"
                  for r in errs]

    # coverage by tag
    tags = {}
    for r in scored:
        for t in r["tests"]:
            d = tags.setdefault(t, [0, 0])
            d[0] += 1
            d[1] += 1 if r["passed"] else 0
    lines.append("\nCOVERAGE (tag: pass/total):")
    lines += [f"  {t}: {p}/{n}" for t, (n, p) in sorted(tags.items())]
    return "\n".join(lines)


def format_rationales(results):
    """The judge's reasoning per case (rationale + possible_mismatch) -- appended to
    the saved report so 'what did it say about X' is answerable without re-scoring."""
    lines = ["", "=" * 70, "RATIONALES (judge reasoning per case)", "=" * 70]
    for r in results:
        if r["error"]:
            continue
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(f"\n[{mark}] {r['score']:.2f}  {r['pmid']}  {_ascii(r['title'])[:72]}")
        lines.append(f"  rationale: {_ascii(r.get('rationale'))}")
        lines.append(f"  mismatch : {_ascii(r.get('possible_mismatch'))}")
    return "\n".join(lines)


def write_report(report_text, runs_dir=None):
    """Save a report to a timestamped file in the runs dir; return the path. Each run
    is kept (not overwritten) so you can compare across edits."""
    from datetime import datetime
    runs_dir = runs_dir or config.JUDGE_HARNESS_RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"judge_harness_{datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(report_text, encoding="utf-8")
    return path


def dry_run(cases=None, conn=None):
    """Verify every fixture pmid resolves and print the cases, without scoring."""
    close = False
    if conn is None:
        conn = db_interface.get_connection()
        close = True
    try:
        cases = cases if cases is not None else load_cases()
        lines = [f"{len(cases)} cases in {config.JUDGE_HARNESS_CASES_FILE.name}"]
        missing = 0
        for c in cases:
            art = db_interface.get_article(conn, c["pmid"])
            ok = "ok " if art else "MISSING"
            if not art:
                missing += 1
            title = _ascii(art["title"])[:60] if art else _ascii(c.get("title"))[:60]
            lines.append(f"  {ok} {c['kind']:<10} want {_band(c):<8} "
                         f"{','.join(c['tests'])} | {c['pmid']} {title}")
        lines.append(f"\n{len(cases) - missing} resolved, {missing} missing.")
        return "\n".join(lines)
    finally:
        if close:
            conn.close()


# ---------------------------------------------------------------------------
# --quick: re-grade the newest saved report, for free
# ---------------------------------------------------------------------------
# Every case here is the judge scoring a paper, so unlike the analysis harness there is no
# free half to split off. What IS free is re-grading scores already on disk: each saved report
# carries every paper's score and reasoning, so a changed range in the fixture, or just another
# look at the output, costs nothing. It cannot tell you anything about an edit to the prompt or
# the profile, because nothing is re-scored, and it says so at the top.

_HEADER = re.compile(r"^JUDGE HARNESS -- prompt (\S+) \| profile (\S+)", re.M)
_CASE = re.compile(r"^\[(?:PASS|FAIL)\] (\d+\.\d+)\s+(\S+)\s+(.*)$")


def latest_report(runs_dir=None):
    """The newest saved judge harness report, or None if there is none."""
    runs_dir = runs_dir or config.JUDGE_HARNESS_RUNS_DIR
    reports = sorted(runs_dir.glob("judge_harness_*.md")) if runs_dir.exists() else []
    return reports[-1] if reports else None


def read_report(text):
    """(prompt_fp, profile_fp, {pmid: {score, rationale, possible_mismatch}}) from a saved
    report. Reads the RATIONALES section that format_rationales writes. Scores come back at the
    two decimals the report prints them with. A paper that could not be scored in that run is
    absent from the dict."""
    header = _HEADER.search(text)
    if header is None or "RATIONALES" not in text:
        raise ValueError("not a judge harness report (no header or no RATIONALES section)")
    scores, current = {}, None
    for line in text.split("RATIONALES", 1)[1].splitlines():
        case = _CASE.match(line)
        if case:
            current = {"score": float(case.group(1)), "rationale": None,
                       "possible_mismatch": None}
            scores[case.group(2)] = current
        elif current and line.startswith("  rationale: "):
            current["rationale"] = line[len("  rationale: "):]
        elif current and line.startswith("  mismatch : "):
            current["possible_mismatch"] = line[len("  mismatch : "):]
    return header.group(1), header.group(2), scores


def regrade(cases, scores):
    """Results shaped exactly like run_tests', from saved scores and TODAY's ranges."""
    results = []
    for c in cases:
        saved = scores.get(c["pmid"])
        if saved is None:
            results.append({**c, "journal": None, "score": None, "passed": None,
                            "rationale": None, "possible_mismatch": None,
                            "error": "not scored in that run"})
            continue
        results.append({**c, "journal": None, "score": saved["score"],
                        "passed": _in_band(c, saved["score"]),
                        "rationale": saved["rationale"],
                        "possible_mismatch": saved["possible_mismatch"], "error": None})
    return results


def _reader_matches_writer():
    """Writes a small report with today's formatter and reads it back, so a change to the
    report's layout breaks --quick loudly instead of making it misread scores."""
    cases = [{"pmid": "P1", "low": 0.0, "high": 0.25, "kind": "guard", "tests": ["t"],
              "title": "one"},
             {"pmid": "P2", "low": 0.75, "high": 1.0, "kind": "regression", "tests": ["t"],
              "title": "two"},
             {"pmid": "P3", "low": 0.0, "high": 0.25, "kind": "guard", "tests": ["t"],
              "title": "three"}]
    written = regrade(cases, {"P1": {"score": 0.12, "rationale": "r1", "possible_mismatch": "m1"},
                              "P2": {"score": 0.62, "rationale": "r2", "possible_mismatch": "m2"}})
    text = format_report(written, "PROMPTFP", "PROFILEFP") + "\n" + format_rationales(written)
    prompt_fp, profile_fp, scores = read_report(text)
    expected = {"P1": {"score": 0.12, "rationale": "r1", "possible_mismatch": "m1"},
                "P2": {"score": 0.62, "rationale": "r2", "possible_mismatch": "m2"}}
    if (prompt_fp, profile_fp, scores) != ("PROMPTFP", "PROFILEFP", expected):
        raise RuntimeError("judge_harness.read_report no longer matches the report format, so "
                           "--quick would misread scores. Fix read_report to match "
                           "format_report / format_rationales.")


def quick(cases=None, runs_dir=None):
    """Re-grade the newest saved report against the current fixture. Returns
    (results, prompt_fp, profile_fp, report_path, warnings). Nothing is scored or saved."""
    _reader_matches_writer()
    path = latest_report(runs_dir)
    if path is None:
        raise FileNotFoundError("no saved judge harness report yet -- run the real one once")
    prompt_fp, profile_fp, scores = read_report(path.read_text(encoding="utf-8"))
    results = regrade(cases if cases is not None else load_cases(), scores)
    warnings = []
    active_prompt = _fp(prompt_interface.read_active_or_empty())
    active_profile = _fp(profile_interface.read_active_or_empty())
    if prompt_fp != active_prompt:
        warnings.append(f"your active judge prompt ({active_prompt}) is not the one these scores "
                        f"came from ({prompt_fp}), so they say nothing about it")
    if profile_fp != active_profile:
        warnings.append(f"your active profile ({active_profile}) is not the one these scores came "
                        f"from ({profile_fp}), so they say nothing about it")
    return results, prompt_fp, profile_fp, path, warnings
