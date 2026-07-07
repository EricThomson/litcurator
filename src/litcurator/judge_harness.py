"""
judge_harness.py -- the JUDGE's UNIT TESTS (fast floor-of-competence gate).

Distinct from the benchmark (the slow statistical eval): a small fixture of OBVIOUS
papers with loose bands (reject <= 0.25, keep >= 0.75, a dead middle), each tagged
with the boundary it covers. Every case is scored under the ACTIVE profile plus a
given prompt -- the active prompt on disk by default, or a DRAFT passed in from the
workbench -- so it tests the WHOLE JUDGE and guards BOTH biconvex knobs (prompt AND
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
"""

import hashlib
import json

from litcurator import config, db_interface, judge, profile_interface, prompt_interface


def _ascii(text):
    """Console-safe: the corpus is full of non-cp1252 glyphs (Greek, em-dashes)."""
    return (text or "").encode("ascii", "replace").decode()


def _fp(text):
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def load_cases(path=None):
    path = path or config.JUDGE_HARNESS_CASES_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def run_tests(prompt_text=None, profile_text=None, cases=None, conn=None):
    """Score every case under (profile, prompt) and check its band. Returns a list of
    result dicts (the case fields plus score/passed/title/journal/rationale/
    possible_mismatch/error). Pure: does not print. prompt_text/profile_text default
    to whatever is active on disk."""
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
        items = [{"title": articles[c["pmid"]]["title"],
                  "abstract": articles[c["pmid"]].get("abstract"),
                  "journal": articles[c["pmid"]].get("journal"),
                  "pages": articles[c["pmid"]].get("pages")} for c in scorable]

        judgments = []
        if items:
            judgments, _cost = judge.judge_articles_batch(items, profile_text, system_prompt=prompt_text)

        judged = {c["pmid"]: j for c, j in zip(scorable, judgments)}
        results = []
        for c in cases:
            art = articles[c["pmid"]]
            if art is None:
                results.append({**c, "title": c.get("title"), "journal": None,
                                "score": None, "passed": None, "rationale": None,
                                "possible_mismatch": None, "error": "pmid not in articles"})
                continue
            j = judged[c["pmid"]]
            score = j["estimated_score"]
            results.append({**c, "title": art["title"], "journal": art["journal"],
                            "score": score, "passed": c["low"] <= score <= c["high"],
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
             f"{len(scored)} scored | {len(passed)} pass, {len(fails)} fail"
             + (f" | {len(errs)} unresolved" if errs else "")]

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
        lines.append("\nUNRESOLVED (pmid missing from DB):")
        lines += [f"  {r['pmid']} {_ascii(r.get('title'))[:70]}" for r in errs]

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
