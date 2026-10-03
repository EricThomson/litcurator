"""
report.py -- render a run into something readable, and save it.

Mirrors judge_harness: the console gets the summary, the saved file gets the summary plus the
full per-gate transcripts and ALL CHECKS, every check on its own line. Sixty rounds of narrative
is unreadable on a console but is exactly what you want in the artifact when something is red,
and ALL CHECKS is what --quick rebuilds the report from.

Failures come first and carry their interpretation. Empty sections are omitted entirely, so a
clean run has no failure headers at all.
"""

import hashlib
import re
from datetime import datetime

from litcurator import config

def fingerprint(text):
    """The 12-char content address of a prompt, so an archived report says exactly which text
    produced it. Same scheme as the profile and prompt ids in the database."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def _ascii(text):
    """Model-written pattern names can carry glyphs a cp1252 console cannot print."""
    return (text or "").encode("ascii", "replace").decode("ascii")


def _counts(results):
    checks = [c for r in results for c in r["checks"]]
    return len(checks), sum(1 for ok, _, _ in checks if ok)


def verdict_line(results, n_skipped_groups=0):
    """The one line that answers "did everything pass?". It heads the report and the CLI prints
    it again as the very last line, so the answer never needs scrolling for. On screen a check is
    a TEST and a gate is a GROUP of tests -- the user's wording (2026-10-03); "gate" stays the
    word in the code."""
    total, passed = _counts(results)
    line = f"{passed}/{total} TESTS PASSED"
    failed = [r["name"] + (" (could not run)" if r["error"] else "")
              for r in results if not r["passed"] or r["error"]]
    if failed:
        line += " -- failures in " + ", ".join(failed)
    if n_skipped_groups:
        line += f" | {n_skipped_groups} paid groups skipped: fix the failing free tests first"
    return line


def format_report(results, cluster_fp, consolidate_fp, drafts=(), cached=None):
    """The console report. `drafts` names any draft prompt files under test."""
    cost = sum(r["cost"] for r in results)
    seconds = sum(r["seconds"] for r in results)

    head = f"ANALYSIS HARNESS -- cluster {cluster_fp} | consolidate {consolidate_fp}"
    if drafts:
        head += "  (DRAFT: " + ", ".join(drafts) + ")"
    out = [head,
           verdict_line(results)
           + f" | ${cost:.4f} | {int(seconds // 60)}m{int(seconds % 60):02d}s"]
    if cached is not None:
        out.append(f"cluster calls served from cache: {cached}")

    red = [r for r in results if not r["passed"] or r["error"]]
    if red:
        out.append("\nFAILURES")
        for r in red:
            if r["error"]:
                out.append(f"\n  [ERROR] {r['name']} (layer {r['layer']}) -- could not run")
                out.append(f"    {_ascii(r['error'])}")
                continue
            bad = [c for c in r["checks"] if not c[0]]
            out.append(f"\n  [FAIL] {r['name']} (layer {r['layer']}) -- "
                       f"{sum(1 for c in r['checks'] if c[0])}/{len(r['checks'])} tests")
            for _ok, label, detail in bad:
                out.append(f"    [FAIL] {_ascii(label)}"
                           + (f"  ({_ascii(detail)})" if detail else ""))
            if r.get("when_red"):
                out.append(f"    when red: {r['when_red']}")

    green = [r for r in results if r["passed"] and not r["error"]]
    if green:
        out.append("\nPASSING")
        for r in green:
            n = len(r["checks"])
            out.append(f"  [PASS] {r['name']:<20} {n}/{n}"
                       + (f"   ${r['cost']:.4f}" if r["cost"] else ""))

    coverage = _coverage(results)
    if coverage:
        out.append("\nCOVERAGE (property: pass/total)")
        for prop in sorted(coverage):
            ok, tot = coverage[prop]
            out.append(f"  {prop:<34} {ok}/{tot}")
    return "\n".join(out)


def _coverage(results):
    """Group every check by the PROPERTY it guards, so an under-covered area is visible.

    The property is the text before the first colon in a check label -- the labels are already
    written that way ('purity: ...', 'merge: ...'), so this needs no extra bookkeeping."""
    out = {}
    for r in results:
        for ok, label, _ in r["checks"]:
            prop = label.split(":", 1)[0].strip() if ":" in label else r["name"]
            hit, total = out.get(prop, (0, 0))
            out[prop] = (hit + int(bool(ok)), total + 1)
    return out


def format_transcripts(results):
    """The per-gate narrative, appended to the SAVED report only."""
    out = ["", "=" * 70, "TRANSCRIPTS (what each gate actually did)", "=" * 70]
    for r in results:
        if not r.get("transcript"):
            continue
        out += ["", "-" * 70, f"{r['name']} (layer {r['layer']})", "-" * 70,
                _ascii(r["transcript"])]
    return "\n".join(out)


def write_report(text, runs_dir=None, selector=None):
    """Save a timestamped report. Never overwritten, so you accumulate a history to compare
    across edits, exactly like the judge harness.

    `selector` is what was asked for -- None (everything), 'free', or one gate name -- and it
    goes in the FILENAME. Without it a full suite, a free-only run and a single named gate all produce
    identically-shaped names, so the only way to tell them apart is to open them. That is not
    hypothetical: a `quick` run and a `note-carry` run sat next to each other on 2026-08-16 and
    the newest-file-wins assumption read the wrong one. Same fix as the `_dryrun` marker on the
    suggestions report, for the same reason."""
    runs_dir = runs_dir or config.ANALYSIS_HARNESS_RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9-]+", "-", selector) if selector else "all"
    path = runs_dir / f"analysis_harness_{tag}_{datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# ALL CHECKS: the full record of a run, and reading it back (--quick)
# ---------------------------------------------------------------------------
# Every check of every group on its own line, so a run can be dug into later (one check tracked
# across many runs, say) and --quick can rebuild the report exactly. The judge harness's
# RATIONALES section does the same job for its scores.

_ALL_CHECKS = "ALL CHECKS (every check, one line each"
_HEADER = re.compile(r"^ANALYSIS HARNESS -- cluster (\S+) \| consolidate (\S+)"
                     r"(?:  \(DRAFT: (.*)\))?$", re.M)
_GROUP = re.compile(r"^GROUP (\S+)  layer=(\S+)  passed=(yes|no)  cost=([\d.]+)"
                    r"  seconds=([\d.]+)$")
_CHECK = re.compile(r"^  \[(PASS|FAIL)\] ?(.*)$")


class ReportPredatesAllChecks(ValueError):
    """A report saved before ALL CHECKS existed: it can be shown, not rebuilt."""


def _one_line(text):
    """ASCII with whitespace collapsed: every field has to fit on one line to read back."""
    return " ".join(_ascii(str(text or "")).split())


def format_all_checks(results):
    """Every group and every check, one line each, in the layout read_report parses back:

        GROUP <name>  layer=<n>  passed=<yes|no>  cost=<dollars>  seconds=<s>
          error: <why it could not run>        (only when it could not)
          when red: <how to read a failure>    (only when the group has one)
          [PASS] <check>
          [FAIL] <check>
            detail: <detail>                   (only when the check has one)
    """
    out = ["", "=" * 70, f"{_ALL_CHECKS}; --quick rebuilds the report from this)", "=" * 70]
    for r in results:
        out.append(f"GROUP {r['name']}  layer={r['layer']}  passed={'yes' if r['passed'] else 'no'}"
                   f"  cost={r['cost']:.4f}  seconds={r['seconds']:.1f}")
        if r["error"]:
            out.append(f"  error: {_one_line(r['error'])}")
        if r.get("when_red"):
            out.append(f"  when red: {_one_line(r['when_red'])}")
        for ok, label, detail in r["checks"]:
            out.append(f"  [{'PASS' if ok else 'FAIL'}] {_one_line(label)}")
            if detail:
                out.append(f"    detail: {_one_line(detail)}")
    return "\n".join(out)


def read_report(text):
    """(cluster_fp, consolidate_fp, drafts, results) rebuilt from a saved report's ALL CHECKS
    section, the results shaped like run_gates' minus the transcripts. Raises
    ReportPredatesAllChecks for a report saved before the section existed."""
    header = _HEADER.search(text)
    if header is None:
        raise ValueError("not an analysis harness report")
    if _ALL_CHECKS not in text:
        raise ReportPredatesAllChecks("saved before reports kept every check")
    results, group = [], None
    for line in text.split(_ALL_CHECKS, 1)[1].splitlines():
        found = _GROUP.match(line)
        if found:
            layer = found.group(2)
            group = {"name": found.group(1), "layer": int(layer) if layer.isdigit() else layer,
                     "passed": found.group(3) == "yes", "cost": float(found.group(4)),
                     "seconds": float(found.group(5)), "error": None, "when_red": "",
                     "transcript": "", "checks": []}
            results.append(group)
        elif group is None:
            continue
        elif line.startswith("  error: "):
            group["error"] = line[len("  error: "):]
        elif line.startswith("  when red: "):
            group["when_red"] = line[len("  when red: "):]
        elif line.startswith("    detail: ") and group["checks"]:
            ok, label, _ = group["checks"][-1]
            group["checks"][-1] = (ok, label, line[len("    detail: "):])
        elif (check := _CHECK.match(line)):
            group["checks"].append((check.group(1) == "PASS", check.group(2), ""))
    drafts = tuple(header.group(3).split(", ")) if header.group(3) else ()
    return header.group(1), header.group(2), drafts, results


def _reader_matches_writer():
    """Writes a small report with today's formatter and reads it back, so a change to the
    layout breaks --quick loudly instead of making it misread results."""
    written = [
        {"name": "free-one", "layer": 1, "passed": True, "cost": 0.0, "seconds": 1.5,
         "error": None, "when_red": "", "transcript": "",
         "checks": [(True, "first check", ""), (True, "purity: second", "0.9 >= 0.6")]},
        {"name": "paid-two", "layer": 2, "passed": False, "cost": 0.1208, "seconds": 30.0,
         "error": None, "when_red": "read the fixture first", "transcript": "",
         "checks": [(True, "kept", ""), (False, "note-directive", "expected the instruction")]},
        {"name": "broken-three", "layer": 2, "passed": False, "cost": 0.0, "seconds": 0.5,
         "error": "API down", "when_red": "", "transcript": "",
         "checks": [(False, "broken-three: stopped", "ValueError")]},
    ]
    text = (format_report(written, "CLUSTERFP", "CONSOLIDATEFP", drafts=("d.md",))
            + "\n" + format_all_checks(written))
    keys = ("name", "layer", "passed", "cost", "seconds", "error", "when_red", "checks")
    cluster_fp, consolidate_fp, drafts, read = read_report(text)
    if ((cluster_fp, consolidate_fp, drafts) != ("CLUSTERFP", "CONSOLIDATEFP", ("d.md",))
            or [{k: r[k] for k in keys} for r in read] != [{k: w[k] for k in keys} for w in written]):
        raise RuntimeError("analysis_harness read_report no longer matches the report format, so "
                           "--quick would misread results. Fix read_report to match "
                           "format_report / format_all_checks.")


def latest_report(runs_dir=None):
    """The newest saved report of any kind (full, free or one group), by the timestamp every
    filename ends with -- the names start with the selector, so sorting by name would not do."""
    runs_dir = runs_dir or config.ANALYSIS_HARNESS_RUNS_DIR
    reports = list(runs_dir.glob("analysis_harness_*.md")) if runs_dir.exists() else []
    return max(reports, key=lambda p: p.stem[-15:]) if reports else None


def quick(runs_dir=None):
    """(path, text, rebuilt) for the newest saved report. `rebuilt` is read_report's
    (cluster_fp, consolidate_fp, drafts, results), or None for a report saved before ALL CHECKS
    existed, which can only be shown as saved. Nothing is run and nothing is saved."""
    _reader_matches_writer()
    path = latest_report(runs_dir)
    if path is None:
        raise FileNotFoundError("no saved analysis harness report yet -- run it once")
    text = path.read_text(encoding="utf-8")
    try:
        return path, text, read_report(text)
    except ReportPredatesAllChecks:
        return path, text, None
