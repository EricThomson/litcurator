"""
report.py -- render a run into something readable, and save it.

Mirrors judge_harness: the console gets the summary, the saved file gets the summary plus the
full per-gate transcripts. Sixty rounds of narrative is unreadable on a console but is exactly
what you want in the artifact when something is red.

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

    `selector` is what was asked for -- None (everything), 'quick', or one gate name -- and it
    goes in the FILENAME. Without it a full suite, a `quick` and a single named gate all produce
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
