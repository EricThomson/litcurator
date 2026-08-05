"""
machinery.py -- the scratch world the analysis gates run in.

Builds a throwaway database, inserts synthetic flags, runs ONE review session through
the real cluster -> consolidate -> record pipeline, and reads the provenance graph back
out. The live database is never opened.

No argparse and no printing: every gate drives these functions and does its own
reporting, so this module stays reusable.
"""

import hashlib
import os
from collections import Counter
from dataclasses import dataclass

from litcurator import config, db_interface as DB, profile_analysis as PA


# A pattern's direction says what KIND of claim it makes, and that matters for every check that
# counts patterns.
#
#   over / under        a TASTE gap: the profile misses something, or over-triggers on it. These
#                       are per-taste, so two of them covering one taste is fragmentation and one
#                       of them covering two tastes is a fusion.
#   sharpen             the profile's wording is being misread.
#   judge-not-applying  the profile is clear and the judge ignores it; the PROMPT is the problem.
#
# The last two are observations about the profile or the judge rather than about a taste, so they
# legitimately span several tastes at once ("the judge is penalising specialist journals"). Counting
# them as fragmentation or contamination punishes a real finding, so per-taste counts are restricted
# to TASTE_DIRECTIONS. It lives here rather than in grading.py because machinery must not import
# grading -- grading already imports machinery.
TASTE_DIRECTIONS = ("over", "under")


@dataclass
class GateContext:
    """What every paid gate needs to run: an API client, which models to use, and whether
    the cluster cache may serve a hit. Passed in rather than built per gate, so one run uses
    one configuration and a gate cannot quietly pick its own model."""
    client: object
    cluster_model: str
    consolidate_model: str
    cluster_prompt: str = ""
    consolidate_prompt: str = ""
    use_cache: bool = True


def scratch_db_path(name):
    """A throwaway database beside the user's data, named for the gate using it AND for this
    process.

    The pid is what makes two harness runs able to coexist. Without it every scenario driven by
    long_horizon shared one path, and build_db deletes the file on entry, so a second run would
    delete the first one's database mid-run -- which bites exactly when you want to try an
    experiment while a long sweep is going. Gates within one process still run sequentially and
    share the path harmlessly."""
    return config.DATA_DIR / f"_scratch_analysis_{name}_{os.getpid()}.db"


def cached_cluster(client, papers_block, n_flags, profile, model, use_cache, cluster_prompt):
    """Cluster output is a pure function of (cluster prompt, papers, profile, model), so
    cache it. Two reasons this matters: most iteration is on the CONSOLIDATE prompt, and
    re-running cluster each time is pure waste; and pinning cluster output makes consolidate
    the ONLY variable, which is better science. Editing CLUSTER_PROMPT changes the key and
    correctly invalidates. Returns (text, cost, was_cached)."""
    key = hashlib.sha256("\x00".join(
        [cluster_prompt, papers_block, profile, model]).encode("utf-8")).hexdigest()[:16]
    path = config.ANALYSIS_HARNESS_CACHE_DIR / f"{key}.md"
    if use_cache and path.exists():
        return path.read_text(encoding="utf-8"), 0.0, True
    text, cost = PA.run_cluster_step(client, papers_block, n_flags, profile, model,
                                     prompt=cluster_prompt)
    config.ANALYSIS_HARNESS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text, cost, False


# ---------------------------------------------------------------------------
# Scratch world
# ---------------------------------------------------------------------------

def build_db(path, profile):
    """A brand-new database with only what the scenario needs. Not a copy of the live
    one -- total control, and no way to touch real data.

    `profile` is the synthetic profile the scenario runs under. It is a parameter rather
    than a fixture constant because a scenario may need its own: testing that the judge
    is ignoring text the profile ALREADY states needs a profile that states it."""
    if path.exists():
        path.unlink()
    conn = DB.get_connection(path)
    profile_id = DB.get_or_create_profile(conn, profile, notes="harness synthetic profile")
    run_id = DB.find_or_create_scoring_run(
        conn, "curation", "harness-synthetic", "benchmark",
        profile_id=profile_id, judge_prompt_hash="harness")
    return conn, run_id, profile_id


def add_round_flags(conn, run_id, papers, flag_intended):
    """Insert this round's synthetic articles, their judge scores, and your flags.
    Records flag_id -> intended pattern, which is the ground truth everything else uses."""
    DB.insert_articles(conn, [{
        "pubmed_id": p["pmid"], "title": p["title"], "abstract": p["abstract"],
        "journal": p["journal"], "pub_date": "2026-01-15", "epub_date": None,
        "authors": [], "pub_types": [], "pages": None, "doi": None,
    } for p in papers])
    for p in papers:
        DB.insert_evaluation(conn, p["pmid"], run_id, p["judge_score"], rationale=p["rationale"])
        ev = conn.execute("SELECT id FROM evaluations WHERE pmid=? AND run_id=?",
                          (p["pmid"], run_id)).fetchone()["id"]
        flag_intended[DB.insert_flag(conn, ev, p["user_score"], note=p["note"] or None)] = p["intended"]


# ---------------------------------------------------------------------------
# Provenance graph -> quality metrics (no prose is ever read)
# ---------------------------------------------------------------------------

def pattern_intended(conn, flag_intended):
    """produced-pattern id -> Counter of the INTENDED patterns of the flags attached to it.
    This is the whole basis of grading: it says which ground-truth gaps each produced
    pattern actually drew from.

    A flag may carry SEVERAL intended patterns -- one paper can instantiate more than one
    taste (over-scored for its method while under-scored for its topic), and the machinery is
    meant to attach it to a pattern for each rather than fuse them into one chimera. Each
    label is counted, so a dual-nature flag supports both intended patterns it belongs to."""
    out = {}
    for pid, fid in conn.execute("SELECT pattern_id, flag_id FROM pattern_flags"):
        labels = flag_intended.get(fid, "?")
        for label in ((labels,) if isinstance(labels, str) else labels):
            out.setdefault(pid, Counter())[label] += 1
    return out


def patterns_by_flag(conn):
    """flag id -> the SET of produced patterns it is attached to. pattern_flags is many-to-many,
    so a paper instantiating two tastes should appear under a pattern for each; this is the view
    the dual-nature checks read (pattern_intended goes the other way)."""
    out = {}
    for pid, fid in conn.execute("SELECT pattern_id, flag_id FROM pattern_flags"):
        out.setdefault(fid, set()).add(pid)
    return out


def dominant_intended(counter):
    return counter.most_common(1)[0][0] if counter else None


def purity_of(counter):
    return counter.most_common(1)[0][1] / sum(counter.values()) if counter else 0.0


# ---------------------------------------------------------------------------
# One round
# ---------------------------------------------------------------------------

def run_round(conn, client, profile, cluster_model, consolidate_model, cluster_prompt,
              consolidate_prompt, use_cache=True):
    """One review session through the real pipeline: cluster the unattached flags,
    consolidate the candidates against the pattern memory, record the choices.
    `profile` is the scenario's own synthetic profile (see build_db)."""
    flags = DB.get_flags(conn, exclude_attached=True)
    papers_block, ordered = PA._format_papers(flags)
    open_patterns = DB.get_active_patterns(conn)
    closed_patterns = DB.get_patterns(conn, statuses=("incorporated", "rejected"))
    existing = PA._format_existing_patterns(
        open_patterns, closed_patterns,
        DB.get_pattern_examples(conn, [p["id"] for p in open_patterns + closed_patterns]))
    clusters, c1, hit = cached_cluster(client, papers_block, len(flags), profile,
                                       cluster_model, use_cache, cluster_prompt)
    if hit:
        print("  [cluster: cache hit -- $0.00]", flush=True)
    candidates, c2 = PA.run_consolidate_step(client, clusters, profile, existing,
                                           consolidate_model, prompt=consolidate_prompt)
    summary = PA._record_consolidation(conn, candidates, ordered)
    # `existing` is returned so the report can show EXACTLY what memory the model was
    # shown. When it fails to match a closed pattern, the first question is always "was
    # that closed pattern even in front of it?" -- without this you are guessing.
    return len(flags), candidates, summary, c1 + c2, existing


def apply_actions(conn, profile, actions, flag_intended, pattern_for, log):
    """The scripted human. Resolves an intended pattern to EVERY open produced pattern that
    covers it (via provenance, not by name) and records the decision on each.

    EVERY one, not the biggest. A gap can legitimately arrive as several patterns -- a taste
    ("over: correlational human neuroimaging") plus a different diagnosis of the same papers
    ("sharpen: the causal-mechanism phrase is being over-applied") -- and the grading blesses
    that split. Deciding only one left its siblings OPEN, and an open sibling is a perfectly
    good home for the gap when it returns: the model merges into it, which records a `carried`
    event rather than `recurred`, and the recurrence check goes red for behavior that was
    correct. The old `max(owned, key=total flags)` also ranked an impure pattern (3 B + 2 A)
    above a pure one (4 B), and ties were settled by second-resolution timestamps. Deciding all
    of them removes the ranking, the tiebreak and the open-sibling escape at once.

    It also matches what a person would do: having decided a taste, you clear every card that
    taste produced. Leaving one open forever is the treadmill this system exists to avoid.

    pattern_for[label] = {"action": ..., "ids": {pattern_id, ...}} -- the patterns later
    sessions check recurrence, merges and closed-status against. An EMPTY id set is recorded
    explicitly rather than omitted, so an unresolved decision fails LOUDLY downstream instead
    of passing vacuously.

    An action may carry an optional third element, ("incorporate", "A", "taste"), to decide only
    the taste patterns and leave meta ones open. Nothing uses it; it exists so the fixture can
    express that case without this function changing."""
    pp = pattern_intended(conn, flag_intended)
    for action, intended, *scope in actions:
        taste_only = bool(scope) and scope[0] == "taste"
        owned = [p for p in DB.get_active_patterns(conn)
                 if dominant_intended(pp.get(p["id"], Counter())) == intended
                 and (not taste_only or p["direction"] in TASTE_DIRECTIONS)]
        pattern_for[intended] = {"action": action, "ids": {p["id"] for p in owned}}
        if not owned:
            log(f"  [human] {action} {intended}: NO open pattern covers it -- UNRESOLVED "
                f"(every cross-session check for {intended} will fail)")
            continue
        for p in owned:
            if action == "incorporate":
                DB.add_pattern_event(conn, p["id"], "incorporated",
                                     profile_id=DB.get_or_create_profile(conn, profile))
            elif action == "reject":
                DB.add_pattern_event(conn, p["id"], "rejected", note="harness: scripted reject")
            else:
                DB.add_pattern_event(conn, p["id"], "carried", note="harness: scripted carry")
            log(f"  [human] {action} {intended} -> pattern {p['id'][:12]} "
                f"[{p['direction']}] {p['flag_count']} flags")


