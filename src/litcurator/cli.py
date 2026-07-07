"""
cli.py -- litcurator command line.

    litcurator run --start 2026-01-01 --end 2026-01-07 [--benchmark]
    litcurator status [--runs] [--flags] [--profiles] [--all] [--start ...] [--end ...]
    litcurator review

Thin dispatch over pipeline.run, a DB summary, and the review feed. The
consolidation commands (suggest, edit) are added as their modules graduate.
"""

import argparse
import hashlib
from datetime import datetime, timezone

from litcurator import pipeline, db_interface, profile_interface, prompt_interface
from litcurator.config import DOMAIN_THRESHOLD, SCORE_THRESHOLD


def _cmd_run(args):
    try:
        pipeline.run(args.start, args.end, benchmark=args.benchmark,
                     final_test=args.final_test)
    except pipeline.LockedTestSetError as e:
        print(f"\nBLOCKED: {e}\n")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# status sections
# ---------------------------------------------------------------------------

def _print_overview(conn):
    n_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    labels = db_interface.count_human_labels(conn)
    n_profiles = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    n_runs = conn.execute("SELECT COUNT(*) FROM scoring_runs").fetchone()[0]
    n_evals = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    n_flags = conn.execute("SELECT COUNT(*) FROM flags").fetchone()[0]
    seed = db_interface.get_seed_profile(conn)
    seed_str = f"seed {seed['id'][:12]}" if seed else "(no seed)"
    print(f"articles: {n_articles}   human labels: {labels['total_labeled']} "
          f"({labels['relevant']} relevant, {labels['curation_labeled']} curation)   "
          f"profiles: {n_profiles}   {seed_str}")
    print(f"runs: {n_runs}   evaluations: {n_evals}   flags: {n_flags}")
    locked = db_interface.locked_test_pmids()
    seal_str = f"{len(locked)} pmids sealed" if locked else "NOT sealed (run: litcurator seal_test_set)"
    print(f"locked test set: {seal_str}")
    n_prompts = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
    active_prompt = prompt_interface.active_version_id()
    prompt_str = (f"{active_prompt}  ({n_prompts} version{'s' if n_prompts != 1 else ''})"
                  if active_prompt else f"not seeded yet (default in code, {n_prompts} in db)")
    print(f"judge prompt: {prompt_str}")


def _run_duration(created_at, completed_at):
    """Wall time of a finished run from its stored timestamps, or '-'.
    Read-only: subtracts two columns already on scoring_runs, writes nothing."""
    if not (created_at and completed_at):
        return "-"
    try:
        c = datetime.fromisoformat(created_at)
        d = datetime.fromisoformat(completed_at)
        if c.tzinfo:
            c = c.astimezone(timezone.utc).replace(tzinfo=None)
        if d.tzinfo:
            d = d.astimezone(timezone.utc).replace(tzinfo=None)
        secs = (d - c).total_seconds()
        return pipeline._fmt_dur(secs) if secs >= 0 else "-"
    except ValueError:
        return "-"


def _print_runs(conn, limit=15):
    runs = conn.execute("""
        SELECT r.stage, r.mode, r.date_start, r.date_end, r.created_at, r.completed_at, r.cost_usd,
               (SELECT COUNT(*) FROM evaluations e WHERE e.run_id = r.id) AS n
        FROM scoring_runs r ORDER BY r.created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    if not runs:
        print("\nruns: none yet")
        return
    print("\nruns (newest first):")
    print(f"  {'stage':<8} {'mode':<9} {'window':<26} {'papers':<7} {'state':<9} {'time':<9} cost")
    for r in runs:
        state = "done" if r["completed_at"] else "in-flight"
        cost = f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "-"
        dur = _run_duration(r["created_at"], r["completed_at"])
        window = f"{r['date_start']}..{r['date_end']}"
        print(f"  {r['stage']:<8} {r['mode']:<9} {window:<26} {r['n']:<7} {state:<9} {dur:<9} {cost}")


def _print_funnel(conn, start=None, end=None):
    """The pipeline funnel over a window: retrieved -> domain-passed -> judged
    -> surfaced. Read-only; counts by pub_date_iso. Blank range = all time."""
    where = ""
    params = []
    if start:
        where += " AND pub_date_iso >= ?"
        params.append(start)
    if end:
        where += " AND pub_date_iso <= ?"
        params.append(end)
    retrieved = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE 1=1" + where, params).fetchone()[0]
    passed = len(db_interface.get_articles_passing_domain_filter(
        conn, start, end, DOMAIN_THRESHOLD))
    judged = db_interface.latest_curation(conn, start, end)
    surfaced = sum(1 for it in judged if it["score"] >= SCORE_THRESHOLD)
    rng = f"{start or 'start'} .. {end or 'end'}"
    rows = [
        ("retrieved (pub date in range)", retrieved),
        (f"passed domain filter (>= {DOMAIN_THRESHOLD:.1f})", passed),
        ("judged", len(judged)),
        (f"surfaced (judge score >= {SCORE_THRESHOLD:.1f})", surfaced),
    ]
    print(f"\nfunnel ({rng}):")
    for label, n in rows:
        print(f"  {label:<34} {n:>6}")


def _print_flags(conn, start=None, end=None):
    flags = db_interface.get_flags(conn, start=start, end=end)
    if not flags:
        print("\nflags: none in range")
        return
    print(f"\nflags ({len(flags)}, latest per paper, by |delta|):")
    print(f"  {'delta':>6}  {'judge':>5} {'you':>5}  {'pmid':<10} title")
    for f in sorted(flags, key=lambda x: abs(x["delta"]), reverse=True):
        ing = "  (ingested)" if f.get("ingested_to_profile_id") else ""
        note = f"   note: {f['note']}" if f.get("note") else ""
        title = (f.get("title") or "")[:58]
        print(f"  {f['delta']:>+6.2f}  {f['judge_score']:>5.2f} {f['your_score']:>5.2f}  "
              f"{f['pmid']:<10} {title}{ing}{note}")


def _print_profiles(conn):
    profs = conn.execute(
        "SELECT id, parent_id, created_at, length(content) AS n, notes "
        "FROM profiles ORDER BY created_at").fetchall()
    if not profs:
        print("\nprofiles: none yet")
        return
    active_id = None
    if profile_interface.exists():
        active_id = hashlib.sha256(profile_interface.load_active().encode("utf-8")).hexdigest()
    print(f"\nprofiles ({len(profs)}, oldest first):")
    for p in profs:
        tags = []
        if p["parent_id"] is None:
            tags.append("seed")
        if p["id"] == active_id:
            tags.append("active")
        tag = "  [" + ",".join(tags) + "]" if tags else ""
        notes = f"   {p['notes']}" if p["notes"] else ""
        print(f"  {p['id'][:12]}  {(p['created_at'] or '')[:19]}  {p['n']:>5} chars{tag}{notes}")


def _cmd_status(args):
    show_funnel = args.funnel or args.all
    show_runs = args.runs or args.all
    show_flags = args.flags or args.all
    show_profiles = args.profiles or args.all
    conn = db_interface.get_connection()
    try:
        _print_overview(conn)
        if show_funnel:
            _print_funnel(conn, args.start, args.end)
        if show_runs:
            _print_runs(conn)
        if show_flags:
            _print_flags(conn, args.start, args.end)
        if show_profiles:
            _print_profiles(conn)
        if not (show_funnel or show_runs or show_flags or show_profiles):
            print("\n(detail: --funnel  --runs  --flags  --profiles  --all"
                  "   [--start/--end scope funnel + flags])")
    finally:
        conn.close()


def _cmd_review(args):
    from litcurator.apps import review_feed   # lazy: don't import dash for run/status
    review_feed.run_app(start=args.start, end=args.end)


def _cmd_profile_analysis(args):
    from litcurator import profile_analysis
    profile_analysis.suggest_edits(start=args.start, end=args.end)


def _cmd_profile_workbench(args):
    from litcurator.apps import profile_workbench
    profile_workbench.run_app()


def _cmd_prompt_workbench(args):
    from litcurator.apps import prompt_workbench
    prompt_workbench.run_app()


def _cmd_judge_harness(args):
    from pathlib import Path
    from litcurator import judge_harness
    if args.dry_run:
        print(judge_harness.dry_run())
        return
    prompt_text = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else None
    results, prompt_fp, profile_fp = judge_harness.run_tests(prompt_text=prompt_text)
    report = judge_harness.format_report(results, prompt_fp, profile_fp)
    print(report)
    path = judge_harness.write_report(report + "\n" + judge_harness.format_rationales(results))
    print(f"\nsaved to {path}")


def _cmd_label_relevance(args):
    from litcurator.apps import relevance_labeler
    relevance_labeler.run_app(start=args.start, end=args.end)


def _cmd_label_curation(args):
    from litcurator.apps import curation_labeler
    curation_labeler.run_app()


def _cmd_seal_test_set(args):
    from litcurator.config import LOCKED_TEST_PMIDS_FILE
    conn = db_interface.get_connection()
    try:
        if LOCKED_TEST_PMIDS_FILE.exists() and not args.force:
            existing = db_interface.locked_test_pmids()
            print(f"Already sealed: {len(existing)} pmids at {LOCKED_TEST_PMIDS_FILE}")
            print("The held-out set is frozen. Re-seal with --force ONLY if development "
                  "has not yet started.")
            return
        pmids = db_interface.freeze_locked_test_set(conn, overwrite=args.force)
    finally:
        conn.close()
    print(f"Sealed {len(pmids)} labeled pmids as the locked test set.")
    print(f"Written to {LOCKED_TEST_PMIDS_FILE}")
    print("Development benchmark/label queries now subtract this set by construction.")


def _cmd_backfill_pages(args):
    from litcurator import pipeline
    conn = db_interface.get_connection()
    try:
        if args.labeled_only:
            # final_test=True: pages are neutral bibliographic metadata, safe to pull
            # for November too (and wanted, so the test set matches live judge input).
            articles = db_interface.labeled_articles(conn, args.start, args.end,
                                                     relevant=None, final_test=True)
        else:
            articles = db_interface.articles_in_range(conn, args.start, args.end)
        missing = [a for a in articles if not a.get("pages")]
        if args.limit:
            missing = missing[:args.limit]
        print(f"{len(missing)} articles missing pages in scope -- fetching...")
        if not missing:
            print("nothing to do.")
            return
        checked, filled = pipeline.backfill_pages(conn, missing)
        print(f"done: {filled} page ranges filled; {checked - filled} have none "
              f"(electronic-only / article-number; re-checkable next run).")
    finally:
        conn.close()


def _cmd_prepare_labeling(args):
    import json
    from litcurator.config import LABELING_QUEUE_FILE
    months = [m.strip() for m in args.months.split(",")]
    conn = db_interface.get_connection()
    try:
        pmids = db_interface.sample_unlabeled_by_month(
            conn, months, n_per_month=args.n_per_month, seed=args.seed
        )
    finally:
        conn.close()
    LABELING_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"months": months, "n_per_month": args.n_per_month, "pmids": pmids}
    LABELING_QUEUE_FILE.write_text(json.dumps(payload, indent=2))
    print(f"Selected {len(pmids)} articles ({args.n_per_month}/month across {len(months)} months)")
    print(f"Saved to {LABELING_QUEUE_FILE}")
    print("Run `litcurator label_relevance` to start labeling.")


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="litcurator",
        description="litcurator: personalized PubMed curation",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="retrieve -> domain filter -> judge over a date range")
    run_p.add_argument("--start", required=True, help="ISO start date, YYYY-MM-DD")
    run_p.add_argument("--end", required=True, help="ISO end date, YYYY-MM-DD")
    run_p.add_argument("--benchmark", action="store_true",
                       help="judge the human-labeled set in the window (skip retrieve + domain filter)")
    run_p.add_argument("--final-test", action="store_true",
                       help="unlock the held-out November 2025 test set (spends it -- use once, ever)")
    run_p.set_defaults(func=_cmd_run)

    status_p = sub.add_parser("status", help="database summary (flags select detail sections)")
    status_p.add_argument("--funnel", action="store_true",
                          help="retrieved -> domain -> judged -> surfaced over --start/--end")
    status_p.add_argument("--runs", action="store_true", help="list recent scoring runs")
    status_p.add_argument("--flags", action="store_true", help="list flags (your corrections)")
    status_p.add_argument("--profiles", action="store_true", help="show the profile lineage")
    status_p.add_argument("--all", action="store_true", help="show every section")
    status_p.add_argument("--start", default=None, help="scope --funnel/--flags to pub dates >= this")
    status_p.add_argument("--end", default=None, help="scope --funnel/--flags to pub dates <= this")
    status_p.set_defaults(func=_cmd_status)

    review_p = sub.add_parser("review", help="launch the review feed (browse judged papers, flag)")
    review_p.add_argument("--start", default=None, help="pre-fill the pub-date filter start (YYYY-MM-DD)")
    review_p.add_argument("--end", default=None, help="pre-fill the pub-date filter end (YYYY-MM-DD)")
    review_p.set_defaults(func=_cmd_review)

    pa_p = sub.add_parser("profile_analysis",
                           help="cluster flags -> ranked profile-edit suggestions")
    pa_p.add_argument("--start", default=None, help="scope flags to pub dates >= this (YYYY-MM-DD)")
    pa_p.add_argument("--end", default=None, help="scope flags to pub dates <= this (YYYY-MM-DD)")
    pa_p.set_defaults(func=_cmd_profile_analysis)

    pw_p = sub.add_parser("profile_workbench",
                           help="launch the profile workbench (review suggestions, edit, set active)")
    pw_p.set_defaults(func=_cmd_profile_workbench)

    ptw_p = sub.add_parser("prompt_workbench",
                            help="launch the prompt workbench (edit + version the judge prompt)")
    ptw_p.set_defaults(func=_cmd_prompt_workbench)

    jh_p = sub.add_parser("judge_harness",
                           help="run the judge harness -- fast floor-of-competence gate (tests prompt + profile)")
    jh_p.add_argument("--prompt", default=None,
                      help="path to a draft prompt to test (default: active prompt on disk)")
    jh_p.add_argument("--dry-run", action="store_true",
                      help="verify the fixture pmids resolve and list cases, without scoring")
    jh_p.set_defaults(func=_cmd_judge_harness)

    lr_p = sub.add_parser("label_relevance",
                           help="launch the relevance labeler (relevant 0/1, date-masked)")
    lr_p.add_argument("--start", default="2025-01-01", metavar="YYYY-MM-DD")
    lr_p.add_argument("--end", default="2025-11-30", metavar="YYYY-MM-DD")
    lr_p.set_defaults(func=_cmd_label_relevance)

    lc_p = sub.add_parser("label_curation",
                           help="launch the curation labeler (rating 0-5 on relevant=1 articles)")
    lc_p.set_defaults(func=_cmd_label_curation)

    st_p = sub.add_parser("seal_test_set",
                           help="freeze the November held-out pmid set (run once, before development)")
    st_p.add_argument("--force", action="store_true",
                      help="re-seal even if a seal exists (only before development starts)")
    st_p.set_defaults(func=_cmd_seal_test_set)

    bp_p = sub.add_parser("backfill_pages",
                           help="fetch + store page ranges for articles missing one (re-checkable)")
    bp_p.add_argument("--start", default=None, help="scope to pub dates >= this (YYYY-MM-DD)")
    bp_p.add_argument("--end", default=None, help="scope to pub dates <= this (YYYY-MM-DD)")
    bp_p.add_argument("--labeled-only", action="store_true",
                      help="only the human-labeled (benchmark) set")
    bp_p.add_argument("--limit", type=int, default=None, help="cap how many to fetch (for testing)")
    bp_p.set_defaults(func=_cmd_backfill_pages)

    pl_p = sub.add_parser("prepare_labeling",
                           help="pre-select a balanced unlabeled sample for a labeling round")
    pl_p.add_argument("--months", default="2025-01,2025-03,2025-05,2025-07,2025-09,2025-11",
                      help="comma-separated YYYY-MM month prefixes to sample from")
    pl_p.add_argument("--n-per-month", type=int, default=130,
                      help="articles to select per month (default 130 -> 780 total across 6 months)")
    pl_p.add_argument("--seed", type=int, default=42,
                      help="random seed for reproducibility (default 42)")
    pl_p.set_defaults(func=_cmd_prepare_labeling)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
