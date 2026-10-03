"""
cli.py -- litcurator command line.

    litcurator run --start 2026-01-01 --end 2026-01-07 [--benchmark]
    litcurator status [--runs] [--flags] [--profiles] [--all] [--start ...] [--end ...]
    litcurator review

Thin dispatch over the pipeline (run), DB summaries (status), the Dash apps (review,
judge_workbench, analysis_workbench, the labelers), the offline error_analysis
suggester, and the two harnesses: judge_harness for the judge, analysis_harness for the
error-analysis machinery. Each subcommand is a thin wrapper over its module.
"""

import argparse
import hashlib
from datetime import datetime, timezone

from litcurator import pipeline, db_interface, profile_interface, prompt_interface
from litcurator.config import BEST_OF_RUNS, DOMAIN_THRESHOLD, SCORE_THRESHOLD


def _cmd_run(args):
    """--profile-file / --prompt-file score a window under a regime that is NOT the active
    one, without touching the live artifacts. That is how an edit gets measured: re-score the
    same papers under the draft and compare class means. Each combination is its own
    scoring_run, stamped with the profile and prompt that produced it."""
    import pathlib as _pl

    def _read(path):
        return _pl.Path(path).read_text(encoding="utf-8", errors="replace") if path else None

    try:
        pipeline.run(args.start, args.end, benchmark=args.benchmark,
                     final_test=args.final_test,
                     profile_text=_read(args.profile_file),
                     prompt_text=_read(args.prompt_file))
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
    -> surfaced. Read-only; counts by issue_date_iso (the same bucketing axis the
    domain/curation queries below use, so the funnel stays coherent). Blank range =
    all time."""
    where = ""
    params = []
    if start:
        where += " AND issue_date_iso >= ?"
        params.append(start)
    if end:
        where += " AND issue_date_iso <= ?"
        params.append(end)
    retrieved = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE 1=1" + where, params).fetchone()[0]
    passed = len(db_interface.get_articles_passing_domain_filter(
        conn, start, end, DOMAIN_THRESHOLD))
    judged = db_interface.latest_curation(conn, start, end)
    surfaced = sum(1 for it in judged if it["score"] >= SCORE_THRESHOLD)
    rng = f"{start or 'start'} .. {end or 'end'}"
    rows = [
        ("retrieved (issue date in range)", retrieved),
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
        note = f"   note: {f['note']}" if f.get("note") else ""
        title = (f.get("title") or "")[:58]
        print(f"  {f['delta']:>+6.2f}  {f['judge_score']:>5.2f} {f['user_score']:>5.2f}  "
              f"{f['pmid']:<10} {title}{note}")


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


def _cmd_error_analysis(args):
    from litcurator import error_analysis
    # Only pass overrides that were given, so error_analysis keeps its own defaults.
    overrides = {k: v for k, v in (("cluster_model", args.cluster_model),
                                   ("consolidate_model", args.consolidate_model)) if v}
    error_analysis.suggest_edits(start=args.start, end=args.end,
                                   persist=not args.dry_run,
                                   shuffle_seed=args.shuffle_candidates,
                                   include_attached=args.include_attached,
                                   reuse_clusters=args.reuse_clusters,
                                   best_of=args.best_of, pick_model=args.pick_model,
                                   **overrides)


def _cmd_pick_best(args):
    """Pick between consolidation rounds that already exist, without paying to re-run them.

    error_analysis already does this at the end of every round. This is for picking again --
    a different seed, to check the answer does not depend on presentation order -- or for
    comparing reports that were never part of one round."""
    from litcurator import consolidation_picker

    model = args.model or consolidation_picker.PICKER_MODEL
    verdict, rounds, cost = consolidation_picker.pick_best(
        args.reports, start=args.start, end=args.end, model=model, seed=args.seed)
    by_label = {label: path for label, path, _ in rounds}
    print("\nPresented as: " + ", ".join(f"{lb}={p.name}" for lb, p, _ in rounds))
    if verdict.get("none_are_good"):
        print("\n*** The picker judged every round weak. ***")
    print(f"\nRanking: {' > '.join(verdict['ranking'])}")
    for entry in verdict["assessments"]:
        print(f"  {entry['run']}: {entry['worst_problem']}")
    print(f"\nWhy the winner: {verdict['why_the_winner']}")
    print(f"\nWINNER: {by_label[verdict['ranking'][0]]}")
    print(f"[cost: ${cost:.4f}]")
    print(f"Saved to {consolidation_picker.write_verdict(verdict, rounds, cost, model)}")


def _stamped_or_active(conn, stamps, key, table, register_active):
    """The artifact id a report names, or today's active one with a printed reason.

    Never silent: a round recorded against the wrong profile or prompt is unfixable later and
    looks exactly like a correct one, so every fallback announces itself."""
    short = stamps.get(key)
    if short:
        resolved = db_interface.resolve_short_id(conn, table, short)
        if resolved:
            return resolved
        print(f"WARNING: this report ran under {key} {short}, which is not in the database. "
              f"Recording against the ACTIVE {key} instead -- the stamp is in the report if "
              f"you need to repair it.")
    else:
        print(f"NOTE: this report predates artifact stamps, so its {key} is unknown. "
              f"Recording against the active one.")
    return register_active()


def _cmd_promote_suggestions(args):
    """Record a run you already have, from its report, instead of paying to re-roll one.

    WHY. A ROUND is unstable enough that quality swings hard on identical input -- on
    2026-08-30 a run the user called a shit show was followed immediately by one he called
    amazing, with nothing changed between them. That makes best-of-N the sensible workflow:
    dry-run two or three times, read them, keep the one you like. Which only works if liking
    one lets you keep it, and until now a dry run threw its parsed candidates away.

    The report is the record -- no second file, no JSON sidecar. This parses it back and hands
    the candidates to the SAME `_record_consolidation` the live path uses, so nothing can
    diverge between what you previewed and what gets written.

    THE GUARD THAT MATTERS. The [N] paper numbers index into the ordered flag list, which
    `_format_papers` derives deterministically from the unattached flags. Re-derive it and the
    numbering reproduces exactly -- but only if the flag set has not moved. So the report's own
    flag count is compared against the live one, and a mismatch refuses rather than silently
    attaching every pattern to the wrong papers."""
    from litcurator import db_interface, error_analysis, profile_interface
    from litcurator import analysis_prompt_interface

    candidates, n_then = error_analysis.parse_consolidation_md(args.report)
    conn = db_interface.get_connection()
    try:
        flags = db_interface.get_flags(conn, start=args.start, end=args.end,
                                       exclude_attached=True)
        n_now = len(flags)
        if n_then is not None and n_then != n_now:
            print(f"REFUSED: that report clustered {n_then} flags; {n_now} are unattached now.\n"
                  f"The [N] paper numbers index into the ordered flag list, so recording it "
                  f"against a different set would attach every pattern to the wrong papers.")
            raise SystemExit(1)

        _papers, ordered_flags = error_analysis._format_papers(flags)
        print(f"{args.report}\n  {len(candidates)} candidates over {n_now} flags")
        if not args.yes:
            if input("Type 'record' to write this into the pattern memory: ").strip().lower() \
                    != "record":
                print("Cancelled -- nothing was written.")
                return

        # Same provenance as a live run: the analysis_run row exists before any pattern, and
        # the prompt is registered (content-addressed, so an unchanged one adds no row).
        #
        # PREFER THE REPORT'S OWN STAMPS over whatever is active today. This round ran under
        # particular artifacts, possibly days ago, and registering today's would attach it to
        # files it never read -- the same class of untruth as stamping a profile version for a
        # prompt-blamed pattern. Reports written before 2026-09-19 carry no stamps, so they
        # fall back to active and SAY SO rather than substituting silently.
        stamps = error_analysis.parse_report_stamps(args.report)
        analysis_prompt_id = _stamped_or_active(
            conn, stamps, "analysis prompt", "prompts",
            lambda: db_interface.get_or_create_prompt(
                conn, analysis_prompt_interface.load_active(), kind="analysis"))
        profile_id = _stamped_or_active(
            conn, stamps, "profile", "profiles",
            lambda: db_interface.get_or_create_profile(
                conn, profile_interface.load_active()))
        judge_prompt_id = _stamped_or_active(
            conn, stamps, "judge prompt", "prompts",
            lambda: db_interface.get_or_create_prompt(
                conn, prompt_interface.load_active(), kind="judge"))

        run_id = db_interface.create_analysis_run(
            conn,
            analysis_prompt_id,
            error_analysis.DEFAULT_CLUSTER_MODEL,
            error_analysis.DEFAULT_CONSOLIDATE_MODEL,
            profile_id=profile_id,
            date_start=args.start, date_end=args.end, n_flags=n_now, cost_usd=0.0,
            judge_prompt_id=judge_prompt_id,
            # No pick prompt, and this is not an omission: promote_suggestions IS the human
            # overriding the picker, so stamping one would assert a machine choice that never
            # happened.
            pick_prompt_id=None)
        summary = error_analysis._record_consolidation(
            conn, candidates, ordered_flags, analysis_run_id=run_id)
        print(error_analysis._summary_line(summary))
        print(f"Recorded from {args.report} -- no model calls, $0.")
    finally:
        conn.close()


def _cmd_undo_error_analysis(args):
    """Undo the most recent error_analysis (or promote_suggestions) recording.

    LATEST ONLY, by design: the most recent run is the only one guaranteed to have nothing
    built on top of it, so undoing it can never orphan a later round's merges. Run it again
    to peel the previous one (LIFO). It stops by itself at any round whose patterns carry
    your own decisions -- you cannot peel past your own curation.

    Deletes only what that run wrote: its minted patterns, the attachments its merges added,
    the events it fired, and the run row. Flags are never touched -- every flag the run had
    attached returns to the unattached pool, so a re-run rebuilds from the same evidence."""
    from litcurator import db_interface

    conn = db_interface.get_connection()
    try:
        run = db_interface.latest_analysis_run(conn)
        if run is None:
            print("No analysis runs recorded -- nothing to undo.")
            return
        m = db_interface.analysis_run_manifest(conn, run["id"])
        print(f"Latest analysis run: {run['created_at']}  |  "
              f"{run['n_flags']} flags  |  ${run['cost_usd'] or 0:.4f}")
        if m["minted"]:
            print(f"\nWould delete {len(m['minted'])} pattern(s) this run minted:")
            for p in m["minted"]:
                print(f"  - [{p['direction']}] {p['name']}")
        if m["foreign_attaches"]:
            print("\nWould detach what its merges added to earlier patterns:")
            for r in m["foreign_attaches"]:
                print(f"  - {r['pmid']} out of '{r['name']}'")
        if m["foreign_events"]:
            print("\nWould remove the events it fired on earlier patterns:")
            for r in m["foreign_events"]:
                print(f"  - {r['event']} on '{r['name']}'")
        if m["blockers"]:
            print("\nREFUSED -- this run has your own work on it:")
            for b in m["blockers"]:
                print(f"  - {b}")
            print("Undo would eat those decisions. If you truly want this round gone, "
                  "unwind your curation first (it is yours, not the round's).")
            raise SystemExit(1)
        if not args.yes:
            if input("\nType 'undo' to delete all of the above: ").strip().lower() != "undo":
                print("Cancelled -- nothing was deleted.")
                return
        counts = db_interface.delete_analysis_run(conn, run["id"])
        pool = len(db_interface.get_flags(conn, exclude_attached=True))
        print(f"Undone: {counts['patterns']} patterns, {counts['attachments']} attachments, "
              f"{counts['events']} events removed. {pool} flags now unattached.")
    finally:
        conn.close()


def _cmd_reset_patterns(args):
    """Drop the pattern memory and NOTHING ELSE, so a bad consolidation round can be redone.

    WHY THIS EXISTS. The obvious way to undo a round is to restore a database snapshot, and on
    2026-08-30 that quietly cost an afternoon of flag edits: the backup predated them, so
    rolling back the patterns rolled back the notes too, and the flag count going 38 -> 33 was
    the only sign. A snapshot is the wrong granularity -- it undoes everything since, not the
    thing you meant.

    This touches four tables and no others. Flags, human labels, evaluations, articles,
    profiles and prompts are all untouched by construction, so the mistake above cannot
    recur. Re-running error_analysis afterwards starts the round again from the same flags."""
    from litcurator import db_interface

    conn = db_interface.get_connection()
    try:
        before = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("patterns", "pattern_flags", "pattern_events", "analysis_runs")}
        n_flags = conn.execute("SELECT COUNT(*) FROM flags").fetchone()[0]
        if not before["patterns"] and not before["analysis_runs"]:
            print("Nothing to reset -- the pattern memory is already empty.")
            return
        print("This will delete the pattern memory:")
        for t, n in before.items():
            print(f"  {t:<16} {n}")
        print(f"\nFlags ({n_flags} rows), labels, evaluations and profiles are NOT touched.")
        if not args.yes:
            if input("Type 'reset' to confirm: ").strip().lower() != "reset":
                print("Cancelled -- nothing was deleted.")
                return
        # Children before parents: pattern_flags and pattern_events both reference patterns,
        # and patterns references analysis_runs.
        for t in ("pattern_flags", "pattern_events", "patterns", "analysis_runs"):
            conn.execute(f"DELETE FROM {t}")
        conn.commit()
        unattached = len(db_interface.get_flags(conn, exclude_attached=True))
        print(f"Pattern memory cleared. {n_flags} flag rows kept; {unattached} papers are "
              f"unattached and ready to re-cluster.")
    finally:
        conn.close()


def _cmd_judge_workbench(args):
    from litcurator.apps import judge_workbench
    judge_workbench.run_app()


def _cmd_analysis_workbench(args):
    from litcurator.apps import analysis_workbench
    analysis_workbench.run_app()


def _cmd_judge_harness(args):
    from pathlib import Path
    from litcurator import judge_harness, pipeline
    if args.dry_run:
        print(judge_harness.dry_run())
        return
    # The harness scores under whatever judge.MODEL is -- say so before spending.
    pipeline.print_model_banner(benchmark=True)
    prompt_text = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else None
    profile_text = Path(args.profile).read_text(encoding="utf-8") if args.profile else None
    results, prompt_fp, profile_fp = judge_harness.run_tests(
        prompt_text=prompt_text,
        profile_text=profile_text,
        progress=lambda done, total: print(f"  scored {done}/{total}", flush=True))
    report = judge_harness.format_report(results, prompt_fp, profile_fp)
    print(report)
    path = judge_harness.write_report(report + "\n" + judge_harness.format_rationales(results))
    print(f"\nsaved to {path}")
    print("\n" + judge_harness.verdict_line(results))   # last, so the answer needs no scrolling


def _cmd_analysis_harness(args):
    from pathlib import Path
    from litcurator import analysis_harness as AH
    try:
        gates = AH.select_gates(args.gate)
    except KeyError as e:
        print(e)
        raise SystemExit(2)

    read = lambda p: Path(p).read_text(encoding="utf-8") if p else None
    cluster_prompt, consolidate_prompt = read(args.cluster_prompt), read(args.consolidate_prompt)
    drafts = [Path(p).name for p in (args.cluster_prompt, args.consolidate_prompt) if p]

    if args.dry_run:
        # A draft still gets applied first, so the dry run's fingerprint answers "did I paste
        # the right path". Applying it must not RUN anything -- see apply_draft_prompts.
        text, ok = AH.dry_run(gates, cluster_prompt, consolidate_prompt)
        print(text)
        raise SystemExit(0 if ok else 2)

    if any(g not in AH.FREE_GATES for g in gates):
        print(f"models: cluster={AH.PA.DEFAULT_CLUSTER_MODEL}  "
              f"consolidate={AH.PA.DEFAULT_CONSOLIDATE_MODEL}", flush=True)
    results, cluster_fp, consolidate_fp = AH.run_gates(
        gates, cluster_prompt=cluster_prompt, consolidate_prompt=consolidate_prompt,
        progress=lambda gate, done, total: print(f"  [{done}/{total}] {gate}", flush=True))

    report = AH.format_report(results, cluster_fp, consolidate_fp, drafts=drafts)
    print("\n" + report)
    path = AH.write_report(report + "\n" + AH.format_transcripts(results),
                           selector=args.gate)
    print(f"\nsaved to {path}")
    # Last, so the answer needs no scrolling. It also says when paid groups were skipped.
    print("\n" + AH.verdict_line(results, n_skipped_groups=len(gates) - len(results)))
    raise SystemExit(AH.exit_code(results, gates))


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

    run_p.add_argument("--profile-file", default=None, metavar="PATH",
                       help="score under this profile instead of the active one (a file from "
                            "profile/versions/, or a draft). The live profile is untouched; "
                            "the run is stamped with whichever profile actually produced it.")
    run_p.add_argument("--prompt-file", default=None, metavar="PATH",
                       help="score under this judge prompt instead of the active one. Pair it "
                            "with --profile-file to re-score a window under any past or draft "
                            "regime and measure what an edit actually did.")

    review_p = sub.add_parser("review", help="launch the review feed (browse judged papers, flag)")
    review_p.add_argument("--start", default=None, help="pre-fill the pub-date filter start (YYYY-MM-DD)")
    review_p.add_argument("--end", default=None, help="pre-fill the pub-date filter end (YYYY-MM-DD)")
    review_p.set_defaults(func=_cmd_review)

    pa_p = sub.add_parser("error_analysis",
                           help="cluster flags -> consolidate into tracked profile patterns")
    pa_p.add_argument("--start", default=None, help="scope flags to pub dates >= this (YYYY-MM-DD)")
    pa_p.add_argument("--end", default=None, help="scope flags to pub dates <= this (YYYY-MM-DD)")
    pa_p.add_argument("--shuffle-candidates", type=int, default=None, metavar="SEED",
                      help="DIAGNOSTIC: present cluster's candidates to consolidate in a "
                           "shuffled order, so `rank` can be correlated against the order they "
                           "were shown in. On the first real run rank tracked cluster's own "
                           "order at rho=0.79, which would make it anchoring rather than "
                           "judgement; this tells the two apart. The permutation is printed and "
                           "saved in the report.")
    pa_p.add_argument("--reuse-clusters", default=None, metavar="REPORT.md",
                      help="skip step 1 and reuse the cluster output from a previous suggestions "
                           "report. Saves the expensive half of the run, and -- the real point -- "
                           "removes cluster's own stochasticity from a comparison, so two "
                           "consolidate models see the SAME candidates instead of two different "
                           "draws. Refuses if the flag count has changed since that report.")
    pa_p.add_argument("--include-attached", action="store_true",
                      help="COMPARISON MODE: cluster every flag in the window, not just the "
                           "unattached ones, so the same evidence can be re-run under a different "
                           "prompt or model without resetting the database. Requires --dry-run; "
                           "recording it would mint duplicate patterns over papers that already "
                           "have them.")
    pa_p.add_argument("--dry-run", action="store_true",
                      help="write the suggestions markdown but do NOT persist patterns")
    pa_p.add_argument("--cluster-model", default=None,
                      help="override the cluster (recall) model")
    pa_p.add_argument("--consolidate-model", default=None,
                      help="override the consolidate (choice) model, e.g. claude-opus-4-8")
    pa_p.add_argument("--best-of", type=int, default=None, metavar="N",
                      help=f"run the whole round N times and let the picker choose which to "
                           f"record (default {BEST_OF_RUNS}, from config). Consolidation "
                           f"is unreliable enough that one round is a lottery. 1 runs a single "
                           f"round and skips the picker, which needs no pick prompt.")
    pa_p.add_argument("--pick-model", default=None,
                      help="override the model that picks between rounds")
    pa_p.set_defaults(func=_cmd_error_analysis)

    pb_p = sub.add_parser("pick_best",
                          help="pick between consolidation rounds that already exist. "
                               "error_analysis does this itself; use this to re-pick with a "
                               "different seed, or to compare reports from separate rounds.")
    pb_p.add_argument("reports", nargs="+",
                      help="a round directory, or two or more suggestions reports")
    pb_p.add_argument("--start", default=None, help="the window the reports were run over")
    pb_p.add_argument("--end", default=None)
    pb_p.add_argument("--model", default=None, help="override the picker model")
    pb_p.add_argument("--seed", type=int, default=None,
                      help="fix the presentation order, to check the answer does not depend "
                           "on it")
    pb_p.set_defaults(func=_cmd_pick_best)

    ps_p = sub.add_parser("promote_suggestions",
                          help="record a dry run you liked, from its report, with no model "
                               "calls. A round is unstable enough that quality swings on "
                               "identical input, so the workflow is: dry-run two or three "
                               "times, read them, promote the good one.")
    ps_p.add_argument("report", help="path to a suggestions markdown report")
    ps_p.add_argument("--start", default=None, help="the window the report was run over")
    ps_p.add_argument("--end", default=None)
    ps_p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ps_p.set_defaults(func=_cmd_promote_suggestions)

    up_p = sub.add_parser("undo_error_analysis",
                          help="undo the most recent error_analysis recording: delete the "
                               "patterns it minted, the attachments its merges added, and the "
                               "events it fired. Latest run only; repeat to peel further back. "
                               "Refuses once it reaches patterns you have curated. Flags are "
                               "untouched and return to the unattached pool.")
    up_p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    up_p.set_defaults(func=_cmd_undo_error_analysis)

    rp_p = sub.add_parser("reset_patterns",
                          help="delete the pattern memory (patterns / provenance / events / "
                               "analysis runs) so a bad consolidation round can be redone. "
                               "Flags, labels and profiles are untouched -- restoring a whole "
                               "database snapshot is the wrong granularity and has already "
                               "cost a round of flag edits once.")
    rp_p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    rp_p.set_defaults(func=_cmd_reset_patterns)

    pw_p = sub.add_parser("judge_workbench",
                           help="launch the judge workbench (review patterns, edit the profile, set active)")
    pw_p.set_defaults(func=_cmd_judge_workbench)

    aw_p = sub.add_parser("analysis_workbench",
                           help="launch the analysis workbench (edit the cluster and consolidate "
                                "prompts, preview a draft on your flags)")
    aw_p.set_defaults(func=_cmd_analysis_workbench)


    jh_p = sub.add_parser("judge_harness",
                           help="run the judge harness -- fast floor-of-competence gate (tests prompt + profile)")
    jh_p.add_argument("--prompt", default=None,
                      help="path to a draft prompt to test (default: active prompt on disk)")
    jh_p.add_argument("--profile", default=None,
                      help="path to a draft profile to test (default: active profile on disk)")
    jh_p.add_argument("--dry-run", action="store_true",
                      help="verify the fixture pmids resolve and list cases, without scoring")
    jh_p.set_defaults(func=_cmd_judge_harness)

    ah_p = sub.add_parser("analysis_harness",
                           help="run the error-analysis gates (free ones first, then the paid ones)")
    ah_p.add_argument("gate", nargs="?", default=None,
                      help="'quick' for the free gates only, or one gate by name "
                           "(default: every gate)")
    ah_p.add_argument("--cluster-prompt", default=None, metavar="FILE",
                      help="path to a draft cluster prompt to test (default: the live one)")
    ah_p.add_argument("--consolidate-prompt", default=None, metavar="FILE",
                      help="path to a draft consolidate prompt to test (default: the live one)")
    ah_p.add_argument("--dry-run", action="store_true",
                      help="list the gates and check preconditions, without running or spending")
    ah_p.set_defaults(func=_cmd_analysis_harness)

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
