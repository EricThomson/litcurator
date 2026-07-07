"""
pipeline.py -- THE litcurator pipeline: retrieve -> domain filter -> judge.

The core engine of litcurator in one module. Give it a date range and it gives
you papers -- fetched, screened, and scored against your profile, sitting in the
DB ready to read. If you just want papers, run this.

Each stage is a scoring run, append-only. Re-invoking the same window+profile
RESUMES the run: only un-evaluated pmids are scored, so you never re-pay for the
expensive judge. Editing the seed makes a NEW run under the new profile_id, and
the old run's scores stay put -- that side-by-side history is the convergence study.

Live mode runs all three stages. Benchmark mode skips retrieve + domain filter
and judges the human-labeled set in the window directly (relevance gate already
applied by the labels). Benchmark labels are NEVER written as evaluations.

Everything else -- examining papers, flagging them, scoring how the judge did,
plotting -- is built on top of what this produces and lives in other modules
(the dashes, profile_analysis). This module ends at the judge.
"""

import hashlib
import time

from litcurator import (retrieve, domain_filter, judge, summarize, db_interface,
                        profile_interface, prompt_interface)
from litcurator.config import DOMAIN_THRESHOLD, LOCKED_TEST_START, LOCKED_TEST_END

JUDGE_BATCH_SIZE = 5


class LockedTestSetError(RuntimeError):
    """Raised when a run would judge the held-out November 2025 test set."""


def _overlaps_locked_test(start, end):
    """True if [start, end] intersects the locked test window. ISO date strings
    compare lexically, so plain string comparison is correct here."""
    return start <= LOCKED_TEST_END and end >= LOCKED_TEST_START


def run(start, end, benchmark=False, final_test=False, domain_threshold=DOMAIN_THRESHOLD,
        judge_batch_size=JUDGE_BATCH_SIZE):
    """Run the pipeline over [start, end] (ISO dates). Idempotent/resumable.
    Returns {judged, cost, mode}.

    Refuses any window overlapping the locked November 2025 test set unless
    final_test=True -- that test must be judged exactly once, at the end."""
    if not final_test and _overlaps_locked_test(start, end):
        raise LockedTestSetError(
            f"window {start}..{end} overlaps the LOCKED November 2025 test set "
            f"({LOCKED_TEST_START}..{LOCKED_TEST_END}). It is held out for the final "
            f"evaluation and must be judged exactly once. If this really is that final "
            f"run, pass final_test=True (CLI: --final-test). Doing so spends the test set."
        )
    t0 = time.monotonic()
    profile_text = profile_interface.load_active()   # raises if no active profile
    mode = "benchmark" if benchmark else "live"
    print(f"=== litcurator pipeline [{start} .. {end}] mode={mode} ===")

    if not benchmark:
        _retrieve_stage(start, end)

    conn = db_interface.get_connection()
    try:
        profile_id = db_interface.get_or_create_profile(conn, profile_text)
        print(f"profile: {profile_interface.active_path().name} ({profile_id[:12]})")

        if benchmark:
            # final_test passes through so the held-out November set is included
            # ONLY on the explicit final run; every dev benchmark subtracts it.
            survivors = db_interface.labeled_articles(conn, start, end, relevant=1,
                                                      final_test=final_test)
            print(f"[benchmark] {len(survivors)} relevance-gated labeled articles")
        else:
            _domain_stage(conn, start, end, mode, domain_threshold)
            survivors = db_interface.get_articles_passing_domain_filter(
                conn, start, end, domain_threshold)
            print(f"domain survivors: {len(survivors)}")

        _summarize_stage(conn, survivors)
        _pagination_stage(conn, survivors)

        judged, cost = _judge_stage(conn, survivors, profile_text, profile_id, mode,
                                    start, end, domain_threshold, judge_batch_size)
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    print(f"=== done: {judged} judged this run | est. judge cost ${cost:.4f} | wall {_fmt_dur(elapsed)} ===")
    return {"judged": judged, "cost": cost, "mode": mode, "seconds": elapsed}


def _retrieve_stage(start, end):
    print("[retrieve] querying PubMed...")
    retrieve.retrieve_range(start, end)   # inserts + dedups in its own connection


def _domain_stage(conn, start, end, mode, threshold):
    """Stage 1: score every not-yet-scored article in the window for domain
    relevance, as an append-only domain run."""
    candidates = db_interface.articles_in_range(conn, start, end)
    # Stamp + key the domain run on its prompt too (the title prompt is the one this
    # stage uses), so a domain-prompt change mints a new run instead of silently
    # resuming the old one -- same provenance guarantee as the curation stage.
    prompt_hash = hashlib.sha256(domain_filter.DOMAIN_FILTER_PROMPT_TITLE.encode("utf-8")).hexdigest()
    run_id = db_interface.find_or_create_scoring_run(
        conn, "domain", domain_filter.DOMAIN_FILTER_MODEL, mode,
        judge_prompt_version=domain_filter.DOMAIN_FILTER_PROMPT_VERSION,
        judge_prompt_hash=prompt_hash,
        date_start=start, date_end=end, threshold=threshold)
    todo = db_interface.unevaluated_in_run(conn, run_id, [a["pmid"] for a in candidates])
    if not todo:
        print("[domain] nothing to score")
        db_interface.complete_scoring_run(conn, run_id)
        return
    by_pmid = {a["pmid"]: a for a in candidates}
    print(f"[domain] scoring {len(todo)} titles...")
    in_tok = out_tok = 0
    cost = 0.0
    for batch_pmids in _chunks(todo, domain_filter.BATCH_SIZE_TITLE):
        batch = [by_pmid[p] for p in batch_pmids]
        results, usage = domain_filter.score_domain_batch(batch, mode="title")
        in_tok += usage.input_tokens
        out_tok += usage.output_tokens
        cost += (usage.input_tokens * domain_filter.DOMAIN_FILTER_COST_PER_M_INPUT
                 + usage.output_tokens * domain_filter.DOMAIN_FILTER_COST_PER_M_OUTPUT) / 1_000_000
        for art, (score, reasoning) in zip(batch, results):
            db_interface.insert_evaluation(conn, art["pmid"], run_id, score, rationale=reasoning)
    db_interface.complete_scoring_run(conn, run_id, in_tok, out_tok, cost)


def _summarize_stage(conn, survivors):
    """Generate the neutral review-feed summary for any survivor that lacks one.
    Idempotent: a summary is a stable paper fact, so already-summarized papers are
    skipped and a re-run never re-pays. Non-critical -- a batch that fails to parse
    is logged and skipped, never aborting the run. Runs over survivors in both
    modes (it is keyed to the article, independent of profile/run)."""
    todo = [a for a in survivors if not a.get("summary")]
    if not todo:
        return
    print(f"[summarize] summarizing {len(todo)} papers...", flush=True)
    cost = 0.0
    done = 0
    for batch in _chunks(todo, summarize.BATCH_SIZE):
        try:
            summaries, usage = summarize.summarize_batch(batch)
        except ValueError as e:
            print(f"  [summarize] batch of {len(batch)} failed to parse ({e}); skipping", flush=True)
            continue
        cost += (usage.input_tokens * summarize.COST_PER_M_INPUT
                 + usage.output_tokens * summarize.COST_PER_M_OUTPUT) / 1_000_000
        for art, summary in zip(batch, summaries):
            db_interface.set_article_summary(conn, art["pmid"], summary)
            done += 1
        print(f"  summarized {done}/{len(todo)}  (${cost:.4f})", flush=True)
    print(f"[summarize] done: {done} summarized | est. cost ${cost:.4f}")


def backfill_pages(conn, articles):
    """Re-fetch and store page ranges for `articles` that lack one. PubMed assigns
    pagination only when the print issue appears, so papers caught ahead-of-print
    have none at retrieval; this fills them as they paginate. Idempotent (skips
    papers that already have pages); electronic-only / article-number journals
    never get one, so they stay NULL and are simply re-checked next time. Network-
    only, no LLM. Returns (n_checked, n_filled). Shared by the live pipeline's
    pagination stage and the `litcurator backfill_pages` command."""
    todo = [a for a in articles if not a.get("pages")]
    filled = checked = 0
    for batch in _chunks(todo, 100):
        fetched = {f["pubmed_id"]: f for f in retrieve.fetch_articles([a["pmid"] for a in batch])}
        for art in batch:
            pages = (fetched.get(art["pmid"]) or {}).get("pages")
            if pages:
                db_interface.set_article_pages(conn, art["pmid"], pages)
                art["pages"] = pages   # so a same-run judge sees it
                filled += 1
        checked += len(batch)
        print(f"  pages: {filled} filled / {checked} checked", flush=True)
    return len(todo), filled


def _pagination_stage(conn, survivors):
    """Backfill page ranges for survivors lacking one (see backfill_pages). The
    page range is a signal to the judge -- a short span flags a News & Views /
    commentary rather than a substantive piece."""
    todo = [a for a in survivors if not a.get("pages")]
    if not todo:
        return
    print(f"[pagination] checking {len(todo)} papers for page ranges...")
    _, filled = backfill_pages(conn, survivors)
    print(f"[pagination] done: {filled} filled (rest ahead-of-print or electronic-only)")


def _judge_stage(conn, survivors, profile_text, profile_id, mode,
                 start, end, threshold, batch_size):
    """Stage 2: judge each not-yet-judged survivor against the profile, as an
    append-only curation run stamped with the profile + prompt that produced it.
    The active judge prompt is loaded from disk and registered (content-addressed)
    so the run's judge_prompt_hash == prompts.id -- prompt provenance by JOIN."""
    system_prompt = prompt_interface.load_active()
    prompt_id = db_interface.get_or_create_prompt(conn, system_prompt)
    run_id = db_interface.find_or_create_scoring_run(
        conn, "curation", judge.MODEL, mode, profile_id=profile_id,
        judge_prompt_version=prompt_interface.content_hash(system_prompt),
        judge_prompt_hash=prompt_id,
        date_start=start, date_end=end, threshold=threshold)
    todo = db_interface.unevaluated_in_run(conn, run_id, [a["pmid"] for a in survivors])
    if not todo:
        print("[judge] nothing to judge")
        db_interface.complete_scoring_run(conn, run_id)
        return 0, 0.0
    by_pmid = {a["pmid"]: a for a in survivors}
    print(f"[judge] judging {len(todo)} papers (run {run_id})...")
    cost = 0.0
    done = 0
    for batch_pmids in _chunks(todo, batch_size):
        batch = [by_pmid[p] for p in batch_pmids]
        items = [{"title": a["title"], "abstract": a["abstract"], "journal": a["journal"],
                  "pages": a.get("pages")}
                 for a in batch]
        judgments, batch_cost = judge.judge_articles_batch(
            items, profile_text, system_prompt=system_prompt)
        cost += batch_cost
        for art, j in zip(batch, judgments):
            db_interface.insert_evaluation(
                conn, art["pmid"], run_id, j["estimated_score"],
                rationale=j.get("curation_rationale"),
                surface_decision=j.get("surface_decision"),
                possible_mismatch=j.get("possible_mismatch"))
            done += 1
        print(f"  judged {done}/{len(todo)}")
    db_interface.complete_scoring_run(conn, run_id, cost_usd=cost)
    return done, cost


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _fmt_dur(seconds):
    """Human wall time: 5s / 2m 34s / 1h 02m 05s."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"
