"""
Curation feed: review LLM curation scores and flag mistakes for the profile builder.

Defaults to live mode (everyday use). Benchmark mode is an explicit developer flag:

    streamlit run apps/curation_feed.py                       # live mode (default)
    streamlit run apps/curation_feed.py -- --benchmark        # benchmark mode (dev only)
    streamlit run apps/curation_feed.py -- --start 2025-01-01 --end 2025-01-31
    streamlit run apps/curation_feed.py -- --benchmark --start 2025-01-01 --end 2025-01-31

Benchmark mode groups articles by FN/FP/TP/TN against your curation labels and
gates Sonnet on human relevance (not Haiku) for clean stage-isolated evaluation.
Live mode runs the full pipeline (retrieve -> Haiku -> Sonnet) and shows
above-threshold articles ranked by score.
"""

import argparse
import html as _html
import subprocess
import sys
from collections import defaultdict
from datetime import date
from dotenv import load_dotenv
import streamlit as st
from litcurator import db, evaluate, retrieve
from litcurator.config import LITCURATOR_DB, LLM_SCORE_THRESHOLD
from litcurator.label import render_authors

load_dotenv()

_parser = argparse.ArgumentParser()
_parser.add_argument("--start", default=None)
_parser.add_argument("--end", default=None)
_parser.add_argument("--benchmark", action="store_true", help="Developer flag: enable benchmark mode against ground-truth labels")
_args, _ = _parser.parse_known_args()

CLI_START = date.fromisoformat(_args.start) if _args.start else None
CLI_END = date.fromisoformat(_args.end) if _args.end else None
BENCHMARK_MODE = _args.benchmark

DOMAIN_THRESHOLD = 0.5
DOMAIN_BORDERLINE_WINDOW = 0.15


def _months_in_range(date_start, date_end):
    """Return list of 'YYYY-MM' strings covering the date range."""
    d = date.fromisoformat(date_start)
    end = date.fromisoformat(date_end)
    months = []
    while d <= end:
        months.append(d.strftime("%Y-%m"))
        d = d.replace(month=d.month + 1, day=1) if d.month < 12 else d.replace(year=d.year + 1, month=1, day=1)
    return list(dict.fromkeys(months))


def score_color(score):
    if score < 0.2:
        return "#888888"
    elif score < 0.4:
        return "#6c3483"
    elif score < 0.6:
        return "#1a3a8f"
    elif score < 0.8:
        return "#f0b429"
    elif score < 0.9:
        return "#e05c1a"
    else:
        return "#b01010"


def _error_order(article, threshold):
    is_pass = article["score"] >= threshold
    is_above_noise = article["curation_label"] is not None and article["curation_label"] >= 1
    if not is_pass and is_above_noise:
        return 0  # FN — costliest
    if is_pass and not is_above_noise:
        return 1  # FP
    if is_pass and is_above_noise:
        return 2  # TP
    return 3      # TN


def _error_label(order):
    return {0: "False Negatives", 1: "False Positives", 2: "True Positives", 3: "True Negatives"}[order]


def render_article_card(article, threshold, total, index, feedback, prior_feedback, uningested_prior, conn, show_score=True):
    pmid = article["pmid"]
    eval_id = article["id"]
    score = article["score"]
    rationale = article["rationale"]
    doi = article["doi"]

    is_flagged_current = eval_id in feedback
    prior_flag = uningested_prior.get(pmid) if not is_flagged_current else None
    is_flagged_prior = prior_flag is not None
    is_flagged = is_flagged_current or is_flagged_prior

    color = score_color(score)
    if is_flagged_current:
        flag_indicator, badge_color = "🚩 ", color
    elif is_flagged_prior:
        flag_indicator, badge_color = "📌 ", "#6b4fa0"
    else:
        flag_indicator, badge_color = "", color

    score_badge = (
        f'<span style="background:{badge_color}; color:white; padding:2px 10px; '
        f'border-radius:4px; font-weight:bold; font-size:1.1em;">'
        f'{flag_indicator}{score:.2f}</span>'
        f'<span style="color:#888888; font-size:0.85em; margin-left:8px;">{index}/{total}</span>'
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(score_badge, unsafe_allow_html=True)
        st.markdown(f"**{article['title']}**")
        st.caption(f"{article['journal']} | {article['pub_date'][:7]}")
        st.write(render_authors(article["authors_json"]))
        if rationale:
            st.markdown(f"_{rationale}_")

        has_ingested_history = not is_flagged_current and not is_flagged_prior and pmid in prior_feedback
        if is_flagged_current:
            popover_label = "🚩 Flagged"
        elif is_flagged_prior:
            popover_label = "📌 Prior flag"
        elif has_ingested_history:
            popover_label = "🔖 Flag"
        else:
            popover_label = "Flag"

        with st.popover(popover_label):
            if is_flagged_current:
                current_note = feedback[eval_id]["note"]
            elif is_flagged_prior:
                current_note = prior_flag["note"]
                st.caption(f"Pending flag from previous session (score was {prior_flag['score']:.2f})")
            elif pmid in prior_feedback:
                current_note = prior_feedback[pmid]["note"]
                st.caption(f"You flagged this in a previous session (score was {prior_feedback[pmid]['score']:.2f})")
            else:
                current_note = ""
            note = st.text_area(
                "Notes on this score",
                value=current_note,
                key=f"note_{pmid}",
                height=100,
                placeholder="What's wrong, what's right, or what you want more of",
            )
            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("Save", key=f"save_{pmid}", type="primary"):
                db.upsert_feedback(conn, eval_id, note)
                if is_flagged_prior:
                    db.delete_feedback_for_evaluation(conn, prior_flag["evaluation_id"])
                st.rerun()
            remove_label = "Remove flag" if is_flagged_current else ("Discard prior flag" if is_flagged_prior else None)
            if remove_label and btn_col2.button(remove_label, key=f"remove_{pmid}"):
                target_eval_id = eval_id if is_flagged_current else prior_flag["evaluation_id"]
                db.delete_feedback_for_evaluation(conn, target_eval_id)
                st.rerun()

    with col_right:
        if doi:
            st.markdown(
                f'<a href="https://doi.org/{doi}" target="_blank" '
                f'style="font-size:1.05em; font-weight:bold;">View Article</a>',
                unsafe_allow_html=True
            )
        with st.expander("Abstract"):
            st.write(article["abstract"] or "_No abstract_")
        if article["relevant"] is not None:
            label_str = "relevant" if article["relevant"] == 1 else "not relevant"
            if article["curation_label"] is not None:
                label_str += f", score {article['curation_label']}/5"
            st.caption(f"Your judgment: {label_str}")
        if is_flagged_current:
            st.caption(f"Your note: {feedback[eval_id]['note']}")
        elif is_flagged_prior:
            st.markdown(
                f'<div style="background:#e8e4f0;border-left:3px solid #9b59b6;'
                f'padding:6px 10px;border-radius:4px;margin-top:4px;font-size:0.85em;color:#1a1a1a;">'
                f'📌 <strong>Pending from previous session:</strong><br>'
                f'{_html.escape(prior_flag["note"])}'
                f'</div>',
                unsafe_allow_html=True
            )
            btn_k, btn_d = st.columns(2)
            if btn_k.button("Keep", key=f"keep_prior_{pmid}", type="secondary"):
                db.upsert_feedback(conn, eval_id, prior_flag["note"])
                db.delete_feedback_for_evaluation(conn, prior_flag["evaluation_id"])
                st.rerun()
            if btn_d.button("Delete flag", key=f"delete_prior_{pmid}", type="primary"):
                db.delete_feedback_for_evaluation(conn, prior_flag["evaluation_id"])
                st.rerun()

    st.divider()


st.set_page_config(layout="wide")
st.title("Curation Feed" + (" (Benchmark)" if BENCHMARK_MODE else ""))

conn = db.get_connection(LITCURATOR_DB)

# --- Coverage view ---
coverage = db.get_pipeline_coverage(conn)
if coverage:
    period_stages = defaultdict(set)
    for row in coverage:
        period_stages[(row["date_start"], row["date_end"])].add(row["stage"])

    with st.expander("Coverage", expanded=False):
        header = st.columns([2, 2, 1, 1])
        header[0].markdown("**From**")
        header[1].markdown("**To**")
        header[2].markdown("**Domain**")
        header[3].markdown("**Curation**")
        for (ds, de), stages in sorted(period_stages.items()):
            row_cols = st.columns([2, 2, 1, 1])
            row_cols[0].write(ds)
            row_cols[1].write(de)
            row_cols[2].write("✓" if "domain" in stages else "—")
            row_cols[3].write("✓" if "curation" in stages else "—")

# --- Date range ---
col_date, col_spacer = st.columns([3, 5])
with col_date:
    default_range = (CLI_START, CLI_END) if CLI_START and CLI_END else []
    date_range = st.date_input("Date range", value=default_range)

if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
    st.info("Select a start and end date.")
    conn.close()
    st.stop()

date_start = date_range[0].isoformat()
date_end = date_range[1].isoformat()

# =========================================================
# BENCHMARK MODE: developer flag; group by error type vs labels
# =========================================================
if BENCHMARK_MODE:
    labeled_count = db.count_curation_labeled_for_date_range(conn, date_start, date_end)
    if labeled_count == 0:
        st.warning(
            "Benchmark mode requires hand-curated articles in this date range. "
            "No labeled articles found — pick a labeled month (Jan/Mar/May/Jul/Sep 2025) "
            "or relaunch without --benchmark for live mode."
        )
        conn.close()
        st.stop()

    st.caption(f"Benchmark mode: {labeled_count} hand-curated articles in this range.")

    has_curation = db.has_evaluations_for_date_range(conn, "curation", date_start, date_end)

    if not has_curation:
        if not st.session_state.get("curation_running"):
            if st.button("Run benchmark curation", type="primary"):
                st.session_state.curation_running = True
                st.rerun()
        else:
            with st.spinner("Scoring articles with Sonnet..."):
                _progress = st.empty()
                def _cb(batch_num, total_batches):
                    _progress.markdown(f"**Batch {batch_num}/{total_batches}**")
                evaluate.curation_score(
                    conn, date_start=date_start, date_end=date_end,
                    use_human_relevance_gate=True, progress_callback=_cb,
                )
            st.session_state.curation_running = False
            st.rerun()
        conn.close()
        st.stop()

    # Re-score with current profile (e.g., after editing profile.md)
    if st.session_state.get("curation_rerunning"):
        if st.session_state.get("curation_discard_flags"):
            db.discard_uningested_feedback(conn, _months_in_range(date_start, date_end))
        with st.spinner("Re-scoring with current profile..."):
            _progress = st.empty()
            def _cb(batch_num, total_batches):
                _progress.markdown(f"**Batch {batch_num}/{total_batches}**")
            evaluate.curation_score(
                conn, date_start=date_start, date_end=date_end,
                use_human_relevance_gate=True, progress_callback=_cb,
            )
        st.session_state.curation_rerunning = False
        st.session_state.pop("curation_discard_flags", None)
        st.rerun()
    discard_cb = st.checkbox("Discard pending flags for this period", value=False, key="bm_discard_cb")
    if st.button("Re-score with current profile"):
        st.session_state.curation_rerunning = True
        st.session_state.curation_discard_flags = st.session_state.get("bm_discard_cb", False)
        st.rerun()

    # Load latest curation evaluations, restrict to labeled
    all_articles = db.get_latest_evaluations(
        conn, stage="curation",
        date_start=date_start,
        date_end=date_end,
    )
    articles = [a for a in all_articles if a["curation_label"] is not None]

    if not articles:
        st.warning(
            "No curation-labeled articles have Sonnet scores yet. "
            "If you just ran evaluation curation, give it a moment and reload."
        )
        conn.close()
        st.stop()

    threshold = st.slider(
        "Score threshold",
        min_value=0.0, max_value=1.0,
        value=float(LLM_SCORE_THRESHOLD),
        step=0.05,
    )

    eval_ids = [a["id"] for a in articles]
    feedback = db.get_feedback_for_evaluations(conn, eval_ids)
    pmids = [a["pmid"] for a in articles]
    prior_feedback = db.get_latest_feedback_by_pmids(conn, pmids)
    uningested_prior = {
        pmid: row for pmid, row in db.get_uningested_feedback_by_pmids(conn, pmids).items()
        if row["evaluation_id"] not in eval_ids
    }

    tp = sum(1 for a in articles if a["score"] >= threshold and a["curation_label"] >= 1)
    fp = sum(1 for a in articles if a["score"] >= threshold and a["curation_label"] == 0)
    fn = sum(1 for a in articles if a["score"] < threshold and a["curation_label"] >= 1)
    tn = sum(1 for a in articles if a["score"] < threshold and a["curation_label"] == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    flagged_current = len(feedback)
    flagged_prior = sum(1 for a in articles if a["pmid"] in uningested_prior)
    flagged_total = flagged_current + flagged_prior

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Labeled", len(articles))
    c2.metric("TP", tp)
    c3.metric("FP", fp)
    c4.metric("FN", fn, help="Articles you scored 1-5 but Sonnet missed — the costly errors")
    c5.metric("Precision", f"{precision:.2f}")
    c6.metric("Recall", f"{recall:.2f}")
    c7.metric("Flagged", flagged_total,
              help=f"{flagged_current} this session, {flagged_prior} from previous sessions")

    st.divider()

    sorted_articles = sorted(articles, key=lambda a: (_error_order(a, threshold), -a["score"]))

    current_group = None
    for i, article in enumerate(sorted_articles, 1):
        order = _error_order(article, threshold)
        if order != current_group:
            current_group = order
            label = _error_label(order)
            count = sum(1 for a in articles if _error_order(a, threshold) == order)
            st.subheader(f"{label} ({count})")
        render_article_card(article, threshold, len(articles), i, feedback, prior_feedback, uningested_prior, conn)

# =========================================================
# LIVE MODE: no labels; run full pipeline, browse
# =========================================================
else:
    has_articles = db.has_articles_for_date_range(conn, date_start, date_end)
    has_domain = db.has_evaluations_for_date_range(conn, "domain", date_start, date_end)
    has_curation = db.has_evaluations_for_date_range(conn, "curation", date_start, date_end)

    status_cols = st.columns(3)
    status_cols[0].metric("Retrieved", "yes" if has_articles else "no")
    status_cols[1].metric("Domain filtered", "yes" if has_domain else "no")
    status_cols[2].metric("Curation scored", "yes" if has_curation else "no")

    if not (has_articles and has_domain and has_curation):
        if st.button("Process period", type="primary"):
            if not has_articles:
                with st.spinner("Retrieving articles from PubMed..."):
                    retrieve.retrieve_range(date_start, date_end, db_path=LITCURATOR_DB)
            if not has_domain:
                with st.spinner("Running domain filter..."):
                    evaluate.domain_filter(conn, date_start, date_end)
            if not has_curation:
                with st.spinner("Scoring articles with Sonnet..."):
                    _progress = st.empty()
                    def _cb(batch_num, total_batches):
                        _progress.markdown(f"**Batch {batch_num}/{total_batches}**")
                    evaluate.curation_score(conn, date_start=date_start, date_end=date_end, progress_callback=_cb)
            st.rerun()

    if not has_curation:
        conn.close()
        st.stop()

    # Re-score with current profile (e.g., after editing profile.md)
    if st.session_state.get("curation_rerunning"):
        if st.session_state.get("curation_discard_flags"):
            db.discard_uningested_feedback(conn, _months_in_range(date_start, date_end))
        with st.spinner("Re-scoring with current profile..."):
            _progress = st.empty()
            def _cb(batch_num, total_batches):
                _progress.markdown(f"**Batch {batch_num}/{total_batches}**")
            evaluate.curation_score(conn, date_start=date_start, date_end=date_end, progress_callback=_cb)
        st.session_state.curation_rerunning = False
        st.session_state.pop("curation_discard_flags", None)
        st.rerun()
    discard_cb = st.checkbox("Discard pending flags for this period", value=False, key="live_discard_cb")
    if st.button("Re-score with current profile"):
        st.session_state.curation_rerunning = True
        st.session_state.curation_discard_flags = st.session_state.get("live_discard_cb", False)
        st.rerun()

    st.divider()

    threshold = st.slider(
        "Score threshold",
        min_value=0.0, max_value=1.0,
        value=float(LLM_SCORE_THRESHOLD),
        step=0.05,
    )

    all_articles = db.get_latest_evaluations(
        conn, stage="curation",
        date_start=date_start,
        date_end=date_end,
    )

    if not all_articles:
        st.warning("No scored articles found for this date range.")
        conn.close()
        st.stop()

    eval_ids = [a["id"] for a in all_articles]
    feedback = db.get_feedback_for_evaluations(conn, eval_ids)
    pmids = [a["pmid"] for a in all_articles]
    prior_feedback = db.get_latest_feedback_by_pmids(conn, pmids)
    uningested_prior = {
        pmid: row for pmid, row in db.get_uningested_feedback_by_pmids(conn, pmids).items()
        if row["evaluation_id"] not in eval_ids
    }

    articles = [a for a in all_articles if a["score"] >= threshold]
    total = len(articles)
    flagged_current = len(feedback)
    flagged_prior = sum(1 for a in articles if a["pmid"] in uningested_prior)
    flagged_total = flagged_current + flagged_prior

    c1, c2, c3 = st.columns(3)
    c1.metric("Shown", total)
    c2.metric("Threshold", f"{threshold:.2f}")
    c3.metric("Flagged", flagged_total,
              help=f"{flagged_current} this session, {flagged_prior} from previous sessions")

    st.divider()

    for i, article in enumerate(articles, 1):
        render_article_card(article, threshold, total, i, feedback, prior_feedback, uningested_prior, conn)

    # Domain borderlines (live mode only)
    borderlines = db.get_domain_borderline_articles(
        conn, date_start, date_end,
        threshold=DOMAIN_THRESHOLD,
        window=DOMAIN_BORDERLINE_WINDOW,
    )
    if borderlines:
        with st.expander(f"Domain filter borderlines ({len(borderlines)} articles near threshold {DOMAIN_THRESHOLD})"):
            st.caption(
                "Articles within 0.15 of the domain threshold in either direction. "
                "Read-only — if you see a pattern, adjust the prompt in config.py."
            )
            prev_above = None
            for article in borderlines:
                currently_above = article["score"] >= DOMAIN_THRESHOLD
                if prev_above is True and not currently_above:
                    st.markdown(
                        "<hr style='border: 2px dashed #888888; margin: 8px 0;'>"
                        f"<p style='color:#888888; font-size:0.8em; text-align:center; margin-top:-10px;'>"
                        f"— threshold {DOMAIN_THRESHOLD} —</p>",
                        unsafe_allow_html=True,
                    )
                prev_above = currently_above
                color = score_color(article["score"])
                badge = (
                    f'<span style="background:{color}; color:white; padding:2px 8px; '
                    f'border-radius:4px; font-weight:bold;">{article["score"]:.2f}</span>'
                )
                st.markdown(badge, unsafe_allow_html=True)
                st.markdown(f"**{article['title']}**")
                st.caption(f"{article['journal']} | {article['pub_date'][:7]}")
                if article["rationale"]:
                    st.markdown(f"_{article['rationale']}_")
                st.divider()

# --- Done flagging (shared) ---
st.divider()
col_done, col_info = st.columns([2, 6])
with col_done:
    if st.button("Done flagging", type="primary"):
        st.session_state["review_done"] = True
        st.rerun()

if st.session_state.get("review_done"):
    uningested = db.get_uningested_feedback_periods(conn)
    total_pending = sum(c for _, c in uningested)
    flagged_total = len(db.get_feedback_for_evaluations(
        conn,
        [a["id"] for a in db.get_latest_evaluations(conn, stage="curation", date_start=date_start, date_end=date_end)]
    ))
    st.success(f"Flagged {flagged_total} article{'s' if flagged_total != 1 else ''} this session.")
    if total_pending > 0:
        period_summary = ", ".join(f"{m} ({c})" for m, c in uningested)
        st.info(f"{total_pending} flags pending profile update across: {period_summary}")
        if st.button("Launch profile builder", type="primary"):
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", "apps/profile_builder.py"])
            st.caption("Profile builder launching in a new tab. Safe to close this tab.")
    else:
        st.info("No pending flags — nothing to update in the profile builder yet.")

conn.close()
