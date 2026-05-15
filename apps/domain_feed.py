"""
Domain filter review: inspect Haiku domain filter output for a date range.

Shows all domain-evaluated articles sorted by score, with a moveable threshold
divider so you can see at a glance what passes and what gets filtered out.
To tune the filter, edit DOMAIN_FILTER_PROMPT in src/litcurator/config.py,
then re-run the domain filter here.

Run from repo root:
    streamlit run apps/domain_feed.py
    streamlit run apps/domain_feed.py -- --start 2025-07-01 --end 2025-07-14
"""

import argparse
from datetime import date

from dotenv import load_dotenv
import streamlit as st

from litcurator import db, evaluate
from litcurator.config import LITCURATOR_DB
from litcurator.label import render_authors

load_dotenv()

_parser = argparse.ArgumentParser()
_parser.add_argument("--start", default=None)
_parser.add_argument("--end", default=None)
_args, _ = _parser.parse_known_args()

CLI_START = date.fromisoformat(_args.start) if _args.start else None
CLI_END = date.fromisoformat(_args.end) if _args.end else None


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


st.set_page_config(layout="wide")
st.title("Domain Filter Review")

conn = db.get_connection(LITCURATOR_DB)

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

has_articles = db.has_articles_for_date_range(conn, date_start, date_end)
has_domain = db.has_evaluations_for_date_range(conn, "domain", date_start, date_end)

status_cols = st.columns(2)
status_cols[0].metric("Retrieved", "yes" if has_articles else "no")
status_cols[1].metric("Domain filtered", "yes" if has_domain else "no")

if not has_articles:
    st.warning("No articles retrieved for this date range. Retrieve them first via curation_feed.py.")
    conn.close()
    st.stop()

if not has_domain:
    if not st.session_state.get("domain_filter_running"):
        if st.button("Run domain filter", type="primary"):
            st.session_state.domain_filter_running = True
            st.rerun()
    else:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _on_progress(batch_num, total_batches, passed):
            progress_bar.progress(batch_num / total_batches)
            status_text.text(f"Batch {batch_num}/{total_batches} — {passed} passed so far")

        evaluate.domain_filter(conn, date_start, date_end, progress_callback=_on_progress)
        st.session_state.domain_filter_running = False
        st.rerun()
    conn.close()
    st.stop()

# --- Threshold slider ---
threshold = st.slider(
    "Domain threshold",
    min_value=0.0, max_value=1.0,
    value=0.5,
    step=0.05,
)
st.caption(
    "To tune the domain filter prompt, edit `DOMAIN_FILTER_PROMPT` in "
    "`src/litcurator/config.py`, then re-run the domain stage via curation_feed.py."
)

# --- Load all domain evaluations ---
all_articles = db.get_latest_evaluations(
    conn, stage="domain",
    date_start=date_start,
    date_end=date_end,
)

if not all_articles:
    st.warning("No domain evaluations found for this date range.")
    conn.close()
    st.stop()

# Restrict to hand-labeled articles only
articles = [a for a in all_articles if a["relevant"] is not None]
if not articles:
    st.warning("No hand-labeled articles found in this date range. Label some articles first.")
    conn.close()
    st.stop()

st.caption(
    f"Showing {len(articles)} hand-labeled articles "
    f"({len(all_articles)} total domain-evaluated in this range)."
)

above = [a for a in articles if a["score"] >= threshold]
below = [a for a in articles if a["score"] < threshold]

tp = sum(1 for a in articles if a["score"] >= threshold and a["relevant"] == 1)
fp = sum(1 for a in articles if a["score"] >= threshold and a["relevant"] == 0)
fn = sum(1 for a in articles if a["score"] < threshold and a["relevant"] == 1)
tn = sum(1 for a in articles if a["score"] < threshold and a["relevant"] == 0)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# --- Stats ---
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Labeled", len(articles))
c2.metric("TP", tp)
c3.metric("FP", fp)
c4.metric("FN", fn, help="Relevant articles dropped by the filter — the costly errors")
c5.metric("Precision", f"{precision:.2f}")
c6.metric("Recall", f"{recall:.2f}")

st.divider()


def _error_order(article, threshold):
    is_pass = article["score"] >= threshold
    is_relevant = article["relevant"] == 1
    if not is_pass and is_relevant:
        return 0  # FN — most important
    if is_pass and not is_relevant:
        return 1  # FP
    if is_pass and is_relevant:
        return 2  # TP
    return 3      # TN


def _error_label(order):
    return {0: "False Negatives", 1: "False Positives", 2: "True Positives", 3: "True Negatives"}[order]


# --- Article list grouped by error type ---
sorted_articles = sorted(articles, key=lambda a: (_error_order(a, threshold), -a["score"]))

current_group = None
for i, article in enumerate(sorted_articles, 1):
    score = article["score"]
    human_relevant = article["relevant"] == 1
    order = _error_order(article, threshold)

    if order != current_group:
        current_group = order
        label = _error_label(order)
        count = sum(1 for a in articles if _error_order(a, threshold) == order)
        st.subheader(f"{label} ({count})")

    color = score_color(score)
    score_badge = (
        f'<span style="background:{color}; color:white; padding:2px 10px; '
        f'border-radius:4px; font-weight:bold; font-size:1.1em;">'
        f'{score:.2f}</span>'
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(score_badge, unsafe_allow_html=True)
        st.markdown(f"**{article['title']}**")
        st.caption(f"{article['journal']} | {article['pub_date'][:7]}")
        st.write(render_authors(article["authors_json"]))
        if article["rationale"]:
            st.markdown(f"_{article['rationale']}_")

    with col_right:
        with st.expander("Abstract"):
            st.write(article["abstract"] or "_No abstract_")
        label_str = "relevant" if human_relevant else "not relevant"
        if article["curation_label"] is not None:
            label_str += f", score {article['curation_label']}/5"
        st.caption(f"Your judgment: {label_str}")

    st.divider()

conn.close()
