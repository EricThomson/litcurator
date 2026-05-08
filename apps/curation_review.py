"""
Streamlit app for human review of curation scores.

Shows articles scored by curation_score(), sorted by score descending.
Flag articles the LLM scored incorrectly with notes — flags feed into
the profile builder.

Run from repo root:
    streamlit run apps/curation_review.py
    streamlit run apps/curation_review.py -- --start 2025-01-01 --end 2025-01-14
"""

import argparse
import subprocess
import sys
from datetime import date
from dotenv import load_dotenv
import streamlit as st
from litcurator import db
from litcurator.config import (
    LITCURATOR_DB, LLM_SCORE_THRESHOLD,
)
from litcurator.label import render_authors

load_dotenv()

_parser = argparse.ArgumentParser()
_parser.add_argument("--start", default=None)
_parser.add_argument("--end", default=None)
_args, _ = _parser.parse_known_args()

CLI_START = date.fromisoformat(_args.start) if _args.start else None
CLI_END = date.fromisoformat(_args.end) if _args.end else None

NEAR_THRESHOLD_WINDOW = 0.15


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
st.title("Ranked Articles")

conn = db.get_connection(LITCURATOR_DB)

# --- Date range and view mode ---
col_date, col_filter, col_spacer = st.columns([2, 3, 3])
with col_date:
    default_range = (CLI_START, CLI_END) if CLI_START and CLI_END else []
    date_range = st.date_input(
        "Date range",
        value=default_range,
    )

with col_filter:
    view_mode = st.radio(
        "Show",
        ["All", "Above threshold", "Near threshold", "Flagged"],
        horizontal=True,
    )

if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
    st.info("Select a start and end date.")
    st.stop()

date_start, date_end = date_range
date_key = f"{date_start.isoformat()}_{date_end.isoformat()}"

all_articles = db.get_latest_evaluations(
    conn, stage="curation",
    date_start=date_start.isoformat(),
    date_end=date_end.isoformat(),
)

if not all_articles:
    st.warning("No scored articles in this date range. Run curation_rank() first.")
    conn.close()
    st.stop()

eval_ids = [a["id"] for a in all_articles]
feedback = db.get_feedback_for_evaluations(conn, eval_ids)
pmids = [a["pmid"] for a in all_articles]
prior_feedback = db.get_latest_feedback_by_pmids(conn, pmids)


def is_near_threshold(score):
    return abs(score - LLM_SCORE_THRESHOLD) <= NEAR_THRESHOLD_WINDOW


articles = all_articles
if view_mode == "Above threshold":
    articles = [a for a in articles if a["score"] >= LLM_SCORE_THRESHOLD]
elif view_mode == "Near threshold":
    articles = [a for a in articles if is_near_threshold(a["score"])]
elif view_mode == "Flagged":
    articles = [a for a in articles if a["id"] in feedback]

total = len(articles)
above = sum(1 for a in articles if a["score"] >= LLM_SCORE_THRESHOLD)
near = sum(1 for a in articles if is_near_threshold(a["score"]))
flagged_count = len(feedback)

# --- Stats bar ---
c1, c2 = st.columns(2)
c1.metric("Shown", total)
c2.metric(f"Above threshold (≥{LLM_SCORE_THRESHOLD})", above)
c3, c4 = st.columns(2)
c3.metric("Near threshold", near)
c4.metric("Flagged", flagged_count)

st.divider()

# --- Ranked list ---
prev_above_threshold = None

for i, article in enumerate(articles, 1):
    pmid = article["pmid"]
    eval_id = article["id"]
    score = article["score"]
    rationale = article["rationale"]
    doi = article["doi"]
    is_flagged = eval_id in feedback

    # Threshold boundary divider
    currently_above = score >= LLM_SCORE_THRESHOLD
    if prev_above_threshold is True and not currently_above:
        st.markdown(
            "<hr style='border: 2px dashed #888888; margin: 16px 0;'>"
            "<p style='color:#888888; font-size:0.85em; text-align:center; margin-top:-12px;'>"
            "— near threshold —</p>",
            unsafe_allow_html=True
        )
    prev_above_threshold = currently_above

    color = score_color(score)
    flag_indicator = "🚩 " if is_flagged else ""
    score_badge = (
        f'<span style="background:{color}; color:white; padding:2px 10px; '
        f'border-radius:4px; font-weight:bold; font-size:1.1em;">'
        f'{flag_indicator}{score:.2f}</span>'
        f'<span style="color:#888888; font-size:0.85em; margin-left:8px;">{i}/{total}</span>'
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(score_badge, unsafe_allow_html=True)
        st.markdown(f"**{article['title']}**")
        st.caption(f"{article['journal']} | {article['pub_date'][:7]}")
        st.write(render_authors(article["authors_json"]))
        if rationale:
            st.markdown(f"_{rationale}_")

        popover_label = "🚩 Flagged" if is_flagged else "Flag"
        with st.popover(popover_label):
            if is_flagged:
                current_note = feedback[eval_id]["note"]
            elif pmid in prior_feedback:
                current_note = prior_feedback[pmid]["note"]
            else:
                current_note = ""
            if not is_flagged and pmid in prior_feedback:
                st.caption(f"Pre-filled from previous flag (score was {prior_feedback[pmid]['score']:.2f})")
            note = st.text_area(
                "What did the LLM get wrong?",
                value=current_note,
                key=f"note_{pmid}",
                height=100,
            )
            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("Save", key=f"save_{pmid}", type="primary"):
                db.upsert_feedback(conn, eval_id, note)
                st.rerun()
            if is_flagged and btn_col2.button("Remove flag", key=f"remove_{pmid}"):
                db.delete_feedback_for_evaluation(conn, eval_id)
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
        if article["curation_label"] is not None:
            st.caption(f"Your label: {article['curation_label']}")
        if is_flagged:
            st.caption(f"Your note: {feedback[eval_id]['note']}")

    st.divider()

# --- Done flagging ---
st.divider()
col_done, col_info = st.columns([2, 6])
with col_done:
    if st.button("Done flagging", type="primary"):
        st.session_state["review_done"] = True
        st.rerun()

if st.session_state.get("review_done"):
    uningested = db.get_uningested_feedback_periods(conn)
    total_pending = sum(c for _, c in uningested)
    st.success(f"Flagged {flagged_count} article{'s' if flagged_count != 1 else ''} this session.")
    if total_pending > 0:
        period_summary = ", ".join(f"{m} ({c})" for m, c in uningested)
        st.info(f"{total_pending} flags pending profile update across: {period_summary}")
        if st.button("Launch profile builder", type="primary"):
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", "apps/profile_builder.py"])
            st.caption("Profile builder launching in a new tab...")
    else:
        st.info("No pending flags — nothing to update in the profile builder yet.")

conn.close()
