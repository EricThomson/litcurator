"""
Streamlit app for fast relevance labeling.

Shows articles selected_for_review=1 one at a time. For each article,
click Relevant or Not Relevant.

Run from repo root:
    streamlit run apps/relevance_labeler.py
"""

import random
import time
import streamlit as st
from litcurator import db
from litcurator.config import GROUND_TRUTH_DB
from litcurator.config import RELEVANCE_BATCH_SIZE as BATCH_SIZE
from litcurator.label import render_authors, read_batch_state, write_batch_state, BATCH_STATE_FILE, get_status

st.title("LitCurator — Relevance Labeling")

st.markdown("""
<style>
button[kind="primary"] {
    background-color: #2d7a2d !important;
    border-color: #2d7a2d !important;
    color: white !important;
}
button[kind="secondary"] {
    background-color: #888888 !important;
    border-color: #888888 !important;
    color: white !important;
}
div[data-testid="stColumn"]:nth-child(2) button[kind="secondary"] {
    background-color: #c0392b !important;
    border-color: #c0392b !important;
}
</style>
""", unsafe_allow_html=True)


def load_batch_history(prefix):
    if BATCH_STATE_FILE.exists():
        data = {}
        for line in BATCH_STATE_FILE.read_text().splitlines():
            key, val = line.split("=", 1)
            data[key.strip()] = val.strip()
        history_str = data.get(f"{prefix}_batch_history", "")
        return history_str.split(",") if history_str else []
    return []


def save_batch_history(prefix, history):
    from litcurator.config import DATA_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if BATCH_STATE_FILE.exists():
        for line in BATCH_STATE_FILE.read_text().splitlines():
            key, val = line.split("=", 1)
            data[key.strip()] = val.strip()
    data[f"{prefix}_batch_history"] = ",".join(history)
    BATCH_STATE_FILE.write_text("\n".join(f"{k}={v}" for k, v in sorted(data.items())) + "\n")


conn = db.get_connection(GROUND_TRUTH_DB)

all_articles = {a["pmid"]: dict(a) for a in conn.execute(
    "SELECT pmid, title, abstract, authors_json, journal, pub_date, relevant "
    "FROM articles WHERE selected_for_review = 1"
).fetchall()}

if "pmid_order" not in st.session_state:
    pmids = list(all_articles.keys())
    random.seed(42)
    random.shuffle(pmids)
    st.session_state.pmid_order = pmids

if "batch_history" not in st.session_state:
    st.session_state.batch_history = load_batch_history("relevance")
if "review_index" not in st.session_state:
    st.session_state.review_index = None

articles = [all_articles[p] for p in st.session_state.pmid_order if p in all_articles]
total = len(articles)
labeled = sum(1 for a in articles if a["relevant"] is not None)
remaining = [a for a in articles if a["relevant"] is None]

if st.session_state.get("done"):
    st.title("See you next time! 😊👋")
    st.divider()
    s = get_status(conn)
    col_rel, col_cur = st.columns(2)
    col_rel.metric("Relevance labeled", f"{s['relevance_labeled']} / {s['selected']}")
    col_rel.metric("Relevant", f"{s['relevant']}  ({s['pct_relevant']:.1f}%)")
    col_cur.metric("Curation labeled", f"{s['curation_labeled']} / {s['relevant']}")
    col_cur.metric("Above the noise (1+)", f"{s['above_noise']}  ({s['pct_above_noise']:.1f}%)")
    conn.close()
    st.stop()

if not remaining:
    st.success("All articles labeled!")
    st.info(f"{labeled} of {total} labeled overall")
    conn.close()
    st.stop()

batch_start_count, batch_elapsed, total_elapsed = read_batch_state("relevance")
if batch_start_count is None:
    batch_start_count = labeled
    batch_elapsed = 0.0
    total_elapsed = 0.0
    write_batch_state("relevance", batch_start_count, batch_elapsed, total_elapsed)

fresh_session = "last_rerun_time" not in st.session_state
if fresh_session:
    st.session_state.last_rerun_time = time.time()

now = time.time()
delta = now - st.session_state.last_rerun_time
batch_elapsed += delta
total_elapsed += delta
st.session_state.last_rerun_time = now
write_batch_state("relevance", batch_start_count, batch_elapsed, total_elapsed)

batch_count = labeled - batch_start_count
overall_avg = total_elapsed / labeled if labeled > 0 else 0
per_article = batch_elapsed / batch_count if batch_count > 0 else 0

# Auto-reset on fresh load so a completed batch doesn't greet the user with the break screen
if batch_count >= BATCH_SIZE and fresh_session:
    write_batch_state("relevance", labeled, 0.0, total_elapsed)
    st.session_state.batch_history = []
    save_batch_history("relevance", [])
    st.session_state.review_index = None
    st.rerun()

if batch_count >= BATCH_SIZE and st.session_state.review_index is None:
    mins, secs = divmod(int(batch_elapsed), 60)
    st.divider()
    st.success(f"Batch of {BATCH_SIZE} done — take a break!")
    st.info(f"{labeled} of {total} labeled overall  |  {mins}m {secs}s this batch  |  {per_article:.1f}s per article  |  {overall_avg:.1f}s overall avg")
    col_cont, col_done, _ = st.columns([1, 1, 4])
    if col_cont.button("Continue", type="secondary"):
        write_batch_state("relevance", labeled, 0.0, total_elapsed)
        st.session_state.batch_history = []
        save_batch_history("relevance", [])
        st.session_state.review_index = None
        st.rerun()
    if col_done.button("Done", type="secondary"):
        write_batch_state("relevance", labeled, 0.0, total_elapsed)
        st.session_state.batch_history = []
        save_batch_history("relevance", [])
        st.session_state.review_index = None
        st.session_state.done = True
        st.rerun()
    if st.session_state.batch_history:
        if st.button("← Back", key="break_back"):
            st.session_state.review_index = len(st.session_state.batch_history) - 1
            st.rerun()
    conn.close()
    st.stop()

st.divider()

# --- REVIEW MODE ---
if st.session_state.review_index is not None:
    idx = st.session_state.review_index
    review_pmid = st.session_state.batch_history[idx]
    review_article = all_articles[review_pmid]
    current_label = review_article["relevant"]
    label_text = "✅ Relevant" if current_label == 1 else "❌ Not Relevant"

    st.subheader(review_article["title"])
    st.caption(f"{review_article['journal']} | {review_article['pub_date'][:7]}")
    st.write(render_authors(review_article["authors_json"]))

    with st.expander("Abstract", expanded=False):
        st.write(review_article["abstract"] or "_No abstract_")

    st.write("")
    col1, col2 = st.columns(2)
    if col1.button("✅ Relevant", type="primary", use_container_width=True, key="review_relevant"):
        conn.execute("UPDATE articles SET relevant = 1 WHERE pmid = ?", (review_pmid,))
        conn.commit()
        st.session_state.review_index = None
        st.rerun()
    if col2.button("❌ Not Relevant", type="secondary", use_container_width=True, key="review_not_relevant"):
        conn.execute("UPDATE articles SET relevant = 0 WHERE pmid = ?", (review_pmid,))
        conn.commit()
        st.session_state.review_index = None
        st.rerun()

    st.info(f"Review mode — article {idx + 1} of {len(st.session_state.batch_history)} in batch | Currently: {label_text}")

    col_back, _, col_fwd, _ = st.columns([1, 1, 2, 2])
    if idx > 0 and col_back.button("← Back", key="review_back"):
        st.session_state.review_index -= 1
        st.rerun()
    if col_fwd.button("→ Continue Labeling", key="review_fwd"):
        st.session_state.review_index = None
        st.rerun()

    conn.close()
    st.stop()

# --- NORMAL LABELING MODE ---
article = remaining[0]
st.subheader(article["title"])
st.caption(f"{article['journal']} | {article['pub_date'][:7]}")
st.write(render_authors(article["authors_json"]))

with st.expander("Abstract", expanded=False):
    st.write(article["abstract"] or "_No abstract_")

st.write("")
col1, col2 = st.columns(2)

if col1.button("✅ Relevant", type="primary", use_container_width=True):
    conn.execute("UPDATE articles SET relevant = 1 WHERE pmid = ?", (article["pmid"],))
    conn.commit()
    st.session_state.batch_history.append(article["pmid"])
    save_batch_history("relevance", st.session_state.batch_history)
    st.rerun()

if col2.button("❌ Not Relevant", type="secondary", use_container_width=True):
    conn.execute("UPDATE articles SET relevant = 0 WHERE pmid = ?", (article["pmid"],))
    conn.commit()
    st.session_state.batch_history.append(article["pmid"])
    save_batch_history("relevance", st.session_state.batch_history)
    st.rerun()

st.write(f"**Batch: {batch_count} / {BATCH_SIZE}**")
st.caption(f"{labeled} of {total} labeled overall  |  {overall_avg:.1f}s per article (overall avg)")

st.write("")
if st.session_state.batch_history:
    if st.button("← Back", key="back_btn"):
        st.session_state.review_index = len(st.session_state.batch_history) - 1
        st.rerun()

conn.close()
