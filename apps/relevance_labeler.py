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
from litcurator.label import render_authors, read_batch_state, write_batch_state, BATCH_SIZE

st.title("LitCurator — Relevance Labeling")

st.markdown("""
<style>
button[kind="primary"] {
    background-color: #2d7a2d !important;
    border-color: #2d7a2d !important;
    color: white !important;
}
button[kind="secondary"] {
    background-color: #c0392b !important;
    border-color: #c0392b !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

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

articles = [all_articles[p] for p in st.session_state.pmid_order if p in all_articles]
total = len(articles)
labeled = sum(1 for a in articles if a["relevant"] is not None)
remaining = [a for a in articles if a["relevant"] is None]

if st.session_state.get("done"):
    st.title("See you next time! 😊👋")
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

if "last_rerun_time" not in st.session_state:
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

st.write(f"**Batch: {batch_count} / {BATCH_SIZE}**")
st.caption(f"{labeled} of {total} labeled overall  |  {overall_avg:.1f}s per article (overall avg)")

if batch_count >= BATCH_SIZE:
    mins, secs = divmod(int(batch_elapsed), 60)
    st.divider()
    st.success(f"Batch of {BATCH_SIZE} done — take a break!")
    st.info(f"{labeled} of {total} labeled overall  |  {mins}m {secs}s this batch  |  {per_article:.1f}s per article  |  {overall_avg:.1f}s overall avg")
    col_cont, col_done, _ = st.columns([1, 1, 4])
    if col_cont.button("Continue", type="secondary"):
        write_batch_state("relevance", labeled, 0.0, total_elapsed)
        st.rerun()
    if col_done.button("Done", type="secondary"):
        st.session_state.done = True
        st.rerun()
    conn.close()
    st.stop()

st.divider()

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
    st.rerun()

if col2.button("❌ Not Relevant", type="secondary", use_container_width=True):
    conn.execute("UPDATE articles SET relevant = 0 WHERE pmid = ?", (article["pmid"],))
    conn.commit()
    st.rerun()

conn.close()
