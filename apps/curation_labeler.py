"""
Streamlit app for curation labeling of relevant articles.

Shows articles marked relevant=1 and asks for a rating:
  0: didn't make the cut (reject button)
  1-5: above the noise (1) to can't miss (5)

Run from repo root:
    streamlit run apps/curation_labeler.py
"""

import random
import time
import streamlit as st
from litcurator import db
from litcurator.config import GROUND_TRUTH_DB
from litcurator.config import CURATION_BATCH_SIZE as BATCH_SIZE
from litcurator.label import render_authors, read_batch_state, write_batch_state, BATCH_STATE_FILE

st.title("LitCurator — Curation Labeling")

st.markdown("""
<style>
button[kind="secondary"] {
    background-color: #888888 !important;
    border-color: #888888 !important;
    color: white !important;
}
button[kind="primary"] { color: white !important; }
div[data-testid="stColumn"]:nth-child(1) button[kind="primary"],
div[data-testid="column"]:nth-child(1) button[kind="primary"] {
    background-color: #6c3483 !important; border-color: #6c3483 !important;
}
div[data-testid="stColumn"]:nth-child(2) button[kind="primary"],
div[data-testid="column"]:nth-child(2) button[kind="primary"] {
    background-color: #1a3a8f !important; border-color: #1a3a8f !important;
}
div[data-testid="stColumn"]:nth-child(3) button[kind="primary"],
div[data-testid="column"]:nth-child(3) button[kind="primary"] {
    background-color: #f0b429 !important; border-color: #f0b429 !important;
}
div[data-testid="stColumn"]:nth-child(4) button[kind="primary"],
div[data-testid="column"]:nth-child(4) button[kind="primary"] {
    background-color: #e05c1a !important; border-color: #e05c1a !important;
}
div[data-testid="stColumn"]:nth-child(5) button[kind="primary"],
div[data-testid="column"]:nth-child(5) button[kind="primary"] {
    background-color: #b01010 !important; border-color: #b01010 !important;
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
    "SELECT pmid, title, abstract, authors_json, journal, pub_date, doi, curation_label "
    "FROM articles WHERE relevant = 1"
).fetchall()}

if "curation_order" not in st.session_state:
    pmids = list(all_articles.keys())
    random.seed(42)
    random.shuffle(pmids)
    st.session_state.curation_order = pmids

if "batch_history" not in st.session_state:
    st.session_state.batch_history = load_batch_history("curation")
if "review_index" not in st.session_state:
    st.session_state.review_index = None

articles = [all_articles[p] for p in st.session_state.curation_order if p in all_articles]
total = len(articles)
labeled = sum(1 for a in articles if a["curation_label"] is not None)
remaining = [a for a in articles if a["curation_label"] is None]

if st.session_state.get("done"):
    st.title("See you next time! 😊👋")
    conn.close()
    st.stop()

if not remaining:
    st.success("All relevant articles rated!")
    st.info(f"{labeled} of {total} curated overall")
    conn.close()
    st.stop()

batch_start_count, batch_elapsed, total_elapsed = read_batch_state("curation")
if batch_start_count is None:
    batch_start_count = labeled
    batch_elapsed = 0.0
    total_elapsed = 0.0
    write_batch_state("curation", batch_start_count, batch_elapsed, total_elapsed)

fresh_session = "last_rerun_time" not in st.session_state
if fresh_session:
    st.session_state.last_rerun_time = time.time()

now = time.time()
delta = now - st.session_state.last_rerun_time
batch_elapsed += delta
total_elapsed += delta
st.session_state.last_rerun_time = now
write_batch_state("curation", batch_start_count, batch_elapsed, total_elapsed)

batch_count = labeled - batch_start_count
overall_avg = total_elapsed / labeled if labeled > 0 else 0
per_article = batch_elapsed / batch_count if batch_count > 0 else 0

# Auto-reset on fresh load so a completed batch doesn't greet the user with the break screen
if batch_count >= BATCH_SIZE and fresh_session:
    write_batch_state("curation", labeled, 0.0, total_elapsed)
    st.session_state.batch_history = []
    save_batch_history("curation", [])
    st.session_state.review_index = None
    st.rerun()

if batch_count >= BATCH_SIZE and st.session_state.review_index is None:
    mins, secs = divmod(int(batch_elapsed), 60)
    st.divider()
    st.success(f"Batch of {BATCH_SIZE} done — take a break!")
    st.info(f"{labeled} of {total} curated overall  |  {mins}m {secs}s this batch  |  {per_article:.1f}s per article  |  {overall_avg:.1f}s overall avg")
    col_cont, col_done, _ = st.columns([1, 1, 4])
    if col_cont.button("Continue", type="secondary"):
        write_batch_state("curation", labeled, 0.0, total_elapsed)
        st.session_state.batch_history = []
        save_batch_history("curation", [])
        st.session_state.review_index = None
        st.rerun()
    if col_done.button("Done", type="secondary"):
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
    current_label = review_article["curation_label"]

    st.subheader(review_article["title"])
    doi = review_article.get("doi")
    st.caption(f"{review_article['journal']} | {review_article['pub_date'][:7]}")
    if doi:
        st.markdown(f"[View Article](https://doi.org/{doi})")
    st.write(render_authors(review_article["authors_json"]))

    with st.expander("Abstract", expanded=True):
        st.write(review_article["abstract"] or "_No abstract_")

    st.write("")
    col_reject, col_skip_spacer, _ = st.columns([2, 1, 6])
    if col_reject.button("0 — Didn't make it", key="review_btn_0", type="secondary", use_container_width=True):
        db.update_curation(conn, review_pmid, label=0)
        st.session_state.review_index = None
        st.rerun()

    st.write("")
    st.markdown("<p style='color: black; font-size: 1rem; margin-bottom: 4px;'>1 = above the noise &nbsp;&nbsp;|&nbsp;&nbsp; 3 = very cool &nbsp;&nbsp;|&nbsp;&nbsp; 5 = can't miss</p>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5, _, _, _, _ = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1])
    for col, val in [(c1, 1), (c2, 2), (c3, 3), (c4, 4), (c5, 5)]:
        if col.button(str(val), key=f"review_btn_{val}", type="primary", use_container_width=True):
            db.update_curation(conn, review_pmid, label=val)
            st.session_state.review_index = None
            st.rerun()

    st.info(f"Review mode — article {idx + 1} of {len(st.session_state.batch_history)} in batch | **Current label: {current_label}**")

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
doi = article.get("doi")
st.subheader(article["title"])
st.caption(f"{article['journal']} | {article['pub_date'][:7]}")
if doi:
    st.markdown(f"[View Article](https://doi.org/{doi})")
st.write(render_authors(article["authors_json"]))

with st.expander("Abstract", expanded=True):
    st.write(article["abstract"] or "_No abstract_")

st.write("")

col_reject, col_skip, _ = st.columns([2, 1, 6])

if col_reject.button("0 — Didn't make it", key="btn_0", type="secondary", use_container_width=True):
    db.update_curation(conn, article["pmid"], label=0)
    st.session_state.batch_history.append(article["pmid"])
    save_batch_history("curation", st.session_state.batch_history)
    st.rerun()

if col_skip.button("Skip", key="btn_skip", use_container_width=True):
    order = st.session_state.curation_order
    order.remove(article["pmid"])
    order.append(article["pmid"])
    st.rerun()

st.write("")

st.markdown("<p style='color: black; font-size: 1rem; margin-bottom: 4px;'>1 = above the noise &nbsp;&nbsp;|&nbsp;&nbsp; 3 = very cool &nbsp;&nbsp;|&nbsp;&nbsp; 5 = can't miss</p>", unsafe_allow_html=True)

c1, c2, c3, c4, c5, _, _, _, _ = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1])
for col, val in [(c1, 1), (c2, 2), (c3, 3), (c4, 4), (c5, 5)]:
    if col.button(str(val), key=f"btn_{val}_{article['pmid']}", type="primary", use_container_width=True):
        db.update_curation(conn, article["pmid"], label=val)
        st.session_state.batch_history.append(article["pmid"])
        save_batch_history("curation", st.session_state.batch_history)
        st.rerun()

st.write(f"**Batch: {batch_count} / {BATCH_SIZE}**")
st.caption(f"{labeled} of {total} curated overall  |  {overall_avg:.1f}s per article (overall avg)")

st.write("")
if st.session_state.batch_history:
    if st.button("← Back", key="back_btn"):
        st.session_state.review_index = len(st.session_state.batch_history) - 1
        st.rerun()

conn.close()
