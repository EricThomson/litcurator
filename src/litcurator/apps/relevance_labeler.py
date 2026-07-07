"""
relevance_labeler.py -- fast relevance labeling of unlabeled articles.

Shows articles with no human_labels row, one at a time, date-masked.
Label each: Relevant (1) or Not Relevant (0).

Date masking: pub_date_iso is not shown. Jan-Nov 2025 papers are shuffled in
random order (fixed seed), so you label November blind, shuffled in with the
rest. This removes month-specific bias and protects the November lock.

If `litcurator prepare_labeling` has written a queue file, this labels that
pre-selected balanced sample; otherwise it falls back to the full unlabeled pool.

Dash note: every fixed-ID button (Relevant, Not Relevant, Back, Continue, Done)
stays mounted at all times -- the view swaps CONTENT and visibility, never the
button IDs the callbacks depend on. See the feedback_dash_inline_fixed_ids memory.

Launch:
    litcurator label_relevance [--start YYYY-MM-DD] [--end YYYY-MM-DD]
    python -m litcurator.apps.relevance_labeler [--ui_test]
"""

import argparse
import json
import random
import time

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback, ctx, dcc, html, no_update

from litcurator import db_interface
from litcurator.config import (
    LABELING_QUEUE_FILE,
    LITCURATOR_DB,
    RELEVANCE_BATCH_SIZE,
    UI_TEST_RELEVANCE_DB,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser()
_parser.add_argument("--ui_test", action="store_true",
                     help="use an isolated test DB, fresh each launch")
_parser.add_argument("--start", default="2025-01-01", metavar="YYYY-MM-DD")
_parser.add_argument("--end", default="2025-11-30", metavar="YYYY-MM-DD")
_cli_args, _ = _parser.parse_known_args()

UI_TEST = _cli_args.ui_test
DB_PATH = UI_TEST_RELEVANCE_DB if UI_TEST else LITCURATOR_DB
BATCH_SIZE = 10 if UI_TEST else RELEVANCE_BATCH_SIZE
DATE_START = _cli_args.start
DATE_END = _cli_args.end

if UI_TEST:
    db_interface.setup_ui_test_labeler_db(DB_PATH, mode="relevance")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

SHOW = {}
HIDE = {"display": "none"}

CARD_STYLE = {
    "backgroundColor": "#f7f5fc",
    "border": "1px solid #e3dcf2",
    "borderRadius": "10px",
    "padding": "1.75rem 2rem",
    "boxShadow": "0 1px 4px rgba(60,40,110,0.06)",
}
TITLE_STYLE = {
    "fontSize": "1.05rem",
    "fontWeight": 700,
    "lineHeight": "1.3",
    "color": "#2a2a3a",
    "marginBottom": "0.35rem",
}
GREEN = "#2d7a2d"
RED = "#c0392b"
BTN_RELEVANT_STYLE = {"backgroundColor": GREEN, "borderColor": GREEN,
                      "color": "white", "fontWeight": 600}
BTN_NOT_STYLE = {"backgroundColor": RED, "borderColor": RED,
                 "color": "white", "fontWeight": 600}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_queue():
    """Shuffled pmid list + total labeled count at load.

    If LABELING_QUEUE_FILE exists (written by `litcurator prepare_labeling`),
    uses that pre-selected list filtered to still-unlabeled articles.
    Otherwise falls back to all unlabeled articles in DATE_START..DATE_END.
    """
    conn = db_interface.get_connection(DB_PATH)
    try:
        labeled_count = conn.execute("SELECT COUNT(*) FROM human_labels").fetchone()[0]

        if not UI_TEST and LABELING_QUEUE_FILE.exists():
            payload = json.loads(LABELING_QUEUE_FILE.read_text())
            candidate_pmids = payload["pmids"]
            labeled_pmids = {
                r["pmid"] for r in conn.execute("SELECT pmid FROM human_labels").fetchall()
            }
            pmids = [p for p in candidate_pmids if p not in labeled_pmids]
        else:
            articles = db_interface.unlabeled_articles(conn, DATE_START, DATE_END)
            pmids = [a["pmid"] for a in articles]
            random.seed(42)
            random.shuffle(pmids)
    finally:
        conn.close()
    return pmids, labeled_count


def _fetch_article(pmid):
    conn = db_interface.get_connection(DB_PATH)
    try:
        return db_interface.get_article(conn, pmid)
    finally:
        conn.close()


def _fetch_label(pmid):
    conn = db_interface.get_connection(DB_PATH)
    try:
        row = conn.execute(
            "SELECT relevant FROM human_labels WHERE pmid = ?", (pmid,)
        ).fetchone()
        return row["relevant"] if row else None
    finally:
        conn.close()


def _fetch_labeled_count():
    conn = db_interface.get_connection(DB_PATH)
    try:
        return conn.execute("SELECT COUNT(*) FROM human_labels").fetchone()[0]
    finally:
        conn.close()


def _render_authors(authors_json):
    """Bold names + affiliation in parens, ' ; ' separated; first 2 and last 2 if
    more than 4 (matches the old Streamlit labeler and the review feed)."""
    if not authors_json:
        return "(unknown authors)"
    try:
        authors = json.loads(authors_json)
    except (json.JSONDecodeError, TypeError):
        return "(unknown authors)"
    if not authors:
        return "(unknown authors)"
    display = (authors[:2] + [{"name": "..."}] + authors[-2:]
               if len(authors) > 4 else authors)
    out = []
    for a in display:
        name = a.get("name", "")
        if not name:
            continue
        if out:
            out.append(" ; ")
        if name == "...":
            out.append("...")
        elif a.get("affiliation"):
            out.append(html.Span([html.B(name),
                                  html.Span(f" ({a['affiliation']})",
                                            className="fst-italic")]))
        else:
            out.append(html.B(name))
    return out or "(unknown authors)"


# ---------------------------------------------------------------------------
# Content builders (return children only -- never button IDs)
# ---------------------------------------------------------------------------


def _article_left(article):
    """Title, journal (+ page range), authors -- the left column's article fields."""
    journal = article.get("journal") or "(journal unknown)"
    pages = article.get("pages")
    pages_span = (html.Span(f"   ·   pp. {pages}", className="fw-normal")
                  if pages else None)
    return [
        html.Div(article["title"], style=TITLE_STYLE),
        html.Div([journal, pages_span],
                 className="text-muted fst-italic mb-1", style={"fontSize": "0.95rem"}),
        html.Div(_render_authors(article.get("authors_json")),
                 className="text-muted", style={"fontSize": "0.85rem"}),
    ]


def _article_right(article):
    """DOI link + abstract (collapsed by default for relevance)."""
    doi = article.get("doi")
    doi_link = (
        html.A("View article ↗", href=f"https://doi.org/{doi}", target="_blank",
               className="fw-bold text-decoration-none",
               style={"fontSize": "1.05rem"})
        if doi else html.Span()
    )
    return [
        html.Div(doi_link, className="mb-2"),
        html.Details([
            html.Summary("Abstract", className="text-muted fw-bold",
                         style={"cursor": "pointer"}),
            html.Div(article.get("abstract") or "(no abstract)",
                     className="mt-2",
                     style={"whiteSpace": "pre-wrap", "fontSize": "0.9rem",
                            "lineHeight": "1.55"}),
        ], open=False),
    ]


def _review_info(pos, total, current_label):
    label_text = {1: "Relevant", 0: "Not Relevant"}.get(current_label, "unlabeled")
    color = GREEN if current_label == 1 else RED if current_label == 0 else "#888"
    return html.Div([
        html.Span(f"Review mode · article {pos + 1} of {total} · current: ",
                  className="text-muted"),
        html.Span(label_text, style={"color": color, "fontWeight": 600}),
    ], className="small")


def _break_stats(labeled_count, batch_start_count, timing):
    batch_n = labeled_count - batch_start_count
    now = time.time()
    elapsed = now - timing.get("session_start", now)
    session_n = labeled_count - timing.get("session_start_count", labeled_count)
    per_article = elapsed / session_n if session_n > 0 else 0
    mins, secs = divmod(int(elapsed), 60)
    return [
        html.H5(f"Batch of {batch_n} done — take a break!", className="text-success"),
        html.P(f"{labeled_count} labeled overall  |  {mins}m {secs}s this session  "
               f"|  {per_article:.1f}s per article",
               className="text-muted"),
    ]


def _done_content(labeled_count, timing, all_done=False):
    now = time.time()
    elapsed = now - timing.get("session_start", now)
    session_n = labeled_count - timing.get("session_start_count", labeled_count)
    per_article = elapsed / session_n if session_n > 0 else 0
    mins, secs = divmod(int(elapsed), 60)
    head = ("All articles in this range are labeled!" if all_done
            else "See you next time! 👋")
    return [
        html.H4(head, className="text-secondary mb-3"),
        html.P(f"{labeled_count} labels overall  |  {session_n} this session  "
               f"|  {mins}m {secs}s  |  {per_article:.1f}s per article",
               className="text-muted"),
    ]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
app.title = "litcurator relevance labeler"

_ui_test_banner = (
    html.Div("UI TEST MODE — writes go to ui_test_relevance.db, not litcurator.db",
             style={"position": "fixed", "bottom": 0, "left": 0, "right": 0,
                    "background": "#fff3cd", "padding": "6px 16px",
                    "textAlign": "center", "zIndex": 999, "fontSize": "0.85em"})
    if UI_TEST else html.Div()
)

# Action buttons: defined ONCE, always mounted. cb_render only toggles their
# visibility and the surrounding content -- never adds/removes these IDs.
_btn_not = dbc.Button("✗ Not Relevant  (1)", id="btn-not-relevant", size="lg",
                      className="w-100", style=BTN_NOT_STYLE)
_btn_relevant = dbc.Button("✓ Relevant  (2)", id="btn-relevant", size="lg",
                           className="w-100", style=BTN_RELEVANT_STYLE)
_btn_back = dbc.Button("← Back", id="btn-back", color="link", size="sm",
                       className="px-0")
_btn_continue_labeling = dbc.Button("Continue labeling →", id="btn-continue-labeling",
                                    color="link", size="sm")
_btn_continue = dbc.Button("Continue", id="btn-continue", color="secondary")
_btn_done = dbc.Button("Done for now", id="btn-done", color="secondary", outline=True)
_btn_break_back = dbc.Button("← Back", id="btn-break-back", color="link")

_card = html.Div(id="card-area", children=dbc.Card(dbc.CardBody(
    dbc.Row([
        dbc.Col([
            html.Div(id="art-left"),
            html.Hr(),
            dbc.Row([
                dbc.Col(_btn_not, width=6),
                dbc.Col(_btn_relevant, width=6),
            ], className="g-2 mb-2"),
            html.Div("Keys:  1 = not relevant  ·  2 = relevant  ·  b = back",
                     className="small text-muted mb-2"),
            html.Div(id="review-info", className="mb-1"),
            html.Div([_btn_back, html.Span(" ", className="mx-2"),
                      _btn_continue_labeling], className="mb-1"),
            html.Div(id="batch-info", className="small text-muted fw-bold"),
        ], width=6),
        dbc.Col(html.Div(id="art-right"), width=6),
    ]),
), style=CARD_STYLE))

_break = html.Div(id="break-area", style=HIDE, children=html.Div([
    html.Div(id="break-stats"),
    html.Div([_btn_continue, html.Span(" ", className="mx-1"),
              _btn_done, html.Span(" ", className="mx-1"), _btn_break_back]),
]))

_done = html.Div(id="done-area", style=HIDE, children=html.Div(id="done-content"))

app.layout = dbc.Container([
    dcc.Store(id="queue-store"),
    dcc.Store(id="nav-store"),
    dcc.Store(id="timing-store"),
    dcc.Store(id="init-trigger", data=0),
    dcc.Store(id="keyboard-sink"),   # clientside keydown listener writes here (unused)

    dbc.Row([
        dbc.Col(html.Div("Relevance Labeling", className="mb-0 fw-bold text-muted",
                         style={"fontSize": "0.95rem", "letterSpacing": "0.04em",
                                "textTransform": "uppercase"}), width="auto"),
        dbc.Col(html.Small(id="header-stats", className="text-muted"),
                width="auto", align="end"),
    ], className="mt-2 mb-2 align-items-center"),

    html.Div([_card, _break, _done],
             style={"maxWidth": "1080px", "margin": "0 auto"}),
    _ui_test_banner,
], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("queue-store", "data"),
    Output("nav-store", "data"),
    Output("timing-store", "data"),
    Input("init-trigger", "data"),
)
def cb_init(_):
    pmids, labeled_count = _load_queue()
    now = time.time()
    return (
        {"pmids": pmids, "batch_start_count": labeled_count},
        {"mode": "label" if pmids else "all_done",
         "label_idx": 0, "review_history": [], "review_pos": 0, "tick": 0},
        {"session_start": now, "session_start_count": labeled_count},
    )


@callback(
    Output("art-left", "children"),
    Output("art-right", "children"),
    Output("review-info", "children"),
    Output("batch-info", "children"),
    Output("break-stats", "children"),
    Output("done-content", "children"),
    Output("card-area", "style"),
    Output("break-area", "style"),
    Output("done-area", "style"),
    Output("btn-back", "style"),
    Output("btn-continue-labeling", "style"),
    Output("header-stats", "children"),
    Input("nav-store", "data"),
    State("queue-store", "data"),
    State("timing-store", "data"),
)
def cb_render(nav, queue, timing):
    if nav is None or queue is None:
        return (no_update,) * 12

    mode = nav["mode"]
    pmids = queue["pmids"]
    batch_start_count = queue["batch_start_count"]
    labeled_count = _fetch_labeled_count()
    remaining = len(pmids) - nav["label_idx"]
    header = f"{labeled_count} labeled  |  {remaining} remaining in pool"

    # Defaults: everything hidden/blank, overridden per mode below.
    out = {
        "art_left": no_update, "art_right": no_update,
        "review_info": "", "batch_info": "", "break_stats": "", "done_content": "",
        "card_style": HIDE, "break_style": HIDE, "done_style": HIDE,
        "back_style": HIDE, "fwd_style": HIDE, "header": header,
    }

    if mode in ("session_done", "all_done"):
        out["done_style"] = SHOW
        out["done_content"] = _done_content(labeled_count, timing,
                                            all_done=(mode == "all_done"))
    elif mode == "batch_done":
        out["break_style"] = SHOW
        out["break_stats"] = _break_stats(labeled_count, batch_start_count, timing)
    elif mode == "review":
        history = nav["review_history"]
        pos = nav["review_pos"]
        article = _fetch_article(history[pos])
        out["card_style"] = SHOW
        out["art_left"] = _article_left(article)
        out["art_right"] = _article_right(article)
        out["review_info"] = _review_info(pos, len(history), _fetch_label(history[pos]))
        out["back_style"] = SHOW if pos > 0 else HIDE
        out["fwd_style"] = SHOW
    else:  # label
        label_idx = nav["label_idx"]
        if label_idx >= len(pmids):
            out["done_style"] = SHOW
            out["done_content"] = _done_content(labeled_count, timing, all_done=True)
        else:
            article = _fetch_article(pmids[label_idx])
            out["card_style"] = SHOW
            out["art_left"] = _article_left(article)
            out["art_right"] = _article_right(article)
            out["batch_info"] = f"Batch: {labeled_count - batch_start_count + 1} / {BATCH_SIZE}"
            out["back_style"] = SHOW if nav["review_history"] else HIDE

    return (out["art_left"], out["art_right"], out["review_info"], out["batch_info"],
            out["break_stats"], out["done_content"], out["card_style"],
            out["break_style"], out["done_style"], out["back_style"],
            out["fwd_style"], out["header"])


@callback(
    Output("nav-store", "data", allow_duplicate=True),
    Output("queue-store", "data", allow_duplicate=True),
    Input("btn-relevant", "n_clicks"),
    Input("btn-not-relevant", "n_clicks"),
    Input("btn-back", "n_clicks"),
    Input("btn-continue-labeling", "n_clicks"),
    Input("btn-continue", "n_clicks"),
    Input("btn-done", "n_clicks"),
    Input("btn-break-back", "n_clicks"),
    State("nav-store", "data"),
    State("queue-store", "data"),
    prevent_initial_call=True,
)
def cb_action(n_rel, n_not, n_back, n_fwd, n_cont, n_done, n_break_back, nav, queue):
    triggered = ctx.triggered_id
    if not triggered or nav is None:
        return no_update, no_update

    if triggered in ("btn-relevant", "btn-not-relevant"):
        return _on_label(triggered, nav, queue)
    if triggered == "btn-back":
        return _on_back(nav), no_update
    if triggered == "btn-continue-labeling":
        return {**nav, "mode": "label"}, no_update
    if triggered == "btn-continue":
        return (
            {**nav, "mode": "label", "review_history": [], "review_pos": 0},
            {**queue, "batch_start_count": _fetch_labeled_count()},
        )
    if triggered == "btn-done":
        return {**nav, "mode": "session_done"}, no_update
    if triggered == "btn-break-back":
        history = nav["review_history"]
        if not history:
            return no_update, no_update
        return {**nav, "mode": "review", "review_pos": len(history) - 1}, no_update

    return no_update, no_update


def _on_label(triggered, nav, queue):
    relevant = 1 if triggered == "btn-relevant" else 0

    if nav["mode"] == "review":
        pmid = nav["review_history"][nav["review_pos"]]
        conn = db_interface.get_connection(DB_PATH)
        try:
            db_interface.set_relevance_label(conn, pmid, relevant)
        finally:
            conn.close()
        return {**nav, "tick": nav.get("tick", 0) + 1}, no_update

    label_idx = nav["label_idx"]
    pmids = queue["pmids"]
    pmid = pmids[label_idx]

    conn = db_interface.get_connection(DB_PATH)
    try:
        db_interface.set_relevance_label(conn, pmid, relevant)
        labeled_count = conn.execute("SELECT COUNT(*) FROM human_labels").fetchone()[0]
    finally:
        conn.close()

    new_history = nav["review_history"] + [pmid]
    new_idx = label_idx + 1
    batch_count = labeled_count - queue["batch_start_count"]

    if new_idx >= len(pmids):
        new_mode = "all_done"
    elif batch_count >= BATCH_SIZE:
        new_mode = "batch_done"
    else:
        new_mode = "label"

    return ({**nav, "mode": new_mode, "label_idx": new_idx,
             "review_history": new_history}, no_update)


def _on_back(nav):
    history = nav["review_history"]
    if not history:
        return no_update
    if nav["mode"] == "review":
        pos = nav["review_pos"]
        return {**nav, "review_pos": pos - 1} if pos > 0 else no_update
    return {**nav, "mode": "review", "review_pos": len(history) - 1}


# ---------------------------------------------------------------------------
# Keyboard shortcuts (clientside)
# ---------------------------------------------------------------------------
# One document-level keydown listener, installed once on load. It clicks the
# button matching the pressed key -- which works precisely because every button
# is always in the DOM (the fixed-id design). offsetParent != null skips buttons
# that are currently hidden (e.g. Back when there is no history), and we only act
# when the card is the visible panel so a stray keypress can't label during a break.
_KEYMAP = {"1": "btn-not-relevant", "2": "btn-relevant", "b": "btn-back"}

app.clientside_callback(
    """
    function(trigger) {
        if (window.__litcurator_kb__) { return ''; }
        window.__litcurator_kb__ = true;
        const KEYMAP = %s;
        document.addEventListener('keydown', function(e) {
            const tag = (e.target.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea' ||
                e.ctrlKey || e.metaKey || e.altKey) { return; }
            const card = document.getElementById('card-area');
            const brk = document.getElementById('break-area');
            const cardVisible = card && card.style.display !== 'none';
            const brkVisible = brk && brk.style.display !== 'none';
            let id = null;
            if (cardVisible) { id = KEYMAP[e.key]; }
            else if (brkVisible && (e.key === 'Enter' || e.key === 'c')) { id = 'btn-continue'; }
            if (id) {
                const el = document.getElementById(id);
                if (el && el.offsetParent !== null) { e.preventDefault(); el.click(); el.blur(); }
            }
        });
        return '';
    }
    """ % json.dumps(_KEYMAP),
    Output("keyboard-sink", "data"),
    Input("init-trigger", "data"),
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_app(start="2025-01-01", end="2025-11-30", port=8054, debug=False):
    global DATE_START, DATE_END
    DATE_START = start
    DATE_END = end
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    run_app(debug=True)
