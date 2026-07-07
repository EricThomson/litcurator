"""
curation_labeler.py -- curation rating of relevant articles.

Shows articles with human_labels.relevant=1 and no curation_label, one at a
time, date-masked. Rate each:
  0 = didn't make the cut
  1-5 = above the noise (1) to can't miss (5)

Skip re-inserts the article at a uniformly random spot in a rolling window
CURATION_SKIP_MIN_GAP..CURATION_SKIP_MAX_GAP articles ahead (for when you're
unsure), so skipped papers come back later this session but do not recur right
away and do not all pile up at the end of the pool.

Date masking: pub_date_iso is not shown. Articles are shuffled with a fixed
seed, so November papers are labeled blind, shuffled in with the rest.

Dash note: every fixed-ID button (0-5 ratings, Skip, Back, Continue, Done)
stays mounted at all times -- the view swaps CONTENT and visibility, never the
button IDs the callbacks depend on. See the feedback_dash_inline_fixed_ids memory.

Launch:
    litcurator label_curation
    python -m litcurator.apps.curation_labeler [--ui_test]
"""

import argparse
import json
import random
import time

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback, ctx, dcc, html, no_update

from litcurator import db_interface
from litcurator.config import (
    CURATION_BATCH_SIZE,
    CURATION_SKIP_MAX_GAP,
    CURATION_SKIP_MIN_GAP,
    LITCURATOR_DB,
    UI_TEST_CURATION_DB,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser()
_parser.add_argument("--ui_test", action="store_true",
                     help="use an isolated test DB, fresh each launch")
_cli_args, _ = _parser.parse_known_args()

UI_TEST = _cli_args.ui_test
DB_PATH = UI_TEST_CURATION_DB if UI_TEST else LITCURATOR_DB
BATCH_SIZE = 10 if UI_TEST else CURATION_BATCH_SIZE

if UI_TEST:
    db_interface.setup_ui_test_labeler_db(DB_PATH, mode="curation")

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
# Rating colors (match review_feed): 0=grey, 1=purple ... 5=red.
RATING_COLORS = {0: "#888888", 1: "#6c3483", 2: "#1a3a8f",
                 3: "#f0b429", 4: "#e05c1a", 5: "#b01010"}


def _rating_style(v):
    return {"backgroundColor": RATING_COLORS[v], "borderColor": RATING_COLORS[v],
            "color": "white", "fontWeight": 600}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_queue():
    """Shuffled pmid list of relevant+uncurated articles + total curated count."""
    conn = db_interface.get_connection(DB_PATH)
    try:
        articles = db_interface.relevant_unlabeled_curation(conn)
        curated_count = conn.execute(
            "SELECT COUNT(*) FROM human_labels WHERE curation_label IS NOT NULL"
        ).fetchone()[0]
    finally:
        conn.close()
    pmids = [a["pmid"] for a in articles]
    random.seed(42)
    random.shuffle(pmids)
    return pmids, curated_count


def _fetch_article(pmid):
    conn = db_interface.get_connection(DB_PATH)
    try:
        return db_interface.get_article(conn, pmid)
    finally:
        conn.close()


def _fetch_curation_label(pmid):
    conn = db_interface.get_connection(DB_PATH)
    try:
        row = conn.execute(
            "SELECT curation_label FROM human_labels WHERE pmid = ?", (pmid,)
        ).fetchone()
        return row["curation_label"] if row else None
    finally:
        conn.close()


def _fetch_curated_count():
    conn = db_interface.get_connection(DB_PATH)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM human_labels WHERE curation_label IS NOT NULL"
        ).fetchone()[0]
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
    """DOI link + abstract (OPEN by default for curation)."""
    doi = article.get("doi")
    doi_link = (
        html.A("View article ↗", href=f"https://doi.org/{doi}", target="_blank",
               className="fw-bold text-decoration-none", style={"fontSize": "1.05rem"})
        if doi else html.Span()
    )
    return [
        html.Div(doi_link, className="mb-2"),
        html.Details([
            html.Summary("Abstract", className="text-muted fw-bold",
                         style={"cursor": "pointer"}),
            html.Div(article.get("abstract") or "(no abstract)", className="mt-2",
                     style={"whiteSpace": "pre-wrap", "fontSize": "0.9rem",
                            "lineHeight": "1.55"}),
        ], open=True),
    ]


def _review_info(pos, total, current_label):
    color = RATING_COLORS.get(current_label, "#888")
    return html.Div([
        html.Span(f"Review mode · article {pos + 1} of {total} · current: ",
                  className="text-muted"),
        html.Span(str(current_label), style={"color": color, "fontWeight": 600}),
    ], className="small")


def _break_stats(curated_count, batch_start_count, timing):
    batch_n = curated_count - batch_start_count
    now = time.time()
    elapsed = now - timing.get("session_start", now)
    session_n = curated_count - timing.get("session_start_count", curated_count)
    per_article = elapsed / session_n if session_n > 0 else 0
    mins, secs = divmod(int(elapsed), 60)
    return [
        html.H5(f"Batch of {batch_n} done — take a break!", className="text-success"),
        html.P(f"{curated_count} rated overall  |  {mins}m {secs}s this session  "
               f"|  {per_article:.1f}s per article", className="text-muted"),
    ]


def _done_content(curated_count, timing, all_done=False):
    now = time.time()
    elapsed = now - timing.get("session_start", now)
    session_n = curated_count - timing.get("session_start_count", curated_count)
    per_article = elapsed / session_n if session_n > 0 else 0
    mins, secs = divmod(int(elapsed), 60)
    head = ("All relevant articles are rated!" if all_done
            else "See you next time! 👋")
    return [
        html.H4(head, className="text-secondary mb-3"),
        html.P(f"{curated_count} curation labels overall  |  {session_n} this session  "
               f"|  {mins}m {secs}s  |  {per_article:.1f}s per article",
               className="text-muted"),
    ]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
app.title = "litcurator curation labeler"

_ui_test_banner = (
    html.Div("UI TEST MODE — writes go to ui_test_curation.db, not litcurator.db",
             style={"position": "fixed", "bottom": 0, "left": 0, "right": 0,
                    "background": "#fff3cd", "padding": "6px 16px",
                    "textAlign": "center", "zIndex": 999, "fontSize": "0.85em"})
    if UI_TEST else html.Div()
)

# Rating buttons (0 reject + 1-5), defined ONCE, always mounted.
_btn_0 = dbc.Button("0 — didn't make it", id="btn-0", size="lg",
                    className="w-100", style=_rating_style(0))
_rating_btns = [
    dbc.Button(str(v), id=f"btn-{v}", size="lg", className="w-100", style=_rating_style(v))
    for v in range(1, 6)
]
_btn_skip = dbc.Button("Skip (s)", id="btn-skip", color="link", size="sm",
                       className="px-0")
_btn_back = dbc.Button("← Back", id="btn-back", color="link", size="sm", className="px-0")
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
            dbc.Row([dbc.Col(_btn_0, width=8)], className="mb-2"),
            html.Div("1 = above the noise   ·   3 = very cool   ·   5 = can't miss",
                     className="small text-muted mb-1"),
            dbc.Row([dbc.Col(b, width=2) for b in _rating_btns],
                    className="g-2 mb-2"),
            html.Div("Keys:  0-5 = rating  ·  s = skip  ·  b = back",
                     className="small text-muted mb-2"),
            html.Div(id="review-info", className="mb-1"),
            html.Div([_btn_back, html.Span(" ", className="mx-2"),
                      _btn_continue_labeling, html.Span(" ", className="mx-2"),
                      _btn_skip], className="mb-1"),
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
        dbc.Col(html.Div("Curation Labeling", className="mb-0 fw-bold text-muted",
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
    pmids, curated_count = _load_queue()
    now = time.time()
    return (
        {"pmids": pmids, "batch_start_count": curated_count},
        {"mode": "label" if pmids else "all_done",
         "label_idx": 0, "review_history": [], "review_pos": 0, "tick": 0},
        {"session_start": now, "session_start_count": curated_count},
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
    Output("btn-skip", "style"),
    Output("header-stats", "children"),
    Input("nav-store", "data"),
    State("queue-store", "data"),
    State("timing-store", "data"),
)
def cb_render(nav, queue, timing):
    if nav is None or queue is None:
        return (no_update,) * 13

    mode = nav["mode"]
    pmids = queue["pmids"]
    batch_start_count = queue["batch_start_count"]
    curated_count = _fetch_curated_count()
    remaining = len(pmids) - nav["label_idx"]
    header = f"{curated_count} rated  |  {remaining} remaining in pool"

    out = {
        "art_left": no_update, "art_right": no_update,
        "review_info": "", "batch_info": "", "break_stats": "", "done_content": "",
        "card_style": HIDE, "break_style": HIDE, "done_style": HIDE,
        "back_style": HIDE, "fwd_style": HIDE, "skip_style": HIDE, "header": header,
    }

    if mode in ("session_done", "all_done"):
        out["done_style"] = SHOW
        out["done_content"] = _done_content(curated_count, timing,
                                            all_done=(mode == "all_done"))
    elif mode == "batch_done":
        out["break_style"] = SHOW
        out["break_stats"] = _break_stats(curated_count, batch_start_count, timing)
    elif mode == "review":
        history = nav["review_history"]
        pos = nav["review_pos"]
        article = _fetch_article(history[pos])
        out["card_style"] = SHOW
        out["art_left"] = _article_left(article)
        out["art_right"] = _article_right(article)
        out["review_info"] = _review_info(pos, len(history),
                                          _fetch_curation_label(history[pos]))
        out["back_style"] = SHOW if pos > 0 else HIDE
        out["fwd_style"] = SHOW
    else:  # label
        label_idx = nav["label_idx"]
        if label_idx >= len(pmids):
            out["done_style"] = SHOW
            out["done_content"] = _done_content(curated_count, timing, all_done=True)
        else:
            article = _fetch_article(pmids[label_idx])
            out["card_style"] = SHOW
            out["art_left"] = _article_left(article)
            out["art_right"] = _article_right(article)
            out["batch_info"] = f"Batch: {curated_count - batch_start_count + 1} / {BATCH_SIZE}"
            out["back_style"] = SHOW if nav["review_history"] else HIDE
            out["skip_style"] = SHOW

    return (out["art_left"], out["art_right"], out["review_info"], out["batch_info"],
            out["break_stats"], out["done_content"], out["card_style"],
            out["break_style"], out["done_style"], out["back_style"],
            out["fwd_style"], out["skip_style"], out["header"])


@callback(
    Output("nav-store", "data", allow_duplicate=True),
    Output("queue-store", "data", allow_duplicate=True),
    Input("btn-0", "n_clicks"),
    Input("btn-1", "n_clicks"),
    Input("btn-2", "n_clicks"),
    Input("btn-3", "n_clicks"),
    Input("btn-4", "n_clicks"),
    Input("btn-5", "n_clicks"),
    Input("btn-skip", "n_clicks"),
    Input("btn-back", "n_clicks"),
    Input("btn-continue-labeling", "n_clicks"),
    Input("btn-continue", "n_clicks"),
    Input("btn-done", "n_clicks"),
    Input("btn-break-back", "n_clicks"),
    State("nav-store", "data"),
    State("queue-store", "data"),
    prevent_initial_call=True,
)
def cb_action(*args):
    nav, queue = args[-2], args[-1]
    triggered = ctx.triggered_id
    if not triggered or nav is None:
        return no_update, no_update

    if triggered and triggered.startswith("btn-") and triggered[4:].isdigit():
        return _on_rate(int(triggered[4:]), nav, queue)
    if triggered == "btn-skip":
        return _on_skip(nav, queue)
    if triggered == "btn-back":
        return _on_back(nav), no_update
    if triggered == "btn-continue-labeling":
        return {**nav, "mode": "label"}, no_update
    if triggered == "btn-continue":
        return (
            {**nav, "mode": "label", "review_history": [], "review_pos": 0},
            {**queue, "batch_start_count": _fetch_curated_count()},
        )
    if triggered == "btn-done":
        return {**nav, "mode": "session_done"}, no_update
    if triggered == "btn-break-back":
        history = nav["review_history"]
        if not history:
            return no_update, no_update
        return {**nav, "mode": "review", "review_pos": len(history) - 1}, no_update

    return no_update, no_update


def _on_rate(rating, nav, queue):
    if nav["mode"] == "review":
        pmid = nav["review_history"][nav["review_pos"]]
        conn = db_interface.get_connection(DB_PATH)
        try:
            db_interface.set_curation_label(conn, pmid, rating)
        finally:
            conn.close()
        return {**nav, "tick": nav.get("tick", 0) + 1}, no_update

    label_idx = nav["label_idx"]
    pmids = queue["pmids"]
    pmid = pmids[label_idx]

    conn = db_interface.get_connection(DB_PATH)
    try:
        db_interface.set_curation_label(conn, pmid, rating)
        curated_count = conn.execute(
            "SELECT COUNT(*) FROM human_labels WHERE curation_label IS NOT NULL"
        ).fetchone()[0]
    finally:
        conn.close()

    new_history = nav["review_history"] + [pmid]
    new_idx = label_idx + 1
    batch_count = curated_count - queue["batch_start_count"]

    if new_idx >= len(pmids):
        new_mode = "all_done"
    elif batch_count >= BATCH_SIZE:
        new_mode = "batch_done"
    else:
        new_mode = "label"

    return ({**nav, "mode": new_mode, "label_idx": new_idx,
             "review_history": new_history}, no_update)


def _on_skip(nav, queue):
    """Re-insert the skipped article at a uniformly random spot in a rolling
    window CURATION_SKIP_MIN_GAP..CURATION_SKIP_MAX_GAP articles ahead of the
    cursor -- far enough not to recur right away, near enough that it comes back
    this session instead of being flung to the end.

    Near the end of the pool the full window no longer fits, so both gaps shrink
    to whatever is ahead (the min gap to a third of it, so the window stays open):
    skipped papers stay spread across the tail rather than stacking at the very
    end. Skip the last handful and there is nowhere left to put them -- that one
    is on you."""
    label_idx = nav["label_idx"]
    pmids = list(queue["pmids"])
    pmid = pmids.pop(label_idx)

    ahead = len(pmids) - label_idx                      # insertable slots in front
    min_gap = min(CURATION_SKIP_MIN_GAP, ahead // 3)
    max_gap = min(CURATION_SKIP_MAX_GAP, ahead)
    insert_at = random.randint(label_idx + min_gap, label_idx + max_gap)
    pmids.insert(insert_at, pmid)

    # label_idx unchanged: the next article slides into the current slot.
    return {**nav}, {**queue, "pmids": pmids}


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
# that are currently hidden (e.g. Skip/Back in review mode), and we only act when
# the card is the visible panel so a stray keypress can't rate during a break.
_KEYMAP = {"0": "btn-0", "1": "btn-1", "2": "btn-2", "3": "btn-3",
           "4": "btn-4", "5": "btn-5", "s": "btn-skip", "b": "btn-back"}

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


def run_app(port=8055, debug=False):
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    run_app(debug=True)
