"""
review_feed.py -- review the judge's output and flag papers.

Reads the most-recent curation evaluation per paper (db_interface.latest_curation),
shows one score-sorted card each, and lets you enter user_score (your own estimated
interest) plus a note, which error_analysis reads (the judge never does). Saving appends a flag via
db_interface.insert_flag, keyed to the evaluation it corrects.

Flags are append-only: re-saving a paper appends a new flag and the latest wins,
so there is no "clear" -- to correct a number, just save the right one.

The delta (user_score - judge_score) is the residual: large |delta| clusters are
where the profile is most wrong. Flags are discovery data; they never feed the judge.

Launch:  litcurator review        (or)  python -m litcurator.apps.review_feed
"""

import argparse
import json
import math
import sqlite3

import dash_bootstrap_components as dbc
from dash import ALL, MATCH, Dash, Input, Output, State, callback, ctx, dcc, html, no_update

from litcurator import db_interface
from litcurator.config import (ARCHIVED_NOTES_FILE, LEVELS_BUCKET_DESCRIPTION,
                               LEVELS_BUCKET_NAME, REVIEW_CARDS_PER_PAGE,
                               REVIEW_CARDS_PER_PAGE_CHOICES)

# Optional CLI dates pre-fill the in-app date picker. parse_known_args so Dash's
# own flags do not choke. Blank = show all.
_parser = argparse.ArgumentParser()
_parser.add_argument("--start", default=None, metavar="YYYY-MM-DD")
_parser.add_argument("--end", default=None, metavar="YYYY-MM-DD")
_cli_args, _ = _parser.parse_known_args()
CLI_START = _cli_args.start
CLI_END = _cli_args.end


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_feed(start=None, end=None):
    """Latest curation evaluation per paper in the window, each merged with its
    latest flag (if any) and whether it is in the Levels Bucket. Sorted by score desc
    (from latest_curation)."""
    conn = db_interface.get_connection()
    try:
        items = db_interface.latest_curation(conn, start, end)
        flags = {f["pmid"]: f for f in db_interface.get_flags(conn, start=start, end=end)}
        bucketed = db_interface.levels_bucket_pmids(conn)
    finally:
        conn.close()
    for it in items:
        it["flag"] = flags.get(it["pmid"])
        it["in_bucket"] = it["pmid"] in bucketed
    return items


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _score_color(score):
    if score < 0.2:   return "#888888"
    if score < 0.4:   return "#6c3483"
    if score < 0.6:   return "#1a3a8f"
    if score < 0.8:   return "#f0b429"
    if score < 0.9:   return "#e05c1a"
    return "#b01010"


_ARCHIVED_NOTES = None


def _archived_notes():
    """{pmid: note} from the last reset, loaded once. Empty when the file is absent, which is
    the normal state -- it only exists after a reset has wiped the flags table, and it goes
    stale harmlessly as real flags accumulate again (a live flag's note always wins)."""
    global _ARCHIVED_NOTES
    if _ARCHIVED_NOTES is None:
        try:
            _ARCHIVED_NOTES = json.loads(ARCHIVED_NOTES_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _ARCHIVED_NOTES = {}
    return _ARCHIVED_NOTES


def _render_authors(authors_json):
    """All authors if <=4, else first 2 + '...' + last 2 (house convention)."""
    if not authors_json:
        return None
    try:
        authors = json.loads(authors_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not authors:
        return None
    display = (authors[:2] + [{"name": "...", "affiliation": ""}] + authors[-2:]
               if len(authors) > 4 else authors)
    rendered = []
    for a in display:
        name = a.get("name", "")
        if not name:
            continue
        if name == "...":
            rendered.append(html.Span("..."))
        elif a.get("affiliation"):
            rendered.append(html.Span([html.B(name), f" ({a['affiliation']})"]))
        else:
            rendered.append(html.B(name))
    if not rendered:
        return None
    out = []
    for i, el in enumerate(rendered):
        if i > 0:
            out.append(" ; ")
        out.append(el)
    return out


def _flag_badge(flag, in_bucket=False):
    """Inner content of the 'your flag' badge, or None when unflagged. The OUTER
    span (carrying the fixed flag-badge id) is always in the DOM, so a save or
    delete fills/empties it surgically instead of rebuilding the card. A paper in
    the Levels Bucket also carries the bucket's tag, visible with the panel closed."""
    if not flag:
        return None
    score = html.Span(f"you: {flag['user_score']:.2f}  (delta {flag['delta']:+.2f})",
                      className="badge bg-danger")
    if not in_bucket:
        return score
    return [score, html.Span(LEVELS_BUCKET_NAME, title=LEVELS_BUCKET_DESCRIPTION,
                             className="badge bg-dark ms-1")]


def _remove_btn_style(flagged, in_bucket=False):
    """Show the Remove-flag button only when flagged and NOT in the Levels Bucket (a
    bucketed paper is taken out first, then removed). Toggled via display so the
    fixed id stays mounted (no conditional render -> no n_clicks-reset glitch)."""
    return {} if flagged and not in_bucket else {"display": "none"}


def _save_btn_look(saved):
    """(label, color) of the Save button: "Saved" while the panel matches what is stored,
    "Save" once there is something to save."""
    return ("Saved", "success") if saved else ("Save", "primary")


def _stored_flag(flag):
    """The score and note as last saved, for the card's stored-flag store; None if unflagged.
    The Save button compares the boxes against this to show Saved or Save."""
    return {"user_score": flag["user_score"], "note": flag["note"]} if flag else None


def _per_card(slots, pmid, value):
    """For an ALL output: set value on one card's slot and leave every other card alone."""
    return [value if s["id"]["pmid"] == pmid else no_update for s in slots]


def _hold(slots):
    """For an ALL output: leave every card alone."""
    return [no_update] * len(slots)


def _bucket_btn_styles(in_bucket):
    """(Add, Take out) button styles: exactly one shows. Both stay mounted and swap
    via display, for the same reason as _remove_btn_style."""
    hidden = {"display": "none"}
    return (hidden, {}) if in_bucket else ({}, hidden)


def _coerce_floor(min_score):
    """A min-score field value (number, blank, or junk) -> a clamped [0,1] floor.
    Blank or invalid means no floor (show everything)."""
    try:
        return max(0.0, min(1.0, float(min_score)))
    except (TypeError, ValueError):
        return 0.0


def _apply_floor(items, min_score):
    """Split the score-sorted feed into (shown, floor): papers at or above the
    min-score floor. floor == 0 shows everything."""
    floor = _coerce_floor(min_score)
    shown = [it for it in items if it["score"] >= floor] if floor > 0 else items
    return shown, floor


def _summary_text(all_items, shown_items, start, end, floor):
    """The one-line feed summary. Single source of truth, used by the initial
    render and by surgical saves so the two never drift. Surfaces how many papers
    the floor is HIDING, so a focused pass never silently buries false negatives."""
    n_flagged = sum(1 for it in shown_items if it.get("flag"))
    hidden = len(all_items) - len(shown_items)
    hidden_txt = f"  |  {hidden} hidden below {floor:g}" if hidden else ""
    rng = f"  |  {start or 'start'} to {end or 'end'}" if (start or end) else ""
    return f"{len(shown_items)} papers  |  {n_flagged} flagged{hidden_txt}{rng}"


def _render_card(item, rank, total):
    score = item["score"]
    pmid = item["pmid"]
    flag = item.get("flag")
    flagged = flag is not None
    in_bucket = item.get("in_bucket", False)

    badge = html.Span(
        f"{score:.2f}",
        style={"backgroundColor": _score_color(score), "color": "white",
               "padding": "2px 10px", "borderRadius": "4px",
               "fontWeight": "bold", "fontSize": "1.05em", "marginRight": "10px"},
    )
    decision = html.Span(item.get("surface_decision") or "",
                         className="badge bg-light text-dark me-2")
    badges = [badge, decision]
    if item.get("curation_label") is not None:
        badges.append(html.Span(f"your label: {item['curation_label']}/5",
                                className="badge bg-secondary me-2"))
    # Always present (fixed id) so a save/delete can fill or empty it surgically.
    badges.append(html.Span(_flag_badge(flag, in_bucket), id={"type": "flag-badge", "pmid": pmid}))

    journal_em = html.Em(item.get("journal") or "(journal unknown)",
                         style={"fontSize": "1rem", "fontWeight": "500", "color": "#5a4b8a"})
    pages_str = f"  |  pp. {item['pages']}" if item.get("pages") else ""
    meta_children = [f"{item.get('pub_date_iso') or ''}  |  pmid {pmid}{pages_str}"]
    if item.get("doi"):
        meta_children += ["  |  ", html.A("DOI", href=f"https://doi.org/{item['doi']}",
                                          target="_blank", className="text-decoration-none")]
    meta_node = html.Div(
        [journal_em, html.Span(meta_children, className="small text-muted ms-2")],
        className="d-flex align-items-baseline flex-wrap")

    pre = flag or {}
    # A paper with no live flag falls back to the note you wrote before the last reset, so the
    # box opens with your own words rather than empty. Only ever a PREFILL: saving writes a
    # normal flag, and once one exists its note wins.
    prior_note = "" if flag else _archived_notes().get(pmid, "")
    remove_btn = dbc.Button("Remove flag", id={"type": "flag-delete", "pmid": pmid},
                            color="danger", outline=True, size="sm",
                            style=_remove_btn_style(flagged, in_bucket))
    # The Levels Bucket buttons. Add saves the panel's score and note STRAIGHT into the bucket,
    # no Save first, so the paper never passes through error_analysis's pool. Take out is the
    # undo, for mis-clicks and regrets. Exactly one shows at a time. They sit up beside the Flag
    # button, away from Save: bucketing is its own decision, and a bucket button right under the
    # note got clicked when Save was meant.
    add_style, out_style = _bucket_btn_styles(in_bucket)
    save_label, save_color = _save_btn_look(saved=flagged)
    bucket_btns = [
        dbc.Button(f"Add to {LEVELS_BUCKET_NAME}",
                   id={"type": "bucket-add", "pmid": pmid},
                   title=LEVELS_BUCKET_DESCRIPTION, color="dark", outline=True, size="sm",
                   className="me-2", style=add_style),
        dbc.Button(f"Take out of {LEVELS_BUCKET_NAME}",
                   id={"type": "bucket-out", "pmid": pmid},
                   title="Your score and note stay, as an ordinary flag for error_analysis.",
                   color="secondary", outline=True, size="sm", className="me-2",
                   style=out_style),
    ]
    # The FIRST keystroke of an edit reaches Dash; the rest wait until you leave the box. While
    # the card reads Saved its boxes report every keystroke (debounce off), so the first one can
    # flip the button to Save; that same callback then switches them to report on leaving the
    # box or Enter (debounce on), and a save switches them back (SAVE_BUTTON_LOOK_JS). Measured
    # (sandbox/review_typing_profiler): every callback costs time for every card on the page,
    # so a callback on every keystroke made typing sticky; this way an edit costs one.
    report_keystrokes = flagged
    flag_panel = dbc.Collapse(
        dbc.Card(dbc.CardBody([
            html.Div("Your estimated interest (0.0 = no interest, 1.0 = must read)",
                     className="small fw-semibold mb-2"),
            dbc.Row([
                dbc.Col(dbc.Input(
                    id={"type": "flag-score", "pmid": pmid},
                    type="text",
                    value=pre.get("user_score", None),
                    placeholder="0.0 - 1.0", size="sm",
                    debounce=not report_keystrokes), width=3),
            ], className="g-2 align-items-center mb-2"),
            dbc.Label("Note", className="small mb-1"),
            dbc.Input(id={"type": "flag-note", "pmid": pmid}, type="text",
                      value=pre.get("note") or prior_note,
                      placeholder="e.g. ECoG, not single-unit", size="sm",
                      debounce=not report_keystrokes),
            *([html.Small("prefilled from your note before the last reset",
                          className="text-muted")] if prior_note else []),
            # Save goes right under the note, where the hand goes after writing it.
            html.Div([
                dbc.Button(save_label, id={"type": "flag-save", "pmid": pmid},
                           color=save_color, size="sm", className="me-2"),
                remove_btn,
            ], className="d-flex align-items-center mt-2"),
        ]), color="light", className="mt-2"),
        id={"type": "flag-collapse", "pmid": pmid},
        is_open=flagged)

    authors_line = _render_authors(item.get("authors_json"))

    return dbc.Card(dbc.CardBody([
        html.Div(badges, className="mb-1"),
        html.Div([html.Span(f"{rank}/{total}", className="text-muted small me-2"),
                  html.Strong(item["title"])]),
        meta_node,
        html.Div(authors_line, className="small text-muted mb-2") if authors_line
        else html.Div(className="mb-2"),
        html.Details([
            html.Summary("Abstract", className="small text-muted fw-bold"),
            html.Div(item.get("abstract") or "(no abstract)", className="small mt-1",
                     style={"whiteSpace": "pre-wrap"}),
        ], open=True),
        html.Div([html.Span("Why: ", className="text-muted small fw-bold"),
                  html.Span(item.get("rationale") or "", className="small")], className="mb-1 mt-2"),
        html.Div([html.Span("Possible Mismatch: ", className="text-muted small fw-bold"),
                  html.Span(item.get("possible_mismatch") or "", className="small")],
                 className="mb-1"),
        html.Div([dbc.Button("Flag / edit score",
                             id={"type": "flag-toggle", "pmid": pmid},
                             color="secondary", outline=True, size="sm", className="me-2"),
                  *bucket_btns],
                 className="d-flex align-items-center mt-2"),
        # Outside the panel, so a bucket click's "enter a score" message shows even when the
        # panel is closed.
        html.Div(id={"type": "flag-error", "pmid": pmid},
                 className="text-danger fw-bold mt-1"),
        # Every part of a card has an id of the same shape (type, pmid), so a callback can wire
        # one card to itself (MATCH). What a card needs beyond its components lives here: the
        # evaluation a flag is keyed to, and the score and note as last saved.
        dcc.Store(id={"type": "evaluation-id", "pmid": pmid}, data=item["evaluation_id"]),
        dcc.Store(id={"type": "stored-flag", "pmid": pmid}, data=_stored_flag(flag)),
        flag_panel,
    ]), className="mb-3",
        style={"backgroundColor": "#f3f0fa", "border": "1px solid #e3dcf2"})


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
app.title = "litcurator review"

def _page_size(selected):
    """Cards per page from the selector's value (a string; the default before any choice)."""
    return int(selected or REVIEW_CARDS_PER_PAGE)


def _pager(pager_id):
    return dbc.Pagination(id=pager_id, max_value=1, active_page=1, first_last=True,
                          previous_next=True, fully_expanded=False, size="sm",
                          className="my-2")


def _build_layout(start_date=None, end_date=None):
    return dbc.Container([
        dcc.Store(id="reload-feed", data=0),
        dbc.Row([
            dbc.Col(html.H3("Review feed", className="mb-0"), width="auto"),
            dbc.Col(html.Small(id="feed-summary", className="text-muted"), width="auto", align="end"),
        ], align="center", className="mt-3 mb-2"),
        dbc.Row([
            dbc.Col([
                html.Small("Filter by pub date (blank = all):", className="text-muted me-2"),
                dcc.DatePickerRange(id="date-filter", display_format="YYYY-MM-DD",
                                    start_date_placeholder_text="start", end_date_placeholder_text="end",
                                    start_date=start_date, end_date=end_date, clearable=True),
            ], width="auto"),
            dbc.Col([
                html.Small("Min score (0 = show all):", className="text-muted me-2"),
                dbc.Input(id="min-score", type="number", min=0, max=1, step="any",
                          value=0, debounce=True, size="sm",
                          style={"width": "90px", "display": "inline-block"}),
            ], width="auto", align="end"),
            dbc.Col([
                html.Small("Cards per page:", className="text-muted me-2",
                           title="More cards per page means every click waits longer: each "
                                 "callback costs time for every card on the page."),
                dbc.Select(id="page-size", size="sm", value=str(REVIEW_CARDS_PER_PAGE),
                           options=[{"label": str(n), "value": str(n)}
                                    for n in REVIEW_CARDS_PER_PAGE_CHOICES],
                           style={"width": "80px", "display": "inline-block"}),
            ], width="auto", align="end"),
        ], className="mb-2 align-items-end"),
        # The rank of the first card on screen, so changing the page size keeps your place.
        dcc.Store(id="feed-first-rank", data=1),
        dbc.Alert(id="flag-alert", is_open=False, duration=3000, color="success"),
        # The feed is paged (REVIEW_CARDS_PER_PAGE): every callback costs time in proportion to
        # the components on the page. Two pagers, kept in step: one to jump, one at the bottom
        # where you finish a page.
        _pager("feed-pager-top"),
        html.Div(id="feed-container"),
        _pager("feed-pager-bottom"),
        dcc.Store(id="scroll-sink"),
    ], fluid=True)


app.layout = _build_layout(CLI_START, CLI_END)


@callback(
    Output("feed-container", "children"),
    Output("feed-summary", "children"),
    Output("feed-pager-top", "max_value"),
    Output("feed-pager-top", "active_page"),
    Output("feed-pager-bottom", "max_value"),
    Output("feed-pager-bottom", "active_page"),
    Output("feed-first-rank", "data"),
    Input("reload-feed", "data"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("min-score", "value"),
    Input("feed-pager-top", "active_page"),
    Input("feed-pager-bottom", "active_page"),
    Input("page-size", "value"),
    State("feed-first-rank", "data"),
)
def cb_render_feed(_n, start, end, min_score, top_page, bottom_page, page_size, first_rank):
    """Render ONE page of the score-sorted feed. A pager click shows that page and moves the
    other pager to match; a reload (after removing a flag) stays on the page; a new page size
    shows the page holding the card that was first on screen; new filters go back to page 1.
    A pager is only written when its page actually changes, since writing it also scrolls to
    the top."""
    per_page = _page_size(page_size)
    trigger = ctx.triggered_id
    if trigger == "feed-pager-top":
        wanted = top_page
    elif trigger == "feed-pager-bottom":
        wanted = bottom_page
    elif trigger == "reload-feed":
        wanted = top_page
    elif trigger == "page-size":
        wanted = ((first_rank or 1) - 1) // per_page + 1
    else:
        wanted = 1   # first load, or the dates or min score changed

    items = _load_feed(start, end)
    shown, floor = _apply_floor(items, min_score) if items else ([], 0.0)
    n_pages = max(1, math.ceil(len(shown) / per_page))
    page = min(max(1, wanted or 1), n_pages)
    top_out = page if page != top_page else no_update
    bottom_out = page if page != bottom_page else no_update
    first = (page - 1) * per_page

    if not items:
        message = html.Div("No judged papers in this range. Run `litcurator run`, or widen the dates.",
                           className="text-muted")
        return message, "", n_pages, top_out, n_pages, bottom_out, 1
    summary = _summary_text(items, shown, start, end, floor)
    if not shown:
        message = html.Div(f"All {len(items)} papers scored below {floor:g}. Lower the min score "
                           f"to see them.", className="text-muted")
        return message, summary, n_pages, top_out, n_pages, bottom_out, 1

    on_this_page = shown[first:first + per_page]
    cards = [_render_card(it, rank, len(shown)) for rank, it in enumerate(on_this_page, first + 1)]
    return cards, summary, n_pages, top_out, n_pages, bottom_out, first + 1


# A new page starts at the top of the page.
app.clientside_callback(
    """function(top, bottom) { window.scrollTo(0, 0); return window.dash_clientside.no_update; }""",
    Output("scroll-sink", "data"),
    Input("feed-pager-top", "active_page"),
    Input("feed-pager-bottom", "active_page"),
    prevent_initial_call=True,
)


# Flag / edit score opens and closes its own card's panel, in the browser: one card (MATCH),
# no server round trip. Measured on 93 cards: 490 ms as an all-cards server callback, 312 ms
# like this, with the rest being the page-size cost that paging removes.
app.clientside_callback(
    """function(_n_clicks, is_open) { return !is_open; }""",
    Output({"type": "flag-collapse", "pmid": MATCH}, "is_open"),
    Input({"type": "flag-toggle", "pmid": MATCH}, "n_clicks"),
    State({"type": "flag-collapse", "pmid": MATCH}, "is_open"),
    prevent_initial_call=True,
)


def _parse_score(raw, doing):
    """(score, None) for a valid 0-1 entry, else (None, message). `doing` finishes the
    sentence "Enter a score before ...", so each button names its own action."""
    if raw is None or str(raw).strip() == "":
        return None, f"Enter a score (0.0 - 1.0) before {doing}."
    try:
        score = float(str(raw).strip())
    except ValueError:
        return None, f"'{raw}' is not a number -- enter a value 0.0 - 1.0."
    if not (0.0 <= score <= 1.0):
        return None, "Score must be between 0.0 and 1.0."
    return score, None


@callback(
    Output("feed-summary", "children", allow_duplicate=True),
    Output({"type": "flag-badge", "pmid": ALL}, "children"),
    Output({"type": "flag-delete", "pmid": ALL}, "style"),
    Output({"type": "flag-error", "pmid": ALL}, "children"),
    Output({"type": "bucket-add", "pmid": ALL}, "style"),
    Output({"type": "bucket-out", "pmid": ALL}, "style"),
    Output({"type": "stored-flag", "pmid": ALL}, "data"),
    Input({"type": "flag-save", "pmid": ALL}, "n_clicks"),
    Input({"type": "bucket-add", "pmid": ALL}, "n_clicks"),
    Input({"type": "bucket-out", "pmid": ALL}, "n_clicks"),
    State({"type": "flag-score", "pmid": ALL}, "value"),
    State({"type": "flag-note", "pmid": ALL}, "value"),
    State({"type": "evaluation-id", "pmid": ALL}, "data"),
    State("date-filter", "start_date"),
    State("date-filter", "end_date"),
    State("min-score", "value"),
    prevent_initial_call=True,
)
def cb_flag_panel_buttons(save_clicks, add_clicks, out_clicks, scores, notes,
                          evaluation_ids, start, end, min_score):
    """The flag panel's three writing buttons. Save records an ordinary flag for
    error_analysis (or, for a paper already in the Levels Bucket, a newer version in the
    bucket). Add to Levels Bucket puts the score and note straight into the bucket. Take out
    turns it back into an ordinary flag. One callback, because all three redraw the same
    parts of the same card."""
    # Surgical update: rewrites ONLY the triggered card's slots and the summary line --
    # every other card is left untouched. This is exactly why litcurator is on Dash and not
    # Streamlit; do NOT regress to bumping a reload counter that re-renders the whole feed.
    # Output order, for ctx.outputs_list: 0 summary, then the per-card ALL outputs
    # 1 flag-badge, 2 flag-delete, 3 flag-error, 4 bucket-add, 5 bucket-out, 6 stored-flag.
    # The Save button's look is not set here: it follows the stored-flag store (see
    # SAVE_BUTTON_LOOK_JS).
    (badge_slots, remove_slots, error_slots, add_slots, out_slots,
     stored_slots) = ctx.outputs_list[1:]
    card_slots = (badge_slots, remove_slots, error_slots, add_slots, out_slots, stored_slots)

    triggered = ctx.triggered_id
    if not triggered or not any(n for n in save_clicks + add_clicks + out_clicks if n):
        return (no_update, *(_hold(s) for s in card_slots))
    pmid, action = triggered["pmid"], triggered["type"]

    if action == "bucket-out":
        conn = db_interface.get_connection()
        try:
            db_interface.remove_from_levels_bucket(conn, pmid)
            flag = db_interface.get_latest_flag(conn, pmid)
        finally:
            conn.close()
        in_bucket = False
    else:
        raw_score = next((s for sid, s in zip(ctx.states_list[0], scores)
                          if sid["id"]["pmid"] == pmid), None)
        note = next((n for nid, n in zip(ctx.states_list[1], notes)
                     if nid["id"]["pmid"] == pmid), None) or ""
        doing = "saving" if action == "flag-save" else f"adding it to the {LEVELS_BUCKET_NAME}"
        user_score, error = _parse_score(raw_score, doing)
        if error:
            # Validation error: loud, persistent, inline on the triggered card only.
            return (no_update, _hold(badge_slots), _hold(remove_slots),
                    _per_card(error_slots, pmid, error), _hold(add_slots), _hold(out_slots),
                    _hold(stored_slots))
        evaluation_id = next(e for sid, e in zip(ctx.states_list[2], evaluation_ids)
                             if sid["id"]["pmid"] == pmid)
        conn = db_interface.get_connection()
        try:
            if action == "bucket-add":
                flag_id, _count = db_interface.add_to_levels_bucket(
                    conn, evaluation_id, user_score, note or None)
                in_bucket = True
            else:
                flag_id, in_bucket = db_interface.save_flag(
                    conn, evaluation_id, user_score, note or None)
            flag = db_interface.get_flag(conn, flag_id)
        finally:
            conn.close()

    items = _load_feed(start, end)   # cheap data-only reload (no rendering) for the count
    shown, floor = _apply_floor(items, min_score)
    add_style, out_style = _bucket_btn_styles(in_bucket)
    return (
        _summary_text(items, shown, start, end, floor),
        _per_card(badge_slots, pmid, _flag_badge(flag, in_bucket)),
        _per_card(remove_slots, pmid, _remove_btn_style(flag is not None, in_bucket)),
        _per_card(error_slots, pmid, ""),   # clear this card's error on success
        _per_card(add_slots, pmid, add_style),
        _per_card(out_slots, pmid, out_style),
        _per_card(stored_slots, pmid, _stored_flag(flag)),
    )


# The Save button's look, for ONE card (MATCH), in the browser: Saved while the score and note
# boxes match the card's stored-flag store, Save otherwise. The same call sets how the boxes
# report: every keystroke while the card is saved (so the first keystroke of an edit lands here),
# only on leaving the box once it is not (so the rest of the edit costs nothing). It runs on that
# first keystroke, on leaving a box, and after a save updates the store. It is the only writer of
# these four props after the card is drawn, and the card is drawn with the right values, so
# there is no call on load.
SAVE_BUTTON_LOOK_JS = """
function(score, note, stored) {
    const looks = %s;
    let saved = false;
    if (stored) {
        const typed = parseFloat(score);
        const sameScore = !isNaN(typed) && Math.abs(typed - stored.user_score) < 1e-9;
        const sameNote = (note || "") === (stored.note || "");
        saved = sameScore && sameNote;
    }
    const look = saved ? looks.saved : looks.unsaved;
    return [look[0], look[1], !saved, !saved];
}
""" % json.dumps({"saved": list(_save_btn_look(saved=True)),
                  "unsaved": list(_save_btn_look(saved=False))})

app.clientside_callback(
    SAVE_BUTTON_LOOK_JS,
    Output({"type": "flag-save", "pmid": MATCH}, "children"),
    Output({"type": "flag-save", "pmid": MATCH}, "color"),
    Output({"type": "flag-score", "pmid": MATCH}, "debounce"),
    Output({"type": "flag-note", "pmid": MATCH}, "debounce"),
    Input({"type": "flag-score", "pmid": MATCH}, "value"),
    Input({"type": "flag-note", "pmid": MATCH}, "value"),
    Input({"type": "stored-flag", "pmid": MATCH}, "data"),
    prevent_initial_call=True,
)


@callback(
    Output("reload-feed", "data", allow_duplicate=True),
    Output("flag-alert", "children", allow_duplicate=True),
    Output("flag-alert", "is_open", allow_duplicate=True),
    # color + duration so a REFUSAL does not arrive as a green flash that vanishes in three
    # seconds. This alert has exactly one writer (saving uses per-card inline errors), so
    # setting them here cannot fight another callback.
    Output("flag-alert", "color", allow_duplicate=True),
    Output("flag-alert", "duration", allow_duplicate=True),
    Input({"type": "flag-delete", "pmid": ALL}, "n_clicks"),
    State("reload-feed", "data"),
    prevent_initial_call=True,
)
def cb_delete_flag(n_clicks_list, reload_n):
    # Delete is the rare path (you seldom un-flag), so it keeps the simple full
    # rebuild: bumping reload-feed returns the card to a pristine unflagged state
    # (badge gone, inputs cleared, panel closed) with no partial-state risk. Save
    # -- the hot path -- is surgical above. Make this surgical too if the
    # asymmetry ever bothers you; it is a deliberate trade, not an oversight.
    triggered = ctx.triggered_id
    if not triggered or not any(n for n in n_clicks_list if n):
        return no_update, no_update, no_update, no_update, no_update
    pmid = triggered["pmid"]
    conn = db_interface.get_connection()
    try:
        db_interface.delete_flag(conn, pmid)
    except sqlite3.IntegrityError:
        # The flag is cited by pattern_flags, so it is a pattern's EVIDENCE, not a stray
        # entry -- and pattern -> flags -> papers is what answers "why is this line in my
        # profile?". PRAGMA foreign_keys is ON and pattern_flags.flag_id has no ON DELETE,
        # so SQLite refuses the delete and the provenance chain cannot be orphaned. The
        # database was already defending itself; this only says so out loud, because the
        # bare raise surfaced as a Dash callback error and the success alert below fired
        # for a delete that never happened.
        #
        # Cannot fire while `patterns` is empty. It starts firing the first time a
        # curation session attaches flags -- i.e. the moment the tool starts working.
        #
        # No override offered on purpose. Correcting a flag's NUMBER never needs one:
        # insert_flag appends and get_flags takes the latest row per paper, so re-saving
        # supersedes. Removal is only for an unattached mis-click.
        # "a flag on this paper", not "this flag": delete_flag is pmid-scoped (it removes
        # every flag row for the paper), so an attached OLDER row blocks the delete even when
        # the row on the card is a newer unattached re-flag. The refusal is still right -- the
        # older row is a pattern's evidence -- but the wording has to name what is actually
        # blocking, or the message sends you looking at the wrong record.
        return no_update, (
            f"Cannot remove {pmid}: a flag on this paper is attached to a pattern and is part "
            f"of its provenance. To change your score, just save the new one -- the latest "
            f"flag wins."), True, "warning", 8000
    finally:
        conn.close()
    return reload_n + 1, f"Flag removed: {pmid}.", True, "success", 3000


def run_app(start=None, end=None, port=8052, debug=False):
    """Launch the review feed. start/end (ISO dates) pre-fill the pub-date filter."""
    if start is not None or end is not None:
        app.layout = _build_layout(start, end)
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    run_app(debug=True)
