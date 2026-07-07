"""
review_feed.py -- review the judge's output and flag papers.

Reads the most-recent curation evaluation per paper (db_interface.latest_curation),
shows one score-sorted card each, and lets you enter your_score (your own estimated
interest) plus an optional private note. Saving appends a flag via
db_interface.insert_flag, keyed to the evaluation it corrects.

Flags are append-only: re-saving a paper appends a new flag and the latest wins,
so there is no "clear" -- to correct a number, just save the right one.

The delta (your_score - judge_score) is the residual: large |delta| clusters are
where the profile is most wrong. Flags are discovery data; they never feed the judge.

Launch:  litcurator review        (or)  python -m litcurator.apps.review_feed
"""

import argparse
import json

import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, callback, ctx, dcc, html, no_update

from litcurator import db_interface

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
    latest flag (if any). Sorted by score desc (from latest_curation)."""
    conn = db_interface.get_connection()
    try:
        items = db_interface.latest_curation(conn, start, end)
        flags = {f["pmid"]: f for f in db_interface.get_flags(conn, start=start, end=end)}
    finally:
        conn.close()
    for it in items:
        it["flag"] = flags.get(it["pmid"])
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


def _flag_badge(flag):
    """Inner content of the 'your flag' badge, or None when unflagged. The OUTER
    span (carrying the fixed flag-badge id) is always in the DOM, so a save or
    delete fills/empties it surgically instead of rebuilding the card."""
    if not flag:
        return None
    return html.Span(f"you: {flag['your_score']:.2f}  (delta {flag['delta']:+.2f})",
                     className="badge bg-danger")


def _remove_btn_style(flagged):
    """Show the Remove-flag button only when flagged. Toggled via display so the
    fixed id stays mounted (no conditional render -> no n_clicks-reset glitch)."""
    return {} if flagged else {"display": "none"}


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
    badges.append(html.Span(_flag_badge(flag), id={"type": "flag-badge", "pmid": pmid}))

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
    remove_btn = dbc.Button("Remove flag", id={"type": "flag-delete", "pmid": pmid},
                            color="danger", outline=True, size="sm", className="mt-2",
                            style=_remove_btn_style(flagged))
    flag_panel = dbc.Collapse(
        dbc.Card(dbc.CardBody([
            html.Div("Your estimated interest (0.0 = no interest, 1.0 = must read)",
                     className="small fw-semibold mb-2"),
            dbc.Row([
                dbc.Col(dbc.Input(
                    id={"type": "flag-score", "pmid": pmid},
                    type="text",
                    value=pre.get("your_score", None),
                    placeholder="0.0 - 1.0", size="sm"), width=3),
                dbc.Col(dbc.Button("Save", id={"type": "flag-save", "pmid": pmid,
                                               "eid": item["evaluation_id"]},
                                   color="primary", size="sm"), width="auto"),
            ], className="g-2 align-items-center mb-2"),
            html.Div(id={"type": "flag-error", "pmid": pmid},
                     className="text-danger fw-bold mb-2"),
            dbc.Label("Note (optional, private)", className="small mb-1"),
            dbc.Input(id={"type": "flag-note", "pmid": pmid}, type="text",
                      value=pre.get("note", ""), placeholder="e.g. ECoG, not single-unit",
                      size="sm"),
            remove_btn,
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
        html.Div([html.Span("Summary: ", className="small fw-bold text-muted"),
                  html.Span(item.get("summary"), className="small")],
                 className="mb-2") if item.get("summary") else None,
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
        html.Div(dbc.Button("Flag / edit score",
                            id={"type": "flag-toggle", "pmid": pmid},
                            color="secondary", outline=True, size="sm"),
                 className="mt-2"),
        flag_panel,
    ]), className="mb-3",
        style={"backgroundColor": "#f3f0fa", "border": "1px solid #e3dcf2"})


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
app.title = "litcurator review"

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
        ], className="mb-2 align-items-end"),
        dbc.Alert(id="flag-alert", is_open=False, duration=3000, color="success"),
        html.Div(id="feed-container"),
    ], fluid=True)


app.layout = _build_layout(CLI_START, CLI_END)


@callback(
    Output("feed-container", "children"),
    Output("feed-summary", "children"),
    Input("reload-feed", "data"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("min-score", "value"),
)
def cb_render_feed(_n, start, end, min_score):
    items = _load_feed(start, end)
    if not items:
        return (html.Div("No judged papers in this range. Run `litcurator run`, or widen the dates.",
                         className="text-muted"), "")
    shown, floor = _apply_floor(items, min_score)
    if not shown:
        return (html.Div(f"All {len(items)} papers scored below {floor:g}. Lower the min score to see them.",
                         className="text-muted"),
                _summary_text(items, shown, start, end, floor))
    total = len(shown)
    cards = [_render_card(it, rank, total) for rank, it in enumerate(shown, 1)]
    return cards, _summary_text(items, shown, start, end, floor)


@callback(
    Output({"type": "flag-collapse", "pmid": ALL}, "is_open"),
    Input({"type": "flag-toggle", "pmid": ALL}, "n_clicks"),
    State({"type": "flag-collapse", "pmid": ALL}, "is_open"),
    prevent_initial_call=True,
)
def cb_toggle_flag(_n, is_open_list):
    triggered = ctx.triggered_id
    if not triggered:
        return [no_update] * len(is_open_list)
    return [(not is_open) if sid["id"]["pmid"] == triggered["pmid"] else is_open
            for sid, is_open in zip(ctx.states_list[0], is_open_list)]


@callback(
    Output("feed-summary", "children", allow_duplicate=True),
    Output({"type": "flag-badge", "pmid": ALL}, "children"),
    Output({"type": "flag-delete", "pmid": ALL}, "style"),
    Output({"type": "flag-error", "pmid": ALL}, "children"),
    Input({"type": "flag-save", "pmid": ALL, "eid": ALL}, "n_clicks"),
    State({"type": "flag-score", "pmid": ALL}, "value"),
    State({"type": "flag-note", "pmid": ALL}, "value"),
    State("date-filter", "start_date"),
    State("date-filter", "end_date"),
    State("min-score", "value"),
    prevent_initial_call=True,
)
def cb_save_flag(n_clicks_list, scores, notes, start, end, min_score):
    # Surgical update: a save rewrites ONLY the triggered card's flag badge, its
    # remove button, and the summary line -- every other card is left untouched.
    # This is exactly why litcurator is on Dash and not Streamlit; do NOT regress
    # to bumping a reload counter that re-renders the whole feed on each save.
    # Output order, for ctx.outputs_list: 0 summary, 1 flag-badge, 2 flag-delete,
    # 3 flag-error -- the last three are per-card ALL outputs.
    badge_slots = ctx.outputs_list[1]
    remove_slots = ctx.outputs_list[2]
    error_slots = ctx.outputs_list[3]

    def per_card(slots, pmid, value, default=no_update):
        """Set value on the triggered card's slot, hold the rest."""
        return [value if s["id"]["pmid"] == pmid else default for s in slots]

    def hold(slots):
        return [no_update] * len(slots)

    def fail(pmid, msg):
        # Validation error: loud, persistent, inline on the triggered card only.
        return no_update, hold(badge_slots), hold(remove_slots), per_card(error_slots, pmid, msg)

    triggered = ctx.triggered_id
    if not triggered or not any(n for n in n_clicks_list if n):
        return no_update, hold(badge_slots), hold(remove_slots), hold(error_slots)
    pmid = triggered["pmid"]
    evaluation_id = triggered["eid"]

    your_score = None
    note = ""
    for sid, s in zip(ctx.states_list[0], scores):
        if sid["id"]["pmid"] == pmid:
            your_score = s
    for nid, n in zip(ctx.states_list[1], notes):
        if nid["id"]["pmid"] == pmid:
            note = n or ""

    if your_score is None or str(your_score).strip() == "":
        return fail(pmid, "Enter a score (0.0 - 1.0) before saving.")
    try:
        your_score = float(str(your_score).strip())
    except ValueError:
        return fail(pmid, f"'{your_score}' is not a number -- enter a value 0.0 - 1.0.")
    if not (0.0 <= your_score <= 1.0):
        return fail(pmid, "Score must be between 0.0 and 1.0.")

    conn = db_interface.get_connection()
    try:
        flag_id = db_interface.insert_flag(conn, evaluation_id, your_score, note or None)
        flag = db_interface.get_flag(conn, flag_id)
    finally:
        conn.close()
    items = _load_feed(start, end)   # cheap data-only reload (no rendering) for the count
    shown, floor = _apply_floor(items, min_score)

    return (
        _summary_text(items, shown, start, end, floor),
        per_card(badge_slots, pmid, _flag_badge(flag)),
        per_card(remove_slots, pmid, _remove_btn_style(True)),
        per_card(error_slots, pmid, ""),   # clear this card's error on success
    )


@callback(
    Output("reload-feed", "data", allow_duplicate=True),
    Output("flag-alert", "children", allow_duplicate=True),
    Output("flag-alert", "is_open", allow_duplicate=True),
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
        return no_update, no_update, no_update
    pmid = triggered["pmid"]
    conn = db_interface.get_connection()
    try:
        db_interface.delete_flag(conn, pmid)
    finally:
        conn.close()
    return reload_n + 1, f"Flag removed: {pmid}.", True


def run_app(start=None, end=None, port=8052, debug=False):
    """Launch the review feed. start/end (ISO dates) pre-fill the pub-date filter."""
    if start is not None or end is not None:
        app.layout = _build_layout(start, end)
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    run_app(debug=True)
