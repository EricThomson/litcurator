"""
profile_workbench.py -- edit and version the active profile from flag suggestions.

Left panel: ranked step-2 suggestions from profile_analysis (loaded from
~/.litcurator/suggestions/), one card each. Mark each Done, Cut, or send to chat.

Right panel: the live active profile, editable. Never overwritten silently:
  - "Save version" writes a timestamped copy to versions/.
  - "Set as active" snapshots the outgoing active into versions/, writes
    user_profile.md, and registers the new version in the DB (parent_id = outgoing).

Bottom: chat sounding-board. Committed profile + live draft both loaded as context
so the assistant can reason about the delta between them.

The human authors every word of the profile.

Run:
    litcurator profile_workbench
    python src/litcurator/apps/profile_workbench.py
"""

import os
import re
from datetime import datetime
from pathlib import Path

import anthropic
import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, callback, ctx, dash_table, dcc, html, no_update
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle
from dotenv import load_dotenv

from litcurator import db_interface, profile_interface
from litcurator.config import DATA_DIR

load_dotenv()

CHAT_MODEL = "claude-sonnet-4-6"
SUGGESTIONS_DIR = DATA_DIR / "suggestions"

# Marks the start of the step-2 ranked suggestions in a profile_analysis output file.
DISTILL_MARKER = "## Distilled suggestions"


# ---------------------------------------------------------------------------
# Suggestions parsing
# ---------------------------------------------------------------------------

def _suggestion_files():
    """Available suggestion .md files, newest first."""
    if not SUGGESTIONS_DIR.exists():
        return []
    return sorted(SUGGESTIONS_DIR.glob("seed_suggestions_*.md"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def _parse_suggestions(md_text):
    """Parse the step-2 section into (items, cut_markdown).

    items: ranked suggestions in document order, each {num, text}.
    cut_markdown: the raw "Considered and cut" block for read-only display.
    Lenient: splits on the numbered list, leaves the model's prose intact.
    """
    marker = md_text.find(DISTILL_MARKER)
    body = md_text[marker:] if marker != -1 else md_text
    body = re.sub(r"(?m)^##\s*Distilled suggestions.*$", "", body, count=1)

    cut_md = ""
    m = re.search(r"(?im)^\s*\**\s*considered and cut\b.*$", body)
    if m:
        cut_md = body[m.end():].strip()
        body = body[:m.start()]

    items = []
    parts = re.split(r"(?m)^\s*(\d+)\.\s+", body)
    pairs = iter(parts[1:])
    for num, text in zip(pairs, pairs):
        text = text.strip()
        if text:
            items.append({"num": num, "text": text})
    return items, cut_md


def _split_item(text):
    """Return (title, body). Title = leading bold span, else truncated first line."""
    mb = re.match(r"\s*\*\*(.+?)\*\*\s*", text)
    if mb:
        return mb.group(1).strip(), text[mb.end():].strip()
    first = (text.splitlines() or [""])[0].strip()
    return (first if len(first) <= 70 else first[:67] + "..."), text


def _items_to_store(items):
    return {f"sugg::{it['num']}": it["text"] for it in items}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def _chat_system(committed, draft):
    return (
        "You are a sharp, concise thinking partner helping a researcher refine their ACTIVE PROFILE "
        "for a personal paper-curation system. The profile is a prose statement of the researcher's "
        "reading taste; an LLM judge scores papers against it.\n\n"
        "You are given TWO versions so you can reason about the DELTA between them:\n"
        "- COMMITTED PROFILE: the stable, on-disk profile (the 'before').\n"
        "- CURRENT DRAFT: what the researcher is editing right now (the 'after'). It may differ from "
        "the committed profile, or be identical.\n\n"
        "This matters: when the researcher asks 'is this already covered?' or 'what do you think of "
        "adding X', do NOT assume X is handled just because it appears in the DRAFT -- they are "
        "mid-edit and may have just typed it. Compare the draft against the committed profile and tell "
        "them whether the change is genuinely NEW, REDUNDANT with existing committed language (quote "
        "it), or IN TENSION with something already there.\n\n"
        "Your job is to help them THINK, not to write the profile for them. Be honest and specific. "
        "Push back hard when a proposed edit would bloat the profile or merely restate something "
        "already present -- this profile has a history of vocabulary drift from over-editing; fewer, "
        "sharper words beat more. Only draft profile prose if explicitly asked, and keep it in their "
        "voice. Keep answers short unless asked to expand. ASCII only.\n\n"
        "----- COMMITTED PROFILE (before) -----\n"
        f"{committed}\n"
        "----- END COMMITTED PROFILE -----\n\n"
        "----- CURRENT DRAFT (after, what they are editing) -----\n"
        f"{draft}\n"
        "----- END CURRENT DRAFT -----"
    )


def _render_thread(history):
    out = []
    for m in history:
        if m["role"] == "user":
            out.append(html.Div(
                dbc.Card(dbc.CardBody(dcc.Markdown(m["content"]), className="py-2 px-3"),
                         color="light", className="mb-2"),
                style={"marginLeft": "12%"}))
        else:
            out.append(html.Div(
                dbc.Card(dbc.CardBody(dcc.Markdown(m["content"]), className="py-2 px-3"),
                         className="mb-2",
                         style={"backgroundColor": "#f3f0fa", "border": "1px solid #e3dcf2"}),
                style={"marginRight": "12%"}))
    return out


# ---------------------------------------------------------------------------
# Suggestion cards
# ---------------------------------------------------------------------------

def _card(item):
    uid = f"sugg::{item['num']}"
    title, body = _split_item(item["text"])
    return html.Div(
        dbc.Card(dbc.CardBody([
            html.Div([
                html.Span(f"{item['num']}.", className="fw-bold me-2",
                          style={"flex": "0 0 auto"}),
                html.Span(title, className="text-truncate",
                          style={"flex": "1 1 auto", "minWidth": 0}),
                html.Span([
                    dbc.Button("Discuss", id={"type": "discuss-btn", "uid": uid},
                               color="primary", outline=True, size="sm", className="me-1"),
                    dbc.Button("Done", id={"type": "done-btn", "uid": uid},
                               color="success", outline=True, size="sm", className="me-1"),
                    dbc.Button("Cut", id={"type": "cut-btn", "uid": uid},
                               color="secondary", outline=True, size="sm"),
                ], className="ms-2", style={"flex": "0 0 auto", "whiteSpace": "nowrap"}),
            ], className="d-flex align-items-center mb-2"),
            dcc.Markdown(body, className="small mb-0"),
        ]), className="mb-2", style={"border": "1px solid #e3dcf2"}),
        id={"type": "card-wrap", "uid": uid},
    )


def _render_suggestions(items, cut_md):
    if not items:
        return [html.Div("No suggestions parsed from this file.", className="text-muted")]
    blocks = [_card(it) for it in items]
    if cut_md:
        blocks.append(html.Details([
            html.Summary("Considered and cut (deferred this round)",
                         className="text-muted small mt-3"),
            dcc.Markdown(cut_md, className="small text-muted mt-2"),
        ]))
    return blocks


# ---------------------------------------------------------------------------
# Flags to retire (per-flag ingestion)
# ---------------------------------------------------------------------------
# Retiring a flag stamps ingested_to_profile_id with the currently-active version,
# so it drops out of future profile_analysis. Un-retired flags carry forward and
# accumulate -- that accumulation is what grows a sparse pattern into one dense
# enough for the analyzer to surface. The human picks what they have addressed;
# nothing is retired automatically.

FLAG_COLUMNS = [
    {"name": "delta", "id": "delta"},
    {"name": "you", "id": "your"},
    {"name": "journal", "id": "journal"},
    {"name": "title", "id": "title"},
    {"name": "your note", "id": "note"},
]


def _load_flag_rows():
    """Un-retired flags as flat table rows (latest flag per paper)."""
    conn = db_interface.get_connection()
    try:
        flags = db_interface.get_flags(conn, only_uningested=True)
    finally:
        conn.close()
    return [{
        "flag_id": f["id"],
        "delta": round(f["delta"], 2),
        "your": f["your_score"],
        "journal": f.get("journal") or "",
        "title": f.get("title") or "",
        "note": f.get("note") or "",
    } for f in flags]


def _filter_flag_rows(rows, term):
    if not term:
        return rows
    t = term.lower()
    return [r for r in rows if t in f"{r['title']} {r['journal']} {r['note']}".lower()]


def _active_profile_id():
    """Register/get the id of the profile currently active on disk -- the version a
    retire stamps onto the flag."""
    conn = db_interface.get_connection()
    try:
        return db_interface.get_or_create_profile(conn, profile_interface.read_active_or_empty())
    finally:
        conn.close()


def _flags_toggle_label(n):
    return f"Retire flags  ({n} un-retired)"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def _latest_version_note():
    v = profile_interface.latest_version()
    return f"latest version: {v.name}" if v else "no saved versions yet"


_files = _suggestion_files()
_file_options = [{"label": p.name, "value": str(p)} for p in _files]
_default_file = str(_files[0]) if _files else None
if _default_file:
    _initial_items, _initial_cut = _parse_suggestions(
        Path(_default_file).read_text(encoding="utf-8"))
else:
    _initial_items, _initial_cut = [], ""

_initial_flag_rows = _load_flag_rows()

_PANE_STYLE = {"height": "56vh", "overflowY": "auto", "padding": "0 14px"}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
app.title = "Profile Workbench"

app.layout = dbc.Container([
    dcc.Store(id="status-store", data={}, storage_type="session"),
    dcc.Store(id="chat-history", data=[], storage_type="session"),
    dcc.Store(id="suggestions-store", data=_items_to_store(_initial_items)),
    dcc.Store(id="flag-rows", data=_initial_flag_rows),
    dbc.Row([
        dbc.Col(html.H4("Profile Workbench", className="mb-0"), width="auto"),
        dbc.Col(dcc.Dropdown(id="file-dropdown", options=_file_options,
                             value=_default_file, clearable=False,
                             style={"minWidth": "340px"}),
                width="auto"),
        dbc.Col(html.Small(id="cut-summary", className="text-muted"),
                width="auto", align="center"),
    ], align="center", className="mt-3 mb-2 g-2"),

    PanelGroup(id="panel-group", direction="horizontal", children=[
        Panel(id="left-panel", defaultSizePercentage=50, children=[
            html.Div([
                html.Div("Suggestions, ranked by importance. Done = folded into the profile, "
                         "Cut = set aside, Discuss = send to chat.",
                         className="text-muted small mb-2"),
                html.Div(id="suggestions-pane",
                         children=_render_suggestions(_initial_items, _initial_cut)),
            ], style=_PANE_STYLE),
        ]),
        PanelResizeHandle(html.Div(style={
            "width": "8px", "backgroundColor": "#d9d2ee", "cursor": "col-resize",
            "height": "56vh"})),
        Panel(id="right-panel", defaultSizePercentage=50, children=[
            html.Div([
                html.Div([
                    dbc.Button("Save version", id="save-version-btn",
                               color="primary", size="sm", className="me-2"),
                    dbc.Button("Set as active", id="set-active-btn",
                               color="danger", outline=True, size="sm", className="me-2"),
                    dbc.Button("Reload from disk", id="reload-profile-btn",
                               color="secondary", outline=True, size="sm", className="me-2"),
                    dbc.Button("Restore autosave", id="restore-autosave-btn",
                               color="warning", outline=True, size="sm"),
                ], className="mb-2"),
                dbc.Alert(id="profile-status", is_open=False, duration=4000,
                          color="success", className="py-1 px-2 small"),
                html.Small(f"active: {profile_interface.active_path()}",
                           className="text-muted d-block"),
                html.Small(id="version-note", className="text-muted d-block"),
                html.Small(id="autosave-note", className="text-muted d-block mb-2"),
                dbc.Textarea(id="profile-editor",
                             value=profile_interface.read_active_or_empty(),
                             persistence=True, persistence_type="local",
                             style={"width": "100%", "height": "44vh",
                                    "fontFamily": "monospace", "fontSize": "0.85rem"}),
                dcc.Interval(id="autosave-timer", interval=20000),
            ], style={"height": "56vh", "overflowY": "auto", "padding": "0 14px"}),
        ]),
    ]),

    html.Div([
        dbc.Button(_flags_toggle_label(len(_initial_flag_rows)), id="flags-toggle",
                   color="link", size="sm", className="px-0"),
        dbc.Collapse(id="flags-collapse", is_open=False, children=html.Div([
            html.Small(
                "Retire the flags you have folded into the profile this round -- they stop "
                "feeding profile_analysis. Un-retired flags carry forward and accumulate. "
                "A retire stamps the CURRENTLY-active version, so Set as active first if this "
                "round minted a new one.",
                className="text-muted d-block mb-2"),
            dbc.Row([
                dbc.Col(dbc.Input(id="flag-search", size="sm",
                                  placeholder="filter by title / journal / your note ..."),
                        width=True),
                dbc.Col(dbc.Button("Retire selected", id="retire-btn",
                                   color="danger", size="sm"), width="auto"),
            ], className="g-2 mb-2"),
            dbc.Alert(id="retire-status", is_open=False, duration=4000,
                      color="success", className="py-1 px-2 small"),
            dash_table.DataTable(
                id="flag-table",
                columns=FLAG_COLUMNS,
                data=_initial_flag_rows,
                row_selectable="multi",
                selected_rows=[],
                sort_action="native",
                page_action="none",
                style_table={"maxHeight": "32vh", "overflowY": "auto"},
                style_header={"fontWeight": "bold"},
                style_cell={"fontSize": "0.78rem", "textAlign": "left", "padding": "2px 6px",
                            "maxWidth": 340, "overflow": "hidden", "textOverflow": "ellipsis",
                            "whiteSpace": "nowrap"},
                style_cell_conditional=[
                    {"if": {"column_id": "delta"}, "width": "55px"},
                    {"if": {"column_id": "your"}, "width": "45px"},
                    {"if": {"column_id": "journal"}, "width": "150px"},
                ],
            ),
        ], style={"padding": "0 14px"})),
    ], style={"padding": "0 14px"}),

    html.Hr(className="my-2"),

    html.Div([
        html.Div([
            html.Span("Chat", className="fw-semibold me-2"),
            html.Small(f"(context: committed profile + live draft | model: {CHAT_MODEL})",
                       className="text-muted"),
            dbc.Button("Clear", id="chat-clear-btn", color="secondary",
                       outline=True, size="sm", className="float-end"),
        ], className="mb-2"),
        dcc.Loading(html.Div(id="chat-thread",
                             style={"height": "20vh", "overflowY": "auto",
                                    "padding": "4px 8px"})),
        dbc.InputGroup([
            dbc.Textarea(id="chat-input",
                         placeholder='Paste a suggestion and ask, e.g. "isn\'t this already in my profile?"',
                         style={"minHeight": "60px"}),
            dbc.Button("Send", id="chat-send-btn", color="primary"),
        ], className="mt-1"),
    ], style={"padding": "0 14px"}),
], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks: suggestions
# ---------------------------------------------------------------------------

@callback(
    Output("suggestions-pane", "children"),
    Output("status-store", "data", allow_duplicate=True),
    Output("suggestions-store", "data"),
    Input("file-dropdown", "value"),
    prevent_initial_call=True,
)
def cb_load_file(path):
    if not path:
        return [html.Div("No file selected.", className="text-muted")], {}, {}
    items, cut_md = _parse_suggestions(Path(path).read_text(encoding="utf-8"))
    return _render_suggestions(items, cut_md), {}, _items_to_store(items)


def _status_style(status):
    collapsed = {"maxHeight": "54px", "overflow": "hidden"}
    if status == "cut":
        return {**collapsed, "opacity": 0.45}
    if status == "done":
        return {**collapsed, "opacity": 0.75, "borderLeft": "4px solid #2f7a4f"}
    return {}


@callback(
    Output({"type": "card-wrap", "uid": ALL}, "style"),
    Output({"type": "cut-btn", "uid": ALL}, "children"),
    Output({"type": "done-btn", "uid": ALL}, "children"),
    Output("status-store", "data"),
    Output("cut-summary", "children"),
    Input({"type": "cut-btn", "uid": ALL}, "n_clicks"),
    Input({"type": "done-btn", "uid": ALL}, "n_clicks"),
    State("status-store", "data"),
    prevent_initial_call=True,
)
def cb_toggle_status(_cut_clicks, _done_clicks, status):
    status = dict(status or {})
    triggered = ctx.triggered_id
    if triggered and triggered.get("type") in ("cut-btn", "done-btn"):
        uid = triggered["uid"]
        kind = "cut" if triggered["type"] == "cut-btn" else "done"
        if status.get(uid) == kind:
            status.pop(uid, None)
        else:
            status[uid] = kind

    styles = [_status_style(status.get(o["id"]["uid"])) for o in ctx.outputs_list[0]]
    cut_labels = ["Restore" if status.get(o["id"]["uid"]) == "cut" else "Cut"
                  for o in ctx.outputs_list[1]]
    done_labels = ["Reopen" if status.get(o["id"]["uid"]) == "done" else "Done"
                   for o in ctx.outputs_list[2]]
    n_done = sum(1 for v in status.values() if v == "done")
    n_cut = sum(1 for v in status.values() if v == "cut")
    parts = [f"{n} {label}" for n, label in [(n_done, "done"), (n_cut, "cut")] if n]
    return styles, cut_labels, done_labels, status, "  |  ".join(parts)


# ---------------------------------------------------------------------------
# Callbacks: profile editor
# ---------------------------------------------------------------------------

@callback(
    Output("profile-status", "children", allow_duplicate=True),
    Output("profile-status", "is_open", allow_duplicate=True),
    Output("version-note", "children", allow_duplicate=True),
    Input("save-version-btn", "n_clicks"),
    State("profile-editor", "value"),
    prevent_initial_call=True,
)
def cb_save_version(_n, text):
    path = profile_interface.save_version(text or "")
    return f"Saved to version: {path.name}", True, _latest_version_note()


@callback(
    Output("profile-status", "children", allow_duplicate=True),
    Output("profile-status", "is_open", allow_duplicate=True),
    Output("version-note", "children", allow_duplicate=True),
    Input("set-active-btn", "n_clicks"),
    State("profile-editor", "value"),
    prevent_initial_call=True,
)
def cb_set_active(_n, text):
    backup = profile_interface.set_active(text or "")
    msg = "Set as active profile."
    if backup:
        msg += f" Outgoing profile backed up to {backup.name}."
    return msg, True, _latest_version_note()


@callback(
    Output("profile-editor", "value"),
    Output("profile-status", "children", allow_duplicate=True),
    Output("profile-status", "is_open", allow_duplicate=True),
    Input("reload-profile-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cb_reload_profile(_n):
    return profile_interface.read_active_or_empty(), "Reloaded profile from disk.", True


@callback(
    Output("version-note", "children"),
    Input("file-dropdown", "value"),
)
def cb_init_version_note(_v):
    return _latest_version_note()


@callback(
    Output("autosave-note", "children"),
    Input("autosave-timer", "n_intervals"),
    State("profile-editor", "value"),
    prevent_initial_call=True,
)
def cb_autosave(_n, text):
    if profile_interface.save_autosave(text):
        return f"autosaved draft {datetime.now():%H:%M:%S}"
    return no_update


@callback(
    Output("profile-editor", "value", allow_duplicate=True),
    Output("profile-status", "children", allow_duplicate=True),
    Output("profile-status", "is_open", allow_duplicate=True),
    Input("restore-autosave-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cb_restore_autosave(_n):
    text = profile_interface.load_autosave()
    if text:
        return text, "Restored autosave draft into the editor.", True
    return no_update, "No autosave draft found.", True


# ---------------------------------------------------------------------------
# Callbacks: flags to retire
# ---------------------------------------------------------------------------

@callback(
    Output("flags-collapse", "is_open"),
    Input("flags-toggle", "n_clicks"),
    State("flags-collapse", "is_open"),
    prevent_initial_call=True,
)
def cb_toggle_flags(_n, is_open):
    return not is_open


@callback(
    Output("flag-table", "data"),
    Output("flag-table", "selected_rows"),
    Input("flag-search", "value"),
    Input("flag-rows", "data"),
)
def cb_filter_flags(term, rows):
    # Re-fires on the search term OR on the store changing (after a retire) -- both
    # want the table re-rendered with the selection cleared.
    return _filter_flag_rows(rows or [], term or ""), []


@callback(
    Output("flag-rows", "data"),
    Output("retire-status", "children"),
    Output("retire-status", "is_open"),
    Output("flags-toggle", "children"),
    Input("retire-btn", "n_clicks"),
    State("flag-table", "data"),
    State("flag-table", "selected_rows"),
    prevent_initial_call=True,
)
def cb_retire(_n, table_data, selected_rows):
    if not selected_rows:
        return no_update, "Select one or more flags first.", True, no_update
    flag_ids = [table_data[i]["flag_id"] for i in selected_rows]
    profile_id = _active_profile_id()
    conn = db_interface.get_connection()
    try:
        db_interface.mark_flags_ingested(conn, flag_ids, profile_id)
    finally:
        conn.close()
    rows = _load_flag_rows()
    msg = (f"Retired {len(flag_ids)} flag(s) into version {profile_id[:12]}.  "
           f"{len(rows)} un-retired remaining.")
    return rows, msg, True, _flags_toggle_label(len(rows))


# ---------------------------------------------------------------------------
# Callbacks: chat
# ---------------------------------------------------------------------------

@callback(
    Output("chat-input", "value", allow_duplicate=True),
    Input({"type": "discuss-btn", "uid": ALL}, "n_clicks"),
    State("suggestions-store", "data"),
    prevent_initial_call=True,
)
def cb_discuss(clicks, store):
    triggered = ctx.triggered_id
    if not triggered or triggered.get("type") != "discuss-btn":
        return no_update
    if not any(clicks or []):
        return no_update
    text = (store or {}).get(triggered["uid"], "")
    if not text:
        return no_update
    return (
        "Is this suggestion already covered by my current profile, or is it a genuine gap? "
        "Quote the overlapping profile text if it is redundant.\n\n"
        f"Suggestion:\n{text}"
    )


@callback(
    Output("chat-history", "data"),
    Output("chat-input", "value"),
    Input("chat-send-btn", "n_clicks"),
    State("chat-input", "value"),
    State("chat-history", "data"),
    State("profile-editor", "value"),
    prevent_initial_call=True,
)
def cb_chat_send(_n, user_text, history, draft):
    if not (user_text and user_text.strip()):
        return no_update, no_update
    history = list(history or [])
    history.append({"role": "user", "content": user_text.strip()})
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=1200,
        system=_chat_system(profile_interface.read_active_or_empty(), draft or ""),
        messages=[{"role": m["role"], "content": m["content"]} for m in history],
    )
    history.append({"role": "assistant", "content": resp.content[0].text})
    return history, ""


@callback(
    Output("chat-history", "data", allow_duplicate=True),
    Input("chat-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cb_chat_clear(_n):
    return []


@callback(
    Output("chat-thread", "children"),
    Input("chat-history", "data"),
)
def cb_render_thread(history):
    return _render_thread(history or [])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_app(port=8053):
    # use_reloader=False: hot reload would reset the editor textarea and wipe
    # unsaved work. Restart manually to pick up code changes.
    app.run(debug=True, use_reloader=False, port=port)


if __name__ == "__main__":
    run_app()
