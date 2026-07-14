"""
profile_workbench.py -- edit the active profile, curating the pattern memory.

Left panel: the ACTIVE PATTERNS (db_interface.get_active_patterns) -- recurring
taste-gaps the suggester surfaced from your flags, each with a drill-down to the
papers behind it. Per pattern you can edit its wording and then decide its fate,
which is written to the append-only pattern_events log (nothing is ever deleted):
  - Incorporate: you folded it into the profile. Stamps the currently-active
    profile version, drops the pattern off the active list.
  - Carry: not yet -- keep it open for a later round.
  - Reject (with a reason): not a real gap. Drops off the active list, kept as a
    tombstone so the suggester will not re-propose it.

Right panel: the live active profile, editable. Never overwritten silently:
  - "Save version" writes a timestamped copy to versions/.
  - "Set as active" snapshots the outgoing active into versions/, writes
    user_profile.md, and registers the new version in the DB (parent_id = outgoing).

Bottom: chat sounding-board. Committed profile + live draft both loaded as context.

The human authors every word of the profile; the machinery only surfaces and
remembers. Typical flow: Discuss / edit a pattern -> author the profile edit on the
right -> Set as active -> Incorporate the pattern (it stamps that version).

Run:
    litcurator profile_workbench
    python src/litcurator/apps/profile_workbench.py
"""

import os
from datetime import datetime

import anthropic
import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, callback, ctx, dcc, html, no_update
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle
from dotenv import load_dotenv

from litcurator import db_interface, profile_interface

load_dotenv()

CHAT_MODEL = "claude-sonnet-4-6"

DIRECTIONS = ["over", "under", "sharpen", "judge-not-applying"]
_DIR_COLOR = {"over": "danger", "under": "success",
              "sharpen": "warning", "judge-not-applying": "info"}


# ---------------------------------------------------------------------------
# Pattern cards
# ---------------------------------------------------------------------------

def _render_provenance(prov):
    """The papers behind a pattern, read-only. Largest |delta| first."""
    if not prov:
        return [html.Div("(no linked papers)", className="text-muted small")]
    out = []
    for f in prov:
        note = f"  -- {f['note']}" if f.get("note") else ""
        out.append(html.Div(
            f"delta {f['delta']:+.2f} (judge {f['judge_score']:.2f} -> you "
            f"{f['your_score']:.2f})  {f.get('journal') or ''}: {f.get('title') or ''}{note}",
            className="small text-muted"))
    return out


def _pattern_card(conn, p):
    pid = p["id"]
    prov = db_interface.get_pattern_provenance(conn, pid)
    return html.Div(dbc.Card(dbc.CardBody([
        html.Div([
            dbc.Badge(p["direction"], color=_DIR_COLOR.get(p["direction"], "secondary"),
                      className="me-2", style={"flex": "0 0 auto"}),
            dbc.Input(id={"type": "pat-name", "pid": pid}, value=p["name"], size="sm",
                      style={"flex": "1 1 auto", "minWidth": 0}),
            dbc.Badge(p["status"], color="light", text_color="dark", className="ms-2",
                      style={"flex": "0 0 auto"}),
        ], className="d-flex align-items-center mb-2"),
        dbc.Select(id={"type": "pat-dir", "pid": pid},
                   options=[{"label": d, "value": d} for d in DIRECTIONS],
                   value=p["direction"], size="sm", className="mb-2"),
        html.Small("description", className="text-muted"),
        dbc.Textarea(id={"type": "pat-desc", "pid": pid}, value=p.get("description") or "",
                     style={"height": "3rem", "fontSize": "0.8rem"}, className="mb-2"),
        html.Small("suggested edit (your working draft -- you author the profile prose)",
                   className="text-muted"),
        dbc.Textarea(id={"type": "pat-sugg", "pid": pid}, value=p.get("suggested_edit") or "",
                     style={"height": "4rem", "fontSize": "0.8rem"}, className="mb-2"),
        html.Details([
            html.Summary(f"{p['flag_count']} papers (provenance)", className="small text-muted"),
            html.Div(_render_provenance(prov), className="mt-1"),
        ], className="mb-2"),
        dbc.Input(id={"type": "pat-reject-note", "pid": pid}, size="sm",
                  placeholder="reason (recorded on Reject)", className="mb-2"),
        html.Div([
            dbc.Button("Save edits", id={"type": "pat-save", "pid": pid},
                       color="primary", outline=True, size="sm", className="me-1"),
            dbc.Button("Discuss", id={"type": "pat-discuss", "pid": pid},
                       color="primary", outline=True, size="sm", className="me-1"),
            dbc.Button("Carry", id={"type": "pat-carry", "pid": pid},
                       color="secondary", outline=True, size="sm", className="me-1"),
            dbc.Button("Incorporate", id={"type": "pat-incorporate", "pid": pid},
                       color="success", size="sm", className="me-1"),
            dbc.Button("Reject", id={"type": "pat-reject", "pid": pid},
                       color="danger", outline=True, size="sm"),
        ]),
    ]), className="mb-3", style={"border": "1px solid #e3dcf2"}),
        id={"type": "pat-card", "pid": pid})


def _render_patterns(conn):
    patterns = db_interface.get_active_patterns(conn)
    if not patterns:
        return [html.Div(
            "No open patterns. Flag papers in the review feed, then run "
            "`litcurator profile_analysis` to surface patterns here.",
            className="text-muted")]
    return [_pattern_card(conn, p) for p in patterns]


def _initial_patterns():
    conn = db_interface.get_connection()
    try:
        return _render_patterns(conn), len(db_interface.get_active_patterns(conn))
    finally:
        conn.close()


def _count_label(n):
    return f"{n} open pattern{'' if n == 1 else 's'}"


def _state_value(states, pid):
    """The value of the pattern-matched State whose id has this pid."""
    for s in states or []:
        if s.get("id", {}).get("pid") == pid:
            return s.get("value")
    return None


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
# App
# ---------------------------------------------------------------------------

def _latest_version_note():
    v = profile_interface.latest_version()
    return f"latest version: {v.name}" if v else "no saved versions yet"


_initial_pattern_children, _initial_count = _initial_patterns()

_PANE_STYLE = {"height": "56vh", "overflowY": "auto", "padding": "0 14px"}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
app.title = "Profile Workbench"

app.layout = dbc.Container([
    dcc.Store(id="chat-history", data=[], storage_type="session"),
    dbc.Row([
        dbc.Col(html.H4("Profile Workbench", className="mb-0"), width="auto"),
        dbc.Col(dbc.Button("Refresh patterns", id="refresh-patterns-btn",
                           color="secondary", outline=True, size="sm"), width="auto"),
        dbc.Col(html.Small(_count_label(_initial_count), id="pattern-count",
                           className="text-muted"), width="auto", align="center"),
    ], align="center", className="mt-3 mb-2 g-2"),

    PanelGroup(id="panel-group", direction="horizontal", children=[
        Panel(id="left-panel", defaultSizePercentage=50, children=[
            html.Div([
                html.Div("Active patterns from your flags. Edit the wording, then decide each "
                         "one's fate. To Incorporate: author the edit on the right, Set as active, "
                         "then Incorporate here (it stamps that version).",
                         className="text-muted small mb-2"),
                dbc.Alert(id="pattern-status", is_open=False, duration=4000,
                          color="success", className="py-1 px-2 small"),
                html.Div(id="patterns-pane", children=_initial_pattern_children),
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
# Callbacks: pattern lifecycle
# ---------------------------------------------------------------------------

@callback(
    Output("patterns-pane", "children", allow_duplicate=True),
    Output("pattern-count", "children", allow_duplicate=True),
    Input("refresh-patterns-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cb_refresh_patterns(_n):
    conn = db_interface.get_connection()
    try:
        return _render_patterns(conn), _count_label(len(db_interface.get_active_patterns(conn)))
    finally:
        conn.close()


@callback(
    Output("patterns-pane", "children", allow_duplicate=True),
    Output("pattern-count", "children", allow_duplicate=True),
    Output("pattern-status", "children"),
    Output("pattern-status", "is_open"),
    Input({"type": "pat-carry", "pid": ALL}, "n_clicks"),
    Input({"type": "pat-incorporate", "pid": ALL}, "n_clicks"),
    Input({"type": "pat-reject", "pid": ALL}, "n_clicks"),
    State({"type": "pat-reject-note", "pid": ALL}, "value"),
    prevent_initial_call=True,
)
def cb_pattern_fate(_carry, _incorp, _reject, _reject_notes):
    trig = ctx.triggered_id
    clicks = (_carry or []) + (_incorp or []) + (_reject or [])
    if not trig or not any(c for c in clicks if c):
        return no_update, no_update, no_update, no_update
    pid, typ = trig["pid"], trig["type"]
    conn = db_interface.get_connection()
    try:
        if typ == "pat-carry":
            db_interface.add_pattern_event(conn, pid, "carried")
            msg = "Carried forward -- still open for a later round."
        elif typ == "pat-incorporate":
            profile_id = db_interface.get_or_create_profile(
                conn, profile_interface.read_active_or_empty())
            db_interface.add_pattern_event(conn, pid, "incorporated", profile_id=profile_id)
            msg = f"Incorporated into active profile {profile_id[:12]}."
        else:  # pat-reject
            note = _state_value(ctx.states_list[0], pid)
            db_interface.add_pattern_event(conn, pid, "rejected", note=note or None)
            msg = "Rejected -- kept as a tombstone; the suggester will not re-propose it."
        children = _render_patterns(conn)
        count = _count_label(len(db_interface.get_active_patterns(conn)))
    finally:
        conn.close()
    return children, count, msg, True


@callback(
    Output("pattern-status", "children", allow_duplicate=True),
    Output("pattern-status", "is_open", allow_duplicate=True),
    Input({"type": "pat-save", "pid": ALL}, "n_clicks"),
    State({"type": "pat-name", "pid": ALL}, "value"),
    State({"type": "pat-dir", "pid": ALL}, "value"),
    State({"type": "pat-desc", "pid": ALL}, "value"),
    State({"type": "pat-sugg", "pid": ALL}, "value"),
    prevent_initial_call=True,
)
def cb_pattern_save(clicks, _names, _dirs, _descs, _suggs):
    trig = ctx.triggered_id
    if not trig or not any(c for c in (clicks or []) if c):
        return no_update, no_update
    pid = trig["pid"]
    conn = db_interface.get_connection()
    try:
        db_interface.update_pattern_content(
            conn, pid,
            name=_state_value(ctx.states_list[0], pid),
            direction=_state_value(ctx.states_list[1], pid),
            description=_state_value(ctx.states_list[2], pid),
            suggested_edit=_state_value(ctx.states_list[3], pid),
        )
    finally:
        conn.close()
    return "Saved pattern edits (content only; the fate log is untouched).", True


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
    Input("refresh-patterns-btn", "n_clicks"),
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
# Callbacks: chat
# ---------------------------------------------------------------------------

@callback(
    Output("chat-input", "value", allow_duplicate=True),
    Input({"type": "pat-discuss", "pid": ALL}, "n_clicks"),
    State({"type": "pat-sugg", "pid": ALL}, "value"),
    State({"type": "pat-name", "pid": ALL}, "value"),
    prevent_initial_call=True,
)
def cb_discuss(clicks, _suggs, _names):
    trig = ctx.triggered_id
    if not trig or not any(c for c in (clicks or []) if c):
        return no_update
    pid = trig["pid"]
    name = _state_value(ctx.states_list[1], pid) or ""
    sugg = _state_value(ctx.states_list[0], pid) or ""
    return (
        "Is this suggestion already covered by my current profile, or is it a genuine gap? "
        "Quote the overlapping profile text if it is redundant.\n\n"
        f"Suggestion ({name}):\n{sugg}"
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
