"""
artifact_editor.py -- the editor shared by every workbench that edits a hand-authored artifact:
the profile and the judge prompt (judge_workbench), and the two sections of the analysis prompt
(analysis_workbench). Before 2026-10-02 it lived inside judge_workbench.py; moving it out is
what lets the analysis workbench reuse it rather than copy it.

One pane per artifact: Save version, Set as active, Reload from disk, Restore autosave, a picker
that loads a saved version back into the box, and the box itself. An app mounts one pane per
artifact plus artifact_selector() to flip between them, and calls register() first.

An artifact's INTERFACE is anything offering active_path, read_active_or_empty, set_active,
save_version, list_versions, latest_version, read_version, save_autosave and load_autosave:
profile_interface and prompt_interface as modules, analysis_prompt_interface.Section per tab.
set_active validates its own artifact and raises ValueError on a refusal, which the pane shows.

register() takes a FUNCTION from artifact to interface, not the interface, and that is
load-bearing. The free gates fake an artifact by replacing an attribute on the APP module, so
the lookup has to run inside the app at call time; an interface captured here at import would
keep the real module and let a free gate overwrite a live file, which is how the 2026-09-22
incident happened.
"""

from datetime import datetime
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import ALL, MATCH, Input, Output, State, callback, ctx, dcc, html, no_update

PANE_STYLE = {"height": "56vh", "overflowY": "auto", "padding": "0 14px"}

# artifact -> (the name shown in its pane, the app's function from artifact to interface)
_REGISTRY = {}


def register(labels, interface_for):
    """Tell the shared callbacks how to reach each artifact. `labels` maps artifact -> the name
    shown in its pane; `interface_for(artifact)` must look the interface up at call time."""
    for artifact, label in labels.items():
        _REGISTRY[artifact] = (label, interface_for)


def label(artifact):
    return _REGISTRY[artifact][0]


def interface(artifact):
    _label, interface_for = _REGISTRY[artifact]
    return interface_for(artifact)


def aid(kind, artifact):
    """A pattern-matched component id for one artifact's editor."""
    return {"type": f"art-{kind}", "artifact": artifact}


def _latest_version_note(artifact):
    v = interface(artifact).latest_version()
    return f"latest version: {v.name}" if v else "no saved versions yet"


def _version_options(artifact):
    return [{"label": p.name, "value": str(p)} for p in interface(artifact).list_versions()]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def artifact_selector(artifacts, selected):
    """Flips between the panes. A pure SELECTOR over sibling divs, never dcc.Tab children: Tab
    children unmount when you switch away, and an unmounted control comes back with n_clicks=0,
    which Dash cannot tell from a click. Every pane stays in the DOM; all but one are hidden.
    Takes the app's own artifacts, since the registry holds every app's in a shared process."""
    return dbc.RadioItems(id="artifact-tabs", value=selected, inline=True,
                          className="mb-2 small",
                          options=[{"label": f" {label(a)} ", "value": a} for a in artifacts])


def editor_pane(artifact, visible=True, extra=()):
    """One artifact's editor, mounted always and hidden by style rather than by being absent --
    the house rule that fixed components stay in the DOM. `extra` is anything one app adds under
    one pane (the judge prompt's harness button)."""
    return html.Div(
        id=aid("pane", artifact),
        style={**PANE_STYLE, **({} if visible else {"display": "none"})},
        children=[
            html.Div([
                dbc.Button("Save version", id=aid("save-version", artifact),
                           color="primary", size="sm", className="me-2"),
                dbc.Button("Set as active", id=aid("set-active", artifact),
                           color="danger", outline=True, size="sm", className="me-2"),
                dbc.Button("Reload from disk", id=aid("reload", artifact),
                           color="secondary", outline=True, size="sm", className="me-2"),
                dbc.Button("Restore autosave", id=aid("restore-autosave", artifact),
                           color="warning", outline=True, size="sm"),
            ], className="mb-2"),
            dbc.Alert(id=aid("status", artifact), is_open=False, duration=4000,
                      color="success", className="py-1 px-2 small"),
            html.Small(f"active {label(artifact)}: {interface(artifact).active_path()}",
                       className="text-muted d-block"),
            html.Small(id=aid("version-note", artifact), className="text-muted d-block"),
            html.Small(id=aid("autosave-note", artifact), className="text-muted d-block mb-2"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id=aid("version-pick", artifact),
                                     placeholder="load a saved version into the box...",
                                     style={"fontSize": "0.8rem"})),
                dbc.Col(dbc.Button("Load", id=aid("load-version", artifact), color="secondary",
                                   outline=True, size="sm"), width="auto"),
            ], className="g-2 mb-2 align-items-center"),
            # read_active_or_empty, never load_active: the latter RAISES when the artifact has
            # not been authored, and this runs at import, so it would take the CLI command and
            # the free workbench gates down with it for a user who has not written one yet.
            dbc.Textarea(id=aid("editor", artifact),
                         value=interface(artifact).read_active_or_empty(),
                         persistence=True, persistence_type="local",
                         style={"width": "100%", "height": "44vh",
                                "fontFamily": "monospace", "fontSize": "0.85rem"}),
            # One timer PER PANE. A single fixed-id Interval cannot drive a MATCH output --
            # Dash requires the matched key to appear in an Input or State of the same callback.
            dcc.Interval(id=aid("autosave-timer", artifact), interval=20000),
            *extra,
        ])


# ---------------------------------------------------------------------------
# Callbacks: ONE registration each, serving every pane in whichever app is running
# ---------------------------------------------------------------------------
# Each is keyed on MATCH, so the same body runs for whichever pane's button was pressed, and
# ctx.triggered_id["artifact"] says which.

@callback(
    Output(aid("version-note", MATCH), "children"),
    Output(aid("version-pick", MATCH), "options"),
    Input(aid("pane", MATCH), "id"),
)
def cb_init_pane(pane_id):
    """Fills a pane's version note and version list on page load. The input is the pane's own
    id, which never changes, so this fires exactly once per pane per load."""
    artifact = pane_id["artifact"]
    return _latest_version_note(artifact), _version_options(artifact)


@callback(
    Output(aid("status", MATCH), "children", allow_duplicate=True),
    Output(aid("status", MATCH), "is_open", allow_duplicate=True),
    Output(aid("status", MATCH), "color", allow_duplicate=True),
    Output(aid("version-note", MATCH), "children", allow_duplicate=True),
    Output(aid("version-pick", MATCH), "options", allow_duplicate=True),
    Input(aid("save-version", MATCH), "n_clicks"),
    State(aid("editor", MATCH), "value"),
    prevent_initial_call=True,
)
def cb_save_version(_n, text):
    artifact = ctx.triggered_id["artifact"]
    path = interface(artifact).save_version(text or "")
    return (f"Saved to version: {path.name}", True, "success",
            _latest_version_note(artifact), _version_options(artifact))


@callback(
    Output(aid("status", MATCH), "children", allow_duplicate=True),
    Output(aid("status", MATCH), "is_open", allow_duplicate=True),
    Output(aid("status", MATCH), "color", allow_duplicate=True),
    Input(aid("set-active", MATCH), "n_clicks"),
    State(aid("editor", MATCH), "value"),
    prevent_initial_call=True,
)
def cb_set_active(_n, text):
    """No per-artifact guard here, deliberately: set_active validates its own artifact and
    raises ValueError, so this shows the refusal and every other caller is covered too."""
    artifact = ctx.triggered_id["artifact"]
    try:
        backup = interface(artifact).set_active(text or "")
    except ValueError as e:
        return str(e), True, "danger"
    msg = f"Set as active {label(artifact)}."
    if backup:
        msg += f" The outgoing version was backed up to {backup.name}."
    return msg, True, "success"


@callback(
    Output(aid("editor", MATCH), "value"),
    Output(aid("status", MATCH), "children", allow_duplicate=True),
    Output(aid("status", MATCH), "is_open", allow_duplicate=True),
    Output(aid("status", MATCH), "color", allow_duplicate=True),
    Input(aid("reload", MATCH), "n_clicks"),
    prevent_initial_call=True,
)
def cb_reload_artifact(_n):
    artifact = ctx.triggered_id["artifact"]
    return (interface(artifact).read_active_or_empty(),
            f"Reloaded {label(artifact)} from disk.", True, "success")


@callback(
    Output(aid("editor", MATCH), "value", allow_duplicate=True),
    Output(aid("status", MATCH), "children", allow_duplicate=True),
    Output(aid("status", MATCH), "is_open", allow_duplicate=True),
    Output(aid("status", MATCH), "color", allow_duplicate=True),
    Input(aid("load-version", MATCH), "n_clicks"),
    State(aid("version-pick", MATCH), "value"),
    prevent_initial_call=True,
)
def cb_load_version(_n, path):
    """Loads a saved version INTO THE BOX only. Nothing is written -- Set as active stays the one
    door to the live file -- so trying an old version costs nothing."""
    artifact = ctx.triggered_id["artifact"]
    if not path:
        return no_update, "Pick a saved version first.", True, "warning"
    try:
        text = interface(artifact).read_version(path)
    except (OSError, ValueError) as e:
        return no_update, f"Could not load {Path(path).name}: {e}", True, "danger"
    return (text, f"Loaded {Path(path).name} into the box. Not active until you set it.",
            True, "success")


@callback(
    Output(aid("autosave-note", MATCH), "children"),
    Input(aid("autosave-timer", MATCH), "n_intervals"),
    State(aid("editor", MATCH), "value"),
    prevent_initial_call=True,
)
def cb_autosave(_n, text):
    artifact = ctx.triggered_id["artifact"]
    if interface(artifact).save_autosave(text):
        return f"autosaved draft {datetime.now():%H:%M:%S}"
    return no_update


@callback(
    Output(aid("editor", MATCH), "value", allow_duplicate=True),
    Output(aid("status", MATCH), "children", allow_duplicate=True),
    Output(aid("status", MATCH), "is_open", allow_duplicate=True),
    Output(aid("status", MATCH), "color", allow_duplicate=True),
    Input(aid("restore-autosave", MATCH), "n_clicks"),
    prevent_initial_call=True,
)
def cb_restore_autosave(_n):
    artifact = ctx.triggered_id["artifact"]
    text = interface(artifact).load_autosave()
    if text:
        return text, "Restored autosave draft into the editor.", True, "success"
    return no_update, "No autosave draft found.", True, "warning"


@callback(
    Output(aid("pane", ALL), "style"),
    Input("artifact-tabs", "value"),
)
def cb_show_artifact(selected):
    """Show one pane, hide the rest. Style only -- nothing unmounts, so every editor keeps its
    draft, its persistence and its n_clicks history when you flip between artifacts."""
    return [{**PANE_STYLE, **({} if o["id"]["artifact"] == selected else {"display": "none"})}
            for o in ctx.outputs_list]
