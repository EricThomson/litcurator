"""
prompt_workbench.py -- edit and version the active JUDGE PROMPT.

The judge prompt is the scoring PROCEDURE (how the judge scores a paper's expected
interest for the user) plus stable user calibrations -- e.g. journal venue weighting
-- that don't belong in the evolving profile. This is the prompt half of the
biconvex problem, edited here by hand. The human authors every word; the LLM is a
bounded critic, never an autonomous editor (that would be the v1 failure at the
prompt level).

  - "Save version" writes a timestamped copy to versions/.
  - "Set as active" snapshots the outgoing active into versions/, writes
    judge_prompt.md, and registers the new version in the DB prompts table
    (parent_id = outgoing). Refuses a prompt missing its '## Output' section,
    since the batch judge derives its output contract from that marker.

Bottom: an Opus critic sounding-board. Committed prompt + live draft both loaded
as context so it can reason about the delta and push back on bloat / over-correction
/ profile-vs-prompt confusion -- it surfaces issues, you author the words.

Run:
    litcurator prompt_workbench
    python src/litcurator/apps/prompt_workbench.py
"""

import os
from datetime import datetime

import anthropic
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback, dcc, html, no_update
from dotenv import load_dotenv

from litcurator import prompt_interface

load_dotenv()

# Opus as the critic: it critiques and surfaces, it does not author prose, which is
# exactly where Opus is strong and the v1 bloat risk is lowest (see the
# feedback_prompt_authorship and feedback_critic_in_loop memories).
CRITIC_MODEL = "claude-opus-4-8"

OUTPUT_MARKER = "## Output"   # structural contract: the batch judge derives from this


# ---------------------------------------------------------------------------
# Chat (bounded critic)
# ---------------------------------------------------------------------------

def _chat_system(committed, draft):
    return (
        "You are a sharp prompt-engineering critic helping a researcher refine the JUDGE PROMPT "
        "for a personal paper-curation system. The judge prompt is the scoring PROCEDURE -- how an "
        "LLM judge scores a paper's expected interest for this user, given the user's SEPARATE "
        "profile. It also holds STABLE user calibrations (e.g. journal venue weighting) that "
        "deliberately do NOT live in the evolving profile.\n\n"
        "You are given TWO versions so you can reason about the DELTA:\n"
        "- COMMITTED PROMPT: the stable, on-disk judge prompt (the 'before').\n"
        "- CURRENT DRAFT: what the researcher is editing right now (the 'after'). It may differ "
        "from the committed prompt, or be identical -- they are mid-edit, so do not assume a draft "
        "line is settled just because it is there.\n\n"
        "Help them THINK and CRITIQUE; do NOT write the prompt for them. Be honest and specific, "
        "and push back hard when an edit risks:\n"
        "- BLOAT / vocabulary drift: this system's original failure was an LLM endlessly elaborating "
        "prose. Fewer, sharper instructions beat more. If a line restates something already present, "
        "say so and quote it.\n"
        "- OVER-CORRECTION: a fix for one failure slice (e.g. the judge over-scoring molecular-"
        "systems borderline papers) that would suppress things the user actually wants (synaptic "
        "plasticity, molecular sensors/tools for systems work). Name the collateral.\n"
        "- PROFILE-vs-PROMPT confusion: the prompt is HOW to score + stable calibrations; the "
        "evolving WHAT (topic taste) belongs in the profile. Flag anything that should live in the "
        "profile instead.\n"
        "- STRUCTURAL breakage: the prompt must keep its '## Output' section (the batch judge "
        "derives its format from it) and its four-key JSON output contract.\n\n"
        "Only draft prompt prose if explicitly asked, and keep it surgical and in their voice. Keep "
        "answers short unless asked to expand. ASCII only.\n\n"
        "----- COMMITTED PROMPT (before) -----\n"
        f"{committed}\n"
        "----- END COMMITTED PROMPT -----\n\n"
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


def _latest_version_note():
    v = prompt_interface.latest_version()
    return f"latest version: {v.name}" if v else "no saved versions yet"


def _active_note():
    vid = prompt_interface.active_version_id()
    return f"active prompt: {prompt_interface.active_path()}  ({vid})" if vid else \
        f"active prompt: {prompt_interface.active_path()}  (not seeded yet)"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
app.title = "Prompt Workbench"

app.layout = dbc.Container([
    dcc.Store(id="chat-history", data=[], storage_type="session"),

    dbc.Row([
        dbc.Col(html.H4("Prompt Workbench", className="mb-0"), width="auto"),
        dbc.Col(html.Small("the JUDGE prompt -- scoring procedure + stable calibrations "
                           "(e.g. journal weighting). The human authors every word.",
                           className="text-muted"), width="auto", align="center"),
    ], align="center", className="mt-3 mb-2 g-2"),

    html.Div([
        dbc.Button("Save version", id="save-version-btn", color="primary",
                   size="sm", className="me-2"),
        dbc.Button("Set as active", id="set-active-btn", color="danger",
                   outline=True, size="sm", className="me-2"),
        dbc.Button("Reload from disk", id="reload-btn", color="secondary",
                   outline=True, size="sm", className="me-2"),
        dbc.Button("Restore autosave", id="restore-autosave-btn", color="warning",
                   outline=True, size="sm", className="me-2"),
        dbc.Button("Run tests", id="run-tests-btn", color="info",
                   outline=True, size="sm"),
    ], className="mb-2"),
    dbc.Alert(id="prompt-status", is_open=False, duration=5000, color="success",
              className="py-1 px-2 small"),
    html.Small(_active_note(), id="active-note", className="text-muted d-block"),
    html.Small(id="version-note", className="text-muted d-block"),
    html.Small(id="autosave-note", className="text-muted d-block mb-2"),

    dbc.Textarea(id="prompt-editor",
                 value=prompt_interface.load_active(),
                 persistence=True, persistence_type="local",
                 style={"width": "100%", "height": "52vh",
                        "fontFamily": "monospace", "fontSize": "0.85rem"}),
    dcc.Interval(id="autosave-timer", interval=20000),

    dcc.Loading(html.Div(id="test-results",
                         style={"whiteSpace": "pre-wrap", "fontFamily": "monospace",
                                "fontSize": "0.78rem", "marginTop": "8px"})),

    html.Hr(className="my-2"),

    html.Div([
        html.Div([
            html.Span("Critic", className="fw-semibold me-2"),
            html.Small(f"(context: committed prompt + live draft | model: {CRITIC_MODEL})",
                       className="text-muted"),
            dbc.Button("Clear", id="chat-clear-btn", color="secondary",
                       outline=True, size="sm", className="float-end"),
        ], className="mb-2"),
        dcc.Loading(html.Div(id="chat-thread",
                             style={"height": "24vh", "overflowY": "auto",
                                    "padding": "4px 8px"})),
        dbc.InputGroup([
            dbc.Textarea(id="chat-input",
                         placeholder='e.g. "I want to add journal weighting -- where, and will it '
                                     'conflict with the article-length section?"',
                         style={"minHeight": "60px"}),
            dbc.Button("Send", id="chat-send-btn", color="primary"),
        ], className="mt-1"),
    ]),
], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks: editor
# ---------------------------------------------------------------------------

@callback(
    Output("prompt-status", "children", allow_duplicate=True),
    Output("prompt-status", "is_open", allow_duplicate=True),
    Output("version-note", "children", allow_duplicate=True),
    Input("save-version-btn", "n_clicks"),
    State("prompt-editor", "value"),
    prevent_initial_call=True,
)
def cb_save_version(_n, text):
    path = prompt_interface.save_version(text or "")
    return f"Saved to version: {path.name}", True, _latest_version_note()


@callback(
    Output("prompt-status", "children", allow_duplicate=True),
    Output("prompt-status", "is_open", allow_duplicate=True),
    Output("prompt-status", "color", allow_duplicate=True),
    Output("version-note", "children", allow_duplicate=True),
    Output("active-note", "children", allow_duplicate=True),
    Input("set-active-btn", "n_clicks"),
    State("prompt-editor", "value"),
    prevent_initial_call=True,
)
def cb_set_active(_n, text):
    text = text or ""
    if OUTPUT_MARKER not in text:
        return (f"Refused: the prompt must contain a '{OUTPUT_MARKER}' section -- the batch judge "
                f"derives its output contract from it. Add it before setting active.",
                True, "danger", no_update, no_update)
    backup = prompt_interface.set_active(text)
    msg = "Set as active judge prompt."
    if backup:
        msg += f" Outgoing prompt backed up to {backup.name}."
    return msg, True, "success", _latest_version_note(), _active_note()


@callback(
    Output("prompt-editor", "value"),
    Output("prompt-status", "children", allow_duplicate=True),
    Output("prompt-status", "is_open", allow_duplicate=True),
    Input("reload-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cb_reload(_n):
    return prompt_interface.load_active(), "Reloaded prompt from disk.", True


@callback(
    Output("test-results", "children"),
    Input("run-tests-btn", "n_clicks"),
    State("prompt-editor", "value"),
    prevent_initial_call=True,
)
def cb_run_tests(_n, draft):
    # Runs the judge harness on the live DRAFT against the active profile -- the
    # in-edit-loop floor gate. Saved report includes the per-case rationales.
    from litcurator import judge_harness
    results, prompt_fp, profile_fp = judge_harness.run_tests(prompt_text=draft or "")
    report = judge_harness.format_report(results, prompt_fp, profile_fp)
    path = judge_harness.write_report(report + "\n" + judge_harness.format_rationales(results))
    return f"{report}\n\nsaved to {path}"


@callback(
    Output("autosave-note", "children"),
    Input("autosave-timer", "n_intervals"),
    State("prompt-editor", "value"),
    prevent_initial_call=True,
)
def cb_autosave(_n, text):
    if prompt_interface.save_autosave(text):
        return f"autosaved draft {datetime.now():%H:%M:%S}"
    return no_update


@callback(
    Output("prompt-editor", "value", allow_duplicate=True),
    Output("prompt-status", "children", allow_duplicate=True),
    Output("prompt-status", "is_open", allow_duplicate=True),
    Input("restore-autosave-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cb_restore_autosave(_n):
    text = prompt_interface.load_autosave()
    if text:
        return text, "Restored autosave draft into the editor.", True
    return no_update, "No autosave draft found.", True


# ---------------------------------------------------------------------------
# Callbacks: critic chat
# ---------------------------------------------------------------------------

@callback(
    Output("chat-history", "data"),
    Output("chat-input", "value"),
    Input("chat-send-btn", "n_clicks"),
    State("chat-input", "value"),
    State("chat-history", "data"),
    State("prompt-editor", "value"),
    prevent_initial_call=True,
)
def cb_chat_send(_n, user_text, history, draft):
    if not (user_text and user_text.strip()):
        return no_update, no_update
    history = list(history or [])
    history.append({"role": "user", "content": user_text.strip()})
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=CRITIC_MODEL,
        max_tokens=1500,
        system=_chat_system(prompt_interface.read_active_or_empty(), draft or ""),
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

def run_app(port=8056):
    # use_reloader=False: hot reload would reset the editor textarea and wipe
    # unsaved work. Restart manually to pick up code changes.
    app.run(debug=True, use_reloader=False, port=port)


if __name__ == "__main__":
    run_app()
