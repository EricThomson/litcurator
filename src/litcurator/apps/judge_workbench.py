"""
judge_workbench.py -- curate the pattern memory and edit the judge's two inputs.

Left panel: the ACTIVE PATTERNS (db_interface.get_active_patterns) -- recurring
taste-gaps the suggester surfaced from your flags, each with a drill-down to the
papers behind it. Per pattern you can edit its wording and then decide its fate,
which is written to the append-only pattern_events log (nothing is ever deleted):
  - Incorporate: you folded it into the profile. Stamps the currently-active
    profile version, drops the pattern off the active list.
  - Hold: not yet. Keeps the pattern, its provenance and its event log, takes it
    off the queue, and returns its flags to the clustering pool so the taste is
    re-decided next round on the evidence. Shown in the Held tab meanwhile.
    (Replaced "Carry", which kept the queue slot and made the list grow forever.)
  - Reject (with a reason): not a real gap. Drops off the active list, kept as a
    closed pattern so the suggester will not re-propose it.

Right panel: the judge's two inputs, the profile and the judge prompt, one tab each, in the
editor shared with the analysis workbench (artifact_editor.py). Never overwritten silently:
"Save version" writes a timestamped copy to versions/, and "Set as active" snapshots the
outgoing active, writes the new one and registers it in the DB (parent_id = outgoing). The
judge prompt tab also runs the judge harness on the draft.

Bottom: chat. A Sonnet partner for the profile, a bounded Opus critic for the judge prompt;
the committed text and the live draft are both loaded as context.

The human authors every word of the profile; the machinery only surfaces and
remembers. Typical flow: Discuss / edit a pattern -> author the profile edit on the
right -> Set as active -> Incorporate the pattern (it stamps that version).

Run:
    litcurator judge_workbench
    python src/litcurator/apps/judge_workbench.py
"""

import os

import anthropic
import dash_bootstrap_components as dbc
from dash import (ALL, Dash, Input, Output, State, callback, ctx, dcc, html,
                  no_update)
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle
from dotenv import load_dotenv

from litcurator import db_interface, profile_interface, prompt_interface
from litcurator.apps import artifact_editor
from litcurator.apps.artifact_editor import aid
from litcurator.config import LEVELS_BUCKET_DESCRIPTION

load_dotenv()

CHAT_MODEL = "claude-sonnet-4-6"          # profile: a thinking PARTNER
CRITIC_MODEL = "claude-opus-4-8"          # judge prompt: a bounded CRITIC

# The chat is the one place the two artifacts genuinely differ, so it is the one place
# with an explicit branch. Editing taste prose and critiquing a scoring procedure are
# different jobs with different models -- a doctrine split predating the merge, carried
# across unchanged rather than harmonised into a single bland assistant.
_CHAT_CONFIG = {
    "profile": {"model": CHAT_MODEL, "max_tokens": 1200},
    "prompt": {"model": CRITIC_MODEL, "max_tokens": 1500},
}

DIRECTIONS = list(db_interface.DIRECTIONS)
# Badge colours, UI only. Asserted so a new direction cannot render as an unexplained grey.
_DIR_COLOR = {"over": "danger", "under": "success"}
assert set(_DIR_COLOR) == set(DIRECTIONS), sorted(set(DIRECTIONS) - set(_DIR_COLOR))


# ---------------------------------------------------------------------------
# Pattern cards
# ---------------------------------------------------------------------------

def _render_provenance(prov):
    """The papers behind a pattern, read-only. Largest |delta| first."""
    if not prov:
        return [html.Div("(no papers)", className="text-muted small")]
    out = []
    for f in prov:
        note = f"  -- {f['note']}" if f.get("note") else ""
        out.append(html.Div(
            f"delta {f['delta']:+.2f} (judge {f['judge_score']:.2f} -> you "
            f"{f['user_score']:.2f})  {f.get('journal') or ''}: {f.get('title') or ''}{note}",
            className="small text-muted"))
    return out


def _pattern_card(conn, p, held=False, prelude=None):
    """One pattern, editable. `held` swaps the fate buttons: a pattern you have not been shown
    yet cannot sensibly be Carried (it is not on your list) or Incorporated (you have not read
    it), so it offers Promote -- put it on the list -- and Reject."""
    pid = p["id"]
    prov = db_interface.get_pattern_provenance(conn, pid)
    return html.Div(dbc.Card(dbc.CardBody([
        *(prelude or []),
        html.Div([
            dbc.Badge(p["direction"], color=_DIR_COLOR.get(p["direction"], "secondary"),
                      className="me-2", style={"flex": "0 0 auto"}),
            dbc.Input(id={"type": "pat-name", "pid": pid}, value=p["name"], size="sm",
                      style={"flex": "1 1 auto", "minWidth": 0}),
            *([dbc.Badge(f"|d| {p['mean_abs_delta']:.2f}", color="light", text_color="dark",
                         title="mean |delta| over this pattern's papers; max "
                               f"{p.get('max_abs_delta') or 0:.2f}",
                         className="ms-2", style={"flex": "0 0 auto"})]
              if p.get("mean_abs_delta") is not None else []),
            dbc.Badge(p["status"], color="light", text_color="dark", className="ms-2",
                      style={"flex": "0 0 auto"}),
            *([dbc.Badge("PROMPT JOB", color="warning", text_color="dark", className="ms-2",
                         title="the profile already states this and the judge scored against "
                               "it anyway -- more profile prose will not help",
                         style={"flex": "0 0 auto"})]
              if p.get("blame") == "prompt" else []),
        ], className="d-flex align-items-center mb-2"),
        dbc.Row([
            dbc.Col(dbc.Select(id={"type": "pat-dir", "pid": pid},
                               options=[{"label": d, "value": d} for d in DIRECTIONS],
                               value=p["direction"], size="sm"), width=6),
            # WHICH ARTIFACT IS AT FAULT. The model proposes it from the note; this is where
            # you overrule it. A prompt-blamed pattern closes against the judge prompt, so the
            # 'incorporated' event stamps the artifact that actually absorbed the fix.
            dbc.Col(dbc.Select(id={"type": "pat-blame", "pid": pid},
                               options=[{"label": "fix: profile", "value": "profile"},
                                        {"label": "fix: judge prompt", "value": "prompt"}],
                               value=p.get("blame") or "profile", size="sm"), width=6),
        ], className="g-2 mb-2"),
        html.Small("description", className="text-muted"),
        dbc.Textarea(id={"type": "pat-desc", "pid": pid}, value=p.get("description") or "",
                     style={"height": "3rem", "fontSize": "0.8rem"}, className="mb-2"),
        html.Small("suggested edit -- what to change in the JUDGE PROMPT (scoring "
                   "procedure); profile prose will not fix this one"
                   if p.get("blame") == "prompt" else
                   "suggested edit (your working draft -- you author the profile prose)",
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
            *([dbc.Button("Promote", id={"type": "pat-promote", "pid": pid},
                          color="success", size="sm", className="me-1")]
              if held else
              [dbc.Button("Hold", id={"type": "pat-hold", "pid": pid},
                          color="secondary", outline=True, size="sm",
                          title="not this round -- keeps the pattern and its evidence, takes it "
                                "off the queue, and puts its flags back in the clustering pool",
                          className="me-1"),
               dbc.Button("Incorporate", id={"type": "pat-incorporate", "pid": pid},
                          color="success", size="sm", className="me-1")]),
            dbc.Button("Reject", id={"type": "pat-reject", "pid": pid},
                       color="danger", outline=True, size="sm"),
        ]),
    ]), className="mb-3", style={"border": "1px solid #e3dcf2"}),
        id={"type": "pat-card", "pid": pid})


def _recurrence_prelude(conn, p):
    """The banner on a closed pattern that keeps coming back. It LEADS WITH THE DECISION -- what
    you chose, when, and how many times it has returned since -- because the whole hazard of
    showing these beside held patterns is re-deciding something you already decided without
    realising it.

    The papers are shown rather than summarised. A count is ambiguous three ways (the edit did
    not take / the rejection was wrong / a vague pattern is attracting false merges) and only
    the papers separate them: same kind of paper is a real return, drift is an attractor."""
    decided = (p.get("status_at") or "")[:10]
    verb = "INCORPORATED" if p["status"] == "incorporated" else "REJECTED"
    meaning = ("the edit did not take -- you stated this and the judge still gets it wrong"
               if p["status"] == "incorporated"
               else "the rejection may have been wrong -- the flags keep bringing it back")
    papers = db_interface.get_post_closure_papers(conn, p["id"])
    return [
        dbc.Alert([
            html.Div([
                dbc.Badge(f"{verb} {decided}", color="dark", className="me-2"),
                dbc.Badge(f"returned {p['recurrence_count']}x", color="warning",
                          text_color="dark"),
            ], className="mb-1"),
            html.Div(meaning, className="small"),
            html.Details([
                html.Summary(f"{len(papers)} paper(s) since you decided",
                             className="small"),
                html.Div([
                    html.Div(f"delta {x['delta']:+.2f}  {x['journal'] or ''}: "
                             f"{x['title'] or ''}"
                             + (f"  -- {x['note']}" if x.get("note") else ""),
                             className="small text-muted")
                    for x in papers] or [html.Div("(none recorded)",
                                                  className="small text-muted")],
                    className="mt-1"),
            ], className="mt-1"),
        ], color="warning", className="py-2 px-2 mb-2"),
    ]


def _levels_bucket_line(conn):
    """The Levels Bucket as ONE line: its name and how many papers are in it, with the
    description on hover. It is not a pattern (see config.LEVELS_BUCKET_NAME), so it gets none
    of the card's fields or buttons, which is also why no callback can touch it and it cannot
    be renamed out from under the review feed's lookup. Nothing shows before the first click."""
    bucket = db_interface.get_levels_bucket(conn)
    if bucket is None:
        return []
    n = bucket["paper_count"]
    return [html.Div(
        html.Span(f"{bucket['name']}: {n} paper{'' if n == 1 else 's'}",
                  title=LEVELS_BUCKET_DESCRIPTION,
                  style={"cursor": "help", "textDecoration": "underline dotted"}),
        className="small text-muted mb-3")]


def _render_patterns(conn, tab="active"):
    held = tab == "held"
    patterns = (db_interface.get_held_patterns(conn) if held
                else db_interface.get_active_patterns(conn))
    bucket = [] if held else _levels_bucket_line(conn)
    # CLOSED PATTERNS THAT KEEP COMING BACK share the Held tab, because the semantics rhyme:
    # held is "not decided, evidence accumulating", recurring-closed is "decided, evidence
    # accumulating AGAINST the decision". Both are accumulation watch-lists wanting the same
    # actions, and Promote already does the right thing on a closed row -- it writes `carried`,
    # which under latest-event-wins REOPENS it. No new event semantics, no second tab.
    #
    # get_closed_recurrences was written, correct and caller-less from 2026-08-25 until now:
    # the signal was recorded faithfully and shown to nobody, while the harness graded it 3/3
    # every sweep. A green test for an orphaned producer is camouflage.
    recurring = db_interface.get_closed_recurrences(conn) if held else []
    if not patterns and not recurring:
        return bucket + [html.Div(
            "Nothing held. The consolidate step records a pattern here when it is real but "
            "not yet worth your attention; it moves to Active once enough evidence arrives."
            if held else
            "No open patterns. Flag papers in the review feed, then run "
            "`litcurator error_analysis` to surface patterns here.",
            className="text-muted")]
    cards = [_pattern_card(conn, p, held=held) for p in patterns]
    cards += [_pattern_card(conn, p, held=True, prelude=_recurrence_prelude(conn, p))
              for p in recurring]
    return bucket + cards


def _counts(conn):
    """Both numbers, always. The held count is the whole safeguard: a held pattern is one you
    were deliberately not shown, so the only thing standing between "recorded" and "invisible"
    is a number you see every session. A held pile that climbs round after round while nothing
    promotes is the model holding too eagerly."""
    return len(db_interface.get_active_patterns(conn)), len(db_interface.get_held_patterns(conn))


def _initial_patterns():
    conn = db_interface.get_connection()
    try:
        return _render_patterns(conn), _counts(conn)
    finally:
        conn.close()


def _count_label(counts):
    n_open, n_held = counts if isinstance(counts, tuple) else (counts, 0)
    held = f"  |  {n_held} held" if n_held else ""
    return f"{n_open} open pattern{'' if n_open == 1 else 's'}{held}"


def _state_value(states, pid):
    """The value of the pattern-matched State whose id has this pid."""
    for s in states or []:
        if s.get("id", {}).get("pid") == pid:
            return s.get("value")
    return None


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def _prompt_critic_system(committed, draft):
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

# ---------------------------------------------------------------------------
# The two artifacts the judge reads
# ---------------------------------------------------------------------------
# The judge is profile x prompt, so this bench edits both, in the editor it shares with the
# analysis workbench (artifact_editor). The panes it mounts, in DOM order:
ARTIFACTS = ("profile", "prompt")
_ARTIFACT_LABEL = {"profile": "profile", "prompt": "judge prompt"}


def _interface(artifact):
    """The interface module for an artifact. profile_interface and prompt_interface are
    deliberate mirrors of each other, so a caller needs nothing but this switch.

    Resolved from THIS module's globals at call time, which is load-bearing rather than
    stylistic: the workbench-actions gate fakes the profile by assigning
    judge_workbench.profile_interface, and a lookup captured at import time (a dict built once,
    say) would keep the real module and let a free gate write the user's actual profile. The
    shared editor reaches artifacts through this same function, for the same reason."""
    return prompt_interface if artifact == "prompt" else profile_interface


artifact_editor.register(_ARTIFACT_LABEL, _interface)

_initial_pattern_children, _initial_count = _initial_patterns()

# The judge prompt pane's one extra: the judge harness on the live draft. There is no profile
# equivalent, so these are fixed ids rather than pattern-matched ones.
_RUN_TESTS = [
    dbc.Button("Run judge harness", id="run-tests-btn", color="info",
               outline=True, size="sm", className="mt-2"),
    dcc.Loading(html.Pre(id="test-results",
                         style={"whiteSpace": "pre-wrap", "fontSize": "0.75rem"})),
]

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
app.title = "Judge Workbench"

app.layout = dbc.Container([
    dcc.Store(id="chat-history", data=[], storage_type="session"),
    dbc.Row([
        dbc.Col(html.H4("Judge Workbench", className="mb-0"), width="auto"),
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
                         "then Incorporate here (it stamps that version). HELD holds patterns the "
                         "consolidate step judged real but not yet worth showing -- check it "
                         "occasionally; a pile that grows while nothing promotes is over-holding.",
                         className="text-muted small mb-2"),
                dcc.Tabs(id="pattern-tabs", value="active", className="mb-2", children=[
                    dcc.Tab(label="Active", value="active"),
                    dcc.Tab(label="Held", value="held"),
                ]),
                dbc.Alert(id="pattern-status", is_open=False, duration=4000,
                          color="success", className="py-1 px-2 small"),
                html.Div(id="patterns-pane", children=_initial_pattern_children),
            ], style=artifact_editor.PANE_STYLE),
        ]),
        PanelResizeHandle(html.Div(style={
            "width": "8px", "backgroundColor": "#d9d2ee", "cursor": "col-resize",
            "height": "56vh"})),
        Panel(id="right-panel", defaultSizePercentage=50, children=[
            artifact_editor.artifact_selector(ARTIFACTS, selected="profile"),
            *[artifact_editor.editor_pane(a, visible=(a == "profile"),
                                          extra=_RUN_TESTS if a == "prompt" else ())
              for a in ARTIFACTS],
        ]),
    ]),

    html.Hr(className="my-2"),

    html.Div([
        html.Div([
            html.Span("Chat", className="fw-semibold me-2"),
            html.Small(id="chat-context-note", className="text-muted"),
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
    Input("pattern-tabs", "value"),
    prevent_initial_call=True,
)
def cb_refresh_patterns(_n, tab):
    """Refresh, and also the tab switch -- both just re-render the pane for the chosen list."""
    conn = db_interface.get_connection()
    try:
        return _render_patterns(conn, tab or "active"), _count_label(_counts(conn))
    finally:
        conn.close()


@callback(
    Output("patterns-pane", "children", allow_duplicate=True),
    Output("pattern-count", "children", allow_duplicate=True),
    Output("pattern-status", "children"),
    Output("pattern-status", "is_open"),
    Input({"type": "pat-hold", "pid": ALL}, "n_clicks"),
    Input({"type": "pat-incorporate", "pid": ALL}, "n_clicks"),
    Input({"type": "pat-reject", "pid": ALL}, "n_clicks"),
    Input({"type": "pat-promote", "pid": ALL}, "n_clicks"),
    State({"type": "pat-reject-note", "pid": ALL}, "value"),
    State("pattern-tabs", "value"),
    # APPENDED, never inserted: _state_value reads ctx.states_list[0] for the reject notes, so
    # putting a new State ahead of it would silently file reject reasons from the wrong list.
    State(aid("editor", ALL), "value"),
    prevent_initial_call=True,
)
def cb_pattern_fate(_hold, _incorp, _reject, _promote, _reject_notes, tab, _drafts):
    trig = ctx.triggered_id
    clicks = (_hold or []) + (_incorp or []) + (_reject or []) + (_promote or [])
    if not trig or not any(c for c in clicks if c):
        return no_update, no_update, no_update, no_update
    pid, typ = trig["pid"], trig["type"]
    conn = db_interface.get_connection()
    try:
        if typ == "pat-promote":
            # 'carried' means "open, not decided yet", which is exactly what a promoted held
            # pattern becomes -- so promotion needs no new event value, and the log reads
            # created -> held -> carried, which is the story you want to read.
            db_interface.add_pattern_event(conn, pid, "carried")
            msg = "Promoted -- it is on your Active list now."
        elif typ == "pat-hold":
            # REPLACED "Carry" 2026-08-30. Carry wrote `carried`, an ACTIVE status, so a
            # carried pattern kept its queue slot until incorporated or rejected -- and the
            # queue cap only bounds NEW patterns, so carried ones accumulated uncapped round
            # after round. That is the treadmill this project exists to avoid.
            #
            # From the human's side there was never a difference: carry and hold both mean "not
            # writing this into my profile today". The only difference was that one guaranteed a
            # queue slot, which removes the decision rather than deferring it. Holding keeps
            # everything -- row, provenance, event log, its place in the Held tab -- and returns
            # its flags to the clustering pool, so the taste is re-decided next round on the
            # evidence rather than by squatting.
            db_interface.add_pattern_event(conn, pid, "held")
            msg = ("Held -- off the queue, kept in memory, and its papers go back into the "
                   "pool for next round.")
        elif typ == "pat-incorporate":
            # SAVE AND STAMP, in that order, in one click. The stamp names the artifact that
            # absorbed the fix -- and now it also names a version that CONTAINS it, because the
            # draft in the editor is set active first.
            #
            # Splitting those two acts is what produced the untruth this is built to end. The
            # Annual Review pattern (2026-09-01) stamped a PROFILE version for a fix that lives
            # only in the judge prompt; and even once blame routed correctly, the editor for a
            # prompt-blamed card was a different app on a different port, so the natural order
            # was click-Incorporate-then-go-edit and the stamp named the PRE-edit prompt every
            # time. One button, one transaction, no ordering to remember.
            row = conn.execute("SELECT blame FROM patterns WHERE id = ?", (pid,)).fetchone()
            artifact = "prompt" if (row and row["blame"] == "prompt") else "profile"
            iface = _interface(artifact)
            draft = next((s["value"] for s in ctx.states_list[2]
                          if s["id"]["artifact"] == artifact), None)
            label = _ARTIFACT_LABEL[artifact]

            saved = ""
            if draft is not None and draft.strip() != iface.read_active_or_empty().strip():
                try:
                    iface.set_active(draft)
                except ValueError as e:
                    # The guards live in set_active, so a refusal means nothing was written --
                    # and nothing must be stamped either, or the event would name a version
                    # that does not exist. Refuse the whole decision rather than half of it.
                    return no_update, no_update, f"Not incorporated. {e}", True
                saved = f" Saved your edited {label} first."
            # Idempotent: set_active has already registered this text, so this returns the id
            # it just minted rather than creating a second row.
            if artifact == "prompt":
                vid = db_interface.get_or_create_prompt(conn, iface.read_active_or_empty())
                db_interface.add_pattern_event(conn, pid, "incorporated", prompt_id=vid)
            else:
                vid = db_interface.get_or_create_profile(conn, iface.read_active_or_empty())
                db_interface.add_pattern_event(conn, pid, "incorporated", profile_id=vid)
            msg = f"Incorporated into active {label} {vid[:12]}.{saved}"
        else:  # pat-reject
            note = _state_value(ctx.states_list[0], pid)
            db_interface.add_pattern_event(conn, pid, "rejected", note=note or None)
            msg = "Rejected -- kept as a closed pattern; the suggester will not re-propose it."
        children = _render_patterns(conn, tab or "active")
        count = _count_label(_counts(conn))
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
    State({"type": "pat-blame", "pid": ALL}, "value"),
    prevent_initial_call=True,
)
def cb_pattern_save(clicks, _names, _dirs, _descs, _suggs, _blames):
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
            blame=_state_value(ctx.states_list[4], pid),
        )
    finally:
        conn.close()
    return "Saved pattern edits (content only; the fate log is untouched).", True


# ---------------------------------------------------------------------------
# Callbacks: the judge prompt pane's harness button (the editor's own live in artifact_editor)
# ---------------------------------------------------------------------------

@callback(
    Output("test-results", "children"),
    Input("run-tests-btn", "n_clicks"),
    State(aid("editor", "prompt"), "value"),
    prevent_initial_call=True,
)
def cb_run_tests(_n, draft):
    """The judge harness on the live DRAFT against the active profile -- the in-edit-loop floor
    gate. Carried over from prompt_workbench; it is the one control that belongs to exactly one
    artifact, so it has a fixed id and reads the prompt pane's textarea directly."""
    from litcurator import judge_harness
    results, prompt_fp, profile_fp = judge_harness.run_tests(prompt_text=draft or "")
    report = judge_harness.format_report(results, prompt_fp, profile_fp)
    path = judge_harness.write_report(
        report + "\n" + judge_harness.format_rationales(results))
    return f"{report}\n\nsaved to {path}"


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
    State("artifact-tabs", "value"),
    State(aid("editor", ALL), "value"),
    prevent_initial_call=True,
)
def cb_chat_send(_n, user_text, history, artifact, drafts):
    """One chat, whose context follows the selected artifact: the profile gets a thinking
    partner, the judge prompt gets a bounded critic (see _CHAT_CONFIG). `drafts` arrives as one
    entry per mounted editor, so the live draft is picked by id rather than by position."""
    if not (user_text and user_text.strip()):
        return no_update, no_update
    draft = next((s["value"] for s in ctx.states_list[3]
                  if s["id"]["artifact"] == artifact), "")
    history = list(history or [])
    history.append({"role": "user", "content": user_text.strip()})
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    cfg = _CHAT_CONFIG[artifact]
    system = (_prompt_critic_system if artifact == "prompt" else _chat_system)(
        _interface(artifact).read_active_or_empty(), draft or "")
    resp = client.messages.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        system=system,
        messages=[{"role": m["role"], "content": m["content"]} for m in history],
    )
    history.append({"role": "assistant", "content": resp.content[0].text})
    return history, ""


@callback(
    Output("chat-context-note", "children"),
    Input("artifact-tabs", "value"),
)
def cb_chat_context_note(artifact):
    """Says which artifact the chat is reasoning about and with which model. Not cosmetic: the
    two critics give different KINDS of advice, and asking the prompt critic about topic taste
    (or the profile partner about scoring mechanics) wastes a turn."""
    cfg = _CHAT_CONFIG[artifact]
    return (f"(context: committed {_ARTIFACT_LABEL[artifact]} + live draft | "
            f"model: {cfg['model']})")


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
