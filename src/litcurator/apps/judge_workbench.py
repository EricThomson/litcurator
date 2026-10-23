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

The human authors every word of the profile; the machinery only surfaces and
remembers. Typical flow: edit a pattern -> author the profile edit on the
right -> Set as active -> Incorporate the pattern (it stamps that version).

There was a chat panel at the bottom (a Sonnet partner for the profile, an Opus critic for the
prompt). Removed 2026-10-23: unused, and its two system prompts were instructions living in
code. A conversation about an edit happens with Claude Code, which can see the flags too.

Run:
    litcurator judge_workbench
    python src/litcurator/apps/judge_workbench.py
"""


import dash_bootstrap_components as dbc
from dash import (ALL, Dash, Input, Output, State, callback, ctx, dcc, html,
                  no_update)
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle

from litcurator import db_interface, profile_interface, prompt_interface
from litcurator.apps import artifact_editor
from litcurator.apps.artifact_editor import aid
from litcurator.config import LEVELS_BUCKET_DESCRIPTION

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
                         title="the fix belongs in the judge prompt (how every paper is "
                               "scored), not the profile; the fix: dropdown below changes it",
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
        html.Small("suggested edit -- a change to the JUDGE PROMPT (how every paper is "
                   "scored); your working draft"
                   if p.get("blame") == "prompt" else
                   "suggested edit -- a change to the PROFILE (what you want); your working draft",
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
    results, prompt_short_id, profile_short_id = judge_harness.run_tests(prompt_text=draft or "")
    report = judge_harness.format_report(results, prompt_short_id, profile_short_id)
    path = judge_harness.write_report(
        report + "\n" + judge_harness.format_rationales(results))
    return f"{report}\n\nsaved to {path}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_app(port=8053):
    # use_reloader=False: hot reload would reset the editor textarea and wipe
    # unsaved work. Restart manually to pick up code changes.
    app.run(debug=True, use_reloader=False, port=port)


if __name__ == "__main__":
    run_app()
