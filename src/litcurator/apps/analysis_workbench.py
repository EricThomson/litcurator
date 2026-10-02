"""
analysis_workbench.py -- edit the ANALYSIS prompt: the instructions error_analysis runs to turn
your flags into patterns. The sibling of judge_workbench, which edits the judge's two inputs;
the two share their editor (artifact_editor.py).

Two tabs, one per section of ~/.litcurator/prompt/analysis_prompt.md:
  - Cluster: step 1, proposes candidate patterns from your unattached flags.
  - Consolidate: step 2, decides what to do with each candidate.
A tab's Save version and Set as active write the WHOLE file -- that tab's text with the other
section as it is active -- so editing one tab can never blank the other.

Below them, PREVIEW: your on-screen drafts of BOTH tabs run through the real error_analysis as a
dry run over a date window. Nothing is recorded, and the report lands in suggestions/ like any
other round's. It replaces the old sandbox lab's Run panel, which ran a private copy of the
pipeline and drifted from the real one.

Run:
    litcurator analysis_workbench
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback, dcc, html

from litcurator import analysis_prompt_interface, error_analysis
from litcurator.apps import artifact_editor
from litcurator.apps.artifact_editor import aid

# How many rounds a preview runs. One, not the three a real run picks between: the preview is
# for trying wording, and a real error_analysis makes the actual decision. Raise it to preview
# exactly what a real run would do (about $0.53 a preview instead of about $0.17).
PREVIEW_ROUNDS = 1

# The panes this bench mounts, in DOM order: one per section of the analysis prompt.
ARTIFACTS = ("cluster", "consolidate")
_ARTIFACT_LABEL = {"cluster": "cluster section", "consolidate": "consolidate section"}


def _interface(artifact):
    """The Section for a tab, resolved from THIS module's globals at call time, so a free gate
    can replace analysis_workbench.analysis_prompt_interface and know nothing reaches the live
    file. See artifact_editor for why the shared editor goes through this function."""
    if artifact == "cluster":
        return analysis_prompt_interface.CLUSTER_SECTION
    return analysis_prompt_interface.CONSOLIDATE_SECTION


artifact_editor.register(_ARTIFACT_LABEL, _interface)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
app.title = "Analysis Workbench"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H4("Analysis Workbench", className="mb-0"), width="auto"),
        dbc.Col(html.Small("the instructions error_analysis runs to turn your flags into "
                           "patterns", className="text-muted"), width="auto", align="center"),
    ], align="center", className="mt-3 mb-2 g-2"),

    artifact_editor.artifact_selector(ARTIFACTS, selected="cluster"),
    *[artifact_editor.editor_pane(a, visible=(a == "cluster")) for a in ARTIFACTS],

    html.Hr(className="my-2"),

    html.Div([
        html.H6("Preview your drafts on real flags", className="mb-1"),
        html.Small(f"Runs what is in BOTH tabs through the real error_analysis as a dry run: "
                   f"nothing is recorded. {PREVIEW_ROUNDS} round(s), not the three a real run "
                   f"picks between. Needs at least {error_analysis.MIN_FLAGS} unattached flags "
                   f"in the window.", className="text-muted d-block mb-2"),
        dbc.Row([
            dbc.Col(dcc.DatePickerRange(id="preview-dates", display_format="YYYY-MM-DD",
                                        clearable=True), width="auto"),
            dbc.Col(dbc.Button("Preview", id="preview-btn", color="success", size="sm"),
                    width="auto"),
        ], className="g-2 mb-2 align-items-center"),
        dcc.Loading(html.Div(id="preview-out")),
    ], style={"padding": "0 14px"}),
], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks: the preview (the editor's own live in artifact_editor)
# ---------------------------------------------------------------------------

@callback(
    Output("preview-out", "children"),
    Input("preview-btn", "n_clicks"),
    State(aid("editor", "cluster"), "value"),
    State(aid("editor", "consolidate"), "value"),
    State("preview-dates", "start_date"),
    State("preview-dates", "end_date"),
    prevent_initial_call=True,
)
def cb_preview(_n, cluster_draft, consolidate_draft, start, end):
    """The drafts on screen, through suggest_edits itself -- the one pipeline, so a preview
    cannot drift from a real run. suggest_edits refuses to record a draft."""
    try:
        analysis_prompt_interface.CLUSTER_SECTION.validate(cluster_draft)
        analysis_prompt_interface.CONSOLIDATE_SECTION.validate(consolidate_draft)
    except ValueError as e:
        return dbc.Alert(f"Not previewed. {e}", color="danger")
    draft = analysis_prompt_interface.compose(cluster_draft, consolidate_draft)
    try:
        round_dir = error_analysis.suggest_edits(
            start=start or None, end=end or None, persist=False, best_of=PREVIEW_ROUNDS,
            analysis_prompt_text=draft)
    except Exception as e:   # surface anything in the page rather than only the terminal
        return dbc.Alert(f"{type(e).__name__}: {e}", color="danger")
    if round_dir is None:
        return dbc.Alert(f"Fewer than {error_analysis.MIN_FLAGS} unattached flags in that "
                         f"window, so there is nothing to preview yet.", color="warning")
    return [dbc.Alert(f"Dry run, nothing recorded. Saved in {round_dir}", color="info",
                      className="py-2"),
            *[dcc.Markdown(report.read_text(encoding="utf-8"))
              for report in sorted(round_dir.glob("run_*.md"))]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_app(port=8057):
    # use_reloader=False: hot reload would reset the editor boxes and wipe unsaved work.
    app.run(debug=True, use_reloader=False, port=port)


if __name__ == "__main__":
    run_app()
