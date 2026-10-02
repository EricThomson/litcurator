"""
analysis_workbench.py -- edit the ANALYSIS prompt: the instructions error_analysis runs to turn
your flags into patterns. The sibling of judge_workbench, which edits the judge's two inputs;
the two share their editor (artifact_editor.py).

Two tabs, one per section of ~/.litcurator/prompt/analysis_prompt.md:
  - Cluster: step 1, proposes candidate patterns from your unattached flags.
  - Consolidate: step 2, decides what to do with each candidate.
A tab's Save version and Set as active write the WHOLE file -- that tab's text with the other
section as it is active -- so editing one tab can never blank the other.

Run:
    litcurator analysis_workbench
"""

import dash_bootstrap_components as dbc
from dash import Dash, html

from litcurator import analysis_prompt_interface
from litcurator.apps import artifact_editor

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
# App (every callback it needs lives in artifact_editor)
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
], fluid=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_app(port=8057):
    # use_reloader=False: hot reload would reset the editor boxes and wipe unsaved work.
    app.run(debug=True, use_reloader=False, port=port)


if __name__ == "__main__":
    run_app()
