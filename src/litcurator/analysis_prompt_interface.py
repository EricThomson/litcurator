"""
analysis_prompt_interface.py -- read, version, and activate the ANALYSIS prompt.

The analysis prompt is how flags become patterns: the CLUSTER step that proposes candidate
patterns from unattached flags, and the CONSOLIDATE step that decides what to do with each
candidate. It is a hand-authored artifact you tune, exactly like the profile and the judge
prompt, and it is versioned the same way -- this module mirrors prompt_interface function for
function.

WHY IT IS ONE FILE AND NOT TWO. Cluster and consolidate are two halves of one pipeline stage:
they are always edited together in the same bench, for the same purpose, and neither is
meaningful without the other. Versioning them separately would double the artifact count, the
version chains and the things to keep straight, to record a distinction ("which half changed")
that a diff answers anyway. So the artifact is one file with two marked sections.

That costs nothing operationally. The cluster cache keys on the CLUSTER SECTION's text, not on
the file, so editing consolidate does not invalidate it. The harness can still override one
section and hold the other fixed. And the analysis workbench edits one section per tab through
Section (below), which always writes the WHOLE file, so one tab can never blank the other.

Layout under PROMPT_DIR (shared with the judge prompt -- one folder for all prompts, and
version filenames are prefixed by artifact so they cannot collide):
    analysis_prompt.md                  the active prompt (what error_analysis runs)
    versions/analysis_prompt_<ts>.md    timestamped snapshots ("save as new version")
    versions/_pre_active_analysis_<ts>.md   the outgoing active, backed up before each promote
    versions/_autosave_analysis_cluster.md, _autosave_analysis_consolidate.md
                                        the workbench's crash-safety drafts, one per tab

Like the profile and the judge prompt, litcurator ships NO content for this: load_active()
raises if you have not authored one. Writing a good minimal default for a new user is a real
task and is on the long-term list -- a mediocre one would silently shape every pattern this
machinery ever produces.
"""

import hashlib
from datetime import datetime
from pathlib import Path

from litcurator.config import ANALYSIS_PROMPT_PATH, PROMPT_DIR

VERSIONS_DIR = PROMPT_DIR / "versions"

# HTML comments rather than markdown headings, deliberately. The judge prompt splits on the
# heading "## Output", which works but is a latent hazard: write that heading in your prose and
# you silently truncate the prompt. A comment marker cannot collide with anything you would
# write, and it stays invisible when the file is rendered.
CLUSTER_MARKER = "<!-- CLUSTER -->"
CONSOLIDATE_MARKER = "<!-- CONSOLIDATE -->"

_TEMPLATE = """# Analysis prompts

How flags become patterns. Two steps, two sections. Edit the prose; leave the marker lines
alone -- they are what splits this file into the two prompts.

{cluster_marker}

{cluster}

{consolidate_marker}

{consolidate}
"""


def _ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# The two sections
# ---------------------------------------------------------------------------

def compose(cluster_text, consolidate_text):
    """Build the file's text from the two prompts."""
    return _TEMPLATE.format(cluster_marker=CLUSTER_MARKER, consolidate_marker=CONSOLIDATE_MARKER,
                            cluster=cluster_text.strip(), consolidate=consolidate_text.strip())


def _sections(text):
    """(cluster_text, consolidate_text), either of which may be empty. Raises ValueError only
    when a marker is missing -- the lenient reader that saved versions are loaded with."""
    if CLUSTER_MARKER not in text:
        raise ValueError(f"analysis prompt is missing the {CLUSTER_MARKER} marker")
    if CONSOLIDATE_MARKER not in text:
        raise ValueError(f"analysis prompt is missing the {CONSOLIDATE_MARKER} marker")
    cluster = text.split(CLUSTER_MARKER, 1)[1]
    cluster, consolidate = cluster.split(CONSOLIDATE_MARKER, 1)
    return cluster.strip(), consolidate.strip()


def split(text):
    """Split the artifact into (cluster_text, consolidate_text).

    Raises ValueError if either marker is missing or either section is empty. That is
    deliberate and it is why the harness checks it: a mistyped marker would otherwise send an
    EMPTY prompt to the model, which surfaces as mysteriously bad clustering rather than as an
    error, and would be very slow to diagnose."""
    cluster, consolidate = _sections(text)
    if not cluster:
        raise ValueError("analysis prompt has an EMPTY cluster section")
    if not consolidate:
        raise ValueError("analysis prompt has an EMPTY consolidate section")
    return cluster, consolidate


# ---------------------------------------------------------------------------
# The active prompt
# ---------------------------------------------------------------------------

def active_path():
    return ANALYSIS_PROMPT_PATH


def exists():
    return ANALYSIS_PROMPT_PATH.exists()


def load_active():
    """Return the active analysis prompt as raw text. Raises FileNotFoundError if there is
    none -- litcurator ships no default."""
    if not ANALYSIS_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"No active analysis prompt at {ANALYSIS_PROMPT_PATH}. litcurator ships no default "
            f"-- author one (cluster + consolidate, two marked sections) at that path, then "
            f"edit it in `litcurator analysis_workbench`.")
    return ANALYSIS_PROMPT_PATH.read_text(encoding="utf-8", errors="replace")


def load_active_sections():
    """The active prompt split into (cluster_text, consolidate_text) -- what the two steps
    actually run."""
    return split(load_active())


def read_active_or_empty():
    """Return the active prompt text, or '' if none exists yet (for the editor)."""
    if ANALYSIS_PROMPT_PATH.exists():
        return ANALYSIS_PROMPT_PATH.read_text(encoding="utf-8", errors="replace")
    return ""


def validate(text):
    """Raise ValueError if `text` could not serve as the analysis prompt (a marker missing or a
    section empty). Separable from set_active so a caller can check without writing, mirroring
    prompt_interface.validate."""
    split(text)


def set_active(text, notes=None):
    """Write text to the active prompt, snapshotting the outgoing active first, and register
    the new version in the DB prompts table with kind='analysis' (parent_id = SHA256 of the
    outgoing active, so the lineage stays a clean chain). Returns the backup path (or None if
    there was no prior active prompt).

    Validates that the text splits into two non-empty sections BEFORE writing anything, so a
    broken marker can never become the active prompt."""
    validate(text)
    backup = None
    parent_id = None
    current = None
    if ANALYSIS_PROMPT_PATH.exists():
        current = ANALYSIS_PROMPT_PATH.read_text(encoding="utf-8", errors="replace")
        parent_id = hashlib.sha256(current.encode("utf-8")).hexdigest()
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        backup = VERSIONS_DIR / f"_pre_active_analysis_{_ts()}.md"
        backup.write_text(current, encoding="utf-8")
    ANALYSIS_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PROMPT_PATH.write_text(text, encoding="utf-8")
    # Lazy import to avoid a circular dependency at module load time.
    from litcurator import db_interface
    conn = db_interface.get_connection()
    try:
        # Outgoing first -- see prompt_interface.set_active for why. The analysis chain
        # carries the same restore wound (eb994c4c702d is an orphan root).
        if current is not None:
            db_interface.get_or_create_prompt(
                conn, current, notes="re-registered during a later promote",
                kind="analysis")
        db_interface.get_or_create_prompt(conn, text, parent_id=parent_id, notes=notes,
                                          kind="analysis")
    finally:
        conn.close()
    return backup


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def save_version(text, ts=None):
    """Write text to a timestamped snapshot in versions/. Returns the path."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = VERSIONS_DIR / f"analysis_prompt_{ts or _ts()}.md"
    path.write_text(text, encoding="utf-8")
    return path


def list_versions():
    """Saved version snapshots, newest first."""
    if not VERSIONS_DIR.exists():
        return []
    return sorted(VERSIONS_DIR.glob("analysis_prompt_*.md"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def latest_version():
    versions = list_versions()
    return versions[0] if versions else None


def read_version(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Version identity (for stamping / display)
# ---------------------------------------------------------------------------

def content_hash(text):
    """Short display id: the 12-char prefix of SHA256, lining up with the DB content-address
    (prompts.id = SHA256(content)). Display only; the load-bearing provenance is the full
    sha256 analysis_prompt_id on each analysis_run."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def active_version_id():
    """content_hash of the active prompt, or None if there is none."""
    if not ANALYSIS_PROMPT_PATH.exists():
        return None
    return content_hash(read_active_or_empty())


# ---------------------------------------------------------------------------
# One section at a time (the analysis workbench's two tabs)
# ---------------------------------------------------------------------------

class Section:
    """One section of the analysis prompt, offering the functions the shared artifact editor
    calls on profile_interface and prompt_interface, so each workbench tab is edited exactly
    like a whole artifact.

    Every write is of the WHOLE file: this section's text composed with the OTHER section as it
    is ACTIVE now. So one tab can never blank or alter the other, and every version on disk is a
    complete prompt error_analysis could run. Each method composes or extracts, then calls the
    module-level function of the same name."""

    def __init__(self, name):
        assert name in ("cluster", "consolidate"), name
        self.name = name

    def _of(self, whole_text):
        cluster, consolidate = _sections(whole_text)
        return cluster if self.name == "cluster" else consolidate

    def _whole_file(self, text):
        """This section's text, composed with the other section as it is active now."""
        cluster, consolidate = _sections(load_active()) if exists() else ("", "")
        if self.name == "cluster":
            return compose(text, consolidate)
        return compose(cluster, text)

    def active_path(self):
        return active_path()

    def read_active_or_empty(self):
        """This section of the active file, or '' before one exists. A file with a broken
        marker RAISES rather than showing an empty tab, which would invite overwriting it."""
        return self._of(load_active()) if exists() else ""

    def validate(self, text):
        if not (text or "").strip():
            raise ValueError(f"Refused: an empty {self.name} section would send the model no "
                             f"instructions at all. Nothing was written.")
        for marker in (CLUSTER_MARKER, CONSOLIDATE_MARKER):
            if marker in text:
                raise ValueError(f"Refused: the {self.name} tab contains the line {marker}. Paste "
                                 f"only this section's prose; the markers belong to the file, and "
                                 f"a second one would split it in the wrong place. Nothing was "
                                 f"written.")

    def set_active(self, text, notes=None):
        self.validate(text)
        return set_active(self._whole_file(text), notes)

    def save_version(self, text, ts=None):
        return save_version(self._whole_file(text), ts)

    def list_versions(self):
        return list_versions()

    def latest_version(self):
        return latest_version()

    def read_version(self, path):
        """This section of a saved version."""
        return self._of(read_version(path))

    def _autosave_path(self):
        # Computed per call, not stored, so a gate that repoints VERSIONS_DIR is honoured.
        return VERSIONS_DIR / f"_autosave_analysis_{self.name}.md"

    def save_autosave(self, text):
        """Persist the tab's draft as a crash/reload safety net. Skips empty and unchanged
        content so it never clobbers a good draft. Returns True if written."""
        text = text or ""
        path = self._autosave_path()
        if not text.strip() or (path.exists() and path.read_text(encoding="utf-8") == text):
            return False
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return True

    def load_autosave(self):
        path = self._autosave_path()
        return path.read_text(encoding="utf-8") if path.exists() else None


CLUSTER_SECTION = Section("cluster")
CONSOLIDATE_SECTION = Section("consolidate")
