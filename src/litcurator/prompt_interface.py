"""
prompt_interface.py -- read, version, and activate the judge prompt.

The judge prompt (config.JUDGE_PROMPT_PATH, conventionally judge_prompt.md) is the
scoring procedure the judge follows -- the OTHER biconvex knob alongside the user
profile. This module is the single gatekeeper for it (the judge via the pipeline
and the prompt workbench all go through here), exactly mirroring profile_interface,
so "set active backs up the outgoing prompt first" lives in one place.

Difference from profile_interface: there is no "no prompt" state. load_active()
SEEDS the active file from judge.DEFAULT_JUDGE_PROMPT the first time, so the judge
always has a prompt and behavior is byte-identical until the user edits it.

Layout under PROMPT_DIR:
    judge_prompt.md                  the active prompt (what the judge runs)
    versions/judge_prompt_<ts>.md    timestamped snapshots ("save as new version")
    versions/_pre_active_<ts>.md     the outgoing active, backed up before each promote
    versions/_autosave.md            the workbench crash-safety draft
"""

import hashlib
from datetime import datetime
from pathlib import Path

from litcurator.config import JUDGE_PROMPT_PATH, PROMPT_DIR

VERSIONS_DIR = PROMPT_DIR / "versions"
AUTOSAVE_PATH = VERSIONS_DIR / "_autosave.md"


def _ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# The active prompt
# ---------------------------------------------------------------------------

def active_path():
    return JUDGE_PROMPT_PATH


def exists():
    return JUDGE_PROMPT_PATH.exists()


def load_active():
    """Return the active judge prompt, seeding it from the default the first time.
    The judge always has a prompt; the seed is byte-identical to the in-code default
    so nothing changes until the user edits it in the workbench."""
    if not JUDGE_PROMPT_PATH.exists():
        _seed_default()
    return JUDGE_PROMPT_PATH.read_text(encoding="utf-8", errors="replace")


def read_active_or_empty():
    """Return the active prompt text, or '' if none exists yet (for the editor)."""
    if JUDGE_PROMPT_PATH.exists():
        return JUDGE_PROMPT_PATH.read_text(encoding="utf-8", errors="replace")
    return ""


def _seed_default():
    """Write the in-code default judge prompt as the seed (root of the lineage)."""
    from litcurator.judge import DEFAULT_JUDGE_PROMPT
    set_active(DEFAULT_JUDGE_PROMPT, notes="seed: default judge prompt")


def set_active(text, notes=None):
    """Write text to the active prompt, snapshotting the outgoing active first, and
    register the new version in the DB prompts table (parent_id = SHA256 of the
    outgoing active, so the lineage stays a clean chain). Returns the backup path
    (or None if there was no prior active prompt)."""
    backup = None
    parent_id = None
    if JUDGE_PROMPT_PATH.exists():
        current = JUDGE_PROMPT_PATH.read_text(encoding="utf-8", errors="replace")
        parent_id = hashlib.sha256(current.encode("utf-8")).hexdigest()
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        backup = VERSIONS_DIR / f"_pre_active_{_ts()}.md"
        backup.write_text(current, encoding="utf-8")
    JUDGE_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_PROMPT_PATH.write_text(text, encoding="utf-8")
    # Lazy import to avoid a circular dependency at module load time.
    from litcurator import db_interface
    conn = db_interface.get_connection()
    try:
        db_interface.get_or_create_prompt(conn, text, parent_id=parent_id, notes=notes)
    finally:
        conn.close()
    return backup


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def save_version(text, ts=None):
    """Write text to a timestamped snapshot in versions/. Returns the path."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = VERSIONS_DIR / f"judge_prompt_{ts or _ts()}.md"
    path.write_text(text, encoding="utf-8")
    return path


def list_versions():
    """Saved version snapshots, newest first."""
    if not VERSIONS_DIR.exists():
        return []
    return sorted(VERSIONS_DIR.glob("judge_prompt_*.md"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def latest_version():
    versions = list_versions()
    return versions[0] if versions else None


def read_version(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Autosave (workbench crash-safety net, separate from explicit versions)
# ---------------------------------------------------------------------------

def save_autosave(text):
    """Persist the workbench draft as a crash/reload safety net. Skips empty and
    unchanged content so it never clobbers a good draft. Returns True if written."""
    text = text or ""
    if not text.strip():
        return False
    if AUTOSAVE_PATH.exists() and AUTOSAVE_PATH.read_text(encoding="utf-8") == text:
        return False
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    AUTOSAVE_PATH.write_text(text, encoding="utf-8")
    return True


def load_autosave():
    if AUTOSAVE_PATH.exists():
        return AUTOSAVE_PATH.read_text(encoding="utf-8")
    return None


# ---------------------------------------------------------------------------
# Version identity (for stamping / display)
# ---------------------------------------------------------------------------

def content_hash(text):
    """Short stable id for a prompt's content (mirrors profile_interface.content_hash
    and judge._fingerprint). Recorded on runs so you can tell which prompt version
    produced a given score."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def active_version_id():
    """content_hash of the active prompt, or None if there is none."""
    if not JUDGE_PROMPT_PATH.exists():
        return None
    return content_hash(read_active_or_empty())
