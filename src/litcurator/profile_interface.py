"""
profile_interface.py -- read, version, and activate the user profile.

The user profile (config.USER_PROFILE_PATH, conventionally user_profile.md) is
the only taste input to the judge. This module is the single gatekeeper for it:
the judge, the pipeline, and the workbench all go through here instead of
touching the files directly, so "set active backs up the outgoing profile first"
lives in exactly one place.

Layout under PROFILE_DIR:
    user_profile.md                  the active profile (what the judge reads)
    versions/user_profile_<ts>.md    timestamped snapshots ("save as new version")
    versions/_pre_active_<ts>.md     the outgoing active, backed up before each promote
    versions/_autosave.md            the workbench crash-safety draft
"""

import hashlib
from datetime import datetime
from pathlib import Path

from litcurator.config import PROFILE_DIR, USER_PROFILE_PATH

VERSIONS_DIR = PROFILE_DIR / "versions"
AUTOSAVE_PATH = VERSIONS_DIR / "_autosave.md"


def _ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# The active profile
# ---------------------------------------------------------------------------

def active_path():
    return USER_PROFILE_PATH


def exists():
    return USER_PROFILE_PATH.exists()


def load_active():
    """Return the active profile text. Raises FileNotFoundError if there is none
    -- the judge must never score against an empty profile."""
    if not USER_PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"No active user profile at {USER_PROFILE_PATH}. "
            f"Create one, or promote a version with set_active()."
        )
    return USER_PROFILE_PATH.read_text(encoding="utf-8", errors="replace")


def read_active_or_empty():
    """Return the active profile text, or '' if none exists (for the editor)."""
    if USER_PROFILE_PATH.exists():
        return USER_PROFILE_PATH.read_text(encoding="utf-8", errors="replace")
    return ""


def set_active(text):
    """Write text to the active profile, snapshotting the outgoing active first,
    and register the new version in the DB (parent_id = SHA256 of the outgoing
    active, so the lineage stays a clean chain). Returns the backup path (or None
    if there was no prior active profile).
    """
    backup = None
    parent_id = None
    if USER_PROFILE_PATH.exists():
        current = USER_PROFILE_PATH.read_text(encoding="utf-8", errors="replace")
        parent_id = hashlib.sha256(current.encode("utf-8")).hexdigest()
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        backup = VERSIONS_DIR / f"_pre_active_{_ts()}.md"
        backup.write_text(current, encoding="utf-8")
    USER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_PROFILE_PATH.write_text(text, encoding="utf-8")
    # Lazy import to avoid circular dependency at module load time.
    from litcurator import db_interface
    conn = db_interface.get_connection()
    try:
        db_interface.get_or_create_profile(conn, text, parent_id=parent_id)
    finally:
        conn.close()
    return backup


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def save_version(text, ts=None):
    """Write text to a timestamped snapshot in versions/. Returns the path."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = VERSIONS_DIR / f"user_profile_{ts or _ts()}.md"
    path.write_text(text, encoding="utf-8")
    return path


def list_versions():
    """Saved version snapshots, newest first."""
    if not VERSIONS_DIR.exists():
        return []
    return sorted(VERSIONS_DIR.glob("user_profile_*.md"),
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
# Version identity (for stamping judgments)
# ---------------------------------------------------------------------------

def content_hash(text):
    """Short stable id for a profile's content. Recorded on judgments so you can
    tell which profile version produced a given score."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def active_version_id():
    """content_hash of the active profile, or None if there is none."""
    if not USER_PROFILE_PATH.exists():
        return None
    return content_hash(read_active_or_empty())
