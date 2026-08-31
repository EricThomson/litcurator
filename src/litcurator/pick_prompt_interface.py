"""
pick_prompt_interface.py -- read the PICK prompt, the fourth hand-authored artifact.

Thinner than the other three prompt interfaces because there is no pick editor: no
set_active, no autosave, no version list. Version history happens on USE instead --
snapshot_active archives the file by content hash when the picker runs, which a hand-edited
file cannot bypass the way a Save hook can.

Not registered in the prompts table: nothing JOINs back to it. content_hash is sha256[:12],
the same address the DB uses, so the hash stamped in a verdict is the id the row would have
had if that ever changes.
"""

import hashlib

from litcurator.config import PROMPT_DIR

PICK_PROMPT_PATH = PROMPT_DIR / "pick_prompt.md"
VERSIONS_DIR = PROMPT_DIR / "versions"


def active_path():
    return PICK_PROMPT_PATH


def load_active():
    """The active pick prompt. Raises if there is none -- litcurator ships no default."""
    if not PICK_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"No pick prompt at {PICK_PROMPT_PATH}. litcurator ships no default -- author one "
            f"(what makes one round of patterns better than another) and save it there.")
    return PICK_PROMPT_PATH.read_text(encoding="utf-8", errors="replace")


def content_hash(text):
    """Short id: sha256[:12], matching the DB content-address. Use this, never a fresh
    hashlib call -- mixed hash schemes are a bug this project has already had once."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def snapshot_active(text):
    """Archive this prompt under versions/, keyed by content hash. No-ops if unchanged."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = VERSIONS_DIR / f"pick_prompt_{content_hash(text)}.md"
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return path
