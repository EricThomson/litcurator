"""
content_address.py -- the one definition of a text's id.

Profiles and prompts are content-addressed: a version's id is the sha256 of its text, so the
same text always gets the same id. profiles.id, prompts.id, every parent_id and every run stamp
are this. The short id, its first 12 characters, is what reports and screens show. Use these,
never a fresh hashlib call: mixed hash schemes are a bug this project has already had once.
"""

import hashlib


def content_address(text):
    """The sha256 of the text: the id a profile or prompt version is stored under."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_id(text):
    """The first 12 characters of content_address, for display."""
    return content_address(text)[:12]
