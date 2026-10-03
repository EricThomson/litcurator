"""
domain_filter.py -- stage 1, the Haiku title gate.

Scores each retrieved title for "is this systems / behavioral neuroscience?", a quick, cheap
pre-screen: a score in [0, 1], and titles at or above DOMAIN_THRESHOLD go on to the judge.
Validated at about 90% recall in v1. The prompt is config.DOMAIN_FILTER_PROMPT_TITLE;
pipeline.py runs the batches and records the scores.
"""

import json

import anthropic

from litcurator.config import DOMAIN_FILTER_MODEL, DOMAIN_FILTER_PROMPT_TITLE

client = anthropic.Anthropic()

BATCH_SIZE_TITLE = 20


def score_domain_batch(articles):
    """Score a batch of titles. `articles` are dicts (or sqlite3.Row) with a 'title'.
    Returns ([(score, reasoning), ...] in input order, response.usage)."""
    lines = [f"{i + 1}. {row['title']}" for i, row in enumerate(articles)]
    system = (
        DOMAIN_FILTER_PROMPT_TITLE
        + "\n\nYou will receive a numbered list of titles. Return a JSON array "
          "where each element has 'score' and 'reasoning', in the same order "
          "as the input."
    )
    response = client.messages.create(
        model=DOMAIN_FILTER_MODEL,
        max_tokens=max(len(articles) * 120, 1),
        system=system,
        messages=[{"role": "user", "content": "\n".join(lines)}],
    )
    raw = _strip_code_fence(response.content[0].text.strip())
    results = json.loads(raw)
    return [(float(r["score"]), r["reasoning"]) for r in results], response.usage


def _strip_code_fence(raw):
    """Strip a leading markdown code fence if the model wrapped its JSON."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw
