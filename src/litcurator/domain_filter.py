"""
Stage 1 of the v2 pipeline: coarse domain relevance filter.

Uses Haiku to score titles (or titles + abstracts) for "is this systems /
behavioral neuroscience?" -- a quick, cheap pre-screen. Scores in [0, 1];
articles >= threshold advance to the rules-engine scoring stage.

Validated at ~90%+ recall against ground truth in v1. Same model, same
prompts, same scoring math as v1; preserved across the v2 pivot deliberately.

This module contains only the pure scoring functions (Haiku call + JSON
parse). The DB-orchestrated entry point that reads articles from the v2
schema, runs a batch, and writes evaluations + scoring_run rows back is
deferred until the v2 storage schema for evaluations is finalized.
"""

import json
import os

import anthropic
from dotenv import load_dotenv

from litcurator.config import (
    DOMAIN_FILTER_PROMPT_ABSTRACT,
    DOMAIN_FILTER_PROMPT_TITLE,
)

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DOMAIN_FILTER_MODEL = "claude-haiku-4-5-20251001"
DOMAIN_FILTER_PROMPT_VERSION = "v1.0"   # stamped on domain runs; bump when the title prompt changes
BATCH_SIZE_TITLE = 20
BATCH_SIZE_FULL = 10

# Per-million-token costs for the filter model, for cost accounting when
# wiring this into a pipeline. Update when model pricing changes.
DOMAIN_FILTER_COST_PER_M_INPUT = 0.80
DOMAIN_FILTER_COST_PER_M_OUTPUT = 4.00


def score_domain_batch(articles, model=None, mode="title"):
    """
    Score a batch of articles for domain relevance.

    Sends a batch to the domain filter LLM and returns a (score, reasoning)
    tuple per article, in the input order.

    Args:
        articles: list of dicts (or sqlite3.Row) each with at least 'title'
                  and, for mode="full", 'abstract'.
        model:    model ID override (default: DOMAIN_FILTER_MODEL).
        mode:     "title" (title only) or "full" (title + abstract).

    Returns:
        Tuple of (list of (score, reasoning) tuples, response.usage object).
    """
    use_model = model or DOMAIN_FILTER_MODEL

    if mode == "full":
        lines = []
        for i, row in enumerate(articles):
            lines.append(f"{i + 1}. {row['title']}\n   {row['abstract'] or ''}")
        user_message = "\n\n".join(lines)
        system = (
            DOMAIN_FILTER_PROMPT_ABSTRACT
            + "\n\nYou will receive a numbered list of articles (title + abstract). "
              "Return a JSON array where each element has 'score' and 'reasoning', "
              "in the same order as the input."
        )
    else:
        lines = [f"{i + 1}. {row['title']}" for i, row in enumerate(articles)]
        user_message = "\n".join(lines)
        system = (
            DOMAIN_FILTER_PROMPT_TITLE
            + "\n\nYou will receive a numbered list of titles. Return a JSON array "
              "where each element has 'score' and 'reasoning', in the same order "
              "as the input."
        )

    response = client.messages.create(
        model=use_model,
        max_tokens=max(len(articles) * 120, 1),
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = _strip_code_fence(response.content[0].text.strip())
    results = json.loads(raw)
    return [(float(r["score"]), r["reasoning"]) for r in results], response.usage


def score_domain(title, prompt=None):
    """
    Score a single article title for domain relevance.

    Convenience wrapper for testing individual titles without running the
    full batch pipeline.

    Args:
        title:  article title string (or title + abstract).
        prompt: optional system prompt override (default:
                DOMAIN_FILTER_PROMPT_TITLE).

    Returns:
        Tuple of (score: float, reasoning: str).
    """
    response = client.messages.create(
        model=DOMAIN_FILTER_MODEL,
        max_tokens=128,
        system=prompt if prompt is not None else DOMAIN_FILTER_PROMPT_TITLE,
        messages=[{"role": "user", "content": title}],
    )
    raw = _strip_code_fence(response.content[0].text.strip())
    result = json.loads(raw)
    return float(result["score"]), result["reasoning"]


def _strip_code_fence(raw):
    """Strip a leading markdown code fence if the model wrapped its JSON."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw
