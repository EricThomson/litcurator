"""
LLM-based ranking of initial PubMed candidates against user interests.

Stage 1 (domain_filter): Coarse filter — is this paper in the right domain?
    Uses Haiku + title-only + batching for speed and low cost.

Stage 2 (curation_rank): Fine-grained filter — does this match the curator's interests?
    Uses Sonnet + full abstract for nuanced scoring.
"""

import json
import os

import anthropic
from dotenv import load_dotenv

from litcurator import db
from litcurator.config import DOMAIN_FILTER_PROMPT

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DOMAIN_FILTER_MODEL = "claude-haiku-4-5-20251001"
CURATION_MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 10


def domain_filter(conn, threshold=0.5, journal=None):
    """
    Run Stage 1 domain filter on status=1 (retrieved) articles.

    Uses Haiku + title-only + batching for speed and low cost. Stores
    domain_score and domain_reasoning in the database.

    Args:
        conn: SQLite connection
        threshold: Minimum domain_score to advance to status=2 (default 0.5)
        journal: Optional journal name string to filter articles (default None = all)

    Returns:
        Tuple of (total scored, passed threshold)
    """
    articles = db.get_articles_by_status(conn, status=1)

    if journal is not None:
        articles = [a for a in articles if a["journal"] == journal]

    total = len(articles)
    passed = 0

    # Process in batches
    for batch_start in range(0, total, BATCH_SIZE):
        batch = articles[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} articles)...")

        scores = _score_domain_batch(batch)

        for row, (score, reasoning) in zip(batch, scores):
            db.update_domain_score(conn, row["pmid"], score, reasoning)
            if score >= threshold:
                passed += 1

    print(f"\nDomain filter complete: {passed}/{total} passed threshold {threshold}")
    return total, passed


def _score_domain_batch(articles):
    """
    Score a batch of articles using title-only input.

    Args:
        articles: List of sqlite3.Row objects with at least 'pmid' and 'title'

    Returns:
        List of (score, reasoning) tuples in the same order as input articles.
    """
    lines = [f"{i+1}. {row['title']}" for i, row in enumerate(articles)]
    user_message = "\n".join(lines)

    response = client.messages.create(
        model=DOMAIN_FILTER_MODEL,
        max_tokens=BATCH_SIZE * 120,
        system=DOMAIN_FILTER_PROMPT + "\n\nYou will receive a numbered list of titles. Return a JSON array where each element has 'score' and 'reasoning', in the same order as the input.",
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    results = json.loads(raw)
    return [(float(r["score"]), r["reasoning"]) for r in results]


def score_domain(title):
    """
    Score a single article title for domain relevance.
    Useful for testing individual articles.

    Args:
        title: Article title string

    Returns:
        Tuple of (score: float, reasoning: str)
    """
    response = client.messages.create(
        model=DOMAIN_FILTER_MODEL,
        max_tokens=128,
        system=DOMAIN_FILTER_PROMPT,
        messages=[
            {"role": "user", "content": title}
        ]
    )

    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)
    return float(result["score"]), result["reasoning"]
