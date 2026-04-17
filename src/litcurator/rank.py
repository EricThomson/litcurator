"""
LLM-based ranking of initial PubMed candidates against user interests.

Stage 1 (domain_filter): Coarse filter — is this paper in the right domain?
    Uses Haiku + truncated abstract (250 chars) + batching for speed and low cost.

Stage 2 (curation_rank): Fine-grained filter — does this match the curator's interests?
    Uses Sonnet + full abstract for nuanced scoring.
"""

import json
import os
import time

import anthropic
from dotenv import load_dotenv

from litcurator import db
from litcurator.config import DOMAIN_FILTER_PROMPT_TITLE, DOMAIN_FILTER_PROMPT_ABSTRACT

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DOMAIN_FILTER_MODEL = "claude-haiku-4-5-20251001"
CURATION_MODEL = "claude-sonnet-4-6"
BATCH_SIZE_TITLE = 20
BATCH_SIZE_FULL = 10
ABSTRACT_EXCERPT_LEN = 250

MODEL_COSTS = {
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
}


def domain_filter(conn, threshold=0.5, journal=None, model=None, mode="title"):
    """
    Run Stage 1 domain filter on status=1 (retrieved) articles.

    Args:
        conn: SQLite connection
        threshold: Minimum domain_score to advance to status=2 (default 0.5)
        journal: Optional journal name string to filter articles (default None = all)
        model: Model ID override (default: DOMAIN_FILTER_MODEL)
        mode: "title" (title only) or "full" (title + full abstract)

    Returns:
        Tuple of (total scored, passed threshold)
    """
    use_model = model or DOMAIN_FILTER_MODEL
    batch_size = BATCH_SIZE_FULL if mode == "full" else BATCH_SIZE_TITLE
    articles = db.get_articles_by_status(conn, status=1)

    if journal is not None:
        articles = [a for a in articles if a["journal"] == journal]

    total = len(articles)
    passed = 0
    total_input_tokens = 0
    total_output_tokens = 0
    t_start = time.time()

    for batch_start in range(0, total, batch_size):
        batch = articles[batch_start: batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} articles)...")

        try:
            scores, usage = _score_domain_batch(batch, model=use_model, mode=mode)
        except Exception as e:
            print(f"  Batch {batch_num} failed ({e}), retrying...")
            time.sleep(2)
            scores, usage = _score_domain_batch(batch, model=use_model, mode=mode)

        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens

        for row, (score, reasoning) in zip(batch, scores):
            db.update_domain_score(conn, row["pmid"], score, reasoning)
            if score >= threshold:
                passed += 1

    elapsed = time.time() - t_start
    costs = MODEL_COSTS.get(use_model, {"input": 0, "output": 0})
    total_cost = (total_input_tokens * costs["input"] + total_output_tokens * costs["output"]) / 1_000_000

    print(f"\nDomain filter complete: {passed}/{total} passed threshold {threshold}")
    print(f"Time: {elapsed:.1f}s | Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out | Cost: ${total_cost:.4f}")
    return total, passed


def _score_domain_batch(articles, model=None, mode="title"):
    """
    Score a batch of articles.

    Args:
        articles: List of sqlite3.Row objects with at least 'pmid', 'title', 'abstract'
        model: Model ID override (default: DOMAIN_FILTER_MODEL)
        mode: "title" (title only) or "full" (title + full abstract)

    Returns:
        List of (score, reasoning) tuples in the same order as input articles.
    """
    use_model = model or DOMAIN_FILTER_MODEL

    if mode == "full":
        lines = []
        for i, row in enumerate(articles):
            lines.append(f"{i+1}. {row['title']}\n   {row['abstract'] or ''}")
        user_message = "\n\n".join(lines)
        system = DOMAIN_FILTER_PROMPT_ABSTRACT + "\n\nYou will receive a numbered list of articles (title + abstract). Return a JSON array where each element has 'score' and 'reasoning', in the same order as the input."
    else:
        lines = [f"{i+1}. {row['title']}" for i, row in enumerate(articles)]
        user_message = "\n".join(lines)
        system = DOMAIN_FILTER_PROMPT_TITLE + "\n\nYou will receive a numbered list of titles. Return a JSON array where each element has 'score' and 'reasoning', in the same order as the input."

    response = client.messages.create(
        model=use_model,
        max_tokens=max(len(articles) * 120, 1),
        system=system,
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

    print(f"  DEBUG: stop_reason={response.stop_reason}, usage={response.usage}, raw={raw[:200]!r}")
    results = json.loads(raw)
    return [(float(r["score"]), r["reasoning"]) for r in results], response.usage


def score_domain(title, prompt=None):
    """
    Score a single article title or abstract for domain relevance.
    Useful for testing individual articles.

    Args:
        title: Article title (or title + abstract) string
        prompt: Optional system prompt override (default: DOMAIN_FILTER_PROMPT)

    Returns:
        Tuple of (score: float, reasoning: str)
    """
    response = client.messages.create(
        model=DOMAIN_FILTER_MODEL,
        max_tokens=128,
        system=prompt if prompt is not None else DOMAIN_FILTER_PROMPT_TITLE,
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
