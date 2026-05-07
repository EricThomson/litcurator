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
from litcurator.config import (
    DOMAIN_FILTER_PROMPT_TITLE, DOMAIN_FILTER_PROMPT_ABSTRACT,
    CURATION_PROMPT, LLM_SCORE_THRESHOLD, JOURNAL_SCORE_ADJUSTMENTS, PROFILE_PATH,
)

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DOMAIN_FILTER_MODEL = "claude-haiku-4-5-20251001"
CURATION_MODEL = "claude-sonnet-4-6"
BATCH_SIZE_TITLE = 20
BATCH_SIZE_FULL = 10
CURATION_BATCH_SIZE = 5
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
    prompt_text = DOMAIN_FILTER_PROMPT_ABSTRACT if mode == "full" else DOMAIN_FILTER_PROMPT_TITLE
    articles = db.get_articles_by_status(conn, status=1)

    if journal is not None:
        articles = [a for a in articles if a["journal"] == journal]

    total = len(articles)
    passed = 0
    total_input_tokens = 0
    total_output_tokens = 0
    t_start = time.time()

    prompt_id = db.get_or_create_prompt(conn, prompt_text)
    run_id = db.create_scoring_run(conn, stage="domain", model=use_model, prompt_id=prompt_id, threshold=threshold)

    try:
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
                db.insert_evaluation(conn, row["pmid"], run_id, score, reasoning)
                if score >= threshold:
                    passed += 1
    except Exception:
        db.fail_scoring_run(conn, run_id)
        raise

    db.complete_scoring_run(conn, run_id)

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


def curation_rank(conn, profile_path=None, model=None, date_start=None, date_end=None):
    """
    Run Stage 2 curation scoring on relevant articles.

    Scores each article 0-5 against the user profile, then applies journal
    adjustments as a postprocessing step. Results saved to curation_score,
    curation_confidence, curation_rationale columns.

    Args:
        conn: SQLite connection
        profile_path: Path to profile markdown file (default: PROFILE_PATH from config)
        model: Model ID override (default: CURATION_MODEL)
        date_start: Optional 'YYYY-MM-DD' string to filter articles by pub_date
        date_end: Optional 'YYYY-MM-DD' string to filter articles by pub_date

    Returns:
        Tuple of (total scored, above threshold count)
    """
    use_model = model or CURATION_MODEL
    use_profile = profile_path or PROFILE_PATH

    if not use_profile.exists():
        seed = use_profile.parent / "seed_profile.md"
        if seed.exists():
            import shutil
            shutil.copy(seed, use_profile)
            print(f"  No profile.md found — copied seed_profile.md to {use_profile}")
        else:
            raise FileNotFoundError(f"Profile not found: {use_profile} (and no seed_profile.md to bootstrap from)")
    profile_text = use_profile.read_text()

    # Query relevant articles with optional date range filter.
    # TODO: for real monthly use without human labels, query by domain_score >= threshold.
    if date_start and date_end:
        articles = conn.execute(
            "SELECT * FROM articles WHERE relevant = 1 AND pub_date >= ? AND pub_date <= ? ORDER BY pub_date",
            (date_start, date_end)
        ).fetchall()
    else:
        articles = conn.execute(
            "SELECT * FROM articles WHERE relevant = 1 ORDER BY pub_date"
        ).fetchall()

    total = len(articles)
    above_threshold = 0
    total_input_tokens = 0
    total_output_tokens = 0
    t_start = time.time()

    profile_id = db.get_or_create_profile(conn, profile_text)
    prompt_id = db.get_or_create_prompt(conn, CURATION_PROMPT)
    run_id = db.create_scoring_run(
        conn, stage="curation", model=use_model,
        prompt_id=prompt_id, profile_id=profile_id,
        date_start=date_start, date_end=date_end,
        threshold=LLM_SCORE_THRESHOLD,
    )

    try:
        for batch_start in range(0, total, CURATION_BATCH_SIZE):
            batch = articles[batch_start: batch_start + CURATION_BATCH_SIZE]
            batch_num = batch_start // CURATION_BATCH_SIZE + 1
            total_batches = (total + CURATION_BATCH_SIZE - 1) // CURATION_BATCH_SIZE
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} articles)...")

            try:
                results, usage = score_curation_batch(batch, profile_text, use_model)
            except Exception as e:
                print(f"  Batch {batch_num} failed ({e}), retrying...")
                time.sleep(2)
                results, usage = score_curation_batch(batch, profile_text, use_model)

            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens

            adjusted_scores = []
            for row, (score, rationale) in zip(batch, results):
                adjustment = JOURNAL_SCORE_ADJUSTMENTS.get(row["journal"], 0.0)
                adjusted_score = min(1.0, max(0.0, score + adjustment))
                db.insert_evaluation(conn, row["pmid"], run_id, adjusted_score, rationale)
                if adjusted_score >= LLM_SCORE_THRESHOLD:
                    above_threshold += 1
                adjusted_scores.append(adjusted_score)

            scores_str = ", ".join(f"{s:.2f}" for s in adjusted_scores)
            print(f"    scores: {scores_str}")
    except Exception:
        db.fail_scoring_run(conn, run_id)
        raise

    db.complete_scoring_run(conn, run_id)

    elapsed = time.time() - t_start
    costs = MODEL_COSTS.get(use_model, {"input": 0, "output": 0})
    total_cost = (total_input_tokens * costs["input"] + total_output_tokens * costs["output"]) / 1_000_000

    print(f"\nCuration scoring complete: {above_threshold}/{total} above threshold (score >= {LLM_SCORE_THRESHOLD})")
    print(f"Time: {elapsed:.1f}s | Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out | Cost: ${total_cost:.4f}")
    return total, above_threshold


def score_curation_batch(articles, profile_text, model):
    """
    Score a batch of articles against a user profile.

    Args:
        articles: List of sqlite3.Row objects with title, abstract, journal
        profile_text: User interest profile as a string
        model: Model ID string

    Returns:
        List of (score, rationale) tuples, and usage object.
    """
    lines = []
    for i, row in enumerate(articles):
        abstract = (row["abstract"] or "").strip() or "(no abstract)"
        lines.append(f"{i+1}. Title: {row['title']}\n   Abstract: {abstract}")

    system = f"{CURATION_PROMPT}\n\n## User Interest Profile\n\n{profile_text}"
    user_message = "## Articles to Score\n\n" + "\n\n".join(lines)

    response = client.messages.create(
        model=model,
        max_tokens=max(len(articles) * 300, 1024),
        system=system,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()
    print(f"  DEBUG: stop_reason={response.stop_reason}, usage={response.usage}, raw={raw[:200]!r}")

    if response.stop_reason == "max_tokens":
        raise ValueError(f"Response truncated (max_tokens); got stop_reason=max_tokens with {response.usage.output_tokens} output tokens")

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    if not raw:
        raise ValueError("Empty response after stripping code fences")

    results = json.loads(raw)
    return [(float(r["score"]), r["rationale"]) for r in results], response.usage


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
