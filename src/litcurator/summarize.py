"""
summarize.py -- neutral one-to-two-sentence paper summaries (Haiku).

A quick-read blurb for the review feed: what the paper IS, independent of the
user's profile, so the reader can grok it before the judge's taste-framed
rationale colors their own read. Descriptive, never evaluative.

A summary is a stable property of the PAPER, not of an evaluation: it does not
change when the profile changes or the judge re-runs. So it is generated once
over the reviewable survivors and cached on the article (db_interface.summary),
NOT emitted by the judge. Pure scoring function here; the pipeline orchestrates.
"""

import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SUMMARIZE_MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 8

COST_PER_M_INPUT = 0.80
COST_PER_M_OUTPUT = 4.00

SYSTEM_PROMPT = """
You write a very short factual summary of a scientific paper from its title and abstract, so a researcher can tell at a glance what it is about.

State what was studied and the key finding or contribution. Be concrete: name the organism, brain system, method, or result when they are central. Lead with the substance.

Rules:
- ONE or TWO sentences. Never three. Shorter is better.
- Plain declarative description of what the paper did and found.
- NO background, motivation, significance claims, hype, or hedging ("sheds light on", "remains poorly understood", "has important implications").
- Describe, do not judge. Never say whether it is interesting, novel, or important.
- ASCII only.

You will receive a numbered list of papers (title + abstract). Return ONLY a JSON array of strings, one summary per paper, in the same order. No preamble, no code fences.
""".strip()


def summarize_batch(articles):
    """Summarize a batch of articles. articles: list of dicts with 'title' and
    'abstract'. Returns (list of summary strings in input order, usage)."""
    blocks = []
    for i, a in enumerate(articles, 1):
        blocks.append(f"{i}. {a.get('title') or '(no title)'}\n   {a.get('abstract') or '(no abstract)'}")
    user_message = "\n\n".join(blocks)

    response = client.messages.create(
        model=SUMMARIZE_MODEL,
        max_tokens=max(len(articles) * 90, 1),
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    summaries = json.loads(raw)
    if not isinstance(summaries, list) or len(summaries) != len(articles):
        raise ValueError(f"expected {len(articles)} summaries, got {summaries!r:.120}")
    return [str(s).strip() for s in summaries], response.usage
