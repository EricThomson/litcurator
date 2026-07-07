"""
judge.py -- the curation judge (v4, profile-only).

The judge reads a paper's title, abstract, and journal and estimates the
user's interest given their user profile alone. No card intermediary, no
precedents.
"""

import hashlib
import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"


def _fingerprint(text):
    """Short content id of a prompt, stamped on each judgment so the judgment
    self-identifies its prompt (mirrors profile_interface.content_hash)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

COST_PER_M_INPUT = 3.0
COST_PER_M_OUTPUT = 15.0

VALID_SURFACE_DECISION = {"surface", "maybe", "do_not_surface"}


# ---------------------------------------------------------------------------
# Primary prompts -- article (title + abstract) mode, profile-only
# ---------------------------------------------------------------------------
# DEFAULT_JUDGE_PROMPT is the SEED: prompt_interface seeds the on-disk active
# prompt from it, so behavior is byte-identical until the user edits it in the
# prompt workbench. The runtime prompt is passed in (loaded from disk), not this.

DEFAULT_JUDGE_PROMPT = """
You decide whether a scientific paper should be surfaced to a specific researcher, given (a) that researcher's user profile describing their interests, and (b) the paper's title and abstract.

You are scoring EXPECTED INTEREST FOR THIS USER, not the paper's scientific quality or general importance. A brilliant molecular-biology paper can be a low score if this user does not care about molecular biology. A modest behavior-only animal study can be a high score if it matches what this user follows. A one-sentence news blurb about an exciting paper is still a low score because it is not a substantive article.

## How to read the user profile

The user profile lists interests, disinterests, method preferences, and article-type preferences. Read it as a description of a person's taste, not as a set of keyword rules.

- The listed interests are AMPLIFIERS, not a closed list. A paper on a topic the profile does not mention is NOT automatically low -- judge whether it fits the spirit of what this person follows. Do not penalize a paper merely because its exact topic is not named in the profile.
- Active disinterests apply only when that topic is the MAIN FOCUS of the paper, not when it appears incidentally.
- If the profile says behavioral, computational, theoretical, or conceptual work can be sufficient on its own, then do NOT require neural data. Absence of neural data is not a strike against a paper whose value is behavioral or conceptual.
- Method preferences modulate interest in a paper that is already topically relevant; they rarely make an off-topic paper interesting on their own.

## What you must NOT do

- Do NOT keyword-match. "The abstract mentions X and the profile mentions X, therefore high" is wrong reasoning. Judge meaning and fit, the way the person themselves would on reading the abstract.
- Do NOT require any specific feature (neural data, a particular organism, a particular method) unless the profile makes it a hard requirement.
- Do NOT score for general scientific importance. Score for THIS user's likely interest.
- Do NOT invent facts. Judge only from the title, abstract, and user profile. If the abstract is thin, that is a reason for lower confidence, not invented detail.
- Treat article type seriously. A one-sentence news / commentary item should usually score low even if the paper it discusses sounds interesting.

## Article length

Page range, when present, signals article type: a one- or two-page span in a paginated journal usually marks a News & Views, Preview, Dispatch, or commentary, not a primary research article -- score those as the non-substantive items they are. A multi-page span marks a substantive article, review, or perspective; do NOT dismiss it as mere commentary just because its abstract is short (perspectives and consortium pieces routinely have brief abstracts). IMPORTANT: a missing page range is UNINFORMATIVE -- many journals are electronic-only and never assign pages, so absence says nothing about substance. Never penalize a paper for lacking a page range.

## Score semantics (expected interest for this user)

The surfacing line is 0.5: papers at or above 0.5 are candidates to show the user; below 0.5 are not. The exact cutoff is tunable downstream, but treat 0.5 as the decision boundary when you score.

- 0.80 - 1.00: Strong, durable interest. Clearly show.
- 0.60 - 0.79: Solid interest. Show.
- 0.50 - 0.59: Marginal keep -- just above the line.
- 0.40 - 0.49: Marginal drop -- just below the line.
- 0.20 - 0.39: Clear non-fit.
- 0.00 - 0.19: Strong mismatch / actively unwanted.

**Precision matters near the boundary, not far from it.** A paper you are sure the user wants can be 0.85 or 0.95 -- the difference does not matter. A paper you are sure they do not want can be 0.05 or 0.20 -- also does not matter. Do NOT agonize over the exact number when a paper is clearly in or clearly out. Spend your judgment where the decision actually flips: papers near 0.5. For those, think hard about which side of the line they belong on, because that is the call that changes what the user sees. Far from the line, a rough score is fine; near the line, be deliberate.

## Output

Return ONLY a JSON object with EXACTLY these four keys (these exact names, all four always present):

{
  "estimated_score": number 0.0-1.0,        // expected interest for THIS user
  "surface_decision": "surface" | "maybe" | "do_not_surface",
  "curation_rationale": "...",              // 1-3 sentences: why this score, in terms of the person's taste
  "possible_mismatch": "..."                // the honest counter-case, or "none"
}

- surface_decision: "surface" for scores >= 0.5, "do_not_surface" below 0.5. Use "maybe" only for genuinely borderline papers right at the line (roughly 0.45-0.55) where you are torn. The decision and the score must agree: do not say "surface" with a score of 0.3.
- curation_rationale: speak in terms of the person's taste ("this person follows X and this paper does Y"), not in terms of generic quality. Be concrete and brief.
- possible_mismatch: the strongest honest reason this score might be wrong, or the strongest reason the paper might NOT fit despite a high score (or might fit despite a low score). Use "none" only when there genuinely is no meaningful counter-case.

ASCII only. Return ONLY the JSON object, no preamble, no code fences.
""".strip()

# Backward-compat alias; tools that referenced SYSTEM_PROMPT still work (they get
# the seed default). The runtime prompt is loaded from disk and passed in.
SYSTEM_PROMPT = DEFAULT_JUDGE_PROMPT

# The editable prompt is a SINGLE-paper prompt; the batch (multi-paper) variant is
# derived from it -- everything up to the '## Output' marker, then this JSON-array
# contract. So there is ONE authored artifact, and '## Output' is a structural
# marker the prompt must keep (the workbench validates it).
_OUTPUT_MARKER = "## Output"

_BATCH_OUTPUT = """## Output

You are judging MULTIPLE papers in one call. Return ONLY a JSON array -- one object per paper, \
in the same order they appear in the user message. Each object must have EXACTLY these four keys:

[
  {
    "estimated_score": number 0.0-1.0,
    "surface_decision": "surface" | "maybe" | "do_not_surface",
    "curation_rationale": "...",
    "possible_mismatch": "..."
  },
  ...
]

Apply the same scoring rules as described above. surface_decision must agree with estimated_score. \
curation_rationale is 1-3 sentences in terms of the person's taste. possible_mismatch is the \
strongest counter-case or "none".

ASCII only. Return ONLY the JSON array starting with [ and ending with ]. No preamble, no code fences.
""".strip()


def _batch_prompt(system_prompt):
    """Derive the batch system prompt from a single-paper prompt: head (up to
    '## Output') + the JSON-array output contract."""
    return system_prompt.split(_OUTPUT_MARKER, 1)[0] + _BATCH_OUTPUT


def _client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _article_to_text(title, abstract, journal, pages=None):
    lines = []
    if journal:
        lines.append(f"Journal: {journal}")
    if pages:
        lines.append(f"Pages: {pages}")
    lines.append(f"Title: {title or '(no title)'}")
    lines.append("")
    lines.append(f"Abstract: {abstract or '(no abstract)'}")
    return "\n".join(lines)


def judge_article(title, abstract, journal, profile_text, pages=None, system_prompt=None):
    """Judge a paper from title + abstract + journal (+ optional page range) against
    the user profile. system_prompt is the active judge prompt (from disk); falls
    back to DEFAULT_JUDGE_PROMPT so direct/standalone callers still work.

    Returns (judgment dict, usage, cost). Retries once on a malformed response.
    """
    system_prompt = system_prompt or DEFAULT_JUDGE_PROMPT
    article_text = _article_to_text(title, abstract, journal, pages)
    user_message = (
        f"# User profile (this user's interests)\n\n{profile_text}\n\n"
        f"# Paper\n\n{article_text}\n\n"
        f"Judge this paper for this user."
    )

    total_cost = 0.0
    last_error = None
    for attempt in range(2):
        response = _client().messages.create(
            model=MODEL,
            max_tokens=700,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        total_cost += (response.usage.input_tokens * COST_PER_M_INPUT
                       + response.usage.output_tokens * COST_PER_M_OUTPUT) / 1_000_000
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            decoder = json.JSONDecoder()
            judgment, _end = decoder.raw_decode(raw)
            _validate(judgment)
        except ValueError as e:
            last_error = e
            continue
        judgment["judge_prompt_version"] = _fingerprint(system_prompt)
        judgment["judge_model"] = MODEL
        return judgment, response.usage, total_cost

    raise ValueError(f"Judge failed after 2 attempts. Last error: {last_error}")


def judge_articles_batch(items, profile_text, system_prompt=None):
    """Judge a batch of articles in a single API call. system_prompt is the active
    judge prompt (from disk); the batch variant is derived from it. Falls back to
    DEFAULT_JUDGE_PROMPT.

    items: list of dicts with keys title, abstract, journal (all optional str).
    Returns (list of judgment dicts, total_cost). Retries once on a malformed response.
    """
    system_prompt = system_prompt or DEFAULT_JUDGE_PROMPT
    batch_prompt = _batch_prompt(system_prompt)
    article_blocks = []
    for i, item in enumerate(items, 1):
        article_text = _article_to_text(
            item.get("title"), item.get("abstract"), item.get("journal"), item.get("pages")
        )
        article_blocks.append(f"## Paper {i}\n\n{article_text}")

    user_message = (
        f"# User profile (this user's interests)\n\n{profile_text}\n\n"
        f"# Papers to judge ({len(items)} total)\n\n"
        + "\n\n---\n\n".join(article_blocks)
        + f"\n\nJudge all {len(items)} papers for this user. "
        f"Return a JSON array with exactly {len(items)} objects, in order."
    )

    total_cost = 0.0
    last_error = None
    for attempt in range(2):
        response = _client().messages.create(
            model=MODEL,
            max_tokens=len(items) * 700,
            system=batch_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        total_cost += (response.usage.input_tokens * COST_PER_M_INPUT
                       + response.usage.output_tokens * COST_PER_M_OUTPUT) / 1_000_000
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            decoder = json.JSONDecoder()
            judgments, _end = decoder.raw_decode(raw)
            if not isinstance(judgments, list):
                raise ValueError(f"Expected JSON array, got {type(judgments).__name__}")
            if len(judgments) != len(items):
                raise ValueError(f"Expected {len(items)} judgments, got {len(judgments)}")
            for j in judgments:
                _validate(j)
                j["judge_prompt_version"] = _fingerprint(system_prompt)
                j["judge_model"] = MODEL
            return judgments, total_cost
        except (ValueError, KeyError) as e:
            last_error = e
            continue

    raise ValueError(f"Batch judge failed after 2 attempts. Last error: {last_error}")


def _validate(j):
    required = {"estimated_score", "surface_decision",
                "curation_rationale", "possible_mismatch"}
    missing = required - set(j.keys())
    if missing:
        raise ValueError(f"Judgment missing fields: {missing}")

    v = j["estimated_score"]
    if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
        raise ValueError(f"estimated_score must be 0.0-1.0, got {v!r}")

    if j["surface_decision"] not in VALID_SURFACE_DECISION:
        raise ValueError(f"surface_decision {j['surface_decision']!r} not in {VALID_SURFACE_DECISION}")

    for key in ("curation_rationale", "possible_mismatch"):
        if not isinstance(j[key], str) or not j[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
