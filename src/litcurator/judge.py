"""
judge.py -- the curation judge (v4, profile-only).

The judge reads a paper's title, abstract, and journal and estimates the
user's interest given their user profile alone. No card intermediary, no
precedents.
"""

import hashlib
import json
import os
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"


def _fingerprint(text):
    """Short display id of a prompt: the 12-char prefix of its SHA256, so it lines up
    with the DB content-address (prompts.id = SHA256(content)); mirrors
    profile_interface.content_hash. Attached to a judgment dict for display; the
    persisted provenance is the full sha256 judge_prompt_hash on the scoring_run."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

# Must track MODEL. Sonnet 4.6 is 3/15; Opus 4.8 is 15/75; Haiku 4.5 is 1/5. A
# stale pair here makes scoring_runs.cost_usd silently wrong by 5x.
COST_PER_M_INPUT = 3.0
COST_PER_M_OUTPUT = 15.0

VALID_SURFACE_DECISION = {"surface", "maybe", "do_not_surface"}

# The judge endpoint intermittently returns an empty completion (or a transient
# overload / connection error). An empty response must NEVER silently drop a
# paper -- in live curation that paper just vanishes from the feed with no trace.
# So we retry with exponential backoff instead of giving up after 2 tries.
_MAX_JUDGE_ATTEMPTS = 5
_TRANSIENT_API_ERRORS = (anthropic.APIStatusError, anthropic.APIConnectionError)


# ---------------------------------------------------------------------------
# Primary prompts -- article (title + abstract) mode, profile-only
# ---------------------------------------------------------------------------
# litcurator ships NO judge prompt. The active one lives at prompt/judge_prompt.md in your data
# directory, is loaded there by the pipeline, and is passed in to every call here. There is no
# default to fall back on -- see PROMPT_NOT_AUTHORED below.

# litcurator ships NO prompt content. The judge runs prompt/judge_prompt.md from your data
# directory; authoring it is your job, exactly like the profile. There is no default here on
# purpose -- a half-reasonable default is worse than none, because it looks authoritative,
# quietly shapes every score, and nobody remembers it is there. Writing a genuinely good
# starting prompt is a real task and is on the long-term list, not a thing to fake in passing.
#
# This sentinel exists so the failure is LOUD and says what to do, instead of an empty string
# reaching the API and producing confident nonsense.
PROMPT_NOT_AUTHORED = "<no judge prompt authored>"

# Backward-compat alias. There is no shipped prompt any more, so this is the sentinel: a
# caller that relied on the module-level default now gets a clear error instead of silent
# behavior from text it never chose.
SYSTEM_PROMPT = PROMPT_NOT_AUTHORED

# The editable prompt is a SINGLE-paper prompt; the batch (multi-paper) variant is
# derived from it -- everything up to the '## Output' marker, then this JSON-array
# contract. So there is ONE authored artifact, and '## Output' is a structural
# marker the prompt must keep.
#
# IMPORTED, NOT REDEFINED (2026-09-22). This module SPLITS on the marker and
# prompt_interface.set_active VALIDATES against it; when each owned a copy, one contract lived
# in two places that had to agree or the judge would silently lose its output spec.
from litcurator.prompt_interface import OUTPUT_MARKER as _OUTPUT_MARKER

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


def _require_prompt(system_prompt):
    """Resolve the judge prompt, or fail loudly. Callers in the package always pass one (the
    pipeline loads the active prompt from disk and hands it down); this catches a direct or
    standalone caller that assumed a default exists."""
    if system_prompt and system_prompt != PROMPT_NOT_AUTHORED:
        return system_prompt
    raise ValueError(
        "No judge prompt. litcurator ships no default -- author one and set it active "
        "(litcurator prompt_workbench), or pass system_prompt= explicitly. "
        "The active prompt lives at prompt/judge_prompt.md in your data directory.")


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


def _decode_json(raw, opener):
    """Decode the first JSON value starting at the first `opener` ("{" or "[") in
    raw, tolerating any prose/preamble or code fences the model emits around it
    (raw_decode stops at the end of the value and ignores trailing text). Raises
    ValueError if no opener is present -- an empty or prose-only response, which
    the caller retries. This is what keeps a stray sentence of reasoning from
    silently dropping a paper."""
    start = raw.find(opener)
    if start == -1:
        raise ValueError(f"no JSON {opener!r} found in response")
    value, _end = json.JSONDecoder().raw_decode(raw, start)
    return value


def _judge_call(system_prompt, user_message, max_tokens, parse):
    """Make one judge API call and parse it, resilient to the transient empty
    completions and overload errors the endpoint occasionally returns. Retries up
    to _MAX_JUDGE_ATTEMPTS with exponential backoff (0.5, 1, 2, 4s). An EMPTY
    response is treated as a retryable failure -- letting an empty completion
    silently drop a paper is the exact failure mode we must avoid. parse(raw_text)
    decodes and validates a non-empty response, raising ValueError/KeyError on a
    malformed one (also retried). Returns (parsed_result, usage, cumulative_cost);
    raises after the last attempt if every attempt fails."""
    last_error = None
    total_cost = 0.0
    usage = None
    for attempt in range(_MAX_JUDGE_ATTEMPTS):
        response = None
        try:
            response = _client().messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except _TRANSIENT_API_ERRORS as e:
            last_error = e

        if response is not None:
            usage = response.usage
            total_cost += (usage.input_tokens * COST_PER_M_INPUT
                           + usage.output_tokens * COST_PER_M_OUTPUT) / 1_000_000
            raw = response.content[0].text.strip() if response.content else ""
            if not raw:
                last_error = ValueError("empty response from judge")
            else:
                try:
                    return parse(raw), usage, total_cost
                except (ValueError, KeyError) as e:
                    last_error = e

        if attempt < _MAX_JUDGE_ATTEMPTS - 1:
            time.sleep(0.5 * 2 ** attempt)

    raise ValueError(f"Judge failed after {_MAX_JUDGE_ATTEMPTS} attempts. "
                     f"Last error: {last_error}")


def judge_article(title, abstract, journal, profile_text, pages=None, system_prompt=None):
    """Judge a paper from title + abstract + journal (+ optional page range) against
    the user profile. system_prompt is the active judge prompt (from disk); falls
    raises if none was given -- there is no shipped default to fall back to.

    Returns (judgment dict, usage, cost). Retries once on a malformed response.
    """
    system_prompt = _require_prompt(system_prompt)
    article_text = _article_to_text(title, abstract, journal, pages)
    user_message = (
        f"# User profile (this user's interests)\n\n{profile_text}\n\n"
        f"# Paper\n\n{article_text}\n\n"
        f"Judge this paper for this user."
    )

    def parse(raw):
        judgment = _decode_json(raw, "{")
        _validate(judgment)
        return judgment

    judgment, usage, cost = _judge_call(system_prompt, user_message, 700, parse)
    judgment["judge_prompt_version"] = _fingerprint(system_prompt)
    judgment["judge_model"] = MODEL
    return judgment, usage, cost


def judge_articles_batch(items, profile_text, system_prompt=None):
    """Judge a batch of articles in a single API call. system_prompt is the active
    judge prompt (from disk); the batch variant is derived from it. Falls back to
    the active prompt; raises if none is supplied.

    items: list of dicts with keys title, abstract, journal (all optional str).
    Returns (list of judgment dicts, total_cost). Retries once on a malformed response.
    """
    system_prompt = _require_prompt(system_prompt)
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

    def parse(raw):
        judgments = _decode_json(raw, "[")
        if not isinstance(judgments, list):
            raise ValueError(f"Expected JSON array, got {type(judgments).__name__}")
        if len(judgments) != len(items):
            raise ValueError(f"Expected {len(items)} judgments, got {len(judgments)}")
        for j in judgments:
            _validate(j)
            j["judge_prompt_version"] = _fingerprint(system_prompt)
            j["judge_model"] = MODEL
        return judgments

    judgments, _usage, total_cost = _judge_call(batch_prompt, user_message, len(items) * 700, parse)
    return judgments, total_cost


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
