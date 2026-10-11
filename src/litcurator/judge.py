"""
judge.py -- stage 2, the curation judge.

The judge reads papers (title, abstract, journal, page range) with the active profile and the
active judge prompt, and estimates the user's interest in each. There is ONE path: a single
paper is a batch of one, so the pipeline, the judge harness and a spot check all send the same
prompt in the same layout.

The judge prompt is sent exactly as authored. It says what the four fields MEAN; the SHAPE of
the reply is JUDGMENT_SCHEMA below, enforced by the API (structured outputs). So the prompt
holds no JSON syntax and this module holds no instructions, only labels.
"""

import json
import time

import anthropic

from litcurator.config import JUDGE_MODEL, api_cost_usd
from litcurator.content_address import short_id


VALID_SURFACE_DECISION = {"surface", "maybe", "do_not_surface"}

# The judge endpoint intermittently returns an empty completion (or a transient
# overload / connection error). An empty response must NEVER silently drop a
# paper -- in live curation that paper just vanishes from the feed with no trace.
# So we retry with exponential backoff instead of giving up after 2 tries.
_MAX_JUDGE_ATTEMPTS = 5
_TRANSIENT_API_ERRORS = (anthropic.APIStatusError, anthropic.APIConnectionError)

# The shape of one reply: a judgment per paper, each carrying the number of the paper it is
# for, so results are matched to papers by number instead of by trusting their order. The API
# cannot enforce the score's 0-1 range or the number of judgments, so the code checks both.
JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_number": {"type": "integer"},
                    "estimated_score": {"type": "number"},
                    "surface_decision": {"type": "string",
                                         "enum": sorted(VALID_SURFACE_DECISION)},
                    "curation_rationale": {"type": "string"},
                    "possible_mismatch": {"type": "string"},
                },
                "required": ["paper_number", "estimated_score", "surface_decision",
                             "curation_rationale", "possible_mismatch"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["judgments"],
    "additionalProperties": False,
}


def _require_prompt(system_prompt):
    """The judge prompt, or a loud failure. litcurator ships no judge prompt on purpose: a
    half-reasonable default looks authoritative and quietly shapes every score. The pipeline
    loads the active one from disk and passes it in; this catches a caller that did not."""
    if system_prompt and system_prompt.strip():
        return system_prompt
    raise ValueError(
        "No judge prompt. litcurator ships no default -- author one and set it active "
        "(litcurator judge_workbench, the judge prompt tab), or pass system_prompt= "
        "explicitly. "
        "The active prompt lives at prompt/judge_prompt.md in your data directory.")


def _client():
    return anthropic.Anthropic()


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
                model=JUDGE_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                output_config={"format": {"type": "json_schema", "schema": JUDGMENT_SCHEMA}},
            )
        except _TRANSIENT_API_ERRORS as e:
            last_error = e

        if response is not None:
            usage = response.usage
            total_cost += api_cost_usd(JUDGE_MODEL, usage)
            raw = next((b.text for b in response.content if b.type == "text"), "").strip()
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


def judge_articles_batch(items, profile_text, system_prompt=None):
    """Judge one or more papers in a single API call. A single paper is a batch of one: the
    prompt and the layout are the same whatever the count.

    items: list of dicts with keys title, abstract, journal, pages (all optional str).
    system_prompt: the judge prompt, sent exactly as authored; raises if none is given.
    Returns (list of judgment dicts in the order of items, total_cost).
    """
    system_prompt = _require_prompt(system_prompt)
    paper_blocks = [
        f"## Paper {number}\n\n" + _article_to_text(
            item.get("title"), item.get("abstract"), item.get("journal"), item.get("pages"))
        for number, item in enumerate(items, 1)
    ]
    user_message = (
        f"# User profile (this user's interests)\n\n{profile_text}\n\n"
        f"# Papers to judge ({len(items)} total)\n\n"
        + "\n\n---\n\n".join(paper_blocks)
    )
    expected_numbers = list(range(1, len(items) + 1))

    def parse(raw):
        by_number = {j.pop("paper_number"): j for j in json.loads(raw)["judgments"]}
        if sorted(by_number) != expected_numbers:
            raise ValueError(f"expected one judgment for each of papers 1-{len(items)}, "
                             f"got paper numbers {sorted(by_number)}")
        judgments = [by_number[number] for number in expected_numbers]
        for j in judgments:
            _validate(j)
            j["judge_prompt_version"] = short_id(system_prompt)
            j["judge_model"] = JUDGE_MODEL
        return judgments

    judgments, _usage, total_cost = _judge_call(system_prompt, user_message,
                                                len(items) * 700, parse)
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
