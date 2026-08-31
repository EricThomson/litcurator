"""
consolidation_picker.py -- given N consolidation rounds over the same flags, pick the best.

Consolidate quality swings hard on identical input, so the workflow is to run a few and keep
the good one. promote_suggestions already keeps a round; what did not scale was reading them.
This reads them, ranks them, and prints the winner's filename. It records nothing.

VOCABULARY: the JUDGE scores papers (judge.py). This is the PICKER, and it chooses between
rounds of the profile-update machinery.

It sees the papers, because structural arithmetic cannot do this job -- a paper-overlap metric
scored a visibly bad round perfectly, since chimeras reduce overlap by construction. It does
not see the raw clusters: those are the input every round shared, not the output being judged.
"""

import json
import pathlib
import random
from datetime import datetime

import anthropic
from dotenv import load_dotenv

from litcurator import (analysis_prompt_interface, db_interface, pick_prompt_interface,
                        profile_analysis as PA, profile_interface)

load_dotenv()

# Opus 5: the only evidence this task is doable is an Opus session doing it on two of these
# rounds and naming the exact causes. ~$0.07 more per round than Sonnet.
PICKER_MODEL = "claude-opus-5"

LABELS = "ABCDEFGH"

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array", "items": {"type": "string"},
            "description": "the round labels, BEST FIRST. Every label exactly once."},
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "run": {"type": "string", "description": "the round label"},
                    "worst_problem": {
                        "type": "string",
                        "description": "the single worst thing in this round, one or two "
                                       "sentences, naming the pattern or paper numbers "
                                       "involved so the claim can be checked"},
                },
                "required": ["run", "worst_problem"],
                "additionalProperties": False,
            },
            "description": "one entry per round, the winner included"},
        "none_are_good": {
            "type": "boolean",
            "description": "true if you would not want ANY of these recorded"},
        "why_the_winner": {
            "type": "string",
            "description": "one short paragraph: what the top-ranked round does better"},
    },
    "required": ["ranking", "assessments", "none_are_good", "why_the_winner"],
    "additionalProperties": False,
}


def _expand(paths):
    """A single round DIRECTORY expands to the run_N.md files in it; anything else is taken as
    an explicit list of reports. Explicit paths are what keeps two arbitrary old reports
    comparable without moving them into a round directory first."""
    paths = [pathlib.Path(p) for p in paths]
    if len(paths) == 1 and paths[0].is_dir():
        runs = sorted(paths[0].glob("run_*.md"),
                      key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
        if not runs:
            raise ValueError(f"{paths[0]} holds no run_*.md reports.")
        return runs
    return paths


def _active_consolidate_prompt():
    """The CONSOLIDATE section of the active analysis prompt -- the standard the rounds are
    graded against. NB it is today's prompt: grading rounds produced under an older one is a
    mismatch, so the verdict header stamps which prompt was used."""
    return analysis_prompt_interface.load_active_sections()[1]


def _papers_block_for(conn, n_expected, start, end):
    """(papers_block, came_from_unattached_pool) for the flags a report was built on.

    The [N] paper numbers index into the ordered flag list, so the block only reproduces
    against the same flags. A normal round clusters the unattached pool; an --include-attached
    comparison run clusters all of them. Try both, refuse if neither matches."""
    for exclude_attached in (True, False):
        flags = db_interface.get_flags(conn, start=start, end=end,
                                       exclude_attached=exclude_attached)
        if len(flags) == n_expected:
            return PA._format_papers(flags)[0], exclude_attached
    raise ValueError(
        f"those reports were built on {n_expected} flags, and neither the unattached pool nor "
        f"the full one holds that many now -- the picker would be shown the wrong papers.")


def _anonymised_rounds(report_paths, rng):
    """[(label, path, rendered)] shuffled, plus the flag count.

    Anonymised by PARSING each report and re-rendering through the shipped renderer, so the
    header, cluster text and filename cannot survive. Beats a hand-written stripper, which
    would be a second thing to keep in step with the report format.

    The queue cap is re-applied because parsing restores the model's ASKED priority; without
    it the picker sees the patterns the model wanted queued, not the ones the user is shown."""
    rounds, n_flags = [], None
    for path in report_paths:
        candidates, n_then = PA.parse_consolidation_md(path)
        if n_flags is None:
            n_flags = n_then
        elif n_then != n_flags:
            raise ValueError(
                f"{pathlib.Path(path).name} was built on {n_then} flags but an earlier report "
                f"used {n_flags} -- these rounds saw different evidence.")
        PA._apply_queue_cap(candidates)
        rounds.append([path, PA._format_consolidation_md(candidates)])
    rng.shuffle(rounds)
    return [(LABELS[i], path, text) for i, (path, text) in enumerate(rounds)], n_flags


def _build_message(profile_text, papers_block, memory_block, rounds, consolidate_prompt):
    # The consolidate prompt IS the standard, so the picker is shown it rather than being given
    # a second copy of the same criteria to drift from. Everything a good pattern is -- chimera
    # vs spray, how to match a merge, how to rank -- is already stated there, better than a
    # hand-kept paraphrase would state it. First, because it is what the rounds are judged
    # against and the evidence only means something once you know the standard.
    # Section headers are `#`, one level ABOVE the `##` headings inside the consolidate prompt
    # and the profile. At the same level their "## Output" and "## Topic interests" would read
    # as siblings of "## ROUND A", which misdescribes the structure of the whole message.
    parts = [
        "# CONSOLIDATE INSTRUCTIONS (what every round below was told to do)\n\n"
        + consolidate_prompt,
        "# The user's current profile\n\n" + profile_text,
        f"# The flagged papers all {len(rounds)} rounds were built from\n\n"
        f"The [N] numbers are the ones each round cites. The consolidate step never saw "
        f"these; you do.\n\n" + papers_block,
    ]
    if memory_block:
        parts.append("# The pattern memory every round was shown\n\n" + memory_block)
    for label, _path, text in rounds:
        parts.append(f"# ROUND {label}\n\n{text}")
    return "\n\n---\n\n".join(parts)


def pick_best(report_paths, start=None, end=None, model=PICKER_MODEL, seed=None):
    """Returns (verdict, rounds, cost). `rounds` is the label -> filename mapping, held here
    and never sent to the model."""
    report_paths = _expand(report_paths)
    if len(report_paths) < 2:
        raise ValueError("give at least two reports -- there is nothing to pick between.")
    pick_prompt = pick_prompt_interface.load_active()
    pick_prompt_interface.snapshot_active(pick_prompt)

    rounds, n_flags = _anonymised_rounds(report_paths, random.Random(seed))

    conn = db_interface.get_connection()
    try:
        papers_block, from_unattached = _papers_block_for(conn, n_flags, start, end)
        # Show the memory only when the rounds could see it. An --include-attached comparison
        # run gets a blank memory on purpose, so showing it here would mark every round down
        # for failing to merge into patterns it was never told about. Which pool the papers
        # came from is the evidence for which mode produced the reports.
        memory = PA.build_memory_block(conn)[0] if from_unattached else ""
    finally:
        conn.close()

    response = anthropic.Anthropic().messages.create(
        model=model,
        max_tokens=16000,
        system=pick_prompt,
        messages=[{"role": "user",
                   "content": _build_message(profile_interface.load_active(), papers_block,
                                             memory, rounds,
                                             _active_consolidate_prompt())}],
        # Structured output, not the forced tool-use consolidate uses: same JSON guarantee and
        # it composes with the adaptive thinking Opus 5 runs by default. No temperature -- it
        # is removed on Opus 5 and sending one is a 400.
        output_config={"format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), rounds, PA._cost(model, response.usage)


def write_verdict(verdict, rounds, cost, model, out_dir=None):
    """Save the verdict beside the reports it judged, with the mapping spelled out."""
    out_dir = pathlib.Path(out_dir or PA.SUGGESTIONS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_label = {label: path for label, path, _ in rounds}
    winner = verdict["ranking"][0]
    prompt_hash = pick_prompt_interface.content_hash(pick_prompt_interface.load_active())

    lines = [
        "# Pick verdict\n",
        f"model: {model}  |  pick prompt: {prompt_hash}  |  graded against consolidate "
        f"prompt: {analysis_prompt_interface.content_hash(_active_consolidate_prompt())}  |  "
        f"cost: ${cost:.4f}  |  {datetime.now():%Y-%m-%d %H:%M:%S}\n",
        f"WINNER: {pathlib.Path(by_label[winner]).name}\n",
    ]
    if verdict.get("none_are_good"):
        lines.append("**The picker says none of these is worth recording.** The ranking below "
                     "is a least-bad ordering.\n")
    lines.append("## Ranking\n")
    for place, label in enumerate(verdict["ranking"], start=1):
        lines.append(f"{place}. round {label} -- {pathlib.Path(by_label[label]).name}")
    lines += ["\n## Why the winner\n", verdict["why_the_winner"],
              "\n## The worst thing in each round\n"]
    for entry in verdict["assessments"]:
        name = pathlib.Path(by_label.get(entry["run"], entry["run"])).name
        lines.append(f"- **{entry['run']}** ({name}): {entry['worst_problem']}")
    lines.append(f"\n## To record the winner\n\n    litcurator promote_suggestions "
                 f"{by_label[winner]}\n")

    out = out_dir / f"pick_verdict_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
