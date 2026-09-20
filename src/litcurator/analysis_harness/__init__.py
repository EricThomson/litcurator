"""
analysis_harness -- the gates for the error-analysis machinery.

The judge is a plain function (one paper in, one score out), so `litcurator judge_harness` can
test it with a list of papers and expected bands. This machinery is different: a whole batch of
flags goes in, a set of patterns comes out, permanent memory changes, and what happens in
session three depends on what you decided in sessions one and two. So its gates are staged
synthetic SCENARIOS rather than a list of cases.

Nothing here calls the judge. Flags are fabricated directly -- a made-up judge score, a made-up
score of your own, on a made-up paper -- so any history can be staged for a few dollars. Every
synthetic flag is tagged with the pattern it is supposed to end up in, and grading reads the
provenance graph (which flags ended up attached to which pattern), never the model's wording,
so a beautifully worded but wrong result still fails.

WHAT IS SIMULATED HERE IS FLAG MECHANICS, NOT PROFILE OR PROMPT MECHANICS. The two hand-authored
artifacts are PARAMETERS, not state: every scenario carries a synthetic profile and a synthetic
judge prompt, the machinery reads both, and neither is ever rewritten. `apply_actions`
"incorporate" stamps `get_or_create_profile` on the SAME unchanged string, so every incorporation
in a run points at one profile version; the lifecycle fixture simulates an edit that worked by
simply not re-emitting that pattern's flags next session. The only way an artifact ever varies is
hand-authoring a variant up front, which is what `named_disinterest` does.

So these gates grade ROUTING and RECORDING -- did the right flags group together, did the decision
land on the right pattern, is the fix addressed to the artifact that can absorb it -- and they
CANNOT grade whether an edit works, because no edit is ever made. That is deliberate: an LLM
rewriting the profile in a loop is the v1 failure this architecture exists to prevent, so a
harness generating edits would be exercising the forbidden thing. "Did the edit take" has its own
instruments -- `judge_harness` for a fast floor, and a paired re-score of the same papers under
the old and new artifact for the real answer.

WHICH IS WHY EVERY FIXTURE PAPER MUST CARRY A NOTE (2026-09-19, 38 of 38; it was 10 of 38). Not
for realism -- structurally. With both artifacts frozen, the NOTE is the only channel through
which the user's intent varies at all, so a note-free fixture runs this machinery with its main
input muted and reduces the whole input to which papers appeared and how big the deltas were.
That is a flag as a number, which is the dead paradigm the rest of this project spent 2026-08
unlearning. It bites hardest on `blame`: the only evidence that can say which document to edit is
what the frozen artifacts state, plus the note. Strip the notes and the question becomes an
inference from paper content alone, which is the undecidable case nothing here grades. See
paper_pools.py for the writing constraint -- vary the surface reason, never write a grouping label.

Testing that a prompt-blamed fix actually LANDS would need a scenario with two hand-authored judge
prompt variants, the gap in one and the fix in the other. Nothing does that today.

Layout:
  machinery.py     the scratch world: build a throwaway database, run one session
  grading.py       the pure graders, plain data in and [(ok, label, detail)] out
  fixtures/        the synthetic profile, papers, paper sets, and named scenarios
  gates_no_llm/    free, instant gates -- run these first
  gates_llm/       the paid gates that drive the real cluster and consolidate steps

Every gate, free or paid, returns the same [(ok, label, detail)], which is what lets one report
span both. Driven by `litcurator analysis_harness`; see cli.py.
"""

import os
import time
import traceback

from litcurator import analysis_prompt_interface, error_analysis as PA

from .gates_llm import COLD_CACHE_GATES, PAID_GATES
from .gates_no_llm import FREE_GATES, run_free_gate
from .machinery import GateContext
from .report import fingerprint, format_report, format_transcripts, write_report

# Every gate in the order it should run: free first, then by layer. A red free gate stops the
# run before anything is spent, which is the advice the docs already give by hand.
GATE_ORDER = list(FREE_GATES) + sorted(PAID_GATES, key=lambda n: (PAID_GATES[n][0], n))


def select_gates(name=None):
    """Resolve a gate selector into an ordered list of gate names.

      None    every gate -- about five minutes on a warm cluster cache
      "quick" the free gates only, a second, spends nothing
      a name  that one gate
    """
    if name is None:
        return list(GATE_ORDER)
    if name == "quick":
        return list(FREE_GATES)
    if name not in GATE_ORDER:
        raise KeyError(f"unknown gate {name!r}. Known: quick, {', '.join(GATE_ORDER)}")
    return [name]


def resolve_prompts(cluster_prompt=None, consolidate_prompt=None):
    """Return the (cluster, consolidate) prompt TEXT this run should use: the active analysis
    prompt from disk, with either section replaced by a draft if one was given.

    Returns text rather than mutating error_analysis globals, which is what the harness used
    to do. Mutating a module global to test a draft means the override leaks to anything else
    in the process and cannot be nested -- the judge never worked that way (judge_articles_batch
    has always taken system_prompt=), and now neither does this.

    Separate from run_gates because the dry run needs the fingerprints WITHOUT running anything.
    It used to get them by calling run_gates([]) purely for the side effect, which spent the
    entire budget: `gates or select_gates()` treats an empty list as "not specified", so the dry
    run silently executed every gate and then printed that nothing had been spent."""
    active_cluster, active_consolidate = analysis_prompt_interface.load_active_sections()
    return (cluster_prompt or active_cluster, consolidate_prompt or active_consolidate)


def budget(gates):
    """(model calls, free gate count) for the selected gates, for the pre-spend estimate."""
    calls = sum(PAID_GATES[g][3] for g in gates if g in PAID_GATES)
    return calls, sum(1 for g in gates if g in FREE_GATES)


def preconditions():
    """[(ok, what, detail)] -- the things a gate needs that are worth reporting BEFORE a run
    rather than as a traceback halfway through."""
    from litcurator import config, db_interface

    out = []
    db = config.LITCURATOR_DB
    if db.exists():
        conn = db_interface.get_connection()
        try:
            n = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        finally:
            conn.close()
        out.append((n >= 6, "live database",
                    f"{db} ({n} articles)" if n >= 6
                    else f"{db} has only {n} articles; the record-stage gates need at least 6"))
    else:
        out.append((False, "live database", f"{db} not found"))
    out.append((bool(os.getenv("ANTHROPIC_API_KEY")), "ANTHROPIC_API_KEY",
                "set" if os.getenv("ANTHROPIC_API_KEY") else "not set -- paid gates cannot run"))
    # The active analysis prompt must split into two non-empty sections. A mistyped marker
    # would otherwise send an EMPTY prompt to the model, which surfaces as mysteriously bad
    # clustering rather than as an error -- the kind of failure that costs a day to find. It
    # belongs here rather than in a gate because it is a can-this-run-at-all question, and this
    # way it shows in --dry-run before anything is spent.
    try:
        c, s = analysis_prompt_interface.load_active_sections()
        out.append((True, "analysis prompt",
                    f"{analysis_prompt_interface.active_version_id()} "
                    f"(cluster {len(c)} chars, consolidate {len(s)} chars)"))
    except Exception as e:                                # noqa: BLE001 - report, never crash
        out.append((False, "analysis prompt", f"{type(e).__name__}: {e}"))
    return out


def _blank(name, layer, **kw):
    row = {"name": name, "layer": layer, "checks": [], "passed": False, "cost": 0.0,
           "seconds": 0.0, "error": None, "transcript": "", "when_red": ""}
    row.update(kw)
    return row


def run_gates(gates=None, cluster_prompt=None, consolidate_prompt=None,
              cluster_model=None, consolidate_model=None, use_cache=True, progress=None):
    """Run the selected gates and return (results, cluster_fp, consolidate_fp).

    Computes and returns data; prints nothing. A draft prompt is applied in memory for this
    process only -- the draft file and error_analysis.py are both left alone.

    Stops before the first paid gate if any free gate is red: the paid ones would only produce
    confusing symptoms on top of a structural break.
    """
    import anthropic

    # `is None`, not falsy: an explicitly EMPTY list means run nothing, and must not be read as
    # "not specified" and quietly expanded to every gate.
    if gates is None:
        gates = select_gates()
    cluster_text, consolidate_text = resolve_prompts(cluster_prompt, consolidate_prompt)
    cluster_fp, consolidate_fp = fingerprint(cluster_text), fingerprint(consolidate_text)

    ctx = None
    results = []
    for i, name in enumerate(gates):
        if progress:
            progress(name, i + 1, len(gates))

        if name in FREE_GATES:
            started = time.time()
            try:
                checks = run_free_gate(name)
                results.append(_blank(name, 1, checks=checks,
                                      passed=all(ok for ok, _, _ in checks),
                                      seconds=time.time() - started))
            except Exception as e:                        # noqa: BLE001 - report, never crash
                results.append(_blank(name, 1, error=f"{type(e).__name__}: {e}",
                                      transcript=traceback.format_exc(),
                                      seconds=time.time() - started))
            continue

        # About to spend. A structural break upstream makes everything below it unreadable.
        if any(r["layer"] == 1 and not r["passed"] for r in results):
            break

        layer, run, when_red, _calls = PAID_GATES[name]
        if ctx is None:
            ctx = GateContext(
                client=anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
                cluster_model=cluster_model or PA.DEFAULT_CLUSTER_MODEL,
                consolidate_model=consolidate_model or PA.DEFAULT_CONSOLIDATE_MODEL,
                cluster_prompt=cluster_text, consolidate_prompt=consolidate_text,
                use_cache=use_cache)
        started = time.time()
        try:
            checks, cost, transcript = run(ctx)
            results.append(_blank(name, layer, checks=checks, cost=cost, transcript=transcript,
                                  passed=all(ok for ok, _, _ in checks), when_red=when_red,
                                  seconds=time.time() - started))
        except Exception as e:                            # noqa: BLE001 - report, never crash
            # The full traceback goes to the transcript section, because an ERROR whose only
            # record is its message costs a debugging session to localise: the 2026-09-19
            # pool-calibration failure surfaced as one line ("'str' object has no attribute
            # 'get'") that had to be reproduced from scratch to find its file and line. NB an
            # exception here can also mean money was spent and never counted -- the cost
            # travels in run()'s return value, which this path never receives.
            results.append(_blank(name, layer, error=f"{type(e).__name__}: {e}",
                                  transcript=traceback.format_exc(),
                                  when_red=when_red, seconds=time.time() - started))
    return results, cluster_fp, consolidate_fp


def exit_code(results, selected=None):
    """0 all good, 1 something is red, 2 something could not RUN.

    The 1/2 split matters: if an API outage produced a red, you would learn to discount reds.
    A 2 means the verdict is unknown, not bad."""
    if any(r["error"] for r in results):
        return 2
    if any(not r["passed"] for r in results):
        return 1
    if selected and len(results) < len(selected):     # stopped early on a red free gate
        return 1
    return 0


def dry_run(gates=None, cluster_prompt=None, consolidate_prompt=None):
    """What WOULD run, and whether it could. Returns (text, ok). Spends nothing.

    Takes the draft prompts so the fingerprints answer "did I paste the right path", without
    running or mutating anything."""
    from litcurator import config

    gates = gates or select_gates()
    cluster_text, consolidate_text = resolve_prompts(cluster_prompt, consolidate_prompt)
    calls, n_free = budget(gates)
    out = ["ANALYSIS HARNESS -- dry run (nothing runs, nothing is spent)",
           f"cluster prompt      {fingerprint(cluster_text)}",
           f"consolidate prompt  {fingerprint(consolidate_text)}",
           f"models              cluster={PA.DEFAULT_CLUSTER_MODEL}  "
           f"consolidate={PA.DEFAULT_CONSOLIDATE_MODEL}",
           "", "preconditions"]
    checks = preconditions()
    for ok, what, detail in checks:
        out.append(f"  {'ok  ' if ok else 'FAIL'} {what:<18} {detail}")

    out += ["", f"gates ({len(gates)})"]
    for name in gates:
        if name in FREE_GATES:
            out.append(f"  free  1  {name:<20} no model calls")
        else:
            layer, _run, _when, n = PAID_GATES[name]
            out.append(f"  llm   {layer}  {name:<20} {n} model calls")
    cached = len(list(config.ANALYSIS_HARNESS_CACHE_DIR.glob("*.md"))) \
        if config.ANALYSIS_HARNESS_CACHE_DIR.exists() else 0
    out += ["", f"budget  {calls} model calls, {n_free} free gates",
            f"        cluster cache holds {cached} entries; a hit costs nothing"]
    multi = [g for g in gates if g in COLD_CACHE_GATES]
    if multi:
        # The key is sha256(cluster prompt, papers block, profile, model), so ANY change to
        # how _format_papers renders invalidates it too -- not just a prompt edit. That is
        # easy to forget, because a renderer change is not a prompt change and leaves no
        # fingerprint in the report.
        out.append(f"        {', '.join(multi)} run many sessions in sequence. Warm that is "
                   f"about a minute each; on a COLD cache (any edit to the cluster prompt OR "
                   f"to the papers-block renderer invalidates every key) it is several "
                   f"minutes each.")
    out.append("        `litcurator analysis_harness quick` runs the free gates only, 0 calls")
    return "\n".join(out), all(ok for ok, _, _ in checks)
