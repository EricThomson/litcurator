"""
analysis_harness -- the gates for the profile-analysis machinery.

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

from litcurator import profile_analysis as PA

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


def is_free(name):
    return name in FREE_GATES


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
    process only -- the draft file and profile_analysis.py are both left alone.

    Stops before the first paid gate if any free gate is red: the paid ones would only produce
    confusing symptoms on top of a structural break.
    """
    import anthropic

    gates = gates or select_gates()
    if cluster_prompt is not None:
        PA.CLUSTER_PROMPT = cluster_prompt
    if consolidate_prompt is not None:
        PA._CONSOLIDATE_SYSTEM = consolidate_prompt
    cluster_fp = fingerprint(PA.CLUSTER_PROMPT)
    consolidate_fp = fingerprint(PA._CONSOLIDATE_SYSTEM)

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
                use_cache=use_cache)
        started = time.time()
        try:
            checks, cost, transcript = run(ctx)
            results.append(_blank(name, layer, checks=checks, cost=cost, transcript=transcript,
                                  passed=all(ok for ok, _, _ in checks), when_red=when_red,
                                  seconds=time.time() - started))
        except Exception as e:                            # noqa: BLE001 - report, never crash
            results.append(_blank(name, layer, error=f"{type(e).__name__}: {e}",
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


def dry_run(gates=None):
    """What WOULD run, and whether it could. Returns (text, ok). Spends nothing."""
    from litcurator import config

    gates = gates or select_gates()
    calls, n_free = budget(gates)
    out = ["ANALYSIS HARNESS -- dry run (nothing runs, nothing is spent)",
           f"cluster prompt      {fingerprint(PA.CLUSTER_PROMPT)}",
           f"consolidate prompt  {fingerprint(PA._CONSOLIDATE_SYSTEM)}",
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
        out.append(f"        {', '.join(multi)} run many sessions in sequence. Warm that is "
                   f"about a minute each; on a COLD cache (a cluster-prompt edit invalidates "
                   f"every key) it is several minutes each.")
    out.append("        `litcurator analysis_harness quick` runs the free gates only, 0 calls")
    return "\n".join(out), all(ok for ok, _, _ in checks)
