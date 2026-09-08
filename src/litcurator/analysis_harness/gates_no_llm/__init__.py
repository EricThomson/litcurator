"""Gates that use no model at all. They run in about a second, cost nothing, and are the
first thing to run after touching any of this code.

If one of these is red, stop: something structural is broken and the paid gates will only
give you confusing symptoms.

These gates are written as plain assert-based scripts, which is the right shape for running
one directly while you work on it. This module adapts them to the same
[(ok, label, detail)] language every other gate speaks, WITHOUT touching their bodies.
There are two shapes to adapt:

  named checks -- the module exposes CHECKS = [fn, ...]. Each is called on its own, so one
                  red check does not hide the others, and its printed summary line becomes
                  the detail column.
  one script   -- the module runs top to bottom in main(), printing one line per stage it
                  completes. Each printed line becomes a passing check; an assertion turns
                  the lines that got through into passes plus one failure saying where it
                  stopped, because a top-to-bottom script genuinely has no results past
                  that point.
"""

import contextlib
import importlib
import io
from pathlib import Path

# Trailers a script prints to summarize itself. They are the old report format, not checks.
_TRAILERS = ("ALL CHECKS PASSED", "SOME CHECKS FAILED")


def _where(exc):
    """file:line of the deepest frame in an assertion's traceback -- where it actually blew
    up, not where we called it from."""
    tb, last = exc.__traceback__, None
    while tb:
        last, tb = tb, tb.tb_next
    if not last:
        return ""
    return f"{Path(last.tb_frame.f_code.co_filename).name}:{last.tb_lineno}"


def _failure_detail(exc):
    first = next((ln for ln in str(exc).strip().splitlines() if ln.strip()),
                 "assertion failed")
    return f"{_where(exc)}  {first}"[:180]


def _stage_lines(captured):
    """The meaningful printed lines: one per stage a script completed."""
    return [ln.strip() for ln in captured.splitlines()
            if ln.strip() and ln.strip() not in _TRAILERS]


def _label_for(gate, line):
    """A stable label from a printed line: the part before the first colon, which is the
    property being checked. What follows the colon varies run to run, so it is detail."""
    head = line.split(":", 1)[0].strip() if ":" in line else line
    return f"{gate}: {head[:60]}"


def _run_named(module, gate):
    """Shape one: a module exposing CHECKS.

    Catches Exception, not just AssertionError. A check that dies some OTHER way -- a leaked
    file handle, a locked scratch database, a typo in the check itself -- used to escape this
    loop, escape run_free_gate, and abort the whole `litcurator analysis_harness` run with a
    traceback and no report at all. That is a worse outcome than any red: you lose the eleven
    gates that would have run next, and you cannot tell a broken check from a broken system.
    It is reported as a failed check carrying the exception type, so the run continues."""
    out = []
    for fn in module.CHECKS:
        label = f"{gate}: {fn.__name__.replace('test_', '', 1)}"
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                fn()
            printed = _stage_lines(buf.getvalue())
            out.append((True, label, printed[-1] if printed else ""))
        except AssertionError as e:
            out.append((False, label, _failure_detail(e)))
        except Exception as e:                       # noqa: BLE001 -- see the docstring
            out.append((False, label,
                        f"{_where(e)}  {type(e).__name__}: {str(e).strip()[:120]}"))
    return out


def _run_script(module, gate):
    """Shape two: a module that runs top to bottom in main().

    Catches Exception for the same reason _run_named does: a script gate that dies on
    something other than an assertion must not take the whole run down with it."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            module.main()
        done = _stage_lines(buf.getvalue())
        # A stage that reports its own failure rather than asserting (pattern_schema does
        # this on one branch) must not count as a pass just because it printed a line.
        return [((("(BAD)" not in ln) and ("FAILED" not in ln)), _label_for(gate, ln), ln)
                for ln in done]
    except Exception as e:                           # noqa: BLE001 -- see the docstring
        done = _stage_lines(buf.getvalue())
        out = [(True, _label_for(gate, ln), ln) for ln in done]
        detail = (_failure_detail(e) if isinstance(e, AssertionError)
                  else f"{_where(e)}  {type(e).__name__}: {str(e).strip()[:120]}")
        out.append((False, f"{gate}: stopped",
                    f"{detail} -- later stages in this gate did not run"))
        return out


# name -> (module name, how to run it). Order matters: the two workbench gates go last because
# they repoint db_interface.LITCURATOR_DB at their own scratch database, and the runner restores
# that afterwards rather than trusting a gate to.
#
# Module NAMES rather than modules: importing them here would mean a gate run with
# `python -m ...gates_no_llm.record_stage` gets imported twice, once by this package and once as
# __main__, which Python warns about and which can behave unpredictably. They are imported when
# a gate actually runs.
FREE_GATES = {
    "terminal-grader": ("terminal_grader", _run_named),
    "round-grader": ("round_grader", _run_named),
    "scenario-compiler": ("scenario_compiler", _run_named),
    "record-stage": ("record_stage", _run_script),
    "pattern-schema": ("pattern_schema", _run_script),
    "workbench-actions": ("workbench_actions", _run_named),
    "workbench-render": ("workbench_render", _run_script),
    "undo-stage": ("undo_stage", _run_named),
    "sinkhole-stage": ("sinkhole_stage", _run_named),
    "blame-stage": ("blame_stage", _run_named),
}


def run_free_gate(name):
    """Run one free gate and return its [(ok, label, detail)]. Snapshots and restores the
    database path these gates repoint, so one gate cannot leak into the next."""
    from litcurator import db_interface

    module_name, runner = FREE_GATES[name]
    module = importlib.import_module(f".{module_name}", __package__)
    saved_db = db_interface.LITCURATOR_DB
    try:
        return runner(module, name)
    finally:
        db_interface.LITCURATOR_DB = saved_db
