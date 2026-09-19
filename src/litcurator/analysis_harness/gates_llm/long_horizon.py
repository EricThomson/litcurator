"""
long_horizon.py -- run a named scenario over many sessions and grade what only the
whole history can show.

This is the ONLY driver. Every paid scenario runs through it, whether it is three sessions
with the human deciding in between or twelve sessions of a slow trickle -- there used to be
a second, hand-rolled loop for the three-session case, which did nothing this does not.

run_scenario plays a whole synthetic history: each session's flags go in, the real cluster ->
consolidate -> record pipeline runs, per-session expectations are graded, and any scripted
human decisions are applied before the next session. check_terminal then grades what only the
whole history can show, like "did this eventually gather into one pattern."

Scenarios are registered by name in fixtures/behaviors.py.
"""

import random
from collections import Counter, defaultdict

from dotenv import load_dotenv

from litcurator import db_interface as DB

from ..fixtures import scenario_gen as GEN
from ..machinery import (scratch_db_path, build_db, add_round_flags, run_round,
                         apply_actions, pattern_intended, patterns_by_flag,
                         dominant_intended, purity_of)
from ..grading import check_round, check_terminal

load_dotenv()

ENGINE_DB = scratch_db_path("long_horizon")


# ---------------------------------------------------------------------------
# Running one scenario end to end
# ---------------------------------------------------------------------------

def run_scenario(spec, rng, client, cluster_model, consolidate_model, cluster_prompt,
                 consolidate_prompt, use_cache=True, db_path=ENGINE_DB, keep_db=False,
                 log=lambda *a: None):
    """Compile the spec and run every session through the real machinery, collecting the
    per-session history the terminal checks read. Returns a result dict:
      {per_round, terminal, history, first_surfaced, cost, n_patterns}"""
    rounds = GEN.build_rounds(spec, rng)
    conn, run_id, _ = build_db(db_path, spec.profile)
    flag_intended, pattern_for = {}, {}
    # {pattern_id: the direction the MODEL asked for}. The recorded direction is computed
    # in code from the flags' deltas, so grading it would grade arithmetic; the grader
    # needs what the model proposed, which lives only in the per-round summary.
    proposed_direction = {}
    history, first_surfaced = [], {}
    per_round, cost = [], 0.0
    try:
        for s, rnd in enumerate(rounds):
            add_round_flags(conn, run_id, rnd["papers"], flag_intended)
            before = {p["id"] for p in DB.get_patterns(conn)}
            open_before = len(DB.get_active_patterns(conn))

            n_flags, candidates, summary, c, memory_shown = run_round(
                conn, client, spec.profile, cluster_model, consolidate_model,
                cluster_prompt, consolidate_prompt, use_cache=use_cache,
                judge_prompt=spec.judge_prompt)
            cost += c
            # What the model was actually given about past patterns. When a recurrence check
            # goes red the first question is always "was that closed pattern even in front of
            # it, and with what evidence?" -- run_round has returned this all along for exactly
            # that purpose, and both callers threw it away, so the question was unanswerable
            # without paying to run the whole scenario again.
            log(f"\n  memory the consolidate step was shown in session {s}:")
            for line in (memory_shown or "(empty -- no history yet)").splitlines():
                log(f"    {line}")
            # WHAT THE HUMAN COULD SEE, not what was written (2026-08-26). A HELD pattern is a
            # real row, so a plain diff over get_patterns would hand held rows to
            # fragmentation, min_purity and no_new_pattern_for -- and a run that held
            # EVERYTHING would start passing checks it should fail while the user was shown
            # nothing. That is the vacuous-pass shape this suite has been bitten by twice, so
            # new_ids is intersected with the active list.
            minted_ids = {p["id"] for p in DB.get_patterns(conn)} - before
            active_now = {p["id"] for p in DB.get_active_patterns(conn)}
            new_ids = minted_ids & active_now
            held_ids = {p["id"] for p in DB.get_held_patterns(conn)}

            pp_now = pattern_intended(conn, flag_intended)
            snap_new = []
            # Held patterns carry a proposed direction too, and direction_not_inverted grades
            # every produced pattern at the end -- so record it for both buckets.
            for cnew in summary["new"] + summary["held"]:
                proposed_direction[cnew["id"]] = (cnew.get("direction_proposed")
                                                  or cnew.get("direction"))
            for cnew in summary["new"]:
                ctr = pp_now.get(cnew["id"], Counter())
                d = dominant_intended(ctr)
                snap_new.append({"id": cnew["id"], "dominant_by_pattern": d, "purity": purity_of(ctr),
                                 "priority": cnew.get("priority"), "direction": cnew.get("direction")})
            # FIRST SURFACED means first VISIBLE, and it is read off the active list rather than
            # off summary["new"]. Those used to be the same thing; since a thin gap can now be
            # minted held at session 0 and promoted later, they are not. Keying on the summary
            # would leave first_surfaced unset for exactly the patterns coalesces_to_one exists
            # to grade, turning a bookkeeping gap into a red that looks like model behaviour.
            for pid in active_now:
                d = dominant_intended(pp_now.get(pid, Counter()))
                if d is not None and d not in first_surfaced:
                    first_surfaced[d] = s
            # `unattached` is the flag count this session was actually handed, and it is the only
            # window onto whether flags are still being ABSORBED as the run goes on. It stays flat
            # when each arriving flag gets attached and climbs when they are being held instead --
            # see pool_drains, which reads it. It was computed here all along and only logged.
            history.append({"session": s, "new": snap_new, "unattached": n_flags,
                            "active_count": len(active_now), "held_count": len(held_ids)})

            per_round += check_round(conn, rnd["expect"], summary, candidates, flag_intended,
                                     new_ids, open_before, pattern_for,
                                     memory_empty=not (memory_shown or "").strip())
            log(f"  session {s}: {n_flags} unattached -> {len(summary['new'])} new, "
                f"{len(summary['merged'])} merged, {len(summary['held'])} held, "
                f"{len(summary['surfaced'])} surfaced, {len(summary['discarded'])} discarded")
            if rnd["then"]:
                apply_actions(conn, spec.profile, rnd["then"], flag_intended, pattern_for, log)

        pp = pattern_intended(conn, flag_intended)
        patterns = [{**p, "direction_proposed": proposed_direction.get(p["id"])}
                    for p in DB.get_patterns(conn)]
        terminal = check_terminal(spec.terminal_expect, pp, patterns, first_surfaced, history,
                                  patterns_by_flag(conn), flag_intended)
        n_patterns = len(patterns)

        # The final provenance table. A terminal check can only report a NUMBER ("2 pattern(s),
        # purity=0.67"), and the first question a red raises is always WHICH patterns and what
        # is in them. Without this the transcript cannot answer that and the only way to find
        # out is to pay for the whole scenario again.
        log("\nfinal patterns, labeled by their intended gap (from provenance, not wording):")
        for p in patterns:
            counter = pp.get(p["id"], Counter())
            if not counter:
                continue
            log(f"  {p['id'][:10]} [{p['status']:<12}] dominant={dominant_intended(counter)} "
                f"purity={purity_of(counter):.2f} flags={dict(counter)}  {p['name']}")
            # The NAME alone is three words and cannot answer "is this one redundant with that
            # one?", which is the question a red here always raises. Print what the pattern
            # actually claims and what edit it is asking for.
            log(f"       direction: {p['direction']}")
            if p.get("description"):
                log(f"       says:      {p['description']}")
            if p.get("suggested_edit"):
                log(f"       asks for:  {p['suggested_edit']}")
    finally:
        conn.close()
        if not keep_db:
            db_path.unlink(missing_ok=True)
    return {"per_round": per_round, "terminal": terminal, "history": history,
            "first_surfaced": first_surfaced, "cost": cost, "n_patterns": n_patterns}


# ---------------------------------------------------------------------------
# Multi-rep: grade with a tolerance vote, not a single flaky run
# ---------------------------------------------------------------------------

def run_reps(spec, reps, pass_frac, client, cluster_model, consolidate_model,
             cluster_prompt, consolidate_prompt, seed=0, use_cache=True,
             log=lambda *a: None):
    """Run the spec `reps` times (same synthetic papers each rep, so the variance measured is
    the LLM's, not the fixture's) and tally each named check. A check is GREEN iff it passes in
    >= pass_frac of reps. Returns (tally, green, total_cost, details) where
    tally[label] = (passes, reps) and details[label] lists why each failing rep failed."""
    tally = defaultdict(lambda: [0, 0])
    details = defaultdict(list)          # label -> what went wrong, per failing rep
    total_cost = 0.0
    for r in range(reps):
        res = run_scenario(spec, random.Random(seed), client, cluster_model,
                           consolidate_model, cluster_prompt, consolidate_prompt,
                           use_cache=use_cache, log=log)
        total_cost += res["cost"]
        # A per-round label can occur once per session; collapse within a rep by AND (every
        # occurrence must pass for the rep to count as a pass), then vote across reps.
        per_label_ok, per_label_detail = {}, {}
        for ok, label, detail in res["per_round"] + res["terminal"]:
            per_label_ok[label] = per_label_ok.get(label, True) and bool(ok)
            if not ok and detail:
                per_label_detail[label] = detail
        for label, ok in per_label_ok.items():
            tally[label][0] += int(ok)
            tally[label][1] += 1
            # Keep WHY a rep failed. Without this a red gate reports only "0/3", which says
            # it broke but not how, and the scenario is expensive to re-run to find out.
            if not ok:
                details[label].append(f"rep {r + 1}: {per_label_detail.get(label, 'failed')}")
        log(f"rep {r + 1}/{reps} done (${res['cost']:.4f})")
    green = all(p >= pass_frac * t for p, t in tally.values()) if tally else False
    return tally, green, total_cost, dict(details)


# ---------------------------------------------------------------------------
# The gate entry point
# ---------------------------------------------------------------------------

def run(ctx, scenario, reps, pass_frac):
    """Run one named scenario `reps` times and return (checks, cost, transcript).

    A check is green only if it passes in at least `pass_frac` of the reps, so its label
    carries the vote (`2/3`) rather than a bare pass or fail -- with a nondeterministic model,
    `2/3` and `0/3` mean very different things and must not look alike."""
    from ..fixtures import behaviors

    # The registry holds constructors, so each run gets a fresh spec whose expectations match
    # its own session count, and one gate cannot mutate the spec the next gate sees.
    spec = behaviors.SCENARIOS[scenario]()
    lines = [f"SCENARIO {spec.name}  |  {spec.n_sessions} sessions x {reps} reps"]

    tally, _green, cost, details = run_reps(
        spec, reps, pass_frac, ctx.client, ctx.cluster_model, ctx.consolidate_model,
        ctx.cluster_prompt, ctx.consolidate_prompt, use_cache=ctx.use_cache,
        log=lines.append)

    checks = []
    for label, (passes, total) in sorted(tally.items()):
        ok = passes >= pass_frac * total
        detail = f"vote {passes}/{total}"
        if not ok and details.get(label):
            detail += " | " + " | ".join(details[label][:3])
        checks.append((ok, label, detail))
    return checks, cost, "\n".join(lines)
