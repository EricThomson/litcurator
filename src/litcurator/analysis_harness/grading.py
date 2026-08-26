"""
grading.py -- the pure graders. Plain data in, [(ok, label, detail)] out.

check_round grades ONE session against its expectations; check_terminal grades
cross-session behavior once a whole scenario has run. Neither touches a database or a
model, which is what makes them unit-testable without spending anything.

Every check reads the PROVENANCE GRAPH -- which flags ended up attached to which
produced pattern -- never the model's wording, so a nicely worded but wrong result
cannot pass.
"""

from collections import Counter

from litcurator import db_interface as DB

from .machinery import pattern_intended, dominant_intended, purity_of

# The cross-cutting carve-outs that used to live here went with the diagnosis directions
# (2026-08-25) -- see the note in machinery.py for why, and for the rule if one is ever
# wanted again (put it in the fixture, not in a field the model writes).
# for why direction decides whether a count is fragmentation or a second finding.


def _decided(pattern_for, label):
    """The produced patterns the scripted human decided for `label`, as a set.

    An EMPTY set means the decision never resolved -- apply_actions found no open pattern
    covering that gap. Every cross-session check fails loudly on that. It used to pass
    silently: the old code read one direction off `pattern_for[label]`, got None when the
    label was missing, and no real pattern has direction None (the column is NOT NULL), so
    the duplicate list came out empty and a genuine duplicate scored green."""
    return set((pattern_for.get(label) or {}).get("ids") or ())


def check_round(conn, expect, summary, candidates, flag_intended, new_ids,
                open_before, pattern_for):
    """Grade ONE session. `candidates` is the raw consolidate output; every caller passes it
    and no check reads it right now -- the act_now check that did was removed as untestable
    against this fixture, and anything about what the model SAID rather than what got recorded
    would need it back."""
    pp = pattern_intended(conn, flag_intended)
    dominant_by_pattern = {pid: dominant_intended(c) for pid, c in pp.items()}   # produced pattern -> its intended pattern
    merged_ids = {c["id"] for c in summary["merged"]}
    recurred_ids = {c["id"] for c in summary["recurred"]}
    # A pattern minted by the RECOVERY path -- the model gave a merge target that did not exist,
    # so the choice was salvaged as a new pattern rather than dropped. It counts as a duplicate,
    # but the report should say where it came from instead of blaming the model for minting it.
    recovered_ids = {c["id"] for c in summary["new"] if c.get("recovered")}
    direction_of = {p["id"]: p["direction"] for p in DB.get_patterns(conn)}
    active = DB.get_active_patterns(conn)
    out = []

    def chk(label, ok, detail=""):
        out.append((bool(ok), label, detail))

    for t in expect.get("covers", []):
        hits = [p for p, d in dominant_by_pattern.items() if d == t]
        chk(f"coverage: intended {t} has a pattern", hits, f"{len(hits)} pattern(s)")

    # FRAGMENTATION is one gap arriving as several patterns that say the SAME thing. Two
    # patterns over the same papers with OPPOSITE directions are two findings rather than one
    # gap fragmented, so count per direction the same way no_new_pattern_for does. Note this
    # got STRICTER when the diagnosis directions went (2026-08-25): the split now has to be a
    # genuine over-vs-under, and computed_sign forces both to agree when they cite the same
    # flags, so two patterns over one gap can no longer hide behind different labels.
    if "max_produced_per_intended" in expect:
        lim = expect["max_produced_per_intended"]
        for t in expect.get("covers", []):
            mine = [p for p, d in dominant_by_pattern.items() if d == t and p in new_ids]
            per_direction = Counter(direction_of.get(p) for p in mine)
            worst = max(per_direction.values(), default=0)
            detail = f"{len(mine)} pattern(s): {dict(per_direction)}"
            if set(mine) & recovered_ids:
                detail += (f"; {len(set(mine) & recovered_ids)} from a RECOVERED malformed "
                           f"candidate, not a deliberate split")
            chk(f"fragmentation: intended {t} <= {lim} new patterns per diagnosis",
                worst <= lim, detail)

    # EVERY new pattern, no exemption. There used to be one: a `sharpen` or `judge-not-applying`
    # pattern was an observation spanning several tastes, so counting it as impure failed a real
    # finding (one built from 8 A flags, 4 B and 3 D, purity 0.53). Those directions are gone, so
    # every pattern is a claim about one taste and the check applies to all of them.
    if "min_purity" in expect and new_ids:
        worst = min((purity_of(pp.get(p, Counter())) for p in new_ids), default=1.0)
        chk(f"purity: every new pattern >= {expect['min_purity']}",
            worst >= expect["min_purity"],
            f"worst {worst:.2f} over {len(new_ids)} pattern(s)")

    if "new_patterns" in expect:
        lo, hi = expect["new_patterns"]["min"], expect["new_patterns"]["max"]
        n = len(summary["new"])
        chk(f"new-count: new patterns in [{lo},{hi}]", lo <= n <= hi, f"{n}")

    for t in expect.get("new_pattern_for", []):
        hits = [p for p in new_ids if dominant_by_pattern.get(p) == t]
        chk(f"addition: new gap {t} got a pattern", hits, f"{len(hits)}")

    # A DUPLICATE is a new pattern restating a gap already tracked. The other case is a new
    # pattern over the same papers with the OPPOSITE direction, which is a second finding.
    # One paper legitimately supports several patterns, so only the duplicate is a failure,
    # and direction is what tells them apart.
    for t in expect.get("no_new_pattern_for", []):
        decided = _decided(pattern_for, t)
        if not decided:
            chk(f"no-duplicate: {t} spawned no new pattern with the same diagnosis", False,
                f"UNRESOLVED -- no pattern was ever decided for {t}, so a duplicate cannot be "
                f"detected (this used to pass vacuously here)")
            continue
        prior = {direction_of.get(p) for p in decided}
        same = [p for p in new_ids
                if dominant_by_pattern.get(p) == t and direction_of.get(p) in prior]
        others = [p for p in new_ids if dominant_by_pattern.get(p) == t and p not in same]
        detail = f"{len(same)} duplicate(s) vs prior diagnosis {sorted(d or '?' for d in prior)}"
        if set(same) & recovered_ids:
            detail += (f"; {len(set(same) & recovered_ids)} from a RECOVERED malformed candidate "
                       f"(a bad merge target), not a deliberate new pattern")
        if others:
            detail += (f"; {len(others)} other diagnosis(es) allowed "
                       f"({', '.join(sorted({direction_of.get(p) or '?' for p in others}))})")
        chk(f"no-duplicate: {t} spawned no new pattern with the same diagnosis", not same, detail)

    # ANY-of: the return is ONE event, and logging it against one of the gap's closed patterns
    # is the behavior under test. Demanding all of them would demand bookkeeping nobody wants.
    for t in expect.get("recurrence_logged_for", []):
        decided = _decided(pattern_for, t)
        hit = decided & recurred_ids
        chk(f"recurrence: logged on one of {t}'s closed patterns", bool(hit),
            f"{len(hit)}/{len(decided)} closed pattern(s) logged a return" if decided
            else f"UNRESOLVED -- the scripted human never decided a pattern for {t}")

    for t in expect.get("merged_into_existing", []):
        decided = _decided(pattern_for, t)
        hit = decided & merged_ids
        chk(f"merge: {t} attached to one of its existing patterns", bool(hit),
            f"{len(hit)}/{len(decided)} pattern(s) gained flags" if decided
            else f"UNRESOLVED -- the scripted human never decided a pattern for {t}")

    # ALL-of: one reopened pattern is a real defect and must not be masked by a sibling that
    # stayed closed.
    if expect.get("stay_closed"):
        status_by_id = {p["id"]: p["status"] for p in DB.get_patterns(conn)}
    for t in expect.get("stay_closed", []):
        decided = _decided(pattern_for, t)
        action = (pattern_for.get(t) or {}).get("action")
        if action not in ("incorporate", "reject"):
            chk(f"stay-closed: every pattern closed for {t} is still closed", False,
                f"FIXTURE ERROR -- the human's last action on {t} was {action or 'nothing'}, "
                f"which closes nothing")
            continue
        bad = sorted((p, status_by_id.get(p)) for p in decided
                     if status_by_id.get(p) not in ("incorporated", "rejected"))
        chk(f"stay-closed: every pattern closed for {t} is still closed",
            bool(decided) and not bad,
            f"{len(decided)} closed, all still closed" if not bad
            else "; ".join(f"{p[:10]}={st}" for p, st in bad))

    # Bounded, not frozen: an extra pattern or two is a real finding; a pile that climbs every
    # session is the treadmill. So convergence is measured as GROWTH, with a ceiling rather than
    # an equality. There is deliberately no check on the pile's absolute size: how many distinct
    # angles a set of papers supports is not fixed, so any absolute cap lands next to the real
    # output and produces coin-flip reds instead of verdicts.
    if "max_open_pattern_growth" in expect:
        grew = len(active) - open_before
        cap = expect["max_open_pattern_growth"]
        chk(f"convergence: open pile grew by at most {cap}", grew <= cap, f"grew by {grew}")

    return out



def coverage_by_intended_pattern(pp, papers_emitted):
    """Per intended pattern, the produced pattern that RECOVERED it: the one holding the most of its
    flags, with how much of the intended pattern it captured (coverage) and how much of that pattern is
    the intended pattern (purity). Returns {label: (home_pattern_id | None, coverage, purity)}.

    This is INTENDED-PATTERN-centric and deliberately different from check_round's min_purity, which is
    PATTERN-centric (the worst dominant-share over new patterns). They answer different
    questions -- "was this intended pattern recovered" versus "is every produced pattern clean" -- so they are
    two reductions, not one duplicated."""
    patterns_holding = {}                       # label -> [(pid, this_class_count, pattern_total)]
    for pid, counter in pp.items():
        total = sum(counter.values())
        for label, n in counter.items():
            patterns_holding.setdefault(label, []).append((pid, n, total))
    out = {}
    for label, emitted in papers_emitted.items():
        entries = patterns_holding.get(label, [])
        if not entries or not emitted:
            out[label] = (None, 0.0, 0.0)
            continue
        pid, n, total = max(entries, key=lambda e: e[1])
        out[label] = (pid, n / emitted, n / total)
    return out


def check_terminal(expect, pp, patterns, first_surfaced, history,
                   flag_patterns=None, flag_intended=None):
    """Grade the terminal_expect of a scenario from plain data:
      pp             : {pattern_id: Counter(intended_label -> flag_count)}   (FINAL graph)
      patterns       : list of pattern dicts, each with 'id', 'direction', 'status'
      first_surfaced : {intended_label: session_index of its first produced pattern}
      history        : [{'session', 'new':[{'id','dominant_by_pattern','purity','priority','direction'}], ...}]
      flag_patterns  : {flag_id: {pattern_id, ...}}  -- only the dual-nature check needs it
      flag_intended  : {flag_id: label | [labels]}   -- ditto
    Returns [(ok, label, detail)] like check_round. No side effects.

    `history` is read by open_pile_settles, which is the only check that needs the shape of the
    run over time rather than its end state."""
    dominant_by_pattern = {pid: dominant_intended(c) for pid, c in pp.items()}
    purity_by_pattern = {pid: purity_of(c) for pid, c in pp.items()}
    by_id = {p["id"]: p for p in patterns}
    direction_of = {p["id"]: p["direction"] for p in patterns}

    def patterns_for(label):
        return [pid for pid, d in dominant_by_pattern.items() if d == label]

    out = []

    def chk(label, ok, detail=""):
        out.append((bool(ok), label, detail))

    # --- accumulation: a weak trickle coalesces to ONE pattern, in a tolerance band --------
    for spec in expect.get("coalesces_to_one", []):
        L, lo = spec["label"], spec.get("min_session", 0)
        hi, floor = spec.get("max_session", 1_000_000), spec.get("min_purity", 0.6)
        pats, fs = patterns_for(L), first_surfaced.get(L)
        one = len(pats) == 1
        in_band = fs is not None and lo <= fs <= hi
        pure = one and purity_by_pattern[pats[0]] >= floor
        detail = f"{len(pats)} pattern(s), first_surfaced={fs}"
        if one:
            detail += f", purity={purity_by_pattern[pats[0]]:.2f}"
        chk(f"coalesce: {L} -> 1 pattern, surfaced in [{lo},{hi}], purity>={floor}",
            one and in_band and pure, detail)

    # --- coverage survives a grown/noisy pool ----------------------------------------------
    for L in expect.get("intended_patterns_surface", []):
        pats = patterns_for(L)
        chk(f"surfaced: {L} has a pattern", bool(pats), f"{len(pats)} pattern(s)")

    # --- adjacent-but-distinct intended patterns never merge --------------------------------
    # Any pattern drawing >=2 flags from each of two intended patterns has fused them. The
    # cross-cutting exemption is gone with the diagnosis directions. There used to be a second term, `set(patterns_for(l1)) &
    # set(patterns_for(l2))`, meant to catch a pattern belonging to both gaps -- but patterns_for
    # keys on the DOMINANT intended pattern, which is single-valued, so that intersection is
    # empty by construction and the term could never fire. It read like a second safeguard and
    # was doing nothing.
    for pair in expect.get("stay_separate", []):
        l1, l2 = pair
        both = [(pid, counter) for pid, counter in pp.items()
                if counter.get(l1, 0) >= 2 and counter.get(l2, 0) >= 2]
        fused = [pid for pid, _ in both]
        chk(f"separate: {l1} vs {l2} stay distinct", not fused, f"{len(fused)} fused")

    # --- one paper, two tastes: it should land in BOTH patterns, not create a chimera -------
    # The failure this guards is the LLM inventing ONE blended pattern to hold two unrelated
    # tastes that merely co-occurred in a paper. Correct behavior is two patterns with the
    # shared paper attached to each (pattern_flags is many-to-many, so this costs nothing).
    for spec in expect.get("shared_flag_in_both", []):
        l1, l2 = spec["labels"]
        duals = [fid for fid, lbls in (flag_intended or {}).items()
                 if not isinstance(lbls, str) and l1 in lbls and l2 in lbls]
        shared = [fid for fid in duals
                  if {dominant_by_pattern.get(p) for p in (flag_patterns or {}).get(fid, ())} >= {l1, l2}]
        # A pattern holding >=2 of EACH intended pattern is the fused chimera being ruled out.
        # The direction filter that used to exempt cross-cutting reads went with the diagnosis
        # directions (2026-08-25).
        chimeras = [pid for pid, c in pp.items()
                    if c.get(l1, 0) >= 2 and c.get(l2, 0) >= 2]
        chk(f"shared: a {l1}+{l2} paper attaches to a pattern of each, no chimera",
            bool(duals) and bool(shared) and not chimeras,
            f"{len(shared)}/{len(duals)} dual flags in both, {len(chimeras)} chimera(s)")

    # --- a decision that will not stick keeps coming back, and it keeps coming back HERE ----
    # A closed pattern whose gap resurfaces logs a `recurred` mark on that same pattern. Doing
    # it once is the per-round recurrence check; doing it AGAIN, on the same pattern, is what
    # makes the count mean something -- "you rejected this and it has now come back twice" is
    # the signal that the decision was wrong, and it only exists if the returns land in one
    # place.
    #
    # The reduction is MAX over the gap's patterns, never SUM. Two sibling patterns with one
    # return each would sum to two and pass, while describing the exact failure this is meant
    # to catch: the gap fragmented, so no single pattern accumulated any history.
    for spec in expect.get("recurrence_accumulates", []):
        L, need = spec["label"], spec["min_recurrences"]
        counts = {pid: by_id.get(pid, {}).get("recurred_count", 0) for pid in patterns_for(L)}
        best = max(counts.values(), default=0)
        detail = (f"most returns on any one pattern: {best} (need {need}); "
                  + ", ".join(f"{p[:10]}[{by_id.get(p, {}).get('status')}]={n}"
                              for p, n in sorted(counts.items()))
                  if counts else f"NO pattern for {L}, so nothing could have recurred")
        chk(f"accumulates: {L} came back >= {need}x on ONE pattern", best >= need, detail)

    # --- the open pile stops climbing -------------------------------------------------------
    # The per-round convergence check measures growth against a count snapshotted AFTER the
    # previous session's decisions, so it cannot see a slow ratchet: +1 every session passes a
    # per-session cap of 1 forever. This reads the pile's actual trajectory across the tail of
    # the run, decisions and all, which is the only place the treadmill is visible.
    for spec in expect.get("open_pile_settles", []):
        window, cap = spec["over_last_sessions"], spec["max_growth"]
        if not history:
            chk(f"settles: open pile grew <= {cap} over the last {window} sessions", False,
                "NO history recorded -- nothing to measure")
            continue
        tail = history[-(window + 1):]
        grew = tail[-1]["active_count"] - tail[0]["active_count"]
        chk(f"settles: open pile grew <= {cap} over the last {window} sessions", grew <= cap,
            f"{' -> '.join(str(h['active_count']) for h in tail)} = {grew:+d}")

    # --- flags keep being absorbed, not just recognised once ---------------------------------
    # This catches the one failure coalesces_to_one structurally cannot. That check reads the END
    # STATE: one pattern, surfaced in the right window, pure. A machinery that mints a pattern
    # early and then HOLDS every later flag instead of attaching it satisfies all three -- one
    # pure pattern, right window -- while every flag after the second was left on the floor. The
    # taste was recognised once and then ignored for the rest of the run, which is exactly the
    # late drift a long scenario exists to expose.
    #
    # The unattached pool is the tell, and it costs nothing: it is the flag count already fed to
    # each session. Absorbing normally it stays flat (one flag arrives, one gets attached);
    # ignoring flags it climbs monotonically.
    for spec in expect.get("pool_drains", []):
        window, cap = spec["over_last_sessions"], spec["max_growth"]
        pool = [h["unattached"] for h in history if "unattached" in h]
        if not pool:
            chk(f"drains: unattached pool grew <= {cap} over the last {window} sessions", False,
                "NO unattached counts recorded -- nothing to measure")
            continue
        tail = pool[-(window + 1):]
        grew = tail[-1] - tail[0]
        chk(f"drains: unattached pool grew <= {cap} over the last {window} sessions", grew <= cap,
            f"{' -> '.join(str(n) for n in tail)} = {grew:+d}")

    # --- a gap the profile ALREADY states must be RECORDED, not dropped as "covered" --------
    # When the profile says something plainly and the judge ignores it anyway, that is a PROMPT
    # problem rather than a hole in the profile. The machinery's job is still to surface it --
    # and the failure being guarded is consolidate deciding the profile already covers this and
    # letting the flags fall on the floor. The default used to allow judge-not-applying; with
    # that value gone the fixture states which direction it expects, defaulting to every legal
    # one (i.e. recording it AT ALL is the check).
    for spec in expect.get("named_disinterest_not_dropped", []):
        L = spec["label"]
        allowed = tuple(spec.get("directions", DB.DIRECTIONS))
        pats = patterns_for(L)
        right = [pid for pid in pats if by_id.get(pid, {}).get("direction") in allowed]
        got = [by_id.get(pid, {}).get("direction") for pid in pats]
        chk(f"named-disinterest: {L} recorded as {'/'.join(allowed)}, not dropped",
            bool(right),
            f"{len(pats)} pattern(s) with directions {got}" if pats else "NO pattern recorded")

    # --- A TASTE PATTERN MUST NOT BE LABELLED WITH THE OPPOSITE ERROR DIRECTION ------------
    # Added 2026-08-07 after the note-carry gate produced a pattern whose every WORD said "the
    # judge scores these too high, get them out of scope" and whose `direction` said `under`,
    # i.e. the judge scored them too low. Nothing caught it, and nothing could have: direction
    # was read by four other checks (fragmentation counts per direction, min_purity filters on
    # it, no_new_pattern_for tells a duplicate from a second finding by it, and the chimera
    # carve-out keys on it) while NOTHING ever validated it. So a wrong direction did not merely
    # slip through, it quietly made those four compare the wrong things.
    #
    # WHAT THIS ASSERTS, and deliberately no more: never the OPPOSITE direction. `under` on a
    # pool whose flags all have negative deltas is never defensible. Since the diagnosis
    # directions went (2026-08-25) there is no third answer to fall back on, so this is now a
    # straight two-way call.
    #
    # NB the failure this caught was a PROMPT defect, not a model one: the consolidate prompt
    # defines under/over by the PROFILE's state ("the profile is MISSING coverage") while the
    # cluster prompt defines them by the JUDGE's error ("the judge OVER-valued the paper"). On a
    # profile that is silent about the topic those two readings come apart, and the silent case
    # is exactly what "add this to my disinterest list" means. Keep this check even after the
    # prompt is fixed -- it is the regression guard for that fix.
    # WHAT THIS GRADES CHANGED 2026-08-16, and the reason is the whole point. The RECORDED
    # direction is now computed in code from the cited flags' deltas, so asserting it would be
    # grading arithmetic -- tautologically green, the vacuous-pass failure this project has hit
    # twice already. So this grades whether the MODEL agreed with the arithmetic. That is the
    # part that can still be wrong, it is what tells you the prompt's definitions are off, and
    # it stays red until they are fixed.
    #
    # What fails is the model proposing the OPPOSITE direction to the flags that built the
    # pattern.
    for spec in expect.get("direction_not_inverted", []):
        L, taste = spec["label"], spec["taste"]
        opposite = "under" if taste == "over" else "over"
        pats = patterns_for(L)
        if not pats:
            chk(f"direction: {L} not proposed as {opposite} (flags say {taste})", False,
                "NO pattern recorded for this label -- nothing to inspect")
            continue
        # Fall back to the recorded direction when nothing was proposed separately, so this
        # still works on runs predating the computed-sign change.
        proposed = {pid: (by_id.get(pid, {}).get("direction_proposed")
                          or by_id.get(pid, {}).get("direction")) for pid in pats}
        bad = [pid for pid, d in proposed.items() if d == opposite]
        chk(f"direction: {L} not proposed as {opposite} (flags say {taste})", not bad,
            f"{len(pats)} pattern(s) proposed {sorted(proposed.values())}"
            + (f" -- {len(bad)} INVERTED against the flags" if bad else ""))

    # --- THE USER'S OWN WORDS MUST REACH THE PATTERN --------------------------------------
    # The ONLY checks in this file that read produced TEXT rather than the provenance graph,
    # and they are here because the graph cannot see this class of failure at all. A flag's
    # note is frequently the user AUTHORING THE FIX ("put this on my disinterest list"), and
    # consolidate never sees the note -- it sees cluster's prose about it. So the user's
    # wording has to survive two hops, and if it is paraphrased on the way the human gets a
    # blurred restatement of a sentence they had already written precisely, AND the blur is
    # what future rounds match against.
    #
    # HOW THESE STAY SINGLE-RUN HONEST, which is the whole design constraint. Never ask a
    # model whether a paraphrase was faithful -- that is unfalsifiable and would force reps
    # and averaging. Instead the fixture plants DISCRETE tokens in the note and we test for
    # their presence, exactly as the graph checks test set membership: a coined term has no
    # natural synonym, so a run either carried it or did not, with no middle to average.
    #
    # The two are deliberately separate because they fail in different places:
    #   note-term        -- did ANY of the user's language survive into the pattern at all
    #   note-directive   -- did the ACTIONABLE part reach suggested_edit, the field whose
    #                       description is literally "the directive as the researcher would
    #                       author it" and the one the prompt's note-wall paragraph governs
    # A green term with a red directive localises the loss to the note wall; both red means
    # the wording never left the cluster step.
    def _text_of(pid, fields):
        p = by_id.get(pid, {})
        return " ".join(str(p.get(f) or "") for f in fields).lower()

    for spec in expect.get("note_wording_survives", []):
        L, markers = spec["label"], [m.lower() for m in spec["markers"]]
        fields = spec.get("fields", ("name", "description", "suggested_edit"))
        where = "+".join(fields)
        pats = patterns_for(L)
        if not pats:
            chk(f"note-{spec.get('kind', 'term')}: {L} carries {markers[0]!r} into {where}",
                False, "NO pattern recorded for this label -- nothing to inspect")
            continue
        hits = {pid: [m for m in markers if m in _text_of(pid, fields)] for pid in pats}
        best = max(hits.values(), key=len)
        chk(f"note-{spec.get('kind', 'term')}: {L} carries {markers[0]!r} into {where}",
            bool(best),
            f"matched {best}" if best else
            f"none of {markers} in {where} of {len(pats)} pattern(s); "
            f"got {[ _text_of(pid, fields)[:90] for pid in pats ]}")

    return out

