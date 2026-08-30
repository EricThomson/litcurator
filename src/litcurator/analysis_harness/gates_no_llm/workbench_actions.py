"""
No-LLM test for the HUMAN'S DECISION PATH: the profile workbench's Carry / Incorporate /
Reject / Save-edits callbacks, driven as the running app drives them, asserting the
pattern_events rows and the profile-version stamp that come out the other side.

This is the half the centaur architecture rests on and it had no test. Everything else in
this harness grades what the MACHINE produced -- how flags become patterns -- while the
point of the design is that a HUMAN decides what happens to each one, by clicking a button.
`workbench-render` asserted those buttons exist in the DOM; nothing asserted what a click
does. So Incorporate writing the wrong event, Reject silently doing nothing, or a decision
failing to stamp the profile version would have passed all eleven gates.

`machinery.apply_actions` -- the scripted human every paid scenario runs on -- writes
pattern_events DIRECTLY. That was the right call (it keeps the scenarios about the
machinery, not about Dash), and it means the paid gates validate the DATA MODEL of a
decision while stepping around the WIRING that produces it in real use. This gate covers
exactly that gap, in the shape record_stage already uses for the recording step: real
functions, scratch database, deterministic, free.

The callbacks really are plain functions -- Dash's @callback registers and returns the
undecorated function -- so they are called here with the argument shapes the app passes.
Two things are substituted, both of them INPUTS rather than the thing under test:

  ctx                 a stand-in carrying triggered_id and states_list, which is how a
                      pattern-matched callback learns WHICH card was clicked and reads the
                      matching card's inputs. Faking it wrong is the bug this gate hunts,
                      so the shapes below mirror Dash's exactly.
  profile_interface   the active profile TEXT is ours, so the expected stamp is a value we
                      can compute (sha256 of known text) rather than whatever happens to be
                      on disk. Everything else on the module passes through untouched.

What it does NOT cover: Dash's own serialization and argument grouping, and anything that
needs a browser (does the click reach the callback, does the card disappear). That is the
job of a manual click-through, and `workbench-render` still covers construction.

Hermetic: build_db fabricates its own articles, so unlike record_stage and pattern_schema
there is no copy of the live database and no precondition on its contents. It DOES repoint
db_interface.LITCURATOR_DB, because the callbacks call get_connection() with no argument --
the runner snapshots and restores that around every free gate, and so does `_world`.

`_world` is a context manager rather than a setup call plus a `_close` at the end of each
check, and that is load-bearing rather than style. A check that fails leaves its scratch
database OPEN, and on Windows the next check's `build_db` cannot unlink a file SQLite still
holds, so it dies with PermissionError before reaching its own assertion. One genuine red
then reports as eleven, and the report stops saying which thing broke. Found by the negative
control in sandbox/workbench_actions_negative_control/, which is what those are for.

    python -m litcurator.analysis_harness.gates_no_llm.workbench_actions
"""

import contextlib

from litcurator import config, db_interface as DB, profile_interface

from .. import machinery as M

# The profile the SCENARIO ran under, and the DIFFERENT text that is "active" when the human
# clicks Incorporate. They must differ: a decision has to stamp the profile the human is
# looking at NOW, and if the two were the same string, a callback that stamped the run's
# original profile -- or any profile at all -- would score green.
SCENARIO_PROFILE = "I follow systems neuroscience: circuits, computation, and behavior."
ACTIVE_PROFILE = SCENARIO_PROFILE + "\n\nInvertebrate neuroethology counts as systems work."

_ORDER = ("pat-hold", "pat-incorporate", "pat-reject", "pat-promote")


class _Ctx:
    """What dash.ctx gives a pattern-matched callback: the id of the component that fired,
    and the States grouped one list per State() declaration, each entry {"id", "value"}."""

    def __init__(self, triggered_id, states_list):
        self.triggered_id = triggered_id
        self.states_list = states_list


class _ProfileShim:
    """The real profile_interface with one method overridden, so the ACTIVE PROFILE TEXT is
    known here and the expected stamp is computable. Delegating the rest means a callback
    that starts using another part of the module keeps working rather than failing in a way
    that looks like a bug in the callback."""

    def __init__(self, text):
        self._text = text

    def read_active_or_empty(self):
        return self._text

    def __getattr__(self, name):
        return getattr(profile_interface, name)


def _paper(intended, n):
    return {"pmid": f"WB{n:05d}", "intended": intended,
            "title": f"Synthetic paper {n} for {intended}",
            "abstract": f"A synthetic abstract for {intended}, paper {n}.",
            "journal": "Journal of Synthetic Results",
            "judge_score": 0.25, "user_score": 0.75,
            "rationale": f"The judge's wrong reason for {intended}.", "note": ""}


@contextlib.contextmanager
def _world(n_patterns=1, papers_each=2):
    """A scratch database holding n open patterns, each built from its own flags, with the
    workbench imported and pointed at it. Always torn down, including on a failed assertion.

    Yields (conn, wb, pids). `pids` is in get_active_patterns order, which is the order Dash
    lays the cards out in and therefore the order of every pattern-matched list below. That
    order is NOT creation order (get_patterns sorts by latest event), so a check that cares
    which card is which must read the pattern, never assume the index."""
    path = M.scratch_db_path("workbench_actions")
    conn, run_id, _ = M.build_db(path, SCENARIO_PROFILE)

    papers, flag_intended, n = [], {}, 0
    for i in range(n_patterns):
        for _ in range(papers_each):
            n += 1
            papers.append(_paper(f"P{i}", n))
    M.add_round_flags(conn, run_id, papers, flag_intended)

    by_label = {}
    for flag_id, label in flag_intended.items():
        by_label.setdefault(label, []).append(flag_id)
    for i in range(n_patterns):
        DB.create_pattern(conn, f"gap {i} under-scored", "under",
                          description=f"description of gap {i}",
                          suggested_edit=f"suggested edit for gap {i}",
                          flag_ids=by_label[f"P{i}"])

    # Repoint BEFORE importing: the module builds its layout at import time, which reads the
    # database. get_connection resolves LITCURATOR_DB per call, so restoring it below is
    # enough to keep later gates on the real one.
    #
    # The try opens BEFORE the import, not after. Everything the finally undoes is already
    # dirty by this line -- LITCURATOR_DB is repointed and the scratch file exists -- so an
    # import that raises would otherwise skip cleanup entirely and leave the next gate
    # pointed at a deleted database. Guard from the first thing that needs undoing.
    DB.LITCURATOR_DB = path
    wb = saved_ctx = None
    try:
        import litcurator.apps.profile_workbench as wb
        saved_ctx = wb.ctx
        wb.profile_interface = _ProfileShim(ACTIVE_PROFILE)
        yield conn, wb, [p["id"] for p in DB.get_active_patterns(conn)]
    finally:
        if wb is not None:
            # ctx as well as profile_interface. The docstring names both as substitutions and
            # frames restoration as the contract, and leaving a stale _Ctx on the shared
            # module for the rest of the process is how one gate quietly conditions the next.
            wb.profile_interface = profile_interface
            if saved_ctx is not None:
                wb.ctx = saved_ctx
        DB.LITCURATOR_DB = config.LITCURATOR_DB      # the module's original binding
        conn.close()
        path.unlink(missing_ok=True)


def _events(conn, pid):
    return [dict(r) for r in conn.execute(
        "SELECT event, note, profile_id FROM pattern_events WHERE pattern_id = ? "
        "ORDER BY id", (pid,))]


def _status(conn, pid):
    return next(p["status"] for p in DB.get_patterns(conn) if p["id"] == pid)


def _open_ids(conn):
    return {p["id"] for p in DB.get_active_patterns(conn)}


def _click_fate(wb, pids, pid, which, reject_notes=None, tab="active"):
    """Drive cb_pattern_fate the way Dash does: one n_clicks list per Input (one slot per
    card, in layout order, the clicked slot set), plus the reject-note States and the tab.

    `tab` is what the pane re-renders after the click. It matters for pat-promote, which is
    only ever clicked from the Held tab."""
    notes = reject_notes or {}
    note_states = [{"id": {"type": "pat-reject-note", "pid": p}, "value": notes.get(p)}
                   for p in pids]
    wb.ctx = _Ctx({"type": which, "pid": pid}, [note_states])
    clicks = {t: [1 if (p == pid and t == which) else None for p in pids] for t in _ORDER}
    return wb.cb_pattern_fate(clicks["pat-hold"], clicks["pat-incorporate"],
                              clicks["pat-reject"], clicks["pat-promote"],
                              [notes.get(p) for p in pids], tab)


def _click_save(wb, pids, pid, values):
    """Drive cb_pattern_save. values maps pid -> (name, direction, description, suggested)
    for EVERY card on screen, because the callback receives them all and has to pick."""
    fields = ("pat-name", "pat-dir", "pat-desc", "pat-sugg")
    states = [[{"id": {"type": f, "pid": p}, "value": values[p][i]} for p in pids]
              for i, f in enumerate(fields)]
    wb.ctx = _Ctx({"type": "pat-save", "pid": pid}, states)
    clicks = [1 if p == pid else None for p in pids]
    return wb.cb_pattern_save(clicks, *[[values[p][i] for p in pids] for i in range(4)])


# ---------------------------------------------------------------------------
# Incorporate -- the decision that writes into the profile's lineage
# ---------------------------------------------------------------------------

def test_incorporate_stamps_the_currently_active_profile():
    """Incorporate must record WHICH profile version absorbed the pattern, and it must be the
    one active at the click -- that stamp is the only link from a line in the profile back to
    the pattern, the flags and the papers that produced it. A NULL stamp, or the scenario's
    original profile, breaks the chain silently and there is no way to repair it later."""
    with _world() as (conn, wb, pids):
        expected = DB._sha256(ACTIVE_PROFILE)
        scenario_id = DB._sha256(SCENARIO_PROFILE)

        _click_fate(wb, pids, pids[0], "pat-incorporate")

        ev = _events(conn, pids[0])
        assert [e["event"] for e in ev] == ["created", "incorporated"], ev
        stamp = ev[-1]["profile_id"]
        assert stamp is not None, "incorporated with NO profile version -- provenance broken"
        assert stamp != scenario_id, "stamped the scenario's profile, not the ACTIVE one"
        assert stamp == expected, f"stamped {stamp!r}, active profile is {expected!r}"
        assert DB.get_profile(conn, stamp)["content"] == ACTIVE_PROFILE
        print("incorporate: stamps the ACTIVE profile version, registered in profiles")


def test_incorporate_walks_back_to_its_papers():
    """The claim the whole design rests on: profile -> pattern -> flags -> papers, answerable
    after the fact. Asserted through the real button rather than through a hand-written event,
    because a decision recorded against the wrong pattern id would still look well-formed."""
    with _world(papers_each=3) as (conn, wb, pids):
        _click_fate(wb, pids, pids[0], "pat-incorporate")

        stamp = _events(conn, pids[0])[-1]["profile_id"]
        pmids = [r["pmid"] for r in conn.execute("""
            SELECT DISTINCT f.pmid
            FROM pattern_events pe
            JOIN pattern_flags pf ON pf.pattern_id = pe.pattern_id
            JOIN flags f ON f.id = pf.flag_id
            JOIN articles a ON a.pmid = f.pmid
            WHERE pe.event = 'incorporated' AND pe.profile_id = ?
        """, (stamp,))]
        assert sorted(pmids) == ["WB00001", "WB00002", "WB00003"], pmids
        print("incorporate: profile version -> pattern -> flags -> papers finds the 3 papers")


def test_incorporate_closes_the_pattern():
    with _world() as (conn, wb, pids):
        _click_fate(wb, pids, pids[0], "pat-incorporate")
        assert _status(conn, pids[0]) == "incorporated", _status(conn, pids[0])
        assert pids[0] not in _open_ids(conn), \
            "an incorporated pattern is still on the open list -- it will be re-decided forever"
        print("incorporate: pattern closes and drops off the open list")


# ---------------------------------------------------------------------------
# Reject and Carry
# ---------------------------------------------------------------------------

def test_reject_records_the_reason_from_the_right_card():
    """Three cards with different reasons typed in. Rejecting the MIDDLE one must record the
    middle one's text: the callback receives every card's note and picks by pid, which is
    where a pattern-matched bug lives. A silent no-op and a note read off the wrong card
    both look identical from the DOM.

    Three cards and the middle one, not two cards and the last one. With two, the clicked
    card IS the last state, so a lookup returning states[-1] regardless of pid returns the
    right value by accident and the check stays green -- it would only discriminate against
    the states[0] twin. Measured: the [0] mutation reds this check, the [-1] mutation did not,
    until the fixture grew a third card. A check that catches one half of a symmetric bug
    reads as full coverage and is not."""
    with _world(n_patterns=3) as (conn, wb, pids):
        notes = {pids[0]: "reason for the first card",
                 pids[1]: "not a real gap, one-off",
                 pids[2]: "reason for the third card"}
        _click_fate(wb, pids, pids[1], "pat-reject", reject_notes=notes)

        ev = _events(conn, pids[1])
        assert [e["event"] for e in ev] == ["created", "rejected"], ev
        assert ev[-1]["note"] == notes[pids[1]], f"recorded {ev[-1]['note']!r}"
        assert _status(conn, pids[1]) == "rejected"
        for other in (pids[0], pids[2]):
            assert _status(conn, other) == "created", "another card was decided too"
        print("reject: closes the pattern with the reason typed on THAT card")


def test_reject_with_no_reason_stores_null():
    """An empty textbox must land as NULL, not as an empty string: 'rejected, no reason
    given' and 'rejected, reason recorded as blank' are different facts, and a tombstone is
    read months later.

    The event type is asserted first, and that is not padding. This check used to read the
    LAST event's note alone, so with Reject no-op'd it inspected the `created` event -- whose
    note is also None -- and passed VACUOUSLY. A check on the last row of an append-only log
    has to pin WHICH row it is looking at, or a missing write reads as a correct one. Same
    disease round_grader was built for; caught here by the negative control."""
    with _world() as (conn, wb, pids):
        _click_fate(wb, pids, pids[0], "pat-reject", reject_notes={pids[0]: ""})
        last = _events(conn, pids[0])[-1]
        assert last["event"] == "rejected", f"no rejection was written at all: {last}"
        assert last["note"] is None, f"a blank reason was stored as {last['note']!r}"
        print("reject: a blank reason is stored as NULL, not as an empty string")


def test_hold_takes_it_off_the_queue_and_keeps_everything():
    """Hold means 'not yet' -- record the decision, take the queue slot back, lose nothing.

    It REPLACED Carry on 2026-08-30. Carry wrote `carried`, an ACTIVE status, so the pattern
    kept its slot until incorporated or rejected -- and the queue cap only bounds NEW patterns,
    so carried ones piled up uncapped round after round. That is the treadmill this project
    exists to avoid, and from the human's side carry and hold always meant the same thing:
    not writing this into the profile today.

    Four things must all hold, and the middle three are what stop this being a delete: the
    event is logged, the pattern and its provenance survive, it is findable in the Held tab,
    and its flags return to the clustering pool so the taste can be re-decided on evidence."""
    with _world() as (conn, wb, pids):
        before = {f["flag_id"] for f in DB.get_pattern_provenance(conn, pids[0])}
        assert before, "fixture: the pattern must own some flags"

        _click_fate(wb, pids, pids[0], "pat-hold")

        ev = _events(conn, pids[0])
        assert [e["event"] for e in ev] == ["created", "held"], ev
        assert pids[0] not in _open_ids(conn), "Hold must give up the queue slot"
        assert pids[0] in {p["id"] for p in DB.get_held_patterns(conn)}, \
            "a held pattern must be findable in the Held tab, or Hold is a delete"
        assert {f["flag_id"] for f in DB.get_pattern_provenance(conn, pids[0])} == before, \
            "Hold must not detach the pattern's papers"
        pool = {f["id"] for f in DB.get_flags(conn, exclude_attached=True)}
        assert before <= pool, \
            "a held pattern's flags must return to the clustering pool, or it can never grow"
        print("hold: logs the decision, leaves the queue, keeps everything, flags back in pool")


def test_only_the_clicked_pattern_is_decided():
    """Three cards, one Incorporate. The other two must be untouched -- still open, still
    carrying nothing but their creation event."""
    with _world(n_patterns=3) as (conn, wb, pids):
        _click_fate(wb, pids, pids[1], "pat-incorporate")

        assert [e["event"] for e in _events(conn, pids[1])] == ["created", "incorporated"]
        for other in (pids[0], pids[2]):
            assert [e["event"] for e in _events(conn, other)] == ["created"], \
                f"deciding one pattern wrote an event on {other[:12]}"
        assert _open_ids(conn) == {pids[0], pids[2]}
        print("fate: exactly one pattern is decided per click, the rest are untouched")


def test_a_render_with_no_click_writes_nothing():
    """Dash fires a pattern-matched callback when the cards are rebuilt, with every n_clicks
    None. That must write nothing. This is the documented Dash hazard in this codebase --
    a component appearing or n_clicks resetting reads as a click -- and here it would append
    a phantom decision to an append-only log that has no undo."""
    with _world(n_patterns=2) as (conn, wb, pids):
        before = {p: _events(conn, p) for p in pids}

        wb.ctx = _Ctx({"type": "pat-incorporate", "pid": pids[0]},
                      [[{"id": {"type": "pat-reject-note", "pid": p}, "value": None}
                        for p in pids]])
        out = wb.cb_pattern_fate([None, None], [None, None], [None, None], [None, None],
                                 [None, None], "active")

        assert all(o is wb.no_update for o in out), f"a no-click render returned writes: {out}"
        assert {p: _events(conn, p) for p in pids} == before, "a no-click render wrote an event"
        print("fate: a rebuild with no click writes nothing and updates nothing")


def test_events_accumulate_and_the_latest_decision_wins():
    """Hold, then Incorporate, on one pattern. Both events survive in order and the status
    is the later one. Append-only is what makes 'how did this line get into my profile' an
    answerable question, so it is worth pinning through the real buttons and not only
    through add_pattern_event."""
    with _world() as (conn, wb, pids):
        _click_fate(wb, pids, pids[0], "pat-hold")
        _click_fate(wb, pids, pids[0], "pat-hold")
        _click_fate(wb, pids, pids[0], "pat-incorporate")

        ev = _events(conn, pids[0])
        assert [e["event"] for e in ev] == ["created", "held", "held", "incorporated"], ev
        assert _status(conn, pids[0]) == "incorporated"
        assert ev[-1]["profile_id"] == DB._sha256(ACTIVE_PROFILE)
        print("fate: decisions append and never overwrite; status is the latest one")


# ---------------------------------------------------------------------------
# Save edits -- content is a draft, fate is the log
# ---------------------------------------------------------------------------

def test_save_edits_changes_content_and_writes_no_event():
    """Editing a pattern's wording is not a decision about it. If Save wrote to the fate log,
    every reword would look like a fresh event and the history would stop being readable."""
    with _world() as (conn, wb, pids):
        _click_save(wb, pids, pids[0],
                    {pids[0]: ("renamed gap", "over", "reworded description", "reworded edit")})

        p = DB.get_pattern(conn, pids[0])
        assert p["name"] == "renamed gap", p
        assert p["direction"] == "over", p
        assert p["description"] == "reworded description", p
        assert p["suggested_edit"] == "reworded edit", p
        assert [e["event"] for e in _events(conn, pids[0])] == ["created"], \
            "Save edits wrote to the append-only fate log"
        assert pids[0] in _open_ids(conn)
        print("save edits: content changes, the fate log and the open list are untouched")


def test_save_edits_targets_the_clicked_card():
    """Three cards on screen, Save clicked on the MIDDLE one. The callback is handed every
    card's values and matches on pid; getting that wrong would silently write one card's
    wording onto another, which no other gate would notice.

    The middle card for the same reason as the reject check above: with the clicked card at
    either end, a lookup that ignores pid and takes the first or the last state gets the right
    answer by accident half the time. The three values are made distinguishable on every field
    that CAN distinguish three -- since 2026-08-25 direction has only two values, so it cannot,
    and name / description / suggested_edit carry the discrimination. The middle card still
    differs from the one above it on direction, which is what a positional slip would grab."""
    with _world(n_patterns=3) as (conn, wb, pids):
        untouched = {p: DB.get_pattern(conn, p) for p in (pids[0], pids[2])}
        values = {pids[0]: ("first untouched", "under", "desc one", "edit one"),
                  pids[1]: ("second edited", "over", "desc two", "edit two"),
                  pids[2]: ("third untouched", "over", "desc three", "edit three")}
        _click_save(wb, pids, pids[1], values)

        edited = DB.get_pattern(conn, pids[1])
        assert edited["name"] == "second edited", edited
        assert edited["direction"] == "over", edited
        assert edited["description"] == "desc two", edited
        assert edited["suggested_edit"] == "edit two", edited
        for pid, was in untouched.items():
            now = DB.get_pattern(conn, pid)
            assert now["name"] == was["name"], f"an unclicked card was rewritten: {now}"
            assert now["description"] == was["description"], now
        print("save edits: only the clicked card is written, by pid not by position")


def test_promote_moves_a_held_pattern_onto_the_active_list():
    """The human half of the held mechanism, driven through the REAL callback.

    A held pattern is one the consolidate step recorded and deliberately did not show. That is
    only safe if there is somewhere to find it, so the workbench has a Held tab whose one extra
    action is Promote. This drives it end to end: the Held tab lists held patterns and nothing
    else, Promote writes an event that moves the pattern onto the Active list, and the counts
    follow. Without this the whole held design rests on a screen nobody tested."""
    with _world(n_patterns=3) as (conn, wb, pids):
        held_pid = pids[1]                      # the MIDDLE card, never an extreme
        DB.add_pattern_event(conn, held_pid, "held", note="thin for now")

        assert [p["id"] for p in DB.get_held_patterns(conn)] == [held_pid]
        assert held_pid not in _open_ids(conn), "a held pattern must leave the active list"
        assert wb._counts(conn) == (2, 1), wb._counts(conn)

        # The Held tab renders held patterns only, and their card offers Promote.
        held_cards = wb._render_patterns(conn, "held")
        assert len(held_cards) == 1, held_cards
        assert "pat-promote" in str(held_cards), "a held card must offer Promote"
        assert "pat-promote" not in str(wb._render_patterns(conn, "active")),             "an active card must NOT offer Promote"

        out = _click_fate(wb, [held_pid], held_pid, "pat-promote", tab="held")
        assert out[0] is not wb.no_update, "Promote must re-render"
        assert _events(conn, held_pid)[-1]["event"] == "carried", _events(conn, held_pid)
        assert held_pid in _open_ids(conn), "Promote must put it on the Active list"
        assert not DB.get_held_patterns(conn), "and take it off the Held list"
        assert wb._counts(conn) == (3, 0), wb._counts(conn)
        print("promote: a held pattern is listed under Held, and Promote moves it to Active")


CHECKS = [
    test_incorporate_stamps_the_currently_active_profile,
    test_incorporate_walks_back_to_its_papers,
    test_incorporate_closes_the_pattern,
    test_reject_records_the_reason_from_the_right_card,
    test_reject_with_no_reason_stores_null,
    test_hold_takes_it_off_the_queue_and_keeps_everything,
    test_only_the_clicked_pattern_is_decided,
    test_a_render_with_no_click_writes_nothing,
    test_events_accumulate_and_the_latest_decision_wins,
    test_save_edits_changes_content_and_writes_no_event,
    test_save_edits_targets_the_clicked_card,
    test_promote_moves_a_held_pattern_onto_the_active_list,
]


if __name__ == "__main__":
    for check in CHECKS:
        check()
    print("\nALL CHECKS PASSED")
