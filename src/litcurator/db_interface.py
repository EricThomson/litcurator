"""
db_interface.py -- SQLite-backed durable state for litcurator (v4, seed-judge).

Append-only provenance, adapted from v1's normalized model (which proved
invaluable): you never lose information by running something again. Every scoring
is a run; re-running adds rows, it never overwrites. So you can always ask "how
did this paper score under the old seed vs the new one?" -- the convergence study.

Tables:
  articles        facts retrieved from PubMed. pub_date_iso = epub-preferred
                  normalized date (sort/display); issue_date_iso = issue-only
                  normalized date (the epub-contamination-free BUCKETING axis).
  human_labels    the 2000 hand labels -- their OWN table, benchmark only.
                  NEVER injected into evaluations as fake 'human' scores (v1's
                  worst mistake: it poisoned the model column forever).
  profiles        content-addressed user-profile snapshots: id = SHA256(content),
                  parent_id chains the lineage, the seed is the root (parent_id NULL).
  prompts         content-addressed judge-prompt snapshots, mirroring profiles --
                  the judge's second input. scoring_runs.judge_prompt_hash == id.
  scoring_runs    one row per scoring session: stage (domain|curation), model,
                  mode, profile_id, judge_prompt_hash, date window, cost,
                  completed_at (NULL = in flight).
  evaluations     one score per (pmid, run_id) -- BOTH stages, unified. Append-only.
  flags           the user's numeric correction on a specific evaluation. Append-only.
  patterns        consolidation ledger: recurring taste-gap patterns (empty until
                  the profile-updater big lift fills them).
  consolidation_runs  one profile-edit round: watermark consumed + doc written.

Authority vs registry: the ACTIVE profile is whatever is in user_profile.md on
disk (read + SHA256 + get_or_create_profile at run time). The DB is a REGISTRY of
what was used, not the authority over the current state.

Most-recent-run-wins (display + domain gate) uses a correlated subquery on
scoring_runs.created_at. Fine at our scale; if this ever hits 100k+ rows per
stage, denormalize an is_latest flag or use a window function instead.

Date handling: pub_date is free-grained text (YYYY / YYYY-MM / YYYY-MM-DD / a
MedlineDate freetext). insert_articles normalizes the best available date into a
sortable pub_date_iso, so a date window selects with a plain BETWEEN.
"""

import hashlib
import json
import random
import re
import sqlite3
import uuid
from datetime import date, datetime, timezone

from litcurator.config import (
    LEVELS_BUCKET_DESCRIPTION,
    LEVELS_BUCKET_NAME,
    LITCURATOR_DB,
    LOCKED_TEST_END,
    LOCKED_TEST_PMIDS_FILE,
    LOCKED_TEST_START,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_ARTICLES = """
CREATE TABLE IF NOT EXISTS articles (
    pmid TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    pages TEXT,
    authors_json TEXT,
    journal TEXT,
    pub_date TEXT,
    pub_date_iso TEXT,
    issue_date_iso TEXT,
    epub_date TEXT,
    doi TEXT,
    pub_types_json TEXT,
    date_added DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

# Presence of a row = the article was sampled and hand-judged. Benchmark only;
# never trains the judge, never enters evaluations.
_CREATE_HUMAN_LABELS = """
CREATE TABLE IF NOT EXISTS human_labels (
    pmid TEXT PRIMARY KEY REFERENCES articles(pmid),
    relevant INTEGER NOT NULL,
    curation_label INTEGER,
    notes TEXT,
    labeled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (relevant IN (0, 1)),
    CHECK (curation_label IS NULL OR curation_label BETWEEN 0 AND 5)
)
"""

# Content-addressed profile snapshots. id = SHA256(content) so identical text
# dedups to one row. parent_id chains the lineage; the seed is the root.
_CREATE_PROFILES = """
CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    parent_id TEXT REFERENCES profiles(id),
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

# Content-addressed judge-prompt snapshots, exactly mirroring profiles: id =
# SHA256(content), parent_id chains the lineage, the seed (the original judge
# prompt) is the root (parent_id NULL). The active prompt is whatever is in
# prompt/judge_prompt.md on disk; at run time it is hashed and registered here.
# scoring_runs.judge_prompt_hash equals this id, so "which prompt produced this
# score" is answerable by JOIN -- the prompt half of the judge's provenance.
# Every hand-authored PROMPT, content-addressed exactly like profiles. `kind` says which
# artifact a row belongs to, so lineage stays per-artifact:
#   judge     the scoring procedure the judge follows (prompt/judge_prompt.md)
#   analysis  cluster + consolidate, how flags become patterns (prompt/analysis_prompt.md)
# Ids are content hashes so two kinds could never collide; kind is what makes "show me this
# artifact's history" a query rather than a guess.
_CREATE_PROMPTS = """
CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    parent_id TEXT REFERENCES prompts(id),
    notes TEXT,
    kind TEXT NOT NULL DEFAULT 'judge',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

# One error_analysis invocation: which prompt and models produced this batch of patterns,
# over which flags, at what cost. The direct mirror of scoring_runs, and for the same reason --
# provenance belongs on the RUN, and each item points at it. Without this, "which prompt made
# this pattern" is unanswerable, which is the one gap the rest of the system does not have.
_CREATE_ANALYSIS_RUNS = """
CREATE TABLE IF NOT EXISTS analysis_runs (
    id TEXT PRIMARY KEY,
    analysis_prompt_id TEXT NOT NULL REFERENCES prompts(id),
    profile_id TEXT REFERENCES profiles(id),
    cluster_model TEXT NOT NULL,
    consolidate_model TEXT NOT NULL,
    date_start TEXT,
    date_end TEXT,
    n_flags INTEGER,
    cost_usd REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

# One row per scoring session. domain runs have profile_id NULL (the domain
# filter is seed-independent); curation runs carry the profile_id used.
_CREATE_SCORING_RUNS = """
CREATE TABLE IF NOT EXISTS scoring_runs (
    id TEXT PRIMARY KEY,                 -- UTC timestamp "YYYYMMDD_HHMMSS_ffffff"
    stage TEXT NOT NULL,                 -- 'domain' | 'curation'
    model TEXT NOT NULL,
    mode TEXT NOT NULL,                  -- 'benchmark' | 'live'  (explicit, never inferred)
    profile_id TEXT REFERENCES profiles(id),
    judge_prompt_hash TEXT,             -- sha256 prompts.id; "which prompt scored this", by JOIN
    date_start TEXT,
    date_end TEXT,
    threshold REAL DEFAULT 0.5,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,               -- NULL = in flight, set = done
    CHECK (stage IN ('domain', 'curation')),
    CHECK (mode IN ('benchmark', 'live'))
)
"""

# One score per article per run, both stages. Domain rows use score + rationale;
# curation rows also set surface_decision + possible_mismatch. Append-only:
# UNIQUE(pmid, run_id) keeps every run's scores side by side.
_CREATE_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid TEXT NOT NULL REFERENCES articles(pmid),
    run_id TEXT NOT NULL REFERENCES scoring_runs(id),
    score REAL NOT NULL,
    rationale TEXT,
    surface_decision TEXT,
    possible_mismatch TEXT,
    evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (pmid, run_id),
    CHECK (score >= 0.0 AND score <= 1.0)
)
"""

# The user's numeric correction on a specific evaluation. Append-only. A flag is
# "handled" by being attached to a pattern (pattern_flags), not by per-flag retirement
# -- so the old ingested_to_profile_id / ingested_at columns are gone (dropped in
# _drop_dead_columns for existing DBs).
_CREATE_FLAGS = """
CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id),
    pmid TEXT NOT NULL REFERENCES articles(pmid),
    judge_score REAL NOT NULL,
    user_score REAL NOT NULL,
    delta REAL NOT NULL,
    note TEXT,
    flagged_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (judge_score >= 0.0 AND judge_score <= 1.0),
    CHECK (user_score >= 0.0 AND user_score <= 1.0)
)
"""

# ---------------------------------------------------------------------------
# Pattern memory -- the subsystem v1 lacked. A PATTERN is a recurring taste gap
# discovered from the papers the user flagged (e.g. "invertebrate neuroethology is
# under-scored"). It is the durable, editable unit of memory. Three tables:
#   patterns        the pattern itself: identity + editable working-draft content
#   pattern_flags   provenance -- which flags produced it (-> the actual papers)
#   pattern_events  the fate, APPEND-ONLY -- created/carried/incorporated/rejected
# Current status = the latest event (most-recent-wins, like evaluations); nothing
# is ever deleted, so a decision (adopt / set aside / throw out) is never lost and
# the log is the legible story of how the taste model evolved. The human authors
# every word of the profile; this memory SURFACES and remembers, it never writes
# profile prose. See the plan starry-brewing-horizon.md.
# ---------------------------------------------------------------------------

# THE ONE PLACE THE DIRECTION VOCABULARY IS WRITTEN DOWN. It lives here, at the bottom
# layer, because everything above imports db_interface already and nothing here imports
# them back -- so the CHECK constraint below, the clamp in error_analysis, the enum in
# the consolidate tool schema and the workbench dropdown all read the same tuple instead of
# restating it.
#
# It was in FIVE places until 2026-08-16, and every copy was individually correct when it
# was written: the CHECK defends the table, the clamp validates a tool argument on the way
# back, the enum steers the model, the dropdown offers the human a choice, TASTE_DIRECTIONS
# answered "which of these are claims about a taste". Four different purposes, four different
# dates -- which is exactly why nobody saw them as copies. NOTE THE GENERAL SHAPE: a DRY
# violation hides when the duplicates serve different purposes. Identical code gets noticed;
# identical VOCABULARY does not. The journal-ratings table (deleted 2026-08-07) wore the same
# disguise -- "judge calibration" and "suggester input" looked like two concerns.
#
# Two of the five failed SILENTLY when they drifted: a value missing from the clamp is
# rewritten to `under`, and a value missing from the enum is simply never proposed.
#
#   over   the judge scores this kind of paper too high
#   under   too low
#
# DIAGNOSIS VALUES REMOVED 2026-08-25. `sharpen` (the profile is vague here) and
# `judge-not-applying` (the profile says it and the judge ignores it) were asking the model
# for an ETIOLOGY, and the inputs cannot support one: consolidate sees the profile but never
# the judge prompt, the papers or the scores, so it can check whether the profile CONTAINS
# text on a topic but not whether that text is strong enough that the judge should have
# followed it. Those come apart -- the neuromorphic case (see "Amplify to lift" in CLAUDE.md)
# had the exception almost verbatim in the profile, the judge recited it and scored 0.35, and
# the fix was still a PROFILE fix. Any reader working from the profile text alone calls that
# judge-not-applying and points at the wrong file. What settles it is an experiment (edit,
# re-run judge_harness, watch the score), so it was never a label to assign by reading.
#
# Direction is what remains, and it is arithmetic: the sign of the cited flags' deltas, which
# `error_analysis.computed_sign` computes and records over whatever the model proposed.
#
# A SECOND REASON, which only showed up on the way out: the harness exempted the two diagnosis
# values from four checks (fragmentation, min_purity, no_new_pattern_for, the chimera
# carve-out) because a cross-cutting pattern has to span tastes to be true. That let the model
# exempt itself from four graders by choosing a label. An allowance like that belongs in the
# fixture, where the test author sets it, never in a field the thing under test writes.
DIRECTIONS = ("over", "under")

# WHICH ARTIFACT IS AT FAULT. A flag measures the JUDGE, and the judge is profile x prompt,
# so the residual belongs to one of them. The question is decidable from what consolidate is
# already shown -- the profile is in its context, and the user's note usually says it
# outright ("Profile clearly says news and views should always be below 0.15, so how was
# this scored at 0.55?" vs "I probably wasn't super clear in my profile").
#   profile  the taste is missing or vague -> the fix is profile prose
#   prompt   the taste is already stated and the judge did not act on it -> fix the
#            scoring procedure; more profile prose provably does not help (measured
#            2026-09-07: 0 of 4 such patterns moved after being incorporated)
# Defaults to profile, which is exactly today's behaviour, so an omitted value is safe.
BLAMES = ("profile", "prompt")

# name/description/suggested_edit are editable working drafts (the human tweaks them in
# place); the fate lives in pattern_events. The CHECK is GENERATED from DIRECTIONS rather
# than restating it -- it is no longer an independent check, and that is correct: independence
# matters for invariants, not for vocabulary. Its job is rejecting anything outside the list,
# which it still does. NB `CREATE TABLE IF NOT EXISTS` never alters an existing table, so
# changing this list still needs a migration for a database that already exists.
_CREATE_PATTERNS = """
CREATE TABLE IF NOT EXISTS patterns (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    direction TEXT NOT NULL,
    description TEXT,
    suggested_edit TEXT,
    analysis_run_id TEXT REFERENCES analysis_runs(id),
    -- The consolidate step's own ordering of the round it came from, 1 = act on this first.
    -- Stored so the workbench can show you the queue in the order the model ranked it, rather
    -- than by most-recent-event (which put whatever you last clicked on top) or by a magnitude
    -- proxy (which is the criterion the ranking prompt was rewritten to stop using).
    rank INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (direction IN (%s))
)
""" % ", ".join(f"'{d}'" for d in DIRECTIONS)

# Provenance join (many-to-many): which flags a pattern was built from. A flag is
# "handled" precisely because it is attached here -- no per-flag retirement needed.
# Joining pattern_flags -> flags -> articles walks a pattern back to its papers.
_CREATE_PATTERN_FLAGS = """
CREATE TABLE IF NOT EXISTS pattern_flags (
    pattern_id TEXT NOT NULL REFERENCES patterns(id),
    flag_id INTEGER NOT NULL REFERENCES flags(id),
    PRIMARY KEY (pattern_id, flag_id)
)
"""

# The pattern's fate, APPEND-ONLY. The four DECISION events (created/carried/
# incorporated/rejected) drive status: the latest DECISION row is the current status
# (get_patterns). 'recurred' is a fifth, NON-decision event -- a closed pattern's
# taste resurfaced in new flags. It is logged (with a note + fresh provenance) but
# never becomes status, so a rejected pattern stays rejected while its flag_count
# grows and it can be surfaced as an alert. profile_id is set on 'incorporated' --
# which profile version absorbed the pattern (so profile -> pattern -> flags ->
# papers is answerable). note carries the reasoning, especially why a pattern was
# rejected.
# THE EVENT VOCABULARY, written down once, for the same reason DIRECTIONS is: the CHECK, the
# status subquery in get_patterns and the migration below all read these instead of restating
# them. A DECISION event becomes the pattern's status (latest wins); an ANNOTATION never does.
#
#   created       minted this round
#   carried       still open, not decided yet -- also what PROMOTES a held pattern, since it
#                 is an active status and status is latest-decision-wins
#   held          recorded but NOT shown to the user (added 2026-08-26; see below)
#   incorporated  folded into the profile, stamped with the profile version
#   rejected      not a real gap
#   recurred      ANNOTATION: a closed pattern's taste came back. Never becomes status, so a
#                 rejected pattern stays rejected while its provenance and counters grow.
#
# WHY `held` EXISTS. Until 2026-08-26 the consolidate step's `hold` choice wrote NOTHING -- no
# row, no flags, no event, only a line in a markdown report no code reads. That is precisely the
# failure error_analysis's own docstring says the redesign killed ("dumped everything it did
# not act on into a free-text line that no code read"), and it meant a recognized-but-not-yet-
# actionable pattern was lost every round and re-derived from scratch. A held pattern is an
# ordinary pattern row with ordinary provenance whose status keeps it off the workbench queue.
DECISION_EVENTS = ("created", "carried", "held", "incorporated", "rejected")
ANNOTATION_EVENTS = ("recurred",)
PATTERN_EVENTS = DECISION_EVENTS + ANNOTATION_EVENTS

# The status groupings, so no caller writes a status tuple by hand.
ACTIVE_STATUSES = ("created", "carried")     # the workbench queue
HELD_STATUSES = ("held",)                    # recorded, deliberately not shown
CLOSED_STATUSES = ("incorporated", "rejected")

_CREATE_PATTERN_EVENTS = """
CREATE TABLE IF NOT EXISTS pattern_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id TEXT NOT NULL REFERENCES patterns(id),
    event TEXT NOT NULL,
    note TEXT,
    profile_id TEXT REFERENCES profiles(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (event IN (%s))
)
""" % ", ".join(f"'{e}'" for e in PATTERN_EVENTS)

_CREATE_STATEMENTS = [
    _CREATE_ARTICLES,
    _CREATE_HUMAN_LABELS,
    _CREATE_PROFILES,
    _CREATE_PROMPTS,
    _CREATE_SCORING_RUNS,
    _CREATE_EVALUATIONS,
    _CREATE_FLAGS,
    _CREATE_ANALYSIS_RUNS,
    _CREATE_PATTERNS,
    _CREATE_PATTERN_FLAGS,
    _CREATE_PATTERN_EVENTS,
]

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_articles_pub_date_iso ON articles(pub_date_iso)",
    "CREATE INDEX IF NOT EXISTS idx_articles_issue_date_iso ON articles(issue_date_iso)",
    "CREATE INDEX IF NOT EXISTS idx_human_labels_relevant ON human_labels(relevant)",
    "CREATE INDEX IF NOT EXISTS idx_human_labels_curation ON human_labels(curation_label)",
    "CREATE INDEX IF NOT EXISTS idx_scoring_runs_stage ON scoring_runs(stage, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_scoring_runs_profile ON scoring_runs(profile_id)",
    "CREATE INDEX IF NOT EXISTS idx_evaluations_pmid ON evaluations(pmid)",
    "CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_flags_evaluation ON flags(evaluation_id)",
    "CREATE INDEX IF NOT EXISTS idx_pattern_flags_pattern ON pattern_flags(pattern_id)",
    "CREATE INDEX IF NOT EXISTS idx_pattern_flags_flag ON pattern_flags(flag_id)",
    "CREATE INDEX IF NOT EXISTS idx_pattern_events_pattern ON pattern_events(pattern_id, created_at)",
]

# Columns added to articles after their initial release; add to an existing DB.
# Idempotent (_migrate swallows "duplicate column name").
_ARTICLE_MIGRATIONS = [
    "ALTER TABLE articles ADD COLUMN pub_date_iso TEXT",
    "ALTER TABLE articles ADD COLUMN pages TEXT",
    # The NLM issue date, normalized issue-ONLY (no epub preference) -- the correct
    # month-bucketing axis for the benchmark/prequential/seal. pub_date_iso is
    # epub-preferred and so mis-buckets ~a third of papers whose epub month differs
    # from their issue month; issue_date_iso fixes that. Both are kept.
    "ALTER TABLE articles ADD COLUMN issue_date_iso TEXT",
]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection(path=None):
    """Open a connection to the litcurator database, creating/migrating as needed."""
    db_path = path or LITCURATOR_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _drop_stale_pattern_tables(conn)   # before CREATE, so the new pattern schema takes effect
    _drop_dead_columns(conn)
    for sql in _CREATE_STATEMENTS:
        conn.execute(sql)
    _migrate_pattern_events(conn)      # after CREATE (table exists), before indexes (recreated below)
    _migrate(conn)
    for sql in _CREATE_INDEXES:
        conn.execute(sql)
    conn.commit()
    _backfill_pub_date_iso(conn)
    _backfill_issue_date_iso(conn)
    return conn


def _drop_stale_pattern_tables(conn):
    """The July readiness pass stubbed a watermark-era `patterns` table (columns
    rounds_seen/evidence_count/fate) and a `consolidation_runs` table. The pattern-
    memory redesign (starry-brewing-horizon.md) replaces both. They were never
    populated, so drop the stale shape here -- ONLY if empty -- so the new
    _CREATE_PATTERNS is created fresh. Idempotent: once migrated the trigger column
    / table is gone and this is a no-op. Refuses to drop a non-empty table (a guard
    against ever silently discarding real data)."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(patterns)").fetchall()]
    if "rounds_seen" in cols:   # present only in the old stub, never the new schema
        n = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        if n:
            raise RuntimeError(f"refusing to drop legacy patterns table with {n} rows")
        conn.execute("DROP TABLE patterns")
    stale = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='consolidation_runs'"
    ).fetchone()
    if stale:
        n = conn.execute("SELECT COUNT(*) FROM consolidation_runs").fetchone()[0]
        if n:
            raise RuntimeError(f"refusing to drop consolidation_runs with {n} rows")
        conn.execute("DROP TABLE consolidation_runs")
    conn.commit()


def _migrate_pattern_events(conn):
    """Rebuild an EXISTING pattern_events table whenever its CHECK no longer matches
    PATTERN_EVENTS. SQLite cannot ALTER a CHECK, so the only way to widen the vocabulary is
    drop-and-recreate preserving every row. Nothing references pattern_events (its FKs are
    outbound to patterns/profiles), so that is safe; the dropped index is recreated by
    _CREATE_INDEXES right after.

    GENERALISED 2026-08-26. It used to test for the literal string "'recurred'", which meant
    the next value added would have silently not migrated -- the table would keep the old
    CHECK and reject the new event at write time, in the record step, mid-consolidation. Now
    it compares every declared value against the stored SQL, so adding one to PATTERN_EVENTS
    is all that adding one takes. Idempotent, and a no-op on a fresh DB where CREATE already
    used the current shape."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='pattern_events'"
    ).fetchone()
    if not row or all(f"'{e}'" in row[0] for e in PATTERN_EVENTS):
        return
    conn.execute(_CREATE_PATTERN_EVENTS.replace(
        "CREATE TABLE IF NOT EXISTS pattern_events", "CREATE TABLE pattern_events__new"))
    conn.execute(
        "INSERT INTO pattern_events__new (id, pattern_id, event, note, profile_id, created_at) "
        "SELECT id, pattern_id, event, note, profile_id, created_at FROM pattern_events")
    conn.execute("DROP TABLE pattern_events")
    conn.execute("ALTER TABLE pattern_events__new RENAME TO pattern_events")
    conn.commit()


def _drop_dead_columns(conn):
    """Drop columns/indexes for removed features (guarded, idempotent). Runs before
    CREATE; a no-op once already migrated (and on a fresh DB, where these tables do not
    exist yet). Needs SQLite >= 3.35 for ALTER TABLE DROP COLUMN.

    - articles.summary: the Haiku review-feed gloss, a display convenience the judge
      never used and the feed shows the abstract anyway; the summarize stage was cut
      2026-07-15.
    - flags.ingested_to_profile_id / ingested_at + their indexes: the per-flag
      retirement the pattern memory replaced. A flag is now "handled" by being attached
      to a pattern (pattern_flags), so these are vestigial. Indexes are dropped FIRST
      because DROP COLUMN fails while a column is used by an index (the partial
      idx_flags_uningested references ingested_to_profile_id)."""
    acols = [r[1] for r in conn.execute("PRAGMA table_info(articles)").fetchall()]
    if "summary" in acols:
        conn.execute("ALTER TABLE articles DROP COLUMN summary")

    conn.execute("DROP INDEX IF EXISTS idx_flags_uningested")
    conn.execute("DROP INDEX IF EXISTS idx_flags_flagged_at")
    fcols = [r[1] for r in conn.execute("PRAGMA table_info(flags)").fetchall()]
    if "ingested_to_profile_id" in fcols:
        conn.execute("ALTER TABLE flags DROP COLUMN ingested_to_profile_id")
    if "ingested_at" in fcols:
        conn.execute("ALTER TABLE flags DROP COLUMN ingested_at")
    if "your_score" in fcols:
        # Rename for clarity: "your" was second-person and ambiguous; the human's score
        # is the user's (pairs with judge_score). SQLite updates the column's CHECK
        # constraint automatically on rename.
        conn.execute("ALTER TABLE flags RENAME COLUMN your_score TO user_score")
    conn.commit()


# Columns added after the tables shipped. ALTER TABLE ADD COLUMN is idempotent-by-exception
# here: a duplicate-column error means the migration already ran.
_LATE_COLUMNS = [
    "ALTER TABLE prompts ADD COLUMN kind TEXT NOT NULL DEFAULT 'judge'",
    "ALTER TABLE patterns ADD COLUMN analysis_run_id TEXT REFERENCES analysis_runs(id)",
    "ALTER TABLE patterns ADD COLUMN rank INTEGER",
    "ALTER TABLE patterns ADD COLUMN blame TEXT NOT NULL DEFAULT 'profile'",
    # WHO WROTE THIS ROW (added 2026-09-04, for undo_error_analysis). A round's writes were
    # identifiable on patterns but not on the attachments and events it added to OTHER rounds'
    # patterns, so "undo the last run" had no clean query. Stamped by the record step; NULL
    # means a human wrote it (workbench decisions), which is exactly what the undo guard
    # refuses to delete.
    "ALTER TABLE pattern_flags ADD COLUMN analysis_run_id TEXT REFERENCES analysis_runs(id)",
    "ALTER TABLE pattern_events ADD COLUMN analysis_run_id TEXT REFERENCES analysis_runs(id)",
    # The symmetric half of profile_id. Without it a prompt-blamed pattern's incorporation
    # stamps a profile version that does not and never will contain the edit -- which is
    # what happened to the Annual Review pattern on 2026-09-01.
    "ALTER TABLE pattern_events ADD COLUMN prompt_id TEXT REFERENCES prompts(id)",
    # WHICH ARTIFACTS THE ROUND ACTUALLY READ (added 2026-09-19). analysis_runs recorded the
    # analysis prompt and the profile, and from 2026-09-18 consolidate ALSO reads the judge
    # prompt, because `blame` is a claim about a specific version of that file ("the prompt
    # already says this and the judge ignored it"). A round that does not name the version it
    # read is a claim that cannot be checked later.
    # NULL IS MEANINGFUL AND IS NEVER BACKFILLED WITH A GUESS: it means the round assigned
    # blame without seeing the judge prompt at all, which is true of every round before
    # 2026-09-18 and is exactly what you want to know when reading their blame values.
    "ALTER TABLE analysis_runs ADD COLUMN judge_prompt_id TEXT REFERENCES prompts(id)",
    # The PICKER chose which of N rounds became these patterns, so it is as causal for "why is
    # this pattern here" as anything else on the row. This is the real column with a real JOIN
    # that the pick prompt was waiting for: until now it was deliberately absent from `prompts`
    # (a kind='pick' row nothing joined to would be the orphan the "what reads this?" test
    # exists to catch), recorded only as a hash stamped into a verdict file on disk.
    "ALTER TABLE analysis_runs ADD COLUMN pick_prompt_id TEXT REFERENCES prompts(id)",
]


def _migrate(conn):
    for sql in _LATE_COLUMNS:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise
    for sql in _ARTICLE_MIGRATIONS:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}


def normalize_pub_date(pub_date, epub_date=None):
    """Best-effort normalize a PubMed date to a sortable YYYY-MM-DD. Prefers
    epub_date, falls back to pub_date; partial dates fill to the first of the
    month/year; returns None if nothing parses."""
    for raw in (epub_date, pub_date):
        iso = _parse_one_date(raw)
        if iso:
            return iso
    return None


def normalize_issue_date(pub_date):
    """Normalize the NLM ISSUE date (pub_date only -- NO epub preference) to a
    sortable YYYY-MM-DD. This is the correct month-bucketing axis: unlike
    normalize_pub_date, it never lets an epub-ahead-of-print date override the
    issue month. Returns None only if pub_date itself does not parse."""
    return _parse_one_date(pub_date)


def _parse_one_date(raw):
    if not raw:
        return None
    raw = raw.strip()
    m = re.match(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$", raw)
    if m:
        return _safe_iso(int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1))
    m = re.match(r"^(\d{4})\s+([A-Za-z]{3})", raw)
    if m:
        return _safe_iso(int(m.group(1)), _MONTHS.get(m.group(2).lower(), 1), 1)
    m = re.match(r"^(\d{4})\b", raw)
    if m:
        return _safe_iso(int(m.group(1)), 1, 1)
    return None


def _safe_iso(year, month, day):
    for mo, da in ((month, day), (month, 1), (1, 1)):
        try:
            return date(year, mo, da).isoformat()
        except ValueError:
            continue
    return None


def _backfill_pub_date_iso(conn):
    rows = conn.execute(
        "SELECT pmid, pub_date, epub_date FROM articles "
        "WHERE pub_date_iso IS NULL AND (pub_date IS NOT NULL OR epub_date IS NOT NULL)"
    ).fetchall()
    updated = 0
    for r in rows:
        iso = normalize_pub_date(r["pub_date"], r["epub_date"])
        if iso:
            conn.execute("UPDATE articles SET pub_date_iso = ? WHERE pmid = ?", (iso, r["pmid"]))
            updated += 1
    if updated:
        conn.commit()
    return updated


def _backfill_issue_date_iso(conn):
    """Fill issue_date_iso from the raw pub_date (issue-only). Idempotent: only
    touches rows where it is still NULL, so it is a no-op on repeat and cheap on
    the common path. Returns count updated."""
    rows = conn.execute(
        "SELECT pmid, pub_date FROM articles "
        "WHERE issue_date_iso IS NULL AND pub_date IS NOT NULL"
    ).fetchall()
    updated = 0
    for r in rows:
        iso = normalize_issue_date(r["pub_date"])
        if iso:
            conn.execute("UPDATE articles SET issue_date_iso = ? WHERE pmid = ?", (iso, r["pmid"]))
            updated += 1
    if updated:
        conn.commit()
    return updated


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------

def insert_articles(conn, articles):
    """Insert article dicts (from retrieve.parse_single_article). INSERT OR IGNORE
    dedups by pmid; normalizes both pub_date_iso (epub-preferred) and issue_date_iso
    (issue-only). Returns count of new rows."""
    inserted = 0
    for a in articles:
        iso = normalize_pub_date(a.get("pub_date"), a.get("epub_date"))
        issue_iso = normalize_issue_date(a.get("pub_date"))
        cur = conn.execute("""
            INSERT OR IGNORE INTO articles
                (pmid, title, abstract, pages, authors_json, journal,
                 pub_date, pub_date_iso, issue_date_iso, epub_date, doi, pub_types_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            a["pubmed_id"], a["title"], a.get("abstract"), a.get("pages"),
            json.dumps(a.get("authors")), a.get("journal"),
            a.get("pub_date"), iso, issue_iso, a.get("epub_date"),
            a.get("doi"), json.dumps(a.get("pub_types")),
        ))
        inserted += cur.rowcount
    conn.commit()
    return inserted


def get_article(conn, pmid):
    row = conn.execute("SELECT * FROM articles WHERE pmid = ?", (pmid,)).fetchone()
    return dict(row) if row else None


def articles_in_range(conn, start=None, end=None):
    """Articles whose ISSUE date falls in the window (issue_date_iso -- the
    epub-contamination-free bucketing axis)."""
    rows = conn.execute("""
        SELECT * FROM articles
        WHERE (? IS NULL OR issue_date_iso >= ?)
          AND (? IS NULL OR issue_date_iso <= ?)
        ORDER BY issue_date_iso
    """, (start, start, end, end)).fetchall()
    return [dict(r) for r in rows]


def set_article_pages(conn, pmid, pages):
    """Store the page range (MedlinePgn) for an article. Pulled lazily over the
    reviewable survivors because PubMed assigns pagination only when the print
    issue appears -- ahead-of-print records (and electronic-only journals) have
    none at retrieval, so this backfills it as it becomes available."""
    conn.execute("UPDATE articles SET pages = ? WHERE pmid = ?", (pages, pmid))
    conn.commit()


# ---------------------------------------------------------------------------
# Profiles (content-addressed; seed = root of the parent_id chain)
# ---------------------------------------------------------------------------

def get_or_create_profile(conn, content, parent_id=None, notes=None):
    """Snapshot a profile. id = SHA256(content); identical content returns the
    existing id (no duplicate row). Returns the profile id."""
    profile_id = _sha256(content)
    existing = conn.execute("SELECT id FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO profiles (id, content, parent_id, notes) VALUES (?, ?, ?, ?)",
            (profile_id, content, parent_id, notes),
        )
        conn.commit()
    return profile_id


def get_profile(conn, profile_id):
    row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    return dict(row) if row else None


def get_seed_profile(conn):
    """The root of the lineage (parent_id IS NULL) -- the immutable seed."""
    row = conn.execute(
        "SELECT * FROM profiles WHERE parent_id IS NULL ORDER BY created_at LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Prompts (content-addressed; seed = root of the parent_id chain) -- mirror of
# profiles, for the judge prompt (the judge's second input).
# ---------------------------------------------------------------------------

def create_analysis_run(conn, analysis_prompt_id, cluster_model, consolidate_model,
                        profile_id=None, date_start=None, date_end=None, n_flags=None,
                        cost_usd=None, judge_prompt_id=None, pick_prompt_id=None):
    """Record one error_analysis invocation and return its id. Unlike scoring runs there is
    no find-or-create: each invocation is its own run even over identical inputs, because two
    runs of a nondeterministic pipeline are two different events and both produced patterns.

    judge_prompt_id and pick_prompt_id are the two artifacts the round READ that are not the
    analysis prompt: the judge prompt consolidate consults to answer `blame`, and the pick
    prompt that chose which round became these patterns. Both may be NULL, and a NULL says
    something true rather than something missing -- see the migration comment."""
    run_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO analysis_runs (id, analysis_prompt_id, profile_id, cluster_model, "
        "consolidate_model, date_start, date_end, n_flags, cost_usd, judge_prompt_id, "
        "pick_prompt_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, analysis_prompt_id, profile_id, cluster_model, consolidate_model,
         date_start, date_end, n_flags, cost_usd, judge_prompt_id, pick_prompt_id),
    )
    conn.commit()
    return run_id


def get_analysis_run(conn, run_id):
    row = conn.execute("SELECT * FROM analysis_runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_or_create_prompt(conn, content, parent_id=None, notes=None, kind="judge"):
    """Snapshot a prompt. id = SHA256(content); identical content returns the existing id (no
    duplicate row). `kind` is 'judge' or 'analysis' -- see the prompts DDL. Returns the id.

    NEVER RAISES ON A MISSING PARENT, and that is the whole point of the guard below.
    parent_id is a FOREIGN KEY and get_connection sets PRAGMA foreign_keys=ON, so a parent that
    is absent makes the INSERT throw. That is exactly what happened: a database snapshot restore
    on 2026-08-30 deleted one prompts row, and every judge-prompt promotion after it threw --
    AFTER set_active had already written the file, so the promotion looked like it worked and
    two active versions went unregistered. One restore silently poisoned the artifact forever.

    A broken chain is bad; a promotion that half-happens is worse. So an unknown parent drops to
    NULL with a loud warning: the new version is always recorded, the break is visible when it
    happens rather than months later, and versions/ still holds what is needed to repair the
    link."""
    prompt_id = _sha256(content)
    existing = conn.execute("SELECT id FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    if not existing:
        if parent_id and not conn.execute(
                "SELECT 1 FROM prompts WHERE id = ?", (parent_id,)).fetchone():
            print(f"WARNING: parent prompt {parent_id[:12]} is not registered, so "
                  f"{prompt_id[:12]} is being recorded as a new root. The lineage is broken "
                  f"here -- usually a database snapshot restore. Repair it from "
                  f"prompt/versions/ once you know which version is missing.")
            parent_id = None
        conn.execute(
            "INSERT INTO prompts (id, content, parent_id, notes, kind) VALUES (?, ?, ?, ?, ?)",
            (prompt_id, content, parent_id, notes, kind),
        )
        conn.commit()
    return prompt_id


def get_prompt(conn, prompt_id):
    row = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    return dict(row) if row else None


def resolve_short_id(conn, table, short_id):
    """The full row id for a 12-char display id, or None if it does not resolve to exactly one.

    Short ids are sha256[:12], the same prefix the content-address itself uses, which is what
    lets a report or a verdict name an artifact without carrying its content. A collision is
    not realistic at 12 hex characters, but an ambiguous prefix returns None rather than
    picking one -- guessing which profile produced a round is worse than admitting we cannot
    tell."""
    if table not in ("profiles", "prompts"):
        raise ValueError(f"resolve_short_id does not serve {table!r}")
    rows = conn.execute(f"SELECT id FROM {table} WHERE id LIKE ?",
                        (f"{short_id}%",)).fetchall()
    return rows[0]["id"] if len(rows) == 1 else None


def get_seed_prompt(conn):
    """The root of the prompt lineage (parent_id IS NULL) -- the original judge prompt."""
    row = conn.execute(
        "SELECT * FROM prompts WHERE parent_id IS NULL ORDER BY created_at LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Scoring runs
# ---------------------------------------------------------------------------

def find_or_create_scoring_run(conn, stage, model, mode, profile_id=None,
                               judge_prompt_hash=None,
                               date_start=None, date_end=None, threshold=0.5):
    """Return the run id for this scoring regime -- (stage, model, mode, profile_id,
    judge_prompt_hash, window) -- creating it if none exists. Reusing an existing run
    is what makes re-invocation RESUME (score only the pmids not yet evaluated in it)
    rather than re-pay. A new profile_id, prompt, OR model mints a NEW run; the old
    run's scores are preserved side by side -- the substrate for the convergence study.
    This never re-scores on its own: an unchanged regime resumes, and a changed regime
    only re-scores the windows you actually choose to re-run. Provenance lives on the
    run, so the run must BE the regime -- model and prompt are part of its identity,
    not just stamped on it.
    """
    row = conn.execute("""
        SELECT id FROM scoring_runs
        WHERE stage = ? AND model = ? AND mode = ?
          AND profile_id IS ?
          AND judge_prompt_hash IS ?
          AND date_start IS ? AND date_end IS ?
        ORDER BY created_at DESC LIMIT 1
    """, (stage, model, mode, profile_id, judge_prompt_hash, date_start, date_end)).fetchone()
    if row:
        return row["id"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    conn.execute("""
        INSERT INTO scoring_runs
            (id, stage, model, mode, profile_id,
             judge_prompt_hash, date_start, date_end, threshold)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, stage, model, mode, profile_id,
          judge_prompt_hash, date_start, date_end, threshold))
    conn.commit()
    return run_id


def complete_scoring_run(conn, run_id, input_tokens=None, output_tokens=None, cost_usd=None):
    """Mark a run done (completed_at) and record its cost. Adds to any existing
    token/cost totals so a resumed run accumulates rather than overwrites."""
    conn.execute("""
        UPDATE scoring_runs
        SET completed_at = ?,
            input_tokens = COALESCE(input_tokens, 0) + COALESCE(?, 0),
            output_tokens = COALESCE(output_tokens, 0) + COALESCE(?, 0),
            cost_usd = COALESCE(cost_usd, 0) + COALESCE(?, 0)
        WHERE id = ?
    """, (_utcnow(), input_tokens, output_tokens, cost_usd, run_id))
    conn.commit()


def get_scoring_run(conn, run_id):
    row = conn.execute("SELECT * FROM scoring_runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Evaluations (append-only, both stages)
# ---------------------------------------------------------------------------

def insert_evaluation(conn, pmid, run_id, score, rationale=None,
                      surface_decision=None, possible_mismatch=None):
    """Append one score for a (pmid, run_id). INSERT OR IGNORE so re-running a
    partially-done run is safe (a pmid already scored in this run is skipped)."""
    conn.execute("""
        INSERT OR IGNORE INTO evaluations
            (pmid, run_id, score, rationale, surface_decision, possible_mismatch)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (pmid, run_id, score, rationale, surface_decision, possible_mismatch))
    conn.commit()


def unevaluated_in_run(conn, run_id, pmids):
    """Of the given pmids, those with no evaluation in this run -- the resume set."""
    done = {r["pmid"] for r in conn.execute(
        "SELECT pmid FROM evaluations WHERE run_id = ?", (run_id,)).fetchall()}
    return [p for p in pmids if p not in done]


def get_articles_passing_domain_filter(conn, start=None, end=None, threshold=0.5):
    """Articles in the window whose MOST RECENT domain evaluation clears the
    threshold (most-recent-run-wins via correlated subquery)."""
    rows = conn.execute("""
        SELECT a.* FROM articles a
        JOIN evaluations e ON e.pmid = a.pmid
        JOIN scoring_runs r ON r.id = e.run_id
        WHERE r.stage = 'domain'
          AND e.score >= ?
          AND r.id = (
              SELECT r2.id FROM evaluations e2
              JOIN scoring_runs r2 ON r2.id = e2.run_id
              WHERE e2.pmid = a.pmid AND r2.stage = 'domain'
              ORDER BY r2.created_at DESC LIMIT 1
          )
          AND (? IS NULL OR a.issue_date_iso >= ?)
          AND (? IS NULL OR a.issue_date_iso <= ?)
        ORDER BY a.issue_date_iso
    """, (threshold, start, start, end, end)).fetchall()
    return [dict(r) for r in rows]


def latest_evaluation(conn, pmid, stage):
    """The most recent evaluation for a pmid at a given stage (most-recent-wins)."""
    row = conn.execute("""
        SELECT e.* FROM evaluations e
        JOIN scoring_runs r ON r.id = e.run_id
        WHERE e.pmid = ? AND r.stage = ?
        ORDER BY r.created_at DESC LIMIT 1
    """, (pmid, stage)).fetchone()
    return dict(row) if row else None


def latest_curation(conn, start=None, end=None):
    """Most-recent curation evaluation per article in the window, joined to the
    article (incl. display fields) and any human label -- what the review feed
    shows. curation_label is NULL for live papers (only benchmark months carry one)."""
    rows = conn.execute("""
        SELECT a.pmid, a.title, a.journal, a.abstract, a.pages, a.pub_date_iso,
               a.doi, a.authors_json,
               e.id AS evaluation_id, e.score, e.surface_decision,
               e.rationale, e.possible_mismatch,
               r.id AS run_id, r.profile_id, r.created_at AS run_created_at,
               hl.curation_label
        FROM articles a
        JOIN evaluations e ON e.pmid = a.pmid
        JOIN scoring_runs r ON r.id = e.run_id
        LEFT JOIN human_labels hl ON hl.pmid = a.pmid
        WHERE r.stage = 'curation'
          AND r.id = (
              SELECT r2.id FROM evaluations e2
              JOIN scoring_runs r2 ON r2.id = e2.run_id
              WHERE e2.pmid = a.pmid AND r2.stage = 'curation'
              ORDER BY r2.created_at DESC LIMIT 1
          )
          AND (? IS NULL OR a.issue_date_iso >= ?)
          AND (? IS NULL OR a.issue_date_iso <= ?)
        ORDER BY e.score DESC
    """, (start, start, end, end)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Flags (append-only numeric corrections)
# ---------------------------------------------------------------------------

def insert_flag(conn, evaluation_id, user_score, note=None, commit=True):
    """Record the user's numeric flag against a specific evaluation. judge_score is
    snapshotted from that evaluation; delta = user_score - judge_score. Returns id.
    commit=False lets add_to_levels_bucket write the flag and its bucket link together."""
    ev = conn.execute("SELECT pmid, score FROM evaluations WHERE id = ?",
                      (evaluation_id,)).fetchone()
    if ev is None:
        raise ValueError(f"no evaluation with id {evaluation_id}")
    judge_score = ev["score"]
    delta = round(user_score - judge_score, 6)
    cur = conn.execute("""
        INSERT INTO flags
            (evaluation_id, pmid, judge_score, user_score, delta, note)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (evaluation_id, ev["pmid"], judge_score, user_score, delta, note))
    if commit:
        conn.commit()
    return cur.lastrowid


def get_latest_flag(conn, pmid):
    """The paper's newest flag as a dict, or None if it was never flagged. Same ordering as
    get_flags, so "newest" means the same thing everywhere."""
    row = conn.execute("SELECT id FROM flags WHERE pmid = ? ORDER BY flagged_at DESC, id DESC "
                       "LIMIT 1", (pmid,)).fetchone()
    return get_flag(conn, row["id"]) if row else None


def get_flag(conn, flag_id):
    row = conn.execute("SELECT * FROM flags WHERE id = ?", (flag_id,)).fetchone()
    return dict(row) if row else None


def delete_flag(conn, pmid):
    """Delete all flag rows for a paper. Flags are the user's own data; a mistaken
    flag should be removable. Deletes all rows so the paper is fully unflagged."""
    conn.execute("DELETE FROM flags WHERE pmid = ?", (pmid,))
    conn.commit()


def get_flags(conn, start=None, end=None, exclude_attached=False):
    """The LATEST flag per paper (flags are append-only; most-recent-wins, so a
    re-flag supersedes without double-counting), joined to its article
    (title/journal/abstract/issue_date_iso) and the evaluation it corrected
    (rationale/surface_decision/possible_mismatch) -- what the suggester and the
    review feed read. start/end filter the article's PUBLICATION window
    (issue_date_iso).

    A flag is "handled" precisely when it is attached to a pattern (pattern_flags).
    exclude_attached=True drops any paper whose latest flag is already attached -- the
    unattached-flags-only pool the suggester clusters, so handled flags never re-cluster
    into duplicate candidates (the primary bloat / idempotency control). The review
    feed keeps the default (exclude_attached=False) so it still shows every flag. If the
    user re-flags a paper AFTER it was patterned, the new latest flag is unattached again
    and correctly re-enters the pool."""
    # ATTACHED IS NOT THE SAME AS HANDLED (2026-08-26). A flag attached to a HELD pattern keeps
    # its provenance but STAYS IN THE POOL, so cluster keeps seeing the paper. That is the whole
    # reason a held pattern can grow: CLUSTER is the only step that reads titles, abstracts and
    # notes, and accumulation works because thin flags pile up until it sees several together.
    # Draining them at first sight would move that job to CONSOLIDATE, which never sees a paper.
    # A flag attached to any NON-held pattern is handled and drops out, exactly as before.
    unattached_only = ("""
        AND NOT EXISTS (
            SELECT 1 FROM pattern_flags pf
            WHERE pf.flag_id = f.id
              AND (SELECT e.event FROM pattern_events e
                   WHERE e.pattern_id = pf.pattern_id AND e.event IN (%(decisions)s)
                   ORDER BY e.created_at DESC, e.id DESC LIMIT 1) NOT IN (%(held)s)
        )""" % {"decisions": ", ".join(f"'{e}'" for e in DECISION_EVENTS),
                "held": ", ".join(f"'{e}'" for e in HELD_STATUSES)}
        if exclude_attached else "")
    rows = conn.execute(f"""
        SELECT f.*, a.title, a.journal, a.abstract, a.issue_date_iso, a.pub_date_iso,
               e.rationale, e.surface_decision, e.possible_mismatch
        FROM flags f
        JOIN articles a ON a.pmid = f.pmid
        JOIN evaluations e ON e.id = f.evaluation_id
        WHERE f.id = (
            SELECT f2.id FROM flags f2 WHERE f2.pmid = f.pmid
            ORDER BY f2.flagged_at DESC, f2.id DESC LIMIT 1
        )
          {unattached_only}
          AND (? IS NULL OR a.issue_date_iso >= ?)
          AND (? IS NULL OR a.issue_date_iso <= ?)
        ORDER BY f.flagged_at
    """, (start, start, end, end)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Pattern memory: patterns + pattern_flags (provenance) + pattern_events (fate).
# The machinery SURFACES and remembers; the human authors every word of the
# profile. Current status = a pattern's latest event. See starry-brewing-horizon.md.
# ---------------------------------------------------------------------------

def create_pattern(conn, name, direction, description=None, suggested_edit=None,
                   flag_ids=(), note=None, analysis_run_id=None, rank=None,
                   blame='profile'):
    """Create a pattern from the flags that produced it, in one transaction: the
    pattern row, its pattern_flags provenance links, and an initial 'created' event.
    Returns the new pattern id. direction in {over, under}.
    note rides the 'created' event -- the suggester passes the consolidate priority +
    rationale here so the first event carries why the pattern was minted.

    analysis_run_id ties the pattern to the run that produced it, and through that to the
    analysis prompt and models -- the same way an evaluation reaches its judge prompt through
    scoring_runs. Nullable only so a test can mint a bare pattern; the real path always sets
    it."""
    pattern_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO patterns (id, name, direction, description, suggested_edit, rank, "
        "analysis_run_id, blame) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (pattern_id, name, direction, description, suggested_edit, rank, analysis_run_id,
         blame if blame in BLAMES else "profile"),
    )
    for fid in dict.fromkeys(flag_ids):   # dedup, preserve order
        conn.execute(
            "INSERT OR IGNORE INTO pattern_flags (pattern_id, flag_id, analysis_run_id) "
            "VALUES (?, ?, ?)",
            (pattern_id, fid, analysis_run_id),
        )
    conn.execute(
        "INSERT INTO pattern_events (pattern_id, event, note, analysis_run_id) "
        "VALUES (?, 'created', ?, ?)",
        (pattern_id, note, analysis_run_id),
    )
    conn.commit()
    return pattern_id


def attach_flags_to_pattern(conn, pattern_id, flag_ids, analysis_run_id=None):
    """Attach additional flags to an EXISTING pattern -- the MERGE primitive: a later
    round's flags attaching to a pattern already tracked (create_pattern only attaches
    at creation). Dedups on the pattern_flags primary key; returns the count NEWLY
    attached. Adds NO fate event on its own -- the caller decides whether new provenance
    warrants a 'carried'/'recurred' event, and skips it when this returns 0, so re-running an
    overlapping window never inflates recurrence.

    RETURNS NEWLY-COVERED PAPERS, NOT NEW ROWS (changed 2026-08-26). A re-flag of a paper this
    pattern already holds is a new flag row, so the row count was non-zero and the caller fired
    a 'carried' / 'recurred' event for a paper the pattern already had -- a recurrence signal
    manufactured by the user editing his own note. 11 of the 20 live flagged papers are
    re-flagged, one of them four times, so this fired routinely. The rows are still written
    (provenance is append-only and the newer flag is the better record); only the return
    changed, and with it what counts as "something new arrived"."""
    before = {r[0] for r in conn.execute(
        "SELECT DISTINCT f.pmid FROM pattern_flags pf JOIN flags f ON f.id = pf.flag_id "
        "WHERE pf.pattern_id = ?", (pattern_id,)).fetchall()}
    for fid in dict.fromkeys(flag_ids):
        conn.execute(
            "INSERT OR IGNORE INTO pattern_flags (pattern_id, flag_id, analysis_run_id) "
            "VALUES (?, ?, ?)",
            (pattern_id, fid, analysis_run_id),
        )
    conn.commit()
    after = {r[0] for r in conn.execute(
        "SELECT DISTINCT f.pmid FROM pattern_flags pf JOIN flags f ON f.id = pf.flag_id "
        "WHERE pf.pattern_id = ?", (pattern_id,)).fetchall()}
    return len(after - before)


def add_pattern_event(conn, pattern_id, event, note=None, profile_id=None,
                      analysis_run_id=None, prompt_id=None):
    """Append a fate event. Append-only. The four DECISION events
    (created|carried|incorporated|rejected) set status = latest decision. 'recurred'
    is a non-decision annotation (a closed pattern's taste came back); it is
    logged but get_patterns ignores it for status, so a rejected pattern stays
    rejected. profile_id is the version that absorbed it, set on 'incorporated'; note
    carries the reasoning (esp. on reject, or the recurrence rationale)."""
    conn.execute(
        "INSERT INTO pattern_events (pattern_id, event, note, profile_id, analysis_run_id, "
        "prompt_id) VALUES (?, ?, ?, ?, ?, ?)",
        (pattern_id, event, note, profile_id, analysis_run_id, prompt_id),
    )
    conn.commit()


def update_pattern_content(conn, pattern_id, name=None, direction=None,
                           description=None, suggested_edit=None, blame=None):
    """Edit a pattern's working-draft content in place (only non-None fields change);
    bumps updated_at. The fate log is untouched -- content is a draft you tweak,
    fate is the append-only memory."""
    sets, params = [], []
    for col, val in (("name", name), ("direction", direction),
                     ("description", description), ("suggested_edit", suggested_edit),
                     ("blame", blame)):
        if val is not None:
            sets.append(f"{col} = ?")
            params.append(val)
    if not sets:
        return
    sets.append("updated_at = ?")
    params.extend([_utcnow(), pattern_id])
    conn.execute(f"UPDATE patterns SET {', '.join(sets)} WHERE id = ?", params)
    conn.commit()


def get_patterns(conn, statuses=None):
    """Patterns with their CURRENT status, the flag count, the recurrence counters,
    and the deciding event's note/profile_id. status = the latest DECISION event
    (created|carried|incorporated|rejected); 'recurred' rows are annotations and never
    become status, so a rejected pattern with later recurrences still reads 'rejected'.
    carried_count / recurred_count are the derived recurrence signals (how many rounds
    it was deferred / how often a closed pattern resurfaced). statuses filters by current
    status (('created','carried') for the active list, ('incorporated','rejected') for
    closed patterns); None returns all. Newest-status-first."""
    rows = conn.execute("""
        SELECT p.*,
               ev.event AS status,
               ev.created_at AS status_at,
               ev.note AS status_note,
               ev.profile_id AS status_profile_id,
               -- PAPERS, not pattern_flags rows. A re-flag of an already-attached paper is a
               -- new flag row, so COUNT(*) counted the same paper twice and inflated every
               -- evidence signal that reads this. 11 of the 20 live flagged papers are
               -- re-flagged, so that was not a corner case.
               (SELECT COUNT(DISTINCT f2.pmid) FROM pattern_flags pf
                    JOIN flags f2 ON f2.id = pf.flag_id
                    WHERE pf.pattern_id = p.id) AS flag_count,
               -- HOW BADLY, alongside how many. `direction` is a sign and `flag_count` is a
               -- head count, so without these a pattern built from three 0.50 deltas and one
               -- built from three 0.08 deltas read identically everywhere -- on the workbench
               -- card, in the memory block the model matches against, and in any ranking.
               --
               -- COMPUTED AND SHOWN, NEVER BRANCHED ON. That is the whole distinction from
               -- DELTA_THRESHOLD (deleted 2026-08-07): a number in code that reshaped the
               -- evidence before anyone saw it. Magnitude is continuous, so it belongs to
               -- judgment -- the model's and the user's -- and both need to be able to see it.
               -- `total_delta` is the quantity the analysis prompt already describes in prose:
               -- "a steady ~0.1 bias across many papers ... can be worth as much as one big
               -- delta on a lone paper". Five flags at 0.1 and one at 0.5 come out equal,
               -- which is exactly what that sentence claims.
               (SELECT ROUND(AVG(ABS(f3.delta)), 3) FROM pattern_flags pf
                    JOIN flags f3 ON f3.id = pf.flag_id
                    WHERE pf.pattern_id = p.id) AS mean_abs_delta,
               (SELECT ROUND(MAX(ABS(f4.delta)), 3) FROM pattern_flags pf
                    JOIN flags f4 ON f4.id = pf.flag_id
                    WHERE pf.pattern_id = p.id) AS max_abs_delta,
               (SELECT ROUND(SUM(ABS(f5.delta)), 3) FROM pattern_flags pf
                    JOIN flags f5 ON f5.id = pf.flag_id
                    WHERE pf.pattern_id = p.id) AS total_delta,
               (SELECT COUNT(*) FROM pattern_events c WHERE c.pattern_id = p.id
                    AND c.event = 'carried') AS carried_count,
               (SELECT COUNT(*) FROM pattern_events r WHERE r.pattern_id = p.id
                    AND r.event = 'recurred') AS recurred_count
        FROM patterns p
        JOIN pattern_events ev ON ev.id = (
            SELECT e2.id FROM pattern_events e2 WHERE e2.pattern_id = p.id
            AND e2.event IN (%(decisions)s)
            ORDER BY e2.created_at DESC, e2.id DESC LIMIT 1
        )
        -- id breaks the tie: created_at is second-resolution, and every pattern minted in one
        -- consolidation pass shares a timestamp, so without this their order is arbitrary and
        -- changes between runs. Same lesson get_closed_recurrences already records.
        -- BY THE MODEL'S OWN RANK, 1 first. Unranked patterns (pre-2026-08-30, or anything
        -- created outside a consolidation) sort last and keep the old recency order among
        -- themselves. Recency remains the tiebreak, which still matters for patterns minted in
        -- one pass: they share a timestamp, so ev.id keeps their order stable between runs.
        ORDER BY (p.rank IS NULL), p.rank ASC, ev.created_at DESC, ev.id DESC
    """ % {"decisions": ", ".join(f"'{e}'" for e in DECISION_EVENTS)}).fetchall()
    result = [dict(r) for r in rows]
    if statuses is not None:
        keep = set(statuses)
        result = [r for r in result if r["status"] in keep]
    return result


def get_active_patterns(conn):
    """The workbench queue: patterns whose latest decision is 'created' or 'carried' --
    the ones the user is shown and still has to decide. HELD patterns are deliberately
    excluded; they are real and recorded but not yet worth the user's attention.

    The LEVELS BUCKET is excluded too: it is stored as a pattern row but is not a pattern (see
    config.LEVELS_BUCKET_NAME). Leaving it out HERE is what keeps it out of error_analysis's
    list of existing patterns, the workbench cards and the workbench count all at once."""
    return [p for p in get_patterns(conn, statuses=ACTIVE_STATUSES)
            if p["name"] != LEVELS_BUCKET_NAME]


def get_levels_bucket(conn):
    """The Levels Bucket's row plus how many papers are in it, or None before the first click.
    The one reader allowed to see it, since get_active_patterns leaves it out on purpose."""
    row = conn.execute(
        "SELECT p.*, COUNT(DISTINCT f.pmid) AS paper_count FROM patterns p "
        "LEFT JOIN pattern_flags pf ON pf.pattern_id = p.id "
        "LEFT JOIN flags f ON f.id = pf.flag_id "
        "WHERE p.name = ? GROUP BY p.id ORDER BY p.rowid DESC LIMIT 1",
        (LEVELS_BUCKET_NAME,)).fetchone()
    return dict(row) if row else None


def get_held_patterns(conn):
    """Recorded but NOT shown: patterns the consolidate step judged real and not yet
    actionable. They stay in the pattern memory (so the model can recognise the gap when it
    returns) and their flags stay in the clustering pool (so CLUSTER keeps accumulating
    evidence for them). A later merge that brings a genuinely new paper writes 'carried',
    which is an active status, and the pattern surfaces.

    Most-accumulated first, which is what makes the workbench's Held tab a ranked
    what-is-nearly-ready list rather than a log."""
    held = get_patterns(conn, statuses=HELD_STATUSES)
    # By EVIDENCE WEIGHT, not head count. Sorting on flag_count put a three-flag trivial
    # pattern above a one-flag severe one -- the volume criterion this project spent a while
    # removing from the definition of a pattern, reintroduced in a sort. total_delta combines
    # breadth and depth the way the analysis prompt already describes them.
    held.sort(key=lambda p: (p.get("total_delta") or 0.0, p.get("flag_count", 0)), reverse=True)
    return held


def get_pattern(conn, pattern_id):
    row = conn.execute("SELECT * FROM patterns WHERE id = ?", (pattern_id,)).fetchone()
    return dict(row) if row else None


def get_pattern_events(conn, pattern_id):
    """A pattern's full fate log, oldest-first -- the story of how it was handled."""
    rows = conn.execute(
        "SELECT * FROM pattern_events WHERE pattern_id = ? ORDER BY created_at, id",
        (pattern_id,)).fetchall()
    return [dict(r) for r in rows]


def get_pattern_provenance(conn, pattern_id):
    """The flags a pattern was built from, joined to their articles -- the papers
    behind the pattern (pattern_flags -> flags -> articles). Largest |delta| first."""
    rows = conn.execute("""
        SELECT f.id AS flag_id, f.pmid, f.judge_score, f.user_score, f.delta, f.note,
               f.flagged_at, a.title, a.journal, a.issue_date_iso
        FROM pattern_flags pf
        JOIN flags f ON f.id = pf.flag_id
        JOIN articles a ON a.pmid = f.pmid
        WHERE pf.pattern_id = ?
        ORDER BY ABS(f.delta) DESC, f.id
    """, (pattern_id,)).fetchall()
    return [dict(r) for r in rows]


def get_pattern_examples(conn, pattern_ids, limit=3):
    """The top `limit` example papers behind EACH of `pattern_ids`, largest |delta| first --
    the batched form of get_pattern_provenance. Returns {pattern_id: [{title, journal, pmid,
    delta}, ...]} with every requested id present, even when it has no papers.

    Batched not for speed (this is local sqlite) but so the caller that renders the pattern
    memory can stay a pure formatter of plain data, which is what makes it testable. Feeding a
    connection into a formatter to do a query per pattern would give that up.

    Ordering matters more than it looks: without the f.id tiebreak, two flags with equal
    |delta| order arbitrarily and the memory block shown to the model changes between runs for
    no reason -- the same second-resolution hazard get_closed_recurrences documents."""
    if not pattern_ids:
        return {}
    ids = list(dict.fromkeys(pattern_ids))
    marks = ",".join("?" * len(ids))
    rows = conn.execute(f"""
        SELECT pattern_id, title, journal, pmid, delta FROM (
            SELECT pattern_id, title, journal, pmid, delta,
                   ROW_NUMBER() OVER (PARTITION BY pattern_id
                                      ORDER BY ABS(delta) DESC, fid) AS rn
            FROM (
                -- One row per PAPER first: a re-flagged paper has several flag rows attached,
                -- and without this it could fill two of the three example slots with itself.
                SELECT pf.pattern_id AS pattern_id, a.title AS title, a.journal AS journal,
                       f.pmid AS pmid, f.delta AS delta, f.id AS fid,
                       ROW_NUMBER() OVER (PARTITION BY pf.pattern_id, f.pmid
                                          ORDER BY ABS(f.delta) DESC, f.id) AS prn
                FROM pattern_flags pf
                JOIN flags f ON f.id = pf.flag_id
                JOIN articles a ON a.pmid = f.pmid
                WHERE pf.pattern_id IN ({marks})
            ) WHERE prn = 1
        ) WHERE rn <= ?
        ORDER BY pattern_id, rn
    """, (*ids, limit)).fetchall()
    out = {pid: [] for pid in ids}
    for r in rows:
        out[r["pattern_id"]].append(dict(r))
    return out


def get_post_closure_papers(conn, pattern_id):
    """The papers attached to a CLOSED pattern AFTER the decision that closed it.

    A rising recurrence count is ambiguous three ways and the count alone cannot separate them:
    an INCORPORATED pattern coming back means the profile edit did not take (the highest-value
    signal this system produces); a REJECTED one coming back means the rejection was probably
    wrong; but EITHER can instead be an ATTRACTOR -- a broad or vaguely-named closed pattern
    collecting false merges, which by count looks identical to both. The disambiguator is
    whether the new papers RESEMBLE the old ones, so the papers are what has to be shown.

    Dated by the analysis_run that attached the flag rather than by the flag's own timestamp:
    pattern_flags carries analysis_run_id (2026-09-04) and a flag can be re-flagged long after
    it was first attached, so flagged_at would misdate an old paper as a fresh return. A human
    attachment carries no run id -- the same absence undo keys on -- and is treated as pre-
    closure, because a human who attaches a paper by hand is not reporting a recurrence."""
    decision = conn.execute("""
        SELECT created_at FROM pattern_events WHERE pattern_id = ?
        AND event IN ('created', 'carried', 'incorporated', 'rejected')
        ORDER BY created_at DESC, id DESC LIMIT 1
    """, (pattern_id,)).fetchone()
    if not decision:
        return []
    rows = conn.execute("""
        SELECT f.delta, f.judge_score, f.user_score, f.note, a.title, a.journal
        FROM pattern_flags pf
        JOIN flags f          ON f.id = pf.flag_id
        JOIN articles a       ON a.pmid = f.pmid
        JOIN analysis_runs r  ON r.id = pf.analysis_run_id
        WHERE pf.pattern_id = ? AND r.created_at > ?
        ORDER BY ABS(f.delta) DESC
    """, (pattern_id, decision["created_at"])).fetchall()
    return [dict(r) for r in rows]


def get_closed_recurrences(conn):
    """Closed patterns (incorporated or rejected) whose gap RESURFACED -- a
    'recurred' event that came AFTER the current deciding event. These are alerts for
    the human ('you closed this, but the flags brought it back'), NOT active patterns:
    they stay off the active list. Returns each pattern row (as get_patterns) with
    recurrence_count + last_recurred_at, most-recently-resurfaced first. Ordering is by
    event id, not timestamp -- timestamps are second-resolution and a decision + its
    recurrence can share one, whereas ids are monotonic."""
    out = []
    for p in get_patterns(conn, statuses=("incorporated", "rejected")):
        decision = conn.execute("""
            SELECT id FROM pattern_events WHERE pattern_id = ?
            AND event IN ('created', 'carried', 'incorporated', 'rejected')
            ORDER BY created_at DESC, id DESC LIMIT 1
        """, (p["id"],)).fetchone()
        row = conn.execute("""
            SELECT COUNT(*) AS n, MAX(created_at) AS last_at
            FROM pattern_events
            WHERE pattern_id = ? AND event = 'recurred' AND id > ?
        """, (p["id"], decision["id"])).fetchone()
        if row["n"]:
            p = dict(p)
            p["recurrence_count"] = row["n"]
            p["last_recurred_at"] = row["last_at"]
            out.append(p)
    out.sort(key=lambda r: r["last_recurred_at"] or "", reverse=True)
    return out


# ---------------------------------------------------------------------------
# The Levels Bucket (see config.LEVELS_BUCKET_NAME)
# ---------------------------------------------------------------------------

def levels_bucket_pmids(conn):
    """Every paper in the Levels Bucket, whichever version of its flag is linked."""
    return {r["pmid"] for r in conn.execute(
        "SELECT DISTINCT f.pmid FROM pattern_flags pf "
        "JOIN flags f ON f.id = pf.flag_id "
        "JOIN patterns p ON p.id = pf.pattern_id WHERE p.name = ?", (LEVELS_BUCKET_NAME,))}


def add_to_levels_bucket(conn, evaluation_id, user_score, note=None):
    """Save the user's score and note STRAIGHT INTO the Levels Bucket. The flag and its
    link to the bucket are committed together, so the paper is never an ordinary flag in
    error_analysis's pool, not even for a moment. Creates the bucket on first use; adding a
    paper already in it saves a newer version there. No analysis_run_id (the human
    signature), so undo_error_analysis cannot touch it. Returns (flag_id, paper_count)."""
    flag_id = insert_flag(conn, evaluation_id, user_score, note, commit=False)
    bucket = conn.execute("SELECT id FROM patterns WHERE name = ? ORDER BY rowid DESC LIMIT 1",
                          (LEVELS_BUCKET_NAME,)).fetchone()
    if bucket is None:
        # direction is a placeholder the schema requires; nothing reads it for the bucket.
        create_pattern(conn, name=LEVELS_BUCKET_NAME, direction="over",
                       description=LEVELS_BUCKET_DESCRIPTION, flag_ids=[flag_id],
                       note="created from the review feed")
    else:
        attach_flags_to_pattern(conn, bucket["id"], [flag_id])
    return flag_id, len(levels_bucket_pmids(conn))


def remove_from_levels_bucket(conn, pmid):
    """Take a paper back out, for mis-clicks and regrets. Deletes only the bucket's links to
    this paper: every score and note stays in `flags`, and the newest becomes an ordinary
    flag in error_analysis's pool again. Nothing else reads these links. Returns the count."""
    conn.execute("DELETE FROM pattern_flags "
                 "WHERE pattern_id IN (SELECT id FROM patterns WHERE name = ?) "
                 "AND flag_id IN (SELECT id FROM flags WHERE pmid = ?)",
                 (LEVELS_BUCKET_NAME, pmid))
    conn.commit()
    return len(levels_bucket_pmids(conn))


def save_flag(conn, evaluation_id, user_score, note=None):
    """The review feed's Save. A paper already in the Levels Bucket stays there: its new
    version goes into the bucket, so re-saving can never leak it back into error_analysis's
    pool. Returns (flag_id, in_bucket)."""
    ev = conn.execute("SELECT pmid FROM evaluations WHERE id = ?", (evaluation_id,)).fetchone()
    if ev is not None and ev["pmid"] in levels_bucket_pmids(conn):
        flag_id, _count = add_to_levels_bucket(conn, evaluation_id, user_score, note)
        return flag_id, True
    return insert_flag(conn, evaluation_id, user_score, note), False


# ---------------------------------------------------------------------------
# Undoing an analysis run (undo_error_analysis)
# ---------------------------------------------------------------------------

def latest_analysis_run(conn):
    """The most recent analysis_runs row, or None. The ONLY run undo may target: the latest
    run is the only one guaranteed to have nothing built on top of it, so undoing it never
    orphans a later round's merges. Older runs are reachable by undoing repeatedly (LIFO)."""
    # rowid, not created_at: the timestamp has second resolution, so two runs in the same
    # second would tie and fall to id -- random hex, a coin flip. The undo-stage gate caught
    # exactly that on its first run. rowid is insertion order, which is what "latest" means.
    return conn.execute(
        "SELECT * FROM analysis_runs ORDER BY rowid DESC LIMIT 1").fetchone()


def analysis_run_manifest(conn, run_id):
    """Everything the run wrote, plus what blocks undoing it. Pure read; the caller shows
    this before delete_analysis_run does anything.

    blockers: human work sitting on the run's minted patterns -- a decision or edit event
    with NO run stamp (the workbench writes unstamped, the record step always stamps), or
    hand-edited wording (update_pattern_content bumps updated_at). Undo deletes the round's
    own writes; it must never eat a human's."""
    minted = conn.execute(
        "SELECT id, name, direction FROM patterns WHERE analysis_run_id = ?",
        (run_id,)).fetchall()
    minted_ids = [p["id"] for p in minted]
    ph = ",".join("?" for _ in minted_ids) or "''"

    foreign_attaches = conn.execute(f"""
        SELECT pf.pattern_id, p.name, f.pmid FROM pattern_flags pf
        JOIN patterns p ON p.id = pf.pattern_id JOIN flags f ON f.id = pf.flag_id
        WHERE pf.analysis_run_id = ? AND pf.pattern_id NOT IN ({ph})
    """, [run_id, *minted_ids]).fetchall()
    foreign_events = conn.execute(f"""
        SELECT pe.pattern_id, p.name, pe.event FROM pattern_events pe
        JOIN patterns p ON p.id = pe.pattern_id
        WHERE pe.analysis_run_id = ? AND pe.pattern_id NOT IN ({ph})
    """, [run_id, *minted_ids]).fetchall()

    blockers = []
    if minted_ids:
        for r in conn.execute(f"""
            SELECT pe.pattern_id, p.name, pe.event FROM pattern_events pe
            JOIN patterns p ON p.id = pe.pattern_id
            WHERE pe.pattern_id IN ({ph}) AND pe.analysis_run_id IS NULL
        """, minted_ids).fetchall():
            blockers.append(f"pattern '{r['name']}' has a human event: {r['event']}")
        # Both timestamps default to CURRENT_TIMESTAMP in the same INSERT, so they are equal
        # at birth; only update_pattern_content (the workbench Save) moves updated_at.
        for r in conn.execute(f"""
            SELECT name FROM patterns
            WHERE id IN ({ph}) AND updated_at != created_at
        """, minted_ids).fetchall():
            blockers.append(f"pattern '{r['name']}' has hand-edited wording")
    return {"run": conn.execute("SELECT * FROM analysis_runs WHERE id = ?",
                                (run_id,)).fetchone(),
            "minted": minted, "foreign_attaches": foreign_attaches,
            "foreign_events": foreign_events, "blockers": blockers}


def delete_analysis_run(conn, run_id):
    """Delete everything one analysis run wrote -- its minted patterns with their provenance
    and events, the attachments its merges added to other rounds' patterns, the events it
    fired on them, and the run row itself. Nothing else; flags are never touched, so every
    flag the run had attached returns to the unattached pool.

    The pattern layer is DERIVED state -- flags are the ground truth -- so this deletes a
    derivation, not a measurement; a re-run rebuilds anything good from the same flags. The
    caller checks analysis_run_manifest()['blockers'] first; this refuses on its own too,
    so no code path can eat human work. Returns {patterns, attachments, events} deleted."""
    manifest = analysis_run_manifest(conn, run_id)
    if manifest["blockers"]:
        raise ValueError("refusing to undo: " + "; ".join(manifest["blockers"]))
    minted_ids = [p["id"] for p in manifest["minted"]]
    ph = ",".join("?" for _ in minted_ids) or "''"
    counts = {}
    counts["events"] = conn.execute(
        f"DELETE FROM pattern_events WHERE analysis_run_id = ? "
        f"OR pattern_id IN ({ph})", [run_id, *minted_ids]).rowcount
    counts["attachments"] = conn.execute(
        f"DELETE FROM pattern_flags WHERE analysis_run_id = ? "
        f"OR pattern_id IN ({ph})", [run_id, *minted_ids]).rowcount
    counts["patterns"] = conn.execute(
        "DELETE FROM patterns WHERE analysis_run_id = ?", (run_id,)).rowcount
    conn.execute("DELETE FROM analysis_runs WHERE id = ?", (run_id,))
    conn.commit()
    return counts


# ---------------------------------------------------------------------------
# Human labels (benchmark only -- their own table, never evaluations)
# ---------------------------------------------------------------------------

# NB there is deliberately no insert_human_label() here. One existed, with zero callers, and
# it was INSERT OR REPLACE over all four columns with curation_label defaulting to None -- so
# any call passing only (pmid, relevant) would have silently blanked an existing curation
# rating. Aimed at the 2000-label benchmark table, which is hand-made and unrecoverable.
# The two live writers below are column-scoped and safe: set_relevance_label upserts ONLY
# relevant (ON CONFLICT DO UPDATE), set_curation_label updates ONLY curation_label.


def get_human_label(conn, pmid):
    row = conn.execute("SELECT * FROM human_labels WHERE pmid = ?", (pmid,)).fetchone()
    return dict(row) if row else None


def labeled_articles(conn, start=None, end=None, relevant=1, final_test=False):
    """Articles with a human label in the window, joined to their label. Used by
    benchmark mode (the judge scores these directly -- labels never become
    evaluations). relevant=1 restricts to the relevance-gated set; None for all.

    The frozen locked-test pmid set (November) is SUBTRACTED unless final_test=True
    -- the development pool is "every labeled pmid MINUS the locked set", enforced
    here at the data layer so any benchmark/analysis caller is held-out-safe by
    construction, not by remembering to pass dev-month windows. Set difference on a
    frozen pmid list is convention-proof: it catches true-November papers even when
    the epub-ahead-of-print artifact mis-buckets their pub_date_iso into October."""
    rows = conn.execute(f"""
        SELECT a.*, hl.relevant, hl.curation_label
        FROM articles a
        JOIN human_labels hl ON hl.pmid = a.pmid
        WHERE ({'hl.relevant = ?' if relevant is not None else '1 = 1'})
          AND (? IS NULL OR a.issue_date_iso >= ?)
          AND (? IS NULL OR a.issue_date_iso <= ?)
        ORDER BY a.issue_date_iso
    """, (([relevant] if relevant is not None else []) + [start, start, end, end])
    ).fetchall()
    result = [dict(r) for r in rows]
    if not final_test:
        locked = locked_test_pmids()
        if locked:
            result = [r for r in result if r["pmid"] not in locked]
    return result


# ---------------------------------------------------------------------------
# Locked test set (frozen November pmid seal -- convention-proof held-out guard)
# ---------------------------------------------------------------------------

def locked_test_pmids(path=None):
    """The frozen locked-test pmid set, or an empty frozenset if not yet sealed.
    Read by labeled_articles to subtract the held-out papers from the dev pool."""
    path = path or LOCKED_TEST_PMIDS_FILE
    if not path.exists():
        return frozenset()
    return frozenset(json.loads(path.read_text()).get("pmids", []))


def freeze_locked_test_set(conn, path=None, overwrite=False):
    """Materialize and FREEZE the locked-test pmid set: the labeled pmids whose
    ISSUE date (issue_date_iso -- the epub-contamination-free axis) falls in the
    locked window. Written to a version-stable JSON file; refuses to overwrite an
    existing seal unless overwrite=True. Returns the sealed pmid list.

    EXTEND-ONLY: the seal may grow but never shrink. Any pmid already in the
    existing seal file is carried forward (UNIONed) even if the new issue-date
    window no longer selects it, so re-sealing on a better date axis can only ADD
    true-November papers, never spend the test set by dropping one. Now that the
    seal is on issue_date_iso, this catches November papers the old epub-preferred
    pub_date_iso mis-bucketed into October."""
    path = path or LOCKED_TEST_PMIDS_FILE
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"locked test set already sealed at {path} -- refusing to overwrite. "
            f"Pass overwrite=True only to EXTEND the seal (union-preserving).")
    rows = conn.execute("""
        SELECT hl.pmid FROM human_labels hl
        JOIN articles a ON a.pmid = hl.pmid
        WHERE a.issue_date_iso >= ? AND a.issue_date_iso <= ?
        ORDER BY hl.pmid
    """, (LOCKED_TEST_START, LOCKED_TEST_END)).fetchall()
    window_pmids = {r["pmid"] for r in rows}
    prior_pmids = set(locked_test_pmids(path))       # extend-only: never drop a sealed pmid
    pmids = sorted(window_pmids | prior_pmids)
    payload = {
        "created_at": _utcnow(),
        "definition": (f"labeled pmids with issue_date_iso in "
                       f"[{LOCKED_TEST_START}, {LOCKED_TEST_END}], UNION any "
                       f"previously-sealed pmids (extend-only)"),
        "locked_window": [LOCKED_TEST_START, LOCKED_TEST_END],
        "count": len(pmids),
        "count_from_issue_window": len(window_pmids),
        "count_carried_from_prior_seal": len(prior_pmids - window_pmids),
        "pmids": pmids,
        "note": ("FROZEN held-out test set -- do not regenerate destructively. "
                 "EXTEND-ONLY: re-sealing UNIONs with the prior seal, never trims."),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return pmids


def count_human_labels(conn):
    total = conn.execute("SELECT COUNT(*) FROM human_labels").fetchone()[0]
    relevant = conn.execute("SELECT COUNT(*) FROM human_labels WHERE relevant = 1").fetchone()[0]
    curation = conn.execute(
        "SELECT COUNT(*) FROM human_labels WHERE curation_label IS NOT NULL"
    ).fetchone()[0]
    return {
        "total_labeled": total,
        "relevant": relevant,
        "not_relevant": total - relevant,
        "curation_labeled": curation,
    }


# ---------------------------------------------------------------------------
# Labeler helpers (relevance_labeler and curation_labeler apps)
# ---------------------------------------------------------------------------

def unlabeled_articles(conn, start=None, end=None):
    """Articles in the window with no human_labels row -- the relevance labeling pool."""
    rows = conn.execute("""
        SELECT a.pmid, a.title, a.abstract, a.authors_json, a.journal, a.doi
        FROM articles a
        LEFT JOIN human_labels hl ON hl.pmid = a.pmid
        WHERE hl.pmid IS NULL
          AND (? IS NULL OR a.issue_date_iso >= ?)
          AND (? IS NULL OR a.issue_date_iso <= ?)
        ORDER BY a.issue_date_iso
    """, (start, start, end, end)).fetchall()
    return [dict(r) for r in rows]


def relevant_unlabeled_curation(conn):
    """Articles with relevant=1 and no curation_label -- the curation labeling pool."""
    rows = conn.execute("""
        SELECT a.pmid, a.title, a.abstract, a.authors_json, a.journal, a.doi
        FROM articles a
        JOIN human_labels hl ON hl.pmid = a.pmid
        WHERE hl.relevant = 1 AND hl.curation_label IS NULL
        ORDER BY a.pub_date_iso
    """).fetchall()
    return [dict(r) for r in rows]


def set_relevance_label(conn, pmid, relevant):
    """Upsert the relevance label (0 or 1) without touching curation_label."""
    conn.execute("""
        INSERT INTO human_labels (pmid, relevant)
        VALUES (?, ?)
        ON CONFLICT(pmid) DO UPDATE SET relevant = excluded.relevant
    """, (pmid, relevant))
    conn.commit()


def set_curation_label(conn, pmid, curation_label):
    """Update curation_label on an existing relevant=1 row."""
    conn.execute(
        "UPDATE human_labels SET curation_label = ? WHERE pmid = ?",
        (curation_label, pmid),
    )
    conn.commit()


def sample_unlabeled_by_month(conn, months, n_per_month, seed=42):
    """Randomly sample n_per_month unlabeled articles from each given month prefix
    (YYYY-MM strings). Returns a shuffled list of pmids across all months.

    months: list of 'YYYY-MM' strings, e.g. ['2025-01', '2025-03']
    n_per_month: max articles to draw per month (takes all available if fewer exist)
    seed: random seed for reproducibility

    Does NOT filter on already-labeled rows inside this call -- that way the
    returned list is stable; the labeler filters at queue-load time.
    """
    rng = random.Random(seed)
    collected = []
    for month in months:
        start = f"{month}-01"
        end = f"{month}-31"   # SQLite BETWEEN is inclusive; 31 covers any month end
        rows = conn.execute("""
            SELECT a.pmid FROM articles a
            LEFT JOIN human_labels hl ON hl.pmid = a.pmid
            WHERE hl.pmid IS NULL
              AND a.issue_date_iso >= ? AND a.issue_date_iso <= ?
            ORDER BY RANDOM()
        """, (start, end)).fetchall()
        pmids = [r["pmid"] for r in rows]
        collected.extend(pmids[:n_per_month])
    rng.shuffle(collected)
    return collected


_TEST_ARTICLE_COLS = (
    "a.pmid, a.title, a.abstract, a.pages, a.authors_json, a.journal, "
    "a.pub_date, a.pub_date_iso, a.epub_date, a.doi, a.pub_types_json"
)


def setup_ui_test_labeler_db(db_path, mode="relevance", n_prelabeled=10, n_unlabeled=10):
    """Create a fresh isolated test DB for --ui_test mode.

    Seeds n_prelabeled already-labeled articles (so Back/review has something to
    show) plus n_unlabeled articles forming the labeling pool:

      relevance: pool = articles with NO human_labels row (label relevant 0/1).
      curation:  pool = relevant=1 articles seeded with curation_label stripped
                 to NULL (rate 0-5). Sourced from any relevant=1 article, since
                 production's relevant rows already carry a curation_label.
    """
    if db_path.exists():
        db_path.unlink()

    prod_conn = get_connection()
    test_conn = get_connection(db_path)
    try:
        if mode == "relevance":
            pre_rows = prod_conn.execute(f"""
                SELECT {_TEST_ARTICLE_COLS}, hl.relevant, hl.curation_label, hl.notes
                FROM articles a JOIN human_labels hl ON hl.pmid = a.pmid
                ORDER BY RANDOM() LIMIT ?
            """, (n_prelabeled,)).fetchall()
            pre_pmids = {r["pmid"] for r in pre_rows}

            un_rows = prod_conn.execute(f"""
                SELECT {_TEST_ARTICLE_COLS}
                FROM articles a
                LEFT JOIN human_labels hl ON hl.pmid = a.pmid
                WHERE hl.pmid IS NULL
                ORDER BY RANDOM() LIMIT ?
            """, (n_unlabeled,)).fetchall()
        else:  # curation
            pre_rows = prod_conn.execute(f"""
                SELECT {_TEST_ARTICLE_COLS}, hl.relevant, hl.curation_label, hl.notes
                FROM articles a JOIN human_labels hl ON hl.pmid = a.pmid
                WHERE hl.relevant = 1 AND hl.curation_label IS NOT NULL
                ORDER BY RANDOM() LIMIT ?
            """, (n_prelabeled,)).fetchall()
            pre_pmids = {r["pmid"] for r in pre_rows}

            ph = ",".join("?" * len(pre_pmids)) or "''"
            un_rows = prod_conn.execute(f"""
                SELECT {_TEST_ARTICLE_COLS}
                FROM articles a JOIN human_labels hl ON hl.pmid = a.pmid
                WHERE hl.relevant = 1 AND a.pmid NOT IN ({ph})
                ORDER BY RANDOM() LIMIT ?
            """, (*pre_pmids, n_unlabeled)).fetchall()
    finally:
        prod_conn.close()

    # sqlite3.Row has no .get(); convert to plain dicts for keyword access.
    pre_rows = [dict(r) for r in pre_rows]
    un_rows = [dict(r) for r in un_rows]

    for row in pre_rows + un_rows:
        test_conn.execute("""
            INSERT OR IGNORE INTO articles
                (pmid, title, abstract, pages, authors_json, journal,
                 pub_date, pub_date_iso, epub_date, doi, pub_types_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (row["pmid"], row["title"], row.get("abstract"), row.get("pages"),
              row.get("authors_json"), row.get("journal"),
              row.get("pub_date"), row.get("pub_date_iso"), row.get("epub_date"),
              row.get("doi"), row.get("pub_types_json")))

    # Pre-labeled rows seed their real label. For curation, the unlabeled pool
    # also needs relevant=1 rows (curation_label NULL) so they show up to rate.
    for row in pre_rows:
        test_conn.execute("""
            INSERT OR IGNORE INTO human_labels (pmid, relevant, curation_label, notes)
            VALUES (?, ?, ?, ?)
        """, (row["pmid"], row["relevant"], row.get("curation_label"), row.get("notes")))

    if mode == "curation":
        for row in un_rows:
            test_conn.execute("""
                INSERT OR IGNORE INTO human_labels (pmid, relevant, curation_label)
                VALUES (?, 1, NULL)
            """, (row["pmid"],))

    test_conn.commit()
    test_conn.close()
