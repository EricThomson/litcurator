"""
db_interface.py -- SQLite-backed durable state for litcurator (v4, seed-judge).

Append-only provenance, adapted from v1's normalized model (which proved
invaluable): you never lose information by running something again. Every scoring
is a run; re-running adds rows, it never overwrites. So you can always ask "how
did this paper score under the old seed vs the new one?" -- the convergence study.

Tables:
  articles        facts retrieved from PubMed (+ pub_date_iso, a normalized date)
  human_labels    the 1220 hand labels from v1 -- their OWN table, benchmark only.
                  NEVER injected into evaluations as fake 'human' scores (v1's
                  worst mistake: it poisoned the model column forever).
  profiles        content-addressed user-profile snapshots: id = SHA256(content),
                  parent_id chains the lineage, the seed is the root (parent_id NULL).
  scoring_runs    one row per scoring session: stage (domain|curation), model,
                  mode, profile_id, prompt version+hash, date window, cost,
                  completed_at (NULL = in flight).
  evaluations     one score per (pmid, run_id) -- BOTH stages, unified. Append-only.
  flags           the user's numeric correction on a specific evaluation. Append-only.

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
from datetime import date, datetime, timezone

from litcurator.config import (
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
    summary TEXT,
    pages TEXT,
    authors_json TEXT,
    journal TEXT,
    pub_date TEXT,
    pub_date_iso TEXT,
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
# score" is answerable by JOIN -- the prompt half of the biconvex provenance.
_CREATE_PROMPTS = """
CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    parent_id TEXT REFERENCES prompts(id),
    notes TEXT,
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
    judge_prompt_version TEXT,
    judge_prompt_hash TEXT,
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

# The user's numeric correction on a specific evaluation. Append-only.
# ingested_to_profile_id marks which seed version absorbed this flag.
_CREATE_FLAGS = """
CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id),
    pmid TEXT NOT NULL REFERENCES articles(pmid),
    judge_score REAL NOT NULL,
    your_score REAL NOT NULL,
    delta REAL NOT NULL,
    note TEXT,
    ingested_to_profile_id TEXT REFERENCES profiles(id),
    flagged_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ingested_at DATETIME,
    CHECK (judge_score >= 0.0 AND judge_score <= 1.0),
    CHECK (your_score >= 0.0 AND your_score <= 1.0)
)
"""

_CREATE_STATEMENTS = [
    _CREATE_ARTICLES,
    _CREATE_HUMAN_LABELS,
    _CREATE_PROFILES,
    _CREATE_PROMPTS,
    _CREATE_SCORING_RUNS,
    _CREATE_EVALUATIONS,
    _CREATE_FLAGS,
]

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_articles_pub_date_iso ON articles(pub_date_iso)",
    "CREATE INDEX IF NOT EXISTS idx_human_labels_relevant ON human_labels(relevant)",
    "CREATE INDEX IF NOT EXISTS idx_human_labels_curation ON human_labels(curation_label)",
    "CREATE INDEX IF NOT EXISTS idx_scoring_runs_stage ON scoring_runs(stage, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_scoring_runs_profile ON scoring_runs(profile_id)",
    "CREATE INDEX IF NOT EXISTS idx_evaluations_pmid ON evaluations(pmid)",
    "CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_flags_evaluation ON flags(evaluation_id)",
    "CREATE INDEX IF NOT EXISTS idx_flags_uningested ON flags(ingested_to_profile_id) WHERE ingested_to_profile_id IS NULL",
]

# Columns added to articles after their initial release; add to an existing DB.
# Idempotent (_migrate swallows "duplicate column name"). summary is the neutral
# one-to-two-sentence paper description shown in the review feed.
_ARTICLE_MIGRATIONS = [
    "ALTER TABLE articles ADD COLUMN pub_date_iso TEXT",
    "ALTER TABLE articles ADD COLUMN summary TEXT",
    "ALTER TABLE articles ADD COLUMN pages TEXT",
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
    for sql in _CREATE_STATEMENTS:
        conn.execute(sql)
    _migrate(conn)
    for sql in _CREATE_INDEXES:
        conn.execute(sql)
    conn.commit()
    _backfill_pub_date_iso(conn)
    return conn


def _migrate(conn):
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


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------

def insert_articles(conn, articles):
    """Insert article dicts (from retrieve.parse_single_article). INSERT OR IGNORE
    dedups by pmid; normalizes pub_date_iso. Returns count of new rows."""
    inserted = 0
    for a in articles:
        iso = normalize_pub_date(a.get("pub_date"), a.get("epub_date"))
        cur = conn.execute("""
            INSERT OR IGNORE INTO articles
                (pmid, title, abstract, pages, authors_json, journal,
                 pub_date, pub_date_iso, epub_date, doi, pub_types_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            a["pubmed_id"], a["title"], a.get("abstract"), a.get("pages"),
            json.dumps(a.get("authors")), a.get("journal"),
            a.get("pub_date"), iso, a.get("epub_date"),
            a.get("doi"), json.dumps(a.get("pub_types")),
        ))
        inserted += cur.rowcount
    conn.commit()
    return inserted


def get_article(conn, pmid):
    row = conn.execute("SELECT * FROM articles WHERE pmid = ?", (pmid,)).fetchone()
    return dict(row) if row else None


def articles_in_range(conn, start=None, end=None):
    rows = conn.execute("""
        SELECT * FROM articles
        WHERE (? IS NULL OR pub_date_iso >= ?)
          AND (? IS NULL OR pub_date_iso <= ?)
        ORDER BY pub_date_iso
    """, (start, start, end, end)).fetchall()
    return [dict(r) for r in rows]


def set_article_summary(conn, pmid, summary):
    """Store the neutral one-to-two-sentence summary for an article. A stable
    paper fact: generated once over the reviewable survivors, reused across every
    run and profile version (never re-generated when the seed changes)."""
    conn.execute("UPDATE articles SET summary = ? WHERE pmid = ?", (summary, pmid))
    conn.commit()


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
# profiles, for the judge prompt (the other biconvex knob).
# ---------------------------------------------------------------------------

def get_or_create_prompt(conn, content, parent_id=None, notes=None):
    """Snapshot a judge prompt. id = SHA256(content); identical content returns the
    existing id (no duplicate row). Returns the prompt id."""
    prompt_id = _sha256(content)
    existing = conn.execute("SELECT id FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO prompts (id, content, parent_id, notes) VALUES (?, ?, ?, ?)",
            (prompt_id, content, parent_id, notes),
        )
        conn.commit()
    return prompt_id


def get_prompt(conn, prompt_id):
    row = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    return dict(row) if row else None


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
                               judge_prompt_version=None, judge_prompt_hash=None,
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
            (id, stage, model, mode, profile_id, judge_prompt_version,
             judge_prompt_hash, date_start, date_end, threshold)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, stage, model, mode, profile_id, judge_prompt_version,
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
          AND (? IS NULL OR a.pub_date_iso >= ?)
          AND (? IS NULL OR a.pub_date_iso <= ?)
        ORDER BY a.pub_date_iso
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
        SELECT a.pmid, a.title, a.journal, a.abstract, a.summary, a.pages, a.pub_date_iso,
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
          AND (? IS NULL OR a.pub_date_iso >= ?)
          AND (? IS NULL OR a.pub_date_iso <= ?)
        ORDER BY e.score DESC
    """, (start, start, end, end)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Flags (append-only numeric corrections)
# ---------------------------------------------------------------------------

def insert_flag(conn, evaluation_id, your_score, note=None):
    """Record the user's numeric flag against a specific evaluation. judge_score is
    snapshotted from that evaluation; delta = your_score - judge_score. Returns id."""
    ev = conn.execute("SELECT pmid, score FROM evaluations WHERE id = ?",
                      (evaluation_id,)).fetchone()
    if ev is None:
        raise ValueError(f"no evaluation with id {evaluation_id}")
    judge_score = ev["score"]
    delta = round(your_score - judge_score, 6)
    cur = conn.execute("""
        INSERT INTO flags
            (evaluation_id, pmid, judge_score, your_score, delta, note)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (evaluation_id, ev["pmid"], judge_score, your_score, delta, note))
    conn.commit()
    return cur.lastrowid


def get_flag(conn, flag_id):
    row = conn.execute("SELECT * FROM flags WHERE id = ?", (flag_id,)).fetchone()
    return dict(row) if row else None


def delete_flag(conn, pmid):
    """Delete all flag rows for a paper. Flags are the user's own data; a mistaken
    flag should be removable. Deletes all rows so the paper is fully unflagged."""
    conn.execute("DELETE FROM flags WHERE pmid = ?", (pmid,))
    conn.commit()


def get_flags(conn, only_uningested=False, start=None, end=None):
    """The LATEST flag per paper (flags are append-only; most-recent-wins, so a
    re-flag supersedes without double-counting), joined to its article
    (title/journal/abstract/pub_date_iso) and the evaluation it corrected
    (rationale/surface_decision/possible_mismatch) -- what the suggester and the
    review feed read. Date window filters on the article's pub date; only_uningested
    restricts to papers whose latest flag is not yet folded into a profile version."""
    rows = conn.execute("""
        SELECT f.*, a.title, a.journal, a.abstract, a.pub_date_iso,
               e.rationale, e.surface_decision, e.possible_mismatch
        FROM flags f
        JOIN articles a ON a.pmid = f.pmid
        JOIN evaluations e ON e.id = f.evaluation_id
        WHERE f.id = (
            SELECT f2.id FROM flags f2 WHERE f2.pmid = f.pmid
            ORDER BY f2.flagged_at DESC, f2.id DESC LIMIT 1
        )
          AND (0 = ? OR f.ingested_to_profile_id IS NULL)
          AND (? IS NULL OR a.pub_date_iso >= ?)
          AND (? IS NULL OR a.pub_date_iso <= ?)
        ORDER BY f.flagged_at
    """, (1 if only_uningested else 0, start, start, end, end)).fetchall()
    return [dict(r) for r in rows]


def mark_flags_ingested(conn, flag_ids, profile_id):
    """Mark flags as folded into a profile version (the learning audit trail)."""
    placeholders = ",".join("?" * len(flag_ids))
    conn.execute(
        f"UPDATE flags SET ingested_to_profile_id = ?, ingested_at = ? "
        f"WHERE id IN ({placeholders})",
        [profile_id, _utcnow()] + list(flag_ids),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Human labels (benchmark only -- their own table, never evaluations)
# ---------------------------------------------------------------------------

def insert_human_label(conn, pmid, relevant, curation_label=None, notes=None):
    conn.execute("""
        INSERT OR REPLACE INTO human_labels (pmid, relevant, curation_label, notes)
        VALUES (?, ?, ?, ?)
    """, (pmid, relevant, curation_label, notes))
    conn.commit()


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
          AND (? IS NULL OR a.pub_date_iso >= ?)
          AND (? IS NULL OR a.pub_date_iso <= ?)
        ORDER BY a.pub_date_iso
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
    pub_date_iso falls in the locked window. Written once to a version-stable JSON
    file; refuses to overwrite an existing seal unless overwrite=True (frozen-once
    so the held-out set provably cannot drift). Returns the sealed pmid list.

    Refinement note: the seal is defined on pub_date_iso today. After NLM issue
    dates are backfilled, the set may only be EXTENDED with newly-identified
    true-November pmids (adding is safe; removing risks spending the test set)."""
    path = path or LOCKED_TEST_PMIDS_FILE
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"locked test set already sealed at {path} -- refusing to overwrite. "
            f"Pass overwrite=True only if development has NOT yet started.")
    rows = conn.execute("""
        SELECT hl.pmid FROM human_labels hl
        JOIN articles a ON a.pmid = hl.pmid
        WHERE a.pub_date_iso >= ? AND a.pub_date_iso <= ?
        ORDER BY hl.pmid
    """, (LOCKED_TEST_START, LOCKED_TEST_END)).fetchall()
    pmids = [r["pmid"] for r in rows]
    payload = {
        "created_at": _utcnow(),
        "definition": (f"labeled pmids with pub_date_iso in "
                       f"[{LOCKED_TEST_START}, {LOCKED_TEST_END}]"),
        "locked_window": [LOCKED_TEST_START, LOCKED_TEST_END],
        "count": len(pmids),
        "pmids": pmids,
        "note": ("FROZEN held-out test set -- do not regenerate. May be EXTENDED "
                 "(never trimmed) with true-November pmids after issue-date backfill."),
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
          AND (? IS NULL OR a.pub_date_iso >= ?)
          AND (? IS NULL OR a.pub_date_iso <= ?)
        ORDER BY a.pub_date_iso
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
              AND a.pub_date_iso >= ? AND a.pub_date_iso <= ?
            ORDER BY RANDOM()
        """, (start, end)).fetchall()
        pmids = [r["pmid"] for r in rows]
        collected.extend(pmids[:n_per_month])
    rng.shuffle(collected)
    return collected


_TEST_ARTICLE_COLS = (
    "a.pmid, a.title, a.abstract, a.pages, a.authors_json, a.journal, "
    "a.pub_date, a.pub_date_iso, a.epub_date, a.doi, a.pub_types_json, a.summary"
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
                 pub_date, pub_date_iso, epub_date, doi, pub_types_json, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (row["pmid"], row["title"], row.get("abstract"), row.get("pages"),
              row.get("authors_json"), row.get("journal"),
              row.get("pub_date"), row.get("pub_date_iso"), row.get("epub_date"),
              row.get("doi"), row.get("pub_types_json"), row.get("summary")))

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
