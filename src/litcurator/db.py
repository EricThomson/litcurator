"""
Database interface for litcurator.

Six-table normalized schema:
  articles      -- permanent PubMed metadata and human labels
  profiles      -- versioned user interest profile snapshots
  prompts       -- versioned scoring prompt snapshots
  scoring_runs  -- provenance for one scoring session
  evaluations   -- one LLM score per article per scoring run
  feedback      -- permanent user flags on specific evaluations

Pipeline status values on articles:
  1 = retrieved (pulled from PubMed)
  2 = domain-scored (has at least one domain evaluation)
  3 = relevant (human label: relevant=1)
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from litcurator.config import DATA_DIR

DB_PATH = DATA_DIR / "litcurator.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_ARTICLES = """
CREATE TABLE IF NOT EXISTS articles (
    pmid TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    authors_json TEXT,
    journal TEXT,
    pub_date TEXT,
    epub_date TEXT,
    doi TEXT,
    pub_types_json TEXT,
    selected_for_review INTEGER DEFAULT 0,
    relevant INTEGER,
    curation_label INTEGER,
    curation_notes TEXT,
    status INTEGER DEFAULT 1,
    date_added DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_PROFILES = """
CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    parent_id TEXT REFERENCES profiles(id),
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_PROMPTS = """
CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_SCORING_RUNS = """
CREATE TABLE IF NOT EXISTS scoring_runs (
    id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    model TEXT NOT NULL,
    profile_id TEXT REFERENCES profiles(id),
    prompt_id TEXT NOT NULL REFERENCES prompts(id),
    date_start TEXT,
    date_end TEXT,
    threshold REAL DEFAULT 0.5,
    status TEXT DEFAULT 'started',
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    CHECK (stage IN ('domain', 'curation')),
    CHECK (status IN ('started', 'completed', 'failed', 'partial'))
)
"""

_CREATE_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid TEXT NOT NULL REFERENCES articles(pmid),
    run_id TEXT NOT NULL REFERENCES scoring_runs(id),
    score REAL NOT NULL,
    rationale TEXT,
    evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (pmid, run_id),
    CHECK (score >= 0.0 AND score <= 1.0)
)
"""

_CREATE_FEEDBACK = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id),
    is_correct INTEGER NOT NULL,
    feedback_label TEXT,
    note TEXT,
    ingested_to_profile_id TEXT REFERENCES profiles(id),
    flagged_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ingested_at DATETIME,
    CHECK (is_correct IN (0, 1))
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_articles_pub_date ON articles(pub_date)",
    "CREATE INDEX IF NOT EXISTS idx_scoring_runs_stage ON scoring_runs(stage, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_scoring_runs_profile ON scoring_runs(profile_id)",
    "CREATE INDEX IF NOT EXISTS idx_scoring_runs_prompt ON scoring_runs(prompt_id)",
    "CREATE INDEX IF NOT EXISTS idx_evaluations_pmid ON evaluations(pmid)",
    "CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_evaluation ON feedback(evaluation_id)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_ingested ON feedback(ingested_to_profile_id)",
]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection(path=None):
    """Open a connection to the litcurator database, creating it if needed."""
    db_path = Path(path) if path else DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    for sql in [_CREATE_ARTICLES, _CREATE_PROFILES, _CREATE_PROMPTS,
                _CREATE_SCORING_RUNS, _CREATE_EVALUATIONS, _CREATE_FEEDBACK]:
        conn.execute(sql)
    for sql in _CREATE_INDEXES:
        conn.execute(sql)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------

def insert_articles(conn, articles):
    """
    Insert a list of article dicts into the database.
    Skips any article whose PMID already exists.

    Returns number of new articles inserted.
    """
    sql = """
        INSERT OR IGNORE INTO articles
            (pmid, title, abstract, authors_json, journal,
             pub_date, epub_date, doi, pub_types_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    """
    inserted = 0
    for article in articles:
        cursor = conn.execute(sql, (
            article["pubmed_id"],
            article["title"],
            article["abstract"],
            json.dumps(article["authors"]),
            article["journal"],
            article["pub_date"],
            article.get("epub_date", ""),
            article["doi"],
            json.dumps(article["pub_types"]),
        ))
        if cursor.rowcount > 0:
            inserted += 1
    conn.commit()
    return inserted


def get_articles_by_status(conn, status):
    """Fetch all articles at a given pipeline status, ordered by date_added."""
    return conn.execute(
        "SELECT * FROM articles WHERE status = ? ORDER BY date_added",
        (status,)
    ).fetchall()


def get_all_articles(conn):
    """Fetch all articles regardless of status, ordered by date_added."""
    return conn.execute("SELECT * FROM articles ORDER BY date_added").fetchall()


def get_article(conn, pmid):
    """Fetch a single article by PMID. Returns sqlite3.Row or None."""
    return conn.execute("SELECT * FROM articles WHERE pmid = ?", (pmid,)).fetchone()


def article_exists(conn, pmid):
    """Return True if an article with this PMID is already in the database."""
    return conn.execute("SELECT 1 FROM articles WHERE pmid = ?", (pmid,)).fetchone() is not None


def update_relevance(conn, pmid, relevant):
    """Store human relevance label (0 or 1)."""
    conn.execute("UPDATE articles SET relevant = ? WHERE pmid = ?", (relevant, pmid))
    conn.commit()


def update_curation(conn, pmid, label, notes=None):
    """Store human curation label (0-5) and advance status to 3."""
    conn.execute(
        "UPDATE articles SET curation_label = ?, curation_notes = ?, status = 3 WHERE pmid = ?",
        (label, notes, pmid)
    )
    conn.commit()


def reset_domain_scores(conn, journal=None):
    """Reset article status back to 1 so domain scoring reruns. Optionally limit to a journal."""
    if journal:
        conn.execute("UPDATE articles SET status = 1 WHERE journal = ?", (journal,))
    else:
        conn.execute("UPDATE articles SET status = 1")
    conn.commit()


def has_articles_for_date_range(conn, date_start, date_end):
    """Return True if any articles with pub_date in this range exist in the DB."""
    return conn.execute(
        "SELECT 1 FROM articles WHERE pub_date >= ? AND pub_date <= ? LIMIT 1",
        (date_start, date_end)
    ).fetchone() is not None


def get_articles_for_date_range(conn, date_start, date_end):
    """Fetch all articles with pub_date in the given range, ordered by pub_date."""
    return conn.execute(
        "SELECT * FROM articles WHERE pub_date >= ? AND pub_date <= ? ORDER BY pub_date",
        (date_start, date_end)
    ).fetchall()


def has_evaluations_for_date_range(conn, stage, date_start, date_end):
    """
    Return True if any articles in this date range have a completed evaluation
    for the given stage. Works for both real LLM runs and synthetic migration runs.
    """
    return conn.execute(
        """
        SELECT 1 FROM articles a
        JOIN evaluations e ON a.pmid = e.pmid
        JOIN scoring_runs r ON e.run_id = r.id
        WHERE r.stage = ? AND r.status = 'completed'
          AND a.pub_date >= ? AND a.pub_date <= ?
        LIMIT 1
        """,
        (stage, date_start, date_end)
    ).fetchone() is not None


def count_curation_labeled_for_date_range(conn, date_start, date_end):
    """Count of articles with a human curation label (0-5) in the given pub_date range."""
    return conn.execute(
        "SELECT COUNT(*) FROM articles WHERE curation_label IS NOT NULL AND pub_date >= ? AND pub_date <= ?",
        (date_start, date_end)
    ).fetchone()[0]


def get_human_relevant_articles_for_date_range(conn, date_start, date_end):
    """
    Articles where the human relevance label is 1 within the given pub_date range.
    Used by curation_score in evaluation mode (gates Sonnet on human labels rather
    than Haiku output, so curation can be evaluated independently of domain filter).
    """
    return conn.execute(
        "SELECT * FROM articles WHERE relevant = 1 AND pub_date >= ? AND pub_date <= ? ORDER BY pub_date",
        (date_start, date_end)
    ).fetchall()


def get_articles_passing_domain_filter(conn, date_start, date_end, threshold=0.5):
    """
    Return articles in this date range that passed the most recent domain filter
    evaluation (score >= threshold). Works for both LLM and synthetic migration runs.
    """
    return conn.execute(
        """
        SELECT a.*
        FROM articles a
        JOIN evaluations e ON a.pmid = e.pmid
        JOIN scoring_runs r ON e.run_id = r.id
        WHERE r.stage = 'domain' AND r.status = 'completed'
          AND a.pub_date >= ? AND a.pub_date <= ?
          AND e.score >= ?
          AND r.id = (
              SELECT r2.id FROM scoring_runs r2
              JOIN evaluations e2 ON e2.run_id = r2.id
              WHERE r2.stage = 'domain' AND e2.pmid = a.pmid
              ORDER BY r2.created_at DESC LIMIT 1
          )
        ORDER BY a.pub_date
        """,
        (date_start, date_end, threshold)
    ).fetchall()


def get_domain_borderline_articles(conn, date_start, date_end, threshold=0.5, window=0.15):
    """
    Return articles with domain scores within window of the threshold, in both
    directions. Useful for spot-checking domain filter behavior.
    """
    return conn.execute(
        """
        SELECT a.pmid, a.title, a.journal, a.pub_date, e.score, e.rationale
        FROM articles a
        JOIN evaluations e ON a.pmid = e.pmid
        JOIN scoring_runs r ON e.run_id = r.id
        WHERE r.stage = 'domain' AND r.status = 'completed'
          AND a.pub_date >= ? AND a.pub_date <= ?
          AND ABS(e.score - ?) <= ?
          AND r.id = (
              SELECT r2.id FROM scoring_runs r2
              JOIN evaluations e2 ON e2.run_id = r2.id
              WHERE r2.stage = 'domain' AND e2.pmid = a.pmid
              ORDER BY r2.created_at DESC LIMIT 1
          )
        ORDER BY e.score DESC
        """,
        (date_start, date_end, threshold, window)
    ).fetchall()


def get_pipeline_coverage(conn):
    """
    Return a summary of completed scoring runs by date range and stage.
    Each entry has date_start, date_end, stage, and latest_run timestamp.
    Excludes synthetic migration runs (which have NULL date_start).
    """
    rows = conn.execute(
        """
        SELECT stage, date_start, date_end, MAX(created_at) AS latest_run
        FROM scoring_runs
        WHERE status = 'completed' AND date_start IS NOT NULL
        GROUP BY stage, date_start, date_end
        ORDER BY date_start, stage
        """
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def _sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def get_or_create_profile(conn, content, notes=None, parent_id=None):
    """
    Snapshot a profile. If identical content already exists, return existing id.

    Returns profile id (SHA256 hash of content).
    """
    profile_id = _sha256(content)
    existing = conn.execute("SELECT id FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO profiles (id, content, parent_id, notes) VALUES (?, ?, ?, ?)",
            (profile_id, content, parent_id, notes)
        )
        conn.commit()
    return profile_id


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def get_or_create_prompt(conn, content, notes=None):
    """
    Snapshot a prompt. If identical content already exists, return existing id.

    Returns prompt id (SHA256 hash of content).
    """
    prompt_id = _sha256(content)
    existing = conn.execute("SELECT id FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO prompts (id, content, notes) VALUES (?, ?, ?)",
            (prompt_id, content, notes)
        )
        conn.commit()
    return prompt_id


# ---------------------------------------------------------------------------
# Scoring runs
# ---------------------------------------------------------------------------

def create_scoring_run(conn, stage, model, prompt_id, profile_id=None,
                       date_start=None, date_end=None, threshold=0.5, notes=None):
    """
    Create a new scoring run row and return its id.

    Run id is a UTC timestamp string: "YYYYMMDD_HHMMSS".
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    conn.execute(
        """
        INSERT INTO scoring_runs
            (id, stage, model, profile_id, prompt_id, date_start, date_end, threshold, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, stage, model, profile_id, prompt_id, date_start, date_end, threshold, notes)
    )
    conn.commit()
    return run_id


def complete_scoring_run(conn, run_id):
    """Mark a scoring run as completed."""
    conn.execute(
        "UPDATE scoring_runs SET status = 'completed', completed_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), run_id)
    )
    conn.commit()


def fail_scoring_run(conn, run_id):
    """Mark a scoring run as failed."""
    conn.execute(
        "UPDATE scoring_runs SET status = 'failed', completed_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), run_id)
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------

def insert_evaluation(conn, pmid, run_id, score, rationale=None):
    """
    Insert one evaluation row. Updates article status based on run stage.

    Returns evaluation id.
    """
    cursor = conn.execute(
        """
        INSERT OR REPLACE INTO evaluations (pmid, run_id, score, rationale)
        VALUES (?, ?, ?, ?)
        """,
        (pmid, run_id, score, rationale)
    )
    run = conn.execute("SELECT stage FROM scoring_runs WHERE id = ?", (run_id,)).fetchone()
    if run and run["stage"] == "domain":
        conn.execute("UPDATE articles SET status = 2 WHERE pmid = ? AND status < 2", (pmid,))
    conn.commit()
    return cursor.lastrowid


def get_latest_evaluations(conn, stage, date_start=None, date_end=None):
    """
    Get the most recent evaluation per article for a given stage.
    Optionally filter articles by pub_date range.

    Returns rows with all evaluation fields plus article metadata.
    """
    date_filter = ""
    params = [stage]
    if date_start and date_end:
        date_filter = "AND a.pub_date >= ? AND a.pub_date <= ?"
        params += [date_start, date_end]

    return conn.execute(
        f"""
        SELECT e.*, a.title, a.abstract, a.journal, a.pub_date, a.doi,
               a.authors_json, a.relevant, a.curation_label, a.curation_notes,
               r.model, r.profile_id, r.prompt_id, r.created_at AS run_created_at
        FROM evaluations e
        JOIN scoring_runs r ON e.run_id = r.id
        JOIN articles a ON e.pmid = a.pmid
        WHERE r.stage = ?
          {date_filter}
          AND r.id = (
              SELECT r2.id FROM scoring_runs r2
              JOIN evaluations e2 ON e2.run_id = r2.id
              WHERE r2.stage = ? AND e2.pmid = e.pmid
              ORDER BY r2.created_at DESC LIMIT 1
          )
        ORDER BY e.score DESC
        """,
        params + [stage]
    ).fetchall()


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def upsert_feedback(conn, evaluation_id, note, is_correct=0, feedback_label=None):
    """Insert or update user feedback for an evaluation. One row per evaluation."""
    existing = conn.execute(
        "SELECT id FROM feedback WHERE evaluation_id = ?", (evaluation_id,)
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE feedback SET note = ?, is_correct = ?, feedback_label = ?, flagged_at = ? WHERE id = ?",
            (note, is_correct, feedback_label, datetime.now(timezone.utc).isoformat(), existing["id"])
        )
    else:
        conn.execute(
            "INSERT INTO feedback (evaluation_id, is_correct, note, feedback_label) VALUES (?, ?, ?, ?)",
            (evaluation_id, is_correct, note, feedback_label)
        )
    conn.commit()


def delete_feedback_for_evaluation(conn, evaluation_id):
    """Remove feedback for a specific evaluation."""
    conn.execute("DELETE FROM feedback WHERE evaluation_id = ?", (evaluation_id,))
    conn.commit()


def get_feedback_for_evaluations(conn, evaluation_ids):
    """
    Get feedback rows for a list of evaluation IDs.
    Returns dict mapping evaluation_id -> feedback Row.
    """
    if not evaluation_ids:
        return {}
    placeholders = ",".join("?" * len(evaluation_ids))
    rows = conn.execute(
        f"SELECT * FROM feedback WHERE evaluation_id IN ({placeholders})",
        list(evaluation_ids)
    ).fetchall()
    return {row["evaluation_id"]: row for row in rows}


def get_uningested_feedback_by_pmids(conn, pmids):
    """
    Get most recent uningested feedback per pmid, keyed by pmid.
    Used to surface pending flags from previous scoring runs in the feed UI.
    Returns dict mapping pmid -> Row with feedback_id, evaluation_id, note, score.
    """
    if not pmids:
        return {}
    placeholders = ",".join("?" * len(pmids))
    rows = conn.execute(f"""
        SELECT a.pmid, f.id AS feedback_id, f.evaluation_id, f.note, e.score
        FROM feedback f
        JOIN evaluations e ON f.evaluation_id = e.id
        JOIN articles a ON e.pmid = a.pmid
        WHERE a.pmid IN ({placeholders})
          AND f.ingested_to_profile_id IS NULL
          AND f.flagged_at = (
              SELECT MAX(f2.flagged_at)
              FROM feedback f2
              JOIN evaluations e2 ON f2.evaluation_id = e2.id
              WHERE e2.pmid = a.pmid
                AND f2.ingested_to_profile_id IS NULL
          )
    """, list(pmids)).fetchall()
    return {row["pmid"]: row for row in rows}


def get_latest_feedback_by_pmids(conn, pmids):
    """
    Get the most recent feedback per pmid across all scoring runs.
    Useful for surfacing prior notes when an article has been re-scored.
    Returns dict mapping pmid -> Row with note, score, flagged_at.
    """
    if not pmids:
        return {}
    placeholders = ",".join("?" * len(pmids))
    rows = conn.execute(
        f"""
        SELECT a.pmid, f.id AS feedback_id, f.note, f.flagged_at, e.score
        FROM feedback f
        JOIN evaluations e ON f.evaluation_id = e.id
        JOIN articles a ON e.pmid = a.pmid
        WHERE a.pmid IN ({placeholders})
        ORDER BY f.flagged_at DESC
        """,
        list(pmids)
    ).fetchall()
    result = {}
    for row in rows:
        if row["pmid"] not in result:
            result[row["pmid"]] = row
    return result


def get_uningested_feedback_periods(conn):
    """
    Return list of (year_month, count) for months that have uningested flags,
    e.g. [("2025-01", 8), ("2025-03", 4)], sorted chronologically.
    """
    rows = conn.execute("""
        SELECT strftime('%Y-%m', a.pub_date) AS month, COUNT(DISTINCT a.pmid) AS cnt
        FROM feedback f
        JOIN evaluations e ON f.evaluation_id = e.id
        JOIN articles a ON e.pmid = a.pmid
        WHERE f.ingested_to_profile_id IS NULL
          AND f.flagged_at = (
              SELECT MAX(f2.flagged_at)
              FROM feedback f2
              JOIN evaluations e2 ON f2.evaluation_id = e2.id
              WHERE e2.pmid = a.pmid
                AND f2.ingested_to_profile_id IS NULL
          )
        GROUP BY month
        ORDER BY month
    """).fetchall()
    return [(r["month"], r["cnt"]) for r in rows]


def get_uningested_feedback(conn, months=None):
    """
    Get the most recent uningested feedback per article,
    joined with evaluation and article data.
    Optionally filter to articles whose pub_date falls in the given months
    (list of 'YYYY-MM' strings).
    """
    month_filter = ""
    params = []
    if months:
        placeholders = ",".join("?" * len(months))
        month_filter = f"AND strftime('%Y-%m', a.pub_date) IN ({placeholders})"
        params = list(months)
    return conn.execute(f"""
        SELECT f.id AS feedback_id, f.note, f.feedback_label,
               e.id AS evaluation_id, e.score, e.rationale,
               a.pmid, a.title, a.abstract, a.journal, a.curation_label
        FROM feedback f
        JOIN evaluations e ON f.evaluation_id = e.id
        JOIN articles a ON e.pmid = a.pmid
        WHERE f.ingested_to_profile_id IS NULL
          {month_filter}
          AND f.flagged_at = (
              SELECT MAX(f2.flagged_at)
              FROM feedback f2
              JOIN evaluations e2 ON f2.evaluation_id = e2.id
              WHERE e2.pmid = a.pmid
                AND f2.ingested_to_profile_id IS NULL
          )
        ORDER BY e.score DESC
    """, params).fetchall()


def get_all_uningested_feedback_ids_for_pmids(conn, pmids):
    """Get all uningested feedback IDs for a list of pmids, including duplicates from older runs."""
    if not pmids:
        return []
    placeholders = ",".join("?" * len(pmids))
    rows = conn.execute(
        f"""
        SELECT f.id FROM feedback f
        JOIN evaluations e ON f.evaluation_id = e.id
        JOIN articles a ON e.pmid = a.pmid
        WHERE a.pmid IN ({placeholders}) AND f.ingested_to_profile_id IS NULL
        """,
        list(pmids)
    ).fetchall()
    return [row["id"] for row in rows]


def discard_uningested_feedback(conn, months=None):
    """
    Delete uningested feedback rows, optionally restricted to specific months.
    months: list of 'YYYY-MM' strings, or None to discard all.
    Does not affect ingested feedback or the pre-fill read path.
    """
    if months:
        placeholders = ",".join("?" * len(months))
        conn.execute(f"""
            DELETE FROM feedback
            WHERE ingested_to_profile_id IS NULL
              AND evaluation_id IN (
                SELECT e.id FROM evaluations e
                JOIN articles a ON e.pmid = a.pmid
                WHERE strftime('%Y-%m', a.pub_date) IN ({placeholders})
              )
        """, list(months))
    else:
        conn.execute("DELETE FROM feedback WHERE ingested_to_profile_id IS NULL")
    conn.commit()


def mark_feedback_ingested(conn, feedback_ids, profile_id):
    """Mark a list of feedback rows as ingested into a profile version."""
    placeholders = ",".join("?" * len(feedback_ids))
    conn.execute(
        f"UPDATE feedback SET ingested_to_profile_id = ?, ingested_at = ? WHERE id IN ({placeholders})",
        [profile_id, datetime.now(timezone.utc).isoformat()] + list(feedback_ids)
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_status(conn):
    """
    Return pipeline counts as a dict.
    Used by CLI --status and labeler apps.
    """
    from litcurator.config import CURATION_THRESHOLD

    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    selected = conn.execute("SELECT COUNT(*) FROM articles WHERE selected_for_review = 1").fetchone()[0]
    relevance_labeled = conn.execute("SELECT COUNT(*) FROM articles WHERE relevant IS NOT NULL").fetchone()[0]
    relevant = conn.execute("SELECT COUNT(*) FROM articles WHERE relevant = 1").fetchone()[0]
    curation_labeled = conn.execute("SELECT COUNT(*) FROM articles WHERE curation_label IS NOT NULL").fetchone()[0]

    curation_counts = {}
    for row in conn.execute(
        "SELECT curation_label, COUNT(*) as n FROM articles WHERE curation_label IS NOT NULL GROUP BY curation_label"
    ).fetchall():
        curation_counts[row["curation_label"]] = row["n"]

    above_noise = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE curation_label >= ?", (CURATION_THRESHOLD,)
    ).fetchone()[0]

    pct_relevant = round(relevant / relevance_labeled * 100, 1) if relevance_labeled else 0
    pct_above_noise = round(above_noise / curation_labeled * 100, 1) if curation_labeled else 0

    return {
        "total": total,
        "selected": selected,
        "relevance_labeled": relevance_labeled,
        "relevant": relevant,
        "pct_relevant": pct_relevant,
        "curation_labeled": curation_labeled,
        "curation_counts": curation_counts,
        "above_noise": above_noise,
        "pct_above_noise": pct_above_noise,
    }


def print_status(conn):
    """Print pipeline status to stdout."""
    s = get_status(conn)
    print(f"Total articles:       {s['total']}")
    print(f"Selected for review:  {s['selected']}")
    print(f"Relevance labeled:    {s['relevance_labeled']} ({s['pct_relevant']}% relevant)")
    print(f"Relevant:             {s['relevant']}")
    print(f"Curation labeled:     {s['curation_labeled']}")
    print(f"Above noise:          {s['above_noise']} ({s['pct_above_noise']}%)")
    if s["curation_counts"]:
        dist = ", ".join(f"{k}:{v}" for k, v in sorted(s["curation_counts"].items()))
        print(f"Curation distribution: {dist}")
