"""
Database interface for litcurator.

Uses a single SQLite table to store articles as they flow through the
retrieval and curation pipeline. The database lives at ~/.litcurator/litcurator.db.

Pipeline status values:
    1 = retrieved (pulled from PubMed)
    2 = domain-scored (passed Stage 1 domain filter, has domain_score)
    3 = curated (has curation_label from Stage 2)
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DATA_DIR = Path.home() / ".litcurator"
DB_PATH = DATA_DIR / "litcurator.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    pmid TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    authors_json TEXT,          -- JSON: list of {name, affiliation}
    journal TEXT,
    pub_date TEXT,              -- issue/print date, whatever PubMed provides
    epub_date TEXT,             -- electronic pub date if available
    doi TEXT,
    pub_types_json TEXT,        -- JSON: list of publication type strings

    -- Stage 1: domain filter (e.g. systems/behavioral/computational neuro)
    domain_score REAL,          -- LLM score 0.0-1.0
    domain_reasoning TEXT,      -- LLM explanation for auditing/debugging

    -- Stage 2: curation (fine-grained relevance to user interests)
    curation_score REAL,        -- LLM score 0.0-1.0
    curation_label INTEGER,     -- 0 (no), 1 (meh/keep), 2 (love it)
    curation_notes TEXT,        -- curator's annotations

    -- Pipeline metadata
    status INTEGER DEFAULT 1,   -- 1: retrieved, 2: systems-scored, 3: curated
    date_added DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""


def get_connection():
    """Open a connection to the litcurator database, creating it if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def insert_articles(conn, articles):
    """
    Insert a list of article dicts into the database.
    Skips any article whose PMID already exists (INSERT OR IGNORE).

    Args:
        conn: SQLite connection
        articles: list of article dicts from retrieve.py

    Returns:
        Number of new articles inserted.
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
    """
    Fetch all articles at a given pipeline status.

    Args:
        conn: SQLite connection
        status: 1 (retrieved), 2 (domain-scored), 3 (curated)

    Returns:
        List of sqlite3.Row objects.
    """
    cursor = conn.execute(
        "SELECT * FROM articles WHERE status = ? ORDER BY date_added",
        (status,)
    )
    return cursor.fetchall()


def update_domain_score(conn, pmid, score, reasoning):
    """
    Store Stage 1 domain filter score and reasoning for an article.
    Advances status to 2 (domain-scored).

    Args:
        conn: SQLite connection
        pmid: PubMed ID string
        score: float 0.0-1.0
        reasoning: LLM explanation string
    """
    conn.execute(
        """
        UPDATE articles
        SET domain_score = ?, domain_reasoning = ?, status = 2
        WHERE pmid = ?
        """,
        (score, reasoning, pmid)
    )
    conn.commit()


def update_curation(conn, pmid, label, score=None, notes=None):
    """
    Store Stage 2 curation label, optional score, and optional notes.
    Advances status to 3 (curated).

    Args:
        conn: SQLite connection
        pmid: PubMed ID string
        label: int 0 (no), 1 (meh/keep), 2 (love it)
        score: optional float 0.0-1.0 from LLM
        notes: optional string annotations from curator
    """
    conn.execute(
        """
        UPDATE articles
        SET curation_label = ?, curation_score = ?, curation_notes = ?, status = 3
        WHERE pmid = ?
        """,
        (label, score, notes, pmid)
    )
    conn.commit()


def reset_domain_scores(conn, journal=None):
    """
    Reset domain_score, domain_reasoning back to NULL and status back to 1.
    Optionally limit to a specific journal.

    Args:
        conn: SQLite connection
        journal: Optional journal name string (default None = all articles)
    """
    if journal:
        conn.execute(
            """
            UPDATE articles
            SET domain_score = NULL, domain_reasoning = NULL, status = 1
            WHERE journal = ?
            """,
            (journal,)
        )
    else:
        conn.execute(
            """
            UPDATE articles
            SET domain_score = NULL, domain_reasoning = NULL, status = 1
            """
        )
    conn.commit()


def get_all_articles(conn):
    """Fetch all articles regardless of status, ordered by date_added."""
    cursor = conn.execute("SELECT * FROM articles ORDER BY date_added")
    return cursor.fetchall()


def get_article(conn, pmid):
    """Fetch a single article by PMID. Returns sqlite3.Row or None."""
    cursor = conn.execute("SELECT * FROM articles WHERE pmid = ?", (pmid,))
    return cursor.fetchone()


def article_exists(conn, pmid):
    """Return True if an article with this PMID is already in the database."""
    cursor = conn.execute("SELECT 1 FROM articles WHERE pmid = ?", (pmid,))
    return cursor.fetchone() is not None
