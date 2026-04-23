"""
Labeling utilities for litcurator.

Shared helpers for the relevance and curation labeling apps, and
the core logic for sampling articles for review.
"""

import json
import random
from litcurator.config import DATA_DIR

BATCH_STATE_FILE = DATA_DIR / "batch_state.txt"
RANDOM_SEED = 42
TARGET_MONTHS = ["2025-01", "2025-03", "2025-05", "2025-07", "2025-09", "2025-11"]


def render_authors(authors_json):
    authors = json.loads(authors_json or "[]")
    if len(authors) > 4:
        display = authors[:2] + [{"name": "...", "affiliation": ""}] + authors[-2:]
    else:
        display = authors
    parts = []
    for a in display:
        if a["name"] == "...":
            parts.append("...")
        elif a.get("affiliation"):
            parts.append(f"**{a['name']}** ({a['affiliation']})")
        else:
            parts.append(f"**{a['name']}**")
    return " ; ".join(parts)


def read_batch_state(prefix):
    if BATCH_STATE_FILE.exists():
        data = {}
        for line in BATCH_STATE_FILE.read_text().splitlines():
            key, val = line.split("=", 1)
            data[key.strip()] = val.strip()
        count_key = f"{prefix}_batch_start_count"
        elapsed_key = f"{prefix}_batch_elapsed_seconds"
        total_key = f"{prefix}_total_elapsed_seconds"
        if count_key in data and elapsed_key in data:
            return int(data[count_key]), float(data[elapsed_key]), float(data.get(total_key, 0))
    return None, None, None


def write_batch_state(prefix, count, elapsed, total_elapsed):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if BATCH_STATE_FILE.exists():
        for line in BATCH_STATE_FILE.read_text().splitlines():
            key, val = line.split("=", 1)
            data[key.strip()] = val.strip()
    data[f"{prefix}_batch_start_count"] = str(count)
    data[f"{prefix}_batch_elapsed_seconds"] = f"{elapsed:.2f}"
    data[f"{prefix}_total_elapsed_seconds"] = f"{total_elapsed:.2f}"
    BATCH_STATE_FILE.write_text("\n".join(f"{k}={v}" for k, v in sorted(data.items())) + "\n")


def get_month_counts(conn):
    counts = {}
    for month in TARGET_MONTHS:
        n = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE substr(pub_date, 1, 7) = ?", (month,)
        ).fetchone()[0]
        counts[month] = n
    return counts


def sample_for_review(conn, n):
    month_counts = get_month_counts(conn)
    total = sum(month_counts.values())
    already_sampled = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE selected_for_review = 1"
    ).fetchone()[0]
    print(f"{already_sampled} already selected, adding {n} more")
    sampled_total = 0
    for month, count in month_counts.items():
        n_month = round(n * count / total)
        pmids = [r[0] for r in conn.execute(
            "SELECT pmid FROM articles WHERE substr(pub_date, 1, 7) = ? "
            "AND selected_for_review = 0 ORDER BY RANDOM()",
            (month,)
        ).fetchall()]
        rng = random.Random(RANDOM_SEED + hash(month))
        selected = rng.sample(pmids, min(n_month, len(pmids)))
        for pmid in selected:
            conn.execute(
                "UPDATE articles SET selected_for_review = 1 WHERE pmid = ?", (pmid,)
            )
        sampled_total += len(selected)
        print(f"  {month}: {len(selected)} selected (of {count})")
    conn.commit()
    print(f"\nTotal selected for review: {sampled_total}")


def get_status(conn):
    month_counts = get_month_counts(conn)
    total = sum(month_counts.values())
    selected = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE selected_for_review = 1"
    ).fetchone()[0]
    relevance_labeled = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE relevant IS NOT NULL"
    ).fetchone()[0]
    relevant = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE relevant = 1"
    ).fetchone()[0]
    curation_counts = dict(conn.execute(
        "SELECT curation_label, COUNT(*) FROM articles "
        "WHERE relevant = 1 AND curation_label IS NOT NULL "
        "GROUP BY curation_label ORDER BY curation_label"
    ).fetchall())
    curation_labeled = sum(curation_counts.values())
    pct_relevant = 100 * relevant / relevance_labeled if relevance_labeled > 0 else 0
    above_noise = sum(curation_counts.get(l, 0) for l in range(1, 6))
    pct_above_noise = 100 * above_noise / curation_labeled if curation_labeled > 0 else 0
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
    s = get_status(conn)
    print(f"Total articles in target months: {s['total']}")
    print(f"Selected for review:  {s['selected']}")
    print(f"Relevance-labeled:    {s['relevance_labeled']} of {s['selected']}")
    print(f"Relevant:             {s['relevant']} ({s['pct_relevant']:.1f}%)")
    print(f"Curation-labeled:     {s['curation_labeled']} of {s['relevant']}")
    print(f"  Breakdown:")
    for label in range(6):
        print(f"  {label}: {s['curation_counts'].get(label, 0)}")
    print(f"  Above the noise (1+): {s['above_noise']} ({s['pct_above_noise']:.1f}% of curated)")


def main():
    import argparse
    from litcurator.config import GROUND_TRUTH_DB
    from litcurator import db

    parser = argparse.ArgumentParser(prog="litcurator")
    parser.add_argument("--n", type=int, default=100, help="Number of articles to add to the sample")
    parser.add_argument("--status", action="store_true", help="Print pipeline snapshot and exit")
    args = parser.parse_args()

    conn = db.get_connection(GROUND_TRUTH_DB)

    for col, typedef in [("selected_for_review", "INTEGER DEFAULT 0"), ("relevant", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {col} {typedef}")
            conn.commit()
        except Exception:
            pass

    if args.status:
        print_status(conn)
    else:
        sample_for_review(conn, args.n)

    conn.close()


if __name__ == "__main__":
    main()
