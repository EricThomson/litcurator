"""
CLI entry point for litcurator.

Commands:
    litcurator status          -- print pipeline snapshot
    litcurator sample --n 100  -- add articles to the review sample
"""

import argparse
from litcurator.config import GROUND_TRUTH_DB
from litcurator import db, label


def _ensure_label_schema(conn):
    for col, typedef in [
        ("selected_for_review", "INTEGER DEFAULT 0"),
        ("relevant", "INTEGER"),
    ]:
        try:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {col} {typedef}")
            conn.commit()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(prog="litcurator")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Print pipeline snapshot")

    sample_parser = subparsers.add_parser("sample", help="Add articles to the review sample")
    sample_parser.add_argument("--n", type=int, default=100, help="Number of articles to add")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    conn = db.get_connection(GROUND_TRUTH_DB)
    _ensure_label_schema(conn)

    if args.command == "status":
        label.print_status(conn)
    elif args.command == "sample":
        label.sample_for_review(conn, args.n)

    conn.close()


if __name__ == "__main__":
    main()
