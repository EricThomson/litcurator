"""
Convenience script: score articles for a date range then launch curation review.

Usage:
    python curate.py --start 2025-01-01 --end 2025-01-14
"""

import argparse
import subprocess
import sys
from litcurator import db, evaluate
from litcurator.config import LITCURATOR_DB

parser = argparse.ArgumentParser()
parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
args = parser.parse_args()

conn = db.get_connection(LITCURATOR_DB)
evaluate.curation_score(conn, date_start=args.start, date_end=args.end)
conn.close()

subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    "apps/curation_review.py", "--",
    "--start", args.start, "--end", args.end,
])
