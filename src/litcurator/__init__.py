"""litcurator: PubMed discovery shaped to one researcher's taste."""

from pathlib import Path

from dotenv import load_dotenv

# The repo's .env, loaded once for every module, before any API client is built. override=True
# makes the file the authority: without it, a stale key exported in the shell silently beats the
# one in .env, and editing .env changes nothing.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
