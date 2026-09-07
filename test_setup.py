"""Phase 0 smoke test: confirm the environment and the OMDb key load correctly."""

import os
import sys

import pandas as pd
import requests
import sklearn
from dotenv import load_dotenv

OMDB_URL = "https://www.omdbapi.com/"


def mask(secret: str) -> str:
    """Show enough of the key to recognise it, not enough to leak it."""
    if len(secret) < 6:
        return "*" * len(secret)
    return f"{secret[:3]}{'*' * (len(secret) - 5)}{secret[-2:]}"


def main() -> int:
    print("Packages")
    print(f"  requests      {requests.__version__}")
    print(f"  scikit-learn  {sklearn.__version__}")
    print(f"  pandas        {pd.__version__}")
    print("  python-dotenv loaded")

    load_dotenv()  # reads .env into os.environ
    api_key = os.getenv("OMDB_API_KEY")

    print("\n.env")
    if not api_key:
        print("  OMDB_API_KEY: MISSING — add it to .env")
        return 1
    print(f"  OMDB_API_KEY: {mask(api_key)} ({len(api_key)} chars, loaded from .env)")

    print("\nOMDb API")
    try:
        resp = requests.get(
            OMDB_URL,
            params={"apikey": api_key, "t": "The Matrix", "y": "1999"},
            timeout=15,
        )
    except requests.RequestException as exc:
        print(f"  could not reach OMDb: {exc}")
        return 1

    if resp.status_code == 401:
        print("  key rejected: HTTP 401 — check OMDB_API_KEY in .env")
        return 1
    if resp.status_code != 200:
        print(f"  unexpected HTTP {resp.status_code}: {resp.text[:120]}")
        return 1

    data = resp.json()
    # OMDb answers 200 even for failures; the real status is in "Response".
    if data.get("Response") != "True":
        print(f"  key rejected: {data.get('Error', 'unknown error')}")
        return 1

    print("  key accepted")
    print(f"  sample lookup: {data['Title']} ({data['Year']}) — IMDb {data['imdbRating']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
