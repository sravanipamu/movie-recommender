"""Phase 1: build a local movie catalog from the OMDb API.

OMDb has no "popular movies" endpoint, so the catalog is assembled by running a
list of seed terms through the search endpoint, collecting unique imdbIDs, then
fetching full details for each one. Results are cached in movies.json so reruns
cost no API calls.

Usage:
    python fetch_movies.py                 # fetch, reusing whatever is cached
    python fetch_movies.py --pages 8       # search deeper per seed term
    python fetch_movies.py --refresh       # ignore the cache, re-fetch everything
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

OMDB_URL = "https://www.omdbapi.com/"
CACHE_FILE = Path(__file__).parent / "movies.json"
REQUEST_DELAY = 0.2  # seconds between calls, to be polite to the free tier

SEED_TERMS = [
    "love",
    "war",
    "space",
    "crime",
    "family",
    "future",
    "night",
    "world",
]

# OMDb reports a blown daily quota through the normal error channel; treat it as
# fatal rather than letting every remaining movie look like a lookup failure.
RATE_LIMIT_MARKER = "request limit reached"


class RateLimited(RuntimeError):
    """The OMDb daily quota is exhausted."""


def omdb_get(session: requests.Session, api_key: str, **params: str) -> dict:
    """Call OMDb and return the decoded body.

    OMDb answers HTTP 200 even for failures, so the caller must still inspect
    the "Response" field.
    """
    params["apikey"] = api_key
    resp = session.get(OMDB_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    error = data.get("Error", "")
    if RATE_LIMIT_MARKER in error.lower():
        raise RateLimited(error)
    return data


def collect_imdb_ids(session: requests.Session, api_key: str, pages: int) -> list[str]:
    """Search every seed term across `pages` pages and return unique imdbIDs."""
    seen: dict[str, None] = {}  # insertion-ordered set

    for term in SEED_TERMS:
        found_for_term = 0
        for page in range(1, pages + 1):
            data = omdb_get(session, api_key, s=term, type="movie", page=str(page))
            time.sleep(REQUEST_DELAY)

            if data.get("Response") != "True":
                # Usually "Movie not found!" past the last page of results.
                break

            for item in data.get("Search", []):
                imdb_id = item.get("imdbID")
                if imdb_id and imdb_id not in seen:
                    seen[imdb_id] = None
                    found_for_term += 1

        print(f"  {term:<8} +{found_for_term:>3} new  (running total: {len(seen)})")

    return list(seen)


def fetch_details(session: requests.Session, api_key: str, imdb_id: str) -> dict | None:
    """Fetch one movie's full record, or None if OMDb has no such title."""
    data = omdb_get(session, api_key, i=imdb_id, plot="full")
    if data.get("Response") != "True":
        return None
    return data


def load_cache(refresh: bool) -> dict[str, dict]:
    """Load movies.json as an imdbID -> record map."""
    if refresh or not CACHE_FILE.exists():
        return {}
    try:
        with CACHE_FILE.open(encoding="utf-8") as fh:
            records = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"warning: {CACHE_FILE.name} is unreadable ({exc}); starting fresh")
        return {}
    return {r["imdbID"]: r for r in records if r.get("imdbID")}


def save_cache(movies: dict[str, dict]) -> None:
    """Write the catalog as a JSON list, ready for pd.DataFrame()."""
    with CACHE_FILE.open("w", encoding="utf-8") as fh:
        json.dump(list(movies.values()), fh, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pages", type=int, default=5, help="search pages per seed term (default: 5)"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="ignore movies.json and re-fetch"
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OMDB_API_KEY")
    if not api_key:
        print("OMDB_API_KEY missing — add it to .env")
        return 1

    movies = load_cache(args.refresh)
    if movies:
        print(f"Loaded {len(movies)} cached movies from {CACHE_FILE.name}\n")

    session = requests.Session()

    try:
        print(f"Searching {len(SEED_TERMS)} seed terms, {args.pages} pages each")
        imdb_ids = collect_imdb_ids(session, api_key, args.pages)
        print(f"\n{len(imdb_ids)} unique imdbIDs found by search")

        pending = [i for i in imdb_ids if i not in movies]
        print(f"{len(imdb_ids) - len(pending)} already cached, {len(pending)} to fetch\n")

        skipped = 0
        for n, imdb_id in enumerate(pending, start=1):
            try:
                record = fetch_details(session, api_key, imdb_id)
            except requests.RequestException as exc:
                print(f"  [{n}/{len(pending)}] {imdb_id} network error: {exc}")
                continue
            finally:
                time.sleep(REQUEST_DELAY)

            if record is None:
                skipped += 1
                continue

            movies[imdb_id] = record
            if n % 25 == 0 or n == len(pending):
                print(f"  [{n}/{len(pending)}] fetched — {len(movies)} in catalog")
                save_cache(movies)  # checkpoint, so a crash costs little

        if skipped:
            print(f"\nSkipped {skipped} ids OMDb had no record for")

    except RateLimited as exc:
        print(f"\nOMDb daily limit hit: {exc}")
        print("Saving what was collected; rerun tomorrow to resume.")
    except KeyboardInterrupt:
        print("\nInterrupted; saving progress.")
    finally:
        save_cache(movies)

    print(f"\n{'=' * 60}")
    print(f"Unique movies collected: {len(movies)}")
    print(f"Saved to: {CACHE_FILE}")
    print(f"{'=' * 60}")

    if not movies:
        print("No movies collected.")
        return 1

    sample = next(iter(movies.values()))
    print("\nSample record:\n")
    print(json.dumps(sample, indent=2, ensure_ascii=False))

    required = ["Title", "Genre", "Director", "Actors", "Plot", "imdbRating"]
    missing = [f for f in required if not sample.get(f)]
    print("\nRequired fields on sample:", "all present" if not missing else f"MISSING {missing}")

    # A field can be present but useless: OMDb writes "N/A" for unknown values.
    na_counts = {
        f: sum(1 for m in movies.values() if m.get(f, "N/A") == "N/A") for f in required
    }
    print("\n\"N/A\" counts across the catalog:")
    for field, count in na_counts.items():
        print(f"  {field:<11} {count:>4} / {len(movies)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
