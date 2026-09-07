"""Phase 2: build a "feature soup" string for every movie in movies.json.

The soup concatenates the fields that describe what a movie *is about* —
genre, director, cast and plot — into one text blob per movie, ready to be
vectorised in Phase 3.

Usage:
    python build_soup.py            # build soups, print samples
    python build_soup.py --save     # also write movies_with_soup.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
MOVIES_FILE = HERE / "movies.json"
OUTPUT_FILE = HERE / "movies_with_soup.json"

# Fields OMDb fills with the literal string "N/A" when it has no value.
SOUP_FIELDS = ["Genre", "Director", "Actors", "Plot"]


def clean(value: object) -> str:
    """Normalise one raw OMDb field to a usable string.

    OMDb signals "missing" with the string "N/A" rather than null, so that has
    to be filtered out explicitly or it becomes a real token in the vocabulary.
    """
    if not isinstance(value, str):
        return ""  # covers None and any non-string OMDb surprise
    value = value.strip()
    if value.upper() == "N/A":
        return ""
    return value


def split_list(value: object) -> list[str]:
    """Split a comma-separated OMDb field into clean parts.

    Individual entries can themselves be "N/A" (e.g. "Ridley Scott, N/A"), so
    each part is checked, not just the field as a whole.
    """
    field = clean(value)
    if not field:
        return []
    return [part for raw in field.split(",") if (part := clean(raw))]


def build_soup(movie: dict) -> str:
    """Combine one movie's Genre, Director, Actors and Plot into one string."""
    genres = split_list(movie.get("Genre"))
    directors = split_list(movie.get("Director"))
    actors = split_list(movie.get("Actors"))
    plot = clean(movie.get("Plot"))

    parts = [" ".join(genres), " ".join(directors), " ".join(actors), plot]

    # Drop empties before joining so missing fields leave no double spaces.
    soup = " ".join(part for part in parts if part)

    # Collapse any whitespace the plot text brought with it (newlines, tabs).
    return re.sub(r"\s+", " ", soup).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", action="store_true", help=f"write {OUTPUT_FILE.name}"
    )
    parser.add_argument(
        "--samples", type=int, default=3, help="how many soups to print (default: 3)"
    )
    args = parser.parse_args()

    if not MOVIES_FILE.exists():
        print(f"{MOVIES_FILE.name} not found — run fetch_movies.py first")
        return 1

    with MOVIES_FILE.open(encoding="utf-8") as fh:
        movies = json.load(fh)

    print(f"Loaded {len(movies)} movies from {MOVIES_FILE.name}\n")

    for movie in movies:
        movie["soup"] = build_soup(movie)

    # --- verification -----------------------------------------------------
    print("=" * 70)
    print(f"SAMPLE SOUPS (first {args.samples})")
    print("=" * 70)

    for movie in movies[: args.samples]:
        print(f"\n{movie.get('Title', '?')} ({movie.get('Year', '?')})")
        print(f"  Genre    : {movie.get('Genre')}")
        print(f"  Director : {movie.get('Director')}")
        print(f"  Actors   : {movie.get('Actors')}")
        print(f"  soup     : {movie['soup'][:300]}"
              + ("..." if len(movie['soup']) > 300 else ""))

    # A movie with a genuinely missing field is the case worth eyeballing, so
    # surface one automatically instead of hoping it lands in the first three.
    incomplete = [
        m for m in movies
        if any(clean(m.get(f)) == "" for f in SOUP_FIELDS)
    ]
    if incomplete:
        m = incomplete[0]
        print("\n" + "=" * 70)
        print("SAMPLE WITH A MISSING FIELD (checks N/A handling)")
        print("=" * 70)
        blanks = [f for f in SOUP_FIELDS if clean(m.get(f)) == ""]
        print(f"\n{m.get('Title', '?')} — missing: {', '.join(blanks)}")
        for f in SOUP_FIELDS:
            print(f"  {f:<9}: {m.get(f)}")
        print(f"  soup     : {m['soup'][:300]}"
              + ("..." if len(m['soup']) > 300 else ""))

    # --- automated sanity checks -----------------------------------------
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    soups = [m["soup"] for m in movies]
    # Case-sensitive and literal: a loose /\bN\/?A\b/i also matches real names
    # such as the actor "Na Moon-hee".
    na_leaks = [m["Title"] for m in movies if re.search(r"\bN/A\b", m["soup"])]
    none_leaks = [m["Title"] for m in movies if "None" in m["soup"]]
    empty = [m["Title"] for m in movies if not m["soup"]]
    doubled = [m["Title"] for m in movies if "  " in m["soup"]]

    lengths = [len(s) for s in soups]
    checks = [
        ('no "N/A" in any soup', na_leaks),
        ('no "None" in any soup', none_leaks),
        ("no empty soups", empty),
        ("no double spaces", doubled),
    ]
    for label, offenders in checks:
        status = "PASS" if not offenders else f"FAIL ({len(offenders)})"
        print(f"  {label:<24} {status}")
        if offenders:
            print(f"      e.g. {offenders[:3]}")

    print(f"\n  soups built              {len(soups)}")
    print(f"  length min/mean/max      {min(lengths)} / {sum(lengths)//len(lengths)} / {max(lengths)}")

    if args.save:
        with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
            json.dump(movies, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote {OUTPUT_FILE.name} ({len(movies)} records, each with a 'soup' key)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
