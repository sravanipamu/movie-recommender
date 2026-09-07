"""Phase 3: turn each movie's soup into a TF-IDF vector.

Reads the Phase 2 output, fits a TfidfVectorizer over every soup, and saves
the fitted vectorizer plus the resulting matrix so Phase 4 can load them
instead of refitting.

Usage:
    python vectorize.py
    python vectorize.py --top 25      # show more vocabulary terms
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).parent
SOUP_FILE = HERE / "movies_with_soup.json"
MODEL_DIR = HERE / "model"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MATRIX_FILE = MODEL_DIR / "tfidf_matrix.joblib"
INDEX_FILE = MODEL_DIR / "movie_index.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top", type=int, default=15, help="how many top terms to print (default: 15)"
    )
    args = parser.parse_args()

    if not SOUP_FILE.exists():
        print(f"{SOUP_FILE.name} not found — run build_soup.py --save first")
        return 1

    with SOUP_FILE.open(encoding="utf-8") as fh:
        movies = json.load(fh)

    soups = [m.get("soup", "") for m in movies]
    if not all(soups):
        print(f"warning: {sum(1 for s in soups if not s)} movies have an empty soup")

    print(f"Loaded {len(movies)} movies from {SOUP_FILE.name}")

    # --- fit ---------------------------------------------------------------
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(soups)

    print("\n" + "=" * 70)
    print("TF-IDF MATRIX")
    print("=" * 70)
    n_movies, n_terms = matrix.shape
    print(f"  shape          {matrix.shape}   [movies x vocabulary]")
    print(f"  movies         {n_movies}")
    print(f"  vocabulary     {n_terms}")
    print(f"  stored values  {matrix.nnz:,} of {n_movies * n_terms:,} cells")
    print(f"  density        {matrix.nnz / (n_movies * n_terms):.4%}  (sparse, as expected)")

    # --- top terms ---------------------------------------------------------
    # Summing each column gives a term's total weight across the whole corpus.
    summed = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    top_idx = summed.argsort()[::-1][: args.top]

    # How many movies each term appears in, for context on *why* it ranks high.
    doc_counts = np.asarray((matrix > 0).sum(axis=0)).ravel()

    print("\n" + "=" * 70)
    print(f"TOP {args.top} TERMS BY SUMMED TF-IDF")
    print("=" * 70)
    print(f"\n  {'#':>3}  {'term':<16} {'summed tf-idf':>13}  {'in movies':>10}")
    print(f"  {'-' * 3}  {'-' * 16} {'-' * 13}  {'-' * 10}")
    for rank, i in enumerate(top_idx, start=1):
        print(f"  {rank:>3}  {terms[i]:<16} {summed[i]:>13.3f}  {doc_counts[i]:>10}")

    # --- vocabulary sanity -------------------------------------------------
    print("\n" + "=" * 70)
    print("VOCABULARY SANITY")
    print("=" * 70)
    numeric = [t for t in terms if t.isdigit()]
    singletons = int((doc_counts == 1).sum())
    print(f"  purely numeric tokens    {len(numeric):>5}  e.g. {numeric[:5]}")
    print(f"  terms in only 1 movie    {singletons:>5}  ({singletons / n_terms:.0%} of vocabulary)")
    print(f"  shortest / longest term  {min(terms, key=len)!r} / {max(terms, key=len)!r}")

    # --- save --------------------------------------------------------------
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(matrix, MATRIX_FILE)

    # Row order is the only link between the matrix and the movies; persist it
    # so Phase 4 can map row N back to a title without re-reading the soups.
    index = [
        {"row": i, "imdbID": m.get("imdbID"), "Title": m.get("Title"), "Year": m.get("Year")}
        for i, m in enumerate(movies)
    ]
    with INDEX_FILE.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)
    for path in (VECTORIZER_FILE, MATRIX_FILE, INDEX_FILE):
        print(f"  {path.relative_to(HERE)}  ({path.stat().st_size / 1024:.0f} KB)")

    # Confirm the artifacts reload and still agree with what we just built.
    reloaded = joblib.load(MATRIX_FILE)
    assert reloaded.shape == matrix.shape, "reloaded matrix shape mismatch"
    print("\n  reload check   PASS — matrix round-trips at", reloaded.shape)

    return 0


if __name__ == "__main__":
    sys.exit(main())
