"""Phase 4: replace TF-IDF with real sentence embeddings.

Encodes every movie's soup with all-MiniLM-L6-v2 (runs locally, no API key)
and saves the vectors alongside movie metadata for the next phase.

Where TF-IDF only matched literal shared words, these embeddings capture
meaning — two movies can score as similar without using the same vocabulary.

Usage:
    python embed_movies.py
    python embed_movies.py --batch-size 32
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SOUP_FILE = HERE / "movies_with_soup.json"
OUTPUT_FILE = HERE / "model" / "embeddings.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_DIM = 384

# Metadata carried alongside the vectors so the next phase needs only this file.
META_FIELDS = ["imdbID", "Title", "Year", "Genre", "Director", "Actors", "imdbRating"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64, help="encode batch size")
    args = parser.parse_args()

    if not SOUP_FILE.exists():
        print(f"{SOUP_FILE.name} not found — run build_soup.py --save first")
        return 1

    with SOUP_FILE.open(encoding="utf-8") as fh:
        movies = json.load(fh)

    soups = [m.get("soup", "") for m in movies]
    print(f"Loaded {len(movies)} movies from {SOUP_FILE.name}")

    empty = sum(1 for s in soups if not s.strip())
    if empty:
        print(f"warning: {empty} movies have an empty soup and will embed as noise")

    # Importing here keeps the missing-data errors above fast to hit.
    print(f"\nLoading {MODEL_NAME} (first run downloads ~90 MB)...")
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as exc:  # network, disk, or hub failure
        print(f"could not load the model: {exc}")
        return 1

    print(f"  device            {model.device}")
    print(f"  max input tokens  {model.max_seq_length}")

    # --- encode ------------------------------------------------------------
    print(f"\nEncoding {len(soups)} soups in batches of {args.batch_size}...")
    started = time.perf_counter()
    embeddings = model.encode(
        soups,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit length: dot product == cosine similarity
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - started

    print("\n" + "=" * 70)
    print("EMBEDDINGS")
    print("=" * 70)
    print(f"  matrix shape        {embeddings.shape}   [movies x dimensions]")
    print(f"  one movie's vector  {embeddings[0].shape}   <- {movies[0].get('Title')}")
    print(f"  dtype               {embeddings.dtype}")
    print(f"  encode time         {elapsed:.1f}s  ({len(soups) / elapsed:.0f} movies/sec)")

    dim = embeddings.shape[1]
    print(f"\n  expected {EXPECTED_DIM} dimensions: "
          f"{'PASS' if dim == EXPECTED_DIM else f'FAIL (got {dim})'}")

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  all unit length:     "
          f"{'PASS' if np.allclose(norms, 1.0, atol=1e-5) else 'FAIL'} "
          f"(min {norms.min():.4f}, max {norms.max():.4f})")
    print(f"  no NaN/inf:          {'PASS' if np.isfinite(embeddings).all() else 'FAIL'}")

    print(f"\n  first 8 values of vector 0:\n    {np.round(embeddings[0][:8], 4)}")

    # --- save --------------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    payload = {
        "model_name": MODEL_NAME,
        "dim": dim,
        "embeddings": embeddings,
        "normalized": True,
        # Row i of `embeddings` describes movies[i] — same order, always.
        "movies": [{k: m.get(k) for k in META_FIELDS} for m in movies],
    }
    with OUTPUT_FILE.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)
    print(f"  {OUTPUT_FILE.relative_to(HERE)}  ({OUTPUT_FILE.stat().st_size / 1024:.0f} KB)")

    with OUTPUT_FILE.open("rb") as fh:
        reloaded = pickle.load(fh)
    ok = (
        reloaded["embeddings"].shape == embeddings.shape
        and len(reloaded["movies"]) == len(movies)
        and reloaded["movies"][0]["Title"] == movies[0].get("Title")
    )
    print(f"  reload check   {'PASS' if ok else 'FAIL'} — "
          f"{reloaded['embeddings'].shape}, {len(reloaded['movies'])} movies aligned")

    return 0


if __name__ == "__main__":
    sys.exit(main())
