"""Phase 5: load the movie embeddings into a persistent ChromaDB collection.

Reads model/embeddings.pkl (Phase 4) and upserts every vector into a local
Chroma collection named "movies", with Title/Genre/Director/imdbID attached as
metadata. The database lives on disk under chroma_db/, so this only needs to
run when the embeddings change.

Usage:
    python load_chroma.py
    python load_chroma.py --reset     # drop and rebuild the collection
"""

import argparse
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).parent
EMBEDDINGS_FILE = HERE / "model" / "embeddings.pkl"
CHROMA_DIR = HERE / "chroma_db"
COLLECTION_NAME = "movies"
BATCH_SIZE = 200


def clean_meta(value: object) -> str:
    """Chroma rejects None metadata values, and OMDb's "N/A" is noise."""
    if not isinstance(value, str) or value.strip().upper() == "N/A":
        return ""
    return value.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true", help="delete the collection before loading"
    )
    args = parser.parse_args()

    if not EMBEDDINGS_FILE.exists():
        print(f"{EMBEDDINGS_FILE.name} not found — run embed_movies.py first")
        return 1

    with EMBEDDINGS_FILE.open("rb") as fh:
        payload = pickle.load(fh)

    embeddings = payload["embeddings"]
    movies = payload["movies"]
    print(f"Loaded {len(movies)} movies, embeddings {embeddings.shape} "
          f"({payload['model_name']})")

    if len(movies) != embeddings.shape[0]:
        print("metadata and embeddings are misaligned — re-run embed_movies.py")
        return 1

    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Dropped existing '{COLLECTION_NAME}' collection")
        except Exception:
            pass  # nothing to drop on a first run

    # embedding_function=None: we supply vectors ourselves, so Chroma must not
    # try to embed anything with its own default model.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )

    # --- build the records -------------------------------------------------
    ids, metadatas, vectors = [], [], []
    seen: set[str] = set()
    duplicates = 0

    for i, movie in enumerate(movies):
        imdb_id = clean_meta(movie.get("imdbID")) or f"row-{i}"
        if imdb_id in seen:
            duplicates += 1
            continue
        seen.add(imdb_id)

        ids.append(imdb_id)
        metadatas.append({
            "Title": clean_meta(movie.get("Title")),
            "Genre": clean_meta(movie.get("Genre")),
            "Director": clean_meta(movie.get("Director")),
            "imdbID": imdb_id,
        })
        vectors.append(embeddings[i].tolist())

    if duplicates:
        print(f"skipped {duplicates} duplicate imdbIDs")

    # --- upsert ------------------------------------------------------------
    # upsert, not add: re-running overwrites rather than erroring on existing ids.
    print(f"\nUpserting {len(ids)} movies in batches of {BATCH_SIZE}...")
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  {min(end, len(ids))}/{len(ids)}")

    # --- verify ------------------------------------------------------------
    count = collection.count()
    print("\n" + "=" * 70)
    print("COLLECTION")
    print("=" * 70)
    print(f"  name              {collection.name}")
    print(f"  path              {CHROMA_DIR.relative_to(HERE)}/")
    print(f"  distance metric   {collection.metadata or 'cosine (configured)'}")
    print(f"  total items       {count}")
    print(f"  expected          {len(movies)}")
    print(f"  match             {'PASS' if count == len(movies) else 'FAIL'}")

    sample = collection.get(ids=[ids[0]], include=["metadatas", "embeddings"])
    print("\n  sample record:")
    print(f"    id        {sample['ids'][0]}")
    for key, value in sample["metadatas"][0].items():
        print(f"    {key:<9} {value}")
    print(f"    vector    {len(sample['embeddings'][0])} dims")

    # A stored vector is only useful if it can be searched; prove it round-trips
    # by querying with movie 0's own embedding — it must return itself first.
    probe = collection.query(
        query_embeddings=[embeddings[0].tolist()],
        n_results=3,
        include=["metadatas", "distances"],
    )
    print("\n  self-query check (nearest neighbours of the first movie):")
    for rank, (meta, dist) in enumerate(
        zip(probe["metadatas"][0], probe["distances"][0]), start=1
    ):
        print(f"    {rank}. {meta['Title'][:45]:<45} distance {dist:.4f}")
    print(f"  returns itself first: "
          f"{'PASS' if probe['ids'][0][0] == ids[0] else 'FAIL'}")

    size = sum(f.stat().st_size for f in CHROMA_DIR.rglob("*") if f.is_file())
    print(f"\n  on-disk size      {size / 1024 / 1024:.1f} MB  (persists across runs)")

    return 0 if count == len(movies) else 1


if __name__ == "__main__":
    sys.exit(main())
