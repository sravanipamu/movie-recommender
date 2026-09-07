"""Phase 6: semantic search over the movie collection.

Embeds a free-text query with the same model used to build the collection,
then returns the nearest movies from ChromaDB.

Usage:
    python recommend.py                          # run the sample queries
    python recommend.py "heist gone wrong"       # one-off query
    python recommend.py --interactive            # type queries in a loop
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
CHROMA_DIR = HERE / "chroma_db"
COLLECTION_NAME = "movies"
MODEL_NAME = "all-MiniLM-L6-v2"  # must match Phase 4, or the vectors won't align

SAMPLE_QUERIES = [
    "assassin who falls in love with his target",
    "a psychological thriller about dreams",
    "space adventure with robots",
    "a father trying to reconnect with his estranged son",
]

_model = None
_collection = None


def get_model():
    """Load the sentence-transformer once and reuse it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection():
    """Open the persistent Chroma collection once and reuse it."""
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # embedding_function=None: queries arrive pre-embedded.
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    return _collection


def recommend_from_search(query: str, n: int = 5) -> list[tuple[str, float]]:
    """Return the n movies closest in meaning to `query`.

    Each tuple is (title, similarity), where similarity is 1 - cosine distance:
    1.0 is identical, 0.0 unrelated.
    """
    model = get_model()
    collection = get_collection()

    # normalize_embeddings must match how the stored vectors were built.
    embedding = model.encode(
        query, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["metadatas", "distances"],
    )

    return [
        (meta.get("Title", "?"), 1.0 - dist)
        for meta, dist in zip(result["metadatas"][0], result["distances"][0])
    ]


def show(query: str, n: int = 5) -> None:
    """Print one query's results with genre context for eyeballing."""
    collection = get_collection()
    model = get_model()

    embedding = model.encode(
        query, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()
    result = collection.query(
        query_embeddings=[embedding], n_results=n, include=["metadatas", "distances"]
    )

    print(f'\n  QUERY: "{query}"')
    print("  " + "-" * 66)
    for rank, (meta, dist) in enumerate(
        zip(result["metadatas"][0], result["distances"][0]), start=1
    ):
        title = meta.get("Title", "?")
        genre = meta.get("Genre", "")
        print(f"    {rank}. {title[:40]:<40} {1.0 - dist:>6.3f}")
        if genre:
            print(f"       {genre}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="a search phrase")
    parser.add_argument("-n", type=int, default=5, help="results per query")
    parser.add_argument(
        "--interactive", action="store_true", help="prompt for queries in a loop"
    )
    args = parser.parse_args()

    if not CHROMA_DIR.exists():
        print("chroma_db/ not found — run load_chroma.py first")
        return 1

    try:
        count = get_collection().count()
    except Exception as exc:
        print(f"could not open the '{COLLECTION_NAME}' collection: {exc}")
        return 1
    print(f"Collection '{COLLECTION_NAME}': {count} movies")

    if args.interactive:
        print("Type a query, or blank to quit.\n")
        while True:
            try:
                query = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                break
            show(query, args.n)
        return 0

    if args.query:
        show(" ".join(args.query), args.n)
        return 0

    print("\n" + "=" * 70)
    print("SAMPLE QUERIES")
    print("=" * 70)
    for query in SAMPLE_QUERIES:
        show(query, args.n)

    # The headline test: none of these words need appear in any plot for the
    # right movies to surface — that is what separates embeddings from TF-IDF.
    print("\n" + "=" * 70)
    print("SEMANTIC vs KEYWORD")
    print("=" * 70)
    probe = "assassin who falls in love with his target"
    top = recommend_from_search(probe, n=5)
    print(f'\n  Query: "{probe}"')
    print("  Returned as (title, similarity) tuples:\n")
    for title, score in top:
        print(f"    ({title!r}, {score:.3f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
