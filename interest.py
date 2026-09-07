"""Phase 3: turn a session's event history into a live "interest vector".

Every event the visitor generates is a hint about what they want:

    search  — embed the query text with all-MiniLM-L6-v2
    view    — reuse the movie's stored vector from ChromaDB (never re-embed)

Averaging those vectors puts the session at a point in the same embedding
space as the movies, so "what should we recommend?" becomes an ordinary
nearest-neighbour query. Recent events count double, so the profile follows
what the visitor just did instead of being anchored by their whole history.

The vectors in the collection are unit length (embed_movies.py normalises
before saving), so a normalised average is a true cosine centroid.

Usage as a script — useful for eyeballing a real session:
    python interest.py                 # newest session in events.db
    python interest.py <session_id>
"""

import sys
import threading
from pathlib import Path

import numpy as np

import tracking

HERE = Path(__file__).parent
CHROMA_DIR = HERE / "chroma_db"
COLLECTION_NAME = "movies"
MODEL_NAME = "all-MiniLM-L6-v2"  # must match embed_movies.py or nothing aligns

# How far back to look. Beyond this the oldest events carry almost no signal
# anyway, and the query/embed work grows for nothing.
HISTORY_LIMIT = 50

# Recency weighting: the newest RECENT_N events count RECENT_WEIGHT each,
# everything older counts BASE_WEIGHT. Two clicks are enough to visibly move
# the profile, but they cannot erase it either.
RECENT_N = 3
RECENT_WEIGHT = 2.0
BASE_WEIGHT = 1.0

_model = None
_collection = None
# Sync endpoints run on uvicorn's thread pool, so two requests can race into a
# cold process. The lock means the model is loaded once, not once per thread.
_lock = threading.Lock()


def get_model():
    """Load the sentence-transformer once and reuse it (a few seconds, once)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection():
    """Open the persistent Chroma collection once and reuse it."""
    global _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                import chromadb

                client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                # embedding_function=None: we always supply vectors ourselves.
                _collection = client.get_collection(
                    COLLECTION_NAME, embedding_function=None
                )
    return _collection


def is_ready() -> bool:
    """True when the collection exists on disk, so callers can degrade politely."""
    return CHROMA_DIR.exists()


def _unit(vector: np.ndarray) -> np.ndarray | None:
    """Scale to length 1. None if the vector is all zeros (nothing to point at)."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return vector / norm


def _weight_for(index: int) -> float:
    """Newest-first index -> weight. Position 0/1/2 are the freshest events."""
    return RECENT_WEIGHT if index < RECENT_N else BASE_WEIGHT


def _stored_vectors(imdb_ids: list[str]) -> dict[str, np.ndarray]:
    """Fetch view embeddings from Chroma in one call, keyed by imdbID.

    Ids the collection does not know about are simply absent from the result,
    which is why the caller keys by id rather than trusting position.
    """
    if not imdb_ids:
        return {}

    got = get_collection().get(ids=imdb_ids, include=["embeddings"])
    vectors = got.get("embeddings")
    if vectors is None or len(vectors) == 0:
        return {}

    return {
        found_id: np.asarray(vectors[i], dtype=np.float32)
        for i, found_id in enumerate(got["ids"])
    }


def build_interest_vector(session_id: str) -> np.ndarray | None:
    """The session's centre of gravity in embedding space.

    Returns a unit-length float32 vector, or None when there is nothing to go
    on — no events at all, or none of them produced a usable vector.
    """
    events = tracking.events_for(session_id, limit=HISTORY_LIMIT)  # newest first
    if not events:
        return None

    # Batch the two expensive lookups instead of doing them per event.
    queries = [
        (i, e["query"].strip())
        for i, e in enumerate(events)
        if e["type"] == "search" and (e.get("query") or "").strip()
    ]
    view_ids = [
        e["imdb_id"] for e in events if e["type"] == "view" and e.get("imdb_id")
    ]

    query_vectors: dict[int, np.ndarray] = {}
    if queries:
        encoded = get_model().encode(
            [text for _, text in queries],
            convert_to_numpy=True,
            normalize_embeddings=True,  # same as the stored vectors
        )
        query_vectors = {
            index: np.asarray(encoded[row], dtype=np.float32)
            for row, (index, _) in enumerate(queries)
        }

    stored = _stored_vectors(sorted(set(view_ids)))

    total = None
    total_weight = 0.0

    for index, event in enumerate(events):
        if event["type"] == "search":
            vector = query_vectors.get(index)
        else:
            vector = stored.get(event.get("imdb_id"))
        if vector is None:  # unusable event — empty query, or a movie we never embedded
            continue

        weight = _weight_for(index)
        contribution = vector * weight
        total = contribution if total is None else total + contribution
        total_weight += weight

    if total is None or total_weight == 0.0:
        return None

    return _unit(total / total_weight)


def recommend_for_session(
    session_id: str,
    n: int = 12,
    exclude: set[str] | None = None,
    vector: np.ndarray | None = None,
) -> list[dict]:
    """Nearest movies to the session's interest vector, best first.

    Each item is {"imdbID", "title", "genre", "score"}, where score is
    1 - cosine distance. Returns [] for a session with no usable history —
    the caller decides what to show instead.

    Pass `vector` when you have already built it: a caller that needs to branch
    on "is there a profile at all?" shouldn't pay to embed the history twice.
    """
    if vector is None:
        vector = build_interest_vector(session_id)
    if vector is None:
        return []

    exclude = exclude or set()

    # Over-fetch so the excluded ids can be dropped and still leave n results.
    collection = get_collection()
    want = min(n + len(exclude), max(collection.count(), 1))

    result = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=want,
        include=["metadatas", "distances"],
    )

    out: list[dict] = []
    for imdb_id, meta, distance in zip(
        result["ids"][0], result["metadatas"][0], result["distances"][0]
    ):
        if imdb_id in exclude:
            continue
        out.append({
            "imdbID": imdb_id,
            "title": meta.get("Title") or "?",
            "genre": meta.get("Genre") or None,
            "score": round(1.0 - float(distance), 4),
        })
        if len(out) == n:
            break
    return out


def explain(session_id: str) -> dict:
    """What went into the profile, for the debug endpoint and the CLI below."""
    events = tracking.events_for(session_id, limit=HISTORY_LIMIT)
    return {
        "session_id": session_id,
        "events_used": len(events),
        "recent_n": RECENT_N,
        "recent_weight": RECENT_WEIGHT,
        "base_weight": BASE_WEIGHT,
        "contributions": [
            {
                "position": i,
                "type": e["type"],
                "signal": e["query"] if e["type"] == "search" else e["imdb_id"],
                "weight": _weight_for(i),
                "created_at": e["created_at"],
            }
            for i, e in enumerate(events)
        ],
    }


def _newest_session() -> str | None:
    with tracking.connect() as conn:
        row = conn.execute(
            "SELECT session_id FROM events ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row["session_id"] if row else None


def main() -> int:
    if not is_ready():
        print("chroma_db/ not found — run load_chroma.py first")
        return 1

    session_id = sys.argv[1] if len(sys.argv) > 1 else _newest_session()
    if not session_id:
        print("no sessions in events.db yet — browse the site first")
        return 1

    detail = explain(session_id)
    print(f"session {session_id}  ({detail['events_used']} events)\n")
    for c in detail["contributions"]:
        print(f"  x{c['weight']:.1f}  {c['type']:<6} {str(c['signal'])[:50]}")

    vector = build_interest_vector(session_id)
    if vector is None:
        print("\nno interest vector — nothing usable in this session's history")
        return 0

    print(f"\ninterest vector: {vector.shape[0]} dims, "
          f"norm {float(np.linalg.norm(vector)):.3f}\n")
    seen = {c["signal"] for c in detail["contributions"] if c["type"] == "view"}
    for rank, item in enumerate(recommend_for_session(session_id, exclude=seen), 1):
        print(f"  {rank:>2}. {item['title'][:45]:<45} {item['score']:.3f}")
        if item["genre"]:
            print(f"      {item['genre']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
