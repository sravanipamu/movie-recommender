"""Head-to-head: do the embeddings beat TF-IDF at matching meaning?

Each probe is a paraphrase of a movie that IS in the catalog, worded to avoid
the plot's own vocabulary. A keyword model can only find these by luck; an
embedding model should rank the target near the top.

Usage:
    python semantic_test.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).parent

# (paraphrase, exact Title of the movie it describes)
PROBES = [
    ("a boy uses a modified car to visit the era when his parents were young",
     "Back to the Future"),
    ("an intelligent machine turns against the crew of a spacecraft",
     "2001: A Space Odyssey"),
    ("people barricade themselves inside a farmhouse as reanimated corpses attack",
     "Night of the Living Dead"),
    ("two lonely neighbours form a quiet bond after learning their spouses are unfaithful",
     "In the Mood for Love"),
    ("a deranged officer triggers a nuclear crisis while politicians scramble to stop it",
     "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb"),
]


def rank_of(titles: list[str], target: str) -> int | None:
    """1-based position of `target` in `titles`, or None if absent."""
    for i, t in enumerate(titles, start=1):
        if t == target:
            return i
    return None


def main() -> int:
    with (HERE / "movies_with_soup.json").open(encoding="utf-8") as fh:
        movies = json.load(fh)
    titles = [m["Title"] for m in movies]

    # --- TF-IDF side (Phase 3 artifacts) ----------------------------------
    vectorizer = joblib.load(HERE / "model" / "tfidf_vectorizer.joblib")
    tfidf_matrix = joblib.load(HERE / "model" / "tfidf_matrix.joblib")

    # --- embedding side (Phase 4/5 artifacts) -----------------------------
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    import chromadb

    collection = chromadb.PersistentClient(
        path=str(HERE / "chroma_db")
    ).get_collection("movies", embedding_function=None)

    print("=" * 78)
    print("PARAPHRASE TEST — can each model find the movie being described?")
    print(f"searching {len(movies)} movies; rank 1 = perfect, '-' = not in top 20")
    print("=" * 78)

    emb_ranks, tfidf_ranks = [], []

    for probe, target in PROBES:
        # TF-IDF: reuse the fitted vectorizer, never refit.
        q_vec = vectorizer.transform([probe])
        sims = cosine_similarity(q_vec, tfidf_matrix).ravel()
        order = sims.argsort()[::-1][:20]
        tfidf_titles = [titles[i] for i in order]
        t_rank = rank_of(tfidf_titles, target)

        # Embeddings.
        q_emb = model.encode(probe, normalize_embeddings=True).tolist()
        res = collection.query(
            query_embeddings=[q_emb], n_results=20, include=["metadatas"]
        )
        emb_titles = [m["Title"] for m in res["metadatas"][0]]
        e_rank = rank_of(emb_titles, target)

        emb_ranks.append(e_rank)
        tfidf_ranks.append(t_rank)

        print(f'\n  "{probe}"')
        print(f"    looking for: {target[:60]}")
        print(f"    embeddings  rank {e_rank or '-':<3}  top hit: {emb_titles[0][:44]}")
        print(f"    tf-idf      rank {t_rank or '-':<3}  top hit: {tfidf_titles[0][:44]}")

    # --- scoreboard --------------------------------------------------------
    def summarise(ranks: list[int | None]) -> str:
        top1 = sum(1 for r in ranks if r == 1)
        top5 = sum(1 for r in ranks if r and r <= 5)
        found = sum(1 for r in ranks if r)
        return f"top-1: {top1}/{len(ranks)}   top-5: {top5}/{len(ranks)}   found: {found}/{len(ranks)}"

    print("\n" + "=" * 78)
    print("SCOREBOARD")
    print("=" * 78)
    print(f"  embeddings   {summarise(emb_ranks)}")
    print(f"  tf-idf       {summarise(tfidf_ranks)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
