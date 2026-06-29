"""
Product search - semantic retrieval over the product catalog, plus the
structured filtering/sorting that shopping questions actually need.

Semantic search alone gets you "things related to relaxing gifts"; real
shopping queries also carry hard constraints ("under $30", "top rated",
"from SoundCore") that aren't a similarity question - they're filters and
sorts over structured fields. This wraps both: Chroma finds the
semantically relevant candidates, then we filter/sort them by the
structured metadata (price, rating, store) we stored at ingest time.

Mirrors agent/retriever.py's structure so the agent graph treats product
search like any other retrieval step.
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DB_DIR = Path(__file__).parent.parent / "data" / "chroma_store"
COLLECTION_NAME = "rafiq_products"

# Below this similarity, a "match" isn't actually relevant - it's just the
# least-bad option in the candidate pool. Without this floor, a query for
# something genuinely not in the catalog (e.g. "headphones" when the
# catalog has no headphones) would still return its closest lexical
# neighbor (e.g. "headband") and present it as if it were a real match -
# which is worse than honestly saying nothing matched.
MIN_RELEVANCE = 0.25


class ProductSearcher:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(DB_DIR))
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embed_fn
        )

    def search(
        self,
        query: str,
        max_price: float | None = None,
        min_price: float | None = None,
        min_rating: float | None = None,
        store: str | None = None,
        sort_by: str | None = None,  # "rating", "price_low", "price_high"
        top_k: int = 5,
        candidate_pool: int = 25,
    ) -> list[dict]:
        """Return a ranked list of product dicts matching the query and any
        structured constraints. We over-fetch a candidate pool semantically,
        then apply hard filters and sorting, so constraints like 'under $30'
        are exact, not approximate."""
        results = self.collection.query(query_texts=[query], n_results=candidate_pool)

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        candidates = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            candidates.append({
                **meta,
                "relevance": 1 - dist,
                # Real Amazon product page, built from the actual ASIN we
                # stored at ingest time - not a placeholder or fake link.
                "url": f"https://www.amazon.com/dp/{meta.get('asin', '')}" if meta.get("asin") else None,
            })

        # Hard structured filters
        def keep(p: dict) -> bool:
            if p["relevance"] < MIN_RELEVANCE:
                return False
            if max_price is not None and (p["price"] < 0 or p["price"] > max_price):
                return False
            if min_price is not None and p["price"] < min_price:
                return False
            if min_rating is not None and p["average_rating"] < min_rating:
                return False
            if store is not None and store.lower() not in p["store"].lower():
                return False
            return True

        filtered = [p for p in candidates if keep(p)]

        # Sorting
        if sort_by == "rating":
            filtered.sort(key=lambda p: (p["average_rating"], p["rating_number"]), reverse=True)
        elif sort_by == "price_low":
            filtered.sort(key=lambda p: p["price"])
        elif sort_by == "price_high":
            filtered.sort(key=lambda p: p["price"], reverse=True)
        # else: keep semantic relevance order

        return filtered[:top_k]
