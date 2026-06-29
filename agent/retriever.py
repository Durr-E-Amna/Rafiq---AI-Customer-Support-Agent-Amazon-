"""
Retriever - queries the Chroma collection built by scripts/ingest.py.

This is the real production retriever. It's wrapped in a small class so
the agent graph can call .retrieve() without caring whether the underlying
store is Chroma (here) or a test substitute (see
scripts/test_agent_sandbox_only.py, which swaps this for a TF-IDF stand-in
since the real ONNX embedding model can't download in a network-restricted
dev sandbox).
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DB_DIR = Path(__file__).parent.parent / "data" / "chroma_store"
COLLECTION_NAME = "rafiq_support_kb"

# Below this similarity score, we don't trust the retrieved context enough
# to answer from it - this is the confidence threshold the eval set (step 3)
# will help us tune properly rather than guess at.
CONFIDENCE_THRESHOLD = 0.35


class KnowledgeRetriever:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(DB_DIR))
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embed_fn
        )

    def retrieve(self, query: str, top_k: int = 3) -> dict:
        results = self.collection.query(query_texts=[query], n_results=top_k)

        docs = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        # Chroma returns cosine *distance*; convert to similarity (1 - distance)
        # so "higher is better" matches intuition everywhere else in the code.
        similarities = [1 - d for d in distances]
        top_confidence = similarities[0] if similarities else 0.0

        return {
            "chunks": docs,
            "sources": [m["title"] for m in metadatas],
            "confidence": top_confidence,
            "low_confidence": top_confidence < CONFIDENCE_THRESHOLD,
        }
