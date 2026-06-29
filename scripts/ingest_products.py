"""
Ingests the product catalog (data/product_catalog/products.json) into its
own Chroma collection, separate from the policy knowledge base.

Why a separate collection: policy docs and product listings are different
*kinds* of knowledge answering different *kinds* of questions ("what's the
return window" vs "recommend a gift under $30"). Mixing them in one
collection would let a policy chunk and a product chunk compete for the
same query and pollute each other's results. Two collections, each
retrieved for the right intent, keeps both clean.

Each product becomes one document whose embedded text is a natural-language
description (title + category + store + features + description), so semantic
search works on meaning ("something relaxing for my mum" -> diffuser,
candles). Structured fields (price, rating, category, store) are stored as
metadata so they can be used for filtering and sorting after retrieval.

Run after build_product_catalog.py:
    python scripts/ingest_products.py
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

CATALOG_FILE = Path(__file__).parent.parent / "data" / "product_catalog" / "products.json"
DB_DIR = Path(__file__).parent.parent / "data" / "chroma_store"
COLLECTION_NAME = "rafiq_products"


def product_to_text(p: dict) -> str:
    """The text that actually gets embedded - written so semantic search
    matches on what a shopper means, not just exact words."""
    parts = [p["title"], f"Category: {p['category']}", f"Brand/store: {p['store']}"]
    if p.get("features"):
        parts.append("Features: " + "; ".join(p["features"]))
    if p.get("description"):
        parts.append(p["description"])
    return ". ".join(parts)


def build_collection():
    if not CATALOG_FILE.exists():
        raise SystemExit(
            f"{CATALOG_FILE} not found. Run `python scripts/build_product_catalog.py` first."
        )

    products = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(products)} products from {CATALOG_FILE}")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas = [], [], []
    for i, p in enumerate(products):
        # Chroma requires a unique id; fall back to index if parent_asin missing
        product_id = p.get("id") or f"product_{i}"
        ids.append(product_id)
        documents.append(product_to_text(p))
        metadatas.append(
            {
                "asin": product_id,
                "title": p["title"],
                "store": p["store"],
                "category": p["category"],
                "price": float(p["price"]) if p.get("price") is not None else -1.0,
                "average_rating": float(p.get("average_rating") or 0.0),
                "rating_number": int(p.get("rating_number") or 0),
            }
        )

    # De-duplicate ids (some datasets repeat parent_asin) to avoid Chroma errors
    seen = set()
    f_ids, f_docs, f_meta = [], [], []
    for _id, doc, meta in zip(ids, documents, metadatas):
        if _id in seen:
            continue
        seen.add(_id)
        f_ids.append(_id)
        f_docs.append(doc)
        f_meta.append(meta)

    collection.add(ids=f_ids, documents=f_docs, metadatas=f_meta)
    print(f"Ingested {len(f_ids)} products into collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    build_collection()
