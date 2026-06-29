"""SANDBOX-ONLY. Verifies product search logic (semantic match + structured
filtering/sorting) against the schema-accurate sample catalog, using a
TF-IDF substitute for embeddings since the real ONNX model can't download
in this network-restricted sandbox. On your machine, ingest_products.py +
product_search.py use real embeddings and the real downloaded catalog."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.ingest_products import CATALOG_FILE, product_to_text

products = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
texts = [product_to_text(p) for p in products]
vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)


def search(query, max_price=None, min_rating=None, store=None, sort_by=None, top_k=5, pool=25):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    order = sims.argsort()[::-1][:pool]
    candidates = []
    for i in order:
        p = dict(products[i])
        p["relevance"] = float(sims[i])
        candidates.append(p)

    def keep(p):
        if max_price is not None and p["price"] > max_price:
            return False
        if min_rating is not None and p["average_rating"] < min_rating:
            return False
        if store is not None and store.lower() not in p["store"].lower():
            return False
        return True

    filtered = [p for p in candidates if keep(p)]
    if sort_by == "rating":
        filtered.sort(key=lambda p: (p["average_rating"], p["rating_number"]), reverse=True)
    elif sort_by == "price_low":
        filtered.sort(key=lambda p: p["price"])
    return filtered[:top_k]


def show(label, results):
    print(f"\n{label}")
    for p in results:
        print(f"  - {p['title'][:50]} | {p['store']} | ${p['price']} | {p['average_rating']}* ({p['rating_number']})")


print("=" * 70)
show("'gift for women' (semantic):", search("gift for women", top_k=3))
show("'gift for women' UNDER $30, sorted by rating:",
     search("gift for women", max_price=30, sort_by="rating", top_k=3))
show("'headphones or speaker' from SoundCore:",
     search("headphones speaker music", store="SoundCore", top_k=3))
show("'something for kids' top rated:",
     search("toy for kids", sort_by="rating", top_k=2))
show("'shoes' (semantic):", search("running shoes", top_k=2))
print("\n" + "=" * 70)
print("If these return sensible products with correct filters applied, the logic is sound.")
