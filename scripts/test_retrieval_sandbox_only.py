"""
SANDBOX-ONLY TEST SCRIPT.

This exists only because my current dev sandbox blocks the download of
Chroma's default ONNX embedding model (it lives on an S3 bucket outside
this environment's network allowlist). It is NOT part of the real project.

It reuses the exact same chunk_markdown() function from scripts/ingest.py
and swaps in a TF-IDF vectorizer (scikit-learn, already installed) just to
prove the chunking + retrieval logic actually surfaces the right answers
for representative support questions. On your own machine, ingest.py runs
as-is with the real all-MiniLM-L6-v2 embeddings and no substitution needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ingest import KB_DIR, chunk_markdown

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Build chunks exactly as the real pipeline would ---
all_chunks = []
for md_file in sorted(KB_DIR.glob("*.md")):
    text = md_file.read_text(encoding="utf-8")
    all_chunks.extend(chunk_markdown(text, source=md_file.stem))

texts = [c["text"] for c in all_chunks]
vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)

# --- Representative test queries a real customer would actually type ---
test_queries = [
    "how long do I have to return something",
    "I paid cash on delivery, when do I get my refund",
    "my order says shipped but tracking hasn't moved in days",
    "can I cancel my order after it's already been sent",
    "they sent me the wrong color, what do I do",
    "is there a fee for installments",
]

print(f"{len(all_chunks)} chunks indexed from {len(set(c['source'] for c in all_chunks))} documents\n")
print("=" * 70)

for q in test_queries:
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = sims.argsort()[::-1][:2]
    print(f"\nQuery: \"{q}\"")
    for rank, idx in enumerate(top_idx, 1):
        chunk = all_chunks[idx]
        preview = chunk["text"].split("\n")[1][:90] if "\n" in chunk["text"] else chunk["text"][:90]
        print(f"  #{rank} [{chunk['title']}] (score {sims[idx]:.3f}) -> {preview}...")
