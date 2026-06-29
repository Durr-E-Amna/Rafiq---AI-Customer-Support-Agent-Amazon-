"""
Ingests the markdown knowledge base into a persistent Chroma collection.

Chunking strategy: split by markdown headers (## sections) rather than fixed
character windows. Each section in our KB articles is already a self-contained
unit of meaning (e.g. "Refund timelines by payment method") - splitting on
headers keeps each chunk semantically whole instead of cutting a table or
a list in half, which is what naive fixed-size chunking would do here.

Embedding model: all-MiniLM-L6-v2, run via Chroma's built-in ONNX-based
embedding function. This is the same model sentence-transformers would give
you, but loaded through ONNX Runtime instead of full PyTorch - smaller
install, faster cold start, identical embedding quality for this use case.
"""

import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
DB_DIR = Path(__file__).parent.parent / "data" / "chroma_store"
COLLECTION_NAME = "rafiq_support_kb"


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Split a markdown doc into chunks at ## headers, keeping the
    top-level title attached to the first chunk for context."""
    lines = text.strip().split("\n")
    title = lines[0].lstrip("# ").strip() if lines[0].startswith("#") else source

    # Split on level-2 headers, keep the header text with its body
    sections = re.split(r"\n(?=## )", text.strip())

    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        # Prepend the document title to every chunk so retrieval still
        # knows which policy area a chunk belongs to, even out of context
        chunk_text = f"[{title}]\n{section}"
        chunks.append(
            {
                "text": chunk_text,
                "source": source,
                "title": title,
                "chunk_index": i,
            }
        )
    return chunks


def build_collection():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Built-in ONNX MiniLM embedding function - downloads the small ONNX
    # model on first run, then caches it locally. No torch, no API calls.
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    # Fresh build each time this script runs, so re-running it after editing
    # KB files always reflects the latest content.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    md_files = sorted(KB_DIR.glob("*.md"))
    print(f"Found {len(md_files)} knowledge base files in {KB_DIR}")

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, source=md_file.stem)
        all_chunks.extend(chunks)
        print(f"  {md_file.name} -> {len(chunks)} chunks")

    collection.add(
        ids=[f"{c['source']}_{c['chunk_index']}" for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[
            {"source": c["source"], "title": c["title"]} for c in all_chunks
        ],
    )

    print(f"\nIngested {len(all_chunks)} chunks into collection '{COLLECTION_NAME}'")
    return collection


if __name__ == "__main__":
    build_collection()
