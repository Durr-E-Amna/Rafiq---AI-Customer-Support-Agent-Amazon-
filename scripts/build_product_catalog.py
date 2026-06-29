"""
Builds a local product catalog from the McAuley-Lab Amazon Reviews 2023
dataset (real product metadata: titles, stores/brands, prices, ratings,
categories), licensed CC BY-SA 4.0.

This is the "real scraped store data, gathered once up front" approach -
NOT live-scraping per query. We pull a bounded sample of real products
from a few categories one time, clean it into a compact local JSON file,
and from then on everything runs offline against that file.

WHY THIS READS PARQUET FILES DIRECTLY, NOT datasets.load_dataset():
This dataset's official HuggingFace config uses a legacy "dataset loading
script" (a custom .py file). As of `datasets` 5.0.0, the library dropped
support for script-based datasets entirely (a real, current breaking
change - confirmed by hitting "Dataset scripts are no longer supported"
directly). The dataset maintainers already auto-converted the raw data to
Parquet files sitting in this same repo, so instead of depending on the
broken script path, we download those Parquet files directly via
huggingface_hub and read them with pyarrow - more efficient anyway, since
we can stream in batches and stop early once we have enough usable rows,
without ever loading a full multi-hundred-MB file into memory at once.

Run once:
    python scripts/build_product_catalog.py

Needs internet for this one run (downloads a handful of Parquet files,
roughly 750MB total across all 5 categories below - each file is cached
by huggingface_hub, so re-running this script doesn't re-download).
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

OUT_DIR = Path(__file__).parent.parent / "data" / "product_catalog"
OUT_FILE = OUT_DIR / "products.json"

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

# (HF repo folder, the specific Parquet shard to pull, friendly label).
# Only the FIRST shard of each category is used - plenty of rows for a
# few hundred usable products without downloading every shard. Categories
# chosen because they're confirmed available as Parquet (some categories,
# like Amazon_Fashion and Home_and_Kitchen, were never auto-converted and
# 404 - verified directly against the repo's actual file listing, not
# assumed) and are reasonably sized rather than multi-gigabyte.
CATEGORIES = [
    ("raw_meta_All_Beauty", "full-00000-of-00001.parquet", "Beauty"),
    ("raw_meta_Handmade_Products", "full-00000-of-00001.parquet", "Handmade & Gifts"),
    ("raw_meta_Musical_Instruments", "full-00000-of-00002.parquet", "Musical Instruments"),
    ("raw_meta_Toys_and_Games", "full-00000-of-00005.parquet", "Toys & Games"),
    ("raw_meta_Cell_Phones_and_Accessories", "full-00000-of-00007.parquet", "Cell Phones & Accessories"),
]

# How many usable products to keep per category. Kept modest so first-time
# ingest (embedding every product) stays reasonable on a laptop.
PER_CATEGORY = 400

# We scan more than PER_CATEGORY rows because many get filtered out for
# missing price/title/rating, and we want PER_CATEGORY *clean* ones.
SCAN_LIMIT = 4000

BATCH_SIZE = 200


def clean_price(raw) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return round(float(raw), 2)
    s = str(raw).replace("$", "").strip()
    if s.lower() in ("none", "", "nan"):
        return None
    s = s.split("-")[0].strip()  # some prices are ranges like "9.99 - 14.99"
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def usable(product: dict) -> bool:
    """Keep only products good enough to recommend - real title, a real
    price, and at least some rating signal. A catalog full of priceless,
    unrated items makes for a bad shopping assistant."""
    if not product.get("title") or len(product["title"]) < 5:
        return False
    if product.get("price") is None:
        return False
    if not product.get("average_rating") or not product.get("rating_number"):
        return False
    return True


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_products = []

    for repo_folder, filename, label in CATEGORIES:
        print(f"\nFetching {label} ({repo_folder})...")
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=f"{repo_folder}/{filename}",
            )
        except Exception as e:
            print(f"  Could not download {repo_folder}: {e}")
            print("  Skipping this category - the rest will still build.")
            continue

        kept = 0
        scanned = 0
        parquet_file = pq.ParquetFile(local_path)

        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
            for item in batch.to_pylist():
                scanned += 1
                if scanned > SCAN_LIMIT or kept >= PER_CATEGORY:
                    break

                price = clean_price(item.get("price"))
                product = {
                    "id": item.get("parent_asin"),
                    "title": (item.get("title") or "").strip(),
                    "store": (item.get("store") or "").strip() or "Unknown store",
                    "category": label,
                    "price": price,
                    "average_rating": item.get("average_rating"),
                    "rating_number": item.get("rating_number"),
                    "features": list(item.get("features") or []),
                    "description": " ".join(item.get("description") or [])[:500],
                }

                if usable(product):
                    all_products.append(product)
                    kept += 1

            if scanned > SCAN_LIMIT or kept >= PER_CATEGORY:
                break

        print(f"  scanned {scanned} rows, kept {kept} usable products")

    OUT_FILE.write_text(json.dumps(all_products, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_products)} products to {OUT_FILE}")
    print("Next: run `python scripts/ingest_products.py` to embed them into Chroma.")


if __name__ == "__main__":
    build()
