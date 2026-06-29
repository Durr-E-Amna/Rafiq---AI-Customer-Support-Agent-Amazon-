"""
Optionally loads the ProductSearcher. Returns None (rather than crashing)
if the product catalog hasn't been built/ingested yet, so the rest of the
agent - orders, returns, refunds, policy - works fine without it. Shopping
queries then get an honest "product search isn't available" reply instead
of a stack trace.

This keeps the product catalog an optional capability layered on top of
the core support agent, not a hard dependency of it.
"""


def load_product_searcher_or_none():
    try:
        from agent.product_search import ProductSearcher

        return ProductSearcher()
    except Exception as e:
        print(f"[info] Product search not available ({type(e).__name__}). "
              "Run scripts/build_product_catalog.py then scripts/ingest_products.py "
              "to enable shopping queries. Orders/returns/refunds/policy work without it.")
        return None
