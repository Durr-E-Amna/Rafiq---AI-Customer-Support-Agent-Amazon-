"""
Run after step 1's ingest.py has been run, and with GROQ_API_KEY set
(in your .env file).

Usage:
    python web/run.py

Then open http://127.0.0.1:8000 in Chrome (voice input needs Chrome's
Web Speech API - it doesn't work in Firefox or Safari).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from oauthlib.uri_validate import port
import uvicorn

from agent.llm_client import GroqClient
from agent.retriever import KnowledgeRetriever
from agent.product_loader import load_product_searcher_or_none
from web.server import create_app


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Add it to your .env file:")
        print("  GROQ_API_KEY=your_key_here")
        sys.exit(1)

    llm = GroqClient()
    retriever = KnowledgeRetriever()
    product_searcher = load_product_searcher_or_none()
    app = create_app(llm, retriever, product_searcher)

    print("Starting Rafiq web server at http://127.0.0.1:8000")
    print("Open it in Chrome for voice input support.\n")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
