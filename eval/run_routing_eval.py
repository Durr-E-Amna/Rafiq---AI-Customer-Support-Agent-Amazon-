"""
Run after GROQ_API_KEY is set and scripts/ingest.py has been run.

Usage:
    python eval/run_routing_eval.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.llm_client import GroqClient
from agent.retriever import KnowledgeRetriever
from eval.routing_eval_core import run_routing_eval


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Set it before running:")
        print("  export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    llm = GroqClient()
    retriever = KnowledgeRetriever()
    run_routing_eval(llm, retriever)


if __name__ == "__main__":
    main()
