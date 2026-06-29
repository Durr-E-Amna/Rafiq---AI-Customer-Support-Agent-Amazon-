"""
Run this after step 1's ingest.py has been run at least once, and with
GROQ_API_KEY set in your environment.

Usage:
    export GROQ_API_KEY=your_key_here       (Mac/Linux)
    set GROQ_API_KEY=your_key_here          (Windows cmd)
    python main.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph
from agent.llm_client import GroqClient
from agent.retriever import KnowledgeRetriever
from agent.product_loader import load_product_searcher_or_none


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Set it before running:")
        print("  export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    llm = GroqClient()
    retriever = KnowledgeRetriever()
    product_searcher = load_product_searcher_or_none()
    graph = build_graph(llm, retriever, product_searcher)
    history = []
    session_context = {}

    print("Rafiq support agent — type a message, or 'quit' to exit.\n")
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in ("quit", "exit"):
            break
        if not user_message:
            continue

        result = graph.invoke(
            {
                "user_message": user_message,
                "history": history,
                "session_context": session_context,
                "intent": "",
                "order_id": None,
                "refund_reason": None,
                "shopping": None,
                "products": None,
                "frustration": False,
                "retrieval": None,
                "tool_result": None,
                "escalate": False,
                "escalate_reason": None,
                "needs_clarification": False,
                "response": None,
            }
        )

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": result["response"]})
        session_context = result.get("session_context", {})

        print(f"\n[intent: {result['intent']}]", end="")
        if result["escalate"]:
            print(f" [ESCALATED: {result['escalate_reason']}]")
        elif result.get("needs_clarification"):
            print(" [asking for clarification]")
        else:
            print()
        print(f"Rafiq: {result['response']}\n")


if __name__ == "__main__":
    main()
