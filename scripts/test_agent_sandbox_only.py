"""
SANDBOX-ONLY. Exercises the real agent graph (agent/graph.py) against the
real mock order DB (agent/tools.py), using FakeLLM + FakeRetriever instead
of Groq/Chroma since neither is reachable from this dev sandbox's network.

If this graph wiring is correct here, it's correct on your machine too -
the only thing that changes when you run main.py for real is which
llm_client/retriever get passed into build_graph().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import build_graph
from scripts.sandbox_substitutes import FakeLLM, FakeRetriever

TEST_MESSAGES = [
    "How long do I have to return something after delivery?",
    "Where is my order ORD-1003? It's been stuck for days.",
    "I want a refund for ORD-1005, I just changed my mind.",
    "I want a refund for ORD-1002, it arrived damaged.",
    "Can I cancel ORD-1004?",
    "Can I cancel ORD-1001?",
    "This is the third time I'm messaging about ORD-1006, still nothing, this is ridiculous.",
    "I think I'm being scammed, I want to talk to a lawyer about this.",
    "What payment methods do you accept?",
    "Where is my order? I don't have the number handy.",
    "asdkj where my thing is at",
]


def main():
    llm = FakeLLM()
    retriever = FakeRetriever()
    graph = build_graph(llm, retriever)

    for msg in TEST_MESSAGES:
        result = graph.invoke(
            {
                "user_message": msg,
                "history": [],
                "intent": "",
                "order_id": None,
                "refund_reason": None,
                "frustration": False,
                "retrieval": None,
                "tool_result": None,
                "escalate": False,
                "escalate_reason": None,
                "needs_clarification": False,
                "response": None,
            }
        )

        print(f"MSG: {msg}")
        print(f"  intent={result['intent']}  order_id={result['order_id']}  frustration={result['frustration']}")
        if result["tool_result"]:
            print(f"  tool_result={result['tool_result']}")
        if result["retrieval"]:
            print(f"  top_source={result['retrieval']['sources'][0]}  confidence={result['retrieval']['confidence']:.3f}")
        if result["escalate"]:
            print(f"  -> ESCALATED: {result['escalate_reason']}")
        elif result.get("needs_clarification"):
            print(f"  -> ASKING FOR CLARIFICATION (not escalated)")
        else:
            print(f"  -> resolved automatically")
        print()


if __name__ == "__main__":
    main()
