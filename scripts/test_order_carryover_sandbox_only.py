"""SANDBOX-ONLY. Verifies the session-state carryover: a follow-up message
with NO order ID in it should still resolve against the order ID
established earlier in the conversation, via explicit session_context -
not by re-parsing chat transcript text."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import build_graph
from scripts.sandbox_substitutes import FakeLLM, FakeRetriever

graph = build_graph(FakeLLM(), FakeRetriever())


def invoke(user_message, history, session_context):
    return graph.invoke(
        {
            "user_message": user_message,
            "history": history,
            "session_context": session_context,
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


# Turn 1: establish an active order via the current message
turn1 = invoke("Where is my order ord dash 1001", history=[], session_context={})
print(f"Turn 1: intent={turn1['intent']}  order_id={turn1['order_id']}  session_context={turn1['session_context']}")
assert turn1["order_id"] == "ORD-1001"
assert turn1["session_context"]["active_order_id"] == "ORD-1001"

# Turn 2: follow-up with NO order ID at all - must resolve via session_context,
# exactly as the caller (main.py / web server) would actually pass it forward.
history = [
    {"role": "user", "content": "Where is my order ord dash 1001"},
    {"role": "assistant", "content": turn1["response"]},
]
turn2 = invoke("what if I want to cancel it can I", history=history, session_context=turn1["session_context"])
print(f"Turn 2: intent={turn2['intent']}  order_id={turn2['order_id']}  escalate={turn2['escalate']}  needs_clarification={turn2.get('needs_clarification')}")
assert turn2["order_id"] == "ORD-1001", f"FAILED: expected ORD-1001 carried over via session_context, got {turn2['order_id']}"
assert not turn2["needs_clarification"], "FAILED: should not have asked for clarification - order ID was already known"

# Turn 3: a NEW order ID mentioned - should replace the active one, not stack
turn3 = invoke("actually what about ORD-1004", history=history, session_context=turn2["session_context"])
print(f"Turn 3: intent={turn3['intent']}  order_id={turn3['order_id']}  session_context={turn3['session_context']}")
assert turn3["order_id"] == "ORD-1004", f"FAILED: new order ID should replace the old one, got {turn3['order_id']}"

print("\nPASSED: session state correctly tracks, carries forward, and replaces the active order ID.")
