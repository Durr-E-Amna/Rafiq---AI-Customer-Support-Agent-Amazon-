"""SANDBOX-ONLY. Verifies two things about the Telegram handler without
needing a live bot token or network access:
1. One chat's context (active order) persists correctly across messages.
2. Two different chats never bleed into each other's session state."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import build_graph
from scripts.sandbox_substitutes import FakeLLM, FakeRetriever
from telegram_bot.handler import handle_message
from telegram_bot.session_store import _sessions

graph = build_graph(FakeLLM(), FakeRetriever())

CHAT_A = 111
CHAT_B = 222

# Chat A: establish an order, then refer back to it without restating the ID
r1 = handle_message(graph, CHAT_A, "Where is my order ord dash 1001")
print(f"Chat A, turn 1: order_id={r1['order_id']}")
assert r1["order_id"] == "ORD-1001"

r2 = handle_message(graph, CHAT_A, "what if I want to cancel it can I")
print(f"Chat A, turn 2: order_id={r2['order_id']}  needs_clarification={r2.get('needs_clarification')}")
assert r2["order_id"] == "ORD-1001", "Chat A should remember its own order"
assert not r2["needs_clarification"]

# Chat B: a completely different conversation, started AFTER chat A already
# has an active order - must not see chat A's order_id at all
r3 = handle_message(graph, CHAT_B, "what if I want to cancel it can I")
print(f"Chat B, turn 1: order_id={r3['order_id']}  needs_clarification={r3.get('needs_clarification')}")
assert r3["order_id"] is None, f"Chat B should NOT see chat A's order, got {r3['order_id']}"
assert r3["needs_clarification"], "Chat B has no context, should ask for the order ID"

print(f"\nActive sessions tracked: {list(_sessions.keys())}")
print("PASSED: sessions persist correctly within a chat and stay isolated across chats.")
