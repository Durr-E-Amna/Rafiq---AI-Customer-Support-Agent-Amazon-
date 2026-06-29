"""
Core per-message handling logic, deliberately separated from
python-telegram-bot's API. This is the same dependency-injection pattern
as build_graph() and create_app(): the graph is built once by the caller
and passed in, so this function can be tested with fake substitutes
(scripts/test_telegram_handler_sandbox_only.py) without a live Telegram
bot token or network access, and the real bot.py just calls it from
inside a python-telegram-bot handler.
"""

from telegram_bot.session_store import get_session, update_session


def handle_message(graph, chat_id: int, text: str) -> dict:
    """Runs one message through the agent graph for a specific chat,
    using and updating that chat's own isolated session state.
    Returns the full graph result dict (caller decides what to send back)."""
    session = get_session(chat_id)

    result = graph.invoke(
        {
            "user_message": text,
            "history": session["history"],
            "session_context": session["session_context"],
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

    new_history = session["history"] + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": result["response"]},
    ]
    # Keep history bounded - same window size used elsewhere in the project
    update_session(chat_id, new_history[-8:], result.get("session_context", {}))

    return result
