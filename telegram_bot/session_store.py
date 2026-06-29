"""
Per-chat session storage for the Telegram bot.

Unlike main.py (one CLI user) or web/run.py (one browser tab per run),
Telegram serves many independent chats from one running process - so
session state has to be keyed per chat_id, or one customer's conversation
would bleed into another's. This is an in-memory dict, which is fine for a
free-tier single-process demo; swapping this for Redis or a database
later (if this ever needed to survive a restart or run across multiple
processes) wouldn't require touching anything outside this file.
"""

_sessions: dict[int, dict] = {}


def get_session(chat_id: int) -> dict:
    if chat_id not in _sessions:
        _sessions[chat_id] = {"history": [], "session_context": {}}
    return _sessions[chat_id]


def update_session(chat_id: int, history: list[dict], session_context: dict) -> None:
    _sessions[chat_id] = {"history": history, "session_context": session_context}


def reset_session(chat_id: int) -> None:
    _sessions.pop(chat_id, None)
