"""
Grounded order-ID resolution.

Voice-to-text mangles short alphanumeric codes badly and *unpredictably* -
"ORD-1001" can come back as "oid 10001", "o r d dash 1001", "OD 10001",
and there's no single regex or prompt instruction that reliably normalizes
every variant, because the mangling isn't a consistent pattern - it's
noise. Asking an LLM to parse it perfectly every time is asking the wrong
tool to do the job: an LLM doesn't know which order IDs actually exist, so
it has nothing to validate its guess against.

The right fix is the one real voice interfaces use for this exact problem:
resolve noisy input against a known reference set ("account lookup" by
fuzzy-matching a spoken account number against real accounts, not just
trusting the raw transcription). Our order database is closed and known
(it's a fixed mock dataset), so we can ground every extraction attempt
against the orders that actually exist.

This is a genuinely different problem from the session-state question
(agent/graph.py) - that was about *remembering dialogue state across
turns*. This is about *resolving noisy input against ground truth within
a single turn*. Conflating the two was part of what went wrong before.
"""

import difflib
import re
from pathlib import Path

from agent.tools import _load_orders  # noqa: reused private loader, same module family

DIGIT_PATTERN = re.compile(r"\d{3,6}")
MATCH_THRESHOLD = 0.8


def known_order_ids() -> list[str]:
    return [o["order_id"] for o in _load_orders()]


def resolve_order_id(text: str) -> str | None:
    """Extract digit sequences from text and fuzzy-match each one against
    known order IDs. Returns the best match if it clears the threshold,
    None otherwise (genuinely no good match - don't guess past that point,
    let the agent ask the customer to repeat it instead)."""
    digit_candidates = DIGIT_PATTERN.findall(text)
    if not digit_candidates:
        return None

    known = known_order_ids()
    known_digits = {oid: oid.split("-")[1] for oid in known}

    best_match, best_score = None, 0.0
    for candidate in digit_candidates:
        for oid, oid_digits in known_digits.items():
            score = difflib.SequenceMatcher(None, candidate, oid_digits).ratio()
            if score > best_score:
                best_score, best_match = score, oid

    return best_match if best_score >= MATCH_THRESHOLD else None
