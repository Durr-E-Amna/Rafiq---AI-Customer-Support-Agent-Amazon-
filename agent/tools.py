"""
Mock order-management tools.

These follow the exact pattern a real integration (e.g. a live order API)
would use: same function signatures, same kind of structured return values.
The only difference is the data source - a local JSON file instead of a
live database - which is the standard way these integrations get built
and tested before they're ever pointed at production data.

Policy logic here (refund eligibility, refund method by payment type,
escalation triggers) mirrors what's documented in
data/knowledge_base/refund_policy.md and damaged_wrong_item.md, so the
tool's decisions stay consistent with what the agent tells customers.
"""

import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "data" / "mock_db" / "orders.json"

RETURN_WINDOW_DAYS = 30
ELECTRONICS_RETURN_WINDOW_DAYS = 15
ELECTRONICS_CATEGORY_ITEMS = {"laptop", "tablet", "smartphone", "phone", "smartwatch", "wearable"}

REFUND_METHOD_BY_PAYMENT = {
    "card": {"destination": "original card", "eta": "3-5 business days"},
    "debit_card": {"destination": "original debit card", "eta": "up to 10 business days"},
    "gift_card_balance": {"destination": "gift card balance", "eta": "within 2-3 hours of approval"},
    "stored_balance": {"destination": "stored payment balance", "eta": "within a few hours of approval"},
}

# Reasons that should never be auto-approved by the agent - these need a
# human to verify (photo evidence, fraud risk), per damaged_wrong_item.md
ESCALATION_REQUIRED_REASONS = {"damaged", "wrong_item", "missing_parts"}


def _load_orders() -> list[dict]:
    return json.loads(DB_PATH.read_text(encoding="utf-8"))


def check_order_status(order_id: str) -> dict:
    """Look up the current status of an order."""
    orders = _load_orders()
    order = next((o for o in orders if o["order_id"].lower() == order_id.lower()), None)

    if order is None:
        return {"found": False, "order_id": order_id}

    result = {
        "found": True,
        "order_id": order["order_id"],
        "item": order["item"],
        "status": order["status"],
        "payment_method": order["payment_method"],
        "amount": order["amount"],
        "placed_days_ago": order["placed_days_ago"],
    }

    # Flag a tracking issue if "in transit" hasn't updated in 3+ days,
    # per the threshold documented in order_tracking.md
    if order["status"] == "in_transit" and order["last_status_update_days_ago"] >= 3:
        result["tracking_stalled"] = True

    if order["status"] == "delivery_failed":
        result["failed_delivery_attempts"] = order["failed_delivery_attempts"]

    return result


def request_refund(order_id: str, reason: str) -> dict:
    """
    Evaluate a refund/return request against policy.

    reason should be one of: "changed_mind", "damaged", "wrong_item",
    "missing_parts", "other"
    """
    orders = _load_orders()
    order = next((o for o in orders if o["order_id"].lower() == order_id.lower()), None)

    if order is None:
        return {"found": False, "order_id": order_id}

    if reason in ESCALATION_REQUIRED_REASONS:
        return {
            "found": True,
            "order_id": order["order_id"],
            "escalate_required": True,
            "escalation_reason": (
                f"Refund reason '{reason}' requires human verification "
                "(photo evidence / fraud check) before approval."
            ),
        }

    if order["status"] != "delivered":
        return {
            "found": True,
            "order_id": order["order_id"],
            "approved": False,
            "reason_denied": "Order has not been delivered yet — a return can only be "
            "requested after delivery, or the order can simply be cancelled if it "
            "hasn't shipped.",
        }

    applicable_window = RETURN_WINDOW_DAYS
    item_lower = order["item"].lower()
    if any(category in item_lower for category in ELECTRONICS_CATEGORY_ITEMS):
        applicable_window = ELECTRONICS_RETURN_WINDOW_DAYS

    if order["delivered_days_ago"] > applicable_window:
        return {
            "found": True,
            "order_id": order["order_id"],
            "approved": False,
            "reason_denied": (
                f"Delivered {order['delivered_days_ago']} days ago, which is past "
                f"the {applicable_window}-day return window for this item."
            ),
        }

    refund_info = REFUND_METHOD_BY_PAYMENT[order["payment_method"]]
    return {
        "found": True,
        "order_id": order["order_id"],
        "approved": True,
        "amount": order["amount"],
        "refund_destination": refund_info["destination"],
        "refund_eta": refund_info["eta"],
    }


def cancel_order(order_id: str) -> dict:
    """Cancel an order if it hasn't shipped yet."""
    orders = _load_orders()
    order = next((o for o in orders if o["order_id"].lower() == order_id.lower()), None)

    if order is None:
        return {"found": False, "order_id": order_id}

    if order["status"] != "processing":
        return {
            "found": True,
            "order_id": order["order_id"],
            "cancelled": False,
            "reason_denied": (
                f"Order is already '{order['status']}' — it can no longer be "
                "cancelled directly since it has shipped. It can be returned "
                "after delivery instead."
            ),
        }

    return {
        "found": True,
        "order_id": order["order_id"],
        "cancelled": True,
        "refund_triggered": order["payment_method"] != "cod",
    }
