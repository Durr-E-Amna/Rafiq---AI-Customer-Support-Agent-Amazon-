"""
Shared core logic for the routing eval, used by both the production runner
(eval/run_routing_eval.py, real Groq + Chroma) and the sandbox dev runner
(scripts/test_routing_eval_sandbox_only.py, fake substitutes) - so both
exercise identical scoring logic and only differ in which llm_client/
retriever get injected.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import build_graph
from eval.routing_test_set import ROUTING_EVAL_SET


def run_routing_eval(llm_client, retriever, verbose: bool = True) -> dict:
    graph = build_graph(llm_client, retriever)
    results = []

    for case in ROUTING_EVAL_SET:
        outcome = graph.invoke(
            {
                "user_message": case["message"],
                "intent": "",
                "order_id": None,
                "refund_reason": None,
                "frustration": False,
                "retrieval": None,
                "tool_result": None,
                "escalate": False,
                "escalate_reason": None,
                "response": None,
            }
        )
        results.append(
            {
                "message": case["message"],
                "expected_intent": case["expected_intent"],
                "actual_intent": outcome["intent"],
                "intent_correct": outcome["intent"] == case["expected_intent"],
                "expected_escalate": case["expected_escalate"],
                "actual_escalate": outcome["escalate"],
                "escalate_correct": outcome["escalate"] == case["expected_escalate"],
            }
        )

    intent_accuracy = sum(r["intent_correct"] for r in results) / len(results)

    true_positives = sum(1 for r in results if r["expected_escalate"] and r["actual_escalate"])
    false_positives = sum(1 for r in results if not r["expected_escalate"] and r["actual_escalate"])
    false_negatives = sum(1 for r in results if r["expected_escalate"] and not r["actual_escalate"])

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else None
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None

    if verbose:
        for r in results:
            flags = []
            if not r["intent_correct"]:
                flags.append(f"INTENT WRONG (expected {r['expected_intent']})")
            if not r["escalate_correct"]:
                flags.append(f"ESCALATION WRONG (expected escalate={r['expected_escalate']})")
            status = " | ".join(flags) if flags else "OK"
            print(f"  [{status}] \"{r['message'][:55]}\" -> intent={r['actual_intent']}, escalate={r['actual_escalate']}")

        print(f"\nIntent classification accuracy: {intent_accuracy:.0%} ({sum(r['intent_correct'] for r in results)}/{len(results)})")
        print(f"Escalation recall (caught the cases that NEEDED a human): {recall:.0%}" if recall is not None else "Escalation recall: n/a")
        print(f"Escalation precision (didn't over-escalate fine cases):   {precision:.0%}" if precision is not None else "Escalation precision: n/a")
        if false_negatives:
            print(f"\n⚠ {false_negatives} case(s) that SHOULD have escalated but didn't — this is the costly failure mode, worth checking first.")

    return {
        "results": results,
        "intent_accuracy": intent_accuracy,
        "escalation_precision": precision,
        "escalation_recall": recall,
        "escalation_f1": f1,
    }
