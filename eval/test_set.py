"""
Eval set for the knowledge-question / RAG path.

Each case has:
- question: what a real customer might type
- ground_truth: a reference answer, written by us directly from the KB
  (used by ContextPrecision/context-grounded metrics)
- reference_contexts: the exact KB passage(s) that should be retrieved to
  answer this correctly (used to score whether retrieval found the right
  thing, independent of whether the LLM's wording matches)

Covers all 6 knowledge base documents at least once, plus harder cases
(ambiguous phrasing, a question the KB only partially answers) to avoid
only testing the easy cases.

IMPORTANT: these ground truths must stay in sync with the actual KB
content in data/knowledge_base/. When the KB was rewritten to reflect real
Amazon policy instead of the earlier Daraz-modeled version, these were
rewritten to match - a RAGAS eval scored against stale ground truths would
silently report false "unfaithful" results even when the agent answered
correctly from the current KB.
"""

RAG_EVAL_SET = [
    {
        "question": "How many days do I have to return an item after it's delivered?",
        "ground_truth": "Most items can be returned within 30 days of the delivery date for a refund or replacement, as long as they're in original or unused condition. Some high-value electronics have a shorter 15-day window instead.",
        "reference_contexts": [
            "Most items can be returned within 30 days of the delivery date for a refund or a replacement, as long as the item is in original or unused condition."
        ],
    },
    {
        "question": "How long does it take to get my refund back after a return?",
        "ground_truth": "Refund timing depends on the destination: gift card balance refunds land within 2-3 hours of approval, credit card refunds take 3-5 business days, and debit card refunds can take up to 10 business days. The refund always returns to whichever method was originally charged.",
        "reference_contexts": [
            "Amazon gift card balance | Within 2-3 hours of approval. Credit card | 3-5 business days (bank processing time, varies by issuer). Debit card | Up to 10 business days."
        ],
    },
    {
        "question": "My order status hasn't changed in 4 days and it still says in transit, is something wrong?",
        "ground_truth": "If there's been no movement for 3 or more business days while marked in transit, that's worth escalating for a manual check with the carrier.",
        "reference_contexts": [
            "If there's been no movement for 3+ business days while marked \"in transit,\" that's worth escalating for a manual check with the carrier rather than continuing to reassure the customer it's probably fine."
        ],
    },
    {
        "question": "Can I cancel my order after it has already been shipped?",
        "ground_truth": "No — once an order has entered the shipping process, it can no longer be cancelled directly. The customer can either let delivery complete and start a standard return, or refuse delivery at the door.",
        "reference_contexts": [
            "Once an order has entered the shipping process, it can no longer be cancelled directly. At this point the options are: Let the delivery complete, then start a standard return if unwanted. Refuse the delivery at the door."
        ],
    },
    {
        "question": "Do you accept cash on delivery?",
        "ground_truth": "No — this platform does not support cash on delivery. All orders require a payment method on file at checkout, such as a credit/debit card or gift card balance.",
        "reference_contexts": [
            "This platform does not support cash on delivery — all orders require a payment method on file at checkout."
        ],
    },
    {
        "question": "I received the wrong color of the product I ordered, what happens now?",
        "ground_truth": "Wrong-item claims are treated as a fulfillment error, not a standard return. The customer doesn't need the item in original unused condition, photo evidence is typically requested, and they can choose a refund or replacement.",
        "reference_contexts": [
            "The item received doesn't match what was ordered (wrong color, wrong model, wrong product entirely)",
            "Customers can choose either a refund or a replacement",
        ],
    },
    {
        "question": "A third-party seller is refusing to refund me, what can I do?",
        "ground_truth": "This is covered by the A-to-Z Guarantee, not standard return policy. Amazon can step in, refund the customer directly, and recover the cost from the seller separately. This should be escalated as a formal claim.",
        "reference_contexts": [
            "When the issue is with a third-party seller rather than a directly-fulfilled order, and the seller hasn't resolved it, the customer is protected by the A-to-Z Guarantee"
        ],
    },
    {
        "question": "Can I return a Final Sale item if it arrives broken?",
        "ground_truth": "Yes — even non-returnable or Final Sale items become eligible for resolution if they arrive damaged, defective, or materially different from the listing. That exception always applies.",
        "reference_contexts": [
            "Even non-returnable or Final Sale items become eligible for resolution if they arrive damaged, defective, or materially different from the listing — this exception always applies."
        ],
    },
]
