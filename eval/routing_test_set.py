"""
Labeled test set for the agentic routing decisions RAGAS doesn't cover -
intent classification accuracy and escalation precision/recall.

These are the decisions that determine whether the agent answers, calls a
tool, or hands off to a human - getting this wrong is arguably more costly
than a slightly imperfect RAG answer (a wrong escalation decision either
annoys a fine customer with unnecessary handoff, or worse, lets the agent
handle something it shouldn't have).

expected_escalate is the ground truth for whether THIS message, in
isolation, should end in escalation - independent of whether intent
classification alone gets it there (e.g. a damaged-item refund escalates
via the tool layer, not via intent classification flagging escalate_directly).
"""

ROUTING_EVAL_SET = [
    {"message": "How long do I have to return something after delivery?",
     "expected_intent": "knowledge_question", "expected_escalate": False},

    {"message": "Where is my order ORD-1003? It's been stuck for days.",
     "expected_intent": "order_status", "expected_escalate": True},  # tracking_stalled

    {"message": "Where is my order ORD-1001?",
     "expected_intent": "order_status", "expected_escalate": False},

    {"message": "I want a refund for ORD-1005, I just changed my mind.",
     "expected_intent": "refund_request", "expected_escalate": False},

    {"message": "I want a refund for ORD-1002, it arrived damaged.",
     "expected_intent": "refund_request", "expected_escalate": True},

    {"message": "Can I cancel ORD-1004?",
     "expected_intent": "cancel_order", "expected_escalate": False},

    {"message": "Can I cancel ORD-1001?",
     "expected_intent": "cancel_order", "expected_escalate": False},

    {"message": "This is the third time I'm messaging about ORD-1006, still nothing, this is ridiculous.",
     "expected_intent": "order_status", "expected_escalate": True},

    {"message": "I think I'm being scammed, I want to talk to a lawyer about this.",
     "expected_intent": "escalate_directly", "expected_escalate": True},

    {"message": "What payment methods do you accept?",
     "expected_intent": "knowledge_question", "expected_escalate": False},

    {"message": "Do you deliver internationally and how long does that take?",
     "expected_intent": "knowledge_question", "expected_escalate": False},

    {"message": "I've contacted support twice already about my missing refund and I'm done waiting.",
     "expected_intent": "other", "expected_escalate": True},
]
