"""
The agent brain - a LangGraph state graph.

Flow:
  classify_intent (sees recent conversation history + persistent session state)
       |
       +-- frustrated / clearly needs a human         --> escalate
       +-- chitchat / identity / small talk            --> generate_meta_response (NOT retrieval, NOT a canned string)
       +-- order action, but no order ID known         --> clarify_order_id (LLM-generated, NOT a canned string)
       +-- order status / refund / cancel, ID known     --> call_tool --+-- needs a human --> escalate
       |                                                                 +-- otherwise     --> generate_tool_answer
       +-- knowledge question                          --> retrieve   --+-- low confidence --> escalate
       +-- unclear / vague (intent "other")             --> retrieve   --+-- low confidence --> clarify_unclear (LLM-generated)
                                                                          +-- otherwise      --> generate_knowledge_answer

ARCHITECTURE NOTE 1 - three categories of input, not two:

A real support agent's inbound messages aren't just "policy questions" and
"order actions" - a meaningful share is chitchat, greetings, and questions
about the agent itself ("what's your name", "who am I talking to", or just
an emoji with no real request). Earlier versions of this graph had no path
for that third category at all: anything that wasn't an order action got
forced through policy-knowledge retrieval, found nothing relevant (because
"what's your name" isn't in a return policy document), and fell back to a
canned clarification string. That's not a tone problem to patch with
better wording - it's a missing branch in the architecture. The fix is a
real third path: "chitchat_or_identity" intent -> generate_meta_response,
which never touches the knowledge base and is never asked to retrieve
anything, because there's nothing to retrieve.

ARCHITECTURE NOTE 2 - clarifications are generated, not canned, on purpose:

Every clarifying message (order ID not found, message unclear) is now
produced by the LLM, grounded in what the customer actually said and the
conversation so far - not a fixed string returned verbatim every time. A
support agent that says the literal same sentence twice in a row,
regardless of what you just told it, reads as obviously scripted; this
was a real, valid complaint, not a style nitpick.

The ONE place a fixed string is still used deliberately is the
human-handoff message in escalate(). That is not an oversight - consistent,
predictable escalation language is a real production requirement (it's
the one message a business actually wants controlled and not improvised
by a model), so it stays a template while everything else does not.

ARCHITECTURE NOTE 3 - session state, not text-mining:

A customer can say "can I cancel it" two turns after mentioning their
order, with no order ID in that message at all. The system tracks "which
order is this conversation currently about" as explicit, structured state
(`session_context`) managed by code - the same pattern real dialogue
systems use (Rasa calls these "slots"). It's set once an order ID is
known and persists until replaced. The caller (main.py, web/server.py,
telegram_bot/handler.py) is responsible for persisting it between turns,
the same way any session token works.

ARCHITECTURE NOTE 4 - order ID extraction is grounded against real data:

Voice transcription mangles short alphanumeric codes unpredictably ("OD
10001", "o r d dash 1001"). The LLM has no way to validate its own guess
against orders that actually exist, so agent/order_resolver.py fuzzy-
matches noisy input against the real order database instead - the same
pattern real voice interfaces use for account lookup.

llm_client and retriever are passed in rather than constructed inside this
file, so the same graph-building logic can run against either the real
Groq + Chroma stack (production) or local test substitutes (sandbox dev -
see scripts/test_agent_sandbox_only.py), without duplicating the routing
logic in two places.
"""

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from agent import tools
from agent.order_resolver import known_order_ids, resolve_order_id

ORDER_INTENTS = ("order_status", "refund_request", "cancel_order")

RAFIQ_PERSONA = (
    "You are Rafiq, an AI shopping and customer support assistant for Amazon "
    "orders. You help with order status, returns, refunds, cancellations, and "
    "finding products to buy, grounded in real store policy and real product "
    "data - you don't make up policy details or invent products. "
    "You're warm, brief, and direct, like a sharp human support agent, not "
    "stiff or corporate. You're comfortable saying you're an independent AI "
    "assistant if asked - no need to dodge the question, and you're not "
    "Amazon's official support line, just a helper built around it."
)


class AgentState(TypedDict):
    user_message: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}, ...] - for LLM context/fluency only
    session_context: dict  # persistent dialogue state, e.g. {"active_order_id": "ORD-1001"} - managed by code, not the LLM
    intent: str
    order_id: Optional[str]
    refund_reason: Optional[str]
    shopping: Optional[dict]
    products: Optional[list]
    frustration: bool
    retrieval: Optional[dict]
    tool_result: Optional[dict]
    escalate: bool
    escalate_reason: Optional[str]
    needs_clarification: bool
    response: Optional[str]


CLASSIFY_SYSTEM_PROMPT = """You are the intent router for an e-commerce support agent called Rafiq.
You will be given the recent conversation history (for context/tone only) and the customer's latest message.

Return ONLY a JSON object with these fields:

- "intent": one of "knowledge_question", "order_status", "refund_request", "cancel_order",
  "shopping_query", "chitchat_or_identity", "off_topic", "escalate_directly", "other"
- "order_id": the order ID IF AND ONLY IF it appears in the latest message (format like ORD-1234), otherwise null.
  Do not try to recall or guess an order ID from earlier turns - that is handled separately by the system.
- "refund_reason": only if intent is "refund_request" - one of "changed_mind", "damaged", "wrong_item", "missing_parts", "other"; otherwise null
- "shopping": only if intent is "shopping_query" - a JSON object describing what they want to buy, with fields:
    "query" (the core thing they're looking for, e.g. "gift for women", "running shoes", "bluetooth speaker"),
    "max_price" (number or null), "min_price" (number or null), "min_rating" (number or null),
    "store" (brand/store name if they named one, else null),
    "sort_by" (one of "rating", "price_low", "price_high", or null - use "rating" for "best"/"top rated",
    "price_low" for "cheapest"/"budget", "price_high" for "premium"/"most expensive").
  Otherwise null.
- "frustration": true ONLY if the customer sounds genuinely angry, explicitly repeats a complaint,
  mentions contacting support before, or threatens to escalate/leave a bad review.

Use "shopping_query" for product discovery: looking for things to buy, gift ideas, deals,
recommendations, "what should I get", "show me shoes", "cheapest headphones", "top rated toys",
"products from <brand>", "something for my mum", etc. Interpret loosely and forgivingly - real
customers type with typos, bad grammar, and incomplete sentences ("chepest blutooth speker",
"gift for wfe under 2000"). Extract intent from messy input; do not require perfect spelling.

Use "chitchat_or_identity" for: greetings ("hi", "hello"), small talk, questions about the
agent itself ("what's your name", "who am I talking to", "what can you do", "are you a bot"),
thanks/goodbyes, or messages with no real request content at all (e.g. just emoji, "test",
random characters). These are NOT policy questions and NOT unclear requests - they're normal
conversational turns that don't need the knowledge base at all.

Use "off_topic" for anything with NO connection to shopping, orders, returns, refunds, or this
store at all - general knowledge questions, trivia, weather, geography, "what's the capital of
X", math problems, requests to write code or essays, anything unrelated to e-commerce support.
This is different from "knowledge_question" (a real store-policy question, even if our
knowledge base happens to lack the answer) and different from "other" (a message that IS about
something support-related but is too unclear to classify). If it's clearly not about this store
or shopping at all, it's "off_topic" - and that should never need a human, since there's nothing
to escalate, just politely decline and redirect.

IMPORTANT: vague, garbled, or unclear phrasing (e.g. from voice-to-text transcription errors) is
NOT frustration and should NOT be "escalate_directly" - just do your best to interpret it, or use
intent "other" if you genuinely can't tell what they're asking. Confusion is not anger.

Real customers often write with typos, poor spelling, mixed languages, and broken grammar.
Interpret generously and forgivingly - "wher my oder", "want refnd", "shoos for weding" should
all be understood, not bounced back as unclear. Only use "other" when you truly cannot tell what
they want even after reading charitably.

Use "escalate_directly" only if the message is clearly something policy can't resolve
(e.g. a fraud accusation, a legal threat, abuse directed at the agent).
"""


def format_history(history: list[dict], max_turns: int = 4) -> str:
    if not history:
        return "(no earlier messages in this conversation)"
    recent = history[-(max_turns * 2):]
    lines = [f"{turn['role']}: {turn['content']}" for turn in recent]
    return "\n".join(lines)


def generate_knowledge_answer_text(llm_client, retrieval: dict, user_message: str, history: list[dict] | None = None) -> str:
    """Shared by the graph node below and eval/run_ragas_eval.py, so the
    eval harness scores the literal production answer-generation code
    rather than a reimplementation of it."""
    context = "\n\n".join(retrieval["chunks"])
    history_block = format_history(history or [])
    system_prompt = (
        f"{RAFIQ_PERSONA}\n\n"
        "Answer the customer's question using ONLY the context below. Use "
        "the conversation history to understand what they're actually "
        "asking about if the latest message is a vague follow-up. If the "
        "context doesn't fully answer it, say what you do know and suggest "
        "contacting support for the rest. Keep it concise.\n\n"
        f"Conversation so far:\n{history_block}\n\n"
        f"Context:\n{context}"
    )
    return llm_client.generate_text(system_prompt, user_message)


def build_graph(llm_client, retriever, product_searcher=None):
    def classify_intent(state: AgentState) -> AgentState:
        history_block = format_history(state.get("history", []))
        user_prompt = f"Conversation so far:\n{history_block}\n\nLatest message: {state['user_message']}"
        result = llm_client.generate_json(CLASSIFY_SYSTEM_PROMPT, user_prompt)

        intent = result.get("intent", "other")
        order_id = result.get("order_id")  # LLM's first attempt at extracting from THIS message

        # Don't trust that extraction blindly - the LLM has no way to
        # validate a guess against orders that actually exist, and voice
        # transcription noise is unpredictable. If what it gave us isn't a
        # real order, try grounded fuzzy resolution against the actual
        # database before giving up on this message entirely.
        if order_id not in known_order_ids():
            order_id = resolve_order_id(state["user_message"]) or None

        session_context = dict(state.get("session_context") or {})

        if order_id:
            # A new order ID was stated this turn - it becomes the active
            # one for the rest of this session, replacing whatever was
            # there before.
            session_context["active_order_id"] = order_id
        elif intent in ORDER_INTENTS:
            # No order ID in this message, but it's clearly about an order
            # action - this is the system's job, not the LLM's: look up
            # the active order from explicit session state.
            order_id = session_context.get("active_order_id")

        return {
            **state,
            "intent": intent,
            "order_id": order_id,
            "session_context": session_context,
            "refund_reason": result.get("refund_reason"),
            "shopping": result.get("shopping"),
            "frustration": bool(result.get("frustration", False)),
        }

    def route_after_classify(state: AgentState) -> str:
        if state["frustration"] or state["intent"] == "escalate_directly":
            return "escalate"
        if state["intent"] == "chitchat_or_identity":
            return "generate_meta_response"
        if state["intent"] == "off_topic":
            return "generate_off_topic_response"
        if state["intent"] == "shopping_query":
            return "product_search"
        if state["intent"] in ORDER_INTENTS:
            if not state.get("order_id"):
                return "clarify_order_id"
            return "call_tool"
        return "retrieve"

    def generate_meta_response(state: AgentState) -> AgentState:
        history_block = format_history(state.get("history", []))
        system_prompt = (
            f"{RAFIQ_PERSONA}\n\n"
            "The customer's latest message is small talk, a greeting, a "
            "question about you, or has no real support request in it "
            "(could even be just an emoji). Respond naturally and briefly - "
            "if they're asking who you are, introduce yourself in one "
            "sentence. If they're joking around or complimenting you, "
            "accept it warmly in a few words, don't riff on it at length. "
            "ALWAYS end your reply by steering back to something concrete "
            "you can actually help with - e.g. ask if they need help with "
            "an order, a return, a refund, or finding a product. Never end "
            "on an open-ended 'just chatting?' note - you're a support "
            "agent having a brief friendly moment, not a companion app.\n\n"
            f"Conversation so far:\n{history_block}"
        )
        response = llm_client.generate_text(system_prompt, state["user_message"])
        return {**state, "response": response, "escalate": False, "needs_clarification": False}

    def generate_off_topic_response(state: AgentState) -> AgentState:
        history_block = format_history(state.get("history", []))
        system_prompt = (
            f"{RAFIQ_PERSONA}\n\n"
            "The customer asked something with NO connection to shopping, "
            "orders, returns, refunds, or this store - general trivia, "
            "unrelated topics, anything outside what you do. Politely say "
            "in one short sentence that it's outside what you can help "
            "with, then immediately redirect to what you CAN do (orders, "
            "returns, refunds, finding products). Do NOT attempt to answer "
            "the off-topic question even partially, and do NOT escalate to "
            "a human - there's nothing for a human to resolve here, it's "
            "simply not something this assistant handles. Keep it to 1-2 "
            "sentences, friendly but efficient.\n\n"
            f"Conversation so far:\n{history_block}"
        )
        response = llm_client.generate_text(system_prompt, state["user_message"])
        return {**state, "response": response, "escalate": False, "needs_clarification": False}

    def product_search(state: AgentState) -> AgentState:
        shopping = state.get("shopping") or {}
        query = shopping.get("query") or state["user_message"]

        if product_searcher is None:
            # Product search isn't wired up (e.g. catalog not built yet) -
            # be honest rather than pretending. Not an escalation; just a
            # capability gap the customer can route around.
            return {
                **state,
                "products": [],
                "response": (
                    "Product search isn't available right now, but I can help "
                    "with orders, returns, refunds, and store policies — want a "
                    "hand with any of those?"
                ),
                "escalate": False,
                "needs_clarification": False,
            }

        products = product_searcher.search(
            query=query,
            max_price=shopping.get("max_price"),
            min_price=shopping.get("min_price"),
            min_rating=shopping.get("min_rating"),
            store=shopping.get("store"),
            sort_by=shopping.get("sort_by"),
            top_k=5,
        )
        return {**state, "products": products}

    def generate_product_answer(state: AgentState) -> AgentState:
        products = state.get("products") or []
        history_block = format_history(state.get("history", []))

        if not products:
            system_prompt = (
                f"{RAFIQ_PERSONA}\n\n"
                "The customer is shopping, but no products matched their "
                "request and any filters (price, rating, store). Tell them "
                "warmly that nothing matched, and suggest loosening a "
                "constraint (e.g. a higher budget or different category). "
                "Don't invent products.\n\n"
                f"Conversation so far:\n{history_block}"
            )
            response = llm_client.generate_text(system_prompt, state["user_message"])
            return {**state, "response": response, "escalate": False, "needs_clarification": False}

        # Give the model the real retrieved products to describe - it must
        # not invent any beyond this list.
        product_lines = [
            f"- {p['title']} | {p['store']} | ${p['price']:.2f} | "
            f"{p['average_rating']:.1f}/5 from {p['rating_number']} ratings"
            for p in products
        ]
        system_prompt = (
            f"{RAFIQ_PERSONA}\n\n"
            "The customer is shopping. Below are the ONLY products you may "
            "recommend - real items retrieved from our catalog. Recommend "
            "from this list in a friendly, helpful way, mentioning price and "
            "rating where useful. Do NOT invent any product, price, or rating "
            "not in this list. Keep it concise - highlight a top pick or two, "
            "don't just dump the whole list robotically.\n\n"
            f"Matching products:\n" + "\n".join(product_lines) + "\n\n"
            f"Conversation so far:\n{history_block}"
        )
        response = llm_client.generate_text(system_prompt, state["user_message"])
        return {**state, "response": response, "escalate": False, "needs_clarification": False}

    def clarify_order_id(state: AgentState) -> AgentState:
        history_block = format_history(state.get("history", []))
        system_prompt = (
            f"{RAFIQ_PERSONA}\n\n"
            "You couldn't identify a valid order number from the "
            "customer's message. Ask them, naturally and briefly, to "
            "share their order number again (it looks like ORD-1234). "
            "Look at the conversation history: if you've already asked "
            "this in the last turn or two, acknowledge that naturally "
            "instead of repeating yourself verbatim - e.g. note that the "
            "number still isn't coming through and suggest checking their "
            "confirmation email/SMS, rather than asking the exact same way again.\n\n"
            f"Conversation so far:\n{history_block}"
        )
        response = llm_client.generate_text(system_prompt, state["user_message"])
        return {**state, "needs_clarification": True, "escalate": False, "response": response}

    def retrieve_knowledge(state: AgentState) -> AgentState:
        retrieval = retriever.retrieve(state["user_message"])
        return {**state, "retrieval": retrieval}

    def route_after_retrieve(state: AgentState) -> str:
        if state["retrieval"]["low_confidence"]:
            # Genuine policy question we don't have an answer for needs a
            # human (we shouldn't guess at policy). Vague/unclear phrasing
            # ("other") just needs the customer to rephrase, not a handoff.
            if state["intent"] == "other":
                return "clarify_unclear"
            return "escalate"
        return "generate_knowledge_answer"

    def clarify_unclear(state: AgentState) -> AgentState:
        history_block = format_history(state.get("history", []))
        system_prompt = (
            f"{RAFIQ_PERSONA}\n\n"
            "You genuinely couldn't tell what the customer is asking for, "
            "and it doesn't match anything in store policy either. Ask "
            "them, naturally and briefly, to clarify - referencing what "
            "they actually said if it helps. If the conversation history "
            "shows you've already asked them to clarify recently, vary "
            "your phrasing rather than repeating the same sentence.\n\n"
            f"Conversation so far:\n{history_block}"
        )
        response = llm_client.generate_text(system_prompt, state["user_message"])
        return {**state, "needs_clarification": True, "escalate": False, "response": response}

    def call_tool(state: AgentState) -> AgentState:
        order_id = state.get("order_id")
        if state["intent"] == "order_status":
            result = tools.check_order_status(order_id)
        elif state["intent"] == "refund_request":
            result = tools.request_refund(order_id, state.get("refund_reason") or "other")
        elif state["intent"] == "cancel_order":
            result = tools.cancel_order(order_id)
        else:
            result = {"found": False}

        session_context = dict(state.get("session_context") or {})
        if not result.get("found", False):
            # This ID doesn't correspond to a real order - stop treating
            # it as the active one, so a retry doesn't silently reuse a
            # bad ID.
            session_context.pop("active_order_id", None)

        return {**state, "tool_result": result, "session_context": session_context}

    def route_after_tool(state: AgentState) -> str:
        result = state["tool_result"]
        if not result.get("found", False):
            # Order ID was present but didn't match anything real - this is
            # different from "we couldn't parse an ID at all", so it's
            # still reasonable to ask the customer to double check it
            # rather than escalate.
            return "clarify_order_id"
        if result.get("escalate_required"):
            return "escalate"
        if result.get("tracking_stalled"):
            return "escalate"
        return "generate_tool_answer"

    def generate_knowledge_answer(state: AgentState) -> AgentState:
        response = generate_knowledge_answer_text(
            llm_client, state["retrieval"], state["user_message"], state.get("history", [])
        )
        return {**state, "response": response, "escalate": False}

    def generate_tool_answer(state: AgentState) -> AgentState:
        history_block = format_history(state.get("history", []))
        system_prompt = (
            f"{RAFIQ_PERSONA}\n\n"
            "Explain the following order/refund result to the customer in "
            "plain language. Don't invent any details not present in the data.\n\n"
            f"Conversation so far:\n{history_block}\n\n"
            f"Data: {state['tool_result']}"
        )
        response = llm_client.generate_text(system_prompt, state["user_message"])
        return {**state, "response": response, "escalate": False}

    def escalate(state: AgentState) -> AgentState:
        if state.get("frustration"):
            reason = "Customer appears frustrated or has raised this issue before."
        elif state.get("intent") == "escalate_directly":
            reason = "Message requires human judgment (fraud/legal/abuse-type issue)."
        elif state.get("retrieval") and state["retrieval"].get("low_confidence"):
            reason = "No knowledge base content matched this question with enough confidence."
        elif state.get("tool_result", {}).get("escalate_required"):
            reason = state["tool_result"].get("escalation_reason", "Requires human verification.")
        elif state.get("tool_result", {}).get("tracking_stalled"):
            reason = (
                f"Order {state['tool_result'].get('order_id')} has shown no tracking "
                "movement for 3+ days while marked in transit — needs a manual check "
                "with the courier, per policy."
            )
        else:
            reason = "Could not be resolved automatically."

        # Deliberately a fixed template, not LLM-generated - unlike every
        # other response in this graph. Escalation hand-off language is
        # exactly the one message a real business wants consistent and
        # controlled, not improvised differently each time.
        response = (
            "I want to make sure this gets handled properly, so I'm connecting "
            "you with a member of our support team who can take it from here. "
            "They'll have the full context of what you've told me."
        )
        return {**state, "escalate": True, "escalate_reason": reason, "response": response}

    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("generate_meta_response", generate_meta_response)
    graph.add_node("generate_off_topic_response", generate_off_topic_response)
    graph.add_node("product_search", product_search)
    graph.add_node("generate_product_answer", generate_product_answer)
    graph.add_node("clarify_order_id", clarify_order_id)
    graph.add_node("clarify_unclear", clarify_unclear)
    graph.add_node("retrieve", retrieve_knowledge)
    graph.add_node("call_tool", call_tool)
    graph.add_node("generate_knowledge_answer", generate_knowledge_answer)
    graph.add_node("generate_tool_answer", generate_tool_answer)
    graph.add_node("escalate", escalate)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "escalate": "escalate",
            "generate_meta_response": "generate_meta_response",
            "generate_off_topic_response": "generate_off_topic_response",
            "product_search": "product_search",
            "clarify_order_id": "clarify_order_id",
            "call_tool": "call_tool",
            "retrieve": "retrieve",
        },
    )
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"escalate": "escalate", "clarify_unclear": "clarify_unclear", "generate_knowledge_answer": "generate_knowledge_answer"},
    )
    graph.add_conditional_edges(
        "call_tool",
        route_after_tool,
        {"escalate": "escalate", "clarify_order_id": "clarify_order_id", "generate_tool_answer": "generate_tool_answer"},
    )
    graph.add_edge("generate_meta_response", END)
    graph.add_edge("generate_off_topic_response", END)
    graph.add_edge("product_search", "generate_product_answer")
    graph.add_edge("generate_product_answer", END)
    graph.add_edge("clarify_order_id", END)
    graph.add_edge("clarify_unclear", END)
    graph.add_edge("generate_knowledge_answer", END)
    graph.add_edge("generate_tool_answer", END)
    graph.add_edge("escalate", END)

    return graph.compile()
