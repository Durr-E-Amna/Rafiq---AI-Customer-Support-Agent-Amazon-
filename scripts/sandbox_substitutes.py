"""
SANDBOX-ONLY. Not part of the real project.

Substitutes for GroqClient and KnowledgeRetriever so the LangGraph wiring
can be exercised end-to-end without a live Groq API key (blocked from this
dev sandbox's network) or the real ONNX embedding download (same reason).

FakeLLM uses simple keyword rules instead of an actual model to decide
intent - it exists only to prove the graph routes correctly, not to
demonstrate real classification quality. The real GroqClient (used in
main.py) is what actually ships.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.ingest import KB_DIR, chunk_markdown


class FakeLLM:
    """Keyword-rule stand-in for GroqClient, sandbox testing only."""

    @staticmethod
    def _is_chitchat_or_identity(text: str) -> bool:
        greetings = ["hi", "hello", "hey", "thanks", "thank you", "bye", "good morning", "good evening"]
        identity_phrases = ["your name", "who are you", "what can you do", "are you a bot", "are you an ai", "are you human"]
        has_letters = any(c.isalpha() for c in text)
        if not has_letters:
            return True  # emoji-only or symbols-only, no real request content
        if any(p in text for p in identity_phrases):
            return True
        # Greeting check: only if it's a short message that's essentially
        # just the greeting, not a longer sentence that happens to contain it
        if len(text.split()) <= 4 and any(g in text for g in greetings):
            return True
        return False

    @staticmethod
    def _is_shopping(text: str) -> bool:
        shopping_signals = [
            "buy", "shop for", "gift", "recommend", "looking for", "want to buy",
            "show me", "deal", "cheap", "best", "top rated", "under ", "suggest",
            "shoes", "headphones", "speaker", "present",
        ]
        return any(s in text for s in shopping_signals)

    @staticmethod
    def _is_off_topic(text: str) -> bool:
        off_topic_signals = [
            "capital of", "weather", "wether", "president of", "who won",
            "write me a", "write a poem", "what is the meaning of",
            "how do i cook", "math problem", "translate", "what year did",
        ]
        return any(s in text for s in off_topic_signals)

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        # The real prompt now wraps history + latest message together;
        # pull out just the latest message for this simple keyword matcher.
        if "Latest message:" in user_prompt:
            latest = user_prompt.split("Latest message:", 1)[1].strip()
        else:
            latest = user_prompt
        text = latest.lower()
        order_match = re.search(r"ord[\s\-]*(?:dash[\s\-]*)?(\d{3,5})", text)
        order_id = f"ORD-{order_match.group(1)}" if order_match else None

        frustration_signals = [
            "again", "still", "third time", "ridiculous", "unacceptable",
            "worst", "never", "scam", "fraud", "angry", "fed up", "done with",
        ]
        frustration = any(sig in text for sig in frustration_signals)

        if "fraud" in text or "lawyer" in text or "legal" in text:
            intent = "escalate_directly"
        elif self._is_off_topic(text):
            intent = "off_topic"
        elif self._is_chitchat_or_identity(text):
            intent = "chitchat_or_identity"
        elif "cancel" in text:
            intent = "cancel_order"
        elif "refund" in text or "return" in text:
            intent = "refund_request"
        elif "track" in text or "where" in text or "status" in text or order_id:
            intent = "order_status"
        elif any(w in text for w in ["policy", "how long", "how do i", "what is", "can i"]):
            intent = "knowledge_question"
        elif self._is_shopping(text):
            intent = "shopping_query"
        else:
            intent = "other"

        refund_reason = None
        if intent == "refund_request":
            if "damaged" in text or "broken" in text:
                refund_reason = "damaged"
            elif "wrong" in text:
                refund_reason = "wrong_item"
            elif "missing" in text:
                refund_reason = "missing_parts"
            else:
                refund_reason = "changed_mind"

        shopping = None
        if intent == "shopping_query":
            import re as _re
            max_price = None
            m = _re.search(r"under\s*\$?(\d+)", text)
            if m:
                max_price = float(m.group(1))
            sort_by = None
            if "best" in text or "top rated" in text or "highest rat" in text:
                sort_by = "rating"
            elif "cheap" in text or "budget" in text:
                sort_by = "price_low"
            shopping = {
                "query": latest,
                "max_price": max_price,
                "min_price": None,
                "min_rating": None,
                "store": None,
                "sort_by": sort_by,
            }

        return {
            "intent": intent,
            "order_id": order_id,
            "refund_reason": refund_reason,
            "shopping": shopping,
            "frustration": frustration,
        }

    def generate_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        # Identify which node's prompt fired by a distinguishing phrase
        # unique to each one (they all share the same persona prefix, so
        # position-based fingerprinting wouldn't distinguish them) - lets
        # tests confirm a node actually called the LLM with the right
        # context, rather than secretly returning a hardcoded string.
        markers = {
            "small talk, a greeting": "meta/chitchat",
            "NO connection to shopping": "off_topic",
            "ONLY products you may": "product_answer",
            "no products matched": "product_no_match",
            "couldn't identify a valid order number": "clarify_order_id",
            "couldn't tell what the customer is asking": "clarify_unclear",
            "Explain the following order/refund result": "tool_answer",
            "ONLY the context below": "knowledge_answer",
        }
        matched = next((label for phrase, label in markers.items() if phrase in system_prompt), "unknown")
        return f"[generated via {matched} prompt]"


class FakeRetriever:
    """TF-IDF stand-in for KnowledgeRetriever, sandbox testing only."""

    def __init__(self, confidence_threshold: float = 0.15):
        self.confidence_threshold = confidence_threshold
        all_chunks = []
        for md_file in sorted(KB_DIR.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            all_chunks.extend(chunk_markdown(text, source=md_file.stem))
        self.chunks = all_chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in all_chunks])

    def retrieve(self, query: str, top_k: int = 3) -> dict:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]

        return {
            "chunks": [self.chunks[i]["text"] for i in top_idx],
            "sources": [self.chunks[i]["title"] for i in top_idx],
            "confidence": float(sims[top_idx[0]]) if len(top_idx) else 0.0,
            "low_confidence": sims[top_idx[0]] < self.confidence_threshold if len(top_idx) else True,
        }

class FakeProductSearcher:
    """TF-IDF stand-in for ProductSearcher, sandbox testing only. Same
    filter/sort logic as the real one, just TF-IDF instead of embeddings."""

    def __init__(self):
        import json
        from scripts.ingest_products import CATALOG_FILE, product_to_text
        self.products = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(
            [product_to_text(p) for p in self.products]
        )

    def search(self, query, max_price=None, min_price=None, min_rating=None,
               store=None, sort_by=None, top_k=5, candidate_pool=25):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        order = sims.argsort()[::-1][:candidate_pool]
        candidates = []
        for i in order:
            p = dict(self.products[i])
            p["relevance"] = float(sims[i])
            p["url"] = f"https://www.amazon.com/dp/{p.get('id', '')}" if p.get("id") else None
            candidates.append(p)

        def keep(p):
            if p["relevance"] < 0.05:  # TF-IDF scale is much lower than real cosine embeddings, threshold scaled down accordingly for this stand-in only
                return False
            if max_price is not None and p["price"] > max_price:
                return False
            if min_price is not None and p["price"] < min_price:
                return False
            if min_rating is not None and p["average_rating"] < min_rating:
                return False
            if store is not None and store.lower() not in p["store"].lower():
                return False
            return True

        filtered = [p for p in candidates if keep(p)]
        if sort_by == "rating":
            filtered.sort(key=lambda p: (p["average_rating"], p["rating_number"]), reverse=True)
        elif sort_by == "price_low":
            filtered.sort(key=lambda p: p["price"])
        elif sort_by == "price_high":
            filtered.sort(key=lambda p: p["price"], reverse=True)
        return filtered[:top_k]
