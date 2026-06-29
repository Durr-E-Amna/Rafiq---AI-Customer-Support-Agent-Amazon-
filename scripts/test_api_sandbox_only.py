"""SANDBOX-ONLY. Verifies the /chat endpoint's routing behavior end-to-end
using FastAPI's in-process TestClient (no real network needed at all,
unlike the live server) and the same fake substitutes used elsewhere.
Not part of the real project."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from scripts.sandbox_substitutes import FakeLLM, FakeRetriever
from web.server import create_app

app = create_app(FakeLLM(), FakeRetriever())
client = TestClient(app)

TEST_MESSAGES = [
    "How long do I have to return something after delivery?",
    "Where is my order ORD-1003? It's been stuck for days.",
    "I want a refund for ORD-1002, it arrived damaged.",
    "Can I cancel ORD-1004?",
]

if __name__ == "__main__":
    print("Health check:", client.get("/health").json())
    print("Homepage serves:", client.get("/").status_code == 200)
    print()

    for msg in TEST_MESSAGES:
        res = client.post("/chat", json={"message": msg})
        data = res.json()
        print(f"MSG: {msg}")
        print(f"  status={res.status_code} intent={data['intent']} escalate={data['escalate']}")
        print(f"  response: {data['response'][:80]}")
        print()
