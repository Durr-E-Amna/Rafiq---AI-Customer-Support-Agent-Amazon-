"""
FastAPI app factory for the voice (and later, Telegram) layer.

Same dependency-injection pattern as agent/graph.py's build_graph(): this
file builds the *shape* of the app, but the actual llm_client/retriever are
passed in by whoever constructs it. Production (web/run.py) passes in real
GroqClient/KnowledgeRetriever; sandbox tests pass in the fake substitutes,
so the routing logic can be verified without any live API calls.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import build_graph

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    session_context: dict = {}


class ChatResponse(BaseModel):
    intent: str
    escalate: bool
    escalate_reason: str | None
    needs_clarification: bool
    response: str
    session_context: dict
    products: list[dict] = []


def create_app(llm_client, retriever, product_searcher=None) -> FastAPI:
    app = FastAPI(title="Rafiq Support Agent")
    graph = build_graph(llm_client, retriever, product_searcher)

    # Wide open CORS since this is a local dev/demo project served from the
    # same origin anyway - tightening this matters once this goes beyond a
    # portfolio demo, not before.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        result = graph.invoke(
            {
                "user_message": req.message,
                "history": req.history,
                "session_context": req.session_context,
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
        return ChatResponse(
            intent=result["intent"],
            escalate=result["escalate"],
            escalate_reason=result["escalate_reason"],
            needs_clarification=result.get("needs_clarification", False),
            response=result["response"],
            session_context=result.get("session_context", {}),
            products=result.get("products") or [],
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def index():
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app
