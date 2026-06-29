# Rafiq — AI Shopping & Support Agent

An AI customer-support and shopping agent for e-commerce, combining **agentic RAG**, **tool-calling**, **multi-channel deployment** (CLI, web with voice, Telegram), and **systematic evaluation** (RAGAS + custom routing metrics) — built entirely on free-tier infrastructure, with zero paid dependencies anywhere in the stack.

Modeled on real Amazon policies (returns, refunds, the A-to-Z Guarantee, cancellation rules) and a real Amazon product dataset for shopping queries. **Not affiliated with, endorsed by, or branded as Amazon** — an independent portfolio project built around real, publicly available policy and product data.

## What this demonstrates

- **Agentic RAG** — retrieval-grounded answers over a real policy knowledge base, not hallucinated ones
- **Tool-calling & agentic orchestration** — a LangGraph state machine that classifies intent and routes to retrieval, tool calls, product search, or escalation
- **Production-style escalation logic** — reserved for what actually needs a human (fraud, repeated frustration, damaged-item claims, stalled tracking), not used as a catch-all for things the agent doesn't understand
- **Eval literacy** — RAGAS (faithfulness, context precision, response relevancy) for the RAG path, plus a custom routing eval (intent accuracy, escalation precision/recall) for the agentic decisions RAGAS doesn't cover
- **Real product data, properly sourced** — McAuley-Lab Amazon Reviews 2023 (CC BY-SA 4.0), ingested once into a vector store, not live-scraped per query
- **Voice interface** — browser Speech-to-Text/Text-to-Speech wired to the same backend
- **Multi-channel deployment** — identical agent logic served via CLI, a web UI, and a live Telegram bot
- **Session-state management** — structured dialogue state (which order is "active" in a conversation), the same pattern production dialogue systems use, kept deliberately separate from raw chat history

## Architecture

```
User input (text / voice / Telegram)
        │
        ▼
  classify_intent  (LLM + grounded order-ID resolution + session state)
        │
        ├─ frustrated / fraud / legal → escalate
        ├─ chitchat / identity        → meta response
        ├─ off-topic                  → polite decline + redirect
        ├─ shopping query             → product search → answer
        ├─ order action, no ID known  → ask for clarification
        ├─ order action, ID known     → call tool → answer (or escalate if it needs a human)
        └─ knowledge question         → retrieve policy → answer (or escalate if unconfident)
```

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Groq (Llama 3.3) | Free tier, no card |
| Agent orchestration | LangGraph | Explicit state machine, not a black box |
| Vector DB | Chroma | Self-hosted, no account needed |
| Embeddings | ONNX MiniLM (via Chroma) | Same quality as sentence-transformers, lighter install |
| Web framework | FastAPI | Powers the `/chat` API and voice UI |
| Voice | Browser Speech APIs | Zero setup, built into Chrome |
| Messaging | Telegram Bot API | Free, no business verification |
| Eval | RAGAS + custom routing eval | Faithfulness/precision/relevancy + escalation accuracy |
| Product data | McAuley-Lab Amazon Reviews 2023 | Real titles, prices, ratings, CC BY-SA 4.0 |

## Quickstart

```bash
git clone <this-repo>
cd supportai
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

pip install chromadb==1.5.9 langgraph==1.2.6 langchain-core==1.4.8 groq==0.37.1
pip install ragas==0.4.3 langchain-groq==1.1.3 pandas==3.0.2
pip install langchain-community==0.3.27 --no-deps --force-reinstall
pip install fastapi uvicorn python-telegram-bot python-dotenv datasets huggingface_hub pyarrow

echo "GROQ_API_KEY=your_key_here" > .env   # free key: console.groq.com

python scripts/ingest.py          # builds the policy knowledge base
python main.py                    # talk to it in the terminal
```

Optional, for shopping queries (real product data, ~700MB one-time download):
```bash
python scripts/build_product_catalog.py
python scripts/ingest_products.py
```

Optional, for the voice web UI:
```bash
python web/run.py     # open http://127.0.0.1:8000 in Chrome
```

Optional, for the Telegram bot — get a free token from **@BotFather** on Telegram (`/newbot`), add it to `.env` as `TELEGRAM_BOT_TOKEN=...`, then:
```bash
python telegram_bot/bot.py
```

## Project structure

```
agent/
  graph.py            LangGraph state machine — the agent's full decision logic
  tools.py             Mock order-management tools (status, refund, cancel)
  retriever.py          Policy knowledge-base retriever (Chroma)
  product_search.py     Product search — semantic + structured filtering/sorting
  order_resolver.py      Grounds noisy order-ID input against the real order database
  llm_client.py           Groq API wrapper
data/
  knowledge_base/       6 policy articles (returns, refunds, tracking, cancellation, payments, damaged/wrong items)
  mock_db/               Sample order data
  product_catalog/       Real product data (generated by build_product_catalog.py)
eval/
  test_set.py / run_ragas_eval.py        RAGAS eval for the RAG path
  routing_test_set.py / run_routing_eval.py   Custom eval for intent + escalation accuracy
web/
  server.py / run.py / static/index.html   FastAPI backend + voice-enabled chat UI
telegram_bot/
  bot.py / handler.py / session_store.py   Telegram channel, per-chat session isolation
main.py                 Interactive CLI
scripts/                Setup scripts (ingestion, catalog building) + test suites
```

## Key engineering decisions

A few decisions worth knowing the reasoning behind (useful context, e.g. for technical interviews):

**Session state is separate from chat history, on purpose.** A customer can say "can I cancel it" two turns after mentioning an order, with no ID in that message at all. Resolving that needs two genuinely different things: `history` (raw transcript, given to the LLM purely for natural wording) and `session_context` (explicit, structured state — "which order is active right now" — managed by code, not inferred by the LLM, the same pattern production dialogue systems call "slots"). An earlier version tried to solve this by having the LLM "remember" via a prompt instruction, with a regex scanning chat transcript as a backstop — a real anti-pattern that conflates "the LLM understands language" with "the system tracks state." Replaced entirely with explicit session state.

**Order-ID extraction is grounded against the real database, not trusted blindly from the LLM.** Voice transcription mangles short codes unpredictably ("ORD-1001" → "oid 10001", "o r d dash 1001", no consistent pattern). An LLM has no way to validate its own guess against orders that actually exist. `agent/order_resolver.py` instead fuzzy-matches noisy input against the real order database — the same approach real voice interfaces use for account lookup. ("10001" scores 0.889 similarity against the real `1001`, vs. 0.667 against every other order — unambiguous.)

**A third intent category for chitchat/identity, not just policy + orders.** Early testing showed that asking "what's your name" or sending an emoji got the *literal same sentence* back regardless of what was said — because anything that wasn't a clear order action was forced through policy retrieval, found nothing relevant, and fell back to one hardcoded string. Fixed with a real third path that never touches the knowledge base, plus an `off_topic` category for things with no connection to the store at all (so "what's the capital of France" gets a polite redirect, not an unnecessary human handoff).

**Escalation is reserved for what actually needs a human.** Fraud/legal threats, repeated genuine frustration, damaged-item claims (need photo verification), stalled tracking (3+ days no movement) — not "the agent doesn't understand," which gets a clarifying question instead. A missing/garbled order ID asks the customer to repeat it rather than escalating — escalating because *we* failed to parse input is a real design flaw, not safe-by-default behavior.

**Real product data, not live scraping.** Live-scraping a retailer per query is slow, actively blocked by most sites, legally grey, and architecturally wrong for RAG anyway (RAG retrieves from an index, not a live page fetch). Instead: a bounded, one-time sample from the McAuley-Lab Amazon Reviews 2023 dataset, ingested into its own Chroma collection (kept separate from policy docs so the two don't pollute each other's retrieval), with semantic search layered under *exact* structured filters (price, rating, store) for queries like "under $30, top rated."

**A real dependency conflict, found and resolved, not avoided.** `ragas` 0.4.3 still imports a `langchain_community` integration path removed in that package's latest release, which directly conflicts with the `langgraph`/`langchain-core` versions the agent needs — installing both together makes pip's resolver refuse outright. Resolved by installing in a specific order and force-reinstalling the older `langchain_community` without re-triggering dependency resolution, then re-running the full test suite to confirm it works at runtime despite the declared version conflict. (Also hit and resolved: `datasets` 5.0.0 dropped support for script-based dataset loading entirely, breaking the official loading path for the product dataset — worked around by reading the dataset's auto-converted Parquet files directly via `huggingface_hub` + `pyarrow`.)

## Evals

Two tracks, because they measure different failure modes:

- **RAGAS** (`eval/run_ragas_eval.py`) scores the RAG path: faithfulness (does the answer stick to retrieved context?), context precision (did retrieval find the right chunks?), response relevancy (does it address the actual question?).
- **Routing eval** (`eval/run_routing_eval.py`) scores the agentic decisions RAGAS doesn't cover: intent classification accuracy, and — more importantly — escalation **recall**. A missed escalation (the agent handling something it shouldn't have) is a worse failure than an unnecessary handoff, so recall matters more than precision here.

Both are testable without live API calls via dependency-injected fakes (`scripts/sandbox_substitutes.py`) — the same `build_graph(llm, retriever, ...)` factory the production code uses accepts lightweight test doubles, so the full routing logic can be verified in CI without burning API quota or needing live credentials.

## Known limitations

- Mock order database (6 sample orders) — built as a test fixture standing in for a real order-management API, the same pattern used in development/staging before any agent touches live customer data
- Product images aren't shown (the catalog doesn't capture image URLs) — product cards are text + real Amazon links only
- Telegram session storage is in-memory (resets on restart) — fine for a demo, would move to Redis/a database for anything persistent

## License & data attribution

Code: MIT (see `LICENSE`). Product data: McAuley-Lab Amazon Reviews 2023, licensed CC BY-SA 4.0. Policy content is original writing modeled on Amazon's publicly documented policies, not copied text.

PREVIEW
<img width="1355" height="599" alt="image" src="https://github.com/user-attachments/assets/2ee4de97-c686-4695-9885-1ee1bc2e3248" />
