"""
RAGAS evaluation harness for the RAG (knowledge-question) path.

Uses the real production retriever (agent/retriever.py) and the real
production answer-generation function
(agent.graph.generate_knowledge_answer_text) — this scores the actual
pipeline, not a parallel reimplementation of it.

Embeddings used for scoring are the SAME free local ONNX MiniLM model the
retriever itself uses, wrapped to satisfy RAGAS's expected interface. This
keeps the whole project free of any OpenAI dependency, even for eval.

Run after:
  - GROQ_API_KEY is set in your environment
  - scripts/ingest.py has been run at least once

Usage:
    python eval/run_ragas_eval.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from chromadb.utils import embedding_functions
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, Faithfulness, ResponseRelevancy

from agent.graph import generate_knowledge_answer_text
from agent.llm_client import GroqClient, MODEL
from agent.retriever import KnowledgeRetriever
from eval.test_set import RAG_EVAL_SET


class ONNXEmbeddings(Embeddings):
    """Wraps Chroma's built-in ONNX MiniLM function to satisfy RAGAS's
    expected Embeddings interface, so eval uses the same free, local model
    as the retriever — no OpenAI key needed anywhere in this project."""

    def __init__(self):
        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return self._fn(texts)

    def embed_query(self, text):
        return self._fn([text])[0]


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Set it before running:")
        print("  export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    llm = GroqClient()
    retriever = KnowledgeRetriever()

    print(f"Running {len(RAG_EVAL_SET)} eval cases through the real retriever + Groq...\n")

    samples = []
    for case in RAG_EVAL_SET:
        retrieval = retriever.retrieve(case["question"])
        response = generate_knowledge_answer_text(llm, retrieval, case["question"])
        samples.append(
            SingleTurnSample(
                user_input=case["question"],
                response=response,
                retrieved_contexts=retrieval["chunks"],
                reference=case["ground_truth"],
                reference_contexts=case["reference_contexts"],
            )
        )
        print(f"  done: {case['question'][:60]}...")

    dataset = EvaluationDataset(samples=samples)

    judge_llm = LangchainLLMWrapper(
        ChatGroq(model=MODEL, temperature=0, api_key=os.environ["GROQ_API_KEY"])
    )
    judge_embeddings = LangchainEmbeddingsWrapper(ONNXEmbeddings())

    print("\nScoring with RAGAS (Faithfulness, ContextPrecision, ResponseRelevancy)...")
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ContextPrecision(), ResponseRelevancy()],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = result.to_pandas()
    score_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference", "reference_contexts")]

    pd.set_option("display.max_colwidth", 50)
    print("\n" + "=" * 70)
    print(df[["user_input"] + score_cols].to_string(index=False))
    print("=" * 70)

    out_path = Path(__file__).parent / "eval_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    print("\nAverage scores across all cases:")
    print(df[score_cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
