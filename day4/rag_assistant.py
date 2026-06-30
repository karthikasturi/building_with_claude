"""
Lab 6 (Part 2) + Lab 7 — RAG Assistant & Evaluation
Module 6 Part 2: Grounded Answers + Citations
Module 7: Evaluation and Output Quality

Run: python rag_assistant.py
Requires: pip install openai numpy
Prerequisite: run day3/rag_pipeline.py first to generate vector_index.json
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    if not os.environ.get(key):
        raise EnvironmentError(f"{key} is not set.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

INDEX_FILE = Path(__file__).parent.parent / "day3" / "vector_index.json"

RAG_SYSTEM = """
You are a credit-policy assistant for Apex Bank.
Answer questions using ONLY the context sections provided below.
If the answer is not in the provided context, say:
  "This topic is not covered in the available policy documents."

Always cite your source at the end of each answer using this format:
  [Source: <section name>, chunk <index>]

Be concise — keep answers under 150 words unless the question requires detail.
"""

TEST_QUERIES = [
    "What is the minimum credit score required for a home loan?",
    "How long does the loan approval process take?",
    "What documents are required for a business loan application?",
    "What is the maximum debt-to-income ratio Apex Bank allows?",
    "Can a self-employed person apply for a vehicle loan?",
]


# ── VectorStore (copy from day3/rag_pipeline.py) ──────────────────────────────

from dataclasses import dataclass


@dataclass
class Chunk:
    text:        str
    doc_id:      str
    section:     str
    chunk_index: int


@dataclass
class IndexedChunk:
    chunk:     Chunk
    embedding: list[float]


class VectorStore:
    def __init__(self):
        self.items: list[IndexedChunk] = []

    def add(self, chunk: Chunk, embedding: list[float]) -> None:
        self.items.append(IndexedChunk(chunk=chunk, embedding=embedding))

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Chunk]:
        import numpy as np
        q = np.array(query_embedding)
        scores = []
        for item in self.items:
            d = np.array(item.embedding)
            score = float(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-9))
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.items[i].chunk for i in ranked[:top_k]]

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        store = cls()
        records = json.loads(path.read_text())
        for r in records:
            chunk = Chunk(
                text=r["text"], doc_id=r["doc_id"],
                section=r["section"], chunk_index=r["chunk_index"],
            )
            store.items.append(IndexedChunk(chunk=chunk, embedding=r["embedding"]))
        print(f"Loaded {len(store.items)} chunks from {path}")
        return store


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(query: str, store: VectorStore, top_k: int = 3) -> list[Chunk]:
    """Embed the query with OpenAI and return the top-k chunks."""
    import openai
    oc = openai.OpenAI()
    query_embedding = oc.embeddings.create(
        input=[query], model="text-embedding-3-small"
    ).data[0].embedding
    return store.search(query_embedding, top_k=top_k)


# ── Grounded answer ────────────────────────────────────────────────────────────

def ask_grounded(query: str, store: VectorStore) -> dict:
    """Retrieve relevant chunks and ask Claude to answer using only that context."""
    chunks = retrieve(query, store, top_k=3)
    context = "\n\n".join(
        f"--- [{c.section}, chunk {c.chunk_index}] ---\n{c.text}"
        for c in chunks
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0,
        system=RAG_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        }],
    )

    answer = next((b.text for b in response.content if b.type == "text"), "")
    return {
        "query":       query,
        "answer":      answer,
        "chunks_used": [f"{c.section}, chunk {c.chunk_index}" for c in chunks],
        "context":     context,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

EVAL_SYSTEM = """
You are an evaluation judge for an AI assistant.
Given a question, the retrieved context, and the assistant's answer, score the answer on:

1. FAITHFULNESS (0–3): Is every claim in the answer supported by the provided context?
   0 = answer contains facts not in context (hallucination)
   1 = mostly grounded, minor unsupported claims
   2 = fully grounded with minor omissions
   3 = fully grounded and complete

2. RELEVANCE (0–3): Does the answer actually address the question asked?
   0 = off-topic  1 = partially relevant  2 = mostly relevant  3 = fully relevant

Respond ONLY with valid JSON:
{"faithfulness": <0-3>, "relevance": <0-3>, "notes": "<one sentence>"}
"""


def evaluate_answer(query: str, context: str, answer: str) -> dict:
    """Ask Claude to judge the answer for faithfulness and relevance."""
    prompt = (
        f"Question: {query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Assistant's answer:\n{answer}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=128,
        temperature=0,
        system=EVAL_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "{}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"faithfulness": 0, "relevance": 0, "notes": f"Parse error: {text[:80]}"}


def main():
    store = VectorStore.load(INDEX_FILE)

    print("\n=== Apex Bank RAG Assistant — Evaluation Run ===\n")
    results = []

    for query in TEST_QUERIES:
        print(f"Q: {query}")
        result = ask_grounded(query, store)
        print(f"A: {result['answer']}")

        scores = evaluate_answer(query, result["context"], result["answer"])
        print(f"   Faithfulness: {scores.get('faithfulness', '?')}/3  "
              f"Relevance: {scores.get('relevance', '?')}/3  — {scores.get('notes', '')}")
        results.append({**result, **scores})
        print()

    # Summary
    faith_scores = [r.get("faithfulness", 0) for r in results]
    rel_scores   = [r.get("relevance", 0)   for r in results]
    low_faith    = sum(1 for s in faith_scores if s < 2)

    print("=== Evaluation Summary ===")
    print(f"Avg faithfulness : {sum(faith_scores)/len(faith_scores):.2f}/3")
    print(f"Avg relevance    : {sum(rel_scores)/len(rel_scores):.2f}/3")
    print(f"Low faithfulness : {low_faith} / {len(results)} queries (score < 2)")


if __name__ == "__main__":
    main()
