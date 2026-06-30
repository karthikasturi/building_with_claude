"""
Lab 6 (Part 1) — RAG Indexing Pipeline
Module 6: Retrieval-Grounded Responses

Part 1 (Day 3): Chunking → Embeddings → Vector Store
Part 2 (Day 4): Retrieval → Grounded answers → Citations (see day4/rag_assistant.py)

Run: python rag_pipeline.py
Requires: pip install openai numpy
Data: ../shared/data/finance_sop/loan_processing_sop.md
      ../shared/data/apex_bank_credit_policy.md
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set. Get a key at platform.openai.com.")

DATA_DIR = Path(__file__).parent.parent / "shared" / "data"
DOCS = [
    {"id": "sop",    "path": DATA_DIR / "finance_sop" / "loan_processing_sop.md",
     "section": "Loan Processing SOP"},
    {"id": "policy", "path": DATA_DIR / "apex_bank_credit_policy.md",
     "section": "Credit Policy"},
]
INDEX_FILE = Path(__file__).parent / "vector_index.json"


# ── Data structures ────────────────────────────────────────────────────────────

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


# ── VectorStore ────────────────────────────────────────────────────────────────

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

    def save(self, path: Path) -> None:
        records = [
            {
                "text":        item.chunk.text,
                "doc_id":      item.chunk.doc_id,
                "section":     item.chunk.section,
                "chunk_index": item.chunk.chunk_index,
                "embedding":   item.embedding,
            }
            for item in self.items
        ]
        path.write_text(json.dumps(records))
        print(f"Saved {len(records)} chunks to {path}")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        store = cls()
        records = json.loads(path.read_text())
        for r in records:
            chunk = Chunk(
                text=r["text"],
                doc_id=r["doc_id"],
                section=r["section"],
                chunk_index=r["chunk_index"],
            )
            store.items.append(IndexedChunk(chunk=chunk, embedding=r["embedding"]))
        print(f"Loaded {len(store.items)} chunks from {path}")
        return store


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Fixed-size word chunking with overlap to preserve context at boundaries."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def chunk_document(doc_text: str, doc_id: str, section: str) -> list[Chunk]:
    """Wrap chunk_text() output into Chunk instances with metadata."""
    raw_chunks = chunk_text(doc_text)
    return [
        Chunk(text=text, doc_id=doc_id, section=section, chunk_index=i)
        for i, text in enumerate(raw_chunks)
    ]


# ── Indexing ───────────────────────────────────────────────────────────────────

def build_index(documents: list[dict]) -> VectorStore:
    """Load each document, chunk it, embed all chunks via OpenAI, build the index."""
    import openai
    oc = openai.OpenAI()

    store = VectorStore()
    all_chunks: list[Chunk] = []

    for doc in documents:
        text = doc["path"].read_text()
        chunks = chunk_document(text, doc["id"], doc["section"])
        all_chunks.extend(chunks)
        print(f"  {doc['id']}: {len(chunks)} chunks from {doc['path'].name}")

    print(f"Embedding {len(all_chunks)} chunks with text-embedding-3-small...")
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i: i + batch_size]
        result = oc.embeddings.create(
            input=[c.text for c in batch], model="text-embedding-3-small"
        )
        for chunk, item in zip(batch, result.data):
            store.add(chunk, item.embedding)

    print(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")
    return store


def main():
    print("Building RAG index from Apex Bank documents...")
    store = build_index(DOCS)
    store.save(INDEX_FILE)

    # Smoke test: reload and search
    print("\nRunning smoke test...")
    loaded = VectorStore.load(INDEX_FILE)

    import openai
    oc = openai.OpenAI()
    query = "What is the maximum loan-to-value ratio for home loans?"
    q_emb = oc.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
    results = loaded.search(q_emb, top_k=3)

    print(f"\nTop 3 results for: '{query}'")
    for rank, chunk in enumerate(results, 1):
        print(f"\n[{rank}] {chunk.section} | chunk {chunk.chunk_index}")
        print(chunk.text[:200] + "...")


if __name__ == "__main__":
    main()
