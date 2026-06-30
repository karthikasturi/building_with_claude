# Lab 6 (Part 1) — RAG Indexing Pipeline
**Module 6: Retrieval-Grounded Responses — Chunking, Embeddings, Vector Store**

## Objective
Build the offline indexing pipeline that converts Apex Bank's SOP and credit policy documents into a searchable vector index.

## Starter file
`day3/rag_pipeline.py`

## What you will build
- `chunk_text()` — fixed-size word chunking with overlap
- `VectorStore` class — in-memory store with cosine similarity search, JSON save/load
- `build_index()` — load docs → chunk → embed with OpenAI → store

## Setup
```bash
pip install openai numpy
# Set OPENAI_API_KEY in .env (get a key at platform.openai.com)
```

## Key concepts
- Fixed-size chunking with overlap preserves context at boundaries
- OpenAI `text-embedding-3-small` model — fast, cost-effective, 1536-dimensional embeddings
- Cosine similarity: `dot(q, d) / (|q| × |d|)`
- Metadata on each chunk (doc_id, section, index) enables citations later

## Documents indexed
| File | Section label |
|------|--------------|
| `shared/data/finance_sop/loan_processing_sop.md` | Loan Processing SOP |
| `shared/data/apex_bank_credit_policy.md` | Credit Policy |

## Success criteria
| Check | Pass condition |
|-------|---------------|
| Chunking | Chunks are ~500 words with 50-word overlap |
| Embeddings | All chunks embedded via `text-embedding-3-small` |
| VectorStore | `search()` returns top-K by cosine similarity (descending) |
| Persistence | Index saves to `vector_index.json` and reloads correctly |
| Smoke test | Top-3 results for a test query are from the correct documents |

## Output
`day3/vector_index.json` — used by `day4/rag_assistant.py`
