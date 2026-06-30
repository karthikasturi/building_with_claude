# Lab 6 (Part 2) + Lab 7 — RAG Assistant & Evaluation
**Module 6 Part 2: Grounded Answers + Citations**
**Module 7: Evaluation and Output Quality**

## Objective
Complete the RAG pipeline by adding answer synthesis with citations (Module 6), then build a Claude-as-judge evaluation harness to measure faithfulness and relevance (Module 7).

## Starter file
`day4/rag_assistant.py`

## Prerequisite
Run `day3/rag_pipeline.py` first — this lab loads `day3/vector_index.json`.

## What you will build

### Module 6 Part 2 — Grounded answers
- `retrieve()` — embed query with OpenAI, search the vector index
- `ask_grounded()` — inject retrieved chunks into Claude's context, enforce citation format

### Module 7 — Evaluation harness
- `evaluate_answer()` — Claude-as-judge scores each answer on faithfulness (0–3) and relevance (0–3)
- Aggregate summary across all 5 test queries

## Key concepts
- RAG prompt pattern: inject `[Source: section, chunk N]` blocks before the question
- Constrained system prompt: "Answer ONLY from the provided context"
- Claude-as-judge: a second Claude call that scores the first call's output
- Faithfulness ≠ relevance — an answer can be on-topic but hallucinate, or grounded but off-topic

## Success criteria
| Check | Pass condition |
|-------|---------------|
| Retrieval | Top-3 chunks retrieved per query using `text-embedding-3-small` |
| Grounded answer | System prompt enforces context-only answers |
| Citations | Every answer ends with `[Source: ...]` |
| Out-of-scope | Returns the "not covered" message for unknown topics |
| Faithfulness scoring | 0–3 score returned for every answer |
| Relevance scoring | 0–3 score returned for every answer |
| Aggregate summary | Mean scores and hallucination count printed at end |
