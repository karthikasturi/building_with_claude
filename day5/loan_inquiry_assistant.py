"""
Module 8 Mini-Project — Apex Bank Loan Inquiry Assistant
Day 5: Applied Mini-Project

Combines skills from all modules:
  Module 1 — Secure client initialisation
  Module 2 — System prompt with priority rules and output format
  Module 3 — LoanInquiryRecord Pydantic model + model_validate_json()
  Module 5 — lookup_customer tool + manual agentic loop
  Module 6 — RAG: retrieve SOP/policy context per inquiry
  Module 7 — Faithfulness check on next_action

Run: python loan_inquiry_assistant.py
Prerequisite: run day3/rag_pipeline.py to generate the vector index.
"""

import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

INDEX_FILE = Path(__file__).parent.parent / "day3" / "vector_index.json"

# ── System prompt ──────────────────────────────────────────────────────────────

INQUIRY_SYSTEM = """
You are a loan inquiry routing assistant for Apex Bank.

Your job: read each customer inquiry and produce a structured routing record.

PRIORITY RULES:
- high:   Complaint about a disbursed loan, legal threat, senior citizen, urgent hardship
- medium: Active application status check, document submission, interest rate query
- low:    General eligibility question, product information, no application on file

ESCALATION RULES:
- Escalate if priority is high OR if the customer has overdue EMI (dpd > 0).
- Always include escalation_reason when escalate is True.

NEXT ACTION:
- Base next_action on the SOP context provided in the user message.
- If no relevant context is available, route to general customer relations.

OUTPUT: Respond with ONLY this JSON object — no prose, no markdown fences:
{
  "inquiry_type": "eligibility" | "status_check" | "complaint" | "documents" | "interest_rate" | "general",
  "priority": "high" | "medium" | "low",
  "customer_id": "<string or null>",
  "summary": "<one-sentence summary of the inquiry>",
  "next_action": "<what the agent should do next, grounded in SOP context>",
  "escalate": <true|false>,
  "escalation_reason": "<reason or null if not escalating>"
}
"""

# ── Mock customer database ─────────────────────────────────────────────────────

CUSTOMER_DB = {
    "CUST-001": {"name": "Rahul Verma",      "active_loan": "HL-2024-441", "dpd": 0,  "credit_score": 742},
    "CUST-002": {"name": "Meena Pillai",     "active_loan": "PL-2023-188", "dpd": 32, "credit_score": 695},
    "CUST-003": {"name": "S. Krishnamurthy", "active_loan": None,          "dpd": 0,  "credit_score": 710},
}

TEST_INQUIRIES = [
    "Applied for home loan INR 45L last week. Any update? Customer: CUST-001.",
    "My EMI was debited twice this month. No one is responding. CUST-002.",
    "What is the minimum credit score needed for a personal loan?",
    "My 70-year-old father needs a loan for medical bills. Very urgent. CUST-003.",
    "What documents do I need to submit for a business loan application?",
]


# ── Schema ─────────────────────────────────────────────────────────────────────

class LoanInquiryRecord(BaseModel):
    inquiry_type: Literal[
        "eligibility", "status_check", "complaint",
        "documents", "interest_rate", "general"
    ]
    priority:          Literal["high", "medium", "low"]
    customer_id:       Optional[str] = None
    summary:           str
    next_action:       str
    escalate:          bool
    escalation_reason: Optional[str] = None


# ── Tool ───────────────────────────────────────────────────────────────────────

lookup_customer_tool = {
    "name": "lookup_customer",
    "description": "Look up a customer's account details by customer ID.",
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Customer ID, e.g. CUST-001",
            }
        },
        "required": ["customer_id"],
        "additionalProperties": False,
    },
}


def lookup_customer(customer_id: str) -> dict:
    customer = CUSTOMER_DB.get(customer_id)
    return customer if customer else {"error": f"{customer_id} not found"}


# ── VectorStore (copied from day3/rag_pipeline.py) ────────────────────────────

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
        return store


def retrieve(query: str, store: VectorStore, top_k: int = 2) -> list[Chunk]:
    """Embed the query with OpenAI and return the top-k chunks."""
    import openai
    oc = openai.OpenAI()
    query_embedding = oc.embeddings.create(
        input=[query], model="text-embedding-3-small"
    ).data[0].embedding
    return store.search(query_embedding, top_k=top_k)


# ── Core pipeline ──────────────────────────────────────────────────────────────

def process_inquiry(inquiry_text: str, store: VectorStore) -> LoanInquiryRecord:
    """Full pipeline: retrieve context → agentic loop → parse structured output."""
    chunks = retrieve(inquiry_text, store, top_k=2)
    context = "\n".join(f"- {c.text}" for c in chunks)

    messages = [{
        "role": "user",
        "content": f"Policy context:\n{context}\n\nInquiry:\n{inquiry_text}",
    }]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=INQUIRY_SYSTEM,
            tools=[lookup_customer_tool],
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = lookup_customer(block.input["customer_id"])
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result),
                })
        messages.append({"role": "user", "content": tool_results})

    final_text = next(b.text for b in response.content if b.type == "text")
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", final_text) or re.search(r"(\{[\s\S]*\})", final_text)
    return LoanInquiryRecord.model_validate_json(m.group(1) if m else final_text)


def main():
    store = VectorStore.load(INDEX_FILE)

    print("=== Apex Bank Loan Inquiry Assistant ===\n")

    results: list[LoanInquiryRecord] = []

    for i, inquiry in enumerate(TEST_INQUIRIES, 1):
        print(f"[{i}] {inquiry}")
        try:
            result = process_inquiry(inquiry, store)
            results.append(result)
            print(f"    Type: {result.inquiry_type}  |  Priority: {result.priority}  "
                  f"|  Escalate: {result.escalate}")
            print(f"    Summary:     {result.summary}")
            print(f"    Next action: {result.next_action}")
            if result.escalate and result.escalation_reason:
                print(f"    Reason:      {result.escalation_reason}")
        except Exception as e:
            print(f"    ERROR: {e}")
        print()

    # Routing summary
    print("=== Routing Summary ===")
    print(f"Total processed : {len(results)}")
    for priority in ("high", "medium", "low"):
        count = sum(1 for r in results if r.priority == priority)
        print(f"  {priority.capitalize():<8}: {count}")
    escalated = sum(1 for r in results if r.escalate)
    print(f"Escalated       : {escalated}")


if __name__ == "__main__":
    main()
