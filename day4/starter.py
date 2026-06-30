"""
Apex Bank — Loan Origination Assistant
Day 4 start: Day 1+2+3 complete. Today: RAG over Loan Processing SOP.

New today:
  - Section-aware SOP chunker
  - Keyword-based retrieval (cosine similarity with Voyage AI in production)
  - SOP context injected into every assessment with prompt caching
"""

import os
import re
import json
import math
import anthropic
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY is not set. Check your .env file.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """
You are a loan intake assistant for Apex Bank.

Your job: collect applicant information, call the available tools to verify
credit and documents, then produce a preliminary loan assessment.

ELIGIBILITY RULES (always cite section when applying):
- [Section 1.1] Minimum credit score: 680 (salaried/government), 700 (self-employed)
- [Section 2.1] Maximum DTI: 40% salaried, 35% self-employed, 45% government
- [Section 2.2] Loans above INR 5,000,000 require credit committee review
- [Section 3.1] Valid PAN and Aadhaar required for all applicants
- [Section 4.1] If bureau unavailable, refer to committee — never decline

DTI formula: existing_emi_total / gross_monthly_income (use only existing EMI from lookup_existing_account)

DECISION PRECEDENCE (in this order):
1. Bureau unavailable → refer_to_committee (Section 4.1; never decline on this alone)
2. Loan > INR 5,000,000 → refer_to_committee (Section 2.2)
3. Credit score below minimum → decline (Section 1.1)
4. DTI exceeds maximum → decline (Section 2.1)
5. Missing mandatory document → documents_verified: false → decline (Section 3.1)
6. All checks pass → proceed

IMPORTANT: Batch system — never ask for clarification. Apply the rules and return JSON immediately.

OUTPUT: After calling all required tools, return ONLY this JSON object — no prose, no markdown fences:
{
  "applicant_type": "salaried" | "self_employed" | "government",
  "annual_income_inr": <number>,
  "loan_type": "home" | "personal" | "business" | "vehicle",
  "loan_amount_requested_inr": <number>,
  "existing_emi_inr": <number from lookup_existing_account or 0>,
  "dti_ratio": <decimal, e.g. 0.42 for 42%>,
  "credit_score": <integer or null if unavailable>,
  "documents_verified": <true|false>,
  "preliminary_decision": "proceed" | "refer_to_committee" | "decline",
  "policy_basis": "<cited sections and reasoning>"
}
"""

SOP_PATH = Path(__file__).parent.parent / "shared/data/finance_sop/loan_processing_sop.md"

# ── Day 2: Schema (complete) ─────────────────────────────────────────────────

class LoanAssessment(BaseModel):
    applicant_type: Literal["salaried", "self_employed", "government"]
    annual_income_inr: float
    loan_type: Literal["home", "personal", "business", "vehicle"]
    loan_amount_requested_inr: float
    existing_emi_inr: float
    dti_ratio: Optional[float] = None
    credit_score: Optional[int] = None
    documents_verified: bool
    preliminary_decision: Literal["proceed", "refer_to_committee", "decline"]
    policy_basis: str

# ── Day 3: Tools + agentic loop (complete) ───────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "check_credit_score",
        "description": (
            "Look up the credit score for an applicant using their customer ID. "
            "Returns score and risk band, or an error if the bureau is unavailable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Internal customer ID, e.g. CUST-001"}
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "lookup_existing_account",
        "description": (
            "Fetch existing loan and EMI details for a customer. "
            "Returns active loans, total existing EMI, and days-past-due."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Internal customer ID, e.g. CUST-001"}
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "validate_documents",
        "description": (
            "Validate a submitted document by type and ID. "
            "Returns validity status, expiry date, and reason."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_type": {"type": "string"},
                "doc_id":   {"type": "string"},
            },
            "required": ["doc_type", "doc_id"],
            "additionalProperties": False,
        },
    },
]

_CREDIT_DB = {
    "CUST-001": {"score": 762, "risk_band": "low"},
    "CUST-002": {"score": 541, "risk_band": "high"},
    "CUST-003": {"score": 680, "risk_band": "medium"},
    "CUST-004": {"error": "Credit bureau temporarily unavailable. Retry later."},
}
_ACCOUNT_DB = {
    "CUST-001": {"active_loans": 1, "existing_emi_total": 8500,  "dpd": 0},
    "CUST-002": {"active_loans": 3, "existing_emi_total": 32000, "dpd": 15},
    "CUST-003": {"active_loans": 0, "existing_emi_total": 0,     "dpd": 0},
    "CUST-004": {"active_loans": 1, "existing_emi_total": 5000,  "dpd": 0},
}
_DOC_DB = {
    "PAN:PAN-VALID":         {"valid": True,  "expiry_date": "N/A",        "reason": "Valid PAN card"},
    "Aadhaar:AADH-VALID":    {"valid": True,  "expiry_date": "N/A",        "reason": "Valid Aadhaar"},
    "Passport:PASS-EXPIRED": {"valid": False, "expiry_date": "2021-03-01", "reason": "Passport expired"},
}


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    if tool_name == "check_credit_score":
        cid = tool_input.get("customer_id", "")
        return _CREDIT_DB.get(cid, {"error": f"Customer {cid} not found"})
    if tool_name == "lookup_existing_account":
        cid = tool_input.get("customer_id", "")
        return _ACCOUNT_DB.get(cid, {"error": f"Customer {cid} not found"})
    if tool_name == "validate_documents":
        key = f"{tool_input.get('doc_type', '')}:{tool_input.get('doc_id', '')}"
        return _DOC_DB.get(key, {"valid": True, "expiry_date": "N/A", "reason": "Document accepted"})
    return {"error": f"Unknown tool: {tool_name}"}


def run_agentic_assessment(application_text: str, system_override: str | None = None) -> LoanAssessment:
    """Full agentic loop with optional system_override (used by RAG to inject SOP context)."""
    system = system_override or SYSTEM_PROMPT
    messages = [{"role": "user", "content": application_text}]
    print("  Starting agentic loop...")

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    print(f"  [tool] {block.name}({block.input}) → {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            print(f"  WARNING: unexpected stop_reason={response.stop_reason}")
            break

    raw = next((b.text for b in response.content if b.type == "text"), "{}")
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw) or re.search(r"(\{[\s\S]*\})", raw)
    return LoanAssessment.model_validate(json.loads(m.group(1) if m else raw))


# ── Day 4: RAG over SOP ───────────────────────────────────────────────────────

class SopChunk:
    def __init__(self, section: str, text: str):
        self.section = section
        self.text = text
        self._tokens = set(text.lower().split())

    def score(self, query_tokens: set[str]) -> float:
        """TF-style overlap score — replace with cosine(Voyage embeddings) in production."""
        if not self._tokens:
            return 0.0
        overlap = len(query_tokens & self._tokens)
        return overlap / math.sqrt(len(self._tokens))


class SopVectorStore:
    """Loads the SOP, splits by Section headers, and retrieves top-k chunks by keyword overlap."""

    def __init__(self, sop_path: Path):
        self.chunks: list[SopChunk] = []
        self._load(sop_path)

    def _load(self, path: Path) -> None:
        text = path.read_text()
        current_section = "Introduction"
        current_lines: list[str] = []

        for line in text.splitlines():
            if line.startswith("## Section") or line.startswith("### Section"):
                if current_lines:
                    self.chunks.append(SopChunk(current_section, "\n".join(current_lines)))
                current_section = line.lstrip("#").strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            self.chunks.append(SopChunk(current_section, "\n".join(current_lines)))

    def retrieve(self, query: str, top_k: int = 3) -> list[SopChunk]:
        query_tokens = set(query.lower().split())
        scored = [(c.score(query_tokens), c) for c in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k] if _ > 0]


sop_store = SopVectorStore(SOP_PATH)


def build_rag_system_prompt(query: str) -> str:
    """Retrieve relevant SOP sections and prepend them to the system prompt.

    The SOP block is marked cache_control=ephemeral so repeated calls
    with the same SOP content cost ~90% less for cached tokens.
    """
    chunks = sop_store.retrieve(query)
    sop_context = "\n\n".join(
        f"[SOP: {c.section}]\n{c.text}" for c in chunks
    )
    return [
        {
            "type": "text",
            "text": (
                "LOAN PROCESSING SOP (authoritative — cite sections in your decision):\n\n"
                + sop_context
            ),
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": SYSTEM_PROMPT},
    ]


def run_rag_assessment(application_text: str) -> LoanAssessment:
    """RAG-enhanced agentic assessment: retrieves SOP context before the loop."""
    system = build_rag_system_prompt(application_text)
    return run_agentic_assessment(application_text, system_override=system)


def main():
    print("=== RAG-Enhanced Loan Assessment ===\n")

    cases = [
        (
            "F-TC-01",
            "Home loan. Salaried, annual income INR 900,000. "
            "Loan: INR 4,500,000. Customer CUST-001. Docs: PAN type PAN-VALID, Aadhaar AADH-VALID.",
        ),
        (
            "F-TC-04",
            "Home loan. Government employee, annual income INR 800,000. "
            "Loan: INR 3,000,000. Customer CUST-004. Docs: PAN type PAN-VALID.",
        ),
    ]

    for case_id, text in cases:
        print(f"{'='*60}")
        print(f"Case {case_id}")
        print("-" * 60)
        try:
            result = run_rag_assessment(text)
            print(f"Decision : {result.preliminary_decision}")
            print(f"Credit   : {result.credit_score}")
            print(f"DTI      : {result.dti_ratio:.1%}" if result.dti_ratio is not None else "DTI      : N/A")
            print(f"Basis    : {result.policy_basis}")
        except NotImplementedError as e:
            print(f"  {e}")
        print()


if __name__ == "__main__":
    main()
