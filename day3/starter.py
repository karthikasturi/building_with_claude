"""
Apex Bank — Loan Origination Assistant
Day 3 start: Day 1+2 complete. Today: Tool Use + Agentic Loop.

New today:
  - Tool definitions (check_credit_score, lookup_existing_account, validate_documents)
  - Mock tool executor
  - Manual while-loop agentic assessment
"""

import os
import re
import json
import anthropic
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

# ── Day 2: Schema + parse (kept complete) ────────────────────────────────────

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


def parse_loan_assessment(application_text: str) -> LoanAssessment:
    """Single-turn structured parse — used when no tools are needed."""
    messages = [{"role": "user", "content": application_text}]

    for attempt in range(2):
        try:
            response = client.messages.parse(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
                output_format=LoanAssessment,
            )
            return response.parsed_output
        except ValidationError as e:
            if attempt == 1:
                raise
            last_text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            messages.append({"role": "assistant", "content": last_text})
            messages.append({
                "role": "user",
                "content": f"Your output had validation errors. Fix and return valid JSON:\n{e}",
            })


class LoanConversationManager:
    TOKEN_WARN  = 30_000
    TOKEN_RESET = 60_000

    def __init__(self):
        self.history: list[dict] = []

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        count = client.messages.count_tokens(
            model=MODEL,
            system=SYSTEM_PROMPT,
            messages=self.history,
        )
        if count.input_tokens > self.TOKEN_RESET:
            self._summarise_and_reset()
        elif count.input_tokens > self.TOKEN_WARN:
            print(f"  [WARNING] conversation at {count.input_tokens} tokens")

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=self.history,
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        self.history.append({"role": "assistant", "content": text})
        return text

    def _summarise_and_reset(self) -> None:
        summary_response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    "Summarise this loan intake conversation in ≤200 words, "
                    "keeping all applicant details and decisions:\n\n"
                    + json.dumps(self.history)
                ),
            }],
        )
        summary = next(
            (b.text for b in summary_response.content if b.type == "text"), ""
        )
        self.history = [{"role": "user", "content": f"[Session summary]\n{summary}"}]
        print("  [INFO] Conversation summarised and reset.")


# ── Day 3: Tools ──────────────────────────────────────────────────────────────

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
                "customer_id": {
                    "type": "string",
                    "description": "Internal customer ID, e.g. CUST-001",
                }
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
                "customer_id": {
                    "type": "string",
                    "description": "Internal customer ID, e.g. CUST-001",
                }
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
                "doc_type": {"type": "string", "description": "e.g. PAN, Aadhaar, Passport"},
                "doc_id":   {"type": "string", "description": "Document identifier / number"},
            },
            "required": ["doc_type", "doc_id"],
            "additionalProperties": False,
        },
    },
]

# Mock database — same data as config/finance.yaml
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
    "PAN:PAN-VALID":    {"valid": True,  "expiry_date": "N/A",        "reason": "Valid PAN card"},
    "Aadhaar:AADH-VALID": {"valid": True, "expiry_date": "N/A",       "reason": "Valid Aadhaar"},
    "Passport:PASS-EXPIRED": {"valid": False, "expiry_date": "2021-03-01", "reason": "Passport expired"},
}


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """Dispatch a tool call to the mock database."""
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


# ── Day 3: Agentic Loop ───────────────────────────────────────────────────────

def run_agentic_assessment(application_text: str) -> LoanAssessment:
    """Run the full agentic loop: tools until end_turn, then parse final JSON."""
    messages = [{"role": "user", "content": application_text}]
    print(f"  Starting agentic loop...")

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
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
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     json.dumps(result),
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})
        else:
            print(f"  WARNING: unexpected stop_reason={response.stop_reason}")
            break

    raw = next((b.text for b in response.content if b.type == "text"), "{}")
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw) or re.search(r"(\{[\s\S]*\})", raw)
    return LoanAssessment.model_validate(json.loads(m.group(1) if m else raw))


def main():
    print("=== Agentic Loan Assessment ===\n")

    cases = [
        (
            "F-TC-01",
            "Home loan application. Salaried applicant, annual income INR 900,000. "
            "Loan requested: INR 4,500,000. Customer ID: CUST-001. "
            "Documents: PAN type PAN-VALID, Aadhaar AADH-VALID.",
        ),
        (
            "F-TC-02",
            "Personal loan. Self-employed, annual income INR 600,000. "
            "Loan: INR 500,000. Customer ID: CUST-002. Documents: PAN type PAN-VALID.",
        ),
        (
            "F-TC-03",
            "Business loan. Salaried, annual income INR 1,200,000. "
            "Loan: INR 8,000,000. Customer ID: CUST-003. Documents: PAN type PAN-VALID.",
        ),
        (
            "F-TC-04",
            "Home loan. Government employee, annual income INR 800,000. "
            "Loan: INR 3,000,000. Customer ID: CUST-004. Documents: PAN type PAN-VALID.",
        ),
    ]

    for case_id, text in cases:
        print(f"{'='*60}")
        print(f"Case {case_id}")
        print("-" * 60)
        try:
            result = run_agentic_assessment(text)
            print(f"Decision : {result.preliminary_decision}")
            print(f"Credit   : {result.credit_score}")
            print(f"DTI      : {result.dti_ratio:.1%}" if result.dti_ratio is not None else "DTI      : N/A")
            print(f"Basis    : {result.policy_basis}")
        except NotImplementedError as e:
            print(f"  {e}")
        print()


if __name__ == "__main__":
    main()
