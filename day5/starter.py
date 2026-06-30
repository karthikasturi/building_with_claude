"""
Apex Bank — Loan Origination Assistant
Day 5 start: Day 1+2+3+4 complete. Today: Evaluation Harness.

New today:
  - LLM-as-judge: judge_faithfulness() and judge_relevance()
  - run_evaluation() over all 4 test cases
  - JSONL logging to eval_logs/finance_v1.0.0.jsonl
"""

import os
import re
import json
import math
import datetime
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
EVAL_LOG  = Path(__file__).parent / "eval_logs/finance_v1.0.0.jsonl"

# ── Days 2–4 complete ────────────────────────────────────────────────────────

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


TOOL_DEFINITIONS = [
    {
        "name": "check_credit_score",
        "description": (
            "Look up the credit score for an applicant using their customer ID. "
            "Returns score and risk band, or an error if the bureau is unavailable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
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
            "properties": {"customer_id": {"type": "string"}},
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


class SopChunk:
    def __init__(self, section: str, text: str):
        self.section = section
        self.text = text
        self._tokens = set(text.lower().split())

    def score(self, query_tokens: set[str]) -> float:
        if not self._tokens:
            return 0.0
        return len(query_tokens & self._tokens) / math.sqrt(len(self._tokens))


class SopVectorStore:
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
        return [c for score, c in scored[:top_k] if score > 0]


sop_store = SopVectorStore(SOP_PATH)


def build_rag_system_prompt(query: str) -> list[dict]:
    chunks = sop_store.retrieve(query)
    sop_context = "\n\n".join(
        f"[SOP: {c.section}]\n{c.text}" for c in chunks
    )
    return [
        {
            "type": "text",
            "text": f"LOAN PROCESSING SOP (authoritative — cite sections in your decision):\n\n{sop_context}",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": SYSTEM_PROMPT},
    ]


def run_agentic_assessment(application_text: str, system_override=None) -> LoanAssessment:
    system = system_override or SYSTEM_PROMPT
    messages = [{"role": "user", "content": application_text}]
    print("  Running agentic loop...")

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
            print(f"  WARNING: stop_reason={response.stop_reason}")
            break

    raw = next((b.text for b in response.content if b.type == "text"), "{}")
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw) or re.search(r"(\{[\s\S]*\})", raw)
    return LoanAssessment.model_validate(json.loads(m.group(1) if m else raw))


def run_rag_assessment(application_text: str) -> LoanAssessment:
    system = build_rag_system_prompt(application_text)
    return run_agentic_assessment(application_text, system_override=system)


# ── Day 5: Evaluation ─────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an impartial evaluator. Score responses on a 1–5 integer scale. "
    "Return JSON only: {\"score\": <int>, \"reasoning\": \"<one sentence>\"}. "
    "No prose, no markdown."
)

FAITHFULNESS_CRITERIA = """
Score how well the assessment decision is grounded in the SOP context and tool results provided.
1 = decision contradicts or ignores the context
3 = partially grounded, some unsupported claims
5 = every claim in the decision is directly traceable to the context or tool results

Application: {application}
SOP context used: {sop_context}
Tool results: {tool_results}
Assessment produced: {assessment}
"""

RELEVANCE_CRITERIA = """
Score how relevant and complete the policy_basis field is to the application.
1 = policy_basis is missing, vague, or cites wrong sections
3 = policy_basis is partially correct but incomplete
5 = policy_basis cites the exact sections, thresholds, and reasoning for the decision

Application: {application}
Assessment produced: {assessment}
"""


def _judge(prompt: str) -> dict:
    """Shared helper: call Claude as judge and parse JSON response."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "{}")
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text) or re.search(r"(\{[\s\S]*\})", text)
    try:
        return json.loads(m.group(1) if m else text)
    except json.JSONDecodeError:
        return {"score": 0, "reasoning": f"Parse error: {text[:80]}"}


def judge_faithfulness(application: str, sop_context: str, tool_results: str,
                        assessment: LoanAssessment) -> dict:
    """Score 1–5: is the assessment grounded in the provided context?"""
    prompt = FAITHFULNESS_CRITERIA.format(
        application=application,
        sop_context=sop_context,
        tool_results=tool_results,
        assessment=json.dumps(assessment.model_dump()),
    )
    return _judge(prompt)


def judge_relevance(application: str, assessment: LoanAssessment) -> dict:
    """Score 1–5: is the policy_basis relevant and correctly cited?"""
    prompt = RELEVANCE_CRITERIA.format(
        application=application,
        assessment=json.dumps(assessment.model_dump()),
    )
    return _judge(prompt)


TEST_CASES = [
    {
        "id": "F-TC-01",
        "description": "Salaried applicant — clean approval path",
        "input": (
            "Home loan. Salaried, annual income INR 900,000. "
            "Loan: INR 4,500,000. Customer CUST-001. Docs: PAN type PAN-VALID, Aadhaar AADH-VALID."
        ),
        "expected_decision": "proceed",
        "expected_section": "Section 1.1",
    },
    {
        "id": "F-TC-02",
        "description": "Self-employed — low credit score → decline",
        "input": (
            "Personal loan. Self-employed, annual income INR 600,000. "
            "Loan: INR 500,000. Customer CUST-002. Docs: PAN type PAN-VALID."
        ),
        "expected_decision": "decline",
        "expected_section": "Section 1.1",
    },
    {
        "id": "F-TC-03",
        "description": "Large business loan → committee referral",
        "input": (
            "Business loan. Salaried, annual income INR 1,200,000. "
            "Loan: INR 8,000,000. Customer CUST-003. Docs: PAN type PAN-VALID."
        ),
        "expected_decision": "refer_to_committee",
        "expected_section": "Section 2.2",
    },
    {
        "id": "F-TC-04",
        "description": "Bureau unavailable → committee referral, not decline",
        "input": (
            "Home loan. Government employee, annual income INR 800,000. "
            "Loan: INR 3,000,000. Customer CUST-004. Docs: PAN type PAN-VALID."
        ),
        "expected_decision": "refer_to_committee",
        "expected_section": "Section 4.1",
    },
]

FAITHFULNESS_THRESHOLD = 2  # keyword retrieval; embedding retrieval would score ~4+
RELEVANCE_THRESHOLD    = 4


def run_evaluation() -> None:
    """Run all test cases, judge each, log to JSONL, print a summary table."""
    EVAL_LOG.parent.mkdir(exist_ok=True)
    results = []

    for tc in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"Evaluating {tc['id']}: {tc['description']}")
        print("-" * 60)

        # Run the full RAG assessment
        rag_system = build_rag_system_prompt(tc["input"])
        chunks = sop_store.retrieve(tc["input"])
        sop_ctx = "\n\n".join(f"[SOP: {c.section}]\n{c.text}" for c in chunks)

        # TODO 3: Call run_rag_assessment(tc["input"]) to get assessment
        # TODO 4: Collect tool_results string (you'll need to capture them — see run_agentic_assessment)
        # For now, use a placeholder
        tool_results_str = "(tool results captured during agentic loop)"

        assessment = run_rag_assessment(tc["input"])

        faith = judge_faithfulness(tc["input"], sop_ctx, tool_results_str, assessment)
        rel   = judge_relevance(tc["input"], assessment)

        decision_match = assessment.preliminary_decision == tc["expected_decision"]
        section_cited  = tc["expected_section"].lower() in assessment.policy_basis.lower()
        faith_pass     = faith["score"] >= FAITHFULNESS_THRESHOLD
        rel_pass       = rel["score"]   >= RELEVANCE_THRESHOLD
        overall_pass   = decision_match and section_cited and faith_pass and rel_pass

        print(f"  Decision : {assessment.preliminary_decision} (expected {tc['expected_decision']}) "
              f"{'✓' if decision_match else '✗'}")
        print(f"  Section  : '{tc['expected_section']}' cited {'✓' if section_cited else '✗'}")
        print(f"  Faithful : {faith['score']}/5 — {faith['reasoning']}")
        print(f"  Relevant : {rel['score']}/5 — {rel['reasoning']}")
        print(f"  RESULT   : {'PASS' if overall_pass else 'FAIL'}")

        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "case_id": tc["id"],
            "description": tc["description"],
            "input": tc["input"],
            "assessment": assessment.model_dump(),
            "expected_decision": tc["expected_decision"],
            "decision_match": decision_match,
            "section_cited": section_cited,
            "faithfulness": faith,
            "relevance": rel,
            "pass": overall_pass,
        }
        results.append(record)

        with open(EVAL_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Summary
    passed = sum(1 for r in results if r["pass"])
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {passed}/{len(results)} PASSED")
    if passed < len(results):
        print("Failed cases:")
        for r in results:
            if not r["pass"]:
                print(f"  {r['case_id']}: {r['description']}")


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
