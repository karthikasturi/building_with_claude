"""
Apex Bank — Loan Origination Assistant
Day 2 start: Day 1 complete. Today: Structured Outputs + Conversation Manager.

New today:
  - LoanAssessment Pydantic schema
  - client.messages.parse() with 2-retry loop on ValidationError
  - LoanConversationManager (token-aware multi-turn session)
"""

import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY is not set. Check your .env file.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """
You are a loan intake assistant for Apex Bank.

Your job: analyse the loan application details provided, then return a structured
JSON assessment using the exact schema specified.

ELIGIBILITY RULES (always cite section when applying):
- [Section 1.1] Minimum credit score: 680 (salaried/government), 700 (self-employed)
- [Section 2.1] Maximum DTI: 40% salaried, 35% self-employed, 45% government
- [Section 2.2] Loans above INR 5,000,000 require credit committee review
- [Section 3.1] Valid PAN and Aadhaar required for all applicants
- [Section 4.1] If bureau unavailable, refer to committee — never decline

DTI formula: (existing_emi + estimated_new_emi) / gross_monthly_income

DECISION VALUES:
- "proceed"            — all checks pass within policy thresholds
- "refer_to_committee" — loan > INR 5M, bureau unavailable, or borderline DTI
- "decline"            — credit score below minimum, DTI exceeds maximum, invalid document

OUTPUT: Return a single JSON object matching the schema. No prose, no markdown fences.
"""

# ── Day 1: single-turn text assessment (kept for reference) ──────────────────

def assess_loan_text(application_text: str) -> str:
    """Day 1 approach — unstructured text response."""
    messages = [{"role": "user", "content": application_text}]

    count = client.messages.count_tokens(
        model=MODEL,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    print(f"  [tokens] input estimate: {count.input_tokens} "
          f"(~${count.input_tokens * 0.000005:.5f})")

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
    except anthropic.AuthenticationError:
        raise SystemExit("ERROR: Invalid API key.")
    except anthropic.RateLimitError as e:
        raise SystemExit(f"ERROR: Rate limited. Retry after "
                         f"{e.response.headers.get('retry-after', '60')}s")
    except anthropic.APIConnectionError:
        raise SystemExit("ERROR: Cannot reach API. Check network.")
    except anthropic.APIStatusError as e:
        raise SystemExit(f"ERROR: API {e.status_code}: {e.message}")

    if response.stop_reason == "max_tokens":
        print("  WARNING: response truncated")

    print(f"  [tokens] input: {response.usage.input_tokens} "
          f"output: {response.usage.output_tokens} "
          f"request_id: {response._request_id}")

    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


# ── Day 2: Structured Outputs ─────────────────────────────────────────────────

from typing import Literal, Optional
from pydantic import BaseModel, ValidationError

# TODO 1: Define the LoanAssessment Pydantic schema with these fields:
#   applicant_type: Literal["salaried", "self_employed", "government"]
#   annual_income_inr: float
#   loan_type: Literal["home", "personal", "business", "vehicle"]
#   loan_amount_requested_inr: float
#   existing_emi_inr: float
#   dti_ratio: float
#   credit_score: Optional[int]      ← null if bureau unavailable
#   documents_verified: bool
#   preliminary_decision: Literal["proceed", "refer_to_committee", "decline"]
#   policy_basis: str                ← must cite the section number
class LoanAssessment(BaseModel):
    applicant_type: Literal["salaried", "self_employed", "government"]
    annual_income_inr: float
    loan_type: Literal["home", "personal", "business", "vehicle"]
    loan_amount_requested_inr: float
    existing_emi_inr: float
    dti_ratio: float
    credit_score: Optional[int] = None
    documents_verified: bool
    preliminary_decision: Literal["proceed", "refer_to_committee", "decline"]
    policy_basis: str


def parse_loan_assessment(application_text: str) -> LoanAssessment:
    """Parse a loan application into a validated LoanAssessment.

    Uses client.messages.parse() which validates against the schema.
    Retries once on ValidationError with the error detail in the re-prompt.
    """
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


# ── Day 2: Conversation Manager ───────────────────────────────────────────────

class LoanConversationManager:
    """Maintains a multi-turn loan intake session with token guardrails."""

    TOKEN_WARN  = 30_000   # print warning when history exceeds this
    TOKEN_RESET = 60_000   # summarise and reset history at this threshold

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


def main():
    print("=== Structured Output Demo ===\n")

    cases = [
        (
            "F-TC-01",
            "Home loan. Salaried, annual income INR 900,000. "
            "Loan: INR 4,500,000. Customer CUST-001. Docs: PAN-VALID, AADH-VALID.",
        ),
        (
            "F-TC-02",
            "Personal loan. Self-employed, annual income INR 600,000. "
            "Loan: INR 500,000. Customer CUST-002. Docs: PAN-VALID.",
        ),
    ]

    for case_id, text in cases:
        print(f"{'='*60}")
        print(f"Case {case_id}")
        print("-" * 60)
        assessment = parse_loan_assessment(text)
        print(f"Decision : {assessment.preliminary_decision}")
        print(f"DTI      : {assessment.dti_ratio:.1%}")
        print(f"Credit   : {assessment.credit_score}")
        print(f"Basis    : {assessment.policy_basis}")
        print()


if __name__ == "__main__":
    main()
