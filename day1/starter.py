"""
Apex Bank — Loan Origination Assistant
Day 1: Secure Foundation

Lab goal: secure client → system prompt → single-turn assessment → error handling
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY is not set. Check your .env file.")

client = anthropic.Anthropic()

SYSTEM_PROMPT = """
You are a loan intake assistant for Apex Bank.

Your job: analyse the loan application details provided and return a clear preliminary
assessment with your reasoning.

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

OUTPUT FORMAT:
Decision: <proceed | refer_to_committee | decline>
Reason: <cite the policy section(s) that apply>
Next steps: <what the loan officer should do>
"""


def assess_loan(application_text: str) -> str:
    """Send a loan application description and return Claude's assessment."""
    messages = [{"role": "user", "content": application_text}]

    try:
        count = client.messages.count_tokens(
            model="claude-haiku-4-5",
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        print(f"  [tokens] input estimate: {count.input_tokens} "
              f"(~${count.input_tokens * 0.000001:.6f})")

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
    except anthropic.AuthenticationError:
        raise SystemExit("ERROR: Invalid API key — check ANTHROPIC_API_KEY in .env")
    except anthropic.RateLimitError as e:
        retry_after = e.response.headers.get("retry-after", "60")
        raise SystemExit(f"ERROR: Rate limited — retry after {retry_after}s")
    except anthropic.APIConnectionError:
        raise SystemExit("ERROR: Cannot reach the API — check your network connection.")
    except anthropic.APIStatusError as e:
        raise SystemExit(f"ERROR: API {e.status_code}: {e.message}")

    if response.stop_reason == "max_tokens":
        print("  WARNING: response truncated — increase max_tokens")

    print(f"  [tokens] input: {response.usage.input_tokens} "
          f"output: {response.usage.output_tokens} "
          f"request_id: {response._request_id}")

    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def main():
    applications = [
        (
            "F-TC-01",
            "Home loan application. Salaried applicant, annual income INR 900,000. "
            "Loan requested: INR 4,500,000. Customer ID: CUST-001. "
            "Documents: PAN-VALID, AADH-VALID.",
        ),
        (
            "F-TC-02",
            "Personal loan application. Self-employed applicant, annual income INR 600,000. "
            "Loan requested: INR 500,000. Customer ID: CUST-002. Documents: PAN-VALID.",
        ),
    ]

    for case_id, application in applications:
        print(f"\n{'='*60}")
        print(f"Case: {case_id}")
        print(f"Application: {application[:80]}...")
        print("-" * 60)
        result = assess_loan(application)
        print(result)


if __name__ == "__main__":
    main()
