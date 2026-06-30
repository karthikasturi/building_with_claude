"""
Lab 3 — Loan Application Data Extraction
Module 3: Structured Outputs and Validation

Run: python loan_application_extractor.py
"""

import os
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, field_validator, ValidationError
from typing import Literal, Optional

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

RAW_APPLICATIONS = [
    "Rajesh Kumar, home loan INR 45L over 20 yrs, monthly income INR 85K, credit score 742.",
    "Sunita Rao, personal loan INR 3L, repay over 3 years, income INR 45K/month.",
    "Arun Mehta, business loan INR 25L, 5-year term, monthly business income INR 2.2L, CIBIL 710.",
    "Deepika Joshi, vehicle loan INR 8.5L, 4 years, income INR 55K/month, credit score 780.",
    "Venkatesh Iyer, home loan INR 70L, 25-year term, household income INR 1.5L/month, score 760.",
]


class LoanApplicationRecord(BaseModel):
    applicant_name: str
    loan_type: Literal["home_loan", "personal_loan", "business_loan", "vehicle_loan"]
    loan_amount_inr: float
    monthly_income_inr: float
    tenure_years: int
    credit_score: Optional[int] = None

    @field_validator("loan_amount_inr")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("loan_amount_inr must be positive")
        return v


def extract_record(raw_text: str, max_retries: int = 2) -> LoanApplicationRecord:
    """Extract a validated LoanApplicationRecord from raw application text."""
    messages = [{"role": "user", "content": f"Extract the loan application fields from: '{raw_text}'"}]

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.parse(
                model=MODEL,
                max_tokens=512,
                temperature=0,
                messages=messages,
                output_format=LoanApplicationRecord,
            )
            return response.parsed_output
        except ValidationError as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to extract after {max_retries} retries. "
                    f"Original text: {raw_text!r}"
                ) from e
            last_text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            messages.append({"role": "assistant", "content": last_text})
            messages.append({
                "role": "user",
                "content": (
                    f"Validation failed:\n{e}\n\n"
                    "Please return a corrected JSON object."
                ),
            })


def main():
    print(f"{'#':<3}  {'Applicant':<20} {'Type':<16} {'Amount (INR)':<14} "
          f"{'Income/mo':<12} {'Tenure':<8} {'Score'}")
    print("-" * 90)

    ok = 0
    fail = 0
    for i, raw in enumerate(RAW_APPLICATIONS, 1):
        try:
            record = extract_record(raw)
            score_str = str(record.credit_score) if record.credit_score else "N/A"
            print(f"{i:<3}  {record.applicant_name:<20} {record.loan_type:<16} "
                  f"{record.loan_amount_inr:<14,.0f} {record.monthly_income_inr:<12,.0f} "
                  f"{record.tenure_years:<8} {score_str}")
            ok += 1
        except Exception as e:
            print(f"{i:<3}  ERROR: {e}")
            fail += 1

    print("-" * 90)
    print(f"Processed: {ok + fail}  |  OK: {ok}  |  Failed: {fail}")


if __name__ == "__main__":
    main()
