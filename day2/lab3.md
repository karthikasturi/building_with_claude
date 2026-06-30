# Lab 3 — Loan Application Data Extraction
**Module 3: Structured Outputs and Validation**

## Objective
Build a structured extraction pipeline that converts raw loan application text into validated `LoanApplicationRecord` objects using Pydantic and `client.messages.parse()`.

## Starter file
`day2/loan_application_extractor.py`

## What you will build
A script that processes 5 raw applicant descriptions from `shared/data/apex_bank_loan_applications.md` and extracts:
- Applicant name
- Loan type (constrained to 4 values via `Literal`)
- Loan amount (INR)
- Monthly income (INR)
- Tenure (years)
- Credit score (optional)

## Key concepts
- `client.messages.parse(output_format=PydanticModel)` — single-call extraction + validation
- `response.parsed_output` — the validated model instance
- `ValidationError` retry loop — re-prompt with the error message, max 2 retries
- `field_validator` — enforce business rules (e.g. amount > 0)

## Success criteria
| Check | Pass condition |
|-------|---------------|
| Pydantic model | All required fields typed; at least one `Literal` field |
| `messages.parse()` | Used — not `messages.create()` + manual JSON parsing |
| Retry loop | `ValidationError` caught; re-prompt attempted up to 2 times |
| All 5 inputs | All 5 applications processed without unhandled exceptions |
| Failed inputs | Failures reported with original text, not silently skipped |
