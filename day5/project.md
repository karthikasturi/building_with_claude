# Module 8 Mini-Project — Apex Bank Loan Inquiry Assistant
**Day 5: Applied Mini-Project**

## Business context
Apex Bank's customer relations team receives hundreds of loan inquiry messages per day via their web portal. Currently a human agent reads each message, categorises it, and routes it to the right team.

**Your task:** Build a Claude-powered assistant that automates this — reading each incoming inquiry, looking up the customer account, retrieving the relevant SOP, and producing a structured routing record.

## Starter file
`day5/loan_inquiry_assistant.py`

## Prerequisite
Run `day3/rag_pipeline.py` first — the assistant loads `day3/vector_index.json`.

## Architecture

```
Raw inquiry text
     │
     ▼
[Module 2] System prompt — priority rules, escalation rules, output format
     │
     ▼
[Module 5] Tool: lookup_customer(customer_id)
     │              ↓ account details, EMI status, credit flag
     ▼
[Module 6] RAG: retrieve SOP / credit policy for inquiry type
     │              ↓ relevant process / eligibility context
     ▼
[Module 3] Structured output: LoanInquiryRecord (messages.parse)
     │
     ▼
Routing record → team queue
```

## Deliverable checklist
- [ ] `LoanInquiryRecord` Pydantic model with all 7 fields
- [ ] System prompt includes priority rules, escalation rules, output format
- [ ] `lookup_customer` tool defined and returning mock data
- [ ] Manual agentic loop handles 0 or 1 tool calls per inquiry
- [ ] RAG retrieval provides policy/SOP context in the user message
- [ ] `model_validate_json()` produces typed output
- [ ] All 5 test inquiries produce valid `LoanInquiryRecord` instances
- [ ] High-priority inquiries have `escalate=True` and `escalation_reason` set
- [ ] Routing summary printed at the end

## Test inquiries
| # | Inquiry | Expected priority | Expected escalate |
|---|---------|-------------------|-------------------|
| 1 | Home loan status check, CUST-001 | medium | False |
| 2 | Double EMI debit complaint, CUST-002 | high | True |
| 3 | General credit score question | low | False |
| 4 | Senior citizen urgent medical loan, CUST-003 | high | True |
| 5 | Business loan document query | medium | False |
