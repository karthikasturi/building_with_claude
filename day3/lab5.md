# Lab 5 — Loan Eligibility & Credit Bureau Tool Agent
**Module 5: Tool Use and Function Integration**

## Objective
Build a manual agentic loop with two tools that Claude uses to check loan eligibility and fetch credit bureau reports from the Apex Bank loan origination system.

## Starter file
`day3/invoice_tool_agent.py`

## What you will build
An agent that:
- Defines two tools: `check_loan_eligibility` and `get_credit_bureau_report`
- Runs the manual agentic loop (not the tool runner) with debug logging
- Handles `is_error` responses for unknown customers
- Tests four scenarios: eligible customer, ineligible customer, bureau report lookup, unknown customer

## Key concepts
- Tool definition: `name`, `description`, `input_schema`
- Manual loop: `while True` → break on `stop_reason == "end_turn"`
- Append `response.content` (not just text) to preserve `tool_use` blocks
- `is_error: True` in tool results — Claude acknowledges errors gracefully

## Success criteria
| Check | Pass condition |
|-------|---------------|
| Two tools defined | `check_loan_eligibility` and `get_credit_bureau_report` present |
| Manual loop | `while True`; breaks on `end_turn` |
| Full history | `response.content` appended each iteration |
| Tool result format | `tool_use_id` matches; `is_error` used for error case |
| Debug logging | Tool name + inputs printed before each execution |
| All 4 scenarios | Eligible, ineligible, bureau lookup, unknown customer all handled |
