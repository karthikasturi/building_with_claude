# Lab 4 — Loan Intake Conversation Manager
**Module 4: Conversation and Context Management**

## Objective
Build a multi-turn conversation manager that collects loan applicant details across several exchanges and produces a structured summary at the end.

## Starter file
`day2/loan_intake_manager.py`

## What you will build
A `ConversationManager` class that:
- Maintains the full message history across turns
- Monitors token count after each turn (warns above 50K)
- Compresses history via summarise-and-reset after 3 exchanges
- Produces a structured JSON summary at the end of the session

## Key concepts
- Stateless API — full `messages` list sent on every call
- `client.messages.count_tokens()` — monitor context growth
- Summarise-and-reset — distil history into a compact summary, restart thread
- Structured final output — `client.messages.parse()` for the session summary

## Success criteria
| Check | Pass condition |
|-------|---------------|
| Multi-turn | At least 4 exchanges maintained correctly |
| Full history | All prior messages sent on every API call |
| Token monitoring | Count printed after each turn; warning above 50K |
| Summarise & reset | Triggered and demonstrated after turn 3 |
| Structured summary | Final JSON matches the required schema |
