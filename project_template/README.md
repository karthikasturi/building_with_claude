# Intelligent Loan Origination Assistant — Apex Bank

A project built during the *Building with Claude* programme. You implement
`loan_origination_assistant.py` in five phases, one day at a time.

## Folder structure

```
loan-origination-assistant/
├── loan_origination_assistant.py   ← your main file (implement this)
├── data/
│   ├── apex_bank_credit_policy.md  ← credit policy reference
│   └── loan_processing_sop.md      ← SOP for RAG (Phase 4)
├── eval_logs/                      ← evaluation output (gitignored)
├── .env.example                    ← copy to .env and fill in keys
├── requirements.txt
└── GIT_WORKFLOW.md                 ← step-by-step GitHub guide
```

## Setup

```bash
# 1. Clone your repo (see GIT_WORKFLOW.md for how to create it)
git clone https://github.com/<your-username>/loan-origination-assistant.git
cd loan-origination-assistant

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate.bat

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Open .env and fill in ANTHROPIC_API_KEY (and OPENAI_API_KEY for Phase 4)

# 5. Verify syntax compiles
python -m py_compile loan_origination_assistant.py && echo "OK"
```

## Running

```bash
python loan_origination_assistant.py
```

Each `NotImplementedError` tells you which phase to implement next.

## Implementation phases

| Phase | Day | What you build |
|-------|-----|----------------|
| 1 | Day 1 | `make_client()`, `LOAN_OFFICER_SYSTEM`, `estimate_cost()` |
| 2 | Day 2 | `LoanApplicationRecord`, `ConversationManager`, `extract_application_record()`, `run_intake_conversation()` |
| 3 | Day 3 | `build_tools()`, `run_agentic_loop()` |
| 4 | Day 3–4 | `build_policy_index()`, `retrieve_policy_context()` |
| 5 | Day 4 | `judge_faithfulness()`, `evaluate_tool_correctness()`, `run_evaluation()` |

## Expected final output (all phases complete)

```
Scenario S1: faithfulness=5, tool_correctness=1.0, decision=✓ → PASS
Scenario S2: faithfulness=5, tool_correctness=1.0, decision=✓ → PASS
Scenario S3: faithfulness=4, tool_correctness=1.0, decision=✓ → PASS
Scenario S4: faithfulness=5, tool_correctness=1.0, decision=✓ → PASS
Overall: 4/4 PASS
```
