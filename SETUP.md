# Setup Guide — Building with Claude

Follow these steps once before your first lab session. You only need to do this one time.

---

## 1. Clone the repository

```bash
git clone https://github.com/karthikasturi/building_with_claude.git
cd building_with_claude
```

---

## 2. Create a Python virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> Your prompt should change to show `(.venv)` when the environment is active. Run `deactivate` to exit it.

---

## 3. Install dependencies

```bash
pip install -r shared/requirements.txt
```

---

## 4. Configure API keys

Copy the example env file and fill in your keys:

```bash
cp shared/.env.example .env
```

Open `.env` in your editor and set the following:

```env
# Required for every lab (Days 1–5)
ANTHROPIC_API_KEY=sk-ant-...

# Required for the RAG labs (Day 3 onwards)
OPENAI_API_KEY=sk-proj-...
```

**Where to get keys:**
- Anthropic: [console.anthropic.com](https://console.anthropic.com) → API Keys
- OpenAI: [platform.openai.com](https://platform.openai.com) → API Keys

> **Never commit `.env`** — it is already listed in `.gitignore`.

---

## 5. Verify your setup

Run the Day 1 lab to confirm the Anthropic key works:

```bash
python day1/secure_call.py
```

You should see a loan assessment printed to the terminal without any authentication errors.

---

## 6. Run the labs in order

Each day builds on the previous one. Run them in this sequence:

```bash
# Day 1 — API basics and secure integration
python day1/secure_call.py

# Day 2 — Structured output and multi-turn conversation
python day2/loan_application_extractor.py
python day2/loan_intake_manager.py

# Day 3 — Tool use, agentic loop, and RAG indexing
python day3/invoice_tool_agent.py
python day3/rag_pipeline.py          # builds day3/vector_index.json — run this before Day 4

# Day 4 — RAG-enhanced assessment
python day4/rag_assistant.py         # requires day3/vector_index.json

# Day 5 — Evaluation harness and mini-project
python day5/loan_inquiry_assistant.py
python day5/starter.py
```

> `day3/rag_pipeline.py` must complete successfully before Day 4 or Day 5 files will run.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `AuthenticationError` or 401 | Check `ANTHROPIC_API_KEY` in `.env` — must start with `sk-ant-` |
| `OPENAI_API_KEY is not set` | Check `OPENAI_API_KEY` in `.env` — uncomment if the line is prefixed with `#` |
| `ModuleNotFoundError` | Run `pip install -r shared/requirements.txt` with the venv active |
| `FileNotFoundError: vector_index.json` | Run `python day3/rag_pipeline.py` first |
| `(.venv)` not showing in prompt | Virtual environment is not active — re-run the `activate` command for your OS |
| PowerShell: *execution policy* error | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once |
