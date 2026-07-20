# Demo 2 — Token & Cost Explorer

**Module 1: API Setup and Secure Integration**
Based on [`day1/secure_call.py`](../../../../day1/secure_call.py).

Shows the two moments that matter for cost awareness:

1. **Before** the call — `client.messages.count_tokens()` — free, no inference.
2. **After** the call — `response.usage.input_tokens` / `output_tokens` — actual.

Sends four questions of increasing length so you can see token counts scale
with input size, then prints a summary table.

Independent [`uv`](https://docs.astral.sh/uv/) project — its own
`pyproject.toml`, lockfile, and `.venv`.

## Setup

```bash
uv sync
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
uv run token_cost_explorer.py
```

## What to notice

- `count_tokens()` costs nothing and returns instantly — use it to sanity
  check a request before sending it.
- `estimated_input` (pre-call) should match `actual_input` (post-call)
  almost exactly for the same input.
- The USD figures use a small **illustrative** rate constant so the shape of
  a cost calculation is concrete — they are **not** real pricing. Check
  current per-model rates at
  [platform.claude.com/docs](https://platform.claude.com/docs) before using
  a number for a real budget.
