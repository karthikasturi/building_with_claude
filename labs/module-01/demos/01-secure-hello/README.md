# Demo 1 — Secure Hello, Claude

**Module 1: API Setup and Secure Integration**
Based on [`day1/secure_call.py`](../../../../day1/secure_call.py).

The smallest possible *correct* Claude call: SDK setup, key loaded from the
environment (never hardcoded), a single request, and a response read safely
(`stop_reason`, content blocks, `usage`).

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
uv run secure_hello.py
uv run secure_hello.py --question "What is the capital of France?"
```

## What to notice

- `load_dotenv()` runs before any `os.environ` read.
- `anthropic.Anthropic()` takes no `api_key=` argument — the SDK reads
  `ANTHROPIC_API_KEY` itself.
- `response.stop_reason` is checked before the content is trusted.
- `response.content` is iterated (filtering `type == "text"`), never indexed
  blindly with `[0]`.
- `response.usage` and `response._request_id` are printed — the minimum
  you'd log in a real application.
