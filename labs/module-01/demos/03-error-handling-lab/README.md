# Demo 3 — Error Handling Playground

**Module 1: API Setup and Secure Integration**
Based on [`day1/secure_call.py`](../../../../day1/secure_call.py), Part 5 (Error Handling).

Shows the four typed exceptions the Anthropic SDK raises and the correct
handler for each — **without** needing to actually break your key, your
network, or wait for a real rate limit.

Independent [`uv`](https://docs.astral.sh/uv/) project — its own
`pyproject.toml`, lockfile, and `.venv`. Also depends on `httpx` directly
(a transitive dependency of `anthropic`) to build fake request/response
objects for the fault-injection helpers.

## Setup

```bash
uv sync
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
# (a placeholder key is enough if you only plan to use --simulate mode)
```

## Run

```bash
uv run error_handling_lab.py                       # real call, happy path
uv run error_handling_lab.py --simulate auth        # AuthenticationError
uv run error_handling_lab.py --simulate rate_limit  # RateLimitError
uv run error_handling_lab.py --simulate connection  # APIConnectionError
uv run error_handling_lab.py --simulate status      # APIStatusError (500)
uv run error_handling_lab.py --simulate all         # all four, back to back
```

## What to notice

- `--simulate` builds **real** instances of each SDK exception class (using
  a fake `httpx` request/response), so every `except` branch fires exactly
  as it would for a genuine failure — see `_fake_response()` in
  `error_handling_lab.py`.
- Handlers are typed and ordered deliberately:
  `AuthenticationError` → fix the key, don't retry.
  `RateLimitError` → read `retry-after`, back off, retry.
  `APIConnectionError` → check connectivity, retry.
  `APIStatusError` → log `status_code` + `message`.
