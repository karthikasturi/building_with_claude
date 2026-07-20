# Module 1 Demos — API Setup and Secure Integration

Three small, beginner-level, single-concept demos that build on the
[`day1/secure_call.py`](../../../day1/secure_call.py) reference and the
[Module 1 guide](../../../guides/module-01-api-setup-and-secure-integration.md).
Each demo isolates **one** idea instead of combining all of Module 1 into one
file, so you can run them in order and see exactly where each concept lives.

Each demo is its **own independent [`uv`](https://docs.astral.sh/uv/)
project** — its own `pyproject.toml`, lockfile, and `.venv` — so you can
`cd` into just one folder and run it without touching the others, or hand
out a single demo folder on its own.

## The three demos

| # | Project | Concept | Needs a real API key? |
|---|---------|---------|------------------------|
| 1 | [`01-secure-hello/`](01-secure-hello/) | SDK setup, key handling, a single request, and reading the response safely (`stop_reason`, content blocks, `usage`) | Yes |
| 2 | [`02-token-cost-explorer/`](02-token-cost-explorer/) | Cost & usage awareness — `count_tokens()` *before* a call vs. `response.usage` *after* it, across several questions | Yes |
| 3 | [`03-error-handling-lab/`](03-error-handling-lab/) | The four typed SDK exceptions (`AuthenticationError`, `RateLimitError`, `APIConnectionError`, `APIStatusError`) and the correct handler for each | No — has a `--simulate` mode |

Each has its own README with setup/run instructions, but the shape is the
same for all three:

```bash
cd labs/module-01/demos/01-secure-hello   # or 02-... / 03-...

uv sync                # creates that project's own .venv
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...

uv run <script>.py
```

## Notes

- Same security rule as the rest of the course: never hardcode a key, never
  commit `.env` (the repo's root `.gitignore` already excludes `.env` and
  every project's `.venv/` here).
- Dollar figures printed by Demo 2 (and any pricing-shaped numbers) use a
  small illustrative rate constant — not real pricing. Check current
  per-model rates at [platform.claude.com/docs](https://platform.claude.com/docs)
  before using a number for an actual budget.
- Demo 3's `--simulate` mode builds *real* instances of each SDK exception
  class (using a fake `httpx` request/response) and raises them locally, so
  you can see every `except` branch fire without needing a broken key, a
  dead network, or an actual rate limit — see `_fake_response()` in
  `03-error-handling-lab/error_handling_lab.py`.
