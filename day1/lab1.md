# Lab 1 — Secure Claude Call
**Module 1: API Setup and Secure Integration**  
**Duration:** 55 minutes

---

## Objective

Build a Python script that calls Claude securely — key loaded from the environment, not the source code — and prints a structured response with token usage and request tracking.

---

## Pre-requisites

1. Python 3.10+ installed
2. API key set in your environment:
   ```bash
   cp ../../shared/.env.example .env
   # Edit .env and set ANTHROPIC_API_KEY=your-key-here
   ```
3. Dependencies installed:
   ```bash
   pip install anthropic python-dotenv
   ```

---

## Part 1 — Secure Client Setup (10 min)

Open `code/secure_call.py`. You'll see a skeleton with TODO comments. Complete each TODO in order.

**Step 1.1** — Load the environment and initialize the client:
```python
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from environment
```

**Step 1.2** — Verify the key is present before making any API call:
```python
import os
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY is not set. Check your .env file.")
```

**Check:** Run `python code/secure_call.py` — you should see a client object printed, not a crash.

---

## Part 2 — Token Counting Before the Call (10 min)

Before spending tokens, count how many the request will use.

```python
system_prompt = "You are a senior credit analyst at Apex Bank..."
messages = [{"role": "user", "content": question}]

count = client.messages.count_tokens(
    model="claude-haiku-4-5",
    system=system_prompt,
    messages=messages,
)
print(f"Estimated input tokens: {count.input_tokens}")
print(f"Estimated cost: ${count.input_tokens * 0.000005:.6f}")
```

**Note:** `count_tokens()` does not make an inference call — it counts the tokens without charging for output.

---

## Part 3 — Make the API Call (15 min)

Send the question and read the response correctly:

```python
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=1024,
    system=system_prompt,
    messages=messages,
)

# Always check stop_reason before reading content
if response.stop_reason == "max_tokens":
    print("WARNING: Response was truncated — increase max_tokens")

for block in response.content:
    if block.type == "text":
        print(block.text)
```

---

## Part 4 — Print Usage and Request ID (5 min)

Every response carries tracking metadata you should always log in production:

```python
print(f"\n--- Usage ---")
print(f"Input tokens:  {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Request ID:    {response._request_id}")
```

The `_request_id` is what you give to Anthropic support if a call behaves unexpectedly.

---

## Part 5 — Error Handling (15 min)

Wrap the API call in the correct exception handlers:

```python
import anthropic

try:
    response = client.messages.create(...)
except anthropic.AuthenticationError:
    print("ERROR: Invalid API key. Check ANTHROPIC_API_KEY in your .env file.")
    raise SystemExit(1)
except anthropic.RateLimitError as e:
    retry_after = int(e.response.headers.get("retry-after", "60"))
    print(f"ERROR: Rate limited. Wait {retry_after} seconds and retry.")
    raise SystemExit(1)
except anthropic.APIConnectionError:
    print("ERROR: Cannot reach the API. Check your network connection.")
    raise SystemExit(1)
except anthropic.APIStatusError as e:
    print(f"ERROR: API returned status {e.status_code}: {e.message}")
    raise SystemExit(1)
```

**Test your error handling:**
- Temporarily set `ANTHROPIC_API_KEY=bad-key` — you should see the `AuthenticationError` message.
- Restore the correct key before continuing.

---

## Success Criteria

Run the check command:
```bash
grep -r "sk-ant" code/      # must return nothing
python code/secure_call.py  # must print response + usage + request_id
```

| Criteria | How to verify |
|----------|--------------|
| No hardcoded key | `grep -r "sk-ant" code/` returns nothing |
| Key from environment | `ANTHROPIC_API_KEY` read via `os.environ` or `dotenv` |
| Token counted before call | `count_tokens()` output printed before the response |
| stop_reason checked | Code checks `response.stop_reason` before reading content |
| Error handlers present | `AuthenticationError` and `RateLimitError` both caught |
| Usage printed | Input tokens, output tokens, and `_request_id` all printed |

---

## Stretch Goals

If you finish early:
1. Add a `--question` CLI argument using `argparse` so the question can be passed from the terminal
2. Try `client.messages.stream()` and print tokens as they arrive
3. Add `max_retries=5` to the client and read the SDK docs on what triggers an auto-retry
