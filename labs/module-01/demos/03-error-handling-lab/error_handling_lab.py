"""
Error Handling Playground
Module 1: API Setup and Secure Integration
Based on: day1/secure_call.py, Part 5 (Error Handling)

Shows the four typed exceptions the Anthropic SDK raises and the correct
handler for each — WITHOUT needing to actually break your key, your network,
or wait for a real rate limit. Use --simulate to inject a fake failure and
watch the matching except-block run.

Run:
    uv run error_handling_lab.py                       # real call, happy path
    uv run error_handling_lab.py --simulate auth        # AuthenticationError
    uv run error_handling_lab.py --simulate rate_limit  # RateLimitError
    uv run error_handling_lab.py --simulate connection  # APIConnectionError
    uv run error_handling_lab.py --simulate status      # APIStatusError (500)
    uv run error_handling_lab.py --simulate all         # run all four, back to back
"""

import argparse
import os

import anthropic
import httpx
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env first.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")


# ── Fault injection helpers (demo-only — never ship this to production) ────
#
# Each helper builds a *real* instance of the SDK's exception class, using a
# fake httpx request/response so it looks exactly like what the SDK would
# raise for a genuine failure. This lets us exercise every except-block on
# demand instead of hoping to hit each one live.

def _fake_response(status_code: int, headers: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status_code, headers=headers or {}, request=request)


def make_auth_error() -> anthropic.AuthenticationError:
    return anthropic.AuthenticationError(
        "invalid x-api-key", response=_fake_response(401), body=None
    )


def make_rate_limit_error() -> anthropic.RateLimitError:
    return anthropic.RateLimitError(
        "rate limit exceeded",
        response=_fake_response(429, headers={"retry-after": "17"}),
        body=None,
    )


def make_connection_error() -> anthropic.APIConnectionError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(message="Connection error.", request=request)


def make_status_error() -> anthropic.APIStatusError:
    return anthropic.APIStatusError(
        "internal server error", response=_fake_response(500), body=None
    )


SIMULATORS = {
    "auth": make_auth_error,
    "rate_limit": make_rate_limit_error,
    "connection": make_connection_error,
    "status": make_status_error,
}


# ── The pattern every lab in this course uses ───────────────────────────────

def guarded_call(question: str, *, inject: Exception | None = None) -> None:
    """Call Claude with full, typed error handling.

    If `inject` is set, we raise it instead of making a real network call —
    that's the only thing that differs from production code below.
    """
    try:
        if inject is not None:
            raise inject

        response = client.messages.create(
            model=MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": question}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        print(f"  ✅ Success: {text}")

    except anthropic.AuthenticationError:
        print("  🔑 AuthenticationError → bad/missing key. Fix ANTHROPIC_API_KEY in .env. Do NOT retry.")

    except anthropic.RateLimitError as e:
        retry_after = e.response.headers.get("retry-after", "60")
        print(f"  ⏳ RateLimitError → back off and retry after {retry_after}s.")

    except anthropic.APIConnectionError:
        print("  🌐 APIConnectionError → network unreachable. Check connectivity, then retry.")

    except anthropic.APIStatusError as e:
        print(f"  🛑 APIStatusError → API returned {e.status_code}. Log and inspect e.message.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulate",
        choices=[*SIMULATORS.keys(), "all"],
        default=None,
        help="Inject a specific failure instead of making a real API call.",
    )
    args = parser.parse_args()

    question = "What is the boiling point of water in Celsius?"

    if args.simulate == "all":
        for name, build in SIMULATORS.items():
            print(f"\n--- Simulating: {name} ---")
            guarded_call(question, inject=build())
        return

    if args.simulate:
        print(f"\n--- Simulating: {args.simulate} ---")
        guarded_call(question, inject=SIMULATORS[args.simulate]())
        return

    print("--- Real call (no simulation) ---")
    guarded_call(question)


if __name__ == "__main__":
    main()
