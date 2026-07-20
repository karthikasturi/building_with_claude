"""
Secure Hello, Claude
Module 1: API Setup and Secure Integration
Based on: day1/secure_call.py

The smallest possible *correct* Claude call: a key loaded from the
environment (never hardcoded), a single request, and a response read safely.

Run:
    uv run secure_hello.py
    uv run secure_hello.py --question "What is the capital of France?"
"""

import argparse
import os

import anthropic
from dotenv import load_dotenv

# ── Step 1: load the environment BEFORE reading anything from it ───────────
load_dotenv()

# ── Step 2: fail fast with a helpful message if the key is missing ─────────
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set.\n"
        "Copy .env.example to .env in this folder and add your key, e.g.:\n"
        "    cp .env.example .env"
    )

# ── Step 3: build the client — no api_key= argument, the SDK reads the env ─
client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

SYSTEM_PROMPT = "You are a friendly, concise assistant. Answer in two sentences or fewer."


def ask(question: str) -> None:
    print(f"\nModel   : {MODEL}")
    print(f"Question: {question}\n")

    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )

    # Always check *why* generation stopped before trusting the content.
    if response.stop_reason == "max_tokens":
        print("WARNING: response was truncated — consider raising max_tokens\n")

    # Iterate content blocks rather than indexing [0] — the safe pattern.
    answer = next((b.text for b in response.content if b.type == "text"), "")
    print(f"Claude  : {answer}")

    print("\n--- Usage (this is what you'd log in a real app) ---")
    print(f"Input tokens  : {response.usage.input_tokens}")
    print(f"Output tokens : {response.usage.output_tokens}")
    print(f"Request ID    : {response._request_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="In one sentence, what is a Large Language Model?",
        help="The question to send to Claude.",
    )
    args = parser.parse_args()
    ask(args.question)


if __name__ == "__main__":
    main()
