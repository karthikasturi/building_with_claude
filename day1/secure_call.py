"""
Lab 1 — Secure Claude Call
Module 1: API Setup and Secure Integration

Run: python secure_call.py
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set. Copy shared/.env.example to .env and add your key."
    )

client = anthropic.Anthropic()

SYSTEM_PROMPT = (
    "You are a senior credit analyst at Apex Bank. "
    "Answer questions about credit policy in plain English for loan officers. "
    "Be concise and cite the relevant policy section when possible."
    "do not invent the answer if it is not in the policy document."
    "instead, respond with: I am not authorized to answer that question based on the provided policy document."
)

QUESTION = "What is the maximum debt-to-income ratio allowed for a home loan?"


def main():
    messages = [{"role": "user", "content": QUESTION}]

    try:
        count = client.messages.count_tokens(
            model="claude-haiku-4-5",
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        print(f"Estimated input tokens : {count.input_tokens}")
        print(f"Estimated cost         : ${count.input_tokens * 0.000001:.6f}\n")

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
    except anthropic.AuthenticationError:
        print("ERROR: Invalid API key — check ANTHROPIC_API_KEY in your .env file.")
        raise SystemExit(1)
    except anthropic.RateLimitError as e:
        retry_after = int(e.response.headers.get("retry-after", "60"))
        print(f"ERROR: Rate limited — retry after {retry_after} seconds.")
        raise SystemExit(1)
    except anthropic.APIConnectionError:
        print("ERROR: Cannot reach the API — check your network connection.")
        raise SystemExit(1)
    except anthropic.APIStatusError as e:
        print(f"ERROR: API returned status {e.status_code}: {e.message}")
        raise SystemExit(1)

    if response.stop_reason == "max_tokens":
        print("WARNING: Response was truncated — consider increasing max_tokens\n")

    print("=== complete Response ===")
    print(response)

    print("=== Response ===")
    for block in response.content:
        if block.type == "text":
            print(block.text)

    print("\n=== Usage ===")
    print(f"Input tokens  : {response.usage.input_tokens}")
    print(f"Output tokens : {response.usage.output_tokens}")
    print(f"Request ID    : {response._request_id}")


if __name__ == "__main__":
    main()
