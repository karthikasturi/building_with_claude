"""
Token & Cost Explorer
Module 1: API Setup and Secure Integration
Based on: day1/secure_call.py

Shows the two moments that matter for cost awareness:
  1. BEFORE the call  -> client.messages.count_tokens() — free, no inference
  2. AFTER the call   -> response.usage.input_tokens / output_tokens — actual

Sends a handful of questions of increasing length so you can see token counts
scale with input size, then prints a summary table.

Run:
    uv run token_cost_explorer.py

Note: the USD figures here use a small illustrative rate constant so the
*shape* of a cost calculation is concrete. Always check current per-model
pricing at https://platform.claude.com/docs before using real numbers.
"""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env first.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

SYSTEM_PROMPT = "You are a concise assistant. Answer in one short sentence."

# Illustrative only — NOT real pricing. Check platform.claude.com/docs for current rates.
ILLUSTRATIVE_RATE_PER_INPUT_TOKEN_USD = 0.000001
ILLUSTRATIVE_RATE_PER_OUTPUT_TOKEN_USD = 0.000005

QUESTIONS = [
    "What is 2 + 2?",
    "Name three primary colors.",
    "In one sentence, explain what an API is to someone who has never coded before.",
    (
        "Summarize, in one sentence, why developers usually load secrets like API keys "
        "from environment variables instead of hardcoding them directly in source files "
        "that might end up committed to a public repository."
    ),
]


def estimate_input_tokens(question: str) -> int:
    """Pre-call estimate — count_tokens() does not run inference, so it's free."""
    count = client.messages.count_tokens(
        model=MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return count.input_tokens


def ask_and_measure(question: str) -> dict:
    estimated_input = estimate_input_tokens(question)

    response = client.messages.create(
        model=MODEL,
        max_tokens=128,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )

    actual_input = response.usage.input_tokens
    actual_output = response.usage.output_tokens
    cost = (
        actual_input * ILLUSTRATIVE_RATE_PER_INPUT_TOKEN_USD
        + actual_output * ILLUSTRATIVE_RATE_PER_OUTPUT_TOKEN_USD
    )

    return {
        "question": question,
        "estimated_input": estimated_input,
        "actual_input": actual_input,
        "actual_output": actual_output,
        "cost_usd": cost,
    }


def main() -> None:
    print(f"Model: {MODEL}\n")
    print(f"{'Q#':<4}{'est.in':>8}{'act.in':>8}{'out':>8}{'~cost($)':>12}  question")
    print("-" * 90)

    rows = []
    total_cost = 0.0
    for i, q in enumerate(QUESTIONS, 1):
        row = ask_and_measure(q)
        rows.append(row)
        total_cost += row["cost_usd"]
        short_q = (q[:48] + "…") if len(q) > 48 else q
        print(
            f"{i:<4}{row['estimated_input']:>8}{row['actual_input']:>8}"
            f"{row['actual_output']:>8}{row['cost_usd']:>12.6f}  {short_q}"
        )

    print("-" * 90)
    print(f"Total illustrative cost for {len(QUESTIONS)} calls: ${total_cost:.6f}")
    print(
        "\nNotice: estimated_input (pre-call) should match actual_input (post-call) "
        "almost exactly — count_tokens() is deterministic for the same input."
    )


if __name__ == "__main__":
    main()
