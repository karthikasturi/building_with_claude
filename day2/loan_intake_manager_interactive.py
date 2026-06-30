"""
Lab 4 — Loan Intake Conversation Manager (Interactive)
Module 4: Conversation and Context Management

Interactive version — type your responses at the prompt.
Type 'exit' or 'quit' to end the session early.

Run: python loan_intake_manager_interactive.py
"""

import os
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel
from typing import Literal

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """
You are a loan intake officer at Apex Bank.
Your job is to collect the following information from the applicant across the conversation:
  - Full name
  - Loan type (home / personal / business / vehicle)
  - Loan amount required (in INR)
  - Monthly income (in INR)
  - Desired tenure (in years)
  - Approximate credit score (if known)

Ask for one or two pieces of information at a time. Be polite and professional.
When you have collected all details, say: "INTAKE COMPLETE" and summarise the application.
"""

TOKEN_WARN_THRESHOLD = 50_000


class ConversationManager:
    def __init__(self, client: anthropic.Anthropic, system: str):
        self.client = client
        self.system = system
        self.messages: list[dict] = []

    def send(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=self.system,
            messages=self.messages,
        )
        reply = next((b.text for b in response.content if b.type == "text"), "")
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def token_count(self) -> int:
        result = self.client.messages.count_tokens(
            model=MODEL,
            system=self.system,
            messages=self.messages,
        )
        return result.input_tokens

    def summarise_and_reset(self) -> str:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self.messages
        )
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": (
                    "Summarise this loan intake conversation in ≤150 words. "
                    "Preserve: applicant name, loan type, amount, income, tenure, credit score.\n\n"
                    f"{history_text}"
                ),
            }],
        )
        summary = next((b.text for b in response.content if b.type == "text"), "")
        self.messages = [{"role": "user", "content": f"[Session summary]\n{summary}"}]
        return summary


class IntakeSummary(BaseModel):
    applicant_type: Literal["salaried", "self_employed", "unknown"]
    loan_amount_inr: float
    credit_checked: bool
    recommended_action: Literal["proceed", "review", "decline"]


def main():
    manager = ConversationManager(client, SYSTEM_PROMPT)

    print("=== Apex Bank — Loan Intake Session ===")
    print("Type 'exit' or 'quit' to end the session.\n")

    # Opening prompt from the officer
    opening = manager.send("Hello, I'd like to apply for a loan.")
    print(f"[Officer] {opening}\n")

    while True:
        user_input = input("[You] ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("\nSession ended by user.")
            break

        reply = manager.send(user_input)
        print(f"\n[Officer] {reply}\n")

        tokens = manager.token_count()
        print(f"  [tokens: {tokens:,}]")
        if tokens > TOKEN_WARN_THRESHOLD:
            print("  [INFO] Compressing conversation history...")
            summary = manager.summarise_and_reset()
            print(f"  [Summary] {summary}")
        print()

        if "INTAKE COMPLETE" in reply:
            break

    if not manager.messages:
        return

    # Structured summary
    print("=== Structured Session Summary ===")
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in manager.messages
    )
    result = client.messages.parse(
        model=MODEL,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                "Based on this loan intake conversation, classify the application:\n\n"
                f"{history_text}"
            ),
        }],
        output_format=IntakeSummary,
    )
    s = result.parsed_output
    print(f"Applicant type     : {s.applicant_type}")
    print(f"Loan amount (INR)  : {s.loan_amount_inr:,.0f}")
    print(f"Credit checked     : {s.credit_checked}")
    print(f"Recommended action : {s.recommended_action}")


if __name__ == "__main__":
    main()
