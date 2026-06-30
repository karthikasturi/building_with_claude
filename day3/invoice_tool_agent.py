"""
Lab 5 — Loan Eligibility & Credit Bureau Tool Agent
Module 5: Tool Use and Function Integration

Run: python invoice_tool_agent.py
"""

import os
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ── Mock data ──────────────────────────────────────────────────────────────────

CUSTOMER_DB = {
    "CUST-001": {"name": "Rahul Verma",  "credit_score": 742, "risk_band": "low",
                 "existing_emi": 8500,  "dpd": 0,  "monthly_income_inr": 75_000},
    "CUST-002": {"name": "Sunita Rao",   "credit_score": 541, "risk_band": "high",
                 "existing_emi": 32000, "dpd": 15, "monthly_income_inr": 50_000},
    "CUST-003": {"name": "Arun Mehta",   "credit_score": 680, "risk_band": "medium",
                 "existing_emi": 0,     "dpd": 0,  "monthly_income_inr": 100_000},
}

ELIGIBILITY_RULES = {
    "home_loan":     {"min_score": 680, "max_dti": 0.40},
    "personal_loan": {"min_score": 720, "max_dti": 0.35},
    "business_loan": {"min_score": 700, "max_dti": 0.40},
    "vehicle_loan":  {"min_score": 680, "max_dti": 0.45},
}

# ── Tool implementations ────────────────────────────────────────────────────────

def check_loan_eligibility(customer_id: str, loan_type: str,
                            loan_amount_inr: float) -> dict:
    """Check whether a customer is eligible for the requested loan."""
    customer = CUSTOMER_DB.get(customer_id)
    if not customer:
        return {"error": f"Customer {customer_id} not found in Apex Bank records."}

    rules = ELIGIBILITY_RULES.get(loan_type)
    if not rules:
        return {"error": f"Unknown loan type: {loan_type}"}

    estimated_new_emi = loan_amount_inr * 0.008
    dti = (customer["existing_emi"] + estimated_new_emi) / customer["monthly_income_inr"]

    eligible = (
        customer["credit_score"] >= rules["min_score"]
        and dti <= rules["max_dti"]
        and customer["dpd"] == 0
    )

    return {
        "customer_id":   customer_id,
        "customer_name": customer["name"],
        "credit_score":  customer["credit_score"],
        "risk_band":     customer["risk_band"],
        "existing_emi":  customer["existing_emi"],
        "estimated_dti": round(dti, 3),
        "eligible":      eligible,
        "reason": (
            "All checks passed." if eligible
            else (f"Failed — score {customer['credit_score']} "
                  f"(min {rules['min_score']}), DTI {dti:.1%} "
                  f"(max {rules['max_dti']:.0%}), DPD {customer['dpd']}")
        ),
    }


def get_credit_bureau_report(customer_id: str) -> dict:
    """Fetch a credit bureau report for a customer."""
    customer = CUSTOMER_DB.get(customer_id)
    if not customer:
        return {"error": f"Customer {customer_id} not found."}
    return {
        "customer_id":  customer_id,
        "bureau":       "CIBIL",
        "credit_score": customer["credit_score"],
        "risk_band":    customer["risk_band"],
        "dpd":          customer["dpd"],
        "report_date":  "2026-06-27",
    }


# ── Tool definitions ────────────────────────────────────────────────────────────

tools = [
    {
        "name": "check_loan_eligibility",
        "description": (
            "Check whether a customer is eligible for a loan. "
            "Returns credit score, DTI ratio, and an eligibility verdict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Apex Bank customer ID, e.g. CUST-001",
                },
                "loan_type": {
                    "type": "string",
                    "description": "One of: home_loan, personal_loan, business_loan, vehicle_loan",
                },
                "loan_amount_inr": {
                    "type": "number",
                    "description": "Requested loan amount in INR",
                },
            },
            "required": ["customer_id", "loan_type", "loan_amount_inr"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_credit_bureau_report",
        "description": (
            "Fetch a CIBIL credit bureau report for a customer. "
            "Returns credit score, risk band, and days-past-due (DPD)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Apex Bank customer ID, e.g. CUST-001",
                },
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
]

TOOL_FN_MAP = {
    "check_loan_eligibility":  check_loan_eligibility,
    "get_credit_bureau_report": get_credit_bureau_report,
}


def run_agent(user_query: str) -> str:
    """Manual agentic loop: send query, execute tool calls, return final answer."""
    messages = [{"role": "user", "content": user_query}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  → Calling {block.name}({block.input})")
                fn = TOOL_FN_MAP.get(block.name)
                if fn:
                    result = fn(**block.input)
                    is_error = "error" in result
                else:
                    result = {"error": f"Unknown tool: {block.name}"}
                    is_error = True
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result),
                    "is_error":    is_error,
                })

        messages.append({"role": "user", "content": tool_results})

    return next((b.text for b in response.content if b.type == "text"), "")


TEST_SCENARIOS = [
    "Is CUST-001 eligible for a home loan of INR 45 lakhs?",
    "Can CUST-002 get a personal loan of INR 5 lakhs?",
    "Pull the CIBIL report for CUST-003 and tell me their risk band.",
    "Check eligibility for CUST-999 for a business loan of INR 10 lakhs.",
]


def main():
    for i, query in enumerate(TEST_SCENARIOS, 1):
        print(f"\n{'='*60}")
        print(f"Scenario {i}: {query}")
        print("-" * 60)
        answer = run_agent(query)
        print(f"\n[Claude] {answer}")


if __name__ == "__main__":
    main()
