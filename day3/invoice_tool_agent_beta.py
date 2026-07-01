"""
Lab 5 — Beta Tool Runner: Loan Eligibility Agent
Module 5: Tool Use and Function Integration

Demonstrates the SDK beta tool runner vs the manual while loop in
invoice_tool_agent.py. Key differences:

  - @beta_tool    : decorator infers the JSON schema from type hints + docstring
  - tool_runner() : SDK drives the agentic loop; no while True / append needed
  - until_done()  : blocks until end_turn, returns the final BetaMessage
  - ToolError     : structured error content with is_error=True
  - tool_choice   : "auto" | "any" | {"type":"tool","name":"..."} to control selection

Run: python invoice_tool_agent_beta.py
"""

import json
import os

from dotenv import load_dotenv

import anthropic
from anthropic.lib.tools import ToolError, beta_tool

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ── Mock data ──────────────────────────────────────────────────────────────────

CUSTOMER_DB = {
    "CUST-001": {"name": "Rahul Verma",  "credit_score": 742, "risk_band": "low",
                 "existing_emi": 8_500,  "dpd": 0,  "monthly_income_inr": 75_000},
    "CUST-002": {"name": "Sunita Rao",   "credit_score": 541, "risk_band": "high",
                 "existing_emi": 32_000, "dpd": 15, "monthly_income_inr": 50_000},
    "CUST-003": {"name": "Arun Mehta",   "credit_score": 680, "risk_band": "medium",
                 "existing_emi": 0,      "dpd": 0,  "monthly_income_inr": 100_000},
}

ELIGIBILITY_RULES = {
    "home_loan":     {"min_score": 680, "max_dti": 0.40},
    "personal_loan": {"min_score": 720, "max_dti": 0.35},
    "business_loan": {"min_score": 700, "max_dti": 0.40},
    "vehicle_loan":  {"min_score": 680, "max_dti": 0.45},
}

# Simulates data returned by a third-party property valuation API
PROPERTY_VALUATIONS = {
    "12 MG Road, Bengaluru":      {"market_value_inr": 8_500_000, "ltv_ceiling": 0.80, "area_sqft": 1200},
    "45 Anna Nagar, Chennai":     {"market_value_inr": 5_200_000, "ltv_ceiling": 0.75, "area_sqft": 950},
    "7 Banjara Hills, Hyderabad": {"market_value_inr": 12_000_000, "ltv_ceiling": 0.80, "area_sqft": 2100},
}


# ── @beta_tool definitions ─────────────────────────────────────────────────────
# Schema is inferred automatically from type hints + Google-style docstring.
# No separate tools=[{...}] dict needed.

@beta_tool
def check_loan_eligibility(customer_id: str, loan_type: str, loan_amount_inr: float) -> str:
    """Check whether a customer is eligible for a loan.

    Returns credit score, DTI ratio, and an eligibility verdict.

    Args:
        customer_id: Apex Bank customer ID, e.g. CUST-001
        loan_type: One of: home_loan, personal_loan, business_loan, vehicle_loan
        loan_amount_inr: Requested loan amount in INR
    """
    customer = CUSTOMER_DB.get(customer_id)
    if not customer:
        # ToolError sets is_error=True in the tool_result — loop continues gracefully
        raise ToolError(f"Customer {customer_id} not found in Apex Bank records.")

    rules = ELIGIBILITY_RULES.get(loan_type)
    if not rules:
        raise ToolError(f"Unknown loan type: {loan_type}. "
                        f"Valid types: {', '.join(ELIGIBILITY_RULES)}")

    estimated_new_emi = loan_amount_inr * 0.008
    dti = (customer["existing_emi"] + estimated_new_emi) / customer["monthly_income_inr"]

    eligible = (
        customer["credit_score"] >= rules["min_score"]
        and dti <= rules["max_dti"]
        and customer["dpd"] == 0
    )

    return json.dumps({
        "customer_id":   customer_id,
        "customer_name": customer["name"],
        "credit_score":  customer["credit_score"],
        "risk_band":     customer["risk_band"],
        "estimated_dti": round(dti, 3),
        "eligible":      eligible,
        "reason": (
            "All checks passed." if eligible
            else (f"Failed — score {customer['credit_score']} "
                  f"(min {rules['min_score']}), DTI {dti:.1%} "
                  f"(max {rules['max_dti']:.0%}), DPD {customer['dpd']}")
        ),
    })


@beta_tool
def get_credit_bureau_report(customer_id: str) -> str:
    """Fetch a CIBIL credit bureau report for a customer.

    Returns credit score, risk band, and days-past-due (DPD).

    Args:
        customer_id: Apex Bank customer ID, e.g. CUST-001
    """
    customer = CUSTOMER_DB.get(customer_id)
    if not customer:
        raise ToolError(f"Customer {customer_id} not found.")

    return json.dumps({
        "customer_id":  customer_id,
        "bureau":       "CIBIL",
        "credit_score": customer["credit_score"],
        "risk_band":    customer["risk_band"],
        "dpd":          customer["dpd"],
        "report_date":  "2026-06-27",
    })


@beta_tool
def fetch_property_valuation(property_address: str, area_sqft: float) -> str:
    """Fetch property market valuation from the external Property Valuation API.

    Simulates a third-party API call to retrieve estimated market value,
    maximum LTV ceiling, and collateral suitability for home loan assessment.

    Args:
        property_address: Full street address of the property, e.g. '12 MG Road, Bengaluru'
        area_sqft: Built-up area of the property in square feet
    """
    # Fuzzy-match on address prefix to tolerate minor variations
    match = next(
        (v for k, v in PROPERTY_VALUATIONS.items()
         if k.lower() in property_address.lower() or property_address.lower() in k.lower()),
        None,
    )

    if match is None:
        # Unknown address — return a generic estimate so the agent can still advise
        estimated_value = int(area_sqft * 6_500)  # ₹6,500 per sqft fallback
        return json.dumps({
            "property_address": property_address,
            "market_value_inr": estimated_value,
            "ltv_ceiling":      0.75,
            "source":           "generic_estimate",
            "note":             "Address not found in valuation database — estimate used.",
        })

    return json.dumps({
        "property_address": property_address,
        "market_value_inr": match["market_value_inr"],
        "ltv_ceiling":      match["ltv_ceiling"],
        "max_loan_inr":     int(match["market_value_inr"] * match["ltv_ceiling"]),
        "source":           "PropValueAPI_v2",
    })


# ── Beta agentic loop ──────────────────────────────────────────────────────────

ALL_TOOLS = [check_loan_eligibility, get_credit_bureau_report, fetch_property_valuation]


def run_agent(user_query: str, tool_choice: dict | None = None) -> str:
    """Run the agentic loop using the SDK beta tool runner.

    Contrast with invoice_tool_agent.py:
      - No while True loop
      - No messages.append() calls
      - No manual tool dispatch
      The runner handles all of that; .until_done() returns the final message.

    Args:
        user_query: The user's question or request
        tool_choice: Optional tool selection control dict, e.g.
                     {"type": "auto"}                         — Claude decides (default)
                     {"type": "any"}                          — must use at least one tool
                     {"type": "tool", "name": "tool_name"}   — force a specific tool
    """
    kwargs: dict = {}
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=1024,
        tools=ALL_TOOLS,
        messages=[{"role": "user", "content": user_query}],
        #**kwargs,
    )

    # Iterate to log tool calls; until_done() alone would work if logging not needed
    for message in runner:
        for block in message.content:
            if block.type == "tool_use":
                print(f"  → [{block.name}] input: {block.input}")

    final = runner.until_done()
    return next((b.text for b in final.content if b.type == "text"), "")


# ── Scenarios ─────────────────────────────────────────────────────────────────

def main() -> None:
    scenarios = [
        {
            "label":       "1 — auto: eligibility check (Claude picks the right tool)",
            "query":       "Is CUST-001 eligible for a home loan of INR 45 lakhs?",
            "tool_choice": {"type": "auto"},
        },
        {
            "label":       "2 — auto: credit report lookup",
            "query":       "Pull the CIBIL report for CUST-003 and tell me their risk band.",
            "tool_choice": {"type": "auto"},
        },
        {
            "label":       "3 — any: force at least one tool (even for a general question)",
            "query":       "Give me a summary of CUST-002's financial standing.",
            "tool_choice": {"type": "any"},
        },
        {
            "label":       "4 — tool: force fetch_property_valuation specifically",
            "query":       "What is the market value of the property at 45 Anna Nagar, Chennai (950 sqft)?",
            "tool_choice": {"type": "tool", "name": "fetch_property_valuation"},
        },
        {
            "label":       "5 — ToolError: unknown customer triggers is_error=True",
            "query":       "Check eligibility for CUST-999 for a business loan of INR 10 lakhs.",
            "tool_choice": {"type": "auto"},
        },
    ]

    for s in scenarios:
        print(f"\n{'='*62}")
        print(f"Scenario {s['label']}")
        print(f"Query   : {s['query']}")
        print(f"Choice  : {s['tool_choice']}")
        print("-" * 62)
        answer = run_agent(s["query"], tool_choice=s["tool_choice"])
        print(f"\n[Claude] {answer}")


if __name__ == "__main__":
    main()
