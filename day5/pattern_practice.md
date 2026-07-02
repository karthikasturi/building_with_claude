# Pattern Practice — Self-Guided Exercises
### Building with Claude

Work through these exercises at your own pace before the mini-project and exit assessment.
Each exercise introduces a new domain so you practise the pattern, not the specific answer.
Try each question yourself first, then scroll to the solution.

---

## Exercise 1 — System Prompt Design

### Scenario

FreshStay Hotels receives guest complaint emails daily. Build a triage assistant that
classifies each email and sets a response priority.

**Categories:** `room_issue` | `service_issue` | `billing_dispute` | `compliment` | `other`

**Priority levels:**
- `urgent` — health or safety risk (pest sighting, broken lock, fire alarm)
- `high` — guest is checking in or out today
- `medium` — active stay issue, can be resolved within hours
- `low` — post-stay feedback or general enquiry

**Your task:** Write a system prompt that includes role, all four priority levels with
criteria, an explicit fallback for ambiguous emails, and the exact JSON output format.

<details>
<summary>Show solution</summary>

```python
system = """
You are a guest experience triage assistant for FreshStay Hotels.

ROLE: Classify incoming guest emails and set a response priority.

CATEGORIES:
- room_issue:       physical room problems (cleanliness, AC, Wi-Fi, plumbing)
- service_issue:    staff behaviour, housekeeping delays, concierge
- billing_dispute:  incorrect charges, refund requests
- compliment:       positive feedback
- other:            anything that does not fit the categories above

PRIORITY LEVELS:
- urgent:  Health or safety risk (pest sighting, broken lock, fire alarm).
           Escalate immediately — do not wait for the next shift.
- high:    Guest is checking in or out today. Same-day resolution required.
- medium:  Active stay issue that can be resolved within 4 hours.
- low:     Post-stay feedback or a general enquiry.

FALLBACK: If the email is too ambiguous to classify confidently, respond:
  {"category": "other", "priority": "medium", "note": "Requires human review."}

OUTPUT: Respond with ONLY this JSON — no prose, no markdown fences:
{"category": "...", "priority": "...", "summary": "...", "next_action": "..."}
"""
```

**Pattern to remember:**
Role → Categories with descriptions → Priority levels with criteria → Fallback → Output format.
This structure works for any triage system prompt regardless of domain.

</details>

---

## Exercise 2 — Structured Output + Validation

### Scenario

QuickCart processes free-text return requests and needs to extract a structured record
from each one.

**Fields:**
- `order_id` — string, must start with `"ORD-"`
- `return_reason` — one of: `"defective"` | `"wrong_item"` | `"changed_mind"` | `"not_as_described"`
- `refund_method` — one of: `"original_payment"` | `"store_credit"`
- `item_condition` — one of: `"unopened"` | `"opened"` | `"damaged"`
- `eligible` — bool: `True` if reason is NOT `"changed_mind"` AND condition is NOT `"damaged"`

**Your task:** Write the Pydantic model with a field validator, then implement an extraction
function that uses `messages.parse()` with up to 2 retries on `ValidationError`.

<details>
<summary>Show solution</summary>

```python
from pydantic import BaseModel, field_validator
from typing import Literal

class ReturnRequest(BaseModel):
    order_id:       str
    return_reason:  Literal["defective", "wrong_item", "changed_mind", "not_as_described"]
    refund_method:  Literal["original_payment", "store_credit"]
    item_condition: Literal["unopened", "opened", "damaged"]
    eligible:       bool

    @classmethod                        # ← @classmethod MUST be above @field_validator
    @field_validator("order_id")
    def validate_order_id(cls, v: str) -> str:
        if not v.startswith("ORD-"):
            raise ValueError("order_id must start with ORD-")
        return v


def extract_return(text: str) -> ReturnRequest:
    messages = [{"role": "user", "content": text}]
    for attempt in range(3):
        try:
            resp = client.messages.parse(
                model=MODEL, max_tokens=256,
                messages=messages,
                output_format=ReturnRequest,
            )
            return resp.parsed_output
        except ValidationError as e:
            if attempt == 2:
                raise
            # Append both the bad response and the error before retrying
            messages.append({"role": "assistant", "content": resp.content[0].text})
            messages.append({"role": "user",
                             "content": f"Validation failed:\n{e}\nReturn corrected JSON."})
```

**Three things that must be right every time:**
1. `@classmethod` above `@field_validator` — reversing them causes a `PydanticUserError` at import
2. `messages.parse()` not `messages.create()` — only `.parse()` validates against the model
3. Retry appends both the bad response AND the error message before the next call

</details>

---

## Exercise 3 — Tool Definitions + Agentic Loop

### Scenario

FreshMart supermarket needs an inventory agent with two tools:
- `check_stock(product_id: str)` → `{"stock_level": int, "reorder_point": int}`
- `reorder_product(product_id: str, quantity: int)` → `{"order_id": str, "estimated_arrival": str}`

**Part A:** Write both tool definitions (the dicts passed to `tools=[...]`).

**Part B:** Four gaps have been introduced into the agentic loop below. Fill them in.

```python
def run_inventory_agent(query: str) -> str:
    messages = [{"role": "user", "content": query}]
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1024,
            tools=tools, messages=messages,
        )
        if response.stop_reason == ___:          # gap 1
            break
        messages.append({
            "role": "assistant",
            "content": ___,                      # gap 2
        })
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": ___,              # gap 3
                })
        messages.append({"role": "user", "content": ___})   # gap 4
    return next(b.text for b in response.content if b.type == "text")
```

<details>
<summary>Show solution</summary>

**Part A — Tool definitions:**

```python
tools = [
    {
        "name": "check_stock",
        "description": "Check current stock level for a product. Call this before deciding to reorder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "FreshMart product ID, e.g. PROD-001"},
            },
            "required": ["product_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "reorder_product",
        "description": "Place a reorder for a product when stock is at or below reorder_point.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "FreshMart product ID"},
                "quantity":   {"type": "integer", "description": "Units to reorder"},
            },
            "required": ["product_id", "quantity"],
            "additionalProperties": False,
        },
    },
]
```

**Part B — Gap answers:**

```
Gap 1: "end_turn"          ← NOT "tool_use" — breaking on "tool_use" exits before tools run
Gap 2: response.content    ← the full content LIST, not response.content[0].text
Gap 3: json.dumps(result)  ← must be a string; passing a dict causes a BadRequestError
Gap 4: tool_results        ← the list of tool_result dicts, not a single dict
```

**Why each mistake is silent and dangerous:**
- Wrong break condition → loop exits immediately, tools never called, no error raised
- Wrong content type → next API call fails with a cryptic missing-block error
- Dict instead of string → API rejects the message with `BadRequestError`

</details>

---

## Exercise 4 — RAG + Faithfulness Evaluation

### Scenario

InfoCorp employees ask questions about company HR policies (leave, travel reimbursement,
performance reviews). The assistant must answer only from the policy documents and cite
its sources. Unknown topics must get a polite "not covered" response.

**Part A:** Design a chunking strategy. What chunk size, overlap, and metadata would you use?

**Part B:** Write the grounding system prompt.

**Part C:** Write a `judge_faithfulness(context, answer)` function that returns
`{"score": 1–5, "reasoning": "..."}`. Score 5 = fully grounded, 1 = hallucinated.

<details>
<summary>Show solution</summary>

**Part A — Chunking strategy:**
- Chunk size: ~400 words
- Overlap: ~50 words (preserves context across chunk boundaries)
- Metadata per chunk: `{"source": "leave_policy.md", "section": "3.1", "chunk_index": 0}`

**Part B — Grounding system prompt:**

```python
system = """
You are an HR policy assistant for InfoCorp.
Answer employee questions using ONLY the policy context provided below.

Rules:
- Every factual statement must cite its source: [Leave Policy, Section 3.1]
- If the answer is not in the provided context, respond with exactly:
  "This topic is not covered in the current HR policies. Please contact hr@infocorp.com."
- Do not use your general training knowledge to fill gaps.
- Be concise. Maximum 150 words per answer.
"""
```

**Part C — Faithfulness judge:**

```python
import re, json

JUDGE_SYSTEM = """Score whether every claim in the answer is supported by the context.
5=fully supported, 4=mostly, 3=mixed, 2=mostly unsupported, 1=hallucination
Return JSON only: {"score": <1-5>, "reasoning": "..."}"""

def judge_faithfulness(context: str, answer: str) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=256, system=JUDGE_SYSTEM,
        messages=[{"role": "user",
                   "content": f"Context:\n{context}\n\nAnswer:\n{answer}"}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    # Always strip fences — Claude wraps JSON even when instructed not to
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text) \
        or re.search(r"(\{[\s\S]*\})", text)
    try:
        return json.loads(m.group(1) if m else text)
    except json.JSONDecodeError:
        return {"score": 0, "reasoning": f"parse error: {text[:80]}"}
```

**Two details that are easy to miss:**
1. The fence-stripping regex is not optional — Claude wraps JSON in ` ```json ` blocks even
   when the system prompt says "return JSON only"
2. The grounding prompt needs an explicit "I don't know" fallback — without it Claude will
   attempt to answer from training knowledge

</details>

---

## Exercise 5 — Code Critique

Find every bug in both programs. For each bug: name the line, describe the problem,
and state the production impact.

### Program A

```python
def get_recommendation(user_query: str) -> str:
    messages = [{"role": "user", "content": user_query}]
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1024, tools=tools, messages=messages
        )
        if response.stop_reason == "tool_use":          # line A
            break
        messages.append({
            "role": "assistant",
            "content": response.content[0].text         # line B
        })
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                messages.append({                        # line C
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)
                })
    return next(b.text for b in response.content if b.type == "text")
```

### Program B

```python
from pydantic import BaseModel, field_validator

class ProductRecord(BaseModel):
    product_id: str
    price: float

    @field_validator("price")                # line D
    def validate_price(cls, v):
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v

def extract_product(text: str) -> ProductRecord:
    resp = client.messages.create(           # line E
        model=MODEL, max_tokens=256,
        messages=[{"role": "user", "content": text}],
        output_format=ProductRecord          # line F
    )
    import os
    api_key = os.environ["ANTHROPIC_API_KEY"]
    print(f"Used key: {api_key}")            # line G
    return ProductRecord(**json.loads(resp.content[0].text))  # line H
```

<details>
<summary>Show solution</summary>

| Line | Bug | Production impact |
|------|-----|-------------------|
| A | `"tool_use"` should be `"end_turn"` | Loop exits before any tool runs — silent failure, no exception |
| B | `.content[0].text` drops `tool_use` blocks from the message | Next API call fails — required content block is missing |
| C | `tool_result` dict appended directly to `messages`, not inside a list | `BadRequestError` — the API expects `content` to be a list |
| D | Missing `@classmethod` above `@field_validator` | `PydanticUserError` raised at class definition time — module fails to import |
| E+F | `messages.create()` does not accept `output_format` (deprecated) | Use `messages.parse(output_format=ProductRecord)` instead |
| G | API key printed to stdout | Key appears in logs, CI output, terminals — immediate security risk; rotate the key |
| H | Unnecessary if using `messages.parse()` | Use `resp.parsed_output` — no manual JSON parsing needed |

**Scan order for future critiques:** break condition → assistant content → tool result wrapper → tool result content type → validator decorator order → parse method → security leaks

</details>

---

## Mini-Project Orientation

Before writing any code, read through `day5/loan_inquiry_assistant.py` section by section
and map each block to the module that introduced it:

| Code section | Introduced in |
|---|---|
| `load_dotenv()` + `anthropic.Anthropic()` | Module 1 — API Setup |
| `INQUIRY_SYSTEM` prompt | Module 2 — Prompt Engineering |
| `LoanInquiryRecord` Pydantic model | Module 3 — Structured Outputs |
| `retrieve()` + vector store | Module 6 — RAG |
| `while True` agentic loop | Module 5 — Tool Use |
| Fence stripping + `model_validate_json()` | Module 3 + 6 |
| `judge_faithfulness()` | Module 7 — Evaluation |

Run the file first, observe the output, then make one of these targeted modifications:

```
Option A (quick)
  Add a new rule to the system prompt:
  "urgent: applicant has a legal complaint or mentions regulatory action"
  Run the file. Does any test case output change?

Option B (moderate)
  Change top_k=2 to top_k=5 in the retrieve() call.
  Run the file. Does the policy_basis field in the output get longer?

Option C (stretch)
  Add a new boolean field flag_for_human_review to LoanInquiryRecord.
  Update the OUTPUT section of the system prompt to include it.
  Run the file. Does the output contain the new field?
```
