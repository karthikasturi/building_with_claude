# Claude API — Patterns Reference Card
### Building with Claude

---

## Pattern 1 — System Prompt Structure (S1)

Every well-formed system prompt has these five sections in this order:

```
ROLE          → who Claude is and what it is responsible for
CATEGORIES    → what values are allowed, with descriptions
RULES         → constraints, one per bullet, unambiguous
FALLBACK      → what to do when the input is ambiguous or out of scope
OUTPUT FORMAT → the exact JSON structure, with field names
```

Worked skeleton:

```python
system = """
You are a [ROLE] for [COMPANY].

CATEGORIES:
- category_a: description
- category_b: description

RULES:
- One rule per bullet.
- Cite source documents: [Policy Section X.Y]
- Never state a decision before all required fields are collected.

FALLBACK: If the input is ambiguous, respond:
{"status": "needs_clarification", "question": "..."}

OUTPUT: Return ONLY this JSON — no prose, no markdown fences:
{"category": "...", "priority": "...", "summary": "..."}
"""
```

---

## Pattern 2 — Structured Output with Retry (S2)

```python
from pydantic import BaseModel, field_validator
from typing import Literal

class MyRecord(BaseModel):
    field_a: str
    field_b: Literal["value1", "value2", "value3"]
    field_c: bool

    @classmethod                    # ← @classmethod MUST come first
    @field_validator("field_a")
    def validate_field_a(cls, v: str) -> str:
        if not v.startswith("PREFIX-"):
            raise ValueError("Must start with PREFIX-")
        return v


def extract(text: str) -> MyRecord:
    messages = [{"role": "user", "content": text}]
    for attempt in range(3):
        try:
            resp = client.messages.parse(   # ← .parse() not .create()
                model=MODEL, max_tokens=512,
                messages=messages,
                output_format=MyRecord,
            )
            return resp.parsed_output       # ← .parsed_output not .content[0].text
        except ValidationError as e:
            if attempt == 2:
                raise
            messages.append({"role": "assistant", "content": resp.content[0].text})
            messages.append({"role": "user",
                             "content": f"Validation failed:\n{e}\nReturn corrected JSON."})
```

**Three things that MUST be right:**
1. `@classmethod` above `@field_validator`
2. `messages.parse()` not `messages.create()`
3. Retry appends BOTH the bad response AND the error

---

## Pattern 3 — Tool Definition + Agentic Loop (S3)

```python
tools = [
    {
        "name": "tool_name",
        "description": "What this tool does and when Claude should call it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
                "param2": {"type": "integer", "description": "..."},
            },
            "required": ["param1", "param2"],
            "additionalProperties": False,
        },
    }
]

def run_agent(query: str) -> str:
    messages = [{"role": "user", "content": query}]
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1024,
            tools=tools, messages=messages,
        )
        if response.stop_reason == "end_turn":  # ← NOT "tool_use"
            break
        messages.append({                        # ← append BEFORE processing tools
            "role": "assistant",
            "content": response.content,         # ← full LIST, not [0].text
        })
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),   # ← string, not dict
                    "is_error": "error" in result,
                })
        messages.append({"role": "user", "content": tool_results})  # ← list
    return next(b.text for b in response.content if b.type == "text")
```

**Four gaps that appear in every broken loop:**

| Gap | Wrong | Right |
|-----|-------|-------|
| Break condition | `"tool_use"` | `"end_turn"` |
| Assistant content | `response.content[0].text` | `response.content` |
| Tool result content | `result` (dict) | `json.dumps(result)` |
| Tool result wrapper | appended directly | inside a list |

---

## Pattern 4 — RAG + Grounding + Faithfulness Judge (S4)

```python
# Chunking strategy (always answer this first)
# chunk_size = 400 words, overlap = 50 words
# metadata per chunk: {"source": "filename", "section": "3.1", "chunk_index": 0}

# Grounding system prompt template
RAG_SYSTEM = """
You are a [DOMAIN] assistant.
Answer questions using ONLY the context provided below.

Rules:
- Cite every fact: [Source: section_title, chunk N]
- If the answer is not in the context, respond exactly:
  "This is not covered in the provided documents."
- Do not use your training knowledge to fill gaps.
"""

# Faithfulness judge (Claude-as-judge)
JUDGE_SYSTEM = """Score whether every claim in the answer is supported by the context.
5=fully grounded, 4=mostly, 3=mixed, 2=mostly unsupported, 1=hallucination.
Return JSON only: {"score": <1-5>, "reasoning": "..."}"""

def judge_faithfulness(context: str, answer: str) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=256, system=JUDGE_SYSTEM,
        messages=[{"role": "user",
                   "content": f"Context:\n{context}\n\nAnswer:\n{answer}"}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    # Strip markdown fences — Claude wraps JSON even when told not to
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text) \
        or re.search(r"(\{[\s\S]*\})", text)
    try:
        return json.loads(m.group(1) if m else text)
    except json.JSONDecodeError:
        return {"score": 0, "reasoning": f"parse error: {text[:80]}"}
```

**Two details that lose marks in S4:**
1. Forgetting fence-stripping → `json.JSONDecodeError` on every judge call
2. Grounding prompt without an explicit "I don't know" fallback → Claude hallucinates

---

## Pattern 5 — Code Critique Checklist (S5)

When reading a broken program, scan in this order:

```
□ 1. Break condition         stop_reason == "end_turn"   (NOT "tool_use")
□ 2. Assistant content       response.content            (NOT [0].text — drops tool blocks)
□ 3. Tool result wrapper     {"role":"user","content":[tool_results]}  (NOT appended raw)
□ 4. Tool result content     json.dumps(result)          (NOT a dict)
□ 5. Pydantic validator      @classmethod above @field_validator
□ 6. Parse method            messages.parse()            (NOT messages.create() + output_format)
□ 7. Return value            resp.parsed_output          (NOT json.loads(resp.content[0].text))
□ 8. Security                No API key printed/logged anywhere
```

---

## Quick Reference — API Conventions

| Topic | Convention |
|-------|-----------|
| Structured output | `client.messages.parse(output_format=Model)` → `.parsed_output` |
| Agentic loop break | `stop_reason == "end_turn"` — never `"tool_use"` |
| Append assistant turn | `{"role": "assistant", "content": response.content}` — full list |
| Tool result content | `json.dumps(result)` — string, not dict |
| Tool error | `{"type": "tool_result", ..., "is_error": True}` |
| Fence stripping | `re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)` |
| Token counting | `client.messages.count_tokens(model, system, messages).input_tokens` |
| Conversation memory | After ~40K tokens: summarise and reset message history |
