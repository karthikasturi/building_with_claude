# Building with Claude — Training Content

> A 19-hour, application-engineering program for technical participants (software/application
> developers, data engineers, solution architects) who build on the **Claude API** and **Python SDK**.
> Hands-on labs and case studies are drawn from **finance, retail, and telecom** contexts.

This repository contains the **full reference learning material** for the program: curated decks,
hands-on labs, runnable sample code, evaluation rubrics, and reading lists — one self-contained
folder per module, plus a shared environment used by every lab.

> ⚠️ **All sample code targets the current, stable Claude API surface** (see
> [Version & currency policy](#version--currency-policy)). Model IDs, the thinking parameter,
> structured-output, and tool-use shapes change over time — this material was written against the
> June 2026 API and notes where older patterns are now deprecated.

---

## Who this is for

| | |
|---|---|
| **Audience** | Technical participants only — developers, data engineers, solution architects building with the Claude API. |
| **Prerequisites** | *Working with Claude* (or equivalent awareness of Claude); hands-on Python; familiarity with REST APIs, JSON, and basic application workflows. |
| **Format** | Instructor-led live connects + self-paced reference material (this repo). Experiential, mentor-supported. |
| **Total duration** | 19 hours across 9 modules. |

## Program outcomes

By the end of the program participants can:

1. Use the Claude API and Python SDK **securely**, with appropriate key and error handling.
2. Design **reliable prompts** and produce **validated structured (JSON) outputs**.
3. Implement **tool use** and function-style integration with business systems.
4. Build **retrieval-grounded (RAG)** responses over enterprise documents.
5. **Evaluate** outputs for accuracy, relevance, and safety.

---

## Module map

| # | Module | Hrs | Case study | Folder |
|---|--------|-----|------------|--------|
| 1 | API Setup and Secure Integration | 2.0 | Secure, env-managed Claude call | [`module_1_api_setup/`](module_1_api_setup/) |
| 2 | Prompt Engineering for Applications | 2.0 | Finance credit-policy explainer | [`module_2_prompt_engineering/`](module_2_prompt_engineering/) |
| 3 | Structured Outputs and Validation | 2.5 | Retail product enrichment | [`module_3_structured_outputs/`](module_3_structured_outputs/) |
| 4 | Conversation and Context Management | 2.0 | Telecom complaint summariser | [`module_4_conversation_context/`](module_4_conversation_context/) |
| 5 | Tool Use and Function Integration | 2.5 | Invoice validation + vendor lookup | [`module_5_tool_use/`](module_5_tool_use/) |
| 6 | Retrieval-Grounded Responses (RAG) | 3.0 | Finance SOP assistant | [`module_6_rag/`](module_6_rag/) |
| 7 | Evaluation and Output Quality | 2.0 | Evaluate the RAG assistant | [`module_7_evaluation/`](module_7_evaluation/) |
| 8 | Applied Mini-Project | 2.0 | Telecom support triage assistant | [`module_8_mini_project/`](module_8_mini_project/) |
| 9 | Exit Test | 1.0 | Scenario assessment | [`module_9_exit_test/`](module_9_exit_test/) |

Each module folder follows the same layout:

```
module_N_<name>/
├── README.md          # Overview, learning outcomes, agenda & timing, key pitfalls
├── slides.md          # Curated deck (Markdown slides, '---' separated)
├── lab.md             # Step-by-step hands-on lab + case study
├── rubric.md          # Evaluation rubric for the lab/skill
├── reading_list.md    # Curated, current reading list
└── code/              # Runnable sample code referenced by the lab
```

---

## Getting started

See **[SETUP.md](SETUP.md)** for the complete setup walkthrough: cloning the repo, creating a virtual environment (macOS / Linux / Windows), installing dependencies, configuring API keys, and running each day's labs in order.

---

## Version & currency policy

This material is written against the **stable Claude API as of June 2026** and the official
`anthropic` Python SDK. The conventions used throughout:

| Topic | Convention used in this course | Notes |
|---|---|---|
| Default model | `claude-opus-4-8` | 1M context. Swap to `claude-haiku-4-5` only for simple, high-volume, latency-sensitive steps. |
| Extended thinking | `thinking={"type": "adaptive"}` | The old `budget_tokens` form is **deprecated on 4.6 and rejected (400) on 4.7 / 4.8**. |
| Reasoning depth | `output_config={"effort": "..."}` | `low | medium | high | max`. |
| Structured output | `client.messages.parse(..., output_format=PydanticModel)` | Validated automatically. Raw form: `output_config={"format": {"type": "json_schema", ...}}`. The bare top-level `output_format=<dict>` on `create()` is deprecated. |
| Streaming | `client.messages.stream(...)` + `.get_final_message()` | Default for long input/output or high `max_tokens`. |
| Tool use | `@beta_tool` + `client.beta.messages.tool_runner(...)`, or a manual agentic loop | |
| Embeddings (RAG) | OpenAI (`openai`) | The Claude API has **no** embeddings endpoint; we use OpenAI's `text-embedding-3-small` model. |

> **Always verify against the live docs before teaching.** The single source of truth is
> [platform.claude.com/docs](https://platform.claude.com/docs). When in doubt, call
> `GET /v1/models` / `client.models.list()` for the live model catalog and capabilities.

## Security ground rules (apply to every module)

1. **Never** hardcode API keys or commit them. Load from the environment (`ANTHROPIC_API_KEY`).
2. Keep `.env` out of version control (a `.gitignore` is provided).
3. Treat model output as **untrusted input** — validate before it touches downstream systems.
4. Never paste secrets, PII, or regulated data into prompts you don't control the retention of.
5. Handle errors with typed exceptions and backoff; fail closed, log the `request_id`.
