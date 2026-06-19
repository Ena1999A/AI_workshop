# LLM Chaining Demo

This script demonstrates intent-based LLM chaining: a first model classifies the user's intent, a Python router picks the right specialist, and a second model handles the request with a domain-specific system prompt.

> **Providers supported:** Gemini and Claude. Switch between them with the `LLM_PROVIDER` environment variable.

---

## What It Does

```
User message
      │
      ▼
LLM 1 — Intent Classifier
  system_prompts/intent_classifier.txt
  Returns JSON: { "intent": "...", "confidence": 0.9, "reasoning": "..." }
      │
      ▼
Python Router
  Reads intent, picks the matching system prompt file
      │
      ├─ update_contact_info      → LLM 2a
      │                              Returns JSON payload + user confirmation
      │
      ├─ repair_status_question   → LLM 2b
      │                              Returns a plain answer
      │
      └─ leasing_policy_question  → LLM 2c
                                     Returns a plain answer
      │
      ▼
logs/demo_<timestamp>.log
```

All LLM calls use `temperature=0.2` to keep output consistent and JSON-parseable.

---

## Setup

### 1 — Install dependencies

```bash
# For Gemini
pip install google-genai

# For Claude
pip install anthropic
```

### 2 — Create a `.env` file

Create a file named `.env` in this folder (or in the project root):

```
# Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Claude (only needed if using LLM_PROVIDER=claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Load it before running:

```bash
source .env && export $(cut -d= -f1 .env)
```

Or export the key you need directly:

```bash
export GEMINI_API_KEY=your_key_here
# or
export ANTHROPIC_API_KEY=your_key_here
```

---

## Run

**With Gemini (default):**

```bash
python src/llm_chaining/main.py
```

**With Claude:**

```bash
LLM_PROVIDER=claude python src/llm_chaining/main.py
```

**With a specific Claude model:**

```bash
LLM_PROVIDER=claude CLAUDE_MODEL=claude-sonnet-4-6 python src/llm_chaining/main.py
```

Logs are written to `src/llm_chaining/logs/demo_<timestamp>.log` and printed to the terminal.

---

## Folder Structure

```
llm_chaining/
├── main.py
├── system_prompts/
│   ├── intent_classifier.txt           ← LLM 1: classifies intent into JSON
│   ├── update_contact_info.txt         ← LLM 2a: produces structured DB payload
│   ├── repair_status_question_rag.txt  ← LLM 2b: answers repair status questions
│   └── leasing_policy_question.txt     ← LLM 2c: answers leasing policy questions
└── user_prompts/
    └── examples.txt                    ← One user message per line
```

### Adding user messages

Add one message per line to `user_prompts/examples.txt`. Empty lines are skipped:

```
I want to change my phone number to 091 555 1234.
What is the status of my repair ticket #4892?
Can I extend my leasing contract for another year?
```

### Supported intents

| Intent | Specialist handler | Output format |
|--------|--------------------|---------------|
| `update_contact_info` | `handle_update_contact_info` | JSON with `db_payload` and `user_message` fields |
| `repair_status_question` | `handle_plain_answer` | Plain text answer |
| `leasing_policy_question` | `handle_plain_answer` | Plain text answer |

Any other intent is logged as unknown and skipped.

---

## Function Reference

| Function | What it does |
|----------|-------------|
| `load_text(path)` | Reads and strips a `.txt` file; used to load system prompts |
| `load_user_prompts()` | Reads all non-empty lines from every `.txt` file in `user_prompts/`; each line is one message |
| `call_llm(client, system_prompt, user_message)` | Sends one request to Gemini or Claude (based on `LLM_PROVIDER`) with `temperature=0.2` and returns the response text |
| `classify_intent(client, user_message)` | Calls LLM 1, parses the JSON response, and returns `(intent, confidence, reasoning)` |
| `route(intent)` | Returns the system prompt `Path` for the given intent, or `None` if unknown |
| `handle_update_contact_info(client, user_message)` | Calls LLM 2a, parses the JSON response, and returns `{"db_payload": ..., "user_answer": ...}` |
| `handle_plain_answer(client, intent, user_message)` | Calls LLM 2b or 2c and returns the plain text answer |
| `run_pipeline(client, user_message)` | Runs the full classify → route → handle pipeline for one message and logs all steps |
| `main()` | Builds the API client, loads user prompts, and runs `run_pipeline` for each |
