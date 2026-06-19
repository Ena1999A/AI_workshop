# Prompt Engineering Demo

This script runs a structured experiment to show how system prompt design affects model output. It walks through prompt engineering stages in order — from a plain instruction all the way to a polished golden prompt — and tests each stage against a set of user messages.

> **Providers supported:** Gemini and Claude. Switch between them with the `LLM_PROVIDER` environment variable.

---

## What It Does

```
system_prompts/
  01_simple/
  02_role/
  03_constraint/
  04_structured_output/
  05_few_shot/
  06_golden_prompt/
        │
        ▼  (for each stage)
user_prompts/leasing_*.txt   or   user_prompts/intent_*.txt
        │
        ▼  (for each user prompt × each stage)
call_llm(system_prompt, user_prompt)
        │
        ▼
logs/demo_<timestamp>.log
```

Each stage folder can hold multiple system prompt files. Files are picked by prefix (`leasing_` or `intent_`), so one run covers one domain at a time.

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
python src/prompt_engineering/main.py
```

**With Claude:**

```bash
LLM_PROVIDER=claude python src/prompt_engineering/main.py
```

**With a specific Claude model:**

```bash
LLM_PROVIDER=claude CLAUDE_MODEL=claude-sonnet-4-6 python src/prompt_engineering/main.py
```

Logs are written to `src/prompt_engineering/logs/demo_<timestamp>.log` and printed to the terminal.

---

## Folder Structure

```
prompt_engineering/
├── main.py
├── system_prompts/
│   ├── 01_simple/          ← Bare instruction prompt
│   ├── 02_role/            ← Role-based prompt
│   ├── 03_constraint/      ← Prompt with hard constraints
│   ├── 04_structured_output/ ← Prompt asking for JSON or structured text
│   ├── 05_few_shot/        ← Prompt with example input/output pairs
│   └── 06_golden_prompt/   ← Final polished prompt
└── user_prompts/
    ├── leasing_users.txt   ← One user message per line (leasing domain)
    └── intent_users.txt    ← One user message per line (intent domain)
```

### Adding system prompts

Place `.txt` files inside the relevant stage folder. Name them with the domain prefix you want to run:

```
system_prompts/03_constraint/leasing_strict.txt   ← picked by run_domain("leasing")
system_prompts/03_constraint/intent_strict.txt    ← picked by run_domain("intent")
```

### Adding user prompts

Add one message per line to the matching file in `user_prompts/`:

```
user_prompts/leasing_users.txt
```

Empty lines are skipped automatically.

### Switching the active domain

In `main.py`, change the call at the bottom of `main()`:

```python
run_domain(client, "leasing")   # run leasing domain
# run_domain(client, "intent")  # run intent domain
```

---

## Function Reference

| Function | What it does |
|----------|-------------|
| `load_system_prompts(prefix)` | Scans all stage subfolders alphabetically and returns `[(stage_folder, file_stem, content)]` for files matching the prefix |
| `load_user_prompts(prefix)` | Reads all non-empty lines from `user_prompts/<prefix>*.txt` files; each line is one user message |
| `call_llm(client, system_prompt, user_prompt)` | Sends one request to Gemini or Claude (based on `LLM_PROVIDER`) and returns the response text |
| `run_domain(client, prefix)` | Runs the full experiment for one domain: every stage × every user prompt, logs all outputs |
| `main()` | Builds the API client for the active provider and calls `run_domain` |
