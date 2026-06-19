# API Parameters Demo

This script demonstrates how generation parameters affect a Gemini model's output. Each scenario runs the same prompt with a different parameter combination so you can see the effect directly in the log.

> **Provider:** Gemini only. This demo uses parameter combinations (e.g. `temperature=2.0` combined with `top_p`) that Claude does not support, so it is kept Gemini-only by design.

---

## What It Does

```
PROMPT (fixed)
      │
      ▼
Scenario 1: temperature=0.2      → deterministic, focused output
Scenario 2: temperature=2.0      → chaotic, varied output
Scenario 3: top_p=0.1            → narrow vocabulary
Scenario 4: top_p=0.95           → wide vocabulary
Scenario 5: max_tokens=40        → hard-cut output
Scenario 6: top_p=0.95 + temp=2.0 → combined randomness
      │
      ▼
logs/demo_<timestamp>.log
```

---

## Setup

### 1 — Install dependencies

```bash
pip install google-genai
```

### 2 — Create a `.env` file

Create a file named `.env` in this folder (or in the project root):

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Then load it in your terminal before running:

```bash
source .env && export $(cut -d= -f1 .env)
```

Or export it directly:

```bash
export GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Run

```bash
python src/API_parameters/api_config_demo.py
```

Logs are written to `src/API_parameters/logs/demo_<timestamp>.log` and also printed to the terminal.

---

## Scenarios

| Scenario | Parameter | Value | What to observe |
|----------|-----------|-------|-----------------|
| Low Temperature | `temperature` | `0.2` | Consistent, predictable word choice |
| High Temperature | `temperature` | `2.0` | Surprising or incoherent word choice |
| Narrow nucleus | `top_p` | `0.1` | Conservative, repetitive phrasing |
| Wide nucleus | `top_p` | `0.95` | Richer, more varied phrasing |
| Short output | `max_tokens` | `40` | Response cut off mid-sentence |
| Combined | `top_p=0.95` + `temperature=2.0` | — | Maximum randomness |

---

## Customising the Prompt

Edit the `PROMPT` constant near the top of `api_config_demo.py`:

```python
PROMPT = """
Your new prompt here.
"""
```

To add a new scenario, append a dict to the `SCENARIOS` list:

```python
{
    "name": "My Scenario",
    "description": "What this parameter does and what to look for.",
    "params": dict(temperature=0.5, max_tokens=200),
}
```

Valid keys for `params`: `temperature`, `top_p`, `max_tokens`.

---

## Function Reference

| Function | What it does |
|----------|-------------|
| `build_client()` | Reads `GEMINI_API_KEY` from the environment and returns a `genai.Client` |
| `call_llm(client, prompt, *, temperature, max_tokens, top_p, system_instruction)` | Sends one request to Gemini with the given generation config and returns the response text |
| `run_scenario(client, scenario)` | Logs the scenario name, description, params, and the model output |
| `main()` | Builds the client and runs every scenario in `SCENARIOS` |
