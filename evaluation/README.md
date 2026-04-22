# LLM Evaluation Workshop

## The Big Idea

The core problem with evaluating LLMs is that you can't simply check *"right or wrong"* the way you do with classical ML.

An LLM might:

- give a correct answer in broken JSON
- hallucinate a confident-sounding fee amount
- classify intent correctly only 50% of the time because the prompt was vague

This workshop builds an **evaluation pipeline** that measures all of these behaviors systematically.

The pipeline follows a single loop:

```
dataset → LLM generates output → metrics → find failures → improve prompt
```

---

# The Two Tasks

Two tasks were chosen because they test **fundamentally different failure modes**.

## 1. Intent Classification

The output is **structured (JSON)** and **deterministic** (one correct answer exists).

Evaluation can be automated without a judge model.

Key question:

> Does the prompt produce reliable JSON and correct labels?
> 

Typical failure modes:

- invalid JSON
- incorrect intent classification
- inconsistent confidence scores
- confusion between similar intents

---

## 2. Leasing Q&A

The output is **free text**.

There is **no single correct answer**.

Primary failure mode:

> hallucination
> 

Examples:

- inventing a fee amount
- guaranteeing a replacement vehicle
- claiming "no penalty" when policy says nothing

Evaluation requires a **second LLM acting as a judge**.

---

# Project Structure

## tasks/intent_classification/dataset.json

Contains **20 labeled examples**.

Each example includes:

- `text`
- `expected_intent`
- `category`

### Categories

These categories simulate real-world user behavior:

| Category | Description |
| --- | --- |
| clear | easy cases that should always pass |
| indirect | implicit meaning (e.g. "I don't want this car anymore" → contract_termination) |
| ambiguous | unclear phrasing requiring best interpretation |
| multi_intent | multiple intents present but only primary counts |
| typos | realistic user spelling mistakes |
| edge_case | off-topic or missing context |

A strong prompt performs well across all categories.

A weak prompt performs well only on *clear* cases.

---

## tasks/intent_classification/prompt_good.txt vs prompt_bad.txt

### Bad prompt

Very short:

```
You are an assistant.
Classify the user's request.
Possible categories: [...]
Return JSON.
```

### Good prompt

Contains:

- definitions for each intent
- disambiguation rules
- structured JSON schema guidance
- 5 few-shot examples demonstrating format

Example rule:

> If unsure, prefer `information_request`
> 

The improved prompt significantly increases both accuracy and format reliability.

---

## tasks/leasing_qa/dataset.json

Contains **15 questions** designed to expose hallucination behavior.

### Important categories

| Category | Purpose |
| --- | --- |
| adversarial | encourages the model to hallucinate |
| prompt_injection | tests instruction robustness |
| ambiguous | requires clarification instead of guessing |
| hallucination_risk | flags questions likely to produce invented facts |

---

## tasks/leasing_qa/prompt_good.txt vs prompt_bad.txt

### Bad prompt

```
Be friendly and helpful.
```

### Good prompt

Includes the full company policy document and explicit constraints:

- do not quote specific monetary amounts
- do not make promises or guarantees
- do not invent benefits not mentioned in policy

Providing grounding context significantly reduces hallucination risk.

---

# Evaluation Components

## evaluation/metrics.py

Pure functions that compute evaluation statistics.

### Intent classification metrics

- **accuracy()**
    - fraction where predicted intent equals expected intent
- **format_valid_rate()**
    - fraction of responses that parsed as valid JSON with required fields
- **avg_confidence()**
    - mean confidence score from valid responses
- **per_intent_accuracy()**
    - accuracy broken down per intent
- **confusion_matrix()**
    - shows which intents are commonly confused

---

### Leasing QA metrics

- **avg_judge_scores()**
    - mean score (0–2) per evaluation criterion
- **hallucination_rate()**
    - fraction of answers where hallucination score = 0

---

## evaluation/judge_llm.py

Implements the **LLM-as-a-judge** pattern.

Instead of manual grading, a second Gemini call evaluates responses.

### Evaluation criteria

Each answer is scored on:

- correctness
- faithfulness
- clarity
- policy_alignment
- hallucination

Scoring scale:

```
0 = poor
1 = partial
2 = good
```

Hallucination scoring is inverted so metrics remain consistent:

```
0 = hallucination detected
2 = no hallucination
```

### Robust parsing

`_parse_judge_response()`:

- strips markdown code fences such as ```json
- returns `1` for fields that fail parsing

This ensures parsing failures remain visible in results.

---

## evaluation/run_gemini.py

Main orchestration script.

Example usage:

```
python evaluation/run_gemini.py--task intent_classification--prompt good
```

---

### Intent classification flow

1. Load dataset, prompt, rubric
2. Call Gemini for each example
3. Parse JSON response
4. Compare predicted intent to expected intent
5. Print per-example results
6. Compute summary metrics
7. Save results to `/results`

---

### Leasing QA flow

1. Load dataset and prompt
2. Generate answer with Gemini
3. Send answer to judge LLM
4. Collect scores
5. Flag hallucinations with ⚠
6. Save results to `/results`

Each run produces a timestamped JSON file for tracking progress.

---

## evaluation/observability.py

Displays results in a terminal dashboard.

```
python evaluation/observability.py
```

### Mock mode

```
python evaluation/observability.py--mock
```

Uses synthetic data for demonstration before running real evaluations.

### Teaching example

| prompt | accuracy | hallucination rate |
| --- | --- | --- |
| bad | 52% | 40% |
| good | 84% | 7% |

---

## synthetic_data/generate_examples.md

Contains ready-to-use prompts for generating datasets with ChatGPT.

Each prompt specifies the required JSON schema so generated examples can be directly pasted into dataset files.

This connects:

```
dataset generation → evaluation pipeline
```

---

# Workshop Flow

## Part 3 — Intent Classification

Participants generate dataset examples:

```
ChatGPT → dataset.json
```

Run evaluation:

```
python evaluation/run_gemini.py--task intent_classification--prompt bad
python evaluation/run_gemini.py--task intent_classification--prompt good
```

Observe:

- improvement in accuracy
- improvement in JSON reliability

Compare results in:

- terminal output
- observability.py

---

## Part 4 — Leasing QA

Participants generate adversarial questions.

Run evaluation:

```
python evaluation/run_gemini.py --task leasing_qa --prompt bad
python evaluation/run_gemini.py --task leasing_qa --prompt good
```

Observe:

- hallucinations with weak prompts
- grounded answers with strong prompts
- judge scores per example

---

## Part 7 — Observability Dashboard

```
python evaluation/observability.py --task leasing_qa
```

Displays all runs:

- highlights failures
- shows hallucination alerts
- tracks improvements over time

---

# Key Takeaway

Prompt quality is measurable.

Not:

> "this prompt sounds better"
> 

But:

- +32% accuracy
- +35% format reliability
- hallucination rate reduced from 40% → 7%

Systematic evaluation transforms prompt engineering from intuition into engineering.