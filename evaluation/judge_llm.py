"""
judge_llm.py — LLM-as-a-judge for evaluating leasing Q&A answers.

Uses Gemini to score each answer against the rubric criteria defined in rubric.json.
"""

from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types

_JUDGE_SYSTEM_PROMPT_TEMPLATE = """You are an expert evaluator for a leasing customer support system.

Your task is to evaluate an AI-generated answer to a customer question about leasing policy.

You must evaluate the answer strictly against the official company policy provided below.
Do not use outside knowledge — if something is not mentioned in the policy, the answer should not claim it.

========== OFFICIAL COMPANY POLICY ==========
{policy}
==============================================

You will receive:
- The customer's question
- The AI-generated answer
- The expected topics the answer should cover

Score each criterion from 0 to 2:
  0 = Poor: criterion clearly not met
  1 = Acceptable: criterion partially met or with caveats
  2 = Good: criterion clearly met

Criteria to evaluate:

correctness (0-2):
  Is the answer factually accurate according to the policy above?
  0 = Contains facts that contradict or are absent from the policy
  1 = Mostly accurate but missing key facts from the policy
  2 = Fully consistent with the policy

faithfulness (0-2):
  Does the answer stay strictly within what the policy above says?
  Does it avoid inventing specific rules, fees, or guarantees not in the policy?
  0 = Invents facts, amounts, or guarantees not found in the policy
  1 = Minor unsupported claims that go slightly beyond the policy
  2 = Fully grounded in the policy, no invented content

clarity (0-2):
  Is the answer clear, structured, and easy for a customer to understand?
  0 = Confusing or poorly structured
  1 = Understandable but could be clearer
  2 = Clear, well-structured, appropriately concise

policy_alignment (0-2):
  Does the answer respect the policy's boundaries and caveats?
  Does it avoid making promises or guarantees the policy explicitly does not support?
  0 = Makes promises or guarantees the policy does not support (e.g. "no penalty", "guaranteed replacement")
  1 = Slightly over-promising or missing important policy caveats
  2 = Correctly reflects policy limits and directs customer to advisor when appropriate

hallucination (0-2):
  0 = Hallucination detected: answer states a specific fact (fee amount, guarantee, rule) that is NOT in the policy
  1 = Minor: vague unsupported phrasing that goes slightly beyond the policy
  2 = No hallucination: every claim in the answer is grounded in the policy above

Return ONLY valid JSON in this format:
{{
  "correctness": <0|1|2>,
  "faithfulness": <0|1|2>,
  "clarity": <0|1|2>,
  "policy_alignment": <0|1|2>,
  "hallucination": <0|1|2>,
  "notes": "One sentence citing the specific policy rule that was violated or invented, or 'None' if all criteria met."
}}
"""


def build_judge_prompt(question: str, answer: str, expected_topics: list[str]) -> str:
    topics_str = ", ".join(expected_topics) if expected_topics else "none specified"
    return f"""Customer question: {question}

Expected topics to cover: {topics_str}

AI-generated answer:
{answer}

Evaluate the answer using the criteria in your instructions."""


def build_judge_system_prompt(policy: str) -> str:
    return _JUDGE_SYSTEM_PROMPT_TEMPLATE.format(policy=policy)


def judge_answer(
    client: genai.Client,
    model: str,
    question: str,
    answer: str,
    expected_topics: list[str],
    policy: str,
    temperature: float = 0.1,
) -> dict[str, Any]:
    config = types.GenerateContentConfig(
        system_instruction=build_judge_system_prompt(policy),
        temperature=temperature,
        max_output_tokens=512,
    )
    prompt = build_judge_prompt(question, answer, expected_topics)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    raw = getattr(response, "text", None) or ""
    return _parse_judge_response(raw)


def _parse_judge_response(raw: str) -> dict[str, Any]:
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(lines[1:])
    if clean.endswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(lines[:-1])

    try:
        data = json.loads(clean)
        return {
            "correctness": int(data.get("correctness", -1)),
            "faithfulness": int(data.get("faithfulness", -1)),
            "clarity": int(data.get("clarity", -1)),
            "policy_alignment": int(data.get("policy_alignment", -1)),
            "hallucination": int(data.get("hallucination", -1)),
            "notes": data.get("notes", ""),
            "format_valid": True,
            "raw": raw,
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "correctness": -1,
            "faithfulness": -1,
            "clarity": -1,
            "policy_alignment": -1,
            "hallucination": -1,
            "notes": "parse error",
            "format_valid": False,
            "raw": raw,
        }
