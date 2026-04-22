"""
metrics.py — Pure metric computation functions for the evaluation pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    if not predictions:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def format_valid_rate(responses: list[dict[str, Any]]) -> float:
    if not responses:
        return 0.0
    valid = sum(1 for r in responses if r.get("format_valid", False))
    return valid / len(responses)


def avg_confidence(responses: list[dict[str, Any]]) -> float:
    scores = [r["confidence"] for r in responses if r.get("format_valid") and "confidence" in r]
    return sum(scores) / len(scores) if scores else 0.0


def requires_human_review_rate(responses: list[dict[str, Any]]) -> float:
    valid = [r for r in responses if r.get("format_valid")]
    if not valid:
        return 0.0
    flagged = sum(1 for r in valid if r.get("requires_human_review", False))
    return flagged / len(valid)


def per_intent_accuracy(
    predictions: list[str],
    ground_truth: list[str],
) -> dict[str, dict[str, int | float]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, gold in zip(predictions, ground_truth):
        counts[gold]["total"] += 1
        if pred == gold:
            counts[gold]["correct"] += 1
    return {
        intent: {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
        }
        for intent, v in counts.items()
    }


def confusion_matrix(
    predictions: list[str],
    ground_truth: list[str],
    labels: list[str],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    for pred, gold in zip(predictions, ground_truth):
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


def avg_judge_scores(judge_results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-criterion scores from LLM-as-judge results."""
    if not judge_results:
        return {}
    criteria = [k for k in judge_results[0] if k not in ("id", "notes", "raw")]
    aggregated: dict[str, list[float]] = defaultdict(list)
    for result in judge_results:
        for criterion in criteria:
            if isinstance(result.get(criterion), (int, float)):
                aggregated[criterion].append(float(result[criterion]))
    return {c: sum(v) / len(v) for c, v in aggregated.items() if v}


def hallucination_rate(judge_results: list[dict[str, Any]]) -> float:
    """Fraction of answers that received a hallucination score of 0."""
    scores = [r.get("hallucination") for r in judge_results if "hallucination" in r]
    if not scores:
        return 0.0
    return sum(1 for s in scores if s == 0) / len(scores)


def intent_classification_summary(
    examples: list[dict],
    responses: list[dict],
    valid_intents: list[str],
) -> dict[str, Any]:
    predictions = [r.get("intent", "") for r in responses]
    ground_truth = [e["expected_intent"] for e in examples]

    return {
        "accuracy": accuracy(predictions, ground_truth),
        "format_valid_rate": format_valid_rate(responses),
        "avg_confidence": avg_confidence(responses),
        "requires_human_review_rate": requires_human_review_rate(responses),
        "per_intent_accuracy": per_intent_accuracy(predictions, ground_truth),
        "confusion_matrix": confusion_matrix(predictions, ground_truth, valid_intents),
        "total_examples": len(examples),
    }


def leasing_qa_summary(judge_results: list[dict]) -> dict[str, Any]:
    return {
        "avg_scores": avg_judge_scores(judge_results),
        "hallucination_rate": hallucination_rate(judge_results),
        "total_examples": len(judge_results),
    }
