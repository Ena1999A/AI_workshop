"""
observability.py — Evaluation results dashboard.

Reads all result JSON files from results/ and prints per-prompt statistics
and worst failures for every run found.

Usage:
    python evaluation/observability.py
    python evaluation/observability.py --task intent_classification
    python evaluation/observability.py --failures 5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# ── loading ──────────────────────────────────────────────────────────────────

def load_results(task_filter: str | None = None) -> list[dict]:
    runs = []
    if not RESULTS_DIR.exists():
        return runs
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            if task_filter and data.get("task") != task_filter:
                continue
            data["_file"] = path.name
            runs.append(data)
        except (json.JSONDecodeError, KeyError):
            pass
    return runs


# ── formatting helpers ────────────────────────────────────────────────────────

W = 70

def sep(char: str = "─") -> None:
    print(char * W)

def header(title: str) -> None:
    print()
    print("┌" + "─" * (W - 2) + "┐")
    print("│" + title.center(W - 2) + "│")
    print("└" + "─" * (W - 2) + "┘")

def wrap(text: str, indent: int = 4, width: int = W) -> str:
    """Wrap long text at word boundaries with a fixed indent."""
    words = text.split()
    lines, line = [], []
    budget = width - indent
    for word in words:
        if sum(len(w) + 1 for w in line) + len(word) > budget:
            lines.append(" " * indent + " ".join(line))
            line = []
        line.append(word)
    if line:
        lines.append(" " * indent + " ".join(line))
    return "\n".join(lines)


# ── intent classification ─────────────────────────────────────────────────────

def print_intent_run(run: dict, n_failures: int) -> None:
    s = run["summary"]
    prompt = run["prompt"]
    ts = run.get("timestamp", "")[:19]

    header(f"INTENT CLASSIFICATION  │  prompt: {prompt}  │  {ts}")

    # stats block
    total = s.get("total_examples", len(run.get("examples", [])))
    correct = int(round(s.get("accuracy", 0) * total))
    print()
    print(f"  accuracy              {s.get('accuracy', 0):>6.1%}   ({correct}/{total} correct)")
    print(f"  format valid          {s.get('format_valid_rate', 0):>6.1%}")
    print(f"  avg confidence        {s.get('avg_confidence', 0):>6.2f}")
    print(f"  requires human review {s.get('requires_human_review_rate', 0):>6.1%}")

    # per-intent breakdown
    per_intent = s.get("per_intent_accuracy", {})
    if per_intent:
        print()
        print("  Per-intent accuracy:")
        sep()
        print(f"  {'intent':<30}  {'correct':>7}  {'total':>5}  {'acc':>6}")
        sep()
        for intent, stats in sorted(per_intent.items()):
            bar = "█" * int(stats["accuracy"] * 16) + "░" * (16 - int(stats["accuracy"] * 16))
            print(f"  {intent:<30}  {stats['correct']:>7}  {stats['total']:>5}  {stats['accuracy']:>5.0%}  {bar}")

    # worst failures
    examples = run.get("examples", [])
    failures = [e for e in examples if not e.get("correct", True)]
    # sort by confidence descending — confident wrong answers are the worst failures
    failures.sort(key=lambda e: e.get("confidence", 0), reverse=True)

    print()
    if not failures:
        print("  No failures.")
        return

    shown = failures[:n_failures]
    print(f"  Worst failures ({len(shown)} of {len(failures)} shown, sorted by confidence):")
    for f in shown:
        print()
        sep()
        # find original text from examples (id lookup)
        text = f.get("text", f"example #{f.get('id', '?')}")
        print(f"  Question  : {text}")
        print(f"  Expected  : {f.get('expected_intent', '?')}")
        print(f"  Predicted : {f.get('intent', '?')}  (confidence {f.get('confidence', 0):.2f})")
        if f.get("explanation"):
            print(wrap(f"Reasoning : {f['explanation']}", indent=2))
        if not f.get("format_valid", True):
            print(f"  ⚠ JSON parse failed — raw: {f.get('raw', '')[:120]!r}")


# ── leasing QA ────────────────────────────────────────────────────────────────

def print_qa_run(run: dict, n_failures: int) -> None:
    s = run["summary"]
    prompt = run["prompt"]
    ts = run.get("timestamp", "")[:19]

    header(f"LEASING Q&A  │  prompt: {prompt}  │  {ts}")

    # stats block
    avg = s.get("avg_scores", {})
    print()
    print(f"  hallucination rate    {s.get('hallucination_rate', 0):>6.1%}")
    print()
    print("  Judge scores (avg, scale 0–2):")
    sep()
    for criterion, score in avg.items():
        print(f"  {criterion:<22}  {score:>4.2f} ")

    # worst failures: lowest combined score + hallucinations first
    examples = run.get("examples", [])

    def failure_score(e: dict) -> float:
        criteria = ["correctness", "faithfulness", "clarity", "policy_alignment"]
        total = sum(e.get(c, 2) for c in criteria)
        halluc_penalty = 10 if e.get("hallucination") == 0 else 0
        return total - halluc_penalty  # lower = worse

    ranked = sorted(examples, key=failure_score)

    print()
    shown = ranked[:n_failures]
    print(f"  Worst answers ({len(shown)} of {len(examples)} shown, hallucinations first):")

    for e in shown:
        print()
        sep()
        halluc_flag = "  ⚠ HALLUCINATION DETECTED" if e.get("hallucination") == 0 else ""
        print(f"  Question  : {e.get('question', '?')}{halluc_flag}")
        print()

        answer = e.get("answer", "").strip()
        print("  Answer:")
        print(wrap(answer, indent=4, width=W))
        print()

        criteria = ["correctness", "faithfulness", "clarity", "policy_alignment", "hallucination"]
        scores_str = "  ".join(f"{c[:8]}={e.get(c, '?')}" for c in criteria)
        print(f"  Scores    : {scores_str}")
        if e.get("notes") and e["notes"] not in ("None", ""):
            print(wrap(f"Judge note: {e['notes']}", indent=2))


# ── main ──────────────────────────────────────────────────────────────────────

def dashboard(runs: list[dict], n_failures: int) -> None:
    if not runs:
        print()
        print("  No results found.")
        print("  Run the evaluation first:")
        print("    python evaluation/run_gemini.py --task intent_classification --prompt good")
        print("    python evaluation/run_gemini.py --task intent_classification --prompt bad")
        print("    python evaluation/run_gemini.py --task leasing_qa --prompt good")
        print("    python evaluation/run_gemini.py --task leasing_qa --prompt bad")
        print()
        return

    print()
    print(f"  Loaded {len(runs)} run(s) from {RESULTS_DIR}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    intent_runs = [r for r in runs if r["task"] == "intent_classification"]
    qa_runs = [r for r in runs if r["task"] == "leasing_qa"]

    for run in intent_runs:
        print_intent_run(run, n_failures)

    for run in qa_runs:
        print_qa_run(run, n_failures)

    sep("═")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM evaluation dashboard")
    parser.add_argument("--task", help="Filter by task name")
    parser.add_argument(
        "--failures",
        type=int,
        default=3,
        metavar="N",
        help="Number of worst failures to show per run (default: 3)",
    )
    args = parser.parse_args()

    runs = load_results(args.task)
    dashboard(runs, args.failures)


if __name__ == "__main__":
    main()
