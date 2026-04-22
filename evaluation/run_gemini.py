"""
run_gemini.py — Main evaluation runner for the LLM evaluation workshop.

Usage:
    python evaluation/run_gemini.py --task intent_classification --prompt good
    python evaluation/run_gemini.py --task leasing_qa --prompt good
    python evaluation/run_gemini.py --task intent_classification --prompt bad

Setup:
    pip install google-genai pyyaml
    export GEMINI_API_KEY="your_api_key_here"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from google import genai
from google.genai import types

# Add project root so sibling imports work when called from any directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.judge_llm import judge_answer
from evaluation import metrics as m

LOG_DIR = PROJECT_ROOT / "evaluation" / "logs"
LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger(__name__)


def setup_logging(task: str, prompt_label: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{task}_{prompt_label}_{ts}.log"

    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(sh)
    return log_file


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> list | dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def call_gemini(
    client: genai.Client,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    log.debug("  [API CALL]")
    log.debug(f"  model       : {model}")
    log.debug(f"  temperature : {temperature}")
    log.debug(f"  max_tokens  : {max_output_tokens}")
    log.debug(f"  system_prompt (first 120 chars): {system_prompt[:120].replace(chr(10), ' ')!r}")
    log.debug(f"  user_message: {user_message!r}")

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=config,
    )
    raw = getattr(response, "text", None) or ""
    log.debug(f"  raw_response: {raw!r}")
    return raw


def parse_intent_response(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.splitlines()[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.splitlines()[:-1])
    try:
        data = json.loads(clean)
        result = {
            "intent": data.get("intent", ""),
            "confidence": float(data.get("confidence", 0.0)),
            "explanation": data.get("explanation", data.get("reasoning", "")),
            "requires_human_review": bool(data.get("requires_human_review", False)),
            "format_valid": True,
            "raw": raw,
        }
        log.debug(f"  parsed      : intent={result['intent']!r} confidence={result['confidence']}")
        return result
    except (json.JSONDecodeError, ValueError):
        log.debug(f"  parse ERROR : could not decode JSON from response")
        return {
            "intent": "",
            "confidence": 0.0,
            "explanation": "",
            "requires_human_review": False,
            "format_valid": False,
            "raw": raw,
        }


def run_intent_classification(
    client: genai.Client,
    cfg: dict,
    task_cfg: dict,
    system_prompt: str,
    prompt_label: str,
    results_dir: Path,
) -> dict:
    dataset = load_json(PROJECT_ROOT / task_cfg["dataset"])
    rubric = load_json(PROJECT_ROOT / task_cfg["rubric"])

    log.info(f"\n{'='*60}")
    log.info(f"Task       : intent_classification")
    log.info(f"Prompt     : {prompt_label}")
    log.info(f"Examples   : {len(dataset)}")
    log.info(f"Model      : {cfg['model']}")
    log.info(f"{'='*60}\n")
    log.debug("SYSTEM PROMPT:\n" + "-" * 40)
    log.debug(system_prompt)
    log.debug("-" * 40 + "\n")

    responses = []
    for i, example in enumerate(dataset, 1):
        log.info(f"[{i:02d}/{len(dataset)}] {example['text'][:60]}...")
        t0 = time.time()
        raw = call_gemini(
            client,
            cfg["model"],
            system_prompt,
            example["text"],
            cfg["temperature"],
            cfg["max_output_tokens"],
        )
        latency_ms = (time.time() - t0) * 1000
        parsed = parse_intent_response(raw)
        parsed["id"] = example["id"]
        parsed["text"] = example["text"]
        parsed["expected_intent"] = example["expected_intent"]
        parsed["latency_ms"] = round(latency_ms, 1)
        parsed["correct"] = parsed["intent"] == example["expected_intent"]
        responses.append(parsed)

        status = "OK" if parsed["format_valid"] else "PARSE_ERR"
        correct_marker = "✓" if parsed["correct"] else "✗"
        log.info(
            f"  [{status}] {correct_marker} predicted={parsed['intent']!r:30s} "
            f"expected={example['expected_intent']!r:25s} conf={parsed['confidence']:.2f}"
        )

    summary = m.intent_classification_summary(
        dataset, responses, rubric["valid_intents"]
    )
    summary["avg_latency_ms"] = sum(r["latency_ms"] for r in responses) / len(responses)

    _print_intent_summary(summary, prompt_label)

    result = {
        "task": "intent_classification",
        "prompt": prompt_label,
        "model": cfg["model"],
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "examples": responses,
    }
    _save_results(results_dir, "intent_classification", prompt_label, result)
    return result


def run_leasing_qa(
    client: genai.Client,
    cfg: dict,
    task_cfg: dict,
    system_prompt: str,
    prompt_label: str,
    results_dir: Path,
) -> dict:
    dataset = load_json(PROJECT_ROOT / task_cfg["dataset"])
    policy = load_text(PROJECT_ROOT / task_cfg["policy"])

    log.info(f"\n{'='*60}")
    log.info(f"Task       : leasing_qa")
    log.info(f"Prompt     : {prompt_label}")
    log.info(f"Examples   : {len(dataset)}")
    log.info(f"Model      : {cfg['model']}")
    log.info(f"{'='*60}\n")
    log.debug("SYSTEM PROMPT:\n" + "-" * 40)
    log.debug(system_prompt)
    log.debug("-" * 40 + "\n")
    log.debug("JUDGE POLICY:\n" + "-" * 40)
    log.debug(policy)
    log.debug("-" * 40 + "\n")

    judge_cfg = cfg.get("judge", {})
    judge_model = judge_cfg.get("model", cfg["model"])
    judge_temp = judge_cfg.get("temperature", 0.1)

    examples_output = []
    judge_results = []

    for i, example in enumerate(dataset, 1):
        log.info(f"[{i:02d}/{len(dataset)}] {example['question'][:60]}...")

        t0 = time.time()
        answer = call_gemini(
            client,
            cfg["model"],
            system_prompt,
            example["question"],
            cfg["temperature"],
            cfg["max_output_tokens"],
        )
        latency_ms = (time.time() - t0) * 1000
        log.debug(f"  answer: {answer!r}")

        log.debug(f"  [JUDGE CALL] question={example['question']!r}")
        scores = judge_answer(
            client,
            judge_model,
            example["question"],
            answer,
            example.get("expected_topics", []),
            policy=policy,
            temperature=judge_temp,
        )
        log.debug(f"  judge raw: {scores.get('raw', '')!r}")
        scores["id"] = example["id"]
        scores["question"] = example["question"]
        scores["answer"] = answer
        scores["risk"] = example.get("risk", "")
        scores["latency_ms"] = round(latency_ms, 1)

        judge_results.append(scores)
        examples_output.append(scores)

        hallucination_flag = " ⚠ HALLUCINATION" if scores["hallucination"] == 0 else ""
        log.info(
            f"  correct={scores['correctness']} faithful={scores['faithfulness']} "
            f"clarity={scores['clarity']} policy={scores['policy_alignment']} "
            f"halluc={scores['hallucination']}{hallucination_flag}"
        )
        if scores.get("notes") and scores["notes"] != "None":
            log.info(f"  notes: {scores['notes']}")

    summary = m.leasing_qa_summary(judge_results)
    summary["avg_latency_ms"] = sum(r["latency_ms"] for r in examples_output) / len(examples_output)

    _print_qa_summary(summary, prompt_label)

    result = {
        "task": "leasing_qa",
        "prompt": prompt_label,
        "model": cfg["model"],
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "examples": examples_output,
    }
    _save_results(results_dir, "leasing_qa", prompt_label, result)
    return result


def _print_intent_summary(summary: dict, prompt_label: str) -> None:
    log.info(f"\n{'='*60}")
    log.info(f"RESULTS — intent_classification ({prompt_label} prompt)")
    log.info(f"{'='*60}")
    log.info(f"  accuracy              : {summary['accuracy']:.2%}")
    log.info(f"  format_valid_rate     : {summary['format_valid_rate']:.2%}")
    log.info(f"  avg_confidence        : {summary['avg_confidence']:.2f}")
    log.info(f"  requires_human_review : {summary['requires_human_review_rate']:.2%}")
    log.info(f"  avg_latency_ms        : {summary['avg_latency_ms']:.0f}ms")
    log.info(f"\nPer-intent accuracy:")
    for intent, stats in summary["per_intent_accuracy"].items():
        log.info(f"  {intent:30s}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.0%})")
    log.info(f"{'='*60}\n")


def _print_qa_summary(summary: dict, prompt_label: str) -> None:
    log.info(f"\n{'='*60}")
    log.info(f"RESULTS — leasing_qa ({prompt_label} prompt)")
    log.info(f"{'='*60}")
    avg = summary["avg_scores"]
    for criterion, score in avg.items():
        log.info(f"  {criterion:20s}: {score:.2f} / 2.0")
    log.info(f"  {'hallucination_rate':20s}: {summary['hallucination_rate']:.2%}")
    log.info(f"  {'avg_latency_ms':20s}: {summary['avg_latency_ms']:.0f}ms")
    log.info(f"{'='*60}\n")


def _save_results(results_dir: Path, task: str, prompt_label: str, data: dict) -> None:
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"{task}_{prompt_label}_{ts}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved to: {filename}")


def resolve_prompt_path(prompt_name: str, task_name: str, task_cfg: dict) -> Path:
    """
    Resolve a prompt name to a file path using three fallback steps:
      1. Named entry in config.yaml  (e.g. "good" → tasks/.../prompt_good.txt)
      2. Convention  tasks/{task}/prompt_{name}.txt
      3. Treat as a direct file path (absolute or relative to PROJECT_ROOT)
    """
    # 1. config.yaml registry
    registered = task_cfg.get("prompts", {})
    if prompt_name in registered:
        return PROJECT_ROOT / registered[prompt_name]

    # 2. convention-based lookup
    convention = PROJECT_ROOT / "tasks" / task_name / f"prompt_{prompt_name}.txt"
    if convention.exists():
        return convention

    # 3. direct path
    direct = Path(prompt_name)
    if not direct.is_absolute():
        direct = PROJECT_ROOT / direct
    if direct.exists():
        return direct

    raise FileNotFoundError(
        f"Cannot find prompt {prompt_name!r}.\n"
        f"  Tried config entry  : {registered.get(prompt_name, 'not registered')}\n"
        f"  Tried convention    : {convention}\n"
        f"  Tried direct path   : {direct}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["intent_classification", "leasing_qa"],
        help="Which task to evaluate",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help=(
            "Prompt to use. Accepts:\n"
            "  - a name registered in config.yaml          (e.g. good, bad)\n"
            "  - a filename in tasks/{task}/prompt_{name}.txt  (e.g. v2, strict)\n"
            "  - a direct file path                        (e.g. my_prompts/test.txt)"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "evaluation" / "config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    cfg = load_config(Path(args.config))
    task_cfg = cfg["tasks"][args.task]
    results_dir = PROJECT_ROOT / cfg.get("results_dir", "results")

    log_file = setup_logging(args.task, args.prompt)
    log.info(f"Log file   : {log_file}")

    prompt_path = resolve_prompt_path(args.prompt, args.task, task_cfg)
    log.info(f"Prompt file: {prompt_path}")
    system_prompt = load_text(prompt_path)
    client = build_client(api_key)

    if args.task == "intent_classification":
        run_intent_classification(client, cfg, task_cfg, system_prompt, args.prompt, results_dir)
    elif args.task == "leasing_qa":
        run_leasing_qa(client, cfg, task_cfg, system_prompt, args.prompt, results_dir)


if __name__ == "__main__":
    main()
