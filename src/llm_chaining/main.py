"""
llm_chaining/main.py

Workshop demo: LLM chaining with intent-based routing.

Pipeline:
  User message
    → LLM 1 : intent classifier          (returns JSON with intent field)
    → Python router                       (reads intent, picks the right LLM)
    → LLM 2a / 2b / 2c (specialized)
        update_contact_info      → returns structured JSON payload + user confirmation
        repair_status_question   → returns a plain user-facing answer
        leasing_policy_question  → returns a plain user-facing answer

System prompts live in:  system_prompts/
User prompt examples in: user_prompts/examples.txt  (one message per line)
Logs saved to:           logs/demo_<timestamp>.log

HOW TO USE:
  export GEMINI_API_KEY="your_key"
  python main.py
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

# Paths
BASE_DIR = Path(__file__).parent
SYSTEM_PROMPTS_DIR = BASE_DIR / "system_prompts"
USER_PROMPTS_DIR = BASE_DIR / "user_prompts"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_NAME = "gemini-2.5-flash-lite"

INTENT_CLASSIFIER_PROMPT = SYSTEM_PROMPTS_DIR / "intent_classifier.txt"
SPECIALIZED_PROMPTS: dict[str, Path] = {
    "update_contact_info":     SYSTEM_PROMPTS_DIR / "update_contact_info.txt",
    "repair_status_question":  SYSTEM_PROMPTS_DIR / "repair_status_question_rag.txt",
    "leasing_policy_question": SYSTEM_PROMPTS_DIR / "leasing_policy_question.txt",
}

# Logging
log_file = LOG_DIR / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# Helpers 

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_user_prompts() -> list[str]:
    prompts: list[str] = []
    for f in sorted(USER_PROMPTS_DIR.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


# Gemini call

def call_gemini(client: genai.Client, system_prompt: str, user_message: str) -> str:
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.2,
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config=config,
    )
    return getattr(response, "text", None) or "<empty response>"


# LLM 1: Intent classifier

def classify_intent(client: genai.Client, user_message: str) -> tuple[str, float, str]:
    """
    Call the intent classifier LLM.
    Returns (intent, confidence, reasoning).
    Falls back to ("unknown", 0.0, raw_response) on parse error.
    """
    system_prompt = load_text(INTENT_CLASSIFIER_PROMPT)
    raw = call_gemini(client, system_prompt, user_message)

    # Strip optional markdown code fences the model sometimes adds
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.splitlines()[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.splitlines()[:-1])

    try:
        data = json.loads(clean)
        return (
            data.get("intent", "unknown"),
            float(data.get("confidence", 0.0)),
            data.get("reasoning", ""),
        )
    except json.JSONDecodeError:
        return "unknown", 0.0, raw


# Python router

def route(intent: str) -> Path | None:
    """Return the system prompt path for the given intent, or None if unknown."""
    return SPECIALIZED_PROMPTS.get(intent)


# LLM 2: Specialized handlers

def handle_update_contact_info(client: genai.Client, user_message: str) -> dict:
    """
    Returns a dict with two keys:
      db_payload  – the structured JSON for the database update
      user_answer – the confirmation message shown to the user
    """
    system_prompt = load_text(SPECIALIZED_PROMPTS["update_contact_info"])
    raw = call_gemini(client, system_prompt, user_message)

    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.splitlines()[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.splitlines()[:-1])

    try:
        payload = json.loads(clean)
        return {
            "db_payload": payload,
            "user_answer": payload.get("user_message", "Your details have been updated."),
        }
    except json.JSONDecodeError:
        return {"db_payload": None, "user_answer": raw}


def handle_plain_answer(client: genai.Client, intent: str, user_message: str) -> str:
    system_prompt = load_text(SPECIALIZED_PROMPTS[intent])
    return call_gemini(client, system_prompt, user_message)


#  Pipeline orchestrator 

def run_pipeline(client: genai.Client, user_message: str) -> None:
    sep = "=" * 70
    log.info(sep)
    log.info(f"  USER MESSAGE : {user_message}")
    log.info("-" * 70)

    # Step 1: Classify intent
    intent, confidence, reasoning = classify_intent(client, user_message)
    log.info(f"  [LLM 1 - CLASSIFIER]")
    log.info(f"  Intent     : {intent}")
    log.info(f"  Confidence : {confidence}")
    log.info(f"  Reasoning  : {reasoning}")
    log.info("-" * 70)

    # Step 2: Route 
    prompt_path = route(intent)
    if prompt_path is None:
        log.info(f"  [ROUTER] Unknown intent '{intent}' — no specialized LLM available.")
        log.info(sep + "\n")
        return

    log.info(f"  [ROUTER] → routed to: {prompt_path.stem}")
    log.info("-" * 70)

    #  Step 3: Specialized LLM 
    if intent == "update_contact_info":
        result = handle_update_contact_info(client, user_message)
        log.info("  [LLM 2 - update_contact_info]")
        log.info("  DB PAYLOAD :")
        for line in json.dumps(result["db_payload"], indent=4, ensure_ascii=False).splitlines():
            log.info(f"    {line}")
        log.info(f"  USER ANSWER : {result['user_answer']}")
    else:
        answer = handle_plain_answer(client, intent, user_message)
        log.info(f"  [LLM 2 - {intent}]")
        log.info("  ANSWER :")
        for line in answer.splitlines():
            log.info(f"    {line}")

    log.info(sep + "\n")


# Entry point
#  
def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please export it before running."
        )
    client = genai.Client(api_key=api_key)

    log.info(f"Model    : {MODEL_NAME}")
    log.info(f"Log file : {log_file}\n")

    user_prompts = load_user_prompts()
    log.info(f"Loaded {len(user_prompts)} user prompt(s) from user_prompts/\n")

    for user_message in user_prompts:
        run_pipeline(client, user_message)

    log.info("Demo complete.")


if __name__ == "__main__":
    main()
