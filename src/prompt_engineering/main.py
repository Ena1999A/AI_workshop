"""
prompt_engineering/main.py

Workshop demo: building a production-ready prompt step by step.

Stages (subfolders of system_prompts/, run in alphabetical order):
  01_simple
  02_role
  03_constraint
  04_structured_output
  05_few_shot
  06_golden_prompt

System prompt files are named by domain:
  leasing_*.txt  →  used in the leasing run
  intent_*.txt   →  used in the intent run

User prompt files live in user_prompts/:
  leasing_*.txt  →  one user prompt per line
  intent_*.txt   →  one user prompt per line

HOW TO USE:
  1. Add system prompt .txt files to the appropriate system_prompts/<stage>/ folder
  2. Add user prompt .txt files (one prompt per line) to user_prompts/
  3. Run:  python main.py
  Logs are saved to logs/demo_<timestamp>.log
"""

from __future__ import annotations

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


# Loaders

def load_system_prompts(prefix: str) -> list[tuple[str, str, str]]:
    """
    Scan all stage subfolders alphabetically and collect system prompt files
    whose name starts with `prefix`.

    Returns [(stage_folder_name, file_stem, content), ...]
    """
    results = []
    for stage_dir in sorted(SYSTEM_PROMPTS_DIR.iterdir()):
        if not stage_dir.is_dir():
            continue
        for f in sorted(stage_dir.glob(f"{prefix}*.txt")):
            results.append((stage_dir.name, f.stem, f.read_text(encoding="utf-8").strip()))
    return results


def load_user_prompts(prefix: str) -> list[str]:
    """
    Load every non-empty line from user_prompts/ files whose name starts with `prefix`.
    Each line is treated as one independent user prompt.
    """
    prompts: list[str] = []
    for f in sorted(USER_PROMPTS_DIR.glob(f"{prefix}*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


# API call 

def call_gemini(client: genai.Client, system_prompt: str, user_prompt: str) -> str:
    config = types.GenerateContentConfig(system_instruction=system_prompt)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=config,
    )
    return getattr(response, "text", None) or "<empty response>"


# Domain runner 

def run_domain(client: genai.Client, prefix: str) -> None:
    domain_title = prefix.upper()
    system_prompts = load_system_prompts(prefix)
    user_prompts = load_user_prompts(prefix)

    banner = "#" * 70
    log.info(banner)
    log.info(f"  DOMAIN: {domain_title}")
    log.info(f"  System prompts : {len(system_prompts)}")
    log.info(f"  User prompts   : {len(user_prompts)}")
    log.info(banner + "\n")

    if not system_prompts:
        log.info(f"  [SKIPPED] No {prefix}_*.txt files found in system_prompts/\n")
        return
    if not user_prompts:
        log.info(f"  [SKIPPED] No {prefix}_*.txt files found in user_prompts/\n")
        return

    for stage_folder, sys_name, sys_content in system_prompts:
        log.info("=" * 70)
        log.info(f"  STAGE           : {stage_folder}")
        log.info(f"  SYSTEM PROMPT   : {sys_name}")
        log.info("-" * 70)
        log.info(sys_content)
        log.info("=" * 70 + "\n")

        for i, user_prompt in enumerate(user_prompts, start=1):
            log.info(f"  >> USER PROMPT [{i}/{len(user_prompts)}] : {user_prompt}")
            log.info("-" * 35)

            output = call_gemini(client, sys_content, user_prompt)

            log.info("  << RESPONSE :")
            for line in output.splitlines():
                log.info(f"     {line}")
            log.info("")

        log.info("")


# Entry point 

def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please export it before running."
        )
    client = genai.Client(api_key=api_key)

    log.info(f"Model    : {MODEL_NAME}")
    log.info(f"Log file : {log_file}\n")

    # run_domain(client, "leasing")
    run_domain(client, "intent")

    log.info("Demo complete.")


if __name__ == "__main__":
    main()
