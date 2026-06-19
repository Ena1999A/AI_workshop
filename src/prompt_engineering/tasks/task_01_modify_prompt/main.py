"""
task_01_modify_prompt/main.py

Run this script, then edit system_prompt.txt or user_prompt.txt and run again.
Watch how the output changes based on your prompt.

Usage:
    python src/prompt_engineering/tasks/task_01_modify_prompt/main.py
    LLM_PROVIDER=claude python src/prompt_engineering/tasks/task_01_modify_prompt/main.py
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

try:
    import anthropic as anthropic_sdk
except ImportError:
    anthropic_sdk = None

HERE = Path(__file__).parent
SYSTEM_PROMPT_FILE = HERE / "system_prompt.txt"
USER_PROMPT_FILE = HERE / "user_prompt.txt"

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL = "gemini-2.5-flash-lite"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")


def build_client():
    if PROVIDER == "claude":
        if anthropic_sdk is None:
            raise ImportError("Install the Anthropic SDK: pip install anthropic")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        return anthropic_sdk.Anthropic(api_key=api_key)
    else:
        if genai is None:
            raise ImportError("Install the Google Gen AI SDK: pip install google-genai")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        return genai.Client(api_key=api_key)


def call_llm(client, system_prompt: str, user_prompt: str) -> str:
    if PROVIDER == "claude":
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text if response.content else "<empty response>"
    else:
        config = genai_types.GenerateContentConfig(system_instruction=system_prompt)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=config,
        )
        return getattr(response, "text", None) or "<empty response>"


def main() -> None:
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    user_prompt = USER_PROMPT_FILE.read_text(encoding="utf-8").strip()

    print(f"Provider      : {PROVIDER.upper()}")
    print(f"System prompt : {system_prompt}")
    print(f"User prompt   : {user_prompt}")
    print("-" * 50)

    client = build_client()
    response = call_llm(client, system_prompt, user_prompt)

    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
