"""
api_config_demo.py

A workshop-friendly script that:
1. Calls Gemini from Python
2. Shows how to configure generation parameters
3. Demonstrates the effect of those parameters with small examples

Setup:
    pip install google-genai
    export GOOGLE_API_KEY="your_api_key_here"

Run:
    python api_config_demo.py
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from textwrap import indent

from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.5-flash-lite"

# Logging setup 
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Prompt - change here
PROMPT = """
Follow the instructions exactly.

Return exactly 3 words.

Text:
"The system experienced a temporary outage because a configuration change was deployed without proper validation checks."
"""
# Parameter scenarios
SCENARIOS = [
    {
        "name": "Low Temperature (deterministic)",
        "description": (
            "temperature=0.2 → the model picks the most likely next token almost every time. "
            "Output is focused, predictable, and consistent across runs."
        ),
        "params": dict(temperature=0.2),
    },
    {
        "name": "High Temperature (creative / chaotic)",
        "description": (
            "temperature=2.0 → the model samples from a much flatter probability distribution. "
            "Output is surprising and varied, sometimes incoherent."
        ),
        "params": dict(temperature=2.0),
    },
    {
        "name": "Low top_p (nucleus sampling – narrow)",
        "description": (
            "top_p=0.1 → only tokens covering the top 10% of probability mass are considered. "
            "Very conservative word choices, repetitive style."
        ),
        "params": dict(top_p=0.1),
    },
    {
        "name": "High top_p (nucleus sampling – wide)",
        "description": (
            "top_p=0.95 → almost the full vocabulary is in play. "
            "Rich, diverse phrasing."
        ),
        "params": dict(top_p=0.95),
    },
    {
        "name": "Very short max_output_tokens",
        "description": (
            "max_output_tokens=40 → the model is hard-stopped early. "
            "The story will be cut off mid-thought."
        ),
        "params": dict(max_output_tokens=40),
    },
    {
        "name": "High temp and top_p",
        "description": (
            "max_output_tokens=500 → the model has room to develop the story fully."
        ),
        "params": dict(top_p=0.95, temperature=2.0),
    },
]


def build_client() -> genai.Client:
    """
    Create a Gemini client.

    The current Google Gen AI SDK supports initializing the client with
    an API key directly or through the GEMINI_API_KEY environment variable.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please export it before running the script."
        )
    return genai.Client(api_key=api_key)


def call_gemini(
    client: genai.Client,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_output_tokens: int = 500,
    top_p: float | None = None,
    system_instruction: str | None = None,
) -> str:
    """
    Make one Gemini request with configurable generation parameters.

    Parameters explained:
    - temperature:
        Lower -> more stable / deterministic
        Higher -> more creative / variable

    - max_output_tokens:
        Hard cap on output length

    - top_p:
        Alternative randomness control based on nucleus sampling.
        Usually you tune either temperature first, then optionally top_p.

    - system_instruction:
        High-level behavior guidance for the model
    """
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        system_instruction=system_instruction,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )

    return response.text if hasattr(response, "text") and response.text else "<empty response>"


def run_scenario(client: genai.Client, scenario: dict) -> None:
    sep = "=" * 70
    log.info(sep)
    log.info(f"SCENARIO : {scenario['name']}")
    log.info(f"WHY      : {scenario['description']}")
    log.info(f"PROMPT   : {PROMPT}")
    log.info(f"PARAMS   : {scenario['params']}")
    log.info("-" * 70)

    output = call_gemini(client, PROMPT, **scenario["params"])

    log.info(f"OUTPUT   :\n{indent(output, '  ')}")
    log.info(sep + "\n")


def main() -> None:
    client = build_client()

    log.info(f"Model    : {MODEL_NAME}")
    log.info(f"Log file : {log_file}\n")

    for scenario in SCENARIOS:
        run_scenario(client, scenario)

    log.info("Demo complete.")



if __name__ == "__main__":
    main()