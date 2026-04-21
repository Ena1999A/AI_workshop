"""
Chatbot pipeline — full turn processing.

Steps per user message:
  1. Sanitize (mask PII)
  2. Fetch recent history + summary from memory
  3. Identify customer from PII (if any found)
  4. Contextualize — rewrite follow-up into standalone query
  5. Intent detection
  6. Retrieval — vector search filtered by intent
  7. Answer generation
  8. Output guard — PII / groundedness check
  9. Update memory (raw history, summary, state)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

from .memory import ConversationMemory
from .pii import identify_customer, mask_pii
from .retrieval import format_chunks_for_prompt, retrieve_chunks

load_dotenv()

BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "system_prompts"
CUSTOMERS_PATH = Path(__file__).parents[2] / "pii_data" / "customers.json"

MODEL_NAME = "gemini-2.5-flash-lite"

log = logging.getLogger(__name__)


# Low-level Gemini helper

def _call_gemini(client: genai.Client, system_prompt: str, user_message: str) -> str:
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.2,
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config=config,
    )
    return getattr(response, "text", None) or ""


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:])
    if text.endswith("```"):
        text = "\n".join(text.splitlines()[:-1])
    return text.strip()


# Pipeline steps

def _rewrite_query(client: genai.Client, user_message: str, memory: ConversationMemory) -> str:
    system_prompt = _load_prompt("query_rewriter.txt")
    context = (
        f"CONVERSATION SUMMARY:\n{memory.summary or 'None'}\n\n"
        f"RECENT HISTORY:\n{memory.format_history_for_prompt()}\n\n"
        f"CURRENT STATE: intent={memory.state['current_intent']}, topic={memory.state['current_topic']}\n\n"
        f"USER MESSAGE:\n{user_message}"
    )
    rewritten = _call_gemini(client, system_prompt, context)
    return rewritten.strip() or user_message


def _classify_intent(client: genai.Client, query: str) -> tuple[str, float, str]:
    system_prompt = _load_prompt("intent_classifier.txt")
    raw = _call_gemini(client, system_prompt, query)
    clean = _strip_fences(raw)
    try:
        data = json.loads(clean)
        return (
            data.get("intent", "unsupported"),
            float(data.get("confidence", 0.0)),
            data.get("reasoning", ""),
        )
    except json.JSONDecodeError:
        return "unsupported", 0.0, raw


def _generate_answer(
    client: genai.Client,
    original_question: str,
    intent: str,
    chunks: list[dict],
    memory: ConversationMemory,
) -> str:
    system_prompt = _load_prompt("answer_generator.txt")
    context_text = format_chunks_for_prompt(chunks)
    user_content = (
        f"INTENT: {intent}\n\n"
        f"CONVERSATION SUMMARY:\n{memory.summary or 'None'}\n\n"
        f"RETRIEVED DOCUMENTS:\n{context_text}\n\n"
        f"USER QUESTION:\n{original_question}"
    )
    return _call_gemini(client, system_prompt, user_content)


def _guard_output(
    client: genai.Client,
    question: str,
    draft_answer: str,
    chunks: list[dict],
) -> str:
    system_prompt = _load_prompt("output_guard.txt")
    source_text = format_chunks_for_prompt(chunks)
    user_content = (
        f"USER_QUESTION:\n{question}\n\n"
        f"DRAFT_ANSWER:\n{draft_answer}\n\n"
        f"SOURCE_DOCUMENTS:\n{source_text}"
    )
    raw = _call_gemini(client, system_prompt, user_content)
    clean = _strip_fences(raw)
    try:
        result = json.loads(clean)
        if not result.get("safe", True):
            log.warning("Output guard flagged issues: %s", result.get("issues", []))
        return result.get("answer", draft_answer)
    except json.JSONDecodeError:
        return draft_answer


def _refresh_summary(client: genai.Client, memory: ConversationMemory) -> None:
    system_prompt = _load_prompt("conversation_summarizer.txt")
    history_text = memory.format_history_for_prompt()
    prompt_with_history = system_prompt.replace("{history}", history_text)
    new_summary = _call_gemini(client, "", prompt_with_history)
    memory.update_summary(new_summary.strip())


# Public API

class ChatbotPipeline:
    def __init__(self, conn, model: SentenceTransformer) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)
        self.conn = conn
        self.model = model
        self.memory = ConversationMemory()

    def process(self, user_input: str) -> str:
        # Step 1: sanitize PII
        sanitized, pii_found = mask_pii(user_input)
        log.debug("PII found: %s", list(pii_found.keys()))

        # Step 2: history already in memory

        # Step 3: identify customer
        customer_result = identify_customer(pii_found, CUSTOMERS_PATH)
        case_id = customer_result[0] if customer_result else None
        if case_id:
            log.debug("Customer identified: %s", case_id)

        # Step 4: rewrite query (use sanitized version for safety)
        rewritten = _rewrite_query(self.client, sanitized, self.memory)
        log.debug("Rewritten query: %s", rewritten)

        # Step 5: intent detection
        intent, confidence, reasoning = _classify_intent(self.client, rewritten)
        log.debug("Intent: %s (%.2f) — %s", intent, confidence, reasoning)

        # Step 6: retrieval
        chunks = retrieve_chunks(self.conn, self.model, rewritten, intent, case_id)
        log.debug("Retrieved %d chunk(s)", len(chunks))

        # Step 7: answer generation (use original question for natural phrasing)
        draft = _generate_answer(self.client, user_input, intent, chunks, self.memory)

        # Step 8: output guard
        final_answer = _guard_output(self.client, user_input, draft, chunks)

        # Step 9: update memory
        self.memory.add_exchange(user_input, final_answer)
        self.memory.update_state(intent=intent, topic=rewritten[:80])
        if self.memory.needs_summary_update():
            _refresh_summary(self.client, self.memory)

        return final_answer
