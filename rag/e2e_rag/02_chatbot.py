"""
02_chatbot.py — interactive RAG chatbot over the documents ingested by 01_ingest_documents.py.

Supports both Gemini and Claude as the answer-generation provider (same
LLM_PROVIDER / try-except-import pattern used across this repo, e.g.
src/llm_chaining/main.py and evaluation/judge_llm.py).

Keeps a short conversation memory so a quick follow-up question
("what about X?") can be rewritten into a standalone query before
retrieval. The memory strategy is selectable with --memory-strategy:

  sliding_window   Keep only the last MEMORY_WINDOW_EXCHANGES exchanges
                    verbatim (default). Oldest exchanges are dropped.
  token_cutoff      Keep verbatim exchanges (most recent first) until adding
                    another one would exceed MEMORY_TOKEN_LIMIT (approximate
                    tokens, no model-specific tokenizer needed).
  summarization     Keep only the last exchange verbatim, plus a running
                    LLM-generated summary of everything older, refreshed
                    every MEMORY_SUMMARY_EVERY exchanges.

Usage:
    python 02_chatbot.py
    python 02_chatbot.py --memory-strategy token_cutoff
    LLM_PROVIDER=claude python 02_chatbot.py --memory-strategy summarization
"""

from __future__ import annotations

import argparse
import os
from collections import deque

from dotenv import load_dotenv
from psycopg import connect
# SentenceTransformer converts text into a numeric vector (embedding). The same model
# must be used here as in 01_ingest_documents.py — only then do question embeddings
# and chunk embeddings live in the same vector space and can be meaningfully compared.
from sentence_transformers import SentenceTransformer

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

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-m3"  # EDITABLE: must match the model used in 01_ingest_documents.py
TOP_K = 4  # EDITABLE: how many chunks to retrieve per question

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # EDITABLE: "gemini" or "claude" (or set LLM_PROVIDER in .env)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")  # EDITABLE: or set GEMINI_MODEL in .env
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")  # EDITABLE: or set CLAUDE_MODEL in .env

# --- sliding_window settings ---
# How many past exchanges to keep verbatim. 1 = only the most recent
# user/assistant pair is remembered; older turns are forgotten.
MEMORY_WINDOW_EXCHANGES = 1  # EDITABLE: number of exchanges to remember verbatim

# --- token_cutoff settings ---
# Approximate token budget for the verbatim history kept in the prompt.
# Token count is estimated as len(text) / 4 (roughly 4 chars per token for
# English text) — good enough for a budget cutoff without adding a
# model-specific tokenizer dependency.
MEMORY_TOKEN_LIMIT = 200  # EDITABLE: approximate token budget for history

# --- summarization settings ---
# Only the most recent exchange is kept verbatim; everything older is
# folded into a running summary, refreshed every N exchanges.
MEMORY_SUMMARY_EVERY = 3  # EDITABLE: how often (in exchanges) to refresh the summary

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the
provided context documents. If the answer is not contained in the context, say you don't know
instead of guessing. Be concise."""
# EDITABLE: ^ the system prompt used for answer generation — change the persona/rules here

REWRITE_SYSTEM_PROMPT = """Rewrite the user's latest message into a standalone question that
makes sense without the prior conversation. If it is already standalone, return it unchanged.
Reply with the rewritten question only, no extra text."""
# EDITABLE: ^ the system prompt used for query rewriting

SUMMARY_SYSTEM_PROMPT = """Update the running conversation summary given the prior summary and
the new exchanges below. Keep it brief (2-4 sentences) and only include facts useful for
understanding follow-up questions. Reply with the updated summary only, no extra text."""
# EDITABLE: ^ the system prompt used to refresh the running summary (summarization strategy only)


# --- LLM provider setup ---------------------------------------------------
# Same dual-provider pattern used in src/llm_chaining/main.py and
# evaluation/judge_llm.py: pick the client based on LLM_PROVIDER, and route
# each call through one function that knows how to talk to either SDK.

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


def call_llm(client, system_prompt: str, user_message: str, temperature: float = 0.2) -> str:  # EDITABLE: temperature default
    if PROVIDER == "claude":
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,  # EDITABLE: max response length (Claude)
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text if response.content else ""
    else:
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config=config,
        )
        return getattr(response, "text", None) or ""


# --- Memory ----------------------------------------------------------------
# Three interchangeable memory strategies, all exposing the same interface
# used by the pipeline: add_exchange(user, assistant) and format_for_prompt().

def _format_history(history) -> str:
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


class SlidingWindowMemory:
    """Keeps only the last `window_exchanges` exchanges verbatim. Simplest
    strategy: a fixed-size deque automatically drops the oldest turn once full."""

    def __init__(self, window_exchanges: int = MEMORY_WINDOW_EXCHANGES) -> None:  # EDITABLE: default exchange count
        self.history: deque = deque(maxlen=window_exchanges * 2)

    def add_exchange(self, user_msg: str, assistant_msg: str) -> None:
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})

    def format_for_prompt(self) -> str:
        return _format_history(self.history) if self.history else "No prior conversation."


class TokenCutoffMemory:
    """Keeps as many of the most recent exchanges as fit under an approximate
    token budget. Unlike a fixed exchange count, this adapts to message
    length — a handful of short exchanges fit, but long ones get dropped sooner."""

    def __init__(self, token_limit: int = MEMORY_TOKEN_LIMIT) -> None:  # EDITABLE: default token budget
        self.token_limit = token_limit
        self._exchanges: list[tuple[str, str]] = []  # oldest first

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # ~4 characters per token is a common rough estimate for English text.
        return max(1, len(text) // 4)  # EDITABLE: chars-per-token ratio

    def add_exchange(self, user_msg: str, assistant_msg: str) -> None:
        self._exchanges.append((user_msg, assistant_msg))

    def format_for_prompt(self) -> str:
        if not self._exchanges:
            return "No prior conversation."

        # Walk backwards from the most recent exchange, keeping whole
        # exchanges until the token budget would be exceeded.
        kept: list[dict] = []
        budget = self.token_limit
        for user_msg, assistant_msg in reversed(self._exchanges):
            exchange_tokens = self._estimate_tokens(user_msg) + self._estimate_tokens(assistant_msg)
            if kept and exchange_tokens > budget:
                break
            budget -= exchange_tokens
            kept.append({"role": "assistant", "content": assistant_msg})
            kept.append({"role": "user", "content": user_msg})

        kept.reverse()
        return _format_history(kept)


class SummarizationMemory:
    """Keeps only the latest exchange verbatim; older exchanges are folded
    into a running LLM-generated summary, refreshed every `summary_every`
    exchanges (mirrors src/chatbot/memory.py's summary layer)."""

    def __init__(
        self,
        client,
        summary_every: int = MEMORY_SUMMARY_EVERY,  # EDITABLE: default refresh interval
    ) -> None:
        self.client = client
        self.summary_every = summary_every
        self.summary = ""
        self.latest_exchange: deque = deque(maxlen=2)
        self._pending: list[tuple[str, str]] = []  # exchanges not yet folded into the summary
        self._exchange_count = 0

    def add_exchange(self, user_msg: str, assistant_msg: str) -> None:
        self.latest_exchange.clear()
        self.latest_exchange.append({"role": "user", "content": user_msg})
        self.latest_exchange.append({"role": "assistant", "content": assistant_msg})

        self._pending.append((user_msg, assistant_msg))
        self._exchange_count += 1

        if self._exchange_count % self.summary_every == 0:
            self._refresh_summary()

    def _refresh_summary(self) -> None:
        pending_text = _format_history(
            [
                {"role": role, "content": content}
                for user_msg, assistant_msg in self._pending
                for role, content in (("user", user_msg), ("assistant", assistant_msg))
            ]
        )
        prior = f"PRIOR SUMMARY:\n{self.summary}\n\n" if self.summary else ""
        context = f"{prior}NEW EXCHANGES:\n{pending_text}"
        new_summary = call_llm(self.client, SUMMARY_SYSTEM_PROMPT, context)
        self.summary = new_summary.strip() or self.summary
        self._pending.clear()

    def format_for_prompt(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"CONVERSATION SUMMARY:\n{self.summary}")
        if self.latest_exchange:
            parts.append(f"MOST RECENT EXCHANGE:\n{_format_history(self.latest_exchange)}")
        return "\n\n".join(parts) if parts else "No prior conversation."


MEMORY_STRATEGIES = {
    "sliding_window": SlidingWindowMemory,
    "token_cutoff": TokenCutoffMemory,
    "summarization": SummarizationMemory,
}


def build_memory(strategy: str, client):
    """Construct the selected memory strategy. `summarization` needs the LLM
    client since it calls the model to refresh its running summary."""
    if strategy == "summarization":
        return SummarizationMemory(client)
    return MEMORY_STRATEGIES[strategy]()


# --- Retrieval ---------------------------------------------------------
# Plain vector search over the single `chunks` table created by 00_init_db.py
# (no doc_type filtering since e2e_rag has one flat collection of documents).

def retrieve_chunks(database_url: str, model: SentenceTransformer, query: str, top_k: int = TOP_K) -> list[dict]:  # EDITABLE: default top_k
    embedding = model.encode(query, normalize_embeddings=True).tolist()
    # Open a fresh connection per query — Neon serverless suspends idle connections,
    # so a long-lived connection held across user input time will go stale.
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, chunk_text, file_name, chunk_order,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (embedding, embedding, top_k),
            )
            rows = cur.fetchall()

    return [
        {"id": row[0], "chunk_text": row[1], "file_name": row[2], "chunk_order": row[3], "similarity": float(row[4])}
        for row in rows
    ]


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant documents found."
    parts = [f"[{i}] ({c['file_name']})\n{c['chunk_text']}" for i, c in enumerate(chunks, 1)]
    return "\n\n---\n\n".join(parts)


# --- Pipeline steps ---------------------------------------------------

_NO_HISTORY_SENTINEL = "No prior conversation."


def rewrite_query(client, user_message: str, memory) -> str:
    """Turn a follow-up message into a standalone query using recent memory."""
    history_text = memory.format_for_prompt()
    if history_text == _NO_HISTORY_SENTINEL:
        return user_message
    context = f"RECENT CONVERSATION:\n{history_text}\n\nLATEST MESSAGE:\n{user_message}"
    rewritten = call_llm(client, REWRITE_SYSTEM_PROMPT, context)
    return rewritten.strip() or user_message


def generate_answer(client, question: str, chunks: list[dict], memory) -> tuple[str, str]:
    """Returns (answer, final_prompt) so the caller can print debug info."""
    context_text = format_chunks_for_prompt(chunks)
    history_text = memory.format_for_prompt()
    history_section = "" if history_text == _NO_HISTORY_SENTINEL else f"CONVERSATION HISTORY:\n{history_text}\n\n"
    user_content = f"{history_section}CONTEXT DOCUMENTS:\n{context_text}\n\nQUESTION:\n{question}"
    return call_llm(client, ANSWER_SYSTEM_PROMPT, user_content), user_content


# --- Main loop ---------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive RAG chatbot.")
    parser.add_argument(
        "--memory-strategy",
        choices=sorted(MEMORY_STRATEGIES),
        default="sliding_window",  # EDITABLE: default memory strategy
        help="Conversation memory strategy to use (default: sliding_window).",
    )
    args = parser.parse_args()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill in your Neon connection string."
        )

    print(f"Provider: {PROVIDER.upper()}")
    client = build_client()

    print("Loading embedding model…")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    memory = build_memory(args.memory_strategy, client)
    print(f"Memory strategy: {args.memory_strategy}")

    print("\nRAG chatbot ready. Type your question, or /quit to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye.")
            break

        # Step 1: rewrite follow-up questions into standalone queries using memory
        query = rewrite_query(client, user_input, memory)

        # Step 2: retrieve relevant chunks (reconnects each time for Neon compatibility)
        chunks = retrieve_chunks(database_url, embedding_model, query)

        # Step 3: generate a grounded answer (history included for conversational context)
        answer, final_prompt = generate_answer(client, user_input, chunks, memory)

        # --- debug info between user message and assistant response ---
        print("\n" + "=" * 60)
        print(f"[RETRIEVED CHUNKS — {len(chunks)} total]")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] id={c['id']}  {c['file_name']} (chunk {c['chunk_order']}, similarity {c['similarity']:.3f}, {len(c['chunk_text'])} chars)")
        print()
        memory_text = memory.format_for_prompt()
        memory_chars = len(memory_text) if memory_text != _NO_HISTORY_SENTINEL else 0
        print(f"[MEMORY — {memory_chars} chars]")
        if memory_chars:
            print(memory_text)
        print()
        print(f"[FINAL PROMPT TO LLM — {len(final_prompt)} chars]")
        print("=" * 60 + "\n")

        print(f"Assistant: {answer}\n")

        # Step 4: update memory (behavior depends on the selected --memory-strategy)
        memory.add_exchange(user_input, answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
