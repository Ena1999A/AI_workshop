"""
Leasing support chatbot — interactive CLI.

Usage:
  export GEMINI_API_KEY="your_key"
  python -m src.chatbot.main

  # Optional overrides:
  python -m src.chatbot.main --model BAAI/bge-m3 --db-host localhost
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from psycopg import connect
from sentence_transformers import SentenceTransformer

from .pipeline import ChatbotPipeline

load_dotenv()

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"chatbot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

WELCOME = """
╔══════════════════════════════════════════════════════════════╗
║          Leasing Support Chatbot  — AI Workshop              ║
║  Type your question. Commands: /history  /state  /quit       ║
╚══════════════════════════════════════════════════════════════╝
"""


def _print_history(pipeline: ChatbotPipeline) -> None:
    history = pipeline.memory.get_recent_history()
    if not history:
        print("  (no history yet)")
        return
    for msg in history:
        role = "You      " if msg["role"] == "user" else "Assistant"
        print(f"  {role}: {msg['content']}")


def _print_state(pipeline: ChatbotPipeline) -> None:
    state = pipeline.memory.state
    summary = pipeline.memory.summary or "(none yet)"
    print(f"  Intent  : {state['current_intent']}")
    print(f"  Topic   : {state['current_topic']}")
    print(f"  Summary : {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leasing support chatbot.")
    parser.add_argument("--model", default="BAAI/bge-m3", help="SentenceTransformer model.")
    parser.add_argument("--device", default=None, help="torch device (cpu / cuda).")
    parser.add_argument("--db-host", default=None)
    parser.add_argument("--db-port", default=None)
    parser.add_argument("--db-name", default=None)
    parser.add_argument("--db-user", default=None)
    parser.add_argument("--db-password", default=None)
    args = parser.parse_args()

    db_host = args.db_host or os.getenv("POSTGRES_HOST", "localhost")
    db_port = args.db_port or os.getenv("POSTGRES_PORT", "5432")
    db_name = args.db_name or os.getenv("POSTGRES_DB", "ai_workshop")
    db_user = args.db_user or os.getenv("POSTGRES_USER", "postgres")
    db_password = args.db_password or os.getenv("POSTGRES_PASSWORD", "postgres")

    conninfo = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}"

    print("Loading embedding model…")
    embedding_model = SentenceTransformer(args.model, device=args.device)

    print("Connecting to database…")
    conn = connect(conninfo)

    pipeline = ChatbotPipeline(conn=conn, model=embedding_model)

    print(WELCOME)
    log.info("Chatbot started. Log: %s", log_file)

    try:
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
            if user_input.lower() == "/history":
                _print_history(pipeline)
                continue
            if user_input.lower() == "/state":
                _print_state(pipeline)
                continue

            log.info("USER: %s", user_input)

            answer = pipeline.process(user_input)
            print(f"\nAssistant: {answer}\n")

            log.info("ASSISTANT: %s", answer)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
