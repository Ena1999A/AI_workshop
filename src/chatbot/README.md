# Leasing Support Chatbot

An AI-powered CLI chatbot for leasing company support. It answers questions by doing semantic search over an internal knowledge base stored in PostgreSQL, using Google Gemini as the language model.

> Before running the chatbot, the database must be initialized and filled with knowledge base documents.
> See [`db/README.md`](../../db/README.md) for those steps.

---

## Prerequisites

- Python 3.10+
- A running PostgreSQL instance with pgvector and the knowledge base already ingested
- A `GEMINI_API_KEY` — get one free at [Google AI Studio](https://aistudio.google.com)
- `.env` file at the project root with at minimum:

```env
GEMINI_API_KEY=your_key_here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ai_workshop
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

---

## Running the Chatbot

From the project root:

```bash
python -m src.chatbot.main
```

All settings are read from `.env` by default. You can override any of them via flags:

```bash
python -m src.chatbot.main \
  --model BAAI/bge-m3 \
  --device cpu \
  --db-host localhost \
  --db-port 5432 \
  --db-name ai_workshop \
  --db-user postgres \
  --db-password postgres
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `BAAI/bge-m3` | SentenceTransformer model used to embed queries |
| `--device` | auto | `cpu` or `cuda` |
| `--db-host/port/name/user/password` | from `.env` | PostgreSQL connection |

---

## CLI Commands

Type your question and press Enter. Special commands available at any time:

| Command    | What it shows |
|------------|---------------|
| `/history` | The last 5 conversation exchanges |
| `/state`   | Current detected intent, topic, and conversation summary |
| `/quit`    | Exit (also `/exit`, `quit`, `exit`) |

After every response, the chatbot automatically prints:
- The detected intent and rewritten query
- All retrieved knowledge base chunks with similarity scores
- The full conversation history
- The current conversation summary

---

## How a Message Is Processed (Step by Step)

Every user message goes through 9 steps defined in `pipeline.py`:

```
User message
     │
     ▼
1. PII Masking       → replace personal data with placeholders before any LLM call
     │
     ▼
2. Memory Fetch      → load recent chat history + conversation summary
     │
     ▼
3. Customer Lookup   → match extracted PII against the customer registry
     │
     ▼
4. Query Rewriting   → turn vague follow-ups into standalone search queries
     │
     ▼
5. Intent Detection  → classify what kind of question this is (6 possible intents)
     │
     ▼
6. Retrieval         → semantic vector search, filtered by intent
     │
     ▼
7. Answer Generation → Gemini writes an answer grounded in retrieved documents
     │
     ▼
8. Output Guard      → safety check: PII leak, hallucination, ungrounded claims
     │
     ▼
9. Memory Update     → store exchange; refresh summary every 3 turns
     │
     ▼
Final answer returned to user
```

---

## File Reference

### `main.py` — Entry Point

Sets up logging, parses CLI arguments, connects to the database, loads the embedding model, and runs the interactive input loop.

| Function | What it does |
|----------|-------------|
| `main()` | Parses args, boots all dependencies, starts the REPL loop |
| `_print_history(pipeline)` | Prints the last 5 exchanges to the terminal |
| `_print_chunks(pipeline)` | Prints retrieved chunks with intent, similarity scores, and doc type |
| `_print_state(pipeline)` | Prints current intent, topic, and conversation summary |

---

### `pipeline.py` — Core Pipeline

Orchestrates all 9 steps for a single turn. Holds the Gemini client, DB connection, embedding model, and memory object.

| Function | What it does |
|----------|-------------|
| `ChatbotPipeline.process(user_input)` | Runs all 9 steps, returns the final answer string |
| `_call_gemini(client, system_prompt, user_message)` | Low-level wrapper — sends one request to Gemini and returns the text response |
| `_load_prompt(name)` | Reads a `.txt` file from `system_prompts/` |
| `_strip_fences(text)` | Removes markdown code fences from LLM output before JSON parsing |
| `_rewrite_query(client, user_message, memory)` | Step 4 — rewrites vague follow-ups into full standalone search queries |
| `_classify_intent(client, query)` | Step 5 — returns `(intent, confidence, reasoning)` parsed from Gemini's JSON |
| `_generate_answer(client, question, intent, chunks, memory)` | Step 7 — builds the prompt with retrieved docs and generates the answer |
| `_guard_output(client, question, draft_answer, chunks)` | Step 8 — checks draft for PII leaks, hallucinations, ungrounded claims; returns safe version |
| `_refresh_summary(client, memory)` | Step 9 — called every 3 turns; builds on the prior summary + recent exchanges to produce an updated one |

After each `process()` call, the pipeline also stores `last_chunks`, `last_intent`, and `last_rewritten` on `self` so `main.py` can print them.

---

### `memory.py` — Conversation Memory

Three-layer in-memory store that persists within a session (not saved to disk between runs).

| Layer | What it stores | Capacity |
|-------|---------------|----------|
| `raw_history` | Last 10 messages (5 user + 5 assistant) | Rolling deque — oldest messages are dropped |
| `summary` | LLM-generated plain-text summary | Refreshed every 3 turns, builds on the previous summary |
| `state` | `current_intent`, `current_topic`, `referenced_entities`, `user_preference` | Updated each turn |

| Function | What it does |
|----------|-------------|
| `add_exchange(user_msg, assistant_msg)` | Appends one turn to the history deque and increments the exchange counter |
| `get_recent_history()` | Returns history as a list of `{"role": ..., "content": ...}` dicts |
| `format_history_for_prompt()` | Formats history as plain text for inclusion in LLM prompts |
| `needs_summary_update()` | Returns `True` every 3 exchanges — used to trigger `_refresh_summary` |
| `update_summary(new_summary)` | Stores the latest LLM-generated summary |
| `update_state(intent, topic)` | Updates `current_intent` and `current_topic` after each turn |

---

### `pii.py` — PII Detection and Masking

Detects Croatian personal data with regex and replaces it with placeholders **before any text is sent to an LLM**. The original values are saved separately for customer lookup.

| Field | Placeholder | Example |
|-------|-------------|---------|
| OIB (personal ID number) | `[OIB]` | `12345678901` |
| IBAN | `[IBAN]` | `HR1210010051863000160` |
| Email | `[EMAIL]` | `ana.horvat@gmail.com` |
| Phone | `[TELEFON]` | `091 234 5678` |
| Date of birth | `[DATUM]` | `15.03.1985` |
| License plate | `[REG_OZNAKA]` | `ZG 123 AB` |
| Contract number | `[BROJ_UGOVORA]` | `UG-2024-00042` |

| Function | What it does |
|----------|-------------|
| `mask_pii(text)` | Returns `(masked_text, found_dict)`. Masked text is safe to send to Gemini. Dict contains the original extracted values keyed by field name. |
| `identify_customer(pii_found, customers_path)` | Loads `pii_data/customers.json` and matches any extracted PII field against registered customers. Returns `(case_id, customer_dict)` or `None`. |

---

### `retrieval.py` — Vector Search

Runs semantic search against the `chunks` table in PostgreSQL using pgvector. Results are filtered by the intent from step 5.

| Intent | Searches in |
|--------|-------------|
| `faq` | FAQ documents |
| `crm_case` | CRM case records |
| `contract_concept` | Concept/glossary documents |
| `procedure` | Procedure documents |
| `product_doc` | Product/module documentation |
| `unsupported` | Nothing — returns empty list |

| Function | What it does |
|----------|-------------|
| `retrieve_chunks(conn, model, query, intent, case_id, top_k)` | Encodes the query into a vector, then: (1) if `case_id` is set and intent is `crm_case`, fetches that customer's case directly by `file_stem`; (2) runs cosine-distance vector search filtered by `doc_type`. Returns up to `top_k` (default 4) chunks. |
| `format_chunks_for_prompt(chunks)` | Formats the chunk list as numbered text blocks for inclusion in the answer-generation prompt |

---

## System Prompts

Each LLM call uses a dedicated prompt file from `system_prompts/`. Keeping them in separate files makes them easy to edit and experiment with without touching Python code.

| File | Pipeline step | What it instructs Gemini to do |
|------|--------------|-------------------------------|
| `query_rewriter.txt` | 4 | Rewrite vague follow-ups into standalone search queries using conversation context |
| `intent_classifier.txt` | 5 | Classify the query into one of 6 intents and return JSON |
| `answer_generator.txt` | 7 | Answer strictly from retrieved documents, in the user's language |
| `output_guard.txt` | 8 | Check draft for PII leaks, hallucinations, and ungrounded claims; return corrected JSON |
| `conversation_summarizer.txt` | 9 | Merge the prior summary with recent exchanges into an updated 3–5 sentence summary |

---

## Logging

Each run creates a new log file at:

```
src/chatbot/logs/chatbot_YYYYMMDD_HHMMSS.log
```

The log includes, at DEBUG level: rewritten queries, detected intents with confidence scores, number of retrieved chunks, whether the output guard flagged anything, and full user/assistant exchanges.
