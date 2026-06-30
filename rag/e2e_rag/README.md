# e2e_rag — end-to-end RAG pipeline

A self-contained Retrieval-Augmented Generation pipeline: drop in markdown
documents, ingest them into a Postgres+pgvector database, then chat with
them. Three numbered scripts, run in order.

## 1. Install dependencies

> **Python version:** use Python **3.12**. Python 3.13 is not supported
> (PyTorch DLL incompatibility on Windows). On Ubuntu / WSL, Python 3.12
> works without any extra steps.

Open a terminal in VS Code (`Terminal → New Terminal`) and create a
virtual environment, then install dependencies.

**Windows (PowerShell terminal in VS Code):**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> On Windows, PyTorch must be installed from the PyTorch CPU index first
> (the line above). The default PyPI build includes CUDA libraries that
> won't load on machines without a GPU driver.

**Ubuntu / WSL (bash terminal in VS Code):**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After activating the venv, VS Code may prompt you to select it as the
workspace interpreter — click **Yes**. You can also set it manually:
open the Command Palette (`Ctrl+Shift+P`) → **Python: Select Interpreter**
→ pick the `.venv` entry.

## 2. Set up `.env`

```
cp .env.example .env
```

You'll be provided with a ready-to-use `.env` file containing:

- a shared `DATABASE_URL` pointing to a pre-configured Neon database
- working API keys for both Gemini and Claude

**Use those to run the pipeline as-is first.** Once everything works,
you're free to swap in your own keys or your own Neon project (see below).

The variables you can change:

| Variable | What it does |
|---|---|
| `DATABASE_URL` | Postgres connection string (Neon or any pgvector-enabled Postgres) |
| `LLM_PROVIDER` | `gemini` or `claude` — which LLM the chatbot uses |
| `GEMINI_API_KEY` | Required when `LLM_PROVIDER=gemini` |
| `ANTHROPIC_API_KEY` | Required when `LLM_PROVIDER=claude` |
| `GEMINI_MODEL` | Optional — override the default Gemini model |
| `CLAUDE_MODEL` | Optional — override the default Claude model |

### What is Neon?

[Neon](https://neon.tech) is a hosted, serverless Postgres provider with
pgvector support on its free tier — no local Docker/Postgres install
needed. To set up your own:

1. Sign up for a free account at [neon.tech](https://neon.tech).
2. Create a project.
3. In the project dashboard, open **Connection Details** and copy the
   connection string (`postgresql://user:password@...neon.tech/dbname?sslmode=require`).
4. Paste it into `DATABASE_URL` in your `.env`.

pgvector itself is enabled automatically by `00_init_db.py`
(`CREATE EXTENSION IF NOT EXISTS vector`) — no manual setup needed.

## 3. Run the pipeline

### Step 0 — create the schema

```
python 00_init_db.py
```

Creates the `chunks` table. Safe to re-run any time — by default it drops
and recreates the table, so re-run it whenever you change chunking
strategy or chunk size, to avoid mixing rows from different approaches.
Use `--no-drop` to keep existing data instead.

### Step 1 — ingest documents

Drop your `.md` files into `documents/` — edit the example files there,
add your own, or replace them completely. See `documents/README.md` for
guidance on how to structure source files for good retrieval. Then:

```
python 01_ingest_documents.py
python 01_ingest_documents.py --strategy fixed --max-chars 500 --overlap 50
python 01_ingest_documents.py --strategy recursive
```

`--strategy` chooses how text is split: `paragraph` (default,
paragraph-aware), `fixed` (naive sliding window), or `recursive`
(LangChain-style separator hierarchy). Re-running on the same file
replaces its previous chunks, so it's safe to re-ingest after editing a
document.

### Step 2 — chat

```
python 02_chatbot.py
python 02_chatbot.py --memory-strategy token_cutoff
LLM_PROVIDER=claude python 02_chatbot.py --memory-strategy summarization
```

`--memory-strategy` chooses how conversation history is kept:
`sliding_window` (default, last N exchanges verbatim), `token_cutoff`
(verbatim exchanges within an approximate token budget), or
`summarization` (latest exchange verbatim + running LLM summary).

Type `/quit` to exit.

`evaluation_qa.md` contains 20 ready-made questions you can paste into the
chatbot — covering questions with clear answers, ambiguous ones, and cases
where the chatbot should reply that it doesn't know. Good starting point if
you're not sure what to ask.

## Experimenting

Every constant worth tweaking is flagged with an `# EDITABLE:` comment in
the three scripts — LLM provider/model, chunking strategy and its
parameters, memory strategy and its parameters, system prompts, `top_k`,
temperature, and more. Search for `EDITABLE` to find them all.

**Suggested flow:** run the three scripts as-is first to see the whole
pipeline work end-to-end, then start changing whatever interests you
(swap providers, try a chunking strategy, tune memory) and re-run.
