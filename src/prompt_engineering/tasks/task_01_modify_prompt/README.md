# Task 01 — Email Classification, Routing & Summarisation

You will engineer a system prompt that instructs the model to:
1. **Classify** a customer email into one of the defined categories
2. **Route** it — identify which team should handle it
3. **Summarise** the content

Do not touch `main.py`. All your work goes into `system_prompt.txt`.

---

## Categories

| Category | Description |
|----------|-------------|
| `prigovor` | Customer complaint about a product or service |
| `zahtjev za povrat` | Request for a refund or reversal of a charge |
| `upit o računu` | Question about an invoice, charge, or payment |
| `tehnički problem` | Technical issue with a product or system |
| `promjena korisničkih podataka` | Request to update personal or contact information |
| `ostalo` | Anything that does not fit the above |

---

## How It Works

```
system_prompt.txt   ← your prompt (edit this)
user_prompt.txt     ← sample customer email (fixed for now)
      │
      ▼
main.py             ← reads both files, calls the LLM, prints the response
```

---

## Run

```bash
# With Gemini (default)
python src/prompt_engineering/tasks/task_01_modify_prompt/main.py

# With Claude
LLM_PROVIDER=claude python src/prompt_engineering/tasks/task_01_modify_prompt/main.py
```

---

## Exercises

Work through these steps in order. Run the script after each change.

**Step 1 — Baseline**

Run the script as-is with the starter `system_prompt.txt`. Note:
- Is the category correct?
- Is the summary useful?
- Is the format consistent and easy to read?

**Step 2 — Add routing**

Update `system_prompt.txt` to also output which team should handle the email. For example:

```
...
3. State which team should handle it: Billing, Technical Support, Customer Relations, or Back Office.
```

**Step 3 — Enforce a structured output**

Make the model return a fixed format every time. For example:

```
Always respond in this exact format:
Category: <category>
Team: <team>
Summary: <one sentence>
```

**Step 4 — Add an urgency flag**

Extend the output with an urgency level (`low`, `medium`, `high`) based on the tone and content of the email.

**Step 5 — Few-shot example**

Add one worked example directly into `system_prompt.txt` to show the model exactly what you expect:

```
Example:
Email: "Želim promijeniti svoju adresu e-pošte u sustavu."
Category: promjena korisničkih podataka
Team: Back Office
Summary: Customer requests an update to their email address.
Urgency: low
```

Then run with the original email. Does the format become more consistent?

**Step 6 — Test with a different email**

Edit `user_prompt.txt` with a different customer message and verify that your prompt handles it correctly.

---

## What to Look For

| What you change | What typically changes |
|-----------------|----------------------|
| Adding a format rule | Output becomes more predictable and parseable |
| Adding routing | Model must reason about category → responsibility |
| Adding urgency | Model reads tone, not just topic |
| Adding a few-shot example | Format consistency improves significantly |
