# documents/

Put the `.md` files you want to ingest here. `01_ingest_documents.py` reads
every `*.md` file in this folder (recursively), so you can organize them
into subfolders if you like.

This is example content for the pipeline — feel free to edit these files,
add your own, or delete everything here and replace it completely. Nothing
in this folder is read by any script except `01_ingest_documents.py`.

## How to write good RAG source files

- **One topic per file (or per clear section).** Each file gets chunked
  independently, so a file mixing unrelated topics makes chunks less
  focused and retrieval less precise.
- **Use real markdown structure.** Headings (`#`, `##`), short paragraphs,
  and lists all help the chunkers find sensible split points — especially
  the `recursive` strategy, which splits on headings and blank lines first.
- **Keep paragraphs self-contained.** Write each paragraph so it makes
  sense on its own, since a chunk may be retrieved without its surrounding
  context.
- **Avoid huge walls of text.** A 5,000-word file with no headings or
  paragraph breaks will get hard-split mid-thought. Break long documents
  into sections with headings.
- **Plain prose over tables/images where possible.** Tables, images, and
  complex formatting don't carry over well into chunked text embeddings.
  If you have tabular data (e.g. pricing tiers), consider also restating
  the key facts as plain sentences.
- **FAQs work well as one Q&A pair per heading**, e.g.:

  ```markdown
  ## Can I cancel my contract early?

  Yes, contracts can be cancelled with 30 days' written notice...
  ```

- **Filenames matter for traceability.** The chatbot shows the source
  `file_name` alongside retrieved chunks, so name files descriptively
  (e.g. `early-termination-policy.md`, not `doc3.md`).

## Example use case

A common way to use this pipeline: a fake company's internal knowledge
base, e.g. a leasing company's policy documents and FAQs (`leasing-terms.md`,
`early-termination-policy.md`, `faq-payments.md`, ...). Each file covers
one policy or FAQ topic, written in plain markdown with headings.

After adding or editing files here, run:

```
python 01_ingest_documents.py
```
