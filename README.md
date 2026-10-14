# distill

A deterministic extractive summarizer in Rust. It picks the most representative
sentences from a document and prints them **verbatim**, in their original order.

- **Cannot hallucinate.** No text is generated. Every output sentence is a
  sentence from the input (whitespace-normalised).
- **Deterministic.** Same input and same binary give byte-identical output.
- **Local.** One binary plus a 90 MB model file. No network at run time.

A small neural model (all-MiniLM-L6-v2) is used, but only to turn each sentence
into a vector for *ranking*. It never writes anything.

## Build and run

```sh
./fetch-model.sh                 # one-time, ~90 MB
cargo build --release
./target/release/distill samples/apache-license.txt
```

On Apple Silicon, `cargo build --release --features accelerate` uses Apple's
Accelerate framework for the matrix maths. (Standard candle feature, but not
tested here: this was built and tested on Linux.)

Input is plain text or Markdown, from a file or stdin. For PDFs, convert first:
`pdftotext -layout paper.pdf - | distill --dehyphenate`.

```
distill [FILE]            summarize FILE (or stdin)
  -n, --sentences N       number of sentences (default: ~10% of the document, 3 to 15)
  -r, --ratio R           or a fraction of the document, e.g. 0.05
      --json              sentence positions, sections and scores
      --show-segments     print how the text was split, without summarizing
      --no-repair         don't add context sentences
      --no-headings       don't print section headings
      --dehyphenate       rejoin words broken across lines (PDF text)
  -v, --verbose           counts and timings
      --lambda, --lead, --backward, --beta, --section-weight, --penalty, --boost
                          ranking knobs; see `distill --help`
```

In the text output, sentences that were adjacent in the source share a line. A
new line means text was skipped in between.

The model folder is looked for in `--model-dir`, `$DISTILL_MODEL`, `./model`,
next to the binary, then `~/.distill/model`. Set `DISTILL_CACHE=<folder>` to
reuse embeddings when re-running on the same document with different options.

## How it works

1. **Segment** (`src/segment.rs`). Detects headings (Markdown, numbered,
   ALL CAPS, `SEC. 2. TITLE.`, `12. Governing Law.`), paragraphs, lists and
   sentences. The sentence splitter is rule-based with abbreviation handling
   for academic and legal text (`et al.`, `Fig. 3`, `42 U.S.C. § 1983`,
   `Smith v. Jones`). A short list is folded into the sentence that introduces
   it, so "X means-- (A) ..., (B) ..., and (C) ..." stays one unit.
2. **Embed** (`src/embed.rs`). Each candidate sentence becomes a 384-dimension
   vector.
3. **Rank** (`src/select.rs`). A sentence scores highly when the rest of the
   document echoes it, with later text counting more than earlier text. That
   one asymmetry captures "important things are stated first, then elaborated".
   Sentences that lean on other text ("However...", "see Figure 2") are
   penalised.
4. **Select.** Each top-level section gets a share of the budget, then
   sentences are picked greedily with MMR (relevance minus similarity to what
   is already picked), so the summary does not repeat itself.
5. **Repair.** If a pick opens with "This", "They", "However" and so on, the
   sentence before it is pulled in. A list item brings its lead-in.

## What was measured

Scored against human-written reference summaries with a simple ROUGE (no
stemming, so only comparable within this table):

| Set | Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| 100 news articles (CNN/DailyMail test), 3 sentences | first 3 sentences | 29.1 | 11.0 | 19.9 |
| | plain centroid ranking | 26.3 | 8.3 | 17.9 |
| | **distill defaults** | 29.4 | 11.7 | 20.5 |
| 60 US bills (BillSum test), 5 sentences | first 5 sentences | 37.7 | 16.8 | 24.4 |
| | plain centroid ranking | 41.0 | 18.5 | 25.7 |
| | **distill defaults** | 41.4 | 19.8 | 26.2 |

"First N sentences" is a famously strong baseline on news; matching it without
being told the text is news is the point. On the bills, distill's five picks
averaged 217 words against 170 for the first five sentences, which flatters it
somewhat. No research papers were scored, only read by eye.

Determinism was checked by hashing the output across repeated runs and across
1, 2 and 4 threads: identical. Across different CPU types the embeddings could
differ in the last float digits, which would only matter if two sentences were
tied to about seven decimal places.

## Limits

- **Extractive only.** It can leave things out, but it cannot rephrase, merge
  two sentences, or explain a term. There is no simplification step yet.
- **Legal text is the weakest case.** Long nested clauses are extracted whole,
  and a clause can depend on definitions elsewhere.
- **Sentence splitting is rule-based**, not a trained parser. Check odd input
  with `--show-segments`.
- **English only** (abbreviation lists, dangling-reference words, the model).
- **Speed is bounded by the embedding model**, not by Rust: roughly 60
  sentences per second on two slow cloud cores, so about 1 s for a news
  article and 3 s for a 3,700-word paper there. Everything else takes
  milliseconds.
- Sentences over 256 word-pieces are ranked on their first 256.
- Garbage in, garbage out: boilerplate left in the file (site navigation,
  licence footers) can be selected.

## Tests

`cargo test --release` runs the splitter, heading, list and selection tests.

## Chrome extension

Highlight text on any page, right-click, choose **Distill**, and the selection is
replaced with its key sentences. The selection is tinted while it works, and a
toast shows how many words were cut. Click the distilled text to compare it with
the original, and press Ctrl+Z / Cmd+Z (or the toast's Undo button) to put the
original back. Works in text boxes too.

```sh
./install.sh   # fetches the model, builds, registers the native helper with Chrome
```

Then open `chrome://extensions`, turn on Developer mode, click **Load unpacked**
and pick the `extension/` folder. Reload any tabs that were already open.

**PDFs.** Chrome's built-in PDF viewer can't be edited by extensions, so using
Distill there reopens the PDF in the Distill reader (same tab) and distills the
same passage. The reader shows the PDF as plain reflowed text (via PDF.js, in
`extension/vendor`), where everything above works. You can also click the
toolbar icon on a PDF, or right-click a PDF link and choose "Open PDF in Distill
reader". For local `file://` PDFs, turn on "Allow access to file URLs" in the
extension's details. Scanned PDFs without a text layer won't work.

The extension can't run the model itself, so it hands the text to
`target/release/distill-host` through Chrome native messaging, which runs
`distill` locally. Nothing leaves your machine.
