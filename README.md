# distill

A deterministic extractive summarizer in Rust. It picks the most representative
sentences from a document and prints them **verbatim**, in their original order.

- **Cannot hallucinate.** No text is generated. Every output sentence is a
  sentence from the input (whitespace-normalised).
- **Deterministic.** Same input and same binary give byte-identical output.
- **Local.** One binary plus a 90 MB model file. No network at run time.

A small neural model (all-MiniLM-L6-v2) is used, but only to turn each sentence
into a vector for *ranking*. It never writes anything. A learned salience score,
a logistic regression over those vectors and a few position and length
features, helps decide which sentences matter. It is a few thousand fixed
numbers compiled into the binary.

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
  -w, --words N           or up to N words, context sentences included
      --json              sentence positions, sections and scores
      --show-segments     print how the text was split, without summarizing
      --no-repair         don't add context sentences
      --no-headings       don't print section headings
      --dehyphenate       rejoin words broken across lines (PDF text)
  -v, --verbose           counts and timings
      --salience W        weight of the learned salience score (default 0.5, 0 = off)
      --lambda, --lead, --backward, --beta, --section-weight, --penalty, --boost
                          ranking knobs; see `distill --help`
```

In the text output, sentences that were adjacent in the source share a line. A
new line means text was skipped in between.

The model folder is looked for in `--model-dir`, `$DISTILL_MODEL`, `./model`,
next to the binary, then `~/.distill/model`. Set `DISTILL_CACHE=<folder>` to
reuse embeddings when re-running on the same document with different options.
The cache is keyed by model as well as text, so switching models is safe.

### Other embedding models

`./fetch-model.sh NAME` downloads another pinned, checksum-verified model into
`./model-NAME`; use it with `--model-dir model-NAME`. The folder's
`distill.json` records the pooling, token limit and text prefix the model
needs. Any BERT-architecture sentence-embedding model works the same way.

| Name | Model | Size | Layers | Token limit | News ROUGE-2 | Bills ROUGE-2 | Embed time, bill / paper |
|---|---|---|---|---|---|---|---|
| `minilm` (default) | all-MiniLM-L6-v2 | 90 MB | 6 | 256 | 9.0 | 19.4 | 0.5 s / 2 s |
| `bge` | bge-small-en-v1.5 | 130 MB | 12 | 512 | 8.9 | 19.0 | 1 s / 4 s |
| `gte` | gte-small | 65 MB | 12 | 512 | 8.8 | 19.5 | 1 s / 4 s |
| `e5` | e5-small-v2 | 130 MB | 12 | 512 | 8.7 | 18.9 | 1 s / 4 s |

ROUGE-2 is on the development sets (500 news articles, 300 bills) at reference
length. None of the differences from the default is significant, including
after retuning the diversity and similarity-threshold knobs for each model, so
the default stays the smallest and fastest. Times are for the 1,662-word bill in
`samples/` and a 5,800-word arXiv paper, on two threads of an M-series Mac.

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
   penalised. Half of the centrality score is then replaced by a learned
   salience score (`src/salience.rs`): a logistic regression over the sentence
   vector plus position, length and dependence features, trained on which
   sentences best match human summaries of news articles and bills.
4. **Select.** Each top-level section gets a share of the budget, then
   sentences are picked greedily with MMR (relevance minus similarity to what
   is already picked), so the summary does not repeat itself.
5. **Repair.** If a pick opens with "This", "They", "However" and so on, the
   sentence before it is pulled in. A list item brings its lead-in.

## What was measured

Everything in this section is reproducible from the repo:

```sh
python3 eval/fetch_data.py   # downloads the frozen sample, checks it against eval/manifest.tsv
cargo build --release
python3 eval/score.py        # about 30 s on an M-series Mac
```

The sample is the first 100 CNN/DailyMail test articles, the first 60 BillSum
test bills and the first 10 arXiv test papers. The text is not committed, but
each document's SHA-256 is, so a different download fails instead of quietly
changing the scores. ROUGE is computed by `eval/score.py` with no stemming, so
the numbers are only comparable within this table.

Each set is scored twice. "N sentences" is the original setup. "ref length"
gives every method a word budget equal to that document's reference summary,
which removes the advantage of picking longer sentences. Under a word budget,
the context sentences distill adds for readability count against the budget.

| Set | Budget | Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Words |
|---|---|---|---|---|---|---|
| cnndm (n=100) | 3 sentences | lead | 29.4 | 12.3 | 21.0 | 65 |
|  |  | centroid | 26.7 | 9.0 | 17.9 | 76 |
|  |  | distill | **31.1** | **13.4** | **22.3** | **74** |
|  | ref length (34 w) | lead | 27.5 | 8.9 | 19.4 | 34 |
|  |  | centroid | 25.8 | 7.0 | 17.4 | 32 |
|  |  | distill | **27.7** | **9.6** | **19.7** | **32** |
| billsum (n=60) | 5 sentences | lead | 36.6 | 16.9 | 24.2 | 115 |
|  |  | centroid | 38.1 | 17.1 | 23.5 | 236 |
|  |  | distill | **41.6** | **21.9** | **27.8** | **222** |
|  | ref length (168 w) | lead | 42.9 | 19.7 | 27.4 | 166 |
|  |  | centroid | 41.2 | 16.3 | 24.6 | 164 |
|  |  | distill | **43.6** | **20.5** | **28.1** | **162** |
| arxiv (n=10) | 6 sentences | lead | 34.3 | 11.2 | 18.9 | 194 |
|  |  | centroid | 28.8 | 8.8 | 15.9 | 326 |
|  |  | distill | **37.5** | **14.3** | **21.2** | **232** |
|  | ref length (142 w) | lead | 36.0 | 11.5 | 20.1 | 142 |
|  |  | centroid | 36.0 | 10.9 | 21.5 | 140 |
|  |  | distill | **39.1** | **13.5** | **21.8** | **139** |

What this shows:

- **News:** distill beats the first sentences, a famously strong baseline on
  news, without being told the text is news. At equal length the margin is
  small (9.6 against 8.9 ROUGE-2). At three sentences it is larger, but partly
  because distill's sentences are longer.
- **Bills:** at equal length distill is nearly a ROUGE-2 point ahead of the
  first sentences, and well ahead of plain centroid ranking.
- **Papers:** distill is ahead too, although no papers were used for training.
  Ten documents is still too few to read much into, and the arXiv set is
  lowercased and pre-tokenised, which is not what the splitter was built for.
- **The learned salience score** is where most of the gain comes from. Against
  the same pipeline without it, ROUGE-2 at equal length rose by 1.9 points on
  the bills and 2.2 on the papers (both significant) and 0.4 on news (not
  significant).

Knobs are tuned on separate development sets (500 CNN/DailyMail validation
articles and 300 BillSum training bills), never on the test sets above. Run
`python3 eval/score.py --sets cnndm-dev,billsum-dev --budget words --compare
eval/dev-baseline.json` after a change. It prints the difference in every cell
and a paired bootstrap 95% interval for distill's ROUGE-2, so a change only
counts when the interval excludes zero. `eval/baseline.json` holds the test
results in the same format.

The salience weights are trained by `python3 eval/train_salience.py` on 3,000
CNN/DailyMail and 1,500 BillSum training documents, none of which overlap the
development or test sets. It labels each document's sentences with a greedy
ROUGE oracle, fits the regression with numpy, and writes `src/salience.json`;
rebuild to compile the new weights in. The weights fit only the encoder they
were trained with, so with another `--model-dir` the score is switched off and
ranking falls back to centrality alone.

Determinism is checked by `eval/determinism.sh`: the committed samples are
summarized in several modes with 1 and 4 threads, and the output hashes must
agree with each other and with `eval/golden.sha256`. Across different CPU types
the embeddings could differ in the last float digits, which would only matter if
two sentences were tied to about seven decimal places, so CI compares the golden
hashes on macOS arm64 (where they were recorded) and checks thread independence
on Linux.

## Limits

- **Extractive only.** It can leave things out, but it cannot rephrase, merge
  two sentences, or explain a term. There is no simplification step yet.
- **Legal text is the weakest case.** Long nested clauses are extracted whole,
  and a clause can depend on definitions elsewhere.
- **Sentence splitting is rule-based**, not a trained parser. Check odd input
  with `--show-segments`.
- **English only** (abbreviation lists, dangling-reference words, the model).
- **The salience score was trained on news and US bills.** It helped on papers
  too, but other genres may differ; `--salience 0` turns it off.
- **Speed is bounded by the embedding model**, not by Rust: roughly 60
  sentences per second on two slow cloud cores, so about 1 s for a news
  article and 3 s for a 3,700-word paper there. Everything else takes
  milliseconds.
- Sentences over 256 word-pieces are ranked on their first 256 (512 with the
  other models above).
- Common web page furniture is filtered out before ranking: consent banners,
  "Read more:" and "Click here" links, share and follow prompts, newsletter
  sign-ups, link menus, copyright and licence footers, and sections such as
  "Related articles". Anything else left in the file can still be selected.

## Tests

`cargo test --release` runs the splitter, heading, list and selection tests.
`eval/determinism.sh` checks that output is reproducible (see above). CI runs
both, plus `cargo clippy -- -D warnings`, on every push and pull request.

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
