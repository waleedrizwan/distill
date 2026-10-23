# distill

A deterministic extractive summarizer in Rust. It picks the most representative
sentences from a document and prints them **verbatim**, in their original order.

- **Cannot hallucinate.** No text is generated. Every output sentence is a
  sentence from the input (whitespace-normalised). With `--compress`, a
  sentence can have pieces left out, but every word it keeps is from the input,
  in order. With `--simplify`, a word can be swapped for a simpler one, but
  only from a fixed, hand-reviewed table (`src/lexicon.tsv`). With `--split`,
  a long sentence can become two, losing a joining "and".
- **Deterministic.** Same input and same binary give byte-identical output.
- **Local.** One binary (12 MB) plus a 90 MB model file. No network at run time.

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

On macOS the build uses Apple's Accelerate framework for the matrix maths
automatically: on an M4 it embeds about twice as fast as the plain build, with
byte-identical output. It is developed on macOS (Apple Silicon); CI also builds
and tests on Linux, where `--features mkl` is the Intel equivalent.

Input is plain text, Markdown or PDF, from a file or stdin. A PDF is read
directly (`distill paper.pdf`); `distill --extract-text paper.pdf` prints the
text it was read as. Scanned PDFs without a text layer have no text to read.

```
distill [FILE]            summarize FILE (or stdin)
  -n, --sentences N       number of sentences (default: ~10% of the document, 3 to 15)
  -r, --ratio R           or a fraction of the document, e.g. 0.05
  -w, --words N           or up to N words, context sentences included
      --by-section[=N]    N sentences (default 2) under each top-level heading
      --json              sentence positions, sections and scores
      --show-segments     print how the text was split, without summarizing
      --compress          leave out asides: parentheticals, citations, "which"/"who"
                          clauses, dash interruptions, openers like "However,"
      --split             split long sentences at ", and" / ", but" / ";" when both
                          halves are full clauses
      --simplify          swap hard words for simpler ones from a reviewed table
                          ("numerous" -> "many"), only where meaning barely moves
      --plain             easier reading: prefer plainer sentences, and turn on
                          --compress, --split, --simplify, --glossary, --expand-acronyms
      --glossary          list the abbreviations the summary uses without spelling
                          out, with the document's own definitions
      --expand-acronyms   write the first such use as "log-likelihood ratio (LLR)"
      --no-repair         don't add context sentences
      --no-headings       don't print section headings
      --dehyphenate       rejoin words broken across lines (text already extracted from a PDF)
      --extract-text      print the input as distill reads it (a PDF's text) and stop
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

0. **Read** (`src/pdf.rs`, PDFs only). `pdf_oxide` finds the positioned text
   lines; the reading order is worked out here. Rotated text (the arXiv margin
   stamp), running headers and footers, page numbers and small print at the
   bottom of a page (footnotes, affiliations) are dropped. Lines that span the
   page's middle cut it into bands, and a band with text on both sides is read
   as two columns, left then right. Runs of text the library splits at font
   changes are rejoined ("BARE-generated", "Fe3O4"), citation numbers set
   small after a word are dropped, and a word hyphenated at a line end is
   rejoined unless the document writes it hyphenated elsewhere or both halves
   are words it uses alone ("resource-constrained"). Paragraphs set smaller
   than the body text (captions, text inside figures, tables) are dropped,
   except the abstract and small-capital headings. Headings are marked in
   Markdown (`# 1 Introduction`, `## 2.1 Data`) when a one-line paragraph is
   numbered, is a heading every paper uses, or is set bold or large, so
   labels inside figures are not taken for sections. Authors, affiliations
   and emails between the title and the "Abstract" heading are dropped.
1. **Segment** (`src/segment.rs`). Detects headings (Markdown, numbered,
   ALL CAPS, `SEC. 2. TITLE.`, `12. Governing Law.`), paragraphs, lists and
   sentences. The sentence splitter is rule-based with abbreviation handling
   for academic and legal text (`et al.`, `Fig. 3`, `42 U.S.C. § 1983`,
   `Smith v. Jones`). A short list is folded into the sentence that introduces
   it, so "X means-- (A) ..., (B) ..., and (C) ..." stays one unit.
   A quotation of up to 60 words stays one sentence even when it contains
   full stops, so a quoted statement is not cut in half. Pieces that are not
   sentences are kept as context but never picked: captions ("Figure B.1:"),
   pseudocode and formulas (most tokens not plain words), and, unless the
   whole text is lowercased, text that starts in lower case (the rest of a
   sentence a PDF column break or a display equation cut off).
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
python3 eval/fetch_papers.py # the arXiv papers (PDFs pinned in eval/papers.tsv; needs pdftotext)
cargo build --release
python3 eval/score.py        # a few minutes on an M-series Mac
```

The sample is the first 100 CNN/DailyMail test articles, the first 60 BillSum
test bills, the first 10 arXiv test papers, the first 30 GovReport test
reports, and 30 research papers as PDFs (below). The text is not committed, but
each document's SHA-256 is, so a different download fails instead of quietly
changing the scores.

The papers set (`eval/fetch_papers.py`) is chosen by rule: for each of six
arXiv categories (computational linguistics, quantitative biology, materials
science, planetary astrophysics, economics, statistical methodology), the first
five papers submitted in the week of 3 March 2025 whose PDF has a capitalised
"Introduction" (or "Background") heading, 2,000 to 20,000 words of body text
and an abstract of at least 80 words. The reference summary is the abstract;
the input is the paper's text from the Introduction heading on, so the
abstract itself is not in it. References, captions and appendices are left
in. Text comes from distill's own PDF reader (`--extract-text`), so the
scores are for what a user gets from `distill paper.pdf`;
`fetch_papers.py --extractor pdftotext` gives the poppler version instead. A
development set of 30 more papers comes from the week of 3 February 2025.

The PDF reader was checked against `pdftotext` on 82 arXiv PDFs (the papers
sets plus papers skipped by the filters): no failures, and a median F1 of
0.979 over words of three or more letters (0.963 over word pairs, which
also checks the order). Most of the difference is running headers, which
`pdftotext` keeps. Summaries scored the same either way: ROUGE-2 +0.11 on the
test papers and -0.28 on the development papers, neither significant. One
PDF of the 82 maps its "ff" ligature to the wrong characters ("aeected"),
for `pdftotext` as well. ROUGE is computed by `eval/score.py` with no stemming, so
the numbers are only comparable within this table.

Each set is scored twice. "N sentences" is the original setup. "ref length"
gives every method a word budget equal to that document's reference summary,
which removes the advantage of picking longer sentences. Under a word budget,
the context sentences distill adds for readability count against the budget.

| Set | Budget | Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | FK | Rare% |
|---|---|---|---|---|---|---|---|---|
| cnndm (n=100) | 3 sentences | lead | 29.4 | 12.3 | 21.1 | 66 | 12.1 | 4.4 |
|  |  | centroid | 26.3 | 8.9 | 17.7 | 77 | 13.6 | 4.9 |
|  |  | distill | **31.0** | **13.4** | **22.4** | **74** | **13.2** | **5.2** |
|  | ref length (34 w) | lead | 27.5 | 8.9 | 19.4 | 33 | 10.4 | 4.0 |
|  |  | centroid | 25.9 | 7.3 | 17.2 | 32 | 10.9 | 5.0 |
|  |  | distill | **27.3** | **9.4** | **19.3** | **32** | **11.0** | **4.7** |
| billsum (n=60) | 5 sentences | lead | 36.6 | 16.9 | 24.2 | 115 | 14.4 | 4.3 |
|  |  | centroid | 38.1 | 17.1 | 23.5 | 237 | 24.2 | 4.2 |
|  |  | distill | **41.6** | **21.9** | **27.8** | **222** | **18.2** | **3.8** |
|  | ref length (168 w) | lead | 42.8 | 19.7 | 27.4 | 166 | 12.8 | 4.5 |
|  |  | centroid | 41.1 | 16.2 | 24.5 | 164 | 17.0 | 4.7 |
|  |  | distill | **43.6** | **20.6** | **28.2** | **162** | **14.7** | **4.2** |
| arxiv (n=10) | 6 sentences | lead | 34.3 | 11.2 | 18.9 | 194 | 16.5 | 16.4 |
|  |  | centroid | 29.7 | 9.3 | 16.7 | 309 | 21.0 | 19.9 |
|  |  | distill | **34.0** | **12.0** | **19.2** | **272** | **19.4** | **20.3** |
|  | ref length (142 w) | lead | 36.0 | 11.5 | 20.1 | 142 | 12.8 | 17.2 |
|  |  | centroid | 35.8 | 10.4 | 21.1 | 141 | 15.8 | 19.5 |
|  |  | distill | **38.5** | **13.7** | **21.8** | **140** | **14.9** | **21.9** |
| govreport (n=30) | 20 sentences | lead | 44.6 | 13.2 | 17.9 | 530 | 17.2 | 6.2 |
|  |  | centroid | 54.0 | 22.5 | 21.6 | 725 | 21.2 | 6.8 |
|  |  | distill | **55.0** | **22.7** | **22.3** | **686** | **20.3** | **6.7** |
|  | ref length (599 w) | lead | 46.3 | 13.7 | 18.0 | 598 | 16.6 | 6.3 |
|  |  | centroid | 56.4 | 22.5 | 21.8 | 596 | 20.8 | 6.8 |
|  |  | distill | **56.6** | **22.5** | **22.7** | **596** | **19.9** | **6.7** |
| papers (n=30) | 8 sentences | lead | 34.5 | 10.6 | 18.0 | 197 | 17.0 | 14.2 |
|  |  | centroid | 37.4 | 9.6 | 16.9 | 236 | 18.1 | 15.9 |
|  |  | distill | **39.5** | **10.8** | **17.9** | **223** | **17.5** | **15.7** |
|  | ref length (194 w) | lead | 35.7 | 10.8 | 17.8 | 194 | 14.3 | 14.4 |
|  |  | centroid | 38.1 | 9.4 | 16.9 | 192 | 13.3 | 16.1 |
|  |  | distill | **38.1** | **10.2** | **17.5** | **192** | **13.6** | **15.8** |

What this shows:

- **News:** distill beats the first sentences, a famously strong baseline on
  news, without being told the text is news. At equal length the margin is
  small (9.4 against 8.9 ROUGE-2). At three sentences it is larger, but partly
  because distill's sentences are longer.
- **Bills:** at equal length distill is nearly a ROUGE-2 point ahead of the
  first sentences, and well ahead of plain centroid ranking.
- **Papers:** distill is ahead too, although no papers were used for training.
  Ten documents is still too few to read much into, and the arXiv set is
  lowercased and pre-tokenised, which is not what the splitter was built for.
- **Long government reports: level with plain centroid ranking.** On
  GovReport (reports of about 8,600 words, summaries of about 600) distill
  and centroid ranking both reach 22.5 ROUGE-2 at equal length (23.8 against
  24.5 on the 100 development reports). Until the length-dependent defaults
  (below) distill was 3.5 points behind: the summaries draw on the whole
  report, and the rule that favours sentences later text repeats works
  against that. The reports arrive as one block of text with no headings or
  paragraphs, so section budgets have nothing to work with.
- **Research papers: close to the first sentences, ahead of centroid.** On
  the 30 test papers at equal length the first sentences of the introduction
  score 10.8 ROUGE-2, distill 10.2 and centroid 9.4; on the 30 development
  papers distill is ahead (10.0, against 9.0 and 7.3). distill's picks are
  plainer than the first sentences (Flesch-Kincaid grade 13.6 against 14.3).
- **The old arXiv set** (10 lowercased papers) lost 2.7 ROUGE-2 at six
  sentences to the length-dependent defaults (below), and is now at 13.7 at
  equal length, ahead of both baselines.
- **The learned salience score** is where most of the gain comes from. Against
  the same pipeline without it, ROUGE-2 at equal length rose by 1.9 points on
  the bills and 2.2 on the papers (both significant) and 0.4 on news (not
  significant).

FK is the Flesch-Kincaid grade of the summary (lower is plainer) and Rare% the
share of its words outside the 20,000 most frequent words of the news training
articles, a rough measure of jargon. They describe the sentences chosen, not
whether they are good ones.

**Length-dependent defaults.** Short documents put the main point first, so
the ranking counts later text that echoes a sentence more than earlier text
(`--backward 0.3`), and keeps a fair amount of diversity (`--lambda 0.7`).
Long documents are summarized from the whole text, where the direction works
against the summary. So between 60 and 150 candidate sentences the defaults
move to `--backward 1` (no direction) and `--lambda 0.9`. No development
bill and only 17 of 500 news articles have 60 candidates, and their output is
unchanged; every report and paper has more. Setting either flag explicitly
overrides this.

Knobs are tuned on separate development sets (500 CNN/DailyMail validation
articles, 300 BillSum training bills, 100 GovReport validation reports and
30 development papers), never on the test sets above. Run `python3
eval/score.py --sets cnndm-dev,billsum-dev,govreport-dev,papers-dev --budget
words --compare eval/dev-baseline.json` after a change. It prints the difference in every cell
and a paired bootstrap 95% interval for distill's ROUGE-2, so a change only
counts when the interval excludes zero. `eval/baseline.json` holds the test
results in the same format.

`--compress` (`src/compress.rs`) is measured the same way. At reference length
it raised ROUGE-2 by 0.1 to 0.4 points on every set, significantly on news and
on the test bills, because the words it frees go to more content. It never
touches list labels, acronym definitions, defined terms, qualifiers such as
"(other than ...)", quoted text, or "which" clauses in legal operative text.

`--simplify` (`src/lexical.rs`) swaps words using `src/lexicon.tsv`: 221
single-word substitutions such as "numerous" to "many", "frequently" to
"often" and "commence" to "start", taken from SimplePPDB and kept only after
being read by hand (`eval/lexicon-approved.txt`; rejected pairs and why are in
`eval/lexicon-exclude.txt`). It handles plurals, verb endings and "a"/"an",
and leaves proper nouns, compounds, quotations and legal operative text
alone. Each swap is kept only if the sentence's embedding stays at a cosine
similarity of at least 0.95 to the original (`--simplify-threshold`); on the
development summaries every swap below 0.94 was wrong and those above 0.957
kept the meaning, with exceptions that were then removed from the table
("listeria contamination" became "listeria pollution") or remain
(in physics, "fundamental mode" became "basic mode"). It does not change ROUGE. It is also rare: 2 to 3% of news
summary sentences and 10% of paper sentences change, bills almost never. Each
swap replaces a word about seven times more common in news and bills. Most
SimplePPDB single-word rules turned out to be wrong in context ("survive" to
"die", "importation" to "export"), which is why the table is an allow-list;
`python3 eval/build_lexicon.py` reports the candidates nobody has reviewed yet,
most frequent first.

`--split` (`src/split.rs`) turns "X, and Y" into "X. Y" (", but" and ", so"
keep the conjunction: "X. But Y"; a semicolon becomes a full stop) when the
sentence has at least 24 words, each part at least 8, and both parts are full
clauses. There is no part-of-speech tagger: a clause needs a finite verb from
closed lists (auxiliaries, modals, irregular past forms, "-ed" forms), and the
second part must open with a subject (a pronoun, a determiner phrase, a name
or a bare noun phrase) that reaches that verb within a few words. It never
splits lists, numbered or lettered clauses, quotations or legal text. It is
precise but rare: across the development and test summaries it split 7 of
about 260 long news sentences, all correctly, and nothing in bills or papers.
The limit is not the clause detector: few long summary sentences contain such
a junction at all. ROUGE is unchanged.

`--glossary` (`src/glossary.rs`) finds the definitions a document gives for
its abbreviations, "log-likelihood ratio (LLR)" and "LLR (log-likelihood
ratio)", and lists after the summary the ones it uses without spelling out:

```
Glossary
  LoRA: low-rank adaptation
  DoRA: Weight-decomposed LoRAs
```

The words are the document's own. A definition is accepted when the short
form's letters are the initials of the words before the bracket (hyphenated
parts count), or failing that when they appear in order in at most a few
words, a capital matching a capital or the start of a word (Schwartz and
Hearst's method, made stricter). In three readings of 100 definitions found
in the 60 papers, precision went from 89% to 98% to 100% as the rules were
tightened; in the summaries, about half get a glossary (33 of 60, 51
entries), and every entry checked in the test papers' summaries was right.
`--expand-acronyms` instead spells out the first use inside the sentence,
which inserts the document's words into it.

`--plain` is for reading something hard. Each sentence's Flesch-Kincaid
grade becomes a ranking term (`src/readability.rs`, by rank within the
document, weight 0.4), so of two sentences with similar content the plainer
one wins; text that does not end like a sentence counts as hardest, so a
short fragment is not mistaken for a plain sentence. It also turns on the
layers above. On the papers at reference length it lowers the summaries'
grade by about three (test 13.6 to 10.8, development 13.1 to 10.3) and
their share of rare words (15.8% to 13.9% on test), at no significant ROUGE
cost (ROUGE-2 +0.61 test, -0.07 development). The weight was chosen on the
development papers: 0.6 and above cost ROUGE significantly.

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

The guarantees are also property tests, run by `cargo test`: on thousands of
generated sentences and documents (`src/testgen.rs`), `--compress` only ever
removes words, `--split` keeps every word but the dropped "and", a lexical swap
never touches quoted text and changes nothing outside its range, the sentence
splitter loses no text, and selection stays in document order within its
budget. The same contracts are fuzz targets in `fuzz/` (`cargo +nightly fuzz
run compress`, likewise `split`, `lexical`, `segment` and `select`; needs
`cargo install cargo-fuzz`). CI runs each target for 30 seconds.

## Limits

- **Extractive only.** It can leave things out, but it cannot rephrase, merge
  two sentences, or explain a term. `--compress` only deletes. It is
  conservative (about 3% of words on news, 1% on bills) and lossy by design:
  a removed "which" clause can carry real information.
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

## Third-party data

`src/lexicon.tsv` is derived from SimplePPDB by Ellie Pavlick and Chris
Callison-Burch ("Simple PPDB: A Paraphrase Database for Simplification", ACL
2016), licensed under
[Creative Commons Attribution 3.0 United States](https://creativecommons.org/licenses/by/3.0/us/).
It has been modified: filtered by `eval/build_lexicon.py` and reviewed by hand.
The original download link is dead; the build uses the Internet Archive's
capture of `https://www.seas.upenn.edu/~nlp/resources/simple-ppdb.tgz`,
whose checksum is recorded in the script.

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
