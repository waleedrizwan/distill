# Roadmap

distill is a deterministic, extractive summarizer. Every step below keeps the two
guarantees that define it: the same input always gives the same output, and no
words appear in a summary that are not in the source (the simplifier layers in
Milestone 4 are opt-in and each one states exactly how far it relaxes this).

The milestones are gates. Each one has to show a number before the next starts,
and every pull request that touches ranking includes the before/after table from
`eval/` in its description.

## Milestone 0: make quality measurable

Today every knob in `src/select.rs` is tuned by eye, the README's ROUGE table
cannot be reproduced from the repo, and the sentence-count budget lets longer
picks score higher. Nothing later can be judged until this exists.

- [x] **Frozen evaluation sample.** `eval/fetch_data.py` downloads a fixed slice
  of CNN/DailyMail test (100 articles), BillSum test (60 bills) and arXiv test
  (10 papers) and checks every document against the SHA-256 in
  `eval/manifest.tsv`. The text is not committed: the news articles are
  copyrighted, and the checksums make the sample just as reproducible.
- [x] **Scorer.** `eval/score.py` runs the release binary over the sample and
  prints ROUGE-1/2/L and average words per summary for lead-N, plain centroid
  ranking, and distill defaults. Standard library only, no stemming, so it
  matches the README's numbers.
- [x] **Word budget.** `--words N` selects sentences up to a word count instead
  of a sentence count. The scorer compares methods at equal length (each
  document's budget is its reference summary's length) as well as at the old
  sentence counts.
- [x] **README table regenerated** from the scorer, with the word-budget results.
- [x] **CI.** `.github/workflows/ci.yml` runs `cargo test`, `cargo clippy`, then
  `eval/determinism.sh`, which summarizes the committed samples in several
  modes, compares SHA-256 hashes to `eval/golden.sha256`, and checks that 1 and
  4 threads give identical output.

## Milestone 1: cheap ranking wins

One pull request each, ordered by expected payoff per hour. Any change that
does not move the table is dropped. Tuned on the development sets (500 news
articles, 300 bills); a change counts only when the paired bootstrap 95%
interval for ROUGE-2 excludes zero.

- [x] **Context counts against the word budget.** A pick now pays for the
  context sentences repair adds for it, so `--words N` bounds the whole
  summary. Bills drop from 234 to 185 words on the development set for a
  ROUGE-2 change of -0.31 (interval -0.92 to +0.32, not significant). Kept as
  a correctness fix: without it, equal-length comparisons are not equal.
- [x] **Length normalisation: dropped.** Lowering relevance by log length did
  nothing at equal length (within ±0.2 in both directions), because the word
  budget already charges long sentences for their words. Picking by gain per
  word was clearly worse: it favours short, low-content sentences.
- [x] **Dangling-sentence policy: no change.** Dependence penalties from 0.15
  to 0.6, with and without repair, and the lead, boost, lambda and
  section-weight knobs all stayed within noise on bills. Raising the penalty
  was significantly worse on news. Turning repair off was significantly worse
  on bills. Defaults are unchanged.
- [x] **Boilerplate filter.** A deterministic pre-pass in `src/segment.rs`
  drops consent banners, "Read more:" and "Click here" links, share, follow and
  newsletter prompts, link menus, copyright and licence footers, bylines and
  datelines as headings, and sections such as "Related articles". On all eval
  data it removes 44 of 29,217 candidate sentences, all of them genuine
  boilerplate. News ROUGE-2 +0.12 on the development set (interval +0.02 to
  +0.23), bills unchanged. `samples/web-article.txt` is the fixture.

**Gate result.** At equal length on the test sets, distill is within noise of
the first sentences on news and about one ROUGE-2 point behind them on bills.
The gate below is not met, so Milestone 2 goes ahead.

## Milestone 2: embedding model swap

`all-MiniLM-L6-v2` is a 2021 model tuned for short-text similarity and truncates
at 256 word-pieces. Same-size BERT checkpoints load through the same candle code,
but two things were hardcoded first:

- [x] The token limit comes from the model folder's `distill.json`, capped by
  the model's position embeddings, instead of `MAX_TOKENS` in `src/embed.rs`.
- [x] Pooling and text prefix are per-model settings in `distill.json`:
  bge-small uses CLS pooling, gte-small mean pooling, e5-small mean pooling
  plus a `"query: "` prefix. Without the file, MiniLM's settings apply, and
  MiniLM output is byte-identical either way.
- [x] `fetch-model.sh NAME` downloads minilm, bge, gte or e5, each with a pinned
  revision and checksums. The embedding cache is now keyed by model too;
  before, switching models silently reused the old model's vectors.
- [x] **Result: keep MiniLM.** On the development sets at reference length,
  no candidate beats it on news or bills, and none of the differences is
  significant. That still holds after retuning lambda and beta for each model.
  The candidates have 12 layers to MiniLM's 6 and take about twice as long to
  embed. The 512-token limit did not help the bills either. Table in README.

**Gate result.** Swapping the encoder leaves the ranking where it was: level
with the first sentences on news and behind them on bills. The unsupervised
signal looks saturated, which is the headroom Milestone 3 needs, so it goes
ahead with MiniLM as the frozen encoder.

## Milestone 3: learned salience head

- [x] `eval/train_salience.py`: greedy oracle extractive labels (mean of
  ROUGE-1 and ROUGE-2 F1) on 3,000 CNN/DailyMail and 1,500 BillSum training
  documents, kept apart from the development and test sets.
- [x] Features: the frozen MiniLM sentence embedding plus document and section
  centrality, position, position in section, length, dangling and
  external-reference flags, list membership and summary-section membership.
  `distill --dump-features` prints exactly what the binary computes.
- [x] Logistic regression fitted offline with numpy (Newton's method, L2 chosen
  on held-out training documents). The weights, about 400 floats plus
  standardisation, live in `src/salience.json` and are compiled in. Scoring is
  a dot product in fixed order. The head checks that the loaded encoder
  matches the one it was trained on and switches off otherwise.
- [x] **Result: kept, on by default at `--salience 0.5`.** Chosen on the
  development sets, where 0.5 gave the best balance (news +0.62 ROUGE-2, not
  significant; bills +1.40, significant). On the test sets at equal length:
  bills +1.86 and papers +2.18 (both significant), news +0.36 (not
  significant). distill now beats the first-sentences baseline at equal length
  on all three test sets.

- [x] Follow-up found while reading output: the sentence splitter broke
  multi-sentence quotations, so a pick could open with an unclosed quote mark
  ("I just let frustration get the better of me."). The head is more willing
  than centrality to pick such fragments. Fixed in `src/segment.rs` (PR #11): a
  closed quotation of up to 60 words stays one sentence. No clear ROUGE change
  on any development set.

## Milestone 4: simplifier, three opt-in layers

Each layer has its own flag and tests. Where the layer keeps output as a subset
of the input's words, a property test enforces that.

- [x] **Deletion-based compression (`--compress`, `src/compress.rs`).** Rules
  on punctuation and a short word list remove parentheticals, bracketed
  citations, comma-bounded "which"/"who" clauses, spaced dash asides and
  sentence-opening connectives. Kept words are the input's, in order (a test
  checks this, and so did a pass over every compressed sentence in the
  development summaries); the only change to a kept word is capitalising a new
  first word. Guards came from reading output: list labels, acronym
  definitions, defined terms, qualifiers ("other than ...", "except ..."),
  anything inside quotations (including bills' ``...'' insertions), "which"
  clauses in legal operative text, clauses that would swallow the main verb,
  and coordinated clauses are all left alone. Result at reference length:
  ROUGE-2 +0.24 on development news (significant), +0.12 on development bills;
  on the test sets +0.12 news and +0.33 bills (both significant), +0.38 papers.
  It removes about 3% of words on news and 1% on bills. Opt-in, since it
  relaxes the whole-sentence guarantee. Not done: the trained token-deletion
  classifier on the Google sentence-compression data.

- [x] **Lexical substitution from a fixed table (`--simplify`, `src/lexical.rs`).**
  `src/lexicon.tsv` holds 221 single-word substitutions from SimplePPDB
  (CC BY 3.0 US; attribution in the README), built by `eval/build_lexicon.py`:
  paraphrase score >= 3.5, simplification score >= 0.75, the replacement at
  least twice as frequent in the training documents, and the pair listed in
  `eval/lexicon-approved.txt`. Every candidate pair is either approved or
  recorded as rejected, with a reason, in `eval/lexicon-exclude.txt`.
  Inflection is handled in code (plurals, -s/-ed/-ing, a few irregular
  replacements, "a"/"an"). A swap is kept only if the sentence embedding stays
  at cosine >= 0.95. Proper nouns, compounds, quotations and legal operative
  text are left alone.
  Findings: SimplePPDB's single-word rules are mostly wrong in context; of the
  200 most frequent candidates after loosening the filters, one was usable.
  On the development summaries, every swap with similarity below 0.94 was
  wrong and those above 0.957 kept the meaning, though a few lost nuance or a
  technical sense ("fundamental mode" in physics became "basic mode"). Effect: no ROUGE change; 2 to 3% of news summary sentences and 10% of
  paper sentences change, bills almost never; each swap uses a word about
  seven times more common. Coverage grows only by reviewing more of the
  unreviewed candidates the build reports.
- [x] **Sentence splitting on coordination (`--split`, `src/split.rs`).** Started
  without the part-of-speech tagger planned here: clause detection uses closed
  word lists (auxiliaries, modals, irregular past forms, "-ed" forms) and
  subject patterns (pronouns, determiner phrases, names, bare noun phrases).
  Splits "X, and Y" / ", but" / ", so" / ";" in sentences of 24+ words when
  both parts are clauses; never in lists, numbered clauses, quotations or
  legal text. Reading every split on the development and test summaries found
  two kinds of error, both fixed with tests: numbered lists split at their
  semicolons, and "and one that has ..." treated as a new subject. After the
  fixes: 7 splits in about 260 long news sentences, all correct; none in bills
  (legal text is exempt) or papers; ROUGE unchanged. A tagger would not change
  much: few long summary sentences contain a clause-level junction at all.
  The first two layers did not clearly prove the demand this layer was gated
  on; it was built on request and stays opt-in.

  Fixed along the way: the legal-text detector shared by all three layers
  matched list labels case-insensitively, so "(CNN)" at the start of a news
  article counted as a label like "(a)" and switched off clause deletion and
  word swaps on those sentences. Labels are now case-sensitive. Two more
  lexicon pairs were removed after reviewing the swaps this enabled
  ("listeria contamination" -> "pollution", "manufacturing plants" ->
  "producing plants").

A small fine-tuned seq2seq model with greedy decoding and a constrained
vocabulary stays off the roadmap until these three ship: it is the one option
that can reorder meaning.

## Decision gate after Milestone 1

If length normalisation and the dangling-sentence changes beat lead-3 on news by
a clear margin at equal length, Milestone 2 becomes optional and Milestone 4 can
start in parallel. (Not met; see Milestone 1.)

# Part 2: understand hard text without an LLM

The first roadmap made distill a good extractive summarizer of news and bills.
The goal now is the one it was built for: take any text, especially a research
paper full of jargon, and make it shorter **and easier to understand**, locally,
for free, with no LLM.

Ground rules for Part 2 (agreed 2026-10-22):

- **No LLM, no paid service.** Anything bundled must be free to redistribute.
- **No invented words in the summary body.** Sentences stay the source's
  (with the opt-in Milestone 4 layers). Explanations come from the document
  itself or from a fixed, bundled, attributed dictionary, and are shown in a
  labelled glossary, never mixed into the sentences.
- **Determinism is kept where it is cheap** (golden hashes, 1 vs 4 threads) but
  is no longer allowed to block a useful feature; any loss is stated.
- **Network only when the argument is a URL** (Milestone 12). Every other path
  stays offline.
- **Primary platform: macOS on Apple Silicon.** Linux stays green in CI.
- **One branch and pull request per change**, merged after CI passes. Ranking
  changes carry the before/after table; changes judged by reading carry the
  read-through counts.
- **Measures.** ROUGE on every set (nothing may regress significantly), plus a
  new papers set, a readability score, and hand-read precision for anything
  ROUGE cannot see.

## Milestone 5: housekeeping

- [x] Regenerate `eval/baseline.json`, `eval/dev-baseline.json` and the README
  test table. PR #11 changed segmentation after they were recorded; news distill
  ROUGE-2 at reference length moved from 9.6 to 9.4 (95% CI -0.87 to +0.37,
  not significant), the other sets by 0.1 or less.
- [x] Tick the quotation follow-up under Milestone 3 (done in PR #11).
- [x] Accelerate is now on by default for macOS builds (target-specific
  dependency features). On an M4 it embeds a 9,400-word report in 1.4 s instead
  of 2.9 s; all golden hashes and every eval cell are unchanged. README says
  where it is developed and tested.

## Milestone 6: a papers evaluation set (gate for everything below)

The arXiv set has 10 documents, lowercased and pre-tokenised, so it cannot
judge PDF handling, structure or jargon.

- [x] `eval/papers.tsv`: 30 test and 30 development arXiv papers, chosen by
  rule rather than by hand (the first five per category in a fixed week that
  pass length and structure filters), in six fields: cs.CL, q-bio.QM,
  cond-mat.mtrl-sci, astro-ph.EP, econ.GN and stat.ME. Pinned by versioned ID,
  with the SHA-256 of each PDF and abstract. `eval/fetch_papers.py` downloads
  them, pacing itself as arXiv asks and caching API answers. Maths papers were
  planned but are not in: with few prose abstracts and no Introduction heading
  they rarely pass the filters, and stat.ME covers formula-heavy writing.
- [x] Reference summary = the abstract. Input = the text from the first
  "Introduction" (or "Background") heading on. Text comes from `pdftotext`
  (dehyphenated as `--dehyphenate` does) until Milestone 7.
- [x] `score.py` gains FK (Flesch-Kincaid grade) and Rare% (share of words
  outside the 20,000 most frequent in the news training articles) columns.
- [x] Baseline in the README. At reference length on the test papers, ROUGE-2:
  lead 10.2, centroid 8.3, distill 8.1. On the development papers: lead 7.1,
  centroid 9.4, distill 8.7. distill is behind on papers, which is what the
  rest of Part 2 has to fix. Its picks are the plainest (FK 12.2 against 13.9
  for lead on test).

## Milestone 7: native PDF input

- [x] `distill paper.pdf` works directly: the input is detected by its `%PDF-`
  header (`src/pdf.rs`). `--extract-text` prints the text as read.
- [x] Bake-off on 18 arXiv PDFs, F1 against `pdftotext` over words of three or
  more letters: `pdf-extract` 0.984 median, one panic, 92 ms; `pdf_oxide`
  0.995, no failures, 70 ms. `pdf_oxide` won, but its own reading order swaps
  lines and drops margin stamps into sentences, and it joins words across the
  column gap. So it is used only for positioned lines and runs, and the
  layout is done here; a parser panic becomes an error.
- [x] Layout: lines split at the column gap; rotated text, running headers
  and footers, page numbers and bottom-of-page small print dropped; columns
  detected per band (a wide abstract above two columns is common); paragraphs
  from line spacing, indents, short last lines and heading-like lines; runs
  rejoined at font changes, subscripts attached, small citation numbers
  dropped; line-end hyphens rejoined unless the document says otherwise.
  Tests on hand-made lines, a committed two-column fixture
  (`samples/two-column-paper.pdf`, written by `samples/make-paper-pdf.py`,
  also in the goldens), property tests and a `pdf_layout` fuzz target: the
  layout only reorders, joins and drops, never invents a character, and the
  result does not depend on the order lines arrive in.
- [x] **Gate met.** On 82 PDFs: no failures, median F1 0.979 against
  `pdftotext` (the gate was 0.97), 0.963 over word pairs. Papers ROUGE-2
  against the `pdftotext` path: +0.11 test, -0.28 development (gate: within
  0.3). Summaries got plainer (FK -0.6 test, -0.8 development), because less
  furniture reaches them. The papers sets now use this reader.
- Not fixed: one PDF in 82 has a broken ligature map ("aeected");
  `pdftotext` reads it the same way.

## Milestone 8: paper structure

- [x] **Non-prose is not a candidate** (`src/segment.rs`, `src/pdf.rs`): a
  paragraph in smaller type than the body (captions, figure text, tables;
  not the abstract or small-capital headings) is dropped from PDFs; a
  sentence whose tokens are mostly not plain words (pseudocode, formulas),
  a caption with a lettered label ("Figure B.1:", "Fig. S3."), or a sentence
  starting in lower case (unless the whole text is lowercased) is not a
  candidate. Found along the way: two-column pages with a gutter narrower
  than 1.5 em (common on A4) were read across the columns; a narrower gap
  that covers the page's middle now splits too.
- Result. ROUGE-2 at reference length: papers test +0.66 (95% CI -0.25 to
  +1.63), papers-dev +0.37 (-0.38 to +1.17), both papers pooled +0.52
  (-0.08 to +1.11), the old arXiv set +1.11 (+0.08 to +2.52, significant);
  news, bills and reports unchanged (within 0.01). **Read-through** of
  every sentence that changed in the 30 development summaries: of 202
  sentences that left, about 140 were fragments, captions, pseudocode,
  reference entries or garbled text; of 172 that came in, about 50 were.
  About 90 fewer junk sentences in about 300. Kept for that.
- Tried and dropped: skipping everything after the reference list
  (appendices are 21% of candidate sentences): papers-dev -0.21, significant
  (they support the centrality of the main text); requiring a sentence to
  end with a full stop: as much junk came in as went out, and it dropped a
  web article's headline.
- Still junk: lines cut off by a display equation that happen to end the
  paragraph ("... can be calculated as"), table rows, reference entries in
  journals' own styles, figure text whose words run together.
- [x] **Headings marked in PDFs.** Short lines inside figures and tables were
  taken for section headings ("BASE", "Embedding0.4 Cosine Similarity0.6"),
  which split section budgets across fake sections. `src/pdf.rs` now marks
  headings in Markdown when a one-line paragraph is numbered ("3.2",
  "III.", appendix "B.1"), a heading every paper uses, or bold or large and
  capitalised; segment stops guessing headings from bare short lines in a
  document that marks at least two. The title block (authors,
  affiliations, emails) before an "Abstract" heading is dropped.
  ROUGE-2 +0.45 test, +0.48 development (not significant alone); other
  sets unchanged.
- [x] `--by-section[=N]`: N sentences (default 2) from every top-level
  section with at least two candidates, under its heading.
- [x] **Gate met.** Both parts against the Milestone 9 baselines at reference
  length: papers test +1.11 (95% CI +0.01 to +2.19), development +0.86
  (-0.08 to +1.82), the 60 papers pooled +0.98 (+0.23 to +1.72). distill
  is now ahead of centroid on both sets and 0.6 behind the first sentences on
  the test papers.
- [ ] Bold run-in paragraph headings ("Prefix Tuning Prefix Tuning, as
  introduced in ...") split off.

## Milestone 9: long-document ranking

distill trailed plain centroid ranking on GovReport by 3.5 ROUGE-2 (test) and
5.2 (development). Papers are long too.

- [x] One knob at a time on govreport-dev at reference length: `--backward 1`
  (undirected centrality) +3.65, centroid centrality +2.30, salience off
  +1.56, `--lambda 1` +1.17, `--lambda 0.9` +0.78, no lead bonus +0.37. In
  pairs: `--backward 1` with `--lambda 0.9` +4.74, with `--salience 0.3`
  +4.77, with `--lambda 1` +4.66. On news and bills `--backward 1` cost -0.30
  and -0.10 (not significant).
- [x] **Kept: length-dependent defaults** (`select::LengthDefaults`). From 60
  to 150 candidate sentences, `lambda` moves from 0.7 to 0.9 and `backward`
  from 0.3 to 1; explicit flags override. Ramps of 100-250 (+3.32 on
  govreport-dev) and 80-120 (+4.43) were tried; 60-150 gave +4.52. News and
  bills are byte-identical on every development and test document.
- [x] **Gate met.** govreport-dev +4.52 (95% CI +3.76 to +5.27); GovReport
  test +3.56 at reference length (+1.93 to +5.19), now level with centroid at
  22.5. Papers better, not significantly: +0.74 development, +0.89 test.
  News and bills unchanged.
- Cost: the old 10-paper arXiv set (lowercased, pre-tokenised) drops 2.73
  ROUGE-2 at six sentences (significant), 0.90 at equal length (not). The
  `backward` change is the cause (`lambda` alone: +0.75 at equal length).
- Dropped: `backward` 0.65 for long documents. papers-dev +1.15
  (significant) but govreport-dev only +2.28, and the papers difference
  between 0.65 and 1 (0.4) is within noise on 30 papers. Not tried yet:
  pseudo-sections for headingless text, and retraining the salience head
  with reports and papers; both stay open below.
- [ ] Pseudo-sections for text with no headings (TextTiling-style cuts).
- [ ] Retrain the salience head with GovReport and arXiv training documents.

## Milestone 10: jargon glossary (`--glossary`)

- [x] **From the document: abbreviations** (`src/glossary.rs`). "long form
  (SF)" and "SF (long form)", by initials first, then a stricter Schwartz and
  Hearst letter match. `--glossary` lists the abbreviations a summary uses
  without spelling out; `--expand-acronyms` spells out the first use in place;
  `--json` carries a `glossary` field. Precision in three readings of 100
  definitions from the 60 papers: 89%, then 98% and 100% after the rules were
  tightened (cited names such as "Cicchetti, 1994", the rarer SF-first order
  only on exact initials, no brackets or commas in a definition, capitals
  matching capitals or word starts). In use: 33 of the 60 paper summaries get
  a glossary, 51 entries, all 27 test-paper entries correct.
- [ ] **From the document: other definitions** ("X is defined as", "we call
  X", "X refers to", "X, i.e. ..."). Not started.
- **From a bundled dictionary: dropped, gate missed.** Built and measured:
  Open English WordNet 2025 (CC BY 4.0), filtered to the 51,000 single words
  the MiniLM tokenizer splits into pieces (a sign of a rare word: "hysteresis"
  yes, "protein" no); senses chosen by embedding similarity between the
  sentence and each "word: definition", shown only with a margin of 0.03 over
  the next sense and a similarity of at least 0.40 (tuned on the development
  papers, where 27 of 32 entries were right after dropping adverbs and
  circular definitions). On the test papers only 14 of 22 entries had the
  right sense (64%, gate 80%), and a higher threshold did worse (58% at
  0.45). WordNet lacks the technical senses papers use: "kinetics" got the
  mechanics sense in a chemistry paper, "homology" the biological one for
  persistent homology, "decarbonizing" an engine, "diagnostics" medicine,
  "neuron" a nerve cell in a language-model paper. A domain dictionary, or
  definitions the document itself gives, would be needed.
- [x] Property tests and a `glossary` fuzz target: every definition and short
  form is the document's own text (whitespace collapsed, as distill holds
  sentences), and expanding only inserts definitions in front of short forms.
  Fuzzing found two real faults (a definition running into a bracket in the
  SF-first order; the span contract stated without the whitespace caveat).
- [x] **Gate (hand-read on the papers set):** at least 90% of document
  definitions correct: met (98% and 100% on two samples of 100). At least 80%
  of dictionary entries with the right sense: missed (64%), so as planned only
  the document part ships.

## Milestone 11: plainer summaries (`--plain`)

- [x] A per-sentence difficulty score as a ranking term
  (`src/readability.rs`): the sentence's Flesch-Kincaid grade, by rank within
  the document so a few extreme values do not squash the rest; text that does
  not end like a sentence counts as hardest (otherwise short PDF fragments
  win). Simpler than planned: rare-word rate and symbol density were not
  needed to move the grade, and the grade is what the gate measures.
- [x] `--plain` turns on the difficulty term (weight 0.4) plus `--compress`,
  `--split`, `--simplify`, `--glossary` and `--expand-acronyms`. Weight
  chosen on papers-dev: 0 (layers only) FK 13.08, 0.2 FK 11.35, 0.4 FK 10.25
  (ROUGE-2 -0.07), 0.6 FK 9.57 (ROUGE-2 -0.87, significant), 1.0 FK 8.84.
- [ ] Grow `src/lexicon.tsv` for academic vocabulary by reviewing the most
  frequent unreviewed SimplePPDB candidates in papers, with the same allow-list
  process.
- [x] **Gate met.** Test papers at reference length: FK 13.64 -> 10.77,
  Rare% 15.8 -> 13.9, ROUGE-2 +0.61 (95% CI -0.51 to +1.76).
  Development papers: FK 13.14 -> 10.25, ROUGE-2 -0.07.

## Milestone 12: browser extension and URLs

- [x] Extension: an options page with **Plain mode** and **Glossary**
  (`chrome.storage.sync`); the glossary is shown under distilled text (inside
  the distilled span, so undo and click-to-compare still restore the
  original exactly); the PDF reader gets **Summarize paper**, which sends the
  PDF bytes (base64) so distill's own PDF reader is used, and shows
  `--by-section=2` with headings and the glossary. The native host passes
  `plain`, `glossary`, `by_section` and `pdf` through and returns the
  glossary as data. Checked in Chrome on a harness page that loads the
  extension's own `content.js` and `reader.js` with `chrome.*` stubbed and
  native messages relayed to the real `distill-host`: a test article
  (glossary shown, Cmd+Z restores the original) and an arXiv PDF (whole-paper
  summary). The installed extension's service worker and context menus were
  not driven (the browser tool cannot open `chrome-extension://` pages);
  reload the unpacked extension to pick up the change. Found along the way:
  appendix-letter headings matched axis labels ("R Rmax"); letters are now
  A to J.
- [ ] `distill https://...`, opt-in by the argument: fetch the page (`ureq`,
  already a dependency); PDFs go through Milestone 7, HTML through a simple
  main-content extractor and then the boilerplate filter. arXiv `abs/` links
  fetch the PDF. No other code path touches the network.

**Order and scope for the first session:** Milestones 5 to 9 are expected to
land. Milestones 10 to 12 are worked in order as time allows; anything
unfinished stays here, unticked, with notes.
