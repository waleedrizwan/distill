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

Follow-up found while reading output: the sentence splitter breaks
multi-sentence quotations, so a pick can open with an unclosed quote mark
("I just let frustration get the better of me."). The head is more willing than
centrality to pick such fragments. Keeping quotations whole in
`src/segment.rs`, or an unbalanced-quote feature, would fix it.

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
  `src/lexicon.tsv` holds 223 single-word substitutions from SimplePPDB
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
  technical sense ("fundamental mode" in physics became "basic mode"). Effect: no ROUGE change; about 2% of news summary sentences and 10% of
  paper sentences change, bills almost never; each swap uses a word about
  seven times more common. Coverage grows only by reviewing more of the
  unreviewed candidates the build reports.
- [ ] **Sentence splitting on coordination** ("X, and Y" into two sentences when
  both halves have a subject and verb). Needs a part-of-speech tagger, so it
  comes last and only if the first two layers prove the demand.

A small fine-tuned seq2seq model with greedy decoding and a constrained
vocabulary stays off the roadmap until these three ship: it is the one option
that can reorder meaning.

## Decision gate after Milestone 1

If length normalisation and the dangling-sentence changes beat lead-3 on news by
a clear margin at equal length, Milestone 2 becomes optional and Milestone 4 can
start in parallel. (Not met; see Milestone 1.)
