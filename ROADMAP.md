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
but two things are hardcoded first:

- [ ] Read the token limit from the model's config instead of `MAX_TOKENS` in
  `src/embed.rs`.
- [ ] Make pooling a per-model setting stored next to the weights: bge-small
  uses CLS pooling, gte-small mean pooling, e5-small mean pooling plus a
  `"query: "` prefix.
- [ ] Extend `fetch-model.sh` to take a model name, each with a pinned revision
  and checksums.
- [ ] Score bge-small-en-v1.5, gte-small and e5-small-v2 against MiniLM. Keep
  the winner, or keep MiniLM. Re-measure speed: 512 tokens costs more on long
  legal sentences.

## Milestone 3: learned salience head (only if Milestone 2 leaves headroom)

- [ ] `eval/oracle.py`: greedy oracle extractive labels against the references
  on the training splits.
- [ ] Features: the frozen sentence embedding plus position, length, position in
  section, and `is_dangling`.
- [ ] Fit logistic regression offline, ship the weights (a few thousand floats)
  as a small file next to the model, apply them in `src/select.rs` as a dot
  product. Deterministic, no new dependencies.
- [ ] Keep it only if it beats Milestone 2 on both news and bills.

## Milestone 4: simplifier, three opt-in layers

Each layer has its own flag and tests. Where the layer keeps output as a subset
of the input's words, a property test enforces that.

- [ ] **Deletion-based compression.** Rule-based removal of parentheticals,
  appositives, citations, "which" clauses and sentence-initial adverbials. The
  output is a subsequence of the input, so every word is still verbatim. Later,
  a token-deletion classifier trained on the Google sentence-compression data
  on top of the frozen encoder.
- [ ] **Lexical substitution from a fixed table** (for example SimplePPDB ranked
  by word frequency). A swap is accepted only if the sentence embedding stays
  within a cosine threshold of the original. The table is a plain file anyone
  can read and edit.
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
