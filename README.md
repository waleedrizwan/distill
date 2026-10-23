```
          distill::distill(text) -> String          Distiller::new(model_dir)?.summarize(text, &opts)?
          (loads the model, plain mode)             (load once, reuse; Options::plain() or ::default())
                         │                                            │
                         │        CLI: distill FILE | stdin  (thin wrapper over the same call)
                         │                        │                   │
                         └────────────────────────┼───────────────────┘
                                                  │ text  (a PDF must be converted first, e.g. pdftotext)
                                                  ▼
 ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
 ║ 1. SEGMENT   segment.rs                                                                          ║
 ║    headings ─┐                                                                                   ║
 ║    paragraphs├─► Document { sections, sentences[] }  (--show-segments prints this)               ║
 ║    sentences ┘   rule-based splitter: "et al.", "Fig. 3", "42 U.S.C. § 1983", short lists and    ║
 ║                  quotes stay one unit; captions/formulas/lowercase fragments = context only      ║
 ╚══════════════════════════════════════════════╤═══════════════════════════════════════════════════╝
                                                │ candidate sentences
                                                ▼
 ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
 ║ 2. EMBED   embed.rs   all-MiniLM-L6-v2 (local ./model, 90 MB)  ──►  384-dim vector per sentence  ║
 ║            used ONLY for ranking, never writes text         (optional DISTILL_CACHE on disk)     ║
 ╚══════════════════════════════════════════════╤═══════════════════════════════════════════════════╝
                                                │ vectors
                                                ▼
 ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
 ║ 3. RANK   select.rs + salience.rs                                                                ║
 ║                                                                                                  ║
 ║   centrality: "how much does LATER text echo me?"     learned salience (salience.json, built in) ║
 ║   (asymmetric: stated first, then elaborated)         logistic regression over vector +          ║
 ║   − penalty for dependent openers ("However", "see    position, length, dependence features      ║
 ║     Figure 2")                                        (trained on news + bills)                  ║
 ║              │                                                   │                               ║
 ║              └────────────── score = ½ · centrality + ½ · salience  (--salience W)               ║
 ╚══════════════════════════════════════════════╤═══════════════════════════════════════════════════╝
                                                ▼
 ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
 ║ 4. SELECT   budget: -n / --ratio / --words (default ≈10%, 3–15 sentences)                        ║
 ║    split the budget across top-level sections ──► greedy MMR per section:                        ║
 ║         pick  argmax [ relevance − λ · similarity_to_already_picked ]   (avoids repetition)      ║
 ╚══════════════════════════════════════════════╤═══════════════════════════════════════════════════╝
                                                ▼
 ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
 ║ 5. REPAIR   pick starts with "This/They/However…"  →  pull in the sentence before it             ║
 ║             pick is a list item                    →  pull in its lead-in      (--no-repair)     ║
 ╚══════════════════════════════════════════════╤═══════════════════════════════════════════════════╝
                                                │ picks (verbatim, in original order)
                                                ▼
 ┌───────────── PLAIN-LANGUAGE REWRITES (on in distill(text) and --plain) ──────────────────────────┐
 │  --compress   compress.rs  drop parentheticals, citations, "which/who" clauses, "However,"       │
 │               (runs before ranking so the word budget counts the shorter text)                   │
 │  --simplify   lexical.rs   swap words from the reviewed table (lexicon.tsv, "numerous"→"many");  │
 │               keep a swap only if re-embedding stays ≥ threshold similar to the original         │
 │  --split      split.rs     break at ", and" / ", but" / ";" when both halves are full clauses    │
 │  --expand-acronyms / --glossary   glossary.rs   use the document's own definitions               │
 │                                                                                                  │
 │  No LLM: every word is from the input or the reviewed table. Nothing is generated.               │
 └──────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                ▼
                    ┌──────────────── OUTPUT ────────────────┐
                    │ text: headings + picks; sentences that │
                    │   were adjacent share a line, a new    │
                    │   line = text was skipped              │
                    │ Summary: positions, scores, rewrites   │
                    │ --json: the same, from the CLI         │
                    └────────────────────────────────────────┘
                    Deterministic: same input + same binary = byte-identical output
```
