```
                            ┌──────────────────────────── ENTRY POINTS ────────────────────────────┐
                            │                                                                      │
   distill FILE | stdin     │   distill https://...          Chrome extension (extension/*.js)     │
   (.txt .md .html .pdf)    │   fetch() — the only time      highlight text / open a PDF           │
            │               │   it uses the network                   │                            │
            │               │   (arXiv abs/ → its PDF)                │ chrome.runtime.            │
            │               │            │                            │ sendNativeMessage          │
            │               │            │                            ▼                            │
            │               │            │                  distill-host (src/bin/distill-host.rs) │
            │               │            │                  4-byte length + JSON on stdin/stdout:  │
            │               │            │                  {text | pdf(base64), plain, glossary}  │
            │               │            │                  → runs the `distill` binary next to it │
            └───────────────┴────────────┴──────────────────────────────┬──────────────────────────┘
                                                                        │ raw bytes
                                                                        ▼
 ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
 ║ 0. READ   pdf.rs (PDF)  /  html.rs (web page)  /  plain text as-is                               ║
 ║    PDF:  positioned lines → reading order (columns, bands), drop headers/footers/page numbers,   ║
 ║          footnotes, captions, author block; rejoin hyphenated words; mark headings as `# ...`    ║
 ║    HTML: keep <article>/<main>, drop scripts, menus, header/footer, asides, lists of links       ║
 ╚══════════════════════════════════════════════╤═══════════════════════════════════════════════════╝
                                                │ clean text  (--extract-text stops here)
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
 ┌──────────────────────── OPTIONAL REWRITES (all on by --plain) ───────────────────────────────────┐
 │  --compress   compress.rs  drop parentheticals, citations, "which/who" clauses, "However,"       │
 │               (runs before ranking so the word budget counts the shorter text)                   │
 │  --simplify   lexical.rs   swap words from the reviewed table (lexicon.tsv, "numerous"→"many");  │
 │               keep a swap only if re-embedding stays ≥ threshold similar to the original         │
 │  --split      split.rs     break at ", and" / ", but" / ";" when both halves are full clauses    │
 │  --expand-acronyms / --glossary   glossary.rs   use the document's own definitions               │
 │                                                                                                  │
 │  Invariant: every kept word comes from the input, in order. Nothing is generated.                │
 └──────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                ▼
                    ┌──────────────── OUTPUT ────────────────┐
                    │ text: headings + picks; sentences that │
                    │   were adjacent share a line, a new    │
                    │   line = text was skipped              │
                    │ --json: positions, sections, scores    │
                    │ extension: {"ok":true,"summary",...}   │
                    └────────────────────────────────────────┘
                    Deterministic: same input + same binary = byte-identical output
```
