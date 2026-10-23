use distill::{compress, embed, lexical, salience, segment, select, split};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Deterministic extractive summarizer.
///
/// Picks the most representative sentences from a document and prints them
/// verbatim, in their original order. No text is generated, so nothing can be
/// invented, and the same input always gives the same output.
#[derive(Parser, Debug)]
#[command(name = "distill", version)]
struct Args {
    /// Text or Markdown file to summarize ("-" or omitted reads stdin)
    file: Option<PathBuf>,

    /// Number of sentences to select
    #[arg(short = 'n', long, conflicts_with = "ratio")]
    sentences: Option<usize>,

    /// Fraction of the document's sentences to select (e.g. 0.1)
    #[arg(short, long)]
    ratio: Option<f32>,

    /// Select sentences up to this many words instead of a sentence count
    /// (context sentences pulled in for readability are extra)
    #[arg(short, long, conflicts_with_all = ["sentences", "ratio"])]
    words: Option<usize>,

    /// Folder with config.json, tokenizer.json and model.safetensors
    /// [default: $DISTILL_MODEL, ./model, <binary folder>/model, ~/.distill/model]
    #[arg(short, long)]
    model_dir: Option<PathBuf>,

    /// Relevance vs. diversity (1.0 = most central sentences, lower = less repetition)
    #[arg(long, default_value_t = 0.7)]
    lambda: f32,

    /// Bonus for sentences that open a section (0 disables)
    #[arg(long, default_value_t = 0.1)]
    lead: f32,

    /// How sentence importance is measured
    #[arg(long, value_enum, default_value_t = select::Centrality::Directed)]
    centrality: select::Centrality,

    /// Directed centrality: weight of similarity to earlier text (later text = 1)
    #[arg(long, default_value_t = 0.3, allow_negative_numbers = true)]
    backward: f32,

    /// Directed centrality: ignore similarities below this point of the min..max range
    #[arg(long, default_value_t = 0.3)]
    beta: f32,

    /// Weight of within-section importance vs whole-document importance
    #[arg(long, default_value_t = 0.4)]
    section_weight: f32,

    /// Penalty for sentences that lean on other text ("However...", "see Figure 2")
    #[arg(long, default_value_t = 0.08)]
    penalty: f32,

    /// Extra budget share for the opening section and abstract/summary/conclusion
    #[arg(long, default_value_t = 1.5)]
    boost: f32,

    /// Weight of the learned salience score against centrality (0 = centrality
    /// only, 1 = salience only). Used only with the model it was trained on.
    #[arg(long, default_value_t = 0.5)]
    salience: f32,

    /// Print every candidate's ranking features as JSON (for eval/train_salience.py)
    #[arg(long, hide = true)]
    dump_features: bool,

    /// Don't pull in the preceding sentence for "This...", "However..." openers
    #[arg(long)]
    no_repair: bool,

    /// Shorten picked sentences by deleting parentheticals, citations,
    /// "which"/"who" clauses, dash asides and openers like "However,"
    /// (nothing is reworded; see src/compress.rs)
    #[arg(long)]
    compress: bool,

    /// Replace hard words with simpler ones from a fixed, reviewed table
    /// (src/lexicon.tsv, from SimplePPDB). A swap is kept only if the
    /// sentence's meaning, as the embedding model sees it, barely moves
    #[arg(long)]
    simplify: bool,

    /// Split long sentences at ", and" / ", but" / ";" when both halves are
    /// full clauses (nothing is added; see src/split.rs)
    #[arg(long)]
    split: bool,

    /// --simplify: least cosine similarity to the original sentence a swap may leave
    #[arg(long, default_value_t = 0.95)]
    simplify_threshold: f32,

    /// Don't print section headings
    #[arg(long)]
    no_headings: bool,

    /// Output JSON with sentence positions and scores
    #[arg(long)]
    json: bool,

    /// Rejoin words hyphenated across line breaks (useful for PDF text)
    #[arg(long)]
    dehyphenate: bool,

    /// Shortest sentence (in words) that can be selected
    #[arg(long, default_value_t = 6)]
    min_words: usize,

    /// Print the detected sections and sentences instead of a summary
    #[arg(long)]
    show_segments: bool,

    /// Print counts and timings to stderr
    #[arg(short, long)]
    verbose: bool,

    #[arg(long, default_value_t = 8, hide = true)]
    batch_size: usize,
}

#[derive(Serialize)]
struct JsonSentence<'a> {
    /// Position of the sentence in the source document (0-based).
    index: usize,
    /// Top-level section heading.
    section: Option<&'a str>,
    /// Nearest heading above the sentence.
    subsection: Option<&'a str>,
    kind: &'static str,
    rank: usize,
    score: f32,
    text: &'a str,
    /// With --compress: the sentence before compression, when it was shortened.
    #[serde(skip_serializing_if = "Option::is_none")]
    original: Option<&'a str>,
    /// With --simplify: (word, replacement, similarity of the sentence after the swap).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    substitutions: Vec<(String, String, f32)>,
}

#[derive(Serialize)]
struct JsonOut<'a> {
    source_sentences: usize,
    candidate_sentences: usize,
    sections: usize,
    summary_words: usize,
    source_words: usize,
    sentences: Vec<JsonSentence<'a>>,
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    let raw = match args.file.as_deref() {
        Some(p) if p != Path::new("-") => {
            let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
            String::from_utf8_lossy(&bytes).into_owned()
        }
        _ => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s).context("reading stdin")?;
            s
        }
    };

    let doc = segment::parse(
        &raw,
        &segment::ParseOptions { min_words: args.min_words, dehyphenate: args.dehyphenate },
    );

    if args.show_segments {
        for (si, sec) in doc.sections.iter().enumerate() {
            if sec.sentences.is_empty() && sec.title.is_none() {
                continue;
            }
            println!(
                "== section {si}: {}{}",
                sec.title.as_deref().unwrap_or("(untitled)"),
                if sec.skip { "  [skipped]" } else { "" }
            );
            for s in &doc.sentences[sec.sentences.clone()] {
                let flag = match (s.candidate, s.list_item) {
                    (true, true) => "c-",
                    (true, false) => "c ",
                    (false, true) => " -",
                    (false, false) => "  ",
                };
                println!("{:>5} p{:<4} {flag} {}", s.index, s.paragraph, s.text);
            }
        }
        return Ok(());
    }

    let candidates: Vec<&segment::Sentence> = doc.sentences.iter().filter(|s| s.candidate).collect();
    if candidates.is_empty() {
        bail!("no usable sentences found in the input");
    }

    let budget = match (args.sentences, args.ratio) {
        (Some(n), _) => n,
        (None, Some(r)) => {
            if !(r > 0.0 && r <= 1.0) {
                bail!("--ratio must be between 0 and 1");
            }
            ((candidates.len() as f32 * r).round() as usize).max(1)
        }
        (None, None) => ((candidates.len() as f32 * 0.1).ceil() as usize).clamp(3, 15),
    };

    let model_dir = find_model(args.model_dir.as_deref())?;
    let t1 = Instant::now();
    let embedder = embed::Embedder::load(&model_dir)?;
    let t2 = Instant::now();
    let texts: Vec<&str> = candidates.iter().map(|s| s.text.as_str()).collect();
    let vectors = embed_cached(&embedder, &texts, args.batch_size)?;
    let t3 = Instant::now();

    let mut params = select::Params {
        budget,
        word_budget: args.words,
        lambda: args.lambda.clamp(0.0, 1.0),
        lead: args.lead,
        centrality: args.centrality,
        backward: args.backward,
        beta: args.beta.clamp(0.0, 1.0),
        section_weight: args.section_weight.clamp(0.0, 1.0),
        dependence_penalty: args.penalty,
        summary_section_boost: args.boost.max(0.0),
        repair: !args.no_repair,
        salience: None,
        salience_weight: args.salience.clamp(0.0, 1.0),
        want_features: args.dump_features,
        word_cost: None,
    };
    // Compressed text per sentence (None = unchanged). Ranking still embeds the
    // original sentence; only the output and the word budget use this.
    let shortened: Vec<Option<String>> = if args.compress {
        doc.sentences.iter().map(|s| compress::compress(&s.text)).collect()
    } else {
        vec![None; doc.sentences.len()]
    };
    if args.compress {
        params.word_cost = Some(
            doc.sentences
                .iter()
                .zip(&shortened)
                .map(|(s, c)| c.as_deref().map(|t| t.split_whitespace().count()).unwrap_or(s.words))
                .collect(),
        );
    }
    let mut head_note = "salience head off";
    if params.salience_weight > 0.0 {
        match salience::Head::builtin() {
            Some(h) if h.encoder == embedder.encoder() => {
                params.salience = Some(std::sync::Arc::new(h));
                head_note = "salience head on";
            }
            Some(_) => head_note = "salience head skipped: trained for a different model",
            None => head_note = "salience head skipped: none built in",
        }
    }
    let selection = select::select(&doc, &vectors, &params);
    if args.dump_features {
        #[derive(Serialize)]
        struct Row<'a> {
            index: usize,
            words: usize,
            text: &'a str,
            features: Vec<f32>,
        }
        let rows: Vec<Row> = selection
            .features
            .iter()
            .map(|(i, f)| Row {
                index: *i,
                words: doc.sentences[*i].words,
                text: &doc.sentences[*i].text,
                // Six significant digits keep the dump small and stable.
                features: f.iter().map(|x| format!("{x:.6e}").parse().unwrap_or(*x)).collect(),
            })
            .collect();
        println!("{}", serde_json::to_string(&serde_json::json!({ "encoder": embedder.encoder(), "candidates": rows }))?);
        return Ok(());
    }
    let picks = &selection.picks;

    // What each picked sentence will print as: compressed, then simplified.
    let mut display: Vec<Option<String>> = shortened.clone();
    let mut swaps_made: Vec<Vec<(String, String, f32)>> = vec![Vec::new(); doc.sentences.len()];
    if args.simplify {
        let base = |i: usize| display[i].clone().unwrap_or_else(|| doc.sentences[i].text.clone());
        let mut jobs: Vec<(usize, String, Vec<lexical::Swap>)> = Vec::new();
        for p in picks {
            let text = base(p.index);
            let c = lexical::candidates(&text);
            if !c.is_empty() {
                jobs.push((p.index, text, c));
            }
        }
        // One embedding batch: each sentence, then each sentence with one swap.
        let mut texts: Vec<String> = Vec::new();
        for (_, text, c) in &jobs {
            texts.push(text.clone());
            for sw in c {
                texts.push(lexical::apply(text, &[sw]));
            }
        }
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let vecs = if refs.is_empty() { Vec::new() } else { embedder.embed(&refs, args.batch_size)? };
        let mut at = 0;
        // (sentence index, text before swaps, swaps that passed with their similarity)
        type Passed = (usize, String, Vec<(lexical::Swap, f32)>);
        let mut combined: Vec<Passed> = Vec::new();
        for (idx, text, c) in jobs {
            let n = c.len();
            let orig = &vecs[at];
            let mut kept: Vec<(lexical::Swap, f32)> = Vec::new();
            for (k, sw) in c.into_iter().enumerate() {
                let sim = embed::dot(orig, &vecs[at + 1 + k]);
                if sim >= args.simplify_threshold {
                    kept.push((sw, sim));
                }
            }
            at += 1 + n;
            combined.push((idx, text, kept));
        }
        // Several swaps in one sentence must also pass together; otherwise
        // keep only the single best one.
        let mut multi: Vec<(usize, String)> = Vec::new();
        for (idx, text, kept) in &combined {
            if kept.len() > 1 {
                multi.push((*idx, lexical::apply(text, &kept.iter().map(|(s, _)| s).collect::<Vec<_>>())));
            }
        }
        let mut pair_texts: Vec<&str> = Vec::new();
        for (idx, all) in &multi {
            let text = &combined.iter().find(|(i, ..)| i == idx).expect("job").1;
            pair_texts.push(text);
            pair_texts.push(all);
        }
        let pair_vecs = if pair_texts.is_empty() { Vec::new() } else { embedder.embed(&pair_texts, args.batch_size)? };
        for (idx, text, mut kept) in combined {
            if kept.is_empty() {
                continue;
            }
            if kept.len() > 1 {
                let m = multi.iter().position(|(i, _)| *i == idx).expect("multi");
                let sim = embed::dot(&pair_vecs[2 * m], &pair_vecs[2 * m + 1]);
                if sim < args.simplify_threshold {
                    // Strict '>' keeps the earliest swap on ties.
                    let best = kept.iter().enumerate().fold(0, |b, (k, (_, s))| if *s > kept[b].1 { k } else { b });
                    kept = vec![kept.swap_remove(best)];
                }
            }
            display[idx] = Some(lexical::apply(&text, &kept.iter().map(|(s, _)| s).collect::<Vec<_>>()));
            swaps_made[idx] = kept.into_iter().map(|(s, sim)| (s.from, s.to, (sim * 1e4).round() / 1e4)).collect();
        }
    }
    if args.split {
        for p in picks {
            let text = display[p.index].clone().unwrap_or_else(|| doc.sentences[p.index].text.clone());
            if let Some(parts) = split::split(&text) {
                display[p.index] = Some(parts);
            }
        }
    }
    let shown = |i: usize| -> &str { display[i].as_deref().unwrap_or(&doc.sentences[i].text) };
    let t4 = Instant::now();

    let source_words: usize = doc.sentences.iter().map(|s| s.words).sum();
    let summary_words: usize = picks.iter().map(|p| shown(p.index).split_whitespace().count()).sum();

    if args.json {
        let out = JsonOut {
            source_sentences: doc.sentences.len(),
            candidate_sentences: candidates.len(),
            sections: doc.sections.iter().filter(|s| !s.sentences.is_empty()).count(),
            summary_words,
            source_words,
            sentences: picks
                .iter()
                .map(|p| {
                    let s = &doc.sentences[p.index];
                    JsonSentence {
                        index: p.index,
                        section: selection.groups[p.group].as_deref(),
                        subsection: doc.sections[s.section].title.as_deref(),
                        kind: match p.kind {
                            select::Kind::Selected => "selected",
                            select::Kind::Context => "context",
                        },
                        rank: p.rank,
                        // Four decimals: enough to rank by, and stable against float noise.
                        score: (p.score * 1e4).round() / 1e4,
                        text: shown(p.index),
                        original: display[p.index].as_ref().map(|_| s.text.as_str()),
                        substitutions: swaps_made[p.index].clone(),
                    }
                })
                .collect(),
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        print_text(&doc, &selection, !args.no_headings, &shown);
    }

    if args.verbose {
        eprintln!(
            "{} sentences ({} candidates) in {} sections -> {} selected + {} context; {} -> {} words ({:.0}%)",
            doc.sentences.len(),
            candidates.len(),
            doc.sections.iter().filter(|s| !s.sentences.is_empty()).count(),
            picks.iter().filter(|p| p.kind == select::Kind::Selected).count(),
            picks.iter().filter(|p| p.kind == select::Kind::Context).count(),
            source_words,
            summary_words,
            100.0 * summary_words as f32 / source_words.max(1) as f32
        );
        if args.simplify {
            let made: usize = swaps_made.iter().map(Vec::len).sum();
            eprintln!("simplify: {made} substitutions made ({} in the table)", lexical::table_size());
        }
        let st = embedder.settings();
        eprintln!(
            "embedding model in {} ({:?} pooling, {} tokens{}), {head_note}",
            model_dir.display(),
            st.pooling,
            st.max_tokens,
            if st.prefix.is_empty() { String::new() } else { format!(", prefix {:?}", st.prefix) }
        );
        eprintln!(
            "parse {:.0?}, model load {:.0?}, embed {:.0?}, select {:.0?}, total {:.0?}",
            t1 - t0,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t4 - t0
        );
    }
    Ok(())
}

/// Sentences are printed in document order. Sentences that were adjacent in
/// the source stay on one line; a new line means text was skipped in between.
fn print_text<'a>(doc: &'a segment::Document, sel: &select::Selection, headings: bool, shown: &dyn Fn(usize) -> &'a str) {
    let titled: std::collections::BTreeSet<usize> = sel
        .picks
        .iter()
        .filter(|p| sel.groups[p.group].is_some())
        .map(|p| p.group)
        .collect();
    let show_headings = headings && titled.len() >= 2;

    let mut last_group: Option<usize> = None;
    let mut last_index: Option<usize> = None;
    let mut line = String::new();
    let mut first_output = true;
    for p in &sel.picks {
        let s = &doc.sentences[p.index];
        let new_group = last_group != Some(p.group);
        let adjacent = last_index.map(|i| i + 1 == p.index).unwrap_or(false) && !new_group && !s.list_item;
        if !adjacent && !line.is_empty() {
            println!("{line}");
            line.clear();
        }
        if new_group {
            if show_headings {
                if !first_output {
                    println!();
                }
                if let Some(t) = &sel.groups[p.group] {
                    println!("[{t}]");
                }
            }
            last_group = Some(p.group);
        }
        first_output = false;
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(shown(p.index));
        last_index = Some(p.index);
    }
    if !line.is_empty() {
        println!("{line}");
    }
}

/// Embeddings are a pure function of the sentence text, so they can be reused.
/// Set DISTILL_CACHE to a folder to skip re-embedding an unchanged document.
fn embed_cached(embedder: &embed::Embedder, texts: &[&str], batch_size: usize) -> Result<Vec<Vec<f32>>> {
    let Some(dir) = std::env::var_os("DISTILL_CACHE").filter(|d| !d.is_empty()).map(PathBuf::from) else {
        return embedder.embed(texts, batch_size);
    };
    // FNV-1a over all sentence texts, seeded with the model's fingerprint so a
    // different model never reuses another model's vectors.
    let mut h: u64 = 0xcbf29ce484222325 ^ embedder.fingerprint();
    for t in texts {
        for b in t.bytes().chain(std::iter::once(0u8)) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    let path = dir.join(format!("{h:016x}-{}.f32", texts.len()));
    if let Ok(bytes) = std::fs::read(&path) {
        let n = texts.len();
        if n > 0 && bytes.len() % (4 * n) == 0 {
            let dim = bytes.len() / (4 * n);
            let flat: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            return Ok(flat.chunks(dim).map(|c| c.to_vec()).collect());
        }
    }
    let vectors = embedder.embed(texts, batch_size)?;
    let mut bytes = Vec::with_capacity(vectors.len() * vectors.first().map(|v| v.len()).unwrap_or(0) * 4);
    for v in &vectors {
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    // Write then rename, so another process reading the cache at the same
    // moment sees the whole file or none of it, never a prefix.
    let _ = std::fs::create_dir_all(&dir);
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    Ok(vectors)
}

fn find_model(explicit: Option<&Path>) -> Result<PathBuf> {
    let mut tried: Vec<PathBuf> = Vec::new();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = explicit {
        candidates.push(p.to_path_buf());
    } else {
        if let Ok(p) = std::env::var("DISTILL_MODEL") {
            candidates.push(PathBuf::from(p));
        }
        candidates.push(PathBuf::from("model"));
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("model"));
                // target/release/distill -> project root
                if let Some(root) = dir.parent().and_then(|d| d.parent()) {
                    candidates.push(root.join("model"));
                }
            }
        }
        if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            candidates.push(PathBuf::from(home).join(".distill").join("model"));
        }
    }
    for c in candidates {
        if c.join("model.safetensors").is_file() && c.join("tokenizer.json").is_file() && c.join("config.json").is_file() {
            return Ok(c);
        }
        tried.push(c);
    }
    bail!(
        "embedding model not found. Run ./fetch-model.sh, or pass --model-dir.\nLooked in:\n{}",
        tried.iter().map(|p| format!("  {}", p.display())).collect::<Vec<_>>().join("\n")
    )
}
