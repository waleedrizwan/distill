//! The whole pipeline as one call: text in, summary out.
//!
//! [`distill`] is the simple entry point: it loads the model, summarizes in
//! plain mode and returns the text. To summarize many documents, load a
//! [`Distiller`] once and call [`Distiller::summarize`] for each, which also
//! gives the positions, scores and rewrites behind the text.

use crate::{compress, embed, glossary, lexical, readability, salience, segment, select, split};

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Summarize `text` in plain mode (see [`Options::plain`]) and return the
/// summary as text. Loads the embedding model on every call; use a
/// [`Distiller`] to summarize more than one document.
pub fn distill(text: &str) -> Result<String> {
    Ok(Distiller::new(None)?.summarize(text, &Options::plain())?.to_text(true))
}

/// How long the summary is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Length {
    /// About 10% of the candidate sentences, 3 to 15.
    Auto,
    /// This many sentences.
    Sentences(usize),
    /// This fraction (0..1] of the candidate sentences.
    Ratio(f32),
    /// Up to this many words, context sentences included.
    Words(usize),
    /// This many sentences under each top-level heading.
    PerSection(usize),
}

/// Every knob of the pipeline. `Default` is the command line's defaults;
/// [`Options::plain`] is what `--plain` turns on.
#[derive(Debug, Clone)]
pub struct Options {
    pub length: Length,
    /// Relevance vs. diversity; `None` picks by document length.
    pub lambda: Option<f32>,
    pub lead: f32,
    pub centrality: select::Centrality,
    /// Weight of similarity to earlier text; `None` picks by document length.
    pub backward: Option<f32>,
    pub beta: f32,
    pub section_weight: f32,
    pub penalty: f32,
    pub boost: f32,
    /// Weight of the learned salience score (0 = off).
    pub salience: f32,
    /// Pull in the sentence before a "This...", "However..." opener.
    pub repair: bool,
    /// Prefer plainer sentences when they say about as much.
    pub prefer_plain: bool,
    pub plainness: f32,
    pub compress: bool,
    pub simplify: bool,
    pub simplify_threshold: f32,
    pub split: bool,
    pub glossary: bool,
    pub expand_acronyms: bool,
    pub min_words: usize,
    pub dehyphenate: bool,
    /// Keep every candidate's ranking features in `Summary::selection.features`.
    pub want_features: bool,
    pub batch_size: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            length: Length::Auto,
            lambda: None,
            lead: 0.1,
            centrality: select::Centrality::Directed,
            backward: None,
            beta: 0.3,
            section_weight: 0.4,
            penalty: 0.08,
            boost: 1.5,
            salience: 0.5,
            repair: true,
            prefer_plain: false,
            plainness: 0.4,
            compress: false,
            simplify: false,
            simplify_threshold: 0.95,
            split: false,
            glossary: false,
            expand_acronyms: false,
            min_words: 6,
            dehyphenate: false,
            want_features: false,
            batch_size: 8,
        }
    }
}

impl Options {
    /// Easier reading: plainer sentences, shortened, split, simpler words,
    /// abbreviations spelled out and listed.
    pub fn plain() -> Self {
        Options::default().with_plain()
    }

    /// These options with everything `--plain` turns on.
    pub fn with_plain(mut self) -> Self {
        self.prefer_plain = true;
        self.compress = true;
        self.split = true;
        self.simplify = true;
        self.glossary = true;
        self.expand_acronyms = true;
        self
    }
}

/// The embedding model, loaded once.
pub struct Distiller {
    embedder: embed::Embedder,
    model_dir: PathBuf,
}

/// A summary and what went into it.
pub struct Summary {
    pub doc: segment::Document,
    pub selection: select::Selection,
    /// Rewritten text per sentence (None = shown as in the source).
    display: Vec<Option<String>>,
    /// With `simplify`: (word, replacement, similarity of the sentence after the swap).
    pub substitutions: Vec<Vec<(String, String, f32)>>,
    /// With `glossary`: abbreviations the summary uses without defining.
    pub glossary: Vec<glossary::Abbreviation>,
    pub candidates: usize,
    /// Whether the learned salience score was used, and why not.
    pub salience_note: &'static str,
    pub parse_time: Duration,
    pub embed_time: Duration,
    pub select_time: Duration,
}

impl Distiller {
    /// Load the model from `model_dir`, or from the first of $DISTILL_MODEL,
    /// ./model, <binary folder>/model and ~/.distill/model that has one.
    pub fn new(model_dir: Option<&Path>) -> Result<Self> {
        let model_dir = find_model(model_dir)?;
        let embedder = embed::Embedder::load(&model_dir)?;
        Ok(Distiller { embedder, model_dir })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn embedder(&self) -> &embed::Embedder {
        &self.embedder
    }

    pub fn summarize(&self, text: &str, opts: &Options) -> Result<Summary> {
        let t0 = Instant::now();
        let doc = parse(text, opts);
        let abbreviations = glossary::abbreviations(&doc.sentences.iter().map(|s| s.text.as_str()).collect::<Vec<_>>());
        let candidates: Vec<&segment::Sentence> = doc.sentences.iter().filter(|s| s.candidate).collect();
        if candidates.is_empty() {
            bail!("no usable sentences found in the input");
        }
        let auto = ((candidates.len() as f32 * 0.1).ceil() as usize).clamp(3, 15);
        let (budget, word_budget, per_section) = match opts.length {
            Length::Auto => (auto, None, None),
            Length::Sentences(n) => (n, None, None),
            Length::Ratio(r) => {
                if !(r > 0.0 && r <= 1.0) {
                    bail!("ratio must be between 0 and 1");
                }
                (((candidates.len() as f32 * r).round() as usize).max(1), None, None)
            }
            Length::Words(w) => (auto, Some(w), None),
            Length::PerSection(k) => (auto, None, Some(k.max(1))),
        };
        let t1 = Instant::now();
        let texts: Vec<&str> = candidates.iter().map(|s| s.text.as_str()).collect();
        let vectors = embed_cached(&self.embedder, &texts, opts.batch_size)?;
        let t2 = Instant::now();

        // Defaults for lambda and backward that depend on the document's length.
        let long = select::LengthDefaults::for_candidates(candidates.len());
        let mut params = select::Params {
            budget,
            word_budget,
            lambda: opts.lambda.unwrap_or(long.lambda).clamp(0.0, 1.0),
            lead: opts.lead,
            centrality: opts.centrality,
            backward: opts.backward.unwrap_or(long.backward),
            beta: opts.beta.clamp(0.0, 1.0),
            section_weight: opts.section_weight.clamp(0.0, 1.0),
            dependence_penalty: opts.penalty,
            summary_section_boost: opts.boost.max(0.0),
            repair: opts.repair,
            salience: None,
            salience_weight: opts.salience.clamp(0.0, 1.0),
            want_features: opts.want_features,
            word_cost: None,
            per_section,
            plainness: if opts.prefer_plain { opts.plainness.max(0.0) } else { 0.0 },
            difficulty: opts.prefer_plain.then(|| doc.sentences.iter().map(|s| readability::difficulty(&s.text)).collect()),
        };
        // Compressed text per sentence (None = unchanged). Ranking still embeds the
        // original sentence; only the output and the word budget use this.
        let shortened: Vec<Option<String>> = if opts.compress {
            doc.sentences.iter().map(|s| compress::compress(&s.text)).collect()
        } else {
            vec![None; doc.sentences.len()]
        };
        if opts.compress {
            params.word_cost = Some(
                doc.sentences
                    .iter()
                    .zip(&shortened)
                    .map(|(s, c)| c.as_deref().map(|t| t.split_whitespace().count()).unwrap_or(s.words))
                    .collect(),
            );
        }
        let mut salience_note = "salience head off";
        if params.salience_weight > 0.0 {
            match salience::Head::builtin() {
                Some(h) if h.encoder == self.embedder.encoder() => {
                    params.salience = Some(std::sync::Arc::new(h));
                    salience_note = "salience head on";
                }
                Some(_) => salience_note = "salience head skipped: trained for a different model",
                None => salience_note = "salience head skipped: none built in",
            }
        }
        let selection = select::select(&doc, &vectors, &params);
        let n_candidates = candidates.len();
        let picks = &selection.picks;

        // What each picked sentence will print as: compressed, then simplified.
        let mut display: Vec<Option<String>> = shortened;
        let mut substitutions: Vec<Vec<(String, String, f32)>> = vec![Vec::new(); doc.sentences.len()];
        if opts.simplify {
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
            let vecs = if refs.is_empty() { Vec::new() } else { self.embedder.embed(&refs, opts.batch_size)? };
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
                    if sim >= opts.simplify_threshold {
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
            let pair_vecs =
                if pair_texts.is_empty() { Vec::new() } else { self.embedder.embed(&pair_texts, opts.batch_size)? };
            for (idx, text, mut kept) in combined {
                if kept.is_empty() {
                    continue;
                }
                if kept.len() > 1 {
                    let m = multi.iter().position(|(i, _)| *i == idx).expect("multi");
                    let sim = embed::dot(&pair_vecs[2 * m], &pair_vecs[2 * m + 1]);
                    if sim < opts.simplify_threshold {
                        // Strict '>' keeps the earliest swap on ties.
                        let best = kept.iter().enumerate().fold(0, |b, (k, (_, s))| if *s > kept[b].1 { k } else { b });
                        kept = vec![kept.swap_remove(best)];
                    }
                }
                display[idx] = Some(lexical::apply(&text, &kept.iter().map(|(s, _)| s).collect::<Vec<_>>()));
                substitutions[idx] = kept.into_iter().map(|(s, sim)| (s.from, s.to, (sim * 1e4).round() / 1e4)).collect();
            }
        }
        if opts.split {
            for p in picks {
                let text = display[p.index].clone().unwrap_or_else(|| doc.sentences[p.index].text.clone());
                if let Some(parts) = split::split(&text) {
                    display[p.index] = Some(parts);
                }
            }
        }
        if opts.expand_acronyms {
            let order: Vec<usize> = picks.iter().map(|p| p.index).collect();
            let texts: Vec<String> =
                order.iter().map(|&i| display[i].clone().unwrap_or_else(|| doc.sentences[i].text.clone())).collect();
            for (&i, new) in order.iter().zip(glossary::expand(&texts, &abbreviations)) {
                if new != doc.sentences[i].text && display[i].as_deref() != Some(new.as_str()) {
                    display[i] = Some(new);
                }
            }
        }
        let shown = |i: usize| -> &str { display[i].as_deref().unwrap_or(&doc.sentences[i].text) };
        let glossary: Vec<glossary::Abbreviation> = if opts.glossary {
            glossary::for_summary(&picks.iter().map(|p| shown(p.index)).collect::<Vec<_>>(), &abbreviations)
                .into_iter()
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        let t3 = Instant::now();
        Ok(Summary {
            doc,
            selection,
            display,
            substitutions,
            glossary,
            candidates: n_candidates,
            salience_note,
            parse_time: t1 - t0,
            embed_time: t2 - t1,
            select_time: t3 - t2,
        })
    }
}

/// Split `text` into sections and sentences, as `summarize` does.
pub fn parse(text: &str, opts: &Options) -> segment::Document {
    segment::parse(text, &segment::ParseOptions { min_words: opts.min_words, dehyphenate: opts.dehyphenate })
}

impl Summary {
    /// Sentence `i` of the document as the summary shows it.
    pub fn shown(&self, i: usize) -> &str {
        self.display[i].as_deref().unwrap_or(&self.doc.sentences[i].text)
    }

    /// Whether sentence `i` is shown rewritten (compressed, simplified, split or expanded).
    pub fn rewritten(&self, i: usize) -> bool {
        self.display[i].is_some()
    }

    pub fn source_words(&self) -> usize {
        self.doc.sentences.iter().map(|s| s.words).sum()
    }

    pub fn summary_words(&self) -> usize {
        self.selection.picks.iter().map(|p| self.shown(p.index).split_whitespace().count()).sum()
    }

    /// The summary as text, followed by the glossary if there is one.
    /// Sentences are in document order. Sentences that were adjacent in the
    /// source stay on one line; a new line means text was skipped in between.
    pub fn to_text(&self, headings: bool) -> String {
        let sel = &self.selection;
        let titled: std::collections::BTreeSet<usize> =
            sel.picks.iter().filter(|p| sel.groups[p.group].is_some()).map(|p| p.group).collect();
        let show_headings = headings && titled.len() >= 2;

        let mut out = String::new();
        let mut last_group: Option<usize> = None;
        let mut last_index: Option<usize> = None;
        let mut line = String::new();
        let mut first_output = true;
        for p in &sel.picks {
            let s = &self.doc.sentences[p.index];
            let new_group = last_group != Some(p.group);
            let adjacent = last_index.map(|i| i + 1 == p.index).unwrap_or(false) && !new_group && !s.list_item;
            if !adjacent && !line.is_empty() {
                out.push_str(&line);
                out.push('\n');
                line.clear();
            }
            if new_group {
                if show_headings {
                    if !first_output {
                        out.push('\n');
                    }
                    if let Some(t) = &sel.groups[p.group] {
                        out.push_str(&format!("[{t}]\n"));
                    }
                }
                last_group = Some(p.group);
            }
            first_output = false;
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(self.shown(p.index));
            last_index = Some(p.index);
        }
        if !line.is_empty() {
            out.push_str(&line);
            out.push('\n');
        }
        if !self.glossary.is_empty() {
            out.push_str("\nGlossary\n");
            for a in &self.glossary {
                out.push_str(&format!("  {}: {}\n", a.short, a.long));
            }
        }
        out
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
