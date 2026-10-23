use distill::pipeline::{self, Distiller, Length, Options};
use distill::{glossary, lexical, select};

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

    /// Summarize section by section: N sentences (default 2; `--by-section=3`) under each
    /// top-level heading, for reading a long document part by part
    #[arg(long, value_name = "N", num_args = 0..=1, default_missing_value = "2", require_equals = true,
          conflicts_with_all = ["sentences", "ratio", "words"])]
    by_section: Option<usize>,

    /// Folder with config.json, tokenizer.json and model.safetensors
    /// [default: $DISTILL_MODEL, ./model, <binary folder>/model, ~/.distill/model]
    #[arg(short, long)]
    model_dir: Option<PathBuf>,

    /// Relevance vs. diversity (1.0 = most central sentences, lower = less
    /// repetition). Default 0.7, rising to 0.9 for long documents
    #[arg(long)]
    lambda: Option<f32>,

    /// Bonus for sentences that open a section (0 disables)
    #[arg(long, default_value_t = 0.1)]
    lead: f32,

    /// How sentence importance is measured
    #[arg(long, value_enum, default_value_t = select::Centrality::Directed)]
    centrality: select::Centrality,

    /// Directed centrality: weight of similarity to earlier text (later text =
    /// 1). Default 0.3, rising to 1 (no direction) for long documents
    #[arg(long, allow_negative_numbers = true)]
    backward: Option<f32>,

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

    /// After the summary, list the abbreviations it uses without spelling
    /// out, with the document's own definition ("LLR: log-likelihood ratio")
    #[arg(long)]
    glossary: bool,

    /// Spell out the first use of each abbreviation the summary leaves
    /// undefined, with the document's words: "log-likelihood ratio (LLR)"
    #[arg(long)]
    expand_acronyms: bool,

    /// Easier reading: prefer plainer sentences (shorter, shorter words) when
    /// they say about as much, and turn on --compress, --split, --simplify,
    /// --glossary and --expand-acronyms
    #[arg(long)]
    plain: bool,

    /// --plain: weight of reading difficulty against relevance
    #[arg(long, hide = true, default_value_t = 0.4)]
    plainness: f32,

    /// Print every abbreviation definition found in the document and exit
    #[arg(long, hide = true)]
    dump_abbreviations: bool,

    /// --simplify: least cosine similarity to the original sentence a swap may leave
    #[arg(long, default_value_t = 0.95)]
    simplify_threshold: f32,

    /// Don't print section headings
    #[arg(long)]
    no_headings: bool,

    /// Output JSON with sentence positions and scores
    #[arg(long)]
    json: bool,

    /// Rejoin words hyphenated across line breaks (useful for text copied from a PDF)
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
    /// With --glossary: abbreviations the summary uses without defining.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    glossary: Vec<JsonGloss<'a>>,
}

#[derive(Serialize)]
struct JsonGloss<'a> {
    term: &'a str,
    definition: &'a str,
    source: &'static str,
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

    let bytes = match args.file.as_deref() {
        Some(p) if p != Path::new("-") => std::fs::read(p).with_context(|| format!("reading {}", p.display()))?,
        _ => {
            let mut b = Vec::new();
            std::io::stdin().read_to_end(&mut b).context("reading stdin")?;
            b
        }
    };
    if bytes.starts_with(b"%PDF-") {
        bail!("distill reads text only; convert the PDF first, e.g. `pdftotext -layout file.pdf -`");
    }
    let raw = String::from_utf8_lossy(&bytes);
    let opts = options(&args)?;

    if args.dump_abbreviations || args.show_segments {
        let doc = pipeline::parse(&raw, &opts);
        if args.dump_abbreviations {
            let texts: Vec<&str> = doc.sentences.iter().map(|s| s.text.as_str()).collect();
            for a in glossary::abbreviations(&texts) {
                println!("{}\t{}", a.short, a.long);
            }
            return Ok(());
        }
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

    let t1 = Instant::now();
    let distiller = Distiller::new(args.model_dir.as_deref())?;
    let t2 = Instant::now();
    let sum = distiller.summarize(&raw, &opts)?;
    let t3 = Instant::now();
    let doc = &sum.doc;
    let selection = &sum.selection;

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
        let encoder = distiller.embedder().encoder();
        println!("{}", serde_json::to_string(&serde_json::json!({ "encoder": encoder, "candidates": rows }))?);
        return Ok(());
    }
    let picks = &selection.picks;
    let source_words = sum.source_words();
    let summary_words = sum.summary_words();

    if args.json {
        let out = JsonOut {
            source_sentences: doc.sentences.len(),
            candidate_sentences: sum.candidates,
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
                        text: sum.shown(p.index),
                        original: sum.rewritten(p.index).then_some(s.text.as_str()),
                        substitutions: sum.substitutions[p.index].clone(),
                    }
                })
                .collect(),
            glossary: sum
                .glossary
                .iter()
                .map(|a| JsonGloss { term: &a.short, definition: &a.long, source: "document" })
                .collect(),
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        print!("{}", sum.to_text(!args.no_headings));
    }

    if args.verbose {
        eprintln!(
            "{} sentences ({} candidates) in {} sections -> {} selected + {} context; {} -> {} words ({:.0}%)",
            doc.sentences.len(),
            sum.candidates,
            doc.sections.iter().filter(|s| !s.sentences.is_empty()).count(),
            picks.iter().filter(|p| p.kind == select::Kind::Selected).count(),
            picks.iter().filter(|p| p.kind == select::Kind::Context).count(),
            source_words,
            summary_words,
            100.0 * summary_words as f32 / source_words.max(1) as f32
        );
        if opts.simplify {
            let made: usize = sum.substitutions.iter().map(Vec::len).sum();
            eprintln!("simplify: {made} substitutions made ({} in the table)", lexical::table_size());
        }
        let st = distiller.embedder().settings();
        eprintln!(
            "embedding model in {} ({:?} pooling, {} tokens{}), {}",
            distiller.model_dir().display(),
            st.pooling,
            st.max_tokens,
            if st.prefix.is_empty() { String::new() } else { format!(", prefix {:?}", st.prefix) },
            sum.salience_note
        );
        eprintln!(
            "read {:.0?}, model load {:.0?}, parse {:.0?}, embed {:.0?}, select {:.0?}, total {:.0?}",
            t1 - t0,
            t2 - t1,
            sum.parse_time,
            sum.embed_time,
            sum.select_time,
            t3 - t0
        );
    }
    Ok(())
}

/// The command line's flags as library options.
fn options(args: &Args) -> Result<Options> {
    let length = match (args.sentences, args.ratio, args.words, args.by_section) {
        (Some(n), ..) => Length::Sentences(n),
        (_, Some(r), ..) => {
            if !(r > 0.0 && r <= 1.0) {
                bail!("--ratio must be between 0 and 1");
            }
            Length::Ratio(r)
        }
        (_, _, Some(w), _) => Length::Words(w),
        (.., Some(k)) => Length::PerSection(k),
        _ => Length::Auto,
    };
    let opts = Options {
        length,
        lambda: args.lambda,
        lead: args.lead,
        centrality: args.centrality,
        backward: args.backward,
        beta: args.beta,
        section_weight: args.section_weight,
        penalty: args.penalty,
        boost: args.boost,
        salience: args.salience,
        repair: !args.no_repair,
        prefer_plain: false,
        plainness: args.plainness,
        compress: args.compress,
        simplify: args.simplify,
        simplify_threshold: args.simplify_threshold,
        split: args.split,
        glossary: args.glossary,
        expand_acronyms: args.expand_acronyms,
        min_words: args.min_words,
        dehyphenate: args.dehyphenate,
        want_features: args.dump_features,
        batch_size: args.batch_size,
    };
    Ok(if args.plain { opts.with_plain() } else { opts })
}
