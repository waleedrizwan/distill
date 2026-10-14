//! proptest strategies shared by the property tests: text shaped like what
//! distill reads, with the asides, openers, quotations, labels and layout the
//! rules key on, plus arbitrary strings for robustness. Test-only.

use proptest::prelude::*;
use proptest::sample::select;

const PLAIN: &[&str] = &[
    "the", "a", "an", "of", "to", "in", "on", "for", "with", "by", "from", "at", "as", "council", "river", "work",
    "plan", "report", "agency", "secretary", "bridge", "students", "officials", "residents", "project", "survey", "law",
    "court", "year", "month", "season", "water", "wall", "cost", "million", "data", "results", "authors", "group",
    "term", "section", "north", "bank", "old", "new", "long", "short", "large", "small", "five", "three", "voted",
    "said", "found", "approved", "issued", "restore", "begin", "finished", "heard", "ruled", "held", "told",
    "promised", "worried", "expects", "will", "would", "may", "shall", "must", "can", "is", "are", "was", "were",
    "has", "have", "had", "been", "be", "not", "also", "only", "very", "more", "than", "that", "this", "these",
    "those", "it", "they", "he", "she", "we", "who", "which", "whose", "and", "but", "so", "or", "because", "after",
    "before", "when", "while", "if", "means", "amended", "commence", "commenced", "enormous", "automobile",
    "subsequently", "utilize", "terminate", "approximately", "demonstrate", "purchase", "additional", "other", "than",
    "except", "including",
];
const NAMES: &[&str] = &["Smith", "Oklahoma", "Geneva", "Tuesday", "March", "Congress", "Boren", "Vallejo", "California"];
const ACRONYMS: &[&str] = &["CNN", "FDA", "USDA", "ICC", "ACL", "GAO"];
const OPENERS: &[&str] = &["However", "Moreover", "In addition", "Meanwhile", "As a result", "Finally", "Of course"];
const LABELS: &[&str] = &["(a)", "(b)", "(1)", "(12)", "(iv)", "(B)", "(A)"];
const ENDS: &[&str] = &[".", ".", ".", ".", "?", "!", ".\"", ".''", ".\u{201D}", "\u{2026}", ""];
const HEADINGS: &[&str] = &[
    "# Introduction", "## Methods and data", "1. Scope", "2. Definitions", "SEC. 2. SHORT TITLE.", "RESULTS",
    "12. Governing Law.", "References", "Acknowledgements", "Related articles",
];
const BOILERPLATE: &[&str] = &[
    "Read more: the full story here", "Click here to subscribe to our newsletter", "Share this article",
    "Follow us on Twitter", "Copyright 2024 All rights reserved",
];
const NOISE: &[&str] = &[
    "\"", "\"", "\u{201C}", "\u{201D}", "``", "''", "'", "\u{2019}", "(", ")", "[", "]", "--", "\u{2014}", "\u{2013}",
    "-", ",", ";", ":", ".", "..", "...", "\u{2026}", "?", "!", " ", " ", "  ", "\n", "\n\n", "\r\n", "\r", "\t",
    "\u{000C}", "\u{00AD}", "\u{00A0}", "e.g.", "Fig.", "et al.", "v.", "Mr.", "U.S.C.", "\u{00A7} 1983", "No.",
    "3.", "#", "# ", "- ", "* ", "1. ", "(a)", "[3]", "[12-15]", "the", "Which", "However,", "said", "shall",
    "\u{00E9}", "\u{00DF}", "\u{4E2D}\u{6587}", "\u{0627}\u{0644}", "e\u{0301}", "\u{1F600}", "\u{200B}", "\u{FEFF}",
    "A", "I", "a", "0", "42",
];

fn capitalise(w: &str) -> String {
    let mut c = w.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

fn word() -> impl Strategy<Value = String> {
    prop_oneof![
        10 => select(PLAIN).prop_map(str::to_string),
        1 => select(NAMES).prop_map(str::to_string),
        1 => select(ACRONYMS).prop_map(str::to_string),
        1 => (1u32..3000).prop_map(|n| n.to_string()),
        1 => select(PLAIN).prop_map(capitalise),
        1 => (select(PLAIN), select(PLAIN)).prop_map(|(a, b)| format!("{a}-{b}")),
        1 => select(&["it's", "he's", "they're", "we've", "didn't", "o'clock"]).prop_map(str::to_string),
    ]
}

/// `n` words joined by single spaces.
pub fn words(n: std::ops::Range<usize>) -> impl Strategy<Value = String> {
    prop::collection::vec(word(), n).prop_map(|v| v.join(" "))
}

/// The spans the compression and splitting rules look for.
fn aside() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => words(1..6).prop_map(|w| format!("({w})")),
        2 => (1u16..40, 0u16..3).prop_map(|(a, b)| if b == 0 { format!("[{a}]") } else { format!("[{a}-{}]", a + b) }),
        1 => words(2..8).prop_map(|w| format!("-- {w} --")),
        1 => words(2..8).prop_map(|w| format!("\u{2014} {w} \u{2014}")),
        3 => (select(&["which", "who", "whose", "whom"]), words(2..12)).prop_map(|(r, w)| format!(", {r} {w},")),
        2 => words(2..12).prop_map(|w| format!("\"{w}.\"")),
        1 => words(2..12).prop_map(|w| format!("\u{201C}{w}\u{201D}")),
        1 => (select(LABELS), words(2..12)).prop_map(|(l, w)| format!("``{l} {w}.''")),
        1 => select(LABELS).prop_map(str::to_string),
    ]
}

fn token() -> impl Strategy<Value = String> {
    prop_oneof![
        12 => word(),
        2 => word().prop_map(|w| format!("{w},")),
        1 => word().prop_map(|w| format!("{w};")),
        1 => word().prop_map(|w| format!("{w}:")),
        3 => aside(),
    ]
}

/// One sentence: an optional dateline or opener, 1 to 40 tokens, an ending.
pub fn sentence() -> impl Strategy<Value = String> {
    (
        prop::option::weighted(0.15, select(&["(CNN)", "(CNN) -- ", "WASHINGTON (Reuters) - "])),
        prop::option::weighted(0.2, select(OPENERS)),
        prop::collection::vec(token(), 1..40),
        select(ENDS),
    )
        .prop_map(|(dateline, opener, toks, end)| {
            let mut s = String::new();
            if let Some(d) = dateline {
                s.push_str(d);
            }
            if let Some(o) = opener {
                s.push_str(o);
                s.push_str(", ");
            }
            let body = toks.join(" ");
            // Capitalise the first word unless an opener already starts the sentence.
            s.push_str(&if s.is_empty() { capitalise(&body) } else { body });
            s.push_str(end);
            s
        })
}

/// One to six sentences separated by single spaces.
pub fn paragraph() -> impl Strategy<Value = String> {
    prop::collection::vec(sentence(), 1..6).prop_map(|v| v.join(" "))
}

fn list() -> impl Strategy<Value = String> {
    (
        words(3..10),
        select(&["- ", "* ", "1. ", "(a) ", "(1) ", "(i) "]),
        prop::collection::vec(words(2..20), 1..6),
        prop::bool::ANY,
    )
        .prop_map(|(lead, marker, items, numbered)| {
            let mut out = format!("{}:", capitalise(&lead));
            for (i, item) in items.iter().enumerate() {
                let m = if numbered && marker == "1. " { format!("{}. ", i + 1) } else { marker.to_string() };
                let sep = if i + 1 == items.len() { "." } else { ";" };
                out.push_str(&format!("\n{m}{item}{sep}"));
            }
            out
        })
}

/// A paragraph hard-wrapped at about eight words, sometimes breaking a word
/// with a hyphen the way PDF text does.
fn wrapped() -> impl Strategy<Value = String> {
    (paragraph(), prop::bool::ANY).prop_map(|(p, hyphenate)| {
        let mut out = String::new();
        for (i, w) in p.split_whitespace().enumerate() {
            if i > 0 {
                out.push(if i % 8 == 0 { '\n' } else { ' ' });
            }
            if hyphenate && i % 11 == 10 && w.len() > 5 && w.chars().all(|c| c.is_ascii_alphabetic()) {
                let (a, b) = w.split_at(3);
                out.push_str(&format!("{a}-\n{b}"));
            } else {
                out.push_str(w);
            }
        }
        out
    })
}

fn bibliography() -> impl Strategy<Value = String> {
    prop::collection::vec((select(NAMES), words(3..9), 1990u16..2025), 1..4).prop_map(|entries| {
        entries
            .iter()
            .enumerate()
            .map(|(i, (n, t, y))| format!("[{}] {n}, J. A. {}. Journal, {y}.", i + 1, capitalise(t)))
            .collect::<Vec<_>>()
            .join(" ")
    })
}

/// A document: paragraphs, headings, lists, wrapped text, boilerplate and
/// references, separated by blank lines (occasionally by a single newline).
pub fn document() -> impl Strategy<Value = String> {
    let block = prop_oneof![
        6 => paragraph(),
        2 => select(HEADINGS).prop_map(str::to_string),
        1 => words(1..5).prop_map(|w| format!("# {}", capitalise(&w))),
        2 => list(),
        2 => wrapped(),
        1 => select(BOILERPLATE).prop_map(str::to_string),
        1 => bibliography(),
    ];
    (prop::collection::vec(block, 1..8), prop::collection::vec(select(&["\n\n", "\n\n", "\n\n", "\n", "\n\n\n"]), 7))
        .prop_map(|(blocks, seps)| {
            let mut out = String::new();
            for (i, b) in blocks.iter().enumerate() {
                if i > 0 {
                    out.push_str(seps[i - 1]);
                }
                out.push_str(b);
            }
            out
        })
}

/// Arbitrary text: printable Unicode, or runs of the punctuation and control
/// characters the splitter cares about, sometimes wrapped around a real sentence.
pub fn messy() -> impl Strategy<Value = String> {
    let noise = prop::collection::vec(select(NOISE), 0..40).prop_map(|v| v.concat());
    prop_oneof![
        2 => "[\\PC\\n\\r\\t]{0,200}",
        2 => noise.clone(),
        1 => (noise.clone(), sentence(), noise).prop_map(|(a, s, b)| format!("{a}{s}{b}")),
    ]
}
