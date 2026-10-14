//! Lexical simplification from a fixed table (`--simplify`).
//!
//! Swaps a word for a simpler one from `src/lexicon.tsv`, a reviewed subset of
//! SimplePPDB (Pavlick and Callison-Burch, 2016; CC BY 3.0 US). Every possible
//! swap comes from that table, so each one can be read and audited. This module
//! only proposes swaps; `main.rs` keeps a swap only if the sentence's embedding
//! stays close to the original's, which filters out wrong word senses.
//!
//! Handled: plural nouns, verb forms (-s, -ed, -ing) with a short list of
//! irregular replacements, the article before a swapped word ("an enormous" ->
//! "a huge"), and capitalisation at the start of a sentence. Left alone: proper
//! nouns, words in compounds ("long-standing"), quotations, and legal operative
//! text, where a synonym can change what the law says.

use std::collections::BTreeMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pos {
    Noun,
    Verb,
    Adj,
    Adv,
}

/// One proposed replacement, as a byte range of the sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Swap {
    pub start: usize,
    pub end: usize,
    pub from: String,
    pub to: String,
}

fn table() -> &'static BTreeMap<String, Vec<(Pos, String)>> {
    static T: OnceLock<BTreeMap<String, Vec<(Pos, String)>>> = OnceLock::new();
    T.get_or_init(|| {
        let mut m: BTreeMap<String, Vec<(Pos, String)>> = BTreeMap::new();
        for line in include_str!("lexicon.tsv").lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let pos = match f[0] {
                "n" => Pos::Noun,
                "v" => Pos::Verb,
                "a" => Pos::Adj,
                "r" => Pos::Adv,
                _ => continue,
            };
            m.entry(f[1].to_string()).or_default().push((pos, f[2].to_string()));
        }
        m
    })
}

/// Number of substitutions in the built-in table.
pub fn table_size() -> usize {
    table().values().map(Vec::len).sum()
}

// Replacement words whose forms do not follow the regular rules:
// (lemma, past, past participle, -ing). An empty past means "-ed forms are
// ambiguous, do not swap them".
const IRREGULAR: &[(&str, &str, &str, &str)] = &[
    ("say", "said", "said", "saying"),
    ("understand", "understood", "understood", "understanding"),
    ("find", "found", "found", "finding"),
    ("grow", "", "", "growing"),
    ("break", "", "", "breaking"),
    ("commit", "committed", "committed", "committing"),
    ("die", "died", "died", "dying"),
    ("keep", "kept", "kept", "keeping"),
    ("build", "built", "built", "building"),
    ("spend", "spent", "spent", "spending"),
];
const IRREGULAR_PLURAL: &[(&str, &str)] = &[("man", "men"), ("woman", "women"), ("child", "children"), ("person", "people")];

fn is_vowel(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u')
}

fn plural(w: &str) -> String {
    if let Some(&(_, p)) = IRREGULAR_PLURAL.iter().find(|(s, _)| *s == w) {
        return p.to_string();
    }
    let b = w.as_bytes();
    if w.ends_with('s') || w.ends_with('x') || w.ends_with('z') || w.ends_with("ch") || w.ends_with("sh") {
        format!("{w}es")
    } else if b.len() > 1 && w.ends_with('y') && !is_vowel(b[b.len() - 2]) {
        format!("{}ies", &w[..w.len() - 1])
    } else {
        format!("{w}s")
    }
}

/// -s, -ed and -ing forms (regular rules, or the irregular table).
fn verb_forms(w: &str) -> [String; 3] {
    if let Some(&(_, past, _, ing)) = IRREGULAR.iter().find(|(s, ..)| *s == w) {
        return [plural(w), past.to_string(), ing.to_string()];
    }
    let b = w.as_bytes();
    let cons_y = b.len() > 1 && w.ends_with('y') && !is_vowel(b[b.len() - 2]);
    let past = if w.ends_with('e') {
        format!("{w}d")
    } else if cons_y {
        format!("{}ied", &w[..w.len() - 1])
    } else {
        format!("{w}ed")
    };
    let ing = if w.ends_with('e') && !w.ends_with("ee") { format!("{}ing", &w[..w.len() - 1]) } else { format!("{w}ing") };
    [plural(w), past, ing]
}

/// The replacement for `word` (lowercase), if the table has one for it or
/// for the lemma it is a regular form of.
fn replacement(word: &str) -> Option<String> {
    let t = table();
    if let Some(entries) = t.get(word) {
        // The base form: the same replacement whatever the part of speech.
        return entries.first().map(|(_, to)| to.clone());
    }
    // Inflected forms: find a lemma whose regular form is `word`. The table is
    // a BTreeMap, so lemmas are tried in a fixed order.
    for (lemma, entries) in t.iter().filter(|(l, _)| word.starts_with(&l[..l.len().min(3)])) {
        for (pos, to) in entries {
            match pos {
                Pos::Noun if plural(lemma) == word => return Some(plural(to)),
                Pos::Verb => {
                    let from = verb_forms(lemma);
                    let into = verb_forms(to);
                    for k in 0..3 {
                        if from[k] == word && !into[k].is_empty() {
                            return Some(into[k].clone());
                        }
                    }
                }
                _ => {}
            }
        }
    }
    None
}

fn capitalise(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

/// Swaps the table allows in `text`, left to right, without overlaps.
pub fn candidates(text: &str) -> Vec<Swap> {
    if crate::compress::is_legal(text) {
        return Vec::new();
    }
    let bytes = text.as_bytes();
    let mut words: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
            words.push((start, i));
        } else {
            // Skip a whole non-ASCII character, not just one byte of it.
            i += text[i..].chars().next().map(char::len_utf8).unwrap_or(1);
        }
    }

    let mut out: Vec<Swap> = Vec::new();
    for (k, &(s, e)) in words.iter().enumerate() {
        let w = &text[s..e];
        let first = k == 0;
        let capital = w.chars().next().is_some_and(char::is_uppercase);
        // Proper nouns and acronyms stay; so do parts of compounds and contractions.
        if (capital && !first) || w.len() > 1 && w.chars().all(|c| c.is_uppercase()) {
            continue;
        }
        let joined = |c: Option<char>| c.is_some_and(|c| c == '-' || c == '\'' || c == '\u{2019}' || c.is_ascii_digit());
        if joined(text[..s].chars().next_back()) || joined(text[e..].chars().next()) {
            continue;
        }
        if crate::compress::inside_quote(text, s) {
            continue;
        }
        let lower = w.to_lowercase();
        let Some(mut to) = replacement(&lower) else { continue };
        if capital {
            to = capitalise(&to);
        }
        let mut swap = Swap { start: s, end: e, from: w.to_string(), to };
        // "an enormous" -> "a huge": fix the article right before the word.
        if k > 0 {
            let (ps, pe) = words[k - 1];
            let art = &text[ps..pe];
            let between = &text[pe..s];
            if (art.eq_ignore_ascii_case("a") || art.eq_ignore_ascii_case("an")) && between == " " {
                let vowel = swap.to.as_bytes().first().is_some_and(|c| is_vowel(c.to_ascii_lowercase()));
                let mut new_art = if vowel { "an" } else { "a" }.to_string();
                if art.starts_with('A') {
                    new_art = capitalise(&new_art);
                }
                if new_art != art {
                    swap = Swap {
                        start: ps,
                        end: e,
                        from: text[ps..e].to_string(),
                        to: format!("{new_art} {}", swap.to),
                    };
                }
            }
        }
        if out.last().is_none_or(|p| p.end <= swap.start) {
            out.push(swap);
        }
    }
    out
}

/// `text` with the given swaps applied (they must not overlap).
pub fn apply(text: &str, swaps: &[&Swap]) -> String {
    let mut sorted: Vec<&&Swap> = swaps.iter().collect();
    sorted.sort_by_key(|s| s.start);
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for s in sorted {
        out.push_str(&text[last..s.start]);
        out.push_str(&s.to);
        last = s.end;
    }
    out.push_str(&text[last..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(text: &str) -> String {
        let c = candidates(text);
        apply(text, &c.iter().collect::<Vec<_>>())
    }

    #[test]
    fn table_loads() {
        assert!(table_size() > 100);
    }

    #[test]
    fn swaps_and_inflection() {
        assert_eq!(all("The project will commence next spring."), "The project will start next spring.");
        assert_eq!(all("Work commenced in March and the crew commences again today."), "Work started in March and the crew starts again today.");
        assert_eq!(all("They are commencing the survey."), "They are starting the survey.");
        assert_eq!(all("Two automobiles were damaged."), "Two cars were damaged.");
        assert_eq!(all("Subsequently, the plan was approved."), "Later, the plan was approved.");
        assert_eq!(all("It was an enormous project."), "It was a huge project.");
        assert_eq!(all("Investigators comprehended the risk."), "Investigators understood the risk.");
        assert_eq!(all("The firm retained its lawyers and expended its budget."), "The firm kept its lawyers and spent its budget.");
        assert_eq!(all("Numerous residents reside nearby."), "Many residents live nearby.");
    }

    #[test]
    fn leaves_alone() {
        // Proper noun, compound, quotation, legal operative text.
        for s in [
            "The band signed with Enormous Records last year.",
            "The long-standing dispute was resolved peacefully in court.",
            "\"We will commence the work tomorrow,\" she said.",
            "The Secretary shall commence proceedings within 30 days.",
        ] {
            assert_eq!(all(s), s, "changed: {s}");
        }
    }

    #[test]
    fn table_is_well_formed() {
        // The TSV is regenerated by a script and then edited by hand, so check
        // every row: a bad row would otherwise be skipped silently or panic.
        let mut seen = std::collections::BTreeSet::new();
        for (n, line) in include_str!("lexicon.tsv").lines().enumerate() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            assert!(f.len() >= 3, "line {}: too few fields: {line:?}", n + 1);
            assert!(matches!(f[0], "n" | "v" | "a" | "r"), "line {}: bad part of speech {:?}", n + 1, f[0]);
            for w in [f[1], f[2]] {
                assert!(!w.is_empty() && w.bytes().all(|b| b.is_ascii_lowercase()), "line {}: {w:?} is not a lowercase word", n + 1);
            }
            assert_ne!(f[1], f[2], "line {}: maps a word to itself", n + 1);
            assert!(seen.insert((f[0], f[1])), "line {}: duplicate entry for {} {}", n + 1, f[0], f[1]);
        }
    }

    #[test]
    fn inflection_rules() {
        assert_eq!(plural("box"), "boxes");
        assert_eq!(plural("church"), "churches");
        assert_eq!(plural("city"), "cities");
        assert_eq!(plural("day"), "days");
        assert_eq!(plural("child"), "children");
        assert_eq!(verb_forms("start"), ["starts", "started", "starting"]);
        assert_eq!(verb_forms("live"), ["lives", "lived", "living"]);
        assert_eq!(verb_forms("try"), ["tries", "tried", "trying"]);
        assert_eq!(verb_forms("agree"), ["agrees", "agreed", "agreeing"]);
        assert_eq!(verb_forms("keep"), ["keeps", "kept", "keeping"]);
        // Ambiguous irregular past: no -ed form, so that form is never produced.
        assert_eq!(verb_forms("grow")[1], "");
    }

    #[test]
    fn swaps_are_consistent_with_the_text() {
        // `apply` and the embedding check in main.rs rely on these invariants.
        let text = "Numerous automobiles were damaged. Subsequently, an enormous crowd of residents — café owners included — commenced protests. They retained lawyers.";
        let c = candidates(text);
        assert!(c.len() >= 4, "{c:#?}");
        for w in c.windows(2) {
            assert!(w[0].end <= w[1].start, "overlap or out of order: {w:?}");
        }
        for s in &c {
            assert!(text.is_char_boundary(s.start) && text.is_char_boundary(s.end));
            assert_eq!(&text[s.start..s.end], s.from);
            assert_ne!(s.from, s.to);
        }
        // Order of the swaps passed in does not matter, and none is a no-op.
        let fwd: Vec<&Swap> = c.iter().collect();
        let rev: Vec<&Swap> = c.iter().rev().collect();
        assert_eq!(apply(text, &fwd), apply(text, &rev));
        assert_eq!(apply(text, &[]), text);
        // Each swap applied alone changes only its own span.
        for s in &c {
            let out = apply(text, &[s]);
            assert_eq!(&out[..s.start], &text[..s.start]);
            assert!(out.ends_with(&text[s.end..]));
        }
    }

    #[test]
    fn articles_and_capitals() {
        // Sentence-initial words keep their capital; a capitalised article is fixed too.
        assert_eq!(all("Numerous people came."), "Many people came.");
        assert_eq!(all("An enormous wave hit the coast."), "A huge wave hit the coast.");
        // The article is only rewritten when the swap changes which one is needed.
        assert_eq!(all("It was a numerous group."), "It was a many group.");
    }

    #[test]
    fn leaves_tokens_alone() {
        for s in [
            // Acronym, contraction, digits attached, hyphenated compound.
            "The NUMEROUS project ran for years in town.",
            "It's commence'd by the crew in error.",
            "Model commence2 was used in the trial.",
            "The pre-commence checks ran for hours.",
        ] {
            assert_eq!(all(s), s, "changed: {s}");
        }
        // Non-ASCII text around a swap is preserved byte for byte.
        assert_eq!(all("Résumé reviewers reside in Zürich."), "Résumé reviewers live in Zürich.");
    }
}
