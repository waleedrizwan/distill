//! Definitions the document gives for its own abbreviations.
//!
//! "log-likelihood ratio (LLR)" and "LLR (log-likelihood ratio)" are found
//! with the method of Schwartz and Hearst ("A simple algorithm for
//! identifying abbreviation definitions in biomedical text", PSB 2003): the
//! short form's letters must appear, in order, in the words before (or inside)
//! the brackets, its first letter at the start of a word, within a window of
//! min(|SF| + 5, 2|SF|) words. The first definition of each short form wins.
//!
//! A summary often keeps "LLR" and drops the sentence that spelled it out.
//! `for_summary` lists the abbreviations a summary uses without defining
//! them; every definition is a span of the document's own text (with runs of
//! whitespace as one space, which is how distill holds its sentences).

use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Abbreviation {
    pub short: String,
    pub long: String,
}

/// Every abbreviation defined in `sentences`, in order of first definition.
pub fn abbreviations<S: AsRef<str>>(sentences: &[S]) -> Vec<Abbreviation> {
    let mut out: Vec<Abbreviation> = Vec::new();
    for s in sentences {
        for a in in_sentence(s.as_ref()) {
            if !out.iter().any(|o| o.short == a.short) {
                out.push(a);
            }
        }
    }
    out
}

/// The abbreviations `summary` uses that it does not define itself, in the
/// order they first appear in it.
pub fn for_summary<'a, S: AsRef<str>>(summary: &[S], defs: &'a [Abbreviation]) -> Vec<&'a Abbreviation> {
    let text: String = summary.iter().map(|s| s.as_ref()).collect::<Vec<_>>().join(" ");
    let lower = text.to_lowercase();
    let mut used: Vec<(usize, &Abbreviation)> = defs
        .iter()
        .filter_map(|d| {
            let at = find_word(&text, &d.short)?;
            // Already spelled out somewhere in the summary.
            if lower.contains(&d.long.to_lowercase()) {
                return None;
            }
            Some((at, d))
        })
        .collect();
    used.sort_by_key(|(at, _)| *at);
    used.into_iter().map(|(_, d)| d).collect()
}

/// Spell out the first use of each abbreviation the summary leaves
/// undefined: "LLR" becomes "log-likelihood ratio (LLR)". Every inserted word
/// is the document's own. Returns the sentences, changed or not.
pub fn expand<S: AsRef<str>>(summary: &[S], defs: &[Abbreviation]) -> Vec<String> {
    let mut out: Vec<String> = summary.iter().map(|s| s.as_ref().to_string()).collect();
    for d in for_summary(summary, defs) {
        if let Some((i, at)) = out.iter().enumerate().find_map(|(i, s)| find_word(s, &d.short).map(|at| (i, at))) {
            let s = &out[i];
            out[i] = format!("{}{} ({}){}", &s[..at], d.long, d.short, &s[at + d.short.len()..]);
        }
    }
    out
}

/// Byte offset of `word` in `text` as a whole word (not inside a longer word).
fn find_word(text: &str, word: &str) -> Option<usize> {
    let is_word = |c: char| c.is_alphanumeric() || c == '-';
    let mut from = 0;
    while let Some(rel) = text[from..].find(word) {
        let at = from + rel;
        let before = text[..at].chars().next_back();
        let after = text[at + word.len()..].chars().next();
        if !before.is_some_and(is_word) && !after.is_some_and(is_word) {
            return Some(at);
        }
        from = at + word.len();
    }
    None
}

fn re_parens() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\(([^()]{1,120})\)").unwrap())
}

/// A plausible short form: one token of 2 to 10 ASCII letters, digits, "-",
/// "_" or "&" (not mathematical italics), starting with a letter or digit, with at least two letters and two capitals (one is
/// enough in three characters or fewer). A cited name ("Cicchetti, 1994")
/// has one capital.
fn is_short_form(s: &str) -> bool {
    let n = s.chars().count();
    let caps = s.chars().filter(|c| c.is_uppercase()).count();
    (2..=10).contains(&n)
        && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_&".contains(c))
        && s.chars().next().is_some_and(char::is_alphanumeric)
        && s.chars().filter(|c| c.is_alphabetic()).count() >= 2
        && (caps >= 2 || (caps == 1 && n <= 3))
}

fn in_sentence(s: &str) -> Vec<Abbreviation> {
    let mut out = Vec::new();
    for m in re_parens().captures_iter(s) {
        let whole = m.get(0).unwrap();
        // "(LLR; Smith 2020)" and "(LLR, see below)": the first part.
        let inner = m[1].split([';', ',']).next().unwrap_or("").trim();
        let before = s[..whole.start()].trim_end();
        if is_short_form(inner) {
            // long form (SF)
            if let Some(long) = long_form(inner, before) {
                out.push(Abbreviation { short: inner.to_string(), long });
            }
        } else if inner.split_whitespace().count() >= 2 {
            // SF (long form)
            let prev = before.rsplit(char::is_whitespace).next().unwrap_or("");
            let prev = prev.trim_matches(|c: char| !c.is_alphanumeric());
            // This order is rarer and noisier ("Team (the Llama 3.1 model
            // family)"), so the bracket must hold exactly the initials.
            if is_short_form(prev) && prev.chars().filter(|c| c.is_uppercase()).count() >= 2 {
                if let Some(long) = by_initials(prev, inner).filter(|l| plausible_long(l)) {
                    if long == inner.trim_matches(|c: char| !c.is_alphanumeric()) {
                        out.push(Abbreviation { short: prev.to_string(), long });
                    }
                }
            }
        }
    }
    out
}

/// The long form `short` abbreviates at the end of `text`, within the word
/// window, or None. First the words whose initials spell it ("log-likelihood
/// ratio" for LLR: hyphenated parts count), then Schwartz and Hearst's
/// backward letter match, which finds "Low-Rank Adaptation" for LoRA.
fn long_form(short: &str, text: &str) -> Option<String> {
    by_initials(short, text).or_else(|| by_letters(short, text)).filter(|l| plausible_long(l))
}

/// A definition does not run into another bracket ("model (2", "experts [29")
/// and has no control characters.
fn plausible_long(long: &str) -> bool {
    !long.contains(['(', ')', '[', ']', '{', '}']) && !long.contains(char::is_control)
}

/// The last words of `text` whose initials spell `short`.
fn by_initials(short: &str, text: &str) -> Option<String> {
    let sf: Vec<char> = short.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect();
    let words: Vec<&str> = text.split_whitespace().collect();
    let window = (sf.len() + 5).min(2 * sf.len());
    for k in 1..=window.min(words.len()) {
        let span = &words[words.len() - k..];
        let initials: Vec<char> = span
            .iter()
            .flat_map(|w| w.split('-'))
            .filter_map(|part| part.chars().find(|c| c.is_alphanumeric()))
            .flat_map(char::to_lowercase)
            .collect();
        if initials == sf {
            let long = span.join(" ").trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if long.chars().count() > short.chars().count() && !long.contains(short) {
                return Some(long);
            }
        }
    }
    None
}

/// Schwartz and Hearst's backward letter match, over at least two words.
fn by_letters(short: &str, text: &str) -> Option<String> {
    let sf: Vec<char> = short.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect();
    // A capital in the short form stands for the start of a word or of a
    // hyphenated part, or for a capital in the text ("DoRA": Weight-decomposed
    // LoRAs); lower-case letters may come from inside a word ("LoRA": Low-Rank
    // Adaptation). "SD" does not stand for "side initially affected".
    let capital: Vec<bool> = short.chars().filter(|c| c.is_alphanumeric()).map(char::is_uppercase).collect();
    let words: Vec<&str> = text.split_whitespace().collect();
    let window = (sf.len() + 5).min(2 * sf.len());
    let start_word = words.len().saturating_sub(window);
    let candidate = words[start_word..].join(" ");
    let lf: Vec<char> = candidate.chars().flat_map(char::to_lowercase).collect();
    let orig: Vec<char> = candidate.chars().collect();
    if lf.len() != orig.len() {
        return None; // a character changes length when lowercased; skip rather than misalign
    }
    let (mut si, mut li) = (sf.len() as isize - 1, lf.len() as isize - 1);
    while si >= 0 {
        let c = sf[si as usize];
        loop {
            if li < 0 {
                return None;
            }
            let here = lf[li as usize];
            let at_word_start = li == 0 || !lf[li as usize - 1].is_alphanumeric();
            let text_capital = orig[li as usize].is_uppercase();
            if here == c && (at_word_start || (si > 0 && (!capital[si as usize] || text_capital))) {
                break;
            }
            li -= 1;
        }
        li -= 1;
        si -= 1;
    }
    // Back to the start of the whole word: "Weight-decomposed LoRAs" for DoRA.
    let mut start = (li + 1) as usize;
    while start > 0 && !lf[start - 1].is_whitespace() {
        start -= 1;
    }
    let long: String = candidate.chars().skip(start).collect();
    let long = long.trim().trim_end_matches(|c: char| !c.is_alphanumeric()).to_string();
    let lw = long.split_whitespace().count();
    // Longer than the short form, not the short form again, and more than one
    // word: inside one word the letters match by chance ("ID" in "identified").
    if long.chars().count() <= short.chars().count() || lw > window || lw < 2 || long.contains(short) || long.contains(',') {
        return None;
    }
    Some(long)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defs(s: &str) -> Vec<(String, String)> {
        abbreviations(&[s]).into_iter().map(|a| (a.short, a.long)).collect()
    }

    #[test]
    fn long_form_before_brackets() {
        assert_eq!(
            defs("We use the log-likelihood ratio (LLR) to compare models."),
            [("LLR".to_string(), "log-likelihood ratio".to_string())]
        );
        assert_eq!(
            defs("Results from a randomized controlled trial (RCT; Smith 2020) are shown."),
            [("RCT".to_string(), "randomized controlled trial".to_string())]
        );
        assert_eq!(
            defs("Low-Rank Adaptation (LoRA) trains small matrices."),
            [("LoRA".to_string(), "Low-Rank Adaptation".to_string())]
        );
    }

    #[test]
    fn letters_inside_words() {
        assert_eq!(
            defs("Weight-decomposed LoRAs (DoRA) were proposed later."),
            [("DoRA".to_string(), "Weight-decomposed LoRAs".to_string())]
        );
        assert_eq!(
            defs("The K-dwarfs Orbited By habitable Exoplanets (KOBE) survey is new."),
            [("KOBE".to_string(), "K-dwarfs Orbited By habitable Exoplanets".to_string())]
        );
    }

    #[test]
    fn short_form_before_brackets() {
        assert_eq!(
            defs("Each LLR (log-likelihood ratio) is computed per window."),
            [("LLR".to_string(), "log-likelihood ratio".to_string())]
        );
    }

    #[test]
    fn not_abbreviations() {
        for s in [
            "This was shown before (Smith et al., 2020).",
            "The value rose sharply (see Figure 2).",
            "Three cases (i) through (iii) are covered.",
            "The effect was large (p < 0.01).",
            "Prices fell (by 20%) last year.",
            "The company (Acme) made a loss.",
            "This is considered excellent for a psychometric test (Cicchetti, 1994).",
            "Each sample was identified (ID) by hand.",
            "Gains rose as in panel D (panel B) and the share of workers in female sectors.",
            "The model of Meta Llama Team (the Llama 3.1 model family) was used.",
            "Lesions were on the side initially affected (SD) in most patients.",
            "Advertisements are a second-best option, on (AB) the other hand.",
        ] {
            assert!(defs(s).is_empty(), "{s} -> {:?}", defs(s));
        }
    }

    #[test]
    fn first_definition_wins() {
        let d = abbreviations(&[
            "A support vector machine (SVM) was trained.",
            "Later, a different sentence vector model (SVM) appears.",
        ]);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].long, "support vector machine");
    }

    #[test]
    fn summary_needs_what_it_does_not_define() {
        let d = abbreviations(&["We use the log-likelihood ratio (LLR) and a support vector machine (SVM)."]);
        let summary = ["The LLR beats the SVM baseline.", "A support vector machine is cheap."];
        let need: Vec<&str> = for_summary(&summary, &d).iter().map(|a| a.short.as_str()).collect();
        assert_eq!(need, ["LLR"], "SVM is spelled out in the summary");
        // Whole words only: "LLRs" and "XLLR" do not count as uses.
        assert!(for_summary(&["XLLR is unrelated."], &d).is_empty());
        let expanded = expand(&summary, &d);
        assert_eq!(expanded[0], "The log-likelihood ratio (LLR) beats the SVM baseline.");
        assert_eq!(expanded[1], summary[1]);
    }

    mod props {
        use super::*;
        use crate::testgen::messy;
        use proptest::prelude::*;

        fn defining() -> impl Strategy<Value = String> {
            let word = "[a-z]{3,9}";
            (prop::collection::vec(word, 2..5), "[a-z ]{0,30}", any::<bool>()).prop_map(|(ws, tail, sf_first)| {
                let long = ws.join(" ");
                let short: String = ws.iter().map(|w| w.chars().next().unwrap().to_ascii_uppercase()).collect();
                if sf_first {
                    format!("The {short} ({long}) was used{tail}.")
                } else {
                    format!("The {long} ({short}) was used{tail}.")
                }
            })
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(1000))]

            /// Every definition is the document's own text, and every short
            /// form is in the document.
            #[test]
            fn definitions_are_spans(s in prop_oneof![defining(), messy()]) {
                for a in abbreviations(&[&s]) {
                    prop_assert!(s.contains(&a.long), "{:?} not in {:?}", a.long, s);
                    prop_assert!(s.contains(&a.short), "{:?} not in {:?}", a.short, s);
                }
            }

            /// The generated definitions are all found.
            #[test]
            fn generated_definitions_found(s in defining()) {
                prop_assert_eq!(abbreviations(&[&s]).len(), 1, "{:?}", s);
            }

            /// Expanding only inserts the document's words in front of a short form.
            #[test]
            fn expand_only_inserts(s in defining(), use_sf in any::<bool>()) {
                let d = abbreviations(&[&s]);
                let Some(a) = d.first() else { return Ok(()) };
                let summary = vec![format!("Here {} {} matters.", if use_sf { "the" } else { "a" }, a.short)];
                let out = expand(&summary, &d);
                prop_assert_eq!(out[0].replacen(&format!("{} ({})", a.long, a.short), &a.short, 1), summary[0].clone());
            }
        }
    }
}
