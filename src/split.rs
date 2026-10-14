//! Splitting long sentences at a clause boundary (`--split`).
//!
//! "X, and Y." becomes "X. Y." when both X and Y are full clauses; ", but" and
//! ", so" keep their conjunction ("X. But Y."), and a semicolon becomes a full
//! stop. Nothing is added: the output's words are the input's in order, minus
//! a dropped "and"; the only change to a kept word is capitalising the first
//! word of the new sentence.
//!
//! There is no part-of-speech tagger. A half counts as a clause only when it
//! has a finite verb from closed lists (auxiliaries and modals, common
//! irregular past forms, regular "-ed" forms) and, for the second half, when
//! it opens with a subject (a personal pronoun, a determiner phrase, a name or
//! a bare noun phrase) that reaches that verb within a few words. That misses many real
//! clause boundaries but rarely splits anything that is not one.

use regex::Regex;
use std::sync::OnceLock;

macro_rules! re {
    ($name:ident, $pat:expr) => {
        fn $name() -> &'static Regex {
            static R: OnceLock<Regex> = OnceLock::new();
            R.get_or_init(|| Regex::new($pat).expect("valid regex"))
        }
    };
}

// Only sentences at least this long are split, and each part must keep at
// least `MIN_PART` words.
const MIN_SENTENCE: usize = 24;
const MIN_PART: usize = 8;

re!(re_junction, r"(?:,\s+(and|but|so)|;)\s+");
// A list label anywhere ("(1)", "( b )", "(iv)", "(B)"): the semicolons and
// "and"s belong to the list, so the sentence is left whole.
re!(re_label, r"\(\s*(?:\d{1,3}|[a-z]|[ivx]{2,5}|[A-Z])\s*\)");
// "he's", "they're", "we've", "she'll", "I'd", "I'm": a subject and a verb.
re!(re_pronoun_contraction, r"^(?:i|you|he|she|it|we|they|there|that)(?:'|\u{2019})(?:s|re|ve|ll|d|m)$");

const PRONOUNS: &[&str] = &["i", "you", "he", "she", "it", "we", "they", "there"];
const DETERMINERS: &[&str] = &[
    "the", "a", "an", "his", "her", "its", "their", "our", "my", "your", "this", "that", "these", "those", "some", "many",
    "most", "several", "each", "every", "no", "both", "all", "another", "one", "two", "three",
];
const AUX: &[&str] = &[
    "is", "are", "was", "were", "has", "have", "had", "will", "would", "can", "could", "should", "may", "might", "must",
    "does", "did", "do", "shall", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "won't",
    "wouldn't", "can't", "couldn't", "shouldn't", "doesn't", "didn't", "don't",
];
const IRREGULAR_PAST: &[&str] = &[
    "said", "told", "went", "came", "made", "took", "found", "gave", "got", "saw", "knew", "thought", "left", "began",
    "became", "held", "kept", "met", "ran", "sat", "stood", "won", "lost", "paid", "sent", "spent", "built", "brought",
    "bought", "caught", "fell", "felt", "fought", "grew", "led", "rose", "sold", "spoke", "struck", "threw", "wrote",
    "drove", "broke", "chose", "flew", "forgot", "hid", "meant", "taught", "understood", "wore", "woke", "ate", "drank",
];
// Openers that start a verb phrase or an adverbial, not a subject.
const NOT_SUBJECT: &[&str] = &[
    "to", "not", "also", "then", "now", "still", "even", "just", "only", "later", "soon", "again", "instead",
    "therefore", "thus", "yet", "never", "often", "always", "sometimes", "usually", "at", "in", "on", "for", "with",
    "by", "from", "after", "before", "as", "if", "when", "while", "because", "although", "which", "who", "whose",
    "what", "where", "how", "why", "more", "less", "about", "around", "over", "under",
];
// Words after a subject that are not verbs even though they look like one.
const NOT_VERB_ED: &[&str] = &["need", "seed", "speed", "bed", "red", "shed", "hundred", "bred", "led"];

fn words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

fn is_finite_verb(w: &str) -> bool {
    AUX.contains(&w)
        || IRREGULAR_PAST.contains(&w)
        || (w.len() > 4 && w.ends_with("ed") && !NOT_VERB_ED.contains(&w))
}

/// Does `part` (the text after the junction) open with a subject followed by
/// a finite verb within a few words?
fn opens_clause(part: &str) -> bool {
    let raw: Vec<&str> = part.split_whitespace().collect();
    let w = words(part);
    if w.len() < 2 {
        return false;
    }
    let first = w[0].as_str();
    if re_pronoun_contraction().is_match(first) {
        return true;
    }
    let subject_len = if PRONOUNS.contains(&first) {
        1
    } else if DETERMINERS.contains(&first) {
        // "the new plan", "their two youngest children": up to three more words.
        match (1..w.len().min(5)).find(|&k| is_finite_verb(&w[k])) {
            Some(k) => k,
            None => return false,
        }
    } else if raw[0].chars().next().is_some_and(char::is_uppercase) {
        // A name: "Smith", "Prime Minister Theresa May", "University of
        // Oklahoma President David Boren".
        let mut k = 1;
        while k < raw.len().min(8)
            && (raw[k].chars().next().is_some_and(char::is_uppercase) || matches!(raw[k], "of" | "de" | "von" | "van"))
        {
            k += 1;
        }
        k
    } else if !is_finite_verb(first) && !first.ends_with("ing") && !NOT_SUBJECT.contains(&first) {
        // A bare noun phrase: "residents near the old mill worried".
        match (1..w.len().min(6)).find(|&k| is_finite_verb(&w[k])) {
            Some(k) => return k > 0,
            None => return false,
        }
    } else {
        return false;
    };
    // "one that has always resisted ...": a relative clause, not a subject and its verb.
    if w[..subject_len].iter().any(|x| matches!(x.as_str(), "that" | "which" | "who" | "whom" | "whose")) {
        return false;
    }
    // The verb must come right after the subject, or after one adverb ("he also said").
    (subject_len..(subject_len + 2).min(w.len())).any(|k| is_finite_verb(&w[k]))
}

/// The sentence split into two or more, or `None` when no junction qualifies.
pub fn split(text: &str) -> Option<String> {
    let text = text.trim();
    if text.split_whitespace().count() < MIN_SENTENCE || crate::compress::is_legal(text) || re_label().is_match(text) {
        return None;
    }
    let mut out = String::new();
    let mut last = 0;
    let mut did = false;
    for c in re_junction().captures_iter(text) {
        let m = c.get(0).expect("whole match");
        let left = &text[last..m.start()];
        let right = &text[m.end()..];
        if crate::compress::inside_quote(text, m.start())
            || left.matches('(').count() != left.matches(')').count()
            || left.split_whitespace().count() < MIN_PART
            || right.split_whitespace().count() < MIN_PART
            // "A, B, and C": a list, not two clauses.
            || c.get(1).is_some_and(|k| k.as_str() == "and") && left.matches(',').count() >= 2
            || !words(left).iter().any(|w| is_finite_verb(w))
            || !opens_clause(right)
        {
            continue;
        }
        let conj = c.get(1).map(|k| k.as_str());
        out.push_str(left.trim_end());
        out.push_str(". ");
        match conj {
            Some("but") => out.push_str("But "),
            Some("so") => out.push_str("So "),
            _ => {}
        }
        // Capitalise the first word of the new sentence (unless "But"/"So" now opens it).
        let rest = &text[m.end()..];
        if matches!(conj, Some("but") | Some("so")) {
            last = m.end();
        } else {
            let mut ch = rest.chars();
            if let Some(f) = ch.next() {
                out.extend(f.to_uppercase());
                last = m.end() + f.len_utf8();
            }
        }
        did = true;
    }
    if !did {
        return None;
    }
    out.push_str(&text[last..]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_clauses() {
        assert_eq!(
            split("The council voted on Tuesday to restore the wetlands along the lower river, and the work will begin next spring after the bird breeding season ends.").as_deref(),
            Some("The council voted on Tuesday to restore the wetlands along the lower river. The work will begin next spring after the bird breeding season ends.")
        );
        assert_eq!(
            split("Supporters say the restored marsh will store floodwater from the river, but residents near the old mill worried that their houses could be damaged.").as_deref(),
            Some("Supporters say the restored marsh will store floodwater from the river. But residents near the old mill worried that their houses could be damaged.")
        );
        assert_eq!(
            split("Lindsey Vonn may have missed out on gold at last month's world championships, but the American skier has set her sights on glory this season.").as_deref(),
            Some("Lindsey Vonn may have missed out on gold at last month's world championships. But the American skier has set her sights on glory this season.")
        );
        assert_eq!(
            split("The national chapter suspended the students on Monday afternoon, but University of Oklahoma President David Boren took it a step further by cutting all ties on Tuesday.").as_deref(),
            Some("The national chapter suspended the students on Monday afternoon. But University of Oklahoma President David Boren took it a step further by cutting all ties on Tuesday.")
        );
        assert_eq!(
            split("The comic has made only a few appearances on that show since popping up in December, but he's better known within the world of standup comedy.").as_deref(),
            Some("The comic has made only a few appearances on that show since popping up in December. But he's better known within the world of standup comedy.")
        );
        assert_eq!(
            split("The engineers finished the survey of the crumbling north bank last month; they found that the old wall was weaker than anyone had expected.").as_deref(),
            Some("The engineers finished the survey of the crumbling north bank last month. They found that the old wall was weaker than anyone had expected.")
        );
    }

    #[test]
    fn leaves_alone() {
        for s in [
            // Too short.
            "The council voted on Tuesday, and the work will begin in spring.",
            // Second half is not a clause (verb phrase sharing the subject).
            "The council voted on Tuesday to restore the wetlands along the lower river, and agreed to fund a groundwater survey before any work begins.",
            // A list.
            "The plan covers the north bank, the old mill and the meadows, and the council expects the work to take about six years in total.",
            // Inside a quotation.
            "\"We have waited a very long time for this decision, and we are delighted that the council finally listened to us,\" she said.",
            // A relative clause after "and", not a new subject.
            "Easter is unique on the Christian calendar, a major point in the cycle of the religious year, and one that has always been able to resist commercialization.",
            // A numbered list.
            "It is the sense of Congress that-- (1) any person convicted of the offence should be detained; and (2) any person convicted of a second offence should be detained for life.",
            // Legal text.
            "The Secretary shall issue regulations within 180 days after the date of enactment, and the regulations shall take effect on the date of publication.",
        ] {
            assert_eq!(split(s), None, "split: {s}");
        }
    }

    #[test]
    fn keeps_words() {
        let s = "Officials said the bridge would reopen by the end of the month after repairs to the deck, and they promised that tolls would not rise this year.";
        let out = split(s).expect("splits");
        let toks = |t: &str| words(t).into_iter().filter(|w| w != "and").collect::<Vec<_>>();
        assert_eq!(toks(&out), toks(s));
    }
    mod props {
        //! Nothing added: on generated sentences a split keeps every word but
        //! the dropped "and", in order, and short or legal sentences are left
        //! alone. No panics on arbitrary text.
        use super::*;
        use crate::testgen::{messy, sentence};
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(2000))]

            #[test]
            fn nothing_added(s in sentence()) {
                let n = s.split_whitespace().count();
                let legal = crate::compress::is_legal(s.trim()) || re_label().is_match(s.trim());
                let Some(out) = split(&s) else { return Ok(()) };
                prop_assert!(n >= MIN_SENTENCE && !legal, "split a sentence it should leave alone: {:?}", s);
                let keep = |t: &str| words(t).into_iter().filter(|w| w != "and").collect::<Vec<_>>();
                prop_assert_eq!(keep(&out), keep(&s));
                let parts: Vec<&str> = out.split(". ").collect();
                prop_assert!(parts.len() >= 2, "{:?}", out);
                let dropped = words(&s).len() - words(&out).len();
                prop_assert!(dropped < parts.len(), "{:?} lost {} words", out, dropped);
                for p in &parts {
                    prop_assert!(p.split_whitespace().count() >= MIN_PART, "short part {:?} in {:?}", p, out);
                }
                for p in &parts[1..] {
                    let f = p.chars().next().unwrap_or(' ');
                    prop_assert!(!f.is_lowercase(), "new sentence not capitalised: {:?}", p);
                }
            }

            #[test]
            fn never_panics(s in messy()) {
                let _ = split(&s);
            }
        }
    }
}
