//! Deletion-only sentence compression (`--compress`).
//!
//! Removes parts of a sentence that a summary can do without, using rules on
//! punctuation and a short word list:
//!
//! - parentheticals, except list labels such as "(a)" or "(iii)", acronym
//!   definitions such as "(EPA)", defined terms ("referred to as ..."),
//!   qualifiers that change meaning ("other than ...", "except ...") and
//!   anything inside a quotation; a leading "(CNN)"-style dateline is dropped
//! - bracketed citations such as "[12]" or "[3, 7-9]"
//! - comma-bounded "which" / "who" / "whose" clauses, outside quotations and
//!   outside legal operative text (where such a clause can restrict a
//!   definition)
//! - interruptions between a pair of spaced dashes ("A -- an aside -- B")
//! - sentence-opening connectives ("However,", "In addition,", ...)
//!
//! Nothing is added or reworded. The output's words are the input's words, in
//! order, with pieces left out; the only change to a kept word is capitalising
//! the new first word when an opener is removed. A sentence that would end up
//! under five words, or under 40% of its length, is left as it was.

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

// A source credit at the very start: "(CNN)", "(The Hollywood Reporter)".
re!(re_dateline, r"^\((?:CNN|AP|Reuters|AFP|UPI|BBC|The [A-Z][\w ]{1,40})\)\s*(?:--\s*)?");
// Parenthetical content that carries meaning and must stay.
re!(
    re_substantive,
    r"(?i)^(?:other than|except|excluding|including|not including|unless|but\b|not\b|without|only\b|if\b|provided|subject to|as (?:defined|amended|added|described))|referred to|hereinafter|hereafter|``|''"
);
// Legal operative text, where a comma-bounded "which" can restrict a definition.
re!(
    re_legal,
    r"(?i)\b(?:shall|means|the term|is amended|are amended|is hereby|this (?:Act|section|subsection|paragraph|title|chapter)|U\.S\.C\.)\b|^\(?[a-z0-9]{1,4}\)|^``"
);
re!(re_paren, r"\s*\(([^()\[\]]{1,200})\)");
// "(a)", "(12)", "(iii)", "(B)": list labels and references to them.
re!(re_label, r"^(?:[A-Za-z]{1,2}|\d{1,3}|(?i:[ivxlcdm]{1,7}))$");
re!(re_acronym, r"^[A-Z][A-Z0-9&.\-]{1,9}s?$");
re!(re_citation, r"\s*\[\d{1,3}(?:\s*[,\u{2013}\-]\s*\d{1,3})*\]");
re!(re_dash_pair, r"\s+(?:--|\u{2014}|\u{2013})\s+[^\u{2014}\u{2013}\-]{1,120}?\s+(?:--|\u{2014}|\u{2013})\s+");
// The clause must be followed by a lowercase word (normally the main verb:
// "The plan, which ..., will cost"), so a comma inside a name or place
// ("..., who lives in Vallejo, California, has") does not end it early.
re!(re_rel_mid, r#",\s+(?:which|who|whom|whose)\s[^,;:()"\u{201C}\u{201D}]{1,140},\s*(\p{Ll})"#);
re!(re_rel_end, r#",\s+(?:which|who|whom|whose)\s[^,;:()"\u{201C}\u{201D}]{1,140}?([.!?]["'\u{201D}]?)$"#);
re!(
    re_opener,
    r"^(?:However|Moreover|Furthermore|In addition|Additionally|Also|Indeed|In fact|Meanwhile|Nevertheless|Nonetheless|Still|Of course|Notably|Importantly|Interestingly|Overall|In other words|As a result|Consequently|Therefore|Thus|Finally|Similarly|Likewise|Instead|Separately|In particular|For its part|At the same time),\s+"
);

/// The compressed sentence, or `None` when no rule applies (or the result
/// would be too short to trust).
pub fn compress(text: &str) -> Option<String> {
    let mut s = text.trim().to_string();
    let before = word_count(&s);
    let mut opened = false;

    if let Some(m) = re_dateline().find(&s) {
        s.replace_range(..m.end(), "");
        opened = true;
    }
    s = re_citation().replace_all(&s, "").into_owned();
    s = replace_outside_quotes(&s, re_paren(), |c| {
        let inner = c[1].trim();
        let keep = re_label().is_match(inner)
            || re_acronym().is_match(inner)
            || re_substantive().is_match(inner)
            || inner.contains(['"', '\u{201C}', '\u{201D}']);
        if keep { c[0].to_string() } else { String::new() }
    });
    s = replace_outside_quotes(&s, re_dash_pair(), |_| " ".to_string());
    if !re_legal().is_match(&s) {
        let snapshot = s.clone();
        s = replace_outside_quotes(&snapshot, re_rel_mid(), |c| {
            let whole = c.get(0).expect("whole match");
            if !clause_ok(whole.as_str()) || re_coordinator().is_match(&snapshot[whole.end() - c[1].len()..]) {
                return whole.as_str().to_string();
            }
            // "In the midst of X, which ..., warriors marked": keep the comma
            // that closes the opening phrase.
            let before = &snapshot[..whole.start()];
            let sep = if re_intro().is_match(before) && !before.contains(',') { ", " } else { " " };
            format!("{sep}{}", &c[1])
        });
        s = replace_outside_quotes(&s, re_rel_end(), |c| {
            if clause_ok(&c[0]) { c[1].to_string() } else { c[0].to_string() }
        });
    }
    if let Some(m) = re_opener().find(&s) {
        s.replace_range(..m.end(), "");
        opened = true;
    }

    let mut s = tidy(&s);
    if opened {
        s = capitalise_first(&s);
    }
    let after = word_count(&s);
    if s == text.trim() || after < 5 || (after as f32) < 0.4 * before as f32 {
        return None;
    }
    Some(s)
}

re!(re_coordinator, r"^(?:and|or|but|nor)\b");
re!(re_intro, r"^(?:In|On|At|After|Before|When|While|As|Although|Though|Because|If|Since|Despite|During|Following|Amid|For|Under|With|Without|Unlike|Like)\b");
re!(re_speech, r"\b(?:said|says|told|tells|added|adds|asked|wrote|explained|according)\b");

/// A "which"/"who" clause is only removed when it is short and does not seem
/// to swallow the sentence's own verb (sources sometimes omit the closing
/// comma: "Mgbachi, who has a stand in the square said that ...").
fn clause_ok(clause: &str) -> bool {
    clause.split_whitespace().count() <= 16 && !re_speech().is_match(clause)
}

/// Does this read as legal operative text ("shall", "means", "is amended")?
pub fn is_legal(text: &str) -> bool {
    re_legal().is_match(text)
}

/// Is byte offset `pos` inside a quotation? Counts straight double quotes,
/// curly quotes and the ``...'' style used in US bills.
pub fn inside_quote(s: &str, pos: usize) -> bool {
    let before = &s[..pos];
    let straight = before.matches('"').count() % 2 == 1;
    let curly = before.matches('\u{201C}').count() > before.matches('\u{201D}').count();
    let legal = before.matches("``").count() > before.matches("''").count();
    straight || curly || legal
}

/// Like `Regex::replace_all`, but leaves matches that start or end inside a
/// quotation untouched, so quoted words are never altered.
fn replace_outside_quotes(s: &str, re: &Regex, mut with: impl FnMut(&regex::Captures) -> String) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last = 0;
    for c in re.captures_iter(s) {
        let m = c.get(0).expect("whole match");
        out.push_str(&s[last..m.start()]);
        if inside_quote(s, m.start()) || inside_quote(s, m.end()) {
            out.push_str(m.as_str());
        } else {
            out.push_str(&with(&c));
        }
        last = m.end();
    }
    out.push_str(&s[last..]);
    out
}

/// Collapse whitespace and repair punctuation left behind by deletions.
fn tidy(s: &str) -> String {
    let mut out = s.split_whitespace().collect::<Vec<_>>().join(" ");
    for (from, to) in [(" ,", ","), (" .", "."), (" ;", ";"), (" :", ":"), (" !", "!"), (" ?", "?"), (",,", ","), (",.", "."), (", .", ".")] {
        while out.contains(from) {
            out = out.replace(from, to);
        }
    }
    out.trim_start_matches([',', ';', ' ']).to_string()
}

fn capitalise_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_lowercase() => c.to_uppercase().chain(chars).collect(),
        _ => s.to_string(),
    }
}

fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lowercased alphanumeric tokens.
    fn toks(s: &str) -> Vec<String> {
        s.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).map(|t| t.to_lowercase()).collect()
    }

    fn is_subsequence(short: &[String], long: &[String]) -> bool {
        let mut it = long.iter();
        short.iter().all(|t| it.any(|u| u == t))
    }

    fn c(s: &str) -> String {
        compress(s).unwrap_or_else(|| s.to_string())
    }

    #[test]
    fn rules() {
        assert_eq!(
            c("(CNN)The city council voted on Tuesday to restore the old wetlands."),
            "The city council voted on Tuesday to restore the old wetlands."
        );
        assert_eq!(
            c("The plan, which was approved on Tuesday after a long debate, will cost 18 million euros."),
            "The plan will cost 18 million euros."
        );
        assert_eq!(
            c("Engineers built the wall in 1956, which residents say made flooding worse."),
            "Engineers built the wall in 1956."
        );
        assert_eq!(
            c("However, the council agreed to fund a groundwater survey first."),
            "The council agreed to fund a groundwater survey first."
        );
        assert_eq!(
            c("Recent models (such as the one used here) improve accuracy on long inputs [12, 14]."),
            "Recent models improve accuracy on long inputs."
        );
        assert_eq!(
            c("The mayor -- who had opposed the plan for years -- voted in favour of the scheme."),
            "The mayor voted in favour of the scheme."
        );
    }

    #[test]
    fn keeps_what_matters() {
        // List labels and references to them.
        let legal = "Except as provided in subparagraph (B), the term includes any purchase described in paragraph (2).";
        assert_eq!(c(legal), legal);
        // Acronym definitions.
        let acr = "The Environmental Protection Agency (EPA) issued the rule on Monday morning.";
        assert_eq!(c(acr), acr);
        // Legal "--" lead-ins are not spaced dash pairs.
        let lead = "(a) Imposition of Tax.--There is hereby imposed a tax on each covered transaction.";
        assert_eq!(c(lead), lead);
        // Quoted insertion text in a bill.
        let ins = "The last paragraph of section 7 of the Clayton Act is amended by inserting ``(excluding transactions described in section 11321 of title 49)'' after ``Board''.";
        assert_eq!(c(ins), ins);
        // Qualifiers and defined terms.
        let carve = "Amounts in the Fund shall be used to redeem any obligation included in the public debt (other than an obligation held by the Trust Fund).";
        assert_eq!(c(carve), carve);
        let defined = "There is established a commission to be known as the Emergency Insurance Commission (in this Act referred to as the ``Commission'').";
        assert_eq!(c(defined), defined);
        // Inside a quotation.
        let quote = "\"The Law of the Jungle, which never orders anything without a reason, forbids every beast to eat Man.\"";
        assert_eq!(c(quote), quote);
        // A restrictive "which" in a legal definition.
        let defn = "(5) Red snapper.--The term ``red snapper'' means members of the species Lutjanus campechanus, which ordinarily are found seaward of the coastal waters.";
        assert_eq!(c(defn), defn);
        // The source omits the comma before the main verb.
        let nocomma = "Newspaper vendor John Mgbachi, who has a stand in Tinubu Square said that far from being bad for business, he welcomes the free readers.";
        assert_eq!(c(nocomma), nocomma);
        // Coordinated clauses.
        let coord = "The coalition struck Houthi rebels, who are backed by Iran, and who could spread its influence in the region.";
        assert_eq!(c(coord), coord);
        // The comma closing an opening phrase stays.
        assert_eq!(
            c("In the midst of the shouting over the new law, which many fear will lead to discrimination, activists marked the anniversary."),
            "In the midst of the shouting over the new law, activists marked the anniversary."
        );
        // A comma inside a place name must not end the clause early.
        let place = "Denise Huskins, who was reported kidnapped two days ago in Vallejo, California, has been found safe.";
        assert_eq!(c(place), place);
        // Citations to the code still go.
        assert_eq!(c("Section 4 of the Deepwater Port Act of 1974 (33 U.S.C. 1503) is amended by striking the last sentence."),
            "Section 4 of the Deepwater Port Act of 1974 is amended by striking the last sentence.");
        // Too much would go: unchanged.
        let short = "It rained (heavily, for most of the long and miserable afternoon, across the whole region).";
        assert_eq!(c(short), short);
    }

    #[test]
    fn output_is_a_subsequence_of_input() {
        for s in [
            "(CNN)Officials said on Monday, who declined to be named, that the talks (held in Geneva) had stalled.",
            "Moreover, the results -- which surprised the authors -- held across all five datasets [3-5].",
            "In addition, the Secretary, who shall consult with the Administrator, may issue regulations under section 4(b).",
            "The court, which heard arguments in March, ruled that the law (as amended in 2019) was valid.",
        ] {
            let out = c(s);
            assert!(is_subsequence(&toks(&out), &toks(s)), "{out:?} is not a subsequence of {s:?}");
        }
    }
}
