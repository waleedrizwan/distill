#![no_main]
//! Abbreviations on arbitrary text: no panic, every definition and short form
//! is the text's own (with whitespace collapsed, as distill's sentences are),
//! and expanding only inserts a definition in front of a short form.

use distill::glossary::{abbreviations, expand};
use libfuzzer_sys::fuzz_target;

fn undo(now: &str, want: &str, defs: &[distill::glossary::Abbreviation], from: usize) -> bool {
    if now == want {
        return true;
    }
    defs.iter().enumerate().skip(from).any(|(k, a)| {
        let with = format!("{} ({})", a.long, a.short);
        now.match_indices(&with).any(|(at, _)| {
            let back = format!("{}{}{}", &now[..at], a.short, &now[at + with.len()..]);
            undo(&back, want, defs, k + 1)
        })
    })
}

fuzz_target!(|s: &str| {
    let sentences: Vec<&str> = s.split('\n').collect();
    let defs = abbreviations(&sentences);
    let collapsed: Vec<String> = sentences.iter().map(|t| t.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
    for a in &defs {
        assert!(collapsed.iter().any(|t| t.contains(&a.long)), "{:?} not in the text", a.long);
        assert!(s.contains(&a.short), "{:?} not in the text", a.short);
    }
    let out = expand(&sentences, &defs);
    for (before, after) in sentences.iter().zip(&out) {
        // At most one definition per abbreviation was inserted, each in front
        // of a short form: taking them out again gives the sentence back.
        assert!(undo(after, before, &defs, 0), "expand changed more than it inserted: {before:?} -> {after:?}");
    }
});
