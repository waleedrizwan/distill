#![no_main]
//! `select` over a parsed arbitrary document, with a hashed bag of words
//! standing in for the encoder: output in document order, only candidates
//! selected, budgets held.

use distill::segment::{parse, ParseOptions};
use distill::select::{select, Kind, Params};
use libfuzzer_sys::fuzz_target;

fn bow(text: &str) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let mut v = vec![0f32; 64];
    for w in text.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        w.to_lowercase().hash(&mut h);
        v[(h.finish() % 64) as usize] += 1.0;
    }
    v[0] += 1e-3; // never the zero vector
    distill::embed::normalize(&mut v);
    v
}

fuzz_target!(|input: (String, u8, u16, bool, bool)| {
    let (text, budget, words, by_words, repair) = input;
    let doc = parse(&text, &ParseOptions::default());
    let e: Vec<Vec<f32>> = doc.sentences.iter().filter(|s| s.candidate).map(|s| bow(&s.text)).collect();
    let p = Params { budget: budget as usize, word_budget: by_words.then_some(words as usize), repair, ..Default::default() };
    let sel = select(&doc, &e, &p);

    assert!(sel.picks.windows(2).all(|w| w[0].index < w[1].index), "not in document order");
    let selected: Vec<usize> = sel.picks.iter().filter(|p| p.kind == Kind::Selected).map(|p| p.index).collect();
    if !by_words {
        assert_eq!(selected.len(), (budget as usize).min(e.len()));
    }
    for pk in &sel.picks {
        assert!(pk.score.is_finite());
        if pk.kind == Kind::Selected {
            assert!(doc.sentences[pk.index].candidate, "non-candidate selected");
        }
    }
    if by_words {
        let total: usize = sel.picks.iter().map(|p| doc.sentences[p.index].words).sum();
        assert!(total <= words as usize || selected.len() == 1, "{total} words over a budget of {words}");
    }
});
