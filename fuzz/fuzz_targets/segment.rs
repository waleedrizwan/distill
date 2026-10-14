#![no_main]
//! `split_sentences` and `parse` on arbitrary text: no panic, no text lost,
//! and the document invariants `select` relies on.

use distill::segment::{parse, split_sentences, ParseOptions};
use libfuzzer_sys::fuzz_target;

fn non_space(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fuzz_target!(|text: &str| {
    let para = text.split_whitespace().collect::<Vec<_>>().join(" ");
    assert_eq!(non_space(&split_sentences(&para).concat()), non_space(&para), "text lost: {para:?}");

    for opts in [ParseOptions::default(), ParseOptions { min_words: 1, dehyphenate: true }] {
        let doc = parse(text, &opts);
        let mut next = 0;
        for sec in &doc.sections {
            assert_eq!(sec.sentences.start, next, "sections do not tile the sentences");
            next = sec.sentences.end;
        }
        assert_eq!(next, doc.sentences.len());
        for (i, s) in doc.sentences.iter().enumerate() {
            assert_eq!(s.index, i);
            assert!(doc.sections[s.section].sentences.contains(&i));
            assert_eq!(s.words, s.text.split_whitespace().count());
            assert!(!s.text.is_empty() && s.text == s.text.trim() && !s.text.contains("  ") && !s.text.contains('\n'));
            assert!(s.lead_in.is_none_or(|l| l < i && s.list_item));
            if s.candidate {
                assert!(s.words >= opts.min_words && !doc.sections[s.section].skip);
            }
            if i > 0 {
                let p = &doc.sentences[i - 1];
                assert!(p.section <= s.section && p.paragraph <= s.paragraph);
            }
        }
    }
});
