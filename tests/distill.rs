//! The library entry point, end to end. Needs the model (./fetch-model.sh).

use distill::{Distiller, Length, Options};

const ARTICLE: &str = include_str!("../samples/web-article.txt");

#[test]
fn distill_returns_a_shorter_plain_summary() {
    let out = distill::distill(ARTICLE).expect("summary");
    let words = |s: &str| s.split_whitespace().count();
    assert!(words(&out) > 0);
    assert!(words(&out) < words(ARTICLE) / 2, "{} of {} words", words(&out), words(ARTICLE));
}

#[test]
fn default_options_print_source_sentences_unchanged() {
    let d = Distiller::new(None).expect("model");
    let sum = d.summarize(ARTICLE, &Options { length: Length::Sentences(3), ..Options::default() }).expect("summary");
    let source = ARTICLE.split_whitespace().collect::<Vec<_>>().join(" ");
    for p in &sum.selection.picks {
        assert!(!sum.rewritten(p.index));
        assert!(source.contains(sum.shown(p.index)), "not in the source: {}", sum.shown(p.index));
    }
}

#[test]
fn same_input_same_output() {
    let d = Distiller::new(None).expect("model");
    let a = d.summarize(ARTICLE, &Options::plain()).expect("summary").to_text(true);
    let b = d.summarize(ARTICLE, &Options::plain()).expect("summary").to_text(true);
    assert_eq!(a, b);
}
