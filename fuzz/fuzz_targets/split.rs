#![no_main]
//! `split` on arbitrary text: no panic, and every word but a dropped "and"
//! survives in order.

use libfuzzer_sys::fuzz_target;

fn words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_lowercase())
        .filter(|w| !w.is_empty() && w != "and")
        .collect()
}

fuzz_target!(|s: &str| {
    if let Some(out) = distill::split::split(s) {
        assert_eq!(words(&out), words(s), "words changed: {s:?} -> {out:?}");
    }
});
