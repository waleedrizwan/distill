#![no_main]
//! `compress` on arbitrary text: no panic, words only ever removed, and the
//! length floors hold.

use libfuzzer_sys::fuzz_target;

fn toks(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).map(str::to_lowercase).collect()
}

fuzz_target!(|s: &str| {
    let _ = distill::compress::is_legal(s);
    let Some(out) = distill::compress::compress(s) else { return };
    let (short, long) = (toks(&out), toks(s));
    let mut it = long.iter();
    assert!(short.iter().all(|t| it.any(|u| u == t)), "{out:?} is not a subsequence of {s:?}");
    let (a, b) = (out.split_whitespace().count(), s.split_whitespace().count());
    assert!(a >= 5 && a as f32 >= 0.4 * b as f32, "too short: {out:?} from {s:?}");
});
