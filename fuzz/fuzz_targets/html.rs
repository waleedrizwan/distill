#![no_main]
//! `html::to_text` on arbitrary input: no panic, and no complete tag of the
//! kinds it removes is left in the output. Text such as "a<div" with no
//! closing ">" is not a tag and may stay.

use libfuzzer_sys::fuzz_target;

/// Whether `s` has `open` followed by a ">" before any other "<".
fn has_tag(s: &str, open: &str) -> bool {
    s.match_indices(open).any(|(at, _)| {
        let rest = &s[at + open.len()..];
        let next = rest.chars().next();
        // "<p" must end there ("<p>", "<p class=..."), not be "<pre" or "<pixel".
        let boundary = next.is_none_or(|c| c == '>' || c.is_whitespace() || c == '/');
        boundary && rest.find('>').is_some_and(|gt| !rest[..gt].contains('<'))
    })
}

fuzz_target!(|s: &str| {
    let out = distill::html::to_text(s);
    // Decoded entities can spell a tag ("&lt;p&gt;"); only check plain input.
    if s.contains('&') {
        return;
    }
    let lower = out.to_lowercase();
    for open in ["<script", "<style", "<nav", "<p", "<div"] {
        assert!(!has_tag(&lower, open), "{open} tag left in the output: {out:?}");
    }
});
