#![no_main]
//! `html::to_text` on arbitrary input: no panic, and the output has no tags.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|s: &str| {
    let out = distill::html::to_text(s);
    // Decoded entities may produce "<", but never a whole tag that was in the input.
    for tag in ["<script", "<style", "<nav", "<p>", "<div"] {
        if !s.contains("&lt;") && !s.contains("&#") {
            assert!(!out.to_lowercase().contains(tag), "{tag} left in the output");
        }
    }
});
