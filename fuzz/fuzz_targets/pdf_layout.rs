#![no_main]
//! PDF layout (`pdf::assemble`) on arbitrary positioned lines: no panic, and
//! every letter and digit in the output comes from the input lines. Ordering,
//! joining and dropping are allowed; inventing text is not.

use distill::pdf::{assemble, Line, Page};
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

fn letters(s: &str) -> HashMap<char, usize> {
    let mut m = HashMap::new();
    for c in s.chars().filter(|c| c.is_alphanumeric()) {
        *m.entry(c).or_default() += 1;
    }
    m
}

fuzz_target!(|data: &[u8]| {
    // Each line: 6 bytes of geometry, then its text up to a newline.
    let pages = vec![Page { width: 612.0, height: 792.0 }; 4];
    let mut lines = Vec::new();
    let mut rest = data;
    while rest.len() > 6 {
        let (g, tail) = rest.split_at(6);
        let end = tail.iter().position(|&b| b == b'\n').unwrap_or(tail.len());
        let text = String::from_utf8_lossy(&tail[..end]).trim().to_string();
        rest = &tail[(end + 1).min(tail.len())..];
        if text.is_empty() {
            continue;
        }
        let size = 4.0 + f32::from(g[4] % 20);
        lines.push(Line {
            page: usize::from(g[0] % 4),
            x: f32::from(g[1]) * 2.4,
            y: f32::from(g[2]) * 3.1,
            w: 1.0 + f32::from(g[3]) * 2.4,
            h: size,
            size,
            bold: g[5] & 1 == 1,
            text,
        });
    }
    let input: String = lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join(" ");
    let out = assemble(&pages, lines);
    let had = letters(&input);
    for (c, n) in letters(&out) {
        assert!(had.get(&c).copied().unwrap_or(0) >= n, "{c:?} appears more often in the output");
    }
});
