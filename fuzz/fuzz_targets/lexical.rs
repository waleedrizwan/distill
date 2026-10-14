#![no_main]
//! `candidates` and `apply` on arbitrary text: ranges are in bounds, ordered,
//! disjoint and hold their `from`; `apply` touches nothing else.

use distill::lexical::{apply, candidates};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|s: &str| {
    let c = candidates(s);
    let mut last = 0;
    let mut expect = String::new();
    for sw in &c {
        assert!(sw.start >= last && sw.start < sw.end && sw.end <= s.len(), "bad range {sw:?} in {s:?}");
        assert_eq!(&s[sw.start..sw.end], sw.from, "{s:?}");
        expect.push_str(&s[last..sw.start]);
        expect.push_str(&sw.to);
        last = sw.end;
    }
    expect.push_str(&s[last..]);
    assert_eq!(apply(s, &c.iter().collect::<Vec<_>>()), expect);
    assert_eq!(apply(s, &[]), s);
});
