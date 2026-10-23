//! Web pages to text, for `distill https://...`.
//!
//! Not a full HTML parser: enough to get a page's prose out in reading order.
//! Scripts, styles, navigation, headers, footers, asides, forms and figures
//! are removed with their contents; the page's `<article>` (the longest one)
//! or `<main>` is kept if it has one, else the `<body>`. Headings become
//! Markdown headings, list items "- " lines, and other blocks paragraphs.
//! Entities are decoded. What is left of menus and footers is for the
//! boilerplate filter in `segment`.

use regex::Regex;
use std::sync::OnceLock;

macro_rules! re {
    ($name:ident, $pat:expr) => {
        fn $name() -> &'static Regex {
            static R: OnceLock<Regex> = OnceLock::new();
            R.get_or_init(|| Regex::new($pat).unwrap())
        }
    };
}

re!(re_comment, r"(?s)<!--.*?-->");
re!(
    re_drop,
    r"(?is)<(script|style|noscript|svg|nav|header|footer|aside|form|figure|template|iframe|button|select)\b[^>]*>.*?</\s*(script|style|noscript|svg|nav|header|footer|aside|form|figure|template|iframe|button|select)\s*>"
);
re!(re_article, r"(?is)<article\b[^>]*>(.*?)</\s*article\s*>");
re!(re_main, r"(?is)<main\b[^>]*>(.*?)</\s*main\s*>");
re!(re_body, r"(?is)<body\b[^>]*>(.*)</\s*body\s*>");
re!(re_heading, r"(?is)<h([1-6])\b[^>]*>(.*?)</\s*h[1-6]\s*>");
re!(re_li, r"(?is)<li\b[^>]*>");
// A list item that is only a link: a menu, a share bar, a list of related pages.
re!(re_link_item, r"(?is)<li\b[^>]*>\s*<a\b[^>]*>[^<]*(?:<(?:/[^a>]|[^/a>])[^>]*>[^<]*)*</a>\s*</li\s*>");
re!(re_block, r"(?i)</?\s*(p|div|section|br|tr|table|ul|ol|li|blockquote|pre|dd|dt|dl|hr)\b[^>]*>");
re!(re_tag, r"(?s)<[^>]*>");
re!(re_entity, r"&(#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z]{2,8});");
re!(re_title, r"(?is)<title\b[^>]*>(.*?)</\s*title\s*>");

/// True when the bytes look like HTML.
pub fn is_html(bytes: &[u8]) -> bool {
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(2048)]).to_lowercase();
    head.contains("<!doctype html") || head.contains("<html")
}

/// The readable text of an HTML page.
pub fn to_text(html: &str) -> String {
    let html = re_comment().replace_all(html, "");
    let html = re_drop().replace_all(&html, " ");
    let title = re_title().captures(&html).map(|c| decode(&strip(&c[1])));
    let content = re_article()
        .captures_iter(&html)
        .map(|c| c.get(1).unwrap().as_str())
        .max_by_key(|a| a.len())
        .or_else(|| re_main().captures(&html).map(|c| c.get(1).unwrap().as_str()))
        .or_else(|| re_body().captures(&html).map(|c| c.get(1).unwrap().as_str()))
        .unwrap_or(&html);
    let marked = re_heading().replace_all(content, |c: &regex::Captures| {
        let level: usize = c[1].parse().unwrap_or(2);
        format!("\n\n{} {}\n\n", "#".repeat(level), strip(&c[2]).split_whitespace().collect::<Vec<_>>().join(" "))
    });
    let marked = re_link_item().replace_all(&marked, " ");
    let marked = re_li().replace_all(&marked, "\n\n- ");
    let marked = re_block().replace_all(&marked, "\n\n");
    let text = decode(&strip(&marked));

    let mut out = String::new();
    if let Some(t) = title.filter(|t| !t.trim().is_empty() && !text.contains(t.trim())) {
        out.push_str(&format!("# {}\n\n", t.split_whitespace().collect::<Vec<_>>().join(" ")));
    }
    for para in text.split("\n\n") {
        let p = para.split_whitespace().collect::<Vec<_>>().join(" ");
        if p.is_empty() || p == "-" {
            continue;
        }
        out.push_str(&p);
        out.push_str("\n\n");
    }
    out
}

/// Inline tags go without a trace, so "bacteria</a>." stays "bacteria.";
/// block tags were already turned into paragraph breaks.
fn strip(s: &str) -> String {
    re_tag().replace_all(s, "").into_owned()
}

/// Named entities for the common cases, and all numeric ones.
fn decode(s: &str) -> String {
    re_entity()
        .replace_all(s, |c: &regex::Captures| {
            let e = &c[1];
            let named = match e {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some(' '),
                "ndash" => Some('\u{2013}'),
                "mdash" => Some('\u{2014}'),
                "lsquo" => Some('\u{2018}'),
                "rsquo" => Some('\u{2019}'),
                "ldquo" => Some('\u{201C}'),
                "rdquo" => Some('\u{201D}'),
                "hellip" => Some('\u{2026}'),
                "eacute" => Some('é'),
                "copy" => Some('\u{00A9}'),
                _ => None,
            };
            let num = e.strip_prefix('#').and_then(|n| match n.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok(),
                None => n.parse().ok(),
            });
            match named.or_else(|| num.and_then(char::from_u32)) {
                Some(ch) if !ch.is_control() || ch == '\n' => ch.to_string(),
                _ => c[0].to_string(),
            }
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str = r#"<!DOCTYPE html><html><head><title>Bread &amp; Science</title>
        <style>p { color: red }</style><script>var x = "<p>not text</p>";</script></head>
        <body><header><nav><a href="/">Home</a> | <a href="/news">News</a></nav></header>
        <main><article><h1>Slow   bread</h1><p>Slow fermentation gives a more <b>open</b> crumb.&nbsp;Tasters
        prefer it.</p><!-- a comment --><figure><img src="x.png"><figcaption>A loaf</figcaption></figure>
        <h2>How to</h2><ul><li>Mix the dough.</li><li>Proof it overnight&hellip;</li></ul>
        <ul class="share"><li><a href="https://x.example">Share on X (offsite)</a></li><li><a href="/e"><span>Email</span></a></li></ul>
        <p>Wild yeast and <a href="/lb">lactobacillus</a>.</p>
        <p>It costs &#8364;2 &mdash; or &#x20AC;3.</p></article></main>
        <aside>Related: cakes</aside><footer>&copy; 2025 Bakery</footer></body></html>"#;

    #[test]
    fn article_text_in_order() {
        let t = to_text(PAGE);
        assert!(t.starts_with("# Bread & Science\n\n# Slow bread\n\n"), "{t}");
        assert!(t.contains("Slow fermentation gives a more open crumb. Tasters prefer it."));
        assert!(t.contains("## How to\n\n- Mix the dough.\n\n- Proof it overnight\u{2026}"));
        assert!(t.contains("It costs \u{20AC}2 \u{2014} or \u{20AC}3."));
        assert!(t.contains("Wild yeast and lactobacillus."), "no space before the full stop: {t}");
        for gone in ["Home", "not text", "color", "A loaf", "Related", "Bakery", "comment", "Share on X", "Email"] {
            assert!(!t.contains(gone), "{gone} should be gone: {t}");
        }
    }

    #[test]
    fn detects_html() {
        assert!(is_html(b"<!doctype html><html>"));
        assert!(is_html(b"\n<HTML lang=en>"));
        assert!(!is_html(b"Plain text"));
        assert!(!is_html(b"%PDF-1.7"));
    }

    #[test]
    fn unknown_entities_left_alone() {
        assert_eq!(decode("a &bogus; b &#0; c"), "a &bogus; b &#0; c");
    }

    mod props {
        use super::*;
        use crate::testgen::messy;
        use proptest::prelude::*;

        fn words(s: &str) -> Vec<String> {
            s.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect()
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]

            /// Text in paragraphs comes out whole and in order, and nothing
            /// is added but the Markdown marks.
            #[test]
            fn paragraphs_survive(paras in prop::collection::vec("[A-Za-z][a-z ,.]{0,60}", 1..6)) {
                let html = format!("<html><body>{}</body></html>", paras.iter().map(|p| format!("<p>{p}</p>")).collect::<String>());
                let out = to_text(&html);
                prop_assert_eq!(words(&out), words(&paras.join(" ")));
            }

            /// Arbitrary input never panics.
            #[test]
            fn never_panics(s in messy()) {
                let _ = to_text(&s);
                let _ = to_text(&format!("<html><body><p>{s}</p><article>{s}</article></body></html>"));
            }
        }
    }
}
