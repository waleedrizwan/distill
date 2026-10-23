//! PDF input: text in reading order, one paragraph per line.
//!
//! `pdf_oxide` finds the text lines and their positions; the reading order is
//! decided here, because the library's own order sometimes swaps lines and
//! drops margin stamps into the middle of a sentence. Per page:
//!
//! - rotated text (the arXiv stamp in the margin) is dropped;
//! - a page is two-column when both halves hold enough narrow lines. Lines
//!   that span the middle (titles, wide figures) cut the page into bands, and
//!   each band is read left column, then right;
//! - running headers and footers (the same text near the top or bottom of
//!   many pages) and bare page numbers are dropped.
//!
//! Lines then join into paragraphs where the vertical gap is larger than the
//! usual line spacing or the font size changes. A word broken across lines
//! with a hyphen is rejoined unless the document spells it with the hyphen
//! elsewhere ("self-supervised" stays, "experi- ment" becomes "experiment").
//!
//! The output is plain text for `segment::parse`. Same bytes in, same text
//! out: nothing here depends on threads or time.

use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};

/// True when the bytes look like a PDF (the spec allows junk before the header).
pub fn is_pdf(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(1024)];
    head.windows(5).any(|w| w == b"%PDF-")
}

/// One positioned line of text. `y` is the baseline box's bottom edge in PDF
/// points, measured upward from the bottom of the page.
#[derive(Clone, Debug)]
pub struct Line {
    pub page: usize,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub size: f32,
    pub bold: bool,
    pub text: String,
}

#[derive(Clone, Copy, Debug)]
pub struct Page {
    pub width: f32,
    pub height: f32,
}

/// Extract a PDF's text in reading order.
pub fn extract(bytes: &[u8]) -> Result<String> {
    let (pages, lines) = read_lines(bytes)?;
    Ok(assemble(&pages, lines))
}

/// Positioned lines from every page, in the library's order. A panic inside
/// the parser (malformed files) becomes an error.
pub fn read_lines(bytes: &[u8]) -> Result<(Vec<Page>, Vec<Line>)> {
    let data = bytes.to_vec();
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(move || -> Result<(Vec<Page>, Vec<Line>)> {
        let doc = pdf_oxide::PdfDocument::from_bytes(data).map_err(|e| anyhow!("{e}"))?;
        let n = doc.page_count().map_err(|e| anyhow!("{e}"))?;
        let mut pages = Vec::with_capacity(n);
        let mut lines = Vec::new();
        for p in 0..n {
            let (x0, y0, x1, y1) = doc.get_page_media_box(p).unwrap_or((0.0, 0.0, 612.0, 792.0));
            pages.push(Page { width: (x1 - x0).abs().max(1.0), height: (y1 - y0).abs().max(1.0) });
            // A page that cannot be read is skipped rather than failing the document.
            let Ok(page_lines) = doc.extract_text_lines(p) else { continue };
            for l in page_lines {
                if l.words.iter().any(|w| w.rotation_degrees.abs() > 1.0) {
                    continue;
                }
                // The library joins words on the same baseline even across the
                // gap between two columns. Split wherever the space between
                // neighbouring words is far wider than a word space.
                let mut words: Vec<_> = l.words.iter().filter(|w| !w.text.trim().is_empty()).collect();
                words.sort_by(|a, b| a.bbox.x.total_cmp(&b.bbox.x));
                let mut start = 0;
                for i in 1..=words.len() {
                    let split = i == words.len() || {
                        let (a, b) = (words[i - 1], words[i]);
                        column_gap(a.bbox.x + a.bbox.width, b.bbox.x, a.avg_font_size, (x0 + x1) / 2.0)
                    };
                    if !split {
                        continue;
                    }
                    let seg = &words[start..i];
                    start = i;
                    let x = seg[0].bbox.x;
                    let right = seg.iter().map(|w| w.bbox.x + w.bbox.width).fold(f32::MIN, f32::max);
                    let y = seg.iter().map(|w| w.bbox.y).fold(f32::MAX, f32::min);
                    let top = seg.iter().map(|w| w.bbox.y + w.bbox.height).fold(f32::MIN, f32::max);
                    let n = seg.len() as f32;
                    let text = join_runs(seg.iter().map(|w| Run {
                        text: &w.text,
                        x0: w.bbox.x,
                        x1: w.bbox.x + w.bbox.width,
                        size: w.avg_font_size,
                    }));
                    lines.push(Line {
                        page: p,
                        x: x - x0,
                        y: y - y0,
                        w: right - x,
                        h: top - y,
                        size: seg.iter().map(|w| w.avg_font_size).sum::<f32>() / n,
                        bold: seg.iter().filter(|w| w.is_bold).count() * 2 > seg.len(),
                        text: clean_line(&text),
                    });
                }
            }
        }
        Ok((pages, lines))
    });
    std::panic::set_hook(prev);
    match r {
        Ok(res) => res,
        Err(_) => Err(anyhow!("the PDF could not be parsed (the parser crashed on it)")),
    }
}

/// Whether the space between two words on one baseline is the gap between
/// columns: far wider than a word space, or a narrower gap (A4 papers often
/// leave about 1.4 em) that covers the page's middle.
fn column_gap(left: f32, right: f32, size: f32, mid: f32) -> bool {
    let gap = right - left;
    gap > (size * 1.5).max(9.0) || (gap > size * 0.7 && left < mid && right > mid)
}

/// A run of text in one font, as the library reports it ("word").
struct Run<'a> {
    text: &'a str,
    x0: f32,
    x1: f32,
    size: f32,
}

/// Join a line's runs into text. The library splits text wherever the font
/// changes, so "BARE" and "-generated" arrive as two runs that touch, and
/// "Fe3O4" as four. Runs that touch are joined without a space; subscripts
/// and superscripts (smaller type) attach to the word before them; citation
/// numbers set small after a word ("pathway³⁴.", "shown¹²,¹³") are dropped.
/// The library reports scripts on the line's baseline, so they are told
/// apart by size and by what precedes them, not by height.
fn join_runs<'a>(runs: impl Iterator<Item = Run<'a>>) -> String {
    let runs: Vec<Run> = runs.collect();
    // The line's own type size: that of its longest run.
    let base = runs
        .iter()
        .max_by(|a, b| a.text.len().cmp(&b.text.len()).then(a.size.total_cmp(&b.size)))
        .map_or(10.0, |r| r.size)
        .max(1.0);
    let is_script = |r: &Run| r.size < base * 0.85;

    let mut out = String::new();
    let mut prev: Option<&Run> = None;
    // Where the last run ended, dropped ones included, for measuring gaps.
    let mut last_x1 = f32::MIN;
    for r in &runs {
        if is_script(r) && is_citation(r.text) && out.ends_with(|c: char| c.is_lowercase() || ".,;:)]".contains(c)) {
            last_x1 = r.x1;
            continue;
        }
        if let Some(p) = prev {
            let gap = r.x0 - last_x1;
            let space = if is_script(r) {
                gap > base * 0.5
            } else if is_script(p) {
                // Script boxes are imprecise, so decide by what follows:
                // "Fe3O4" continues a formula, "Fe3O4 and" does not.
                let t = r.text;
                let element = t.len() <= 2
                    && t.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                    && t.chars().skip(1).all(|c| c.is_ascii_lowercase());
                let punct = t.starts_with(['.', ',', ';', ':', ')', ']', '/']);
                !(element || punct)
            } else {
                gap > p.size.min(r.size) * 0.1
            };
            if space {
                out.push(' ');
            }
        }
        out.push_str(r.text);
        prev = Some(r);
        last_x1 = r.x1;
    }
    out
}

/// "34", "12,13", "5-7", "5\u{2013}7": what a citation number looks like.
/// A single digit is more often an exponent or a footnote mark, so it stays.
fn is_citation(t: &str) -> bool {
    let t = t.trim();
    t.chars().next().is_some_and(|c| c.is_ascii_digit())
        && t.chars().all(|c| c.is_ascii_digit() || matches!(c, ',' | '-' | '\u{2013}'))
        && (t.len() > 1 || t.contains([',', '-', '\u{2013}']))
}

/// Ligatures and odd spaces the parser leaves in.
fn clean_line(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\u{FB00}' => out.push_str("ff"),
            '\u{FB01}' => out.push_str("fi"),
            '\u{FB02}' => out.push_str("fl"),
            '\u{FB03}' => out.push_str("ffi"),
            '\u{FB04}' => out.push_str("ffl"),
            '\u{00A0}' | '\u{2009}' | '\u{202F}' | '\t' => out.push(' '),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Order the lines, drop page furniture, and join paragraphs.
pub fn assemble(pages: &[Page], lines: Vec<Line>) -> String {
    let lines = drop_footnotes(pages, drop_furniture(pages, lines));
    let mut ordered = Vec::with_capacity(lines.len());
    let mut by_page: Vec<Vec<Line>> = vec![Vec::new(); pages.len()];
    for l in lines {
        if l.page < by_page.len() {
            by_page[l.page].push(l);
        }
    }
    for (p, page_lines) in by_page.into_iter().enumerate() {
        ordered.extend(reading_order(pages[p], page_lines));
    }
    let paragraphs = drop_title_block(mark_headings(&ordered, drop_small_print(&ordered, paragraph_groups(&ordered))));
    let joined = dehyphenate(&join_broken_paragraphs(paragraphs));
    let mut out = joined.join("\n\n");
    out.push('\n');
    out
}

/// Paragraphs set smaller than the body text: figure and table captions, text
/// inside figures, tables. The abstract, which papers often set small too,
/// is kept: the paragraphs after an "Abstract" heading, and one that opens
/// with the word.
fn drop_small_print(lines: &[Line], groups: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let body = body_size(lines);
    let mut out = Vec::with_capacity(groups.len());
    let mut in_abstract = false;
    for g in groups {
        let chars: usize = g.iter().map(|&i| lines[i].text.len()).sum::<usize>().max(1);
        let size = g.iter().map(|&i| lines[i].size * lines[i].text.len() as f32).sum::<f32>() / chars as f32;
        let text: Vec<String> = g.iter().map(|&i| lines[i].text.clone()).collect();
        let first = text[0].trim();
        let abstract_heading = text.len() == 1 && first.trim_end_matches([':', '.']).eq_ignore_ascii_case("abstract");
        // The first paragraph in the body's size ends the abstract.
        if !abstract_heading && size >= body * 0.92 {
            in_abstract = false;
        }
        let abstract_text = in_abstract || first.to_lowercase().starts_with("abstract");
        if abstract_heading {
            in_abstract = true;
        }
        // Headings set in small capitals measure small too; keep one-line,
        // heading-shaped paragraphs.
        let heading = text.len() == 1 && first.split(' ').count() <= 12 && !first.ends_with(['.', ',', ';']);
        if size < body * 0.92 && !abstract_text && !abstract_heading && !heading {
            continue;
        }
        out.push(g);
    }
    out
}

/// Authors, affiliations, emails and dates sit between a paper's title and
/// its abstract. When the document has an "Abstract" heading early on, the
/// plain paragraphs before it are dropped; headings (the title) stay.
fn drop_title_block(paragraphs: Vec<Vec<String>>) -> Vec<Vec<String>> {
    let is_abstract = |p: &Vec<String>| {
        p.len() == 1 && p[0].trim_start_matches('#').trim().trim_end_matches([':', '.']).eq_ignore_ascii_case("abstract")
    };
    let Some(at) = paragraphs.iter().take(40).position(is_abstract) else { return paragraphs };
    paragraphs.into_iter().enumerate().filter(|(i, p)| *i >= at || p[0].starts_with('#')).map(|(_, p)| p).collect()
}

/// Mark section headings in Markdown ("# 1 Introduction", "## 2.1 Data"), so
/// segment does not have to guess them from short lines, which in a paper are
/// as often labels inside figures and tables. A heading is a one-line
/// paragraph, at most 12 words, without closing punctuation, that is
/// numbered ("3", "3.2", "III.", "A."), a heading any paper uses ("Results",
/// "References"), or set bold or larger than the body and capitalised.
fn mark_headings(lines: &[Line], groups: Vec<Vec<usize>>) -> Vec<Vec<String>> {
    static NUMBERED: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let numbered = NUMBERED.get_or_init(|| {
        regex::Regex::new(r"^((?:\d{1,2}(?:\.\d{1,2}){0,3}\.?)|(?:[IVX]{1,5}\.)|(?:[A-J](?:\.\d{1,2}){0,3}\.?))\s+\p{Lu}").unwrap()
    });
    let body = body_size(lines);
    groups
        .into_iter()
        .map(|g| {
            let text: Vec<String> = g.iter().map(|&i| lines[i].text.clone()).collect();
            if g.len() != 1 {
                return text;
            }
            let l = &lines[g[0]];
            let t = l.text.trim();
            let words = t.split(' ').count();
            let letters = t.chars().filter(|c| c.is_alphabetic()).count();
            if words > 12 || letters < 3 || t.ends_with(['.', ',', ';', ':']) || t.len() > 100 {
                return text;
            }
            // Appendix letters ("A Sampling Details", "B.1 Data") but not
            // initials ("A. Baker and B. Miller, Department of ...").
            let lettered = |c: &regex::Captures| c[1].starts_with(|ch: char| ch.is_ascii_uppercase());
            let level = if let Some(c) = numbered.captures(t).filter(|c| !(lettered(c) && t.contains(','))) {
                c[1].trim_end_matches('.').split('.').count()
            } else if crate::segment::is_known_heading(t) {
                1
            } else if (l.bold || l.size > body * 1.1)
                && l.size >= body * 0.92
                && words <= 10
                && t.chars().next().is_some_and(char::is_uppercase)
            {
                2
            } else {
                return text;
            };
            vec![format!("{} {t}", "#".repeat(level.min(4)))]
        })
        .collect()
}

/// Text that repeats near the top or bottom edge of many pages, and page numbers.
fn drop_furniture(pages: &[Page], lines: Vec<Line>) -> Vec<Line> {
    let edge = |l: &Line| {
        let h = pages.get(l.page).map_or(792.0, |p| p.height);
        l.y > h * 0.88 || l.y + l.h < h * 0.10
    };
    let key = |s: &str| -> String {
        s.chars().filter(|c| c.is_alphabetic()).flat_map(char::to_lowercase).collect()
    };
    let mut seen: HashMap<String, HashSet<usize>> = HashMap::new();
    for l in lines.iter().filter(|l| edge(l)) {
        let k = key(&l.text);
        if !k.is_empty() {
            seen.entry(k).or_default().insert(l.page);
        }
    }
    let threshold = (pages.len() / 4).max(2);
    lines
        .into_iter()
        .filter(|l| {
            if !edge(l) {
                return true;
            }
            let k = key(&l.text);
            if k.is_empty() {
                // Page numbers and other bare digits or symbols at the edge.
                return false;
            }
            seen.get(&k).is_none_or(|p| p.len() < threshold)
        })
        .collect()
}

/// The body text's font size: the size that covers the most characters.
fn body_size(lines: &[Line]) -> f32 {
    let mut by_size: HashMap<i32, usize> = HashMap::new();
    for l in lines {
        *by_size.entry((l.size * 2.0).round() as i32).or_default() += l.text.len();
    }
    // Ties go to the smaller size, so the result does not depend on map order.
    by_size.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map_or(10.0, |(k, _)| k as f32 / 2.0)
}

/// Small print in the bottom quarter of a page: footnotes, affiliations,
/// copyright lines. They interrupt the sentence that runs on into the next
/// column or page, and are never what a summary needs.
fn drop_footnotes(pages: &[Page], lines: Vec<Line>) -> Vec<Line> {
    let body = body_size(&lines);
    lines
        .into_iter()
        .filter(|l| {
            let h = pages.get(l.page).map_or(792.0, |p| p.height);
            !(l.size < body * 0.88 && l.y + l.h < h * 0.25)
        })
        .collect()
}

/// Lines of one page in reading order.
fn reading_order(page: Page, mut lines: Vec<Line>) -> Vec<Line> {
    // Top to bottom, then left to right.
    lines.sort_by(|a, b| b.y.total_cmp(&a.y).then(a.x.total_cmp(&b.x)).then_with(|| a.text.cmp(&b.text)));
    let mid = page.width / 2.0;
    let side = |l: &Line| -> Option<bool> {
        // Some(false) = left column, Some(true) = right, None = spans the middle.
        if l.x + l.w <= mid + 6.0 {
            Some(false)
        } else if l.x >= mid - 6.0 {
            Some(true)
        } else {
            None
        }
    };
    // Lines that span the middle (titles, a one-column abstract, wide
    // figures) cut the page into bands. A band with lines on both sides is
    // two columns: left first, then right. This is decided per band, since
    // the first page of a paper often has a wide abstract above two columns.
    let mut out = Vec::with_capacity(lines.len());
    let mut band: Vec<Line> = Vec::new();
    let flush = |band: &mut Vec<Line>, out: &mut Vec<Line>| {
        let left = band.iter().filter(|l| side(l) == Some(false)).count();
        let right = band.len() - left;
        if left >= 2 && right >= 2 {
            let (l, r): (Vec<Line>, Vec<Line>) = band.drain(..).partition(|l| side(l) == Some(false));
            out.extend(l);
            out.extend(r);
        } else {
            out.append(band);
        }
    };
    for l in lines {
        if side(&l).is_none() {
            flush(&mut band, &mut out);
            out.push(l);
        } else {
            band.push(l);
        }
    }
    flush(&mut band, &mut out);
    out
}

/// Join lines into paragraphs.
#[cfg(test)]
fn paragraphs(lines: &[Line]) -> Vec<Vec<String>> {
    paragraph_groups(lines).into_iter().map(|g| g.into_iter().map(|i| lines[i].text.clone()).collect()).collect()
}

/// Paragraphs as runs of line indices.
fn paragraph_groups(lines: &[Line]) -> Vec<Vec<usize>> {
    // The usual distance from one line to the next, from lines in the same
    // column and font size.
    let mut steps: Vec<f32> = lines
        .windows(2)
        .filter(|w| w[0].page == w[1].page && (w[0].x - w[1].x).abs() < 30.0)
        .filter(|w| (w[0].size - w[1].size).abs() < 0.5)
        .map(|w| w[0].y - w[1].y)
        .filter(|d| *d > 0.0)
        .collect();
    steps.sort_by(f32::total_cmp);
    let step = steps.get(steps.len() / 2).copied().unwrap_or(12.0).max(1.0);

    // The usual width of a body line, to tell a heading from a full line.
    let mut widths: Vec<f32> = lines.iter().filter(|l| l.text.split(' ').count() >= 6).map(|l| l.w).collect();
    widths.sort_by(f32::total_cmp);
    let full = widths.get(widths.len() * 3 / 4).copied().unwrap_or(400.0);
    let starts_cap = |t: &str| t.chars().next().is_some_and(|c| c.is_uppercase() || c.is_ascii_digit());
    // A heading-like line: short, a few words, capitalised, no closing
    // punctuation, and followed by a line that starts a sentence.
    let heading: Vec<bool> = (0..lines.len())
        .map(|i| {
            let l = &lines[i];
            let t = l.text.trim();
            l.w < full * 0.6
                && t.split(' ').count() <= 12
                && starts_cap(t)
                && !t.ends_with(['.', ',', ';', ':', '-'])
                && lines.get(i + 1).is_some_and(|n| starts_cap(n.text.trim()))
        })
        .collect();

    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let brk = match i.checked_sub(1).map(|j| &lines[j]) {
            None => false,
            Some(_) if heading[i] || heading[i - 1] => true,
            Some(prev) => {
                let same_flow = prev.page == l.page && prev.y > l.y;
                let gap = prev.y - l.y;
                let size_change = (prev.size - l.size).abs() > 0.6;
                let bold_change = prev.bold != l.bold && (prev.w < 300.0 || l.w < 300.0);
                // A short line that ends a sentence usually ends a paragraph.
                let short_end = prev.w < l.w * 0.8 && ends_sentence(&prev.text);
                // A first-line indent, in the same column.
                let indent = same_flow
                    && l.x > prev.x + 6.0
                    && l.x < prev.x + 40.0
                    && ends_sentence(&prev.text)
                    && l.text.chars().next().is_some_and(char::is_uppercase);
                (same_flow && gap > step * 1.35) || size_change || bold_change || short_end || indent
            }
        };
        if brk && !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        current.push(i);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn ends_sentence(s: &str) -> bool {
    let s = s.trim_end_matches(['"', '\'', ')', '\u{201D}', '\u{2019}']);
    s.ends_with(['.', '?', '!', ':'])
}

/// A paragraph that ends in a hyphenated word ("... diagnostics of tur-")
/// continues in the next one ("bulence ..."): a column or page break fell
/// inside it, and a figure or footnote between made two paragraphs of it.
fn join_broken_paragraphs(paragraphs: Vec<Vec<String>>) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::with_capacity(paragraphs.len());
    for p in paragraphs {
        let continues = out.last().is_some_and(|prev: &Vec<String>| {
            let end = prev.last().map(|l| l.trim_end()).unwrap_or("");
            let start = p[0].trim_start();
            !prev[0].starts_with('#')
                && !start.starts_with('#')
                && end.ends_with('-')
                && end[..end.len() - 1].chars().last().is_some_and(char::is_alphabetic)
                && start.chars().next().is_some_and(char::is_lowercase)
        });
        if continues {
            out.last_mut().unwrap().extend(p);
        } else {
            out.push(p);
        }
    }
    out
}

/// Join each paragraph's lines, rejoining words hyphenated at a line end.
fn dehyphenate(paragraphs: &[Vec<String>]) -> Vec<String> {
    // Words the document writes with a hyphen inside a line, and every word
    // (and hyphenated part) it uses on its own.
    let mut hyphenated: HashSet<String> = HashSet::new();
    let mut words: HashSet<String> = HashSet::new();
    for p in paragraphs {
        for (li, line) in p.iter().enumerate() {
            let ws: Vec<&str> = line.split_whitespace().collect();
            for (wi, w) in ws.iter().enumerate() {
                let w = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
                // The halves of a word broken at this line's end are not evidence.
                let at_break = (wi + 1 == ws.len() && w.ends_with('-')) || (wi == 0 && li > 0);
                if at_break {
                    continue;
                }
                if w.contains('-') && !w.starts_with('-') && !w.ends_with('-') {
                    hyphenated.insert(w.to_lowercase());
                } else {
                    words.insert(w.to_lowercase());
                }
            }
        }
    }
    paragraphs
        .iter()
        .map(|p| {
            let mut s = String::new();
            for line in p {
                if s.is_empty() {
                    s.push_str(line);
                    continue;
                }
                let head: String = s.rsplit(' ').next().unwrap_or("").to_string();
                let next_word = line.split(' ').next().unwrap_or("");
                let broken = head.len() > 2
                    && head.ends_with('-')
                    && head[..head.len() - 1].chars().last().is_some_and(char::is_alphabetic)
                    && next_word.chars().next().is_some_and(char::is_lowercase);
                let caps_break = head.len() > 2
                    && head.ends_with('-')
                    && next_word.chars().next().is_some_and(char::is_uppercase);
                if caps_break {
                    // "DeepSeek-" + "AI": a hyphenated name, keep the hyphen.
                    s.push_str(line);
                } else if broken {
                    let stem = head.trim_end_matches('-').trim_start_matches(|c: char| !c.is_alphanumeric());
                    let tail = next_word.trim_end_matches(|c: char| !c.is_alphanumeric());
                    let with = format!("{stem}-{tail}").to_lowercase();
                    let joined = format!("{stem}{tail}").to_lowercase();
                    // Keep the hyphen when the document writes the compound
                    // that way, or when both halves are words it uses alone
                    // ("resource-" + "constrained") and never the joined form.
                    let keep = hyphenated.contains(&with)
                        || (!words.contains(&joined)
                            && words.contains(&stem.to_lowercase())
                            && words.contains(&tail.to_lowercase())
                            && stem.chars().count() >= 3
                            && tail.chars().count() >= 3);
                    if keep {
                        s.push_str(line);
                    } else {
                        s.pop();
                        s.push_str(line);
                    }
                } else {
                    s.push(' ');
                    s.push_str(line);
                }
            }
            s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(page: usize, x: f32, y: f32, w: f32, text: &str) -> Line {
        Line { page, x, y, w, h: 10.0, size: 10.0, bold: false, text: text.into() }
    }

    const PAGE: Page = Page { width: 612.0, height: 792.0 };

    #[test]
    fn detects_pdf_header() {
        assert!(is_pdf(b"%PDF-1.7\n..."));
        assert!(is_pdf(b"\n\n%PDF-1.4"));
        assert!(!is_pdf(b"plain text about %PDF"));
    }

    #[test]
    fn narrow_gutters_split() {
        // "with-" | "significant" on an A4 page (middle 297.6): a 15 pt gap at 10.9 pt.
        assert!(column_gap(290.9, 306.1, 10.9, 297.6));
        // An ordinary word space does not split, nor a wide space away from the middle.
        assert!(!column_gap(160.9, 163.6, 10.9, 297.6));
        assert!(!column_gap(150.0, 162.0, 10.9, 297.6));
        // A very wide gap splits anywhere (tables, a right-aligned date).
        assert!(column_gap(150.0, 180.0, 10.9, 297.6));
    }

    #[test]
    fn two_columns_read_left_then_right() {
        let mut lines = Vec::new();
        for i in 0..10 {
            let y = 700.0 - 12.0 * i as f32;
            lines.push(line(0, 300.0 + 20.0, y, 250.0, &format!("right {i}")));
            lines.push(line(0, 50.0, y, 250.0, &format!("left {i}")));
        }
        lines.push(line(0, 50.0, 740.0, 500.0, "A title across both columns"));
        let order: Vec<String> = reading_order(PAGE, lines).into_iter().map(|l| l.text).collect();
        assert_eq!(order[0], "A title across both columns");
        assert_eq!(order[1], "left 0");
        assert_eq!(order[10], "left 9");
        assert_eq!(order[11], "right 0");
    }

    #[test]
    fn wide_abstract_above_two_columns() {
        let mut lines = Vec::new();
        for i in 0..20 {
            lines.push(line(0, 50.0, 700.0 - 12.0 * i as f32, 510.0, &format!("abstract {i}")));
        }
        for i in 0..4 {
            let y = 400.0 - 12.0 * i as f32;
            lines.push(line(0, 50.0, y, 250.0, &format!("left {i}")));
            lines.push(line(0, 320.0, y + 4.0, 250.0, &format!("right {i}")));
        }
        let order: Vec<String> = reading_order(PAGE, lines).into_iter().map(|l| l.text).collect();
        assert_eq!(&order[20..], ["left 0", "left 1", "left 2", "left 3", "right 0", "right 1", "right 2", "right 3"]);
    }

    #[test]
    fn single_column_sorted_top_down() {
        let lines = vec![line(0, 72.0, 600.0, 400.0, "second"), line(0, 72.0, 612.0, 400.0, "first")];
        let order: Vec<String> = reading_order(PAGE, lines).into_iter().map(|l| l.text).collect();
        assert_eq!(order, ["first", "second"]);
    }

    #[test]
    fn running_headers_and_page_numbers_dropped() {
        let pages = vec![PAGE; 6];
        let mut lines = Vec::new();
        for p in 0..6 {
            lines.push(line(p, 72.0, 760.0, 300.0, &format!("Journal of Things, vol. 3 page {p}")));
            lines.push(line(p, 300.0, 40.0, 10.0, &format!("{}", p + 1)));
            lines.push(line(p, 72.0, 500.0, 400.0, &format!("Body text on page {p}.")));
        }
        let kept = drop_furniture(&pages, lines);
        assert_eq!(kept.len(), 6);
        assert!(kept.iter().all(|l| l.text.starts_with("Body")));
    }

    #[test]
    fn a_title_on_one_page_is_kept() {
        let pages = vec![PAGE; 6];
        let lines = vec![line(0, 72.0, 760.0, 300.0, "A Paper Title")];
        assert_eq!(drop_furniture(&pages, lines).len(), 1);
    }

    #[test]
    fn paragraphs_split_on_wide_gaps() {
        let lines = vec![
            line(0, 72.0, 700.0, 450.0, "The first paragraph runs over"),
            line(0, 72.0, 688.0, 450.0, "two lines of text and then"),
            line(0, 72.0, 676.0, 450.0, "a third line here"),
            line(0, 72.0, 650.0, 450.0, "A second paragraph."),
        ];
        let p = paragraphs(&lines);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].len(), 3);
    }

    #[test]
    fn headings_stand_alone_without_gaps() {
        // Double-spaced, so the gaps say nothing; the short capitalised line does.
        let lines = vec![
            line(0, 72.0, 700.0, 450.0, "Keywords: machine learning, deep learning, social media"),
            line(0, 72.0, 676.0, 80.0, "Introduction"),
            line(0, 72.0, 652.0, 450.0, "Mental illness is common and often goes undetected for"),
            line(0, 72.0, 628.0, 450.0, "years, which is why automatic detection has drawn interest."),
        ];
        let p = paragraphs(&lines);
        assert_eq!(p.len(), 3, "{p:?}");
        assert_eq!(p[1], ["Introduction"]);
    }

    #[test]
    fn small_print_dropped_but_not_the_abstract() {
        let mut lines = Vec::new();
        let mut y = 760.0;
        let mut push = |text: &str, size: f32, lines: &mut Vec<Line>| {
            let mut l = line(0, 72.0, y, 450.0, text);
            l.size = size;
            l.h = size;
            lines.push(l);
            y -= 30.0;
        };
        push("Abstract", 9.0, &mut lines);
        push("We study things in a small font, as two-column papers do.", 9.0, &mut lines);
        push("1 Introduction", 10.0, &mut lines);
        for _ in 0..6 {
            push("Body text set in the normal size of the paper runs on here.", 10.0, &mut lines);
        }
        push("Figure 1: A caption set smaller than the body text.", 8.0, &mut lines);
        let groups = paragraph_groups(&lines);
        let kept: Vec<String> = drop_small_print(&lines, groups)
            .into_iter()
            .map(|g| g.iter().map(|&i| lines[i].text.as_str()).collect::<Vec<_>>().join(" "))
            .collect();
        assert!(kept.iter().any(|p| p.contains("We study things")), "{kept:?}");
        assert!(!kept.iter().any(|p| p.starts_with("Figure 1")), "{kept:?}");
        assert_eq!(kept.join(" ").matches("Body text").count(), 6);
    }

    #[test]
    fn title_block_dropped_before_the_abstract() {
        let p = |t: &str| vec![t.to_string()];
        let paras = vec![p("# A Paper About Bread"), p("A. Baker and B. Miller, Department of Food Science"),
            p("baker@example.org"), p("# Abstract"), p("We study bread."), p("# 1 Introduction")];
        let out: Vec<String> = drop_title_block(paras).into_iter().map(|p| p.join(" ")).collect();
        assert_eq!(out, ["# A Paper About Bread", "# Abstract", "We study bread.", "# 1 Introduction"]);
        // No abstract heading: nothing to go by, nothing dropped.
        let paras = vec![p("# Title"), p("Some author line here."), p("# 1 Introduction")];
        assert_eq!(drop_title_block(paras).len(), 3);
    }

    #[test]
    fn headings_marked() {
        let mut lines = Vec::new();
        let mut y = 760.0;
        for (t, size, bold) in [
            ("A Paper About Bread", 16.0, true),
            ("A. Baker and B. Miller, Department of Food Science", 11.0, false),
            ("1 Introduction", 10.0, true),
            ("Bread is old and people like it a great deal, so we study it here.", 10.0, false),
            ("2.1 Flours", 10.0, false),
            ("A Sampling Details", 10.0, false),
            ("B.1 More Results", 10.0, false),
            ("Results", 10.0, false),
            ("Domain", 7.0, false),
        ] {
            let mut l = line(0, 72.0, y, 200.0, t);
            l.size = size;
            l.bold = bold;
            lines.push(l);
            y -= 40.0;
        }
        let groups: Vec<Vec<usize>> = (0..lines.len()).map(|i| vec![i]).collect();
        let out: Vec<String> = mark_headings(&lines, groups).into_iter().map(|p| p.join(" ")).collect();
        // "A" could be an appendix letter; either way the title is a heading.
        assert_eq!(out[0], "# A Paper About Bread");
        assert_eq!(out[1], "A. Baker and B. Miller, Department of Food Science");
        assert_eq!(out[2], "# 1 Introduction");
        assert!(!out[3].starts_with('#'));
        assert_eq!(out[4], "## 2.1 Flours");
        assert_eq!(out[5], "# A Sampling Details");
        assert_eq!(out[6], "## B.1 More Results");
        assert_eq!(out[7], "# Results");
        assert_eq!(out[8], "Domain", "a small label from a figure is not a heading");
        // Appendix letters run A to J; "R Rmax" is an axis label.
        let l = line(0, 72.0, 300.0, 60.0, "R Rmax");
        assert_eq!(mark_headings(&[l], vec![vec![0]])[0], ["R Rmax"]);
    }

    #[test]
    fn paragraphs_broken_inside_a_word_rejoin() {
        let p = |t: &str| vec![t.to_string()];
        let out = join_broken_paragraphs(vec![p("A more quantitative diagnostics of tur-"), p("bulence follows here."), p("# 3 Results"), p("next-")]);
        assert_eq!(dehyphenate(&out)[0], "A more quantitative diagnostics of turbulence follows here.");
        assert_eq!(out.len(), 3, "a heading never joins");
    }

    #[test]
    fn hyphens_at_line_ends() {
        let paras = vec![
            vec!["we ran the experi-".to_string(), "ment twice with self-".to_string(), "supervised data".to_string()],
            vec!["a self-supervised model".to_string()],
        ];
        let out = dehyphenate(&paras);
        assert_eq!(out[0], "we ran the experiment twice with self-supervised data");
    }

    #[test]
    fn hyphen_kept_between_two_known_words() {
        let paras = vec![
            vec!["for resource-".to_string(), "constrained devices".to_string()],
            vec!["a resource that is constrained".to_string()],
        ];
        assert_eq!(dehyphenate(&paras)[0], "for resource-constrained devices");
    }

    #[test]
    fn footnotes_dropped() {
        let pages = vec![PAGE];
        let mut lines: Vec<Line> = (0..20).map(|i| line(0, 72.0, 700.0 - 12.0 * i as f32, 450.0, "body")).collect();
        let mut foot = line(0, 72.0, 60.0, 450.0, "* Equal contribution");
        foot.size = 8.0;
        lines.push(foot);
        let kept = drop_footnotes(&pages, lines);
        assert_eq!(kept.len(), 20);
    }

    /// A run as the parser reports it; `_y` (always the line's baseline) is
    /// kept in the calls so they read like the parser's output.
    fn run(text: &str, x0: f32, x1: f32, _y: f32, size: f32) -> Run<'_> {
        Run { text, x0, x1, size }
    }

    #[test]
    fn runs_that_touch_join() {
        let runs = [run("BARE", 245.0, 266.0, 311.9, 10.0), run("-generated", 266.0, 307.5, 311.9, 10.0),
            run("samples", 310.0, 341.8, 311.9, 10.0)];
        assert_eq!(join_runs(runs.into_iter()), "BARE-generated samples");
    }

    #[test]
    fn subscripts_attach() {
        // "i.e. Fe3O4 and α-Fe2O3." as the parser reports it.
        let runs = [
            run("i.e.", 281.8, 296.4, 214.8, 12.0), run("Fe", 299.5, 311.5, 214.8, 12.0),
            run("3", 311.4, 315.4, 213.8, 8.0), run("O", 315.5, 324.1, 214.8, 12.0),
            run("4", 324.1, 328.1, 213.8, 8.0), run("and", 327.1, 344.5, 214.8, 12.0),
            run("α-Fe", 347.5, 369.6, 214.8, 12.0), run("2", 373.7, 377.7, 213.8, 8.0),
            run("O", 373.7, 382.4, 214.8, 12.0), run("3", 386.4, 390.4, 213.8, 8.0),
            run(".", 386.4, 389.4, 214.8, 12.0),
        ];
        assert_eq!(join_runs(runs.into_iter()), "i.e. Fe3O4 and α-Fe2O3.");
    }

    #[test]
    fn citation_superscripts_dropped_exponents_kept() {
        // As the parser reports them: scripts sit on the line's baseline.
        let runs = [run("the", 397.4, 414.2, 77.5, 12.0), run("pathway", 468.3, 512.1, 77.5, 12.0),
            run("34", 512.1, 519.6, 77.5, 7.0), run(".", 519.6, 523.0, 77.5, 12.0)];
        assert_eq!(join_runs(runs.into_iter()), "the pathway.");
        let runs = [run("units", 100.0, 125.0, 500.0, 10.0), run("cm", 127.5, 140.0, 500.0, 10.0),
            run("2", 140.2, 144.0, 500.0, 7.0), run("/g.", 144.0, 156.0, 500.0, 10.0)];
        assert_eq!(join_runs(runs.into_iter()), "units cm2/g.");
        // A subscript after a capital is part of a formula, not a citation.
        let runs = [run("pure", 100.0, 120.0, 500.0, 10.0), run("C", 122.5, 129.0, 500.0, 10.0),
            run("60", 129.0, 136.0, 500.0, 7.0), run("films", 138.5, 160.0, 500.0, 10.0)];
        assert_eq!(join_runs(runs.into_iter()), "pure C60 films");
    }

    #[test]
    fn ligatures_expanded() {
        assert_eq!(clean_line("e\u{FB03}cient  \u{FB01}rst"), "efficient first");
    }

    #[test]
    fn two_column_fixture() {
        // samples/two-column-paper.pdf, written by samples/make-paper-pdf.py.
        let text = extract(include_bytes!("../samples/two-column-paper.pdf")).unwrap();
        let at = |s: &str| text.find(s).unwrap_or_else(|| panic!("{s:?} missing from:\n{text}"));
        // Reading order: title, abstract, then the sections in order across
        // both columns and both pages.
        assert!(at("Slow Fermentation Improves") < at("Abstract"));
        assert!(at("1 Introduction") < at("2 Methods"));
        assert!(at("2 Methods") < at("3 Results"));
        assert!(at("3 Results") < at("4 Conclusion"));
        // A paragraph that runs from one column into the next stays one paragraph.
        assert!(text.contains("We used three flours: a white bread flour, a whole wheat flour and a rye blend."));
        // Line-end hyphens: a broken word is rejoined, a real compound is kept.
        assert!(text.contains("six degrees in a refrigerator."));
        assert!(text.contains("at twenty-four degrees"));
        // Furniture is gone: running header, page numbers, margin stamp, footnote.
        for gone in ["Journal of Everyday Science", "preprint", "Corresponding author"] {
            assert!(!text.contains(gone), "{gone:?} should be dropped:\n{text}");
        }
        assert!(!text.lines().any(|l| l.trim().chars().all(|c| c.is_ascii_digit()) && !l.trim().is_empty()));
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        // Must return, whatever it returns; a panic would fail the test.
        let _ = extract(b"%PDF-1.4\n%%EOF garbage");
        let _ = extract(b"%PDF-1.7\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\ntrailer << /Root 1 0 R >>");
        assert!(extract(b"not a pdf").is_err());
    }

    mod props {
        use super::*;
        use proptest::prelude::*;
        use std::collections::HashMap;

        fn letters(s: &str) -> HashMap<char, usize> {
            let mut m = HashMap::new();
            for c in s.chars().filter(|c| c.is_alphanumeric()) {
                *m.entry(c).or_default() += 1;
            }
            m
        }

        fn word() -> impl Strategy<Value = String> {
            prop_oneof![
                "[a-z]{1,10}",
                "[A-Z][a-z]{1,8}",
                "[a-z]{2,6}-",
                "[0-9]{1,3}",
                "[a-z]{2,6}[.,;:]",
                Just("self-supervised".to_string()),
            ]
        }

        fn lines() -> impl Strategy<Value = (Vec<Page>, Vec<Line>)> {
            // Coarse positions, so lines often share a baseline or a start.
            let one = (0usize..3, (0u8..40).prop_map(|v| v as f32 * 15.0), (0u8..60).prop_map(|v| v as f32 * 13.0), 5.0f32..500.0, 6.0f32..18.0, any::<bool>(),
                prop::collection::vec(word(), 1..12));
            prop::collection::vec(one, 0..80).prop_map(|v| {
                let pages = vec![Page { width: 612.0, height: 792.0 }; 3];
                let lines = v
                    .into_iter()
                    .map(|(page, x, y, w, size, bold, words)| Line {
                        page, x, y, w, h: size, size, bold, text: words.join(" "),
                    })
                    .collect();
                (pages, lines)
            })
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(1000))]

            /// Layout only reorders, joins and drops: every letter and digit
            /// in the output comes from the input lines.
            #[test]
            fn assemble_invents_nothing((pages, lines) in lines()) {
                let input: String = lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join(" ");
                let out = assemble(&pages, lines);
                let (have, had) = (letters(&out), letters(&input));
                for (c, n) in &have {
                    prop_assert!(had.get(c).copied().unwrap_or(0) >= *n, "{:?} appears more often in the output", c);
                }
            }

            /// Same lines in, same text out, whatever order they arrive in.
            #[test]
            fn assemble_ignores_input_order((pages, lines) in lines()) {
                let mut rev = lines.clone();
                rev.reverse();
                prop_assert_eq!(assemble(&pages, lines), assemble(&pages, rev));
            }
        }
    }
}
