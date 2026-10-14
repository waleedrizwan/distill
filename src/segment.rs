//! Turns raw text into sections, paragraphs and sentences.
//!
//! Nothing here rewrites the text: sentences are returned verbatim apart from
//! collapsing whitespace (hard line wraps become single spaces).

use regex::Regex;
use std::collections::HashSet;
use std::ops::Range;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct Sentence {
    pub index: usize,
    pub text: String,
    pub section: usize,
    pub paragraph: usize,
    pub words: usize,
    /// True for bullet / numbered list items.
    pub list_item: bool,
    /// For list items: the sentence ending in ':' that introduces the list.
    pub lead_in: Option<usize>,
    /// Eligible to be picked for the summary.
    pub candidate: bool,
}

#[derive(Debug, Clone)]
pub struct Section {
    pub title: Option<String>,
    /// Heading depth: 1 = top level, 2 = subsection, ... (0 = text before any heading).
    pub level: usize,
    /// References, acknowledgements and similar: never summarized.
    pub skip: bool,
    pub sentences: Range<usize>,
}

#[derive(Debug, Default)]
pub struct Document {
    pub sections: Vec<Section>,
    pub sentences: Vec<Sentence>,
}

#[derive(Debug, Clone)]
pub struct ParseOptions {
    pub min_words: usize,
    /// Rejoin words split across a line break with a hyphen ("exam-\nple").
    pub dehyphenate: bool,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self { min_words: 6, dehyphenate: false }
    }
}

macro_rules! re {
    ($name:ident, $pat:expr) => {
        fn $name() -> &'static Regex {
            static R: OnceLock<Regex> = OnceLock::new();
            R.get_or_init(|| Regex::new($pat).expect("valid regex"))
        }
    };
}

re!(re_md_heading, r"^\s{0,3}(?:#{1,6}\s+(.+?)\s*#*|={2,6}\s*([^=].*?)\s*={2,6})\s*$");
re!(re_rule, r"^\s*(?:={3,}|-{3,}|\*{3,}|_{3,}|(?:[-*_]\s){2,}[-*_])\s*$");
re!(
    re_bullet,
    r"^\s*[`\u{201C}\u{2018}\x22']{0,2}(?:([-*\u{2022}\u{2023}\u{25AA}\u{25E6}\u{2013}\u{2014}])\s+|\((\d{1,3}|[a-zA-Z]|[ivxlc]{1,5}|[IVXLC]{1,5})\)\s+|(\d{1,3}|[a-zA-Z]|[ivxlc]{2,5})\)\s+|(\d{1,2})\.\s+)"
);
re!(
    re_numbered_heading,
    r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVXLC]{1,6}\.|[A-Z]\.)\s+[A-Z]"
);
re!(
    re_keyword_heading,
    r"^(?i:section|sec\.|article|chapter|part|appendix|schedule|exhibit|annex|title)\s+(?:\d+[A-Za-z]?|[IVXLC]+|[A-Z])\b"
);
// US-bill style: "SEC. 2. FINDINGS. The Congress finds that ..."
re!(
    re_inline_heading,
    r"^((?:SECTION|SEC\.|ARTICLE|TITLE)\s+(?:\d+[A-Za-z]?|[IVXLC]+)\.?(?:\s+[A-Z][A-Z0-9 ,;'&/()\-]{2,90})?\.)\s+(\S.*\p{Ll}.*)$"
);
// Contract style: "12. Governing Law. This Agreement shall be governed by ..."
re!(
    re_inline_clause,
    r#"^((?:(?:Section|Article|Clause)\s+)?\d{1,2}(?:\.\d{1,2})?\.?\s+[A-Z][A-Za-z'&/\-]*(?:,?\s+(?:[A-Z][A-Za-z'&/\-]*|of|and|the|to|or|for|in|on)){0,6}\.)\s+([A-Z("\u{201C}].*\p{Ll}.*)$"#
);
re!(
    re_skip_title,
    r"(?i)^(?:references?|bibliography|works cited|literature cited|acknowledge?ments?|funding|conflicts? of interests?|competing interests?|declaration of (?:competing )?interests?|author contributions?|authors'? contributions?|table of contents|contents|keywords|index|about the authors?|related (?:articles|links|posts|stories|content|coverage)|see also|comments? \(\d+\)|\d+ comments|reader comments|leave a (?:comment|reply)|join the (?:conversation|discussion)|you (?:may|might) also like|more from [\w .'&-]+|most (?:read|popular|viewed)|recommended(?: for you| reading)?|trending(?: now)?|(?:sign up for )?(?:our )?newsletters?|advertisement|share this (?:article|story|page)|further reading|footnotes|endnotes|data availability(?: statement)?|supplementary (?:materials?|information))$"
);
re!(
    re_known_heading,
    r"(?i)^(?:abstract|summary|executive summary|overview|introduction|background|related work|prior work|preliminaries|methods?|methodology|materials and methods|approach|experiments?|experimental setup|evaluation|results?|results and discussion|findings|analysis|discussion|limitations|future work|conclusions?|concluding remarks|definitions|recitals|preamble|purpose|scope|key takeaways|takeaways)$"
);
re!(
    re_caption,
    r"^(?:Figure|Fig\.|Table|Tab\.|Source|Sources|Photo|Image|Credit|Chart|Exhibit|Listing|Algorithm|Copyright|\u{00A9})\s*[\dIVX.\-]*\s*[:.|]"
);
re!(re_bib_entry, r"^\[\d{1,3}\]\s");
// Web page furniture: consent banners, link labels, social and newsletter
// prompts, copyright and licence footers. Phrased the way sites address the
// reader, so ordinary sentences about cookies or newsletters still count.
re!(
    re_boilerplate,
    r"(?ix)
      ^(?:read\ more|read\ also|also\ read|see\ also|related|more|watch|video|photos?|gallery|listen|recommended|sponsored|advertisement|ad)\s*[:|>\u{00BB}]
    | ^click\ here\b
    | ^skip\ to\ (?:main\ )?content\b
    | ^(?:\u{00A9}|copyright\ \u{00A9}?)\s*\d{4}
    | \ball\ rights\ reserved\b
    | \b(?:is|are)\ (?:licensed|available)\ under\ (?:a|the)\ creative\ commons\b
    | \b(?:we|this\ (?:web)?site|our\ (?:web)?site)\ uses?\ cookies\b
    | \bby\ (?:continuing\ to\ (?:use|browse)|using)\ (?:this|our)\ (?:web)?site\b
    | \b(?:accept|manage)\ (?:all\ )?cookies\b
    | \b(?:sign\ up|subscribe)\ (?:to|for)\ (?:our|the)\ (?:free\ )?(?:\w+\ )?newsletters?\b
    | ^(?:please\ )?(?:follow|like)\ (?:us|@\w+)\b
    | \bfollow\ (?:us|@\w+)\ on\ (?:twitter|x|facebook|instagram|linkedin|tiktok|threads|bluesky)\b
    | ^share\ (?:this|on)\b
    | \b(?:e-?mail|send)\ (?:this|it)\ to\ a\ friend\b
    | \bdownload\ (?:our|the)\ (?:free\ )?app\b
    | \b(?:enable|turn\ on)\ javascript\b|\bjavascript\ is\ (?:disabled|required)\b
    "
);
// Bylines and datelines are page metadata, not section headings.
re!(re_byline, r"(?i)^(?:by\s+\p{Lu}|(?:published|updated|last updated|posted)\b)");
// "This section describes ...", "The rest of the paper is organized as follows."
re!(
    re_meta,
    r"(?i)^(?:in |throughout )?(?:this|the (?:next|following|previous|remaining|rest|remainder)|each)(?: of (?:this|the))? (?:sub)?(?:section|chapter|appendix|paper is|article is|document is|report is)\b|\b(?:organi[sz]ed|structured) as follows\b"
);
re!(re_strip_numbering, r"^(?:(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVXLC]{1,6}\.|[A-Z]\.)\s+)");

/// Parse raw text (plain text or Markdown) into a `Document`.
pub fn parse(raw: &str, opts: &ParseOptions) -> Document {
    let mut text = raw
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\u{000C}', "\n\n")
        .replace('\u{00AD}', "")
        .replace(['\u{00A0}', '\u{2009}', '\u{202F}', '\t'], " ");
    if opts.dehyphenate {
        static R: OnceLock<Regex> = OnceLock::new();
        let r = R.get_or_init(|| Regex::new(r"(\p{Ll})-\n[ ]*(\p{Ll})").unwrap());
        text = r.replace_all(&text, "$1$2").into_owned();
    }

    let lines: Vec<&str> = text.lines().collect();
    let hard_wrapped = is_hard_wrapped(&lines);

    // Group lines into blocks (paragraph-level units).
    let mut blocks: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut in_code = false;
    for line in &lines {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code = !in_code;
            flush(&mut current, &mut blocks);
            continue;
        }
        if in_code {
            continue;
        }
        // Tables and images carry no extractable prose.
        if trimmed.starts_with('|') || trimmed.starts_with("![") {
            flush(&mut current, &mut blocks);
            continue;
        }
        if trimmed.is_empty() {
            flush(&mut current, &mut blocks);
            continue;
        }
        let cleaned = trimmed.trim_start_matches("> ").trim_start_matches('>').trim();
        if cleaned.is_empty() {
            flush(&mut current, &mut blocks);
            continue;
        }
        if re_md_heading().is_match(cleaned) {
            flush(&mut current, &mut blocks);
            blocks.push(vec![cleaned.to_string()]);
            continue;
        }
        current.push(cleaned.to_string());
        if !hard_wrapped {
            flush(&mut current, &mut blocks);
        }
    }
    flush(&mut current, &mut blocks);

    let mut b = Builder::new(opts.clone());
    let n = blocks.len();
    let mut i = 0;
    while i < n {
        let block = &blocks[i];
        let first = block[0].as_str();

        // Markdown "# Heading"
        if let Some(c) = re_md_heading().captures(first) {
            let title = c.get(1).or_else(|| c.get(2)).map(|m| m.as_str().trim().to_string());
            let marks = first.trim_start().chars().take_while(|&ch| ch == '#' || ch == '=').count();
            let level = if first.trim_start().starts_with('=') { marks.saturating_sub(1).max(1) } else { marks };
            b.start_section(title, level);
            if block.len() > 1 {
                b.add_paragraph(&block[1..]);
            }
            i += 1;
            continue;
        }
        // Setext heading: a line underlined with === or ---
        if block.len() == 2 && re_rule().is_match(&block[1]) && word_count(first) <= 14 {
            let level = if block[1].trim_start().starts_with('=') { 1 } else { 2 };
            b.start_section(Some(first.trim().to_string()), level);
            i += 1;
            continue;
        }
        if block.len() == 1 && re_rule().is_match(first) {
            i += 1;
            continue;
        }
        let listy = b.in_list_context() && re_bullet().is_match(first);
        // "SEC. 2. FINDINGS. Body text continues on the same line..."
        if !listy {
            let caps = re_inline_heading().captures(first).or_else(|| re_inline_clause().captures(first));
            if let Some(c) = caps {
                b.start_section(Some(c[1].trim().to_string()), numbering_depth(&c[1]));
                let mut rest: Vec<String> = vec![c[2].to_string()];
                rest.extend(block[1..].iter().cloned());
                b.add_paragraph(&rest);
                i += 1;
                continue;
            }
        }
        // Stand-alone heading line
        if block.len() == 1 && !listy {
            let next_words = blocks
                .get(i + 1)
                .map(|nb| nb.iter().map(|l| word_count(l)).sum::<usize>())
                .unwrap_or(0);
            if is_heading_line(first, next_words) {
                b.start_section(Some(first.trim().to_string()), numbering_depth(first));
                i += 1;
                continue;
            }
        }
        b.add_paragraph(block);
        i += 1;
    }
    b.finish()
}

fn flush(current: &mut Vec<String>, blocks: &mut Vec<Vec<String>>) {
    if !current.is_empty() {
        blocks.push(std::mem::take(current));
    }
}

/// Hard-wrapped text has short lines and uses blank lines between paragraphs.
/// Otherwise every newline is treated as a paragraph break.
fn is_hard_wrapped(lines: &[&str]) -> bool {
    let mut lens: Vec<usize> = lines
        .iter()
        .map(|l| l.trim().chars().count())
        .filter(|&l| l > 0)
        .collect();
    if lens.is_empty() {
        return true;
    }
    lens.sort_unstable();
    let p90 = lens[(lens.len() * 9 / 10).min(lens.len() - 1)];
    p90 <= 160
}

fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

fn is_heading_line(line: &str, next_block_words: usize) -> bool {
    let t = line.trim();
    let words = word_count(t);
    if words == 0 || words > 14 || t.chars().count() > 120 {
        return false;
    }
    let letters = t.chars().filter(|c| c.is_alphabetic()).count();
    if letters < 2 {
        return false;
    }
    let last = t.chars().last().unwrap();
    let ends_terminal = matches!(last, '.' | '?' | '!' | ';' | ',');
    let bare = t.trim_end_matches([':', '.']).trim();
    let unnumbered = re_strip_numbering().replace(bare, "");

    if re_byline().is_match(t) {
        return false;
    }
    if re_known_heading().is_match(unnumbered.trim()) || re_skip_title().is_match(unnumbered.trim()) {
        return true;
    }
    if re_keyword_heading().is_match(t) && (!ends_terminal || last == '.') {
        return true;
    }
    if t.starts_with('\u{00A7}') && words <= 10 {
        return true;
    }
    if re_numbered_heading().is_match(t) && (!ends_terminal || (last == '.' && words <= 8 && is_title_case(t))) {
        return true;
    }
    if ends_terminal || last == ':' {
        return false;
    }
    // ALL CAPS line
    let lower = t.chars().filter(|c| c.is_lowercase()).count();
    if letters >= 4 && lower == 0 && words <= 12 {
        return true;
    }
    let first = t.chars().next().unwrap();
    if !(first.is_uppercase() || first.is_ascii_digit()) {
        return false;
    }
    if words > 10 {
        return false;
    }
    if is_title_case(t) {
        return true;
    }
    // Sentence-case heading: a short unpunctuated line followed by a real paragraph.
    words <= 10 && next_block_words >= 25
}

/// "3 Methods" -> 1, "3.2 Attention" -> 2, "3.2.1 Details" -> 3, unnumbered -> 1.
fn numbering_depth(title: &str) -> usize {
    static R: OnceLock<Regex> = OnceLock::new();
    let r = R.get_or_init(|| Regex::new(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s").unwrap());
    match r.captures(title) {
        Some(c) => c[1].split('.').count(),
        None => 1,
    }
}

fn is_title_case(t: &str) -> bool {
    const SMALL: [&str; 20] = [
        "a", "an", "the", "of", "and", "or", "in", "on", "for", "to", "with", "by", "at", "from",
        "vs", "vs.", "as", "is", "are", "into",
    ];
    t.split_whitespace().all(|w| {
        let w0 = w.trim_start_matches(|c: char| !c.is_alphanumeric());
        match w0.chars().next() {
            None => true,
            Some(c) => c.is_uppercase() || c.is_ascii_digit() || SMALL.contains(&w0.to_lowercase().as_str()),
        }
    })
}

/// Kind of list marker, used to tell siblings from nested items.
type Marker = (u8, u8);

fn marker_of(line: &str, stack: &[ListLevel]) -> Option<(Marker, String)> {
    let c = re_bullet().captures(line)?;
    if c.get(1).is_some() {
        return Some(((0, 0), String::new()));
    }
    let (style, m) = if let Some(m) = c.get(2) {
        (1u8, m.as_str())
    } else if let Some(m) = c.get(3) {
        (2u8, m.as_str())
    } else {
        (3u8, c.get(4)?.as_str())
    };
    let roman = |x: &str| x.chars().all(|ch| matches!(ch.to_ascii_lowercase(), 'i' | 'v' | 'x' | 'l' | 'c'));
    let upper = m.chars().all(|ch| ch.is_uppercase());
    let kind = if m.chars().all(|ch| ch.is_ascii_digit()) {
        1
    } else if m.chars().count() == 1 {
        let ch = m.chars().next().unwrap().to_ascii_lowercase();
        // "(i)" is a roman numeral unless it follows "(h)" in a lettered list.
        let letter_kind = if upper { 3 } else { 2 };
        let follows_letter = stack.iter().rev().find(|l| l.marker == (style, letter_kind)).map(|l| {
            let prev = l.last.chars().next().map(|p| p.to_ascii_lowercase());
            prev.map(|p| (p as u8) + 1 == ch as u8).unwrap_or(false)
        });
        if matches!(ch, 'i' | 'v' | 'x') && follows_letter != Some(true) {
            if upper { 5 } else { 4 }
        } else {
            letter_kind
        }
    } else if roman(m) {
        if upper { 5 } else { 4 }
    } else {
        6
    };
    Some(((style, kind), m.to_string()))
}

struct ListLevel {
    marker: Marker,
    /// The most recent marker text at this level ("h", "3", ...).
    last: String,
    /// Sentence that introduces this list, if any.
    lead: Option<usize>,
}

struct Builder {
    opts: ParseOptions,
    sentences: Vec<Sentence>,
    /// (title, level) for every heading seen, in order; index 0 is the text before any heading.
    headings: Vec<(Option<String>, usize)>,
    paragraph: usize,
    /// Most recent sentence that ends by introducing a list ("...:" or "...--").
    pending_lead: Option<usize>,
    list_stack: Vec<ListLevel>,
}

/// A lead-in plus its items is kept as one unit when it is no longer than this.
const MAX_MERGED_LIST_WORDS: usize = 90;

impl Builder {
    fn new(opts: ParseOptions) -> Self {
        Self {
            opts,
            sentences: Vec::new(),
            headings: vec![(None, 0)],
            paragraph: 0,
            pending_lead: None,
            list_stack: Vec::new(),
        }
    }

    fn in_list_context(&self) -> bool {
        !self.list_stack.is_empty() || self.pending_lead.is_some()
    }

    fn start_section(&mut self, title: Option<String>, level: usize) {
        self.headings.push((title, level.max(1)));
        self.pending_lead = None;
        self.list_stack.clear();
    }

    fn add_paragraph(&mut self, lines: &[String]) {
        // Split the block into prose runs and list items.
        let bullet_lines = lines.iter().filter(|l| re_bullet().is_match(l)).count();
        let mut had_item = false;
        let mut units: Vec<(bool, String)> = Vec::new();
        for line in lines {
            let t = line.trim();
            let symbol = re_bullet().captures(t).map(|c| c.get(1).is_some()).unwrap_or(false);
            // "1." / "(a)" only count as list markers inside a list; a lone
            // numbered paragraph is a clause.
            let is_item = re_bullet().is_match(t)
                && (symbol || had_item || bullet_lines >= 2 || self.in_list_context());
            if is_item || units.is_empty() {
                had_item |= is_item;
                units.push((is_item, t.to_string()));
            } else {
                let last = units.last_mut().unwrap();
                last.1.push(' ');
                last.1.push_str(t);
            }
        }
        let section = self.headings.len() - 1;
        for (is_item, text) in units {
            let text = collapse_ws(&text);
            if text.is_empty() {
                continue;
            }
            let para = self.paragraph;
            self.paragraph += 1;

            let mut lead = None;
            if is_item {
                if let Some((marker, label)) = marker_of(&text, &self.list_stack) {
                    match self.list_stack.iter().rposition(|l| l.marker == marker) {
                        // A sibling of an earlier item: same list, same lead-in.
                        Some(pos) => {
                            self.list_stack.truncate(pos + 1);
                            self.list_stack[pos].last = label;
                            lead = self.list_stack[pos].lead;
                        }
                        // A new (possibly nested) list.
                        None => {
                            lead = self.pending_lead;
                            self.list_stack.push(ListLevel { marker, last: label, lead });
                        }
                    }
                }
            } else {
                self.list_stack.clear();
            }
            self.pending_lead = None;

            for (k, s) in split_sentences(&text).into_iter().enumerate() {
                let index = self.sentences.len();
                let words = word_count(&s);
                let candidate = self.is_candidate(&s, words);
                self.pending_lead = if ends_with_lead_in(&s) { Some(index) } else { None };
                self.sentences.push(Sentence {
                    index,
                    text: s,
                    section,
                    paragraph: para,
                    words,
                    list_item: is_item && k == 0,
                    lead_in: if k == 0 { lead } else { None },
                    candidate,
                });
            }
        }
    }

    fn is_candidate(&self, s: &str, words: usize) -> bool {
        if words < self.opts.min_words {
            return false;
        }
        let t = s.trim_end();
        if t.ends_with('?') || ends_with_lead_in(t) {
            return false;
        }
        if re_caption().is_match(t) || re_bib_entry().is_match(t) || re_meta().is_match(t) {
            return false;
        }
        if re_boilerplate().is_match(t) || is_menu(t) {
            return false;
        }
        let non_space = t.chars().filter(|c| !c.is_whitespace()).count().max(1);
        let alpha = t.chars().filter(|c| c.is_alphabetic()).count();
        if (alpha as f32) / (non_space as f32) < 0.6 {
            return false;
        }
        let lower_words = t
            .split_whitespace()
            .filter(|w| w.chars().any(|c| c.is_lowercase()))
            .count();
        lower_words >= 3
    }

    /// Fold short lists into the sentence that introduces them, so
    /// "The term 'security' means-- (A) stock, (B) bonds, and (C) derivatives."
    /// is one self-contained unit instead of four fragments.
    fn merge_short_lists(&mut self) {
        let n = self.sentences.len();
        let mut absorbed = vec![false; n];
        for i in (0..n).rev() {
            if !ends_with_lead_in(&self.sentences[i].text) {
                continue;
            }
            let mut j = i + 1;
            let mut words = self.sentences[i].words;
            let mut items: Vec<usize> = Vec::new();
            while j < n {
                if absorbed[j] {
                    j += 1;
                    continue;
                }
                let s = &self.sentences[j];
                if !(s.list_item && s.lead_in == Some(i) && s.section == self.sentences[i].section) {
                    break;
                }
                // An item that itself still introduces a (long) sub-list cannot be folded in.
                if ends_with_lead_in(&s.text) {
                    items.clear();
                    break;
                }
                // The item must be a single sentence.
                let next_free = (j + 1..n).find(|&k| !absorbed[k]);
                if let Some(k) = next_free {
                    let nx = &self.sentences[k];
                    if nx.paragraph == s.paragraph {
                        items.clear();
                        break;
                    }
                }
                words += s.words;
                items.push(j);
                j += 1;
            }
            if items.is_empty() || words > MAX_MERGED_LIST_WORDS {
                continue;
            }
            let mut text = std::mem::take(&mut self.sentences[i].text);
            for &k in &items {
                text.push(' ');
                text.push_str(&self.sentences[k].text);
                absorbed[k] = true;
            }
            let candidate = self.is_candidate(&text, words);
            let s = &mut self.sentences[i];
            s.text = text;
            s.words = words;
            s.candidate = candidate;
        }
        if !absorbed.iter().any(|&a| a) {
            return;
        }
        let mut remap = vec![usize::MAX; n];
        let mut kept: Vec<Sentence> = Vec::with_capacity(n);
        for (old, s) in std::mem::take(&mut self.sentences).into_iter().enumerate() {
            if absorbed[old] {
                continue;
            }
            remap[old] = kept.len();
            kept.push(s);
        }
        for (new, s) in kept.iter_mut().enumerate() {
            s.index = new;
            s.lead_in = s.lead_in.and_then(|l| if remap[l] == usize::MAX { None } else { Some(remap[l]) });
        }
        self.sentences = kept;
    }

    fn finish(mut self) -> Document {
        self.merge_short_lists();
        let mut doc = Document::default();
        let n = self.sentences.len();
        let mut start = 0usize;
        // Headings with no text of their own ("6 Results" directly followed by
        // "6.1 ...") are kept, so their subsections still group under them.
        for (h, (title, level)) in self.headings.iter().enumerate() {
            let mut end = start;
            while end < n && self.sentences[end].section == h {
                end += 1;
            }
            let skip = title.as_deref().map(is_skip_title).unwrap_or(false);
            if skip {
                for s in &mut self.sentences[start..end] {
                    s.candidate = false;
                }
            }
            doc.sections.push(Section { title: title.clone(), level: *level, skip, sentences: start..end });
            start = end;
        }
        doc.sentences = self.sentences;
        doc
    }
}

/// "...the following:" / "...is amended--" introduce a list and say little alone.
fn ends_with_lead_in(s: &str) -> bool {
    let t = s.trim_end().trim_end_matches(is_closer).trim_end();
    t.ends_with(':') || t.ends_with("--") || t.ends_with('\u{2014}')
}

/// A run of short links ("Home | World | Business | about us"): three or more
/// items separated by bars or bullets, averaging four words or fewer.
fn is_menu(t: &str) -> bool {
    let items: Vec<&str> = t.split(['|', '\u{2022}', '\u{00B7}']).map(str::trim).filter(|x| !x.is_empty()).collect();
    items.len() >= 3 && items.iter().map(|x| word_count(x)).sum::<usize>() <= 4 * items.len()
}

fn is_skip_title(title: &str) -> bool {
    let bare = title.trim().trim_end_matches([':', '.']).trim();
    let unnumbered = re_strip_numbering().replace(bare, "");
    re_skip_title().is_match(unnumbered.trim())
}

fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for w in s.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(w);
    }
    out
}

// ---------------------------------------------------------------------------
// Sentence splitting
// ---------------------------------------------------------------------------

/// Abbreviations that are never the end of a sentence.
const NEVER: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "messrs", "mme", "mlle", "st", "mt", "ft", "capt", "sgt",
    "cmdr", "adm", "supt", "lt", "e.g", "i.e", "c.f", "n.b", "cf", "viz", "vs", "approx", "est",
    "ca", "resp", "incl", "excl", "avg", "dept", "univ", "figs", "eqs", "eqn", "secs", "sect",
    "chap", "arts", "paras", "vols", "nos", "refs", "pp", "thm", "lem", "subsec", "subdiv",
    "rptr", "exh",
];

/// Abbreviations that are also ordinary words ("art", "app", "mass", "no").
/// Lower-case they only count as abbreviations before a number ("art. 5");
/// capitalised they behave like `MAYBE`.
const AMBIGUOUS: &[&str] = &[
    "pres", "hon", "rev", "gen", "gov", "sen", "rep", "col", "fr", "fig", "eq", "tab", "sec",
    "ch", "art", "para", "par", "cl", "vol", "ref", "def", "prop", "cor", "alg", "app", "supp",
    "cir", "stat", "reg", "regs", "fed", "ct", "pub", "res", "amend", "const", "cong", "sess",
    "ex", "ann", "op", "cal", "mass", "div", "pt", "ser", "tit", "min", "max", "sq", "std", "no",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
];

/// Abbreviations that sometimes end a sentence; split only when the next word
/// is a typical sentence opener.
const MAYBE: &[&str] = &[
    "etc", "al", "inc", "ltd", "co", "corp", "llc", "plc", "bros", "jr", "sr", "esq", "ed", "eds",
    "trans", "seq", "ibid", "id", "cit", "ave", "blvd", "rd", "hrs", "yrs", "mos", "wks",
];

const STARTERS: &[&str] = &[
    "the", "this", "that", "these", "those", "it", "its", "he", "she", "they", "we", "i", "you",
    "his", "her", "their", "our", "my", "a", "an", "in", "on", "at", "for", "if", "as", "but",
    "and", "or", "nor", "so", "yet", "however", "there", "here", "when", "while", "after",
    "before", "although", "though", "since", "because", "to", "by", "with", "from", "under",
    "such", "each", "any", "all", "some", "most", "many", "both", "thus", "therefore", "moreover",
    "furthermore", "then", "what", "which", "who", "where", "why", "how", "not", "nothing",
    "one", "two", "three", "another", "other", "others", "more", "several", "according", "during",
    "despite", "until", "unless", "whether", "once", "now", "today", "also", "instead", "indeed",
    "still", "meanwhile", "finally", "first", "second", "third", "next", "later", "earlier",
    "between", "among", "within", "without", "over", "through", "upon", "only", "even", "just",
    "nevertheless", "consequently", "additionally", "similarly", "accordingly", "hence",
    "otherwise", "overall", "together", "given", "based", "using", "let", "note", "see",
    "subsequent", "subsequently", "recent", "recently", "previous", "previously", "every",
    "neither", "either", "few", "much", "little", "like", "unlike", "notwithstanding",
];

fn set(words: &'static [&'static str], cell: &'static OnceLock<HashSet<&'static str>>) -> &'static HashSet<&'static str> {
    cell.get_or_init(|| words.iter().copied().collect())
}

fn never() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    set(NEVER, &S)
}
fn ambiguous() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    set(AMBIGUOUS, &S)
}
fn maybe() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    set(MAYBE, &S)
}
fn starters() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    set(STARTERS, &S)
}

fn is_closer(c: char) -> bool {
    matches!(c, '"' | '\'' | ')' | ']' | '\u{201D}' | '\u{2019}' | '\u{00BB}')
}

fn is_opener(c: char) -> bool {
    matches!(c, '"' | '\'' | '(' | '[' | '\u{201C}' | '\u{2018}' | '\u{00AB}' | '`')
}

/// Split one paragraph (whitespace already collapsed) into sentences.
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < n {
        let c = chars[i];
        if matches!(c, '.' | '!' | '?' | '\u{2026}') {
            let mut j = i + 1;
            while j < n && matches!(chars[j], '.' | '!' | '?' | '\u{2026}') {
                j += 1;
            }
            let run = j - i;
            while j < n && is_closer(chars[j]) {
                j += 1;
            }
            // Trailing citation markers stay with the sentence: "...shown.[3] Next"
            if j >= n {
                break;
            }
            if !chars[j].is_whitespace() {
                i = j;
                continue;
            }
            let mut k = j;
            while k < n && chars[k].is_whitespace() {
                k += 1;
            }
            if k >= n {
                break;
            }
            if is_boundary(&chars, start, i, run, k) {
                push(&chars[start..j], &mut out);
                start = k;
                i = k;
            } else {
                i = j;
            }
            continue;
        }
        i += 1;
    }
    push(&chars[start..n], &mut out);
    out
}

fn push(chars: &[char], out: &mut Vec<String>) {
    let s: String = chars.iter().collect();
    let s = s.trim();
    if !s.is_empty() {
        out.push(s.to_string());
    }
}

/// Decide whether the punctuation at `p` ends a sentence, given that the next
/// non-space character is at `k`.
fn is_boundary(chars: &[char], start: usize, p: usize, run: usize, k: usize) -> bool {
    // A bibliography entry ("[3] Smith, J. A study of ... Nature, 2019.") is
    // full of initials and inner full stops; it ends only where the next entry
    // starts, so each entry stays one (non-candidate) unit.
    if is_bib_label(chars, start) {
        return is_bib_label(chars, k);
    }
    let n = chars.len();
    let mut a = k;
    while a < n && is_opener(chars[a]) && chars[a] != '(' {
        a += 1;
    }
    // A new enumerated clause: `... security. (b) Rate of tax` / `... thereof. (ii) any`
    if a < n && chars[a] == '(' {
        let mut e = a + 1;
        while e < n && e - a <= 5 && chars[e].is_alphanumeric() {
            e += 1;
        }
        let inner = &chars[a + 1..e];
        let enumerator = (inner.len() <= 2 && !inner.is_empty() && inner.iter().all(|c| c.is_ascii_digit()))
            || (inner.len() == 1 && inner[0].is_alphabetic())
            || (!inner.is_empty() && inner.len() <= 4 && inner.iter().all(|c| matches!(c.to_ascii_lowercase(), 'i' | 'v' | 'x')));
        if enumerator && e < n && chars[e] == ')' && e + 1 < n && chars[e + 1].is_whitespace() {
            return true;
        }
    }
    while a < n && is_opener(chars[a]) {
        a += 1;
    }
    if a >= n {
        return false;
    }
    let after = chars[a];
    if !(after.is_uppercase() || after.is_ascii_digit()) {
        return false;
    }
    if chars[p] != '.' || run > 1 {
        return true;
    }
    // Token immediately before the period.
    let mut ts = p;
    while ts > start && !chars[ts - 1].is_whitespace() {
        ts -= 1;
    }
    let raw: String = chars[ts..p].iter().collect();
    let token = raw.trim_start_matches(|c: char| is_opener(c) || c == '\u{00A7}');
    if token.is_empty() {
        return true;
    }
    // "1. The Supplier shall..." -- a leading enumerator, not a sentence.
    let enumerator = token.chars().all(|c| c.is_ascii_digit())
        || (token.chars().count() == 1 && token.chars().all(|c| c.is_alphabetic()))
        || (token.len() <= 5 && token.chars().all(|c| matches!(c, 'I' | 'V' | 'X' | 'L' | 'C')));
    if enumerator && ts == start {
        return false;
    }
    let token = token.trim_end_matches(is_closer);
    let lower = token.to_lowercase();
    let lower = lower.as_str();
    if never().contains(lower) || token == "v" {
        return false;
    }
    let capitalised = token.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
    let mut maybe_class = maybe().contains(lower);
    if ambiguous().contains(lower) {
        if after.is_ascii_digit() {
            return false;
        }
        if !capitalised {
            return true;
        }
        maybe_class = true;
    }
    let single_letter = lower.chars().count() == 1 && lower.chars().all(|c| c.is_alphabetic());
    let dotted = lower.contains('.')
        && lower
            .split('.')
            .all(|part| !part.is_empty() && part.chars().count() <= 3 && part.chars().all(|c| c.is_alphabetic()));
    if maybe_class || single_letter || dotted {
        if !after.is_uppercase() {
            return false;
        }
        let mut e = a;
        while e < n && (chars[e].is_alphabetic() || chars[e] == '\'') {
            e += 1;
        }
        let next: String = chars[a..e].iter().collect::<String>().to_lowercase();
        return starters().contains(next.as_str());
    }
    true
}

/// Does a numbered reference label such as "[12] " start at `i`? (The same
/// shape `re_bib_entry` matches.)
fn is_bib_label(chars: &[char], i: usize) -> bool {
    if chars.get(i) != Some(&'[') {
        return false;
    }
    let digits = chars[i + 1..].iter().take_while(|c| c.is_ascii_digit()).count();
    (1..=3).contains(&digits)
        && chars.get(i + 1 + digits) == Some(&']')
        && chars.get(i + 2 + digits).is_some_and(|c| c.is_whitespace())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(s: &str) -> Vec<String> {
        split_sentences(s)
    }

    #[test]
    fn basic() {
        assert_eq!(split("Hello world. This is a test. Done!"), vec!["Hello world.", "This is a test.", "Done!"]);
    }

    #[test]
    fn titles_and_initials() {
        let s = split("Dr. Smith met John F. Kennedy in Washington. They spoke briefly.");
        assert_eq!(s.len(), 2);
        assert!(s[0].ends_with("Washington."));
    }

    #[test]
    fn academic() {
        let s = split("Smith et al. (2020) showed this in Fig. 3 and Eq. 2. The effect was small, i.e. Not significant at p < 0.05. We follow Jones et al. The method is simple.");
        assert_eq!(s.len(), 4, "{s:#?}");
        assert!(s[0].ends_with("Eq. 2."));
        assert!(s[2].ends_with("et al."));
    }

    #[test]
    fn legal() {
        let s = split("See 42 U.S.C. § 1983 and Smith v. Jones, 123 F. Supp. 2d 456 (9th Cir. 2001). The court agreed. Pub. L. No. 104-191 applies.");
        assert_eq!(s.len(), 3, "{s:#?}");
        assert!(s[0].ends_with("2001)."));
    }

    #[test]
    fn us_sentence_end() {
        let s = split("He moved to the U.S. The move was hard. The U.S. Senate voted.");
        assert_eq!(s.len(), 3, "{s:#?}");
    }

    #[test]
    fn quotes_and_numbers() {
        let s = split("\"We are done.\" She left at 3.5 percent growth. Revenue hit $4.2 billion in 2019. 42 firms joined.");
        assert_eq!(s.len(), 4, "{s:#?}");
        let s = split("\"Why?\" he asked. Nobody answered.");
        assert_eq!(s.len(), 2, "{s:#?}");
    }

    #[test]
    fn statutory_clauses() {
        let s = split("``(a) Imposition of Tax.--There is hereby imposed a tax on each covered transaction. ``(b) Rate of Tax.--The tax imposed shall be 0.03 percent. (c) Other.--Nothing here applies.");
        assert_eq!(s.len(), 3, "{s:#?}");
        assert!(s[1].starts_with("``(b)"));
    }

    #[test]
    fn enumerated_clause() {
        let s = split("1. The Supplier shall deliver the Goods. The Buyer shall pay.");
        assert_eq!(s.len(), 2, "{s:#?}");
        assert!(s[0].starts_with("1. The Supplier"));
    }

    #[test]
    fn sections_markdown() {
        let doc = parse(
            "# Title\n\nIntro paragraph with enough words to count as a sentence here.\n\n## Methods\n\nWe did a thing with many words in this particular sentence. And another thing happened here too okay.\n\n## References\n\n[1] Someone, A. A paper about things and stuff, 2020.\n",
            &ParseOptions::default(),
        );
        // Section 0 is the (empty) text before the first heading.
        assert_eq!(doc.sections.len(), 4);
        assert_eq!(doc.sections[2].title.as_deref(), Some("Methods"));
        assert_eq!(doc.sections[2].level, 2);
        assert!(doc.sections[3].skip);
        assert!(doc.sentences.iter().filter(|s| s.section == 3).all(|s| !s.candidate));
    }

    #[test]
    fn web_boilerplate() {
        let doc = parse(include_str!("../samples/web-article.txt"), &ParseOptions::default());
        let candidates: Vec<&str> = doc.sentences.iter().filter(|s| s.candidate).map(|s| s.text.as_str()).collect();
        for junk in [
            "Home | World",
            "We use cookies",
            "By continuing to use this site",
            "READ MORE:",
            "Share this article",
            "Sign up for our free daily newsletter",
            "Click here",
            "Follow @CityDesk",
            "best news I have read",
            "Privacy policy |",
            "All rights reserved",
            "Creative Commons",
        ] {
            assert!(!candidates.iter().any(|c| c.contains(junk)), "boilerplate is a candidate: {junk}");
        }
        assert!(candidates.iter().any(|c| c.starts_with("The city council voted")));
        assert!(candidates.iter().any(|c| c.starts_with("Ecologists say")));
        // Byline and dateline are not headings.
        assert!(doc.sections.iter().all(|s| s.title.as_deref().is_none_or(|t| !t.starts_with("By ") && !t.starts_with("Published"))));
        // Ordinary sentences about the same topics are kept.
        for ok in [
            "The bakery says it uses cookies from a family recipe in every gift box.",
            "Readers who subscribe to the print edition receive a weekly supplement.",
            "Copyright law protects the expression of an idea, not the idea itself.",
            "(C) by striking the period at the end and inserting a semicolon.",
            "(c) by redesignating paragraphs (5) and (6) as paragraphs (6) and (7).",
        ] {
            let d = parse(ok, &ParseOptions::default());
            assert!(d.sentences[0].candidate, "dropped: {ok}");
        }
    }

    #[test]
    fn sections_bill() {
        let doc = parse(
            "SECTION 1. SHORT TITLE. This Act may be cited as the Example Act of 2020.\n\nSEC. 2. FINDINGS. Congress finds that examples are useful for testing software. Congress also finds more things.\n",
            &ParseOptions::default(),
        );
        assert_eq!(doc.sections.len(), 3, "{:#?}", doc.sections);
        assert_eq!(doc.sections[2].title.as_deref(), Some("SEC. 2. FINDINGS."));
        assert_eq!(doc.sections[2].sentences.len(), 2);
    }

    #[test]
    fn short_list_merges_into_lead_in() {
        let doc = parse(
            "The Supplier shall do the following:\n- deliver the goods on time to the buyer's premises\n- provide all the required documentation with each delivery\n\nNext paragraph here with words.",
            &ParseOptions::default(),
        );
        assert_eq!(doc.sentences.len(), 2, "{:#?}", doc.sentences);
        assert!(doc.sentences[0].text.ends_with("with each delivery"));
        assert!(doc.sentences[0].candidate);
        assert_eq!(doc.sentences[1].index, 1);
    }

    #[test]
    fn long_list_keeps_lead_in_links() {
        let item = "the party shall deliver all of the goods described in the relevant schedule on time and in full to the premises named by the other party in writing before the delivery date";
        let text = format!("The Supplier shall do the following:\n(a) {item};\n(b) {item}, where:\n(i) {item};\n(ii) {item};\n(c) {item}.\n\nNext paragraph here with words.");
        let doc = parse(&text, &ParseOptions::default());
        let s = &doc.sentences;
        assert_eq!(s.len(), 7, "{s:#?}");
        assert_eq!(s[1].lead_in, Some(0));
        assert_eq!(s[2].lead_in, Some(0));
        assert_eq!(s[3].lead_in, Some(2), "nested item points at its own lead-in");
        assert_eq!(s[4].lead_in, Some(2));
        assert_eq!(s[5].lead_in, Some(0), "back to the outer list");
        assert!(!s[6].list_item);
    }

    #[test]
    fn hard_wrap_joined() {
        let doc = parse(
            "This is a sentence that has been\nwrapped across two lines by an editor. Second\nsentence here.\n\nNew paragraph starts here now.",
            &ParseOptions::default(),
        );
        assert_eq!(doc.sentences.len(), 3);
        assert_eq!(doc.sentences[0].text, "This is a sentence that has been wrapped across two lines by an editor.");
        assert_ne!(doc.sentences[1].paragraph, doc.sentences[2].paragraph);
    }

    const SAMPLES: &[&str] = &[
        include_str!("../samples/web-article.txt"),
        include_str!("../samples/us-bill.txt"),
        include_str!("../samples/apache-license.txt"),
    ];

    /// Invariants `select.rs` relies on without checking.
    fn assert_well_formed(doc: &Document, opts: &ParseOptions) {
        for (i, s) in doc.sentences.iter().enumerate() {
            assert_eq!(s.index, i, "sentence indices are not consecutive");
            assert!(s.section < doc.sections.len());
            assert!(doc.sections[s.section].sentences.contains(&i), "sentence {i} outside its section's range");
            assert_eq!(s.words, s.text.split_whitespace().count(), "word count of {:?}", s.text);
            assert!(!s.text.is_empty() && s.text == s.text.trim() && !s.text.contains("  ") && !s.text.contains('\n'));
            if let Some(l) = s.lead_in {
                assert!(l < i, "lead-in {l} does not precede item {i}");
                assert!(s.list_item);
            }
            if s.candidate {
                assert!(s.words >= opts.min_words, "short candidate {:?}", s.text);
                assert!(!doc.sections[s.section].skip, "candidate in a skipped section: {:?}", s.text);
            }
            if i > 0 {
                let p = &doc.sentences[i - 1];
                assert!(p.section <= s.section && p.paragraph <= s.paragraph, "order broken at {i}");
            }
        }
        // Section ranges tile the sentence list in order.
        let mut next = 0;
        for sec in &doc.sections {
            assert_eq!(sec.sentences.start, next, "{:#?}", doc.sections);
            next = sec.sentences.end;
        }
        assert_eq!(next, doc.sentences.len());
    }

    fn non_space(s: &str) -> String {
        s.chars().filter(|c| !c.is_whitespace()).collect()
    }

    #[test]
    fn samples_are_well_formed() {
        for opts in [ParseOptions::default(), ParseOptions { min_words: 1, dehyphenate: true }] {
            for text in SAMPLES {
                let doc = parse(text, &opts);
                assert!(doc.sentences.iter().filter(|s| s.candidate).count() >= 5);
                assert_well_formed(&doc, &opts);
            }
        }
    }

    #[test]
    fn splitting_loses_no_text() {
        // Sentences are verbatim: rejoined, they give back the paragraph.
        for text in SAMPLES {
            for para in text.split("\n\n") {
                let para = collapse_ws(para);
                let joined: String = split_sentences(&para).concat();
                assert_eq!(non_space(&joined), non_space(&para));
            }
        }
        for s in ["", " ", "...", "?!", "No terminal punctuation", "Ends with an ellipsis\u{2026}", "\"\"", "``''"] {
            assert_eq!(non_space(&split_sentences(s).concat()), non_space(s), "{s:?}");
        }
    }

    #[test]
    fn degenerate_input() {
        for raw in ["", "\n\n\n", "   \t  ", "#", "# \n", "- \n- \n", "\u{000C}\u{00AD}"] {
            let doc = parse(raw, &ParseOptions::default());
            assert!(doc.sentences.iter().all(|s| !s.candidate), "{raw:?}: {:#?}", doc.sentences);
            assert_well_formed(&doc, &ParseOptions::default());
        }
    }

    #[test]
    fn line_endings_do_not_matter() {
        let lf = SAMPLES[0];
        let texts = |d: Document| d.sentences.into_iter().map(|s| (s.text, s.section, s.candidate)).collect::<Vec<_>>();
        let opts = ParseOptions::default();
        assert_eq!(texts(parse(&lf.replace('\n', "\r\n"), &opts)), texts(parse(lf, &opts)));
        assert_eq!(texts(parse(&lf.replace('\n', "\r"), &opts)), texts(parse(lf, &opts)));
    }

    #[test]
    fn dehyphenation() {
        let raw = "The committee examined the environ-\nmental impact of the proposed dam carefully.";
        let on = parse(raw, &ParseOptions { dehyphenate: true, ..Default::default() });
        assert_eq!(on.sentences[0].text, "The committee examined the environmental impact of the proposed dam carefully.");
        let off = parse(raw, &ParseOptions::default());
        assert!(off.sentences[0].text.contains("environ-"), "{:?}", off.sentences[0].text);
    }

    #[test]
    fn non_candidates() {
        let opts = ParseOptions::default();
        for (s, why) in [
            ("What exactly did the committee decide on that day?", "question"),
            ("Figure 3: Accuracy of every model on the test set.", "caption"),
            ("[12] World Health Organization report on air quality and health, 2021 edition.", "bibliography entry"),
            ("[12] Smith, J. A study of things and stuff, 2020.", "bibliography entry with initials"),
            ("This section describes the method in more detail.", "signposting"),
            ("The following items are included in the package:", "lead-in"),
            ("12.5 33.1 47.9 88.0 91.2 14.4 63.0", "numbers"),
            ("Too short to matter.", "short"),
        ] {
            let doc = parse(s, &opts);
            assert!(doc.sentences.iter().all(|x| !x.candidate), "{why} is a candidate: {s}");
        }
    }

    #[test]
    fn bibliography_entries_stay_whole() {
        // Consecutive entries arrive as one paragraph. Each should come out as
        // one unit, not split at initials and inner full stops, where a long
        // title fragment would become a candidate.
        let refs = "[1] Smith, J. A. Learning to summarize long documents with sentence graphs. Journal of Language Engineering, 12(3), 2020. [2] Jones, K. B. and Lee, M. Extractive methods for very long legislative texts. Proc. ACL, pp. 1-9, 2019. [3] Lee, A. A survey of evaluation metrics for automatic text summarization systems, 2018.";
        let s = split(refs);
        assert_eq!(s.len(), 3, "{s:#?}");
        assert!(s[0].starts_with("[1] ") && s[1].starts_with("[2] ") && s[2].starts_with("[3] "));
        let doc = parse(&format!("Some ordinary introductory text that is long enough to count.\n\n{refs}"), &ParseOptions::default());
        assert!(doc.sentences[0].candidate);
        assert!(doc.sentences[1..].iter().all(|x| !x.candidate), "{:#?}", doc.sentences);

        // Citation markers inside prose do not change how prose is split.
        assert_eq!(
            split("The method was proposed earlier [3]. It was later extended [4, 5]. Results improved."),
            vec!["The method was proposed earlier [3].", "It was later extended [4, 5].", "Results improved."]
        );
    }

    #[test]
    fn deterministic() {
        for text in SAMPLES {
            let a = parse(text, &ParseOptions::default());
            let b = parse(text, &ParseOptions::default());
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
        }
    }
    mod props {
        //! The splitter loses no text, and `parse` keeps the invariants
        //! `select.rs` relies on, on generated documents and arbitrary strings.
        use super::*;
        use crate::testgen::{document, messy, paragraph};
        use proptest::prelude::*;

        fn texts(doc: &Document) -> Vec<(String, bool, usize)> {
            doc.sentences.iter().map(|s| (s.text.clone(), s.candidate, s.section)).collect()
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(300))]

            #[test]
            fn splitting_loses_no_text(p in prop_oneof![paragraph(), messy()]) {
                let para = collapse_ws(&p);
                let parts = split_sentences(&para);
                prop_assert_eq!(non_space(&parts.concat()), non_space(&para));
            }

            #[test]
            fn parse_is_well_formed(d in prop_oneof![3 => document(), 1 => messy()]) {
                for opts in [ParseOptions::default(), ParseOptions { min_words: 1, dehyphenate: true }] {
                    let doc = parse(&d, &opts);
                    assert_well_formed(&doc, &opts);
                    prop_assert_eq!(format!("{:?}", parse(&d, &opts)), format!("{:?}", doc), "not deterministic");
                }
            }

            #[test]
            fn sentences_are_verbatim(d in document()) {
                // Every sentence is the input's text, whitespace aside. (List
                // items lose their markers; arbitrary strings can also lose
                // soft hyphens and Markdown syntax, so this is for documents.)
                let doc = parse(&d, &ParseOptions::default());
                let flat = non_space(&d);
                for s in doc.sentences.iter().filter(|s| !s.list_item) {
                    prop_assert!(flat.contains(&non_space(&s.text)), "not verbatim: {:?}", s.text);
                }
            }

            #[test]
            fn line_endings_do_not_matter(d in document()) {
                let opts = ParseOptions::default();
                let lf = texts(&parse(&d, &opts));
                prop_assert_eq!(texts(&parse(&d.replace('\n', "\r\n"), &opts)), lf.clone());
                prop_assert_eq!(texts(&parse(&d.replace('\n', "\r"), &opts)), lf);
            }
        }
    }
}
