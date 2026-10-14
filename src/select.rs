//! Chooses which sentences make up the summary.
//!
//! 1. Score every candidate sentence by how central it is: how strongly the
//!    rest of the document (and the rest of its own section) echoes it.
//! 2. Give each top-level section a share of the budget so long documents are
//!    covered evenly.
//! 3. Pick greedily with MMR: prefer central sentences, penalise ones that
//!    repeat something already picked.
//! 4. Repair dangling references by pulling in the preceding sentence.
//!
//! Everything here is plain arithmetic on fixed vectors with index-ordered
//! tie-breaking, so the same input always produces the same picks.

use crate::embed::{dot, normalize};
use crate::segment::Document;
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Centrality {
    /// Similarity-graph centrality that counts later text more than earlier text.
    Directed,
    /// Cosine similarity to the average sentence vector.
    Centroid,
}

#[derive(Debug, Clone)]
pub struct Params {
    /// Number of sentences to select (before context sentences are added).
    pub budget: usize,
    /// MMR trade-off: 1.0 = relevance only, 0.0 = diversity only.
    pub lambda: f32,
    pub centrality: Centrality,
    /// Directed centrality: weight of similarity to *earlier* sentences
    /// (similarity to later sentences has weight 1).
    pub backward: f32,
    /// Directed centrality: similarities below this point of the min..max range are ignored.
    pub beta: f32,
    /// Bonus for sentences that open a section (lead bias), on a 0..1 relevance scale.
    pub lead: f32,
    /// Weight of section-level centrality vs whole-document centrality.
    pub section_weight: f32,
    /// Penalty for sentences that lean on text outside themselves
    /// ("However, ...", "This ...", "as shown in Figure 3").
    pub dependence_penalty: f32,
    /// Budget multiplier for the opening section and for abstract/summary/conclusion sections.
    pub summary_section_boost: f32,
    /// Pull in the previous sentence when a pick starts with "This", "However", ...
    pub repair: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            budget: 5,
            lambda: 0.7,
            centrality: Centrality::Directed,
            backward: 0.3,
            beta: 0.3,
            lead: 0.1,
            section_weight: 0.4,
            dependence_penalty: 0.08,
            summary_section_boost: 1.5,
            repair: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Chosen by the ranking.
    Selected,
    /// Added so a selected sentence makes sense on its own.
    Context,
}

#[derive(Debug, Clone)]
pub struct Pick {
    pub index: usize,
    pub kind: Kind,
    /// Relevance score, 0..1 within this document (selected sentences only).
    pub score: f32,
    /// Order in which the sentence was chosen (0 = first).
    pub rank: usize,
    /// Top-level section the sentence belongs to (index into `Selection::groups`).
    pub group: usize,
}

pub struct Selection {
    pub picks: Vec<Pick>,
    /// Heading of each top-level section.
    pub groups: Vec<Option<String>>,
}

/// Map every section to its top-level section ("3.2.1 Details" belongs to "3 Model").
fn group_sections(doc: &Document) -> (Vec<usize>, Vec<Option<String>>) {
    let levels: Vec<usize> = doc.sections.iter().map(|s| s.level).collect();
    let max_level = levels.iter().copied().max().unwrap_or(0);
    // The top level is the shallowest heading level that occurs more than once
    // (a lone level-1 heading is the document title, not a section).
    let top = (1..=max_level)
        .find(|&l| levels.iter().filter(|&&x| x == l).count() >= 2)
        .unwrap_or(max_level);
    let mut group_of = Vec::with_capacity(levels.len());
    let mut titles: Vec<Option<String>> = Vec::new();
    for (i, sec) in doc.sections.iter().enumerate() {
        if i == 0 || sec.level <= top {
            titles.push(sec.title.clone());
        }
        group_of.push(titles.len() - 1);
    }
    (group_of, titles)
}

fn summary_like(title: &str) -> bool {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)\b(?:abstract|summary|conclusions?|concluding remarks|key (?:points|takeaways|findings)|takeaways|highlights|overview|purpose|findings)\b").unwrap()
    })
    .is_match(title)
}

/// Sentences that point at figures, tables or equations, or carry leftover markup.
fn points_elsewhere(text: &str) -> bool {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)\b(?:fig(?:ure|s?\.)|tables?|tab\.|eq(?:uation|s?\.)|appendix|algorithm|listing)\s*\(?\d|\\[a-zA-Z]{2,}|_\{|\^\{|\bsee (?:above|below)\b|\bas (?:shown|described|discussed|noted|mentioned) (?:above|below|earlier|previously)\b").unwrap()
    })
    .is_match(text)
}

/// Mean thresholded similarity to the other sentences in `members`, with
/// similarity to later sentences weighted 1 and to earlier ones `backward`.
/// Rescaled to 0..1 over `members`.
fn directed_centrality(members: &[usize], sim: &dyn Fn(usize, usize) -> f32, backward: f32, beta: f32, out: &mut [f32]) {
    let m = members.len();
    if m < 2 {
        for &i in members {
            out[i] = 0.5;
        }
        return;
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for a in 0..m {
        for b in (a + 1)..m {
            let s = sim(members[a], members[b]);
            lo = lo.min(s);
            hi = hi.max(s);
        }
    }
    let threshold = lo + beta * (hi - lo);
    let mut raw = vec![0f32; m];
    for a in 0..m {
        let mut acc = 0f32;
        for b in 0..m {
            if a == b {
                continue;
            }
            let s = (sim(members[a], members[b]) - threshold).max(0.0);
            acc += if b > a { s } else { backward * s };
        }
        raw[a] = acc;
    }
    rescale(&mut raw);
    for (a, &i) in members.iter().enumerate() {
        out[i] = raw[a];
    }
}

fn rescale(v: &mut [f32]) {
    let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = hi - lo;
    for x in v.iter_mut() {
        *x = if span > 1e-9 { (*x - lo) / span } else { 0.5 };
    }
}

/// `embeddings[i]` is the vector for the i-th *candidate* sentence, in document order.
pub fn select(doc: &Document, embeddings: &[Vec<f32>], p: &Params) -> Selection {
    let (group_of_section, group_titles) = group_sections(doc);
    let cand: Vec<usize> = doc.sentences.iter().filter(|s| s.candidate).map(|s| s.index).collect();
    assert_eq!(cand.len(), embeddings.len());
    if cand.is_empty() || p.budget == 0 {
        return Selection { picks: Vec::new(), groups: group_titles };
    }
    let n = cand.len();
    let n_groups = group_titles.len().max(1);
    let group: Vec<usize> = cand.iter().map(|&si| group_of_section[doc.sentences[si].section]).collect();

    // Pairwise similarities (the vectors are unit length, so dot = cosine).
    // Beyond a few thousand sentences the matrix is too big to hold; compute on demand.
    let matrix: Option<Vec<f32>> = if n <= 6000 {
        let mut m = vec![0f32; n * n];
        for a in 0..n {
            m[a * n + a] = 1.0;
            for b in (a + 1)..n {
                let s = dot(&embeddings[a], &embeddings[b]);
                m[a * n + b] = s;
                m[b * n + a] = s;
            }
        }
        Some(m)
    } else {
        None
    };
    let sim = |a: usize, b: usize| -> f32 {
        match &matrix {
            Some(m) => m[a * n + b],
            None => dot(&embeddings[a], &embeddings[b]),
        }
    };

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    let mut group_words = vec![0usize; n_groups];
    for ci in 0..n {
        members[group[ci]].push(ci);
        group_words[group[ci]] += doc.sentences[cand[ci]].words;
    }

    // Centrality within the whole document and within each top-level section.
    let mut doc_c = vec![0f32; n];
    let mut sec_c = vec![0f32; n];
    let use_directed = p.centrality == Centrality::Directed && matrix.is_some();
    if use_directed {
        let all: Vec<usize> = (0..n).collect();
        directed_centrality(&all, &sim, p.backward, p.beta, &mut doc_c);
        for m in &members {
            directed_centrality(m, &sim, p.backward, p.beta, &mut sec_c);
        }
    } else {
        let dim = embeddings[0].len();
        let mut total = vec![0f32; dim];
        let mut per_group = vec![vec![0f32; dim]; n_groups];
        for ci in 0..n {
            for d in 0..dim {
                total[d] += embeddings[ci][d];
                per_group[group[ci]][d] += embeddings[ci][d];
            }
        }
        // Leave-one-out, so a sentence is not rewarded for resembling itself.
        let loo = |sum: &[f32], e: &[f32]| -> f32 {
            let mut c: Vec<f32> = sum.iter().zip(e).map(|(s, x)| s - x).collect();
            normalize(&mut c);
            dot(&c, e)
        };
        for ci in 0..n {
            doc_c[ci] = loo(&total, &embeddings[ci]);
            sec_c[ci] = loo(&per_group[group[ci]], &embeddings[ci]);
        }
        rescale(&mut doc_c);
        for m in &members {
            let mut vals: Vec<f32> = m.iter().map(|&i| sec_c[i]).collect();
            rescale(&mut vals);
            for (k, &i) in m.iter().enumerate() {
                sec_c[i] = vals[k];
            }
        }
    }

    // Relevance = centrality + lead bonus - dependence penalty.
    let mut pos_in_group = vec![0usize; n_groups];
    let mut rel = vec![0f32; n];
    for ci in 0..n {
        let s = &doc.sentences[cand[ci]];
        let g = group[ci];
        let pos = pos_in_group[g];
        pos_in_group[g] += 1;
        // Section-level centrality means little in very small sections.
        let sw = if n_groups == 1 { 0.0 } else { p.section_weight * ((members[g].len() as f32 - 1.0) / 4.0).clamp(0.0, 1.0) };
        let central = (1.0 - sw) * doc_c[ci] + sw * sec_c[ci];
        let lead = p.lead / (1.0 + pos as f32);
        let mut penalty = 0.0;
        if is_dangling(&s.text) {
            penalty += p.dependence_penalty;
        }
        if points_elsewhere(&s.text) {
            penalty += p.dependence_penalty;
        }
        if s.lead_in.is_some() {
            penalty += p.dependence_penalty;
        }
        rel[ci] = central + lead - penalty;
    }

    // Per-section share of the budget. Sub-linear weighting keeps a long
    // section from crowding out short but important ones.
    let budget = p.budget.min(n);
    let first_real_group = (0..n_groups).find(|&g| members[g].len() >= 2);
    let weights: Vec<f64> = (0..n_groups)
        .map(|g| {
            let mut w = (group_words[g] as f64).powf(0.75);
            let titled_summary = group_titles[g].as_deref().map(summary_like).unwrap_or(false);
            if n_groups > 1 && (Some(g) == first_real_group || titled_summary) {
                w *= p.summary_section_boost as f64;
            }
            w
        })
        .collect();
    let total: f64 = weights.iter().sum();
    let quota: Vec<f64> = weights.iter().map(|w| budget as f64 * w / total.max(1e-9)).collect();
    let floor: Vec<usize> = (0..n_groups).map(|g| (quota[g].floor() as usize).min(members[g].len())).collect();
    let cap: Vec<usize> = (0..n_groups)
        .map(|g| (quota[g].ceil() as usize).max(1).min(members[g].len()))
        .collect();

    let mut picked: Vec<usize> = Vec::with_capacity(budget); // candidate indices, in pick order
    let mut taken = vec![false; n];
    let mut per_group = vec![0usize; n_groups];
    let mut max_sim = vec![0f32; n]; // similarity to the closest already-picked sentence

    let pick_best = |allowed: &dyn Fn(usize, &[usize]) -> bool,
                     picked: &mut Vec<usize>,
                     taken: &mut Vec<bool>,
                     per_group: &mut Vec<usize>,
                     max_sim: &mut Vec<f32>|
     -> bool {
        let mut best: Option<(usize, f32)> = None;
        for ci in 0..n {
            if taken[ci] || !allowed(group[ci], per_group) {
                continue;
            }
            let score = p.lambda * rel[ci] - (1.0 - p.lambda) * max_sim[ci];
            // Strict '>' keeps the earliest sentence on ties, so the result is deterministic.
            if best.map(|(_, b)| score > b).unwrap_or(true) {
                best = Some((ci, score));
            }
        }
        match best {
            Some((ci, _)) => {
                taken[ci] = true;
                per_group[group[ci]] += 1;
                for other in 0..n {
                    if !taken[other] {
                        let s = sim(ci, other);
                        if s > max_sim[other] {
                            max_sim[other] = s;
                        }
                    }
                }
                picked.push(ci);
                true
            }
            None => false,
        }
    };

    // Phase 1: every section gets its guaranteed share.
    for g in 0..n_groups {
        for _ in 0..floor[g] {
            if picked.len() >= budget {
                break;
            }
            let only = move |x: usize, _: &[usize]| x == g;
            pick_best(&only, &mut picked, &mut taken, &mut per_group, &mut max_sim);
        }
    }
    // Phase 2: the remainder goes wherever the best sentences are, up to each section's cap.
    while picked.len() < budget {
        let cap_ref = &cap;
        let under_cap = move |x: usize, per: &[usize]| per[x] < cap_ref[x];
        if !pick_best(&under_cap, &mut picked, &mut taken, &mut per_group, &mut max_sim) {
            break;
        }
    }
    // Phase 3: if the caps left budget unused (tiny sections), fill from anywhere.
    while picked.len() < budget {
        let any = |_: usize, _: &[usize]| true;
        if !pick_best(&any, &mut picked, &mut taken, &mut per_group, &mut max_sim) {
            break;
        }
    }

    let sentence_group = |si: usize| group_of_section[doc.sentences[si].section];
    let mut picks: BTreeMap<usize, Pick> = BTreeMap::new();
    for (rank, &ci) in picked.iter().enumerate() {
        picks.insert(
            cand[ci],
            Pick { index: cand[ci], kind: Kind::Selected, score: rel[ci], rank, group: group[ci] },
        );
    }

    if p.repair {
        let max_extra = (budget + 1) / 2;
        let mut extra = 0usize;
        for (rank, &ci) in picked.iter().enumerate() {
            let mut cur = cand[ci];
            // A list item needs the sentence that introduces the list (and, for
            // nested lists, the one that introduces that).
            if doc.sentences[cur].lead_in.is_some() {
                // Not counted against the cap: an item without its lead-in is unreadable.
                for _ in 0..3 {
                    let Some(lead) = doc.sentences[cur].lead_in else { break };
                    if !picks.contains_key(&lead) {
                        picks.insert(
                            lead,
                            Pick { index: lead, kind: Kind::Context, score: 0.0, rank, group: sentence_group(lead) },
                        );
                    }
                    cur = lead;
                }
                continue;
            }
            // A dangling opener needs the sentence before it (at most two back).
            for _ in 0..2 {
                if cur == 0 || extra >= max_extra || !is_dangling(&doc.sentences[cur].text) {
                    break;
                }
                let prev = cur - 1;
                if doc.sentences[prev].section != doc.sentences[cur].section {
                    break;
                }
                if !picks.contains_key(&prev) {
                    picks.insert(
                        prev,
                        Pick { index: prev, kind: Kind::Context, score: 0.0, rank, group: sentence_group(prev) },
                    );
                    extra += 1;
                }
                cur = prev;
            }
        }
    }

    Selection { picks: picks.into_values().collect(), groups: group_titles }
}

/// Does this sentence open with a word that points back at earlier text?
pub fn is_dangling(text: &str) -> bool {
    let cleaned: String = text
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .chars()
        .take(60)
        .collect::<String>()
        .to_lowercase();
    let words: Vec<&str> = cleaned
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .collect();
    let Some(&first) = words.first() else { return false };
    let second = words.get(1).copied().unwrap_or("");

    const PHRASES: &[&str] = &[
        "in addition", "as a result", "for example", "for instance", "in contrast", "by contrast",
        "on the other hand", "in other words", "in particular", "as such", "that is", "after all",
        "in turn", "in that case", "in such case", "in such cases", "to this end", "for this reason",
        "for that reason", "as a consequence", "even so", "despite this", "despite that",
        "because of this", "because of that", "the latter", "the former", "the foregoing",
        "notwithstanding the foregoing", "in doing so", "at the same time", "in this case",
        "in these cases", "in this way", "of these", "of those", "to do so", "in both cases",
        "in either case", "more specifically", "more importantly", "even then", "since then",
        "by then", "until then", "the same", "to that end", "on the contrary", "in response",
        "other than that", "apart from that", "before that", "after that", "prior to that",
        "one is", "one of these", "one of them", "the other", "the first is", "the second is",
        "the third is", "the second", "the third", "the others", "among these", "among them",
    ];
    if PHRASES.iter().any(|ph| {
        cleaned.starts_with(ph)
            && cleaned[ph.len()..].chars().next().map(|c| !c.is_alphanumeric()).unwrap_or(true)
    }) {
        return true;
    }

    const CONNECTIVES: &[&str] = &[
        "however", "therefore", "thus", "hence", "moreover", "furthermore", "additionally", "also",
        "consequently", "nevertheless", "nonetheless", "instead", "otherwise", "similarly",
        "likewise", "accordingly", "meanwhile", "conversely", "yet", "but", "and", "or", "nor",
        "so", "then", "besides", "rather", "second", "secondly", "third", "thirdly", "fourth",
        "finally", "lastly", "next", "specifically", "still", "afterwards", "afterward",
        "subsequently", "alternatively", "indeed", "further", "later", "another", "others",
        "neither", "either", "both", "here", "there", "thereafter", "therein", "thereby", "thereof",
    ];
    if CONNECTIVES.contains(&first) {
        // "There is/are ..." introduces something new rather than pointing back.
        if first == "there" {
            return !matches!(second, "is" | "are" | "was" | "were" | "has" | "have" | "had" | "will" | "may" | "can" | "could" | "would" | "should" | "must" | "remains" | "remain" | "exists" | "exist" | "seems" | "appears");
        }
        return true;
    }

    const PRONOUNS: &[&str] = &[
        "he", "she", "they", "them", "his", "her", "hers", "their", "theirs", "its", "him",
        "he's", "she's", "they're", "she'd", "he'd", "they'd", "they've", "she'll", "he'll", "they'll",
    ];
    if PRONOUNS.contains(&first) {
        return true;
    }
    if first == "it" || first == "it's" {
        // "It is important to ...", "It is clear that ..." do not refer back.
        let expletive = words.iter().skip(1).take(6).any(|w| matches!(*w, "that" | "to" | "whether"));
        return !expletive;
    }
    if matches!(first, "this" | "that" | "these" | "those" | "such") {
        // "This paper", "This Agreement", "These terms" refer to the document itself.
        const SELF_NOUNS: &[&str] = &[
            "paper", "article", "study", "work", "report", "agreement", "act", "section", "chapter",
            "document", "review", "survey", "book", "essay", "thesis", "contract", "license",
            "licence", "policy", "guide", "post", "memo", "memorandum", "note", "research",
            "analysis", "title", "subsection", "paragraph", "clause", "part", "terms", "conditions",
            "lease", "deed", "notice", "order", "opinion", "case", "court", "appeal", "bill",
            "statute", "regulation", "rule", "tutorial", "page", "site", "website", "privacy",
            "letter", "manual", "standard", "specification", "proposal", "brief", "story", "week",
            "year", "month", "morning", "summer", "winter", "spring", "fall", "time",
        ];
        return !SELF_NOUNS.contains(&second);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dangling() {
        assert!(is_dangling("However, the results differ."));
        assert!(is_dangling("This suggests a problem."));
        assert!(is_dangling("They refused to comment."));
        assert!(is_dangling("In addition, the Buyer shall pay."));
        assert!(is_dangling("\"It was awful,\" she said."));
        assert!(is_dangling("She'd since returned on two occasions."));
        assert!(is_dangling("There they found a second body."));
        assert!(!is_dangling("There are many choices of encoding."));
        assert!(!is_dangling("This paper proposes a method."));
        assert!(!is_dangling("This Agreement is governed by Ontario law."));
        assert!(!is_dangling("It is important to note the limits."));
        assert!(!is_dangling("The company reported record profits."));
    }

    #[test]
    fn external_pointers() {
        assert!(points_elsewhere("As shown in Figure 2, accuracy improves."));
        assert!(points_elsewhere("In Table 3 rows (B), we observe a drop."));
        assert!(points_elsewhere("The value d_{k} matters."));
        assert!(!points_elsewhere("The table was set for dinner at eight."));
    }

    #[test]
    fn deterministic_and_diverse() {
        // Four fake unit vectors: 0 and 1 are near-duplicates, 2 and 3 differ.
        let doc = crate::segment::parse(
            "Alpha beta gamma delta epsilon zeta eta. Alpha beta gamma delta epsilon zeta theta. Completely different words appear in this one here. Yet another unrelated sentence about something else entirely.",
            &crate::segment::ParseOptions::default(),
        );
        let mut e = vec![vec![1.0, 0.0, 0.0], vec![0.99, 0.14, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        for v in e.iter_mut() {
            normalize(v);
        }
        let p = Params { budget: 2, repair: false, ..Default::default() };
        let a = select(&doc, &e, &p);
        let b = select(&doc, &e, &p);
        let ia: Vec<usize> = a.picks.iter().map(|x| x.index).collect();
        let ib: Vec<usize> = b.picks.iter().map(|x| x.index).collect();
        assert_eq!(ia, ib);
        assert!(!(ia.contains(&0) && ia.contains(&1)), "near-duplicates both picked: {ia:?}");
    }
}
