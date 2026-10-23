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
use crate::salience::Head;
use crate::segment::Document;
use regex::Regex;
use std::collections::btree_map::{BTreeMap, Entry};
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
    /// Ignored when `word_budget` is set.
    pub budget: usize,
    /// Select sentences up to this many words instead of `budget` sentences.
    /// Context sentences added by repair count against it.
    pub word_budget: Option<usize>,
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
    /// Learned salience head (see `salience.rs`); `None` uses centrality only.
    pub salience: Option<std::sync::Arc<Head>>,
    /// Share of the centrality term replaced by the learned salience score
    /// (0 = centrality only, 1 = salience only). Ignored without a head.
    pub salience_weight: f32,
    /// Return every candidate's feature vector in `Selection::features`.
    pub want_features: bool,
    /// Words each sentence will occupy in the output, by sentence index, when
    /// that differs from `Sentence::words` (compression). Word budget only.
    pub word_cost: Option<Vec<usize>>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            budget: 5,
            word_budget: None,
            lambda: 0.7,
            centrality: Centrality::Directed,
            backward: 0.3,
            beta: 0.3,
            lead: 0.1,
            section_weight: 0.4,
            dependence_penalty: 0.08,
            summary_section_boost: 1.5,
            repair: true,
            salience: None,
            salience_weight: 0.0,
            want_features: false,
            word_cost: None,
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
    /// With `Params::want_features`: (sentence index, features) per candidate,
    /// hand features as in `salience::HAND_FEATURES` followed by the embedding.
    pub features: Vec<(usize, Vec<f32>)>,
}

/// Budget bookkeeping while picking.
struct Fill {
    /// Budget spent per section.
    used: Vec<f64>,
    /// Budget spent overall.
    total: f64,
    /// Sentences already in the output, picked or as context (word budget only).
    covered: Vec<bool>,
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

/// Defaults for the knobs that should change with document length.
///
/// Short documents (news, most bills) put the main point first and elaborate
/// after it, so directed centrality, which counts later text more, finds it.
/// Long documents (reports, papers) are summarized from the whole text, and
/// there the direction works against the summary: on 100 GovReport
/// development reports, `backward` 1 and `lambda` 0.9 gain 4.7 ROUGE-2.
/// Between 60 and 150 candidate sentences the defaults move linearly from the
/// short-document values to the long ones. Of the development documents, no
/// bill and 17 of 500 news articles reach 60 (none reaches 100), while every
/// report and paper does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LengthDefaults {
    pub lambda: f32,
    pub backward: f32,
}

impl LengthDefaults {
    pub const SHORT: LengthDefaults = LengthDefaults { lambda: 0.7, backward: 0.3 };
    pub const LONG: LengthDefaults = LengthDefaults { lambda: 0.9, backward: 1.0 };
    const RAMP: (f32, f32) = (60.0, 150.0);

    pub fn for_candidates(n: usize) -> LengthDefaults {
        let t = ((n as f32 - Self::RAMP.0) / (Self::RAMP.1 - Self::RAMP.0)).clamp(0.0, 1.0);
        let mix = |a: f32, b: f32| a + (b - a) * t;
        LengthDefaults {
            lambda: mix(Self::SHORT.lambda, Self::LONG.lambda),
            backward: mix(Self::SHORT.backward, Self::LONG.backward),
        }
    }
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
    if cand.is_empty() || p.word_budget.unwrap_or(p.budget) == 0 {
        return Selection { picks: Vec::new(), groups: group_titles, features: Vec::new() };
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

    // Relevance = centrality (optionally blended with learned salience)
    // + lead bonus - dependence penalty.
    let first_real_group = (0..n_groups).find(|&g| members[g].len() >= 2);
    let is_summary_group = |g: usize| {
        n_groups > 1 && (Some(g) == first_real_group || group_titles[g].as_deref().map(summary_like).unwrap_or(false))
    };
    let use_head = p.salience.is_some() && p.salience_weight > 0.0;
    let mut pos_in_group = vec![0usize; n_groups];
    let mut central = vec![0f32; n];
    let mut lead_bonus = vec![0f32; n];
    let mut penalties = vec![0f32; n];
    let mut feats: Vec<Vec<f32>> = Vec::new();
    for ci in 0..n {
        let s = &doc.sentences[cand[ci]];
        let g = group[ci];
        let pos = pos_in_group[g];
        pos_in_group[g] += 1;
        // Section-level centrality means little in very small sections.
        let sw = if n_groups == 1 { 0.0 } else { p.section_weight * ((members[g].len() as f32 - 1.0) / 4.0).clamp(0.0, 1.0) };
        central[ci] = (1.0 - sw) * doc_c[ci] + sw * sec_c[ci];
        let lead = p.lead / (1.0 + pos as f32);
        let dangling = is_dangling(&s.text);
        let elsewhere = points_elsewhere(&s.text);
        let mut penalty = 0.0;
        if dangling {
            penalty += p.dependence_penalty;
        }
        if elsewhere {
            penalty += p.dependence_penalty;
        }
        if s.lead_in.is_some() {
            penalty += p.dependence_penalty;
        }
        lead_bonus[ci] = lead;
        penalties[ci] = penalty;

        if use_head || p.want_features {
            let flag = |b: bool| if b { 1.0 } else { 0.0 };
            let mut f = vec![
                doc_c[ci],
                sec_c[ci],
                ci as f32 / (n.max(2) - 1) as f32,
                (1.0 + ci as f32).ln(),
                pos as f32 / (members[g].len().max(2) - 1) as f32,
                flag(pos == 0),
                (s.words.max(1) as f32).ln(),
                flag(dangling),
                flag(elsewhere),
                flag(s.lead_in.is_some()),
                flag(s.list_item),
                flag(is_summary_group(g)),
            ];
            debug_assert_eq!(f.len(), crate::salience::HAND_FEATURES.len());
            f.extend_from_slice(&embeddings[ci]);
            feats.push(f);
        }
    }
    // The head's log-odds, rescaled to 0..1 within the document like centrality.
    let salience: Option<Vec<f32>> = match (&p.salience, use_head) {
        (Some(h), true) if feats.first().map(|f| f.len()) == Some(h.weights.len()) => {
            let mut v: Vec<f32> = feats.iter().map(|f| h.logit(f)).collect();
            rescale(&mut v);
            Some(v)
        }
        _ => None,
    };
    let rel: Vec<f32> = (0..n)
        .map(|ci| {
            let c = match &salience {
                Some(sal) => (1.0 - p.salience_weight) * central[ci] + p.salience_weight * sal[ci],
                None => central[ci],
            };
            c + lead_bonus[ci] - penalties[ci]
        })
        .collect();

    // Per-section share of the budget. Sub-linear weighting keeps a long
    // section from crowding out short but important ones.
    //
    // A pick costs 1 under a sentence budget and its word count under a word
    // budget; quotas, caps and the overall limit are all in that unit. With a
    // cost of 1 the arithmetic below is exactly integer sentence counting.
    //
    // Under a word budget a pick also pays for the context sentences repair will
    // add for it, minus anything already in the output, so the budget bounds
    // the whole summary.
    let out_words = |si: usize| match &p.word_cost {
        Some(w) => w[si],
        None => doc.sentences[si].words,
    } as f64;
    let words_of = |ci: usize| out_words(cand[ci]);
    let chains: Vec<Vec<usize>> = match (p.word_budget, p.repair) {
        (Some(_), true) => cand.iter().map(|&si| context_chain(doc, si).0).collect(),
        _ => vec![Vec::new(); n],
    };
    let cost = |ci: usize, f: &Fill| -> f64 {
        if p.word_budget.is_none() {
            return 1.0;
        }
        let mut c = if f.covered[cand[ci]] { 0.0 } else { words_of(ci) };
        for &si in &chains[ci] {
            if !f.covered[si] {
                c += out_words(si);
            }
        }
        c
    };
    let budget = p.budget.min(n);
    let limit: f64 = match p.word_budget {
        Some(w) => w as f64,
        None => budget as f64,
    };
    let weights: Vec<f64> = (0..n_groups)
        .map(|g| {
            let mut w = (group_words[g] as f64).powf(0.75);
            if is_summary_group(g) {
                w *= p.summary_section_boost as f64;
            }
            w
        })
        .collect();
    let total: f64 = weights.iter().sum();
    let quota: Vec<f64> = weights.iter().map(|w| limit * w / total.max(1e-9)).collect();
    // `floor`: share every section is guaranteed. `cap`: most a section may take
    // before the remainder goes elsewhere. Under a sentence budget these are the
    // rounded-down and rounded-up quotas; under a word budget a section may run
    // over its quota by about one average sentence.
    let avg_words = (0..n).map(words_of).sum::<f64>() / n as f64;
    let (floor, cap): (Vec<f64>, Vec<f64>) = (0..n_groups)
        .map(|g| match p.word_budget {
            Some(_) => (quota[g], quota[g] + avg_words),
            None => {
                let m = members[g].len();
                ((quota[g].floor() as usize).min(m) as f64, (quota[g].ceil() as usize).max(1).min(m) as f64)
            }
        })
        .unzip();

    let mut fill = Fill { used: vec![0.0; n_groups], total: 0.0, covered: vec![false; doc.sentences.len()] };
    let mut picked: Vec<usize> = Vec::with_capacity(budget); // candidate indices, in pick order
    let mut taken = vec![false; n];
    let mut max_sim = vec![0f32; n]; // similarity to the closest already-picked sentence

    let pick_best = |allowed: &dyn Fn(usize, &Fill) -> bool,
                     picked: &mut Vec<usize>,
                     taken: &mut Vec<bool>,
                     fill: &mut Fill,
                     max_sim: &mut Vec<f32>|
     -> bool {
        let mut best: Option<(usize, f32)> = None;
        for ci in 0..n {
            if taken[ci] || !allowed(ci, fill) {
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
                let c = cost(ci, fill);
                fill.used[group[ci]] += c;
                fill.total += c;
                fill.covered[cand[ci]] = true;
                for &si in &chains[ci] {
                    fill.covered[si] = true;
                }
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
    let fits = |ci: usize, f: &Fill| f.total + cost(ci, f) <= limit;

    // Phase 1: every section gets its guaranteed share.
    for (g, &share) in floor.iter().enumerate() {
        let in_share = |ci: usize, f: &Fill| group[ci] == g && f.used[g] + cost(ci, f) <= share && fits(ci, f);
        while pick_best(&in_share, &mut picked, &mut taken, &mut fill, &mut max_sim) {}
    }
    // Phase 2: the remainder goes wherever the best sentences are, up to each section's cap.
    let under_cap = |ci: usize, f: &Fill| f.used[group[ci]] + cost(ci, f) <= cap[group[ci]] && fits(ci, f);
    while pick_best(&under_cap, &mut picked, &mut taken, &mut fill, &mut max_sim) {}
    // Phase 3: if the caps left budget unused (tiny sections), fill from anywhere.
    while pick_best(&fits, &mut picked, &mut taken, &mut fill, &mut max_sim) {}
    // A word budget shorter than every sentence still yields one sentence.
    if picked.is_empty() {
        pick_best(&|_, _| true, &mut picked, &mut taken, &mut fill, &mut max_sim);
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
        // Under a sentence budget, dangling-opener context is capped at half the
        // picks. Under a word budget it was already paid for when the pick was made.
        let max_extra = if p.word_budget.is_some() { usize::MAX } else { picked.len().div_ceil(2) };
        let mut extra = 0usize;
        for (rank, &ci) in picked.iter().enumerate() {
            let (chain, capped) = context_chain(doc, cand[ci]);
            for si in chain {
                if capped && extra >= max_extra {
                    break;
                }
                if let Entry::Vacant(slot) = picks.entry(si) {
                    slot.insert(Pick { index: si, kind: Kind::Context, score: 0.0, rank, group: sentence_group(si) });
                    if capped {
                        extra += 1;
                    }
                }
            }
        }
    }

    let features = if p.want_features { cand.iter().copied().zip(feats).collect() } else { Vec::new() };
    Selection { picks: picks.into_values().collect(), groups: group_titles, features }
}

/// Sentences that must accompany sentence `si` for it to make sense, nearest
/// first, and whether they count against the sentence-mode context cap.
///
/// A list item needs the sentence that introduces its list (and, for nested
/// lists, the one that introduces that); these are never capped, since an item
/// without its lead-in is unreadable. A dangling opener ("This...",
/// "However...") needs the sentence before it, at most two back, within the
/// same section.
fn context_chain(doc: &Document, si: usize) -> (Vec<usize>, bool) {
    let mut chain = Vec::new();
    let mut cur = si;
    if doc.sentences[cur].lead_in.is_some() {
        for _ in 0..3 {
            let Some(lead) = doc.sentences[cur].lead_in else { break };
            chain.push(lead);
            cur = lead;
        }
        return (chain, false);
    }
    for _ in 0..2 {
        if cur == 0 || !is_dangling(&doc.sentences[cur].text) {
            break;
        }
        let prev = cur - 1;
        if doc.sentences[prev].section != doc.sentences[cur].section {
            break;
        }
        chain.push(prev);
        cur = prev;
    }
    (chain, true)
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
    #[test]
    fn length_defaults_ramp() {
        assert_eq!(LengthDefaults::for_candidates(0), LengthDefaults::SHORT);
        assert_eq!(LengthDefaults::for_candidates(60), LengthDefaults::SHORT);
        assert_eq!(LengthDefaults::for_candidates(150), LengthDefaults::LONG);
        assert_eq!(LengthDefaults::for_candidates(5000), LengthDefaults::LONG);
        let mid = LengthDefaults::for_candidates(105);
        assert!((mid.backward - 0.65).abs() < 1e-6 && (mid.lambda - 0.8).abs() < 1e-6);
    }

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

    #[test]
    fn word_budget() {
        let doc = crate::segment::parse(
            "Alpha beta gamma delta epsilon zeta eta. Completely different words appear in this one here today. Yet another unrelated sentence about something else entirely now. A fourth sentence that is quite a lot longer than all of the others put together here.",
            &crate::segment::ParseOptions::default(),
        );
        let mut e = vec![vec![1.0, 0.1, 0.0, 0.0], vec![0.1, 1.0, 0.0, 0.0], vec![0.0, 0.1, 1.0, 0.0], vec![0.0, 0.0, 0.1, 1.0]];
        for v in e.iter_mut() {
            normalize(v);
        }
        let words = |sel: &Selection| sel.picks.iter().map(|x| doc.sentences[x.index].words).sum::<usize>();
        for w in [8, 20, 30] {
            let sel = select(&doc, &e, &Params { word_budget: Some(w), repair: false, ..Default::default() });
            assert!(!sel.picks.is_empty());
            assert!(words(&sel) <= w, "budget {w}, got {} words", words(&sel));
        }
        // With repair on, context sentences count against the budget too:
        // sentence 1 opens "This", so picking it costs its predecessor's words.
        let doc2 = crate::segment::parse(
            "Alpha beta gamma delta epsilon zeta eta theta. This result was then confirmed by several other groups. Completely different words appear in this one here today.",
            &crate::segment::ParseOptions::default(),
        );
        let e2 = vec![vec![0.6, 0.8, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        for w in [10, 18, 30] {
            let sel = select(&doc2, &e2, &Params { word_budget: Some(w), ..Default::default() });
            let total: usize = sel.picks.iter().map(|x| doc2.sentences[x.index].words).sum();
            assert!(total <= w.max(9), "budget {w}, got {total} words including context");
        }
        // Shorter than any sentence: still one sentence.
        let sel = select(&doc, &e, &Params { word_budget: Some(2), repair: false, ..Default::default() });
        assert_eq!(sel.picks.len(), 1);
        // Large enough for everything.
        let sel = select(&doc, &e, &Params { word_budget: Some(1000), repair: false, ..Default::default() });
        assert_eq!(sel.picks.len(), 4);
    }

    use crate::segment::{parse, ParseOptions};

    /// A stand-in for the encoder: hashed bag of lowercase words, unit length.
    /// Sentences sharing words get similar vectors, which is all selection needs.
    fn bow(text: &str) -> Vec<f32> {
        use std::hash::{Hash, Hasher};
        let mut v = vec![0f32; 64];
        for w in text.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            w.to_lowercase().hash(&mut h);
            v[(h.finish() % 64) as usize] += 1.0;
        }
        v[0] += 1e-3; // never the zero vector
        normalize(&mut v);
        v
    }

    fn embed(doc: &Document) -> Vec<Vec<f32>> {
        doc.sentences.iter().filter(|s| s.candidate).map(|s| bow(&s.text)).collect()
    }

    fn selected(sel: &Selection) -> Vec<usize> {
        sel.picks.iter().filter(|p| p.kind == Kind::Selected).map(|p| p.index).collect()
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let mut v = v.to_vec();
        normalize(&mut v);
        v
    }

    #[test]
    fn invariants_on_real_documents() {
        let samples = [
            include_str!("../samples/web-article.txt"),
            include_str!("../samples/us-bill.txt"),
            include_str!("../samples/apache-license.txt"),
        ];
        for text in samples {
            let doc = parse(text, &ParseOptions::default());
            let e = embed(&doc);
            let n = e.len();
            for centrality in [Centrality::Directed, Centrality::Centroid] {
                for (budget, word_budget) in [(1, None), (3, None), (8, None), (n + 5, None), (0, Some(40)), (0, Some(150))] {
                    let p = Params { budget, word_budget, centrality, ..Default::default() };
                    let sel = select(&doc, &e, &p);
                    let what = format!("{centrality:?} budget={budget} words={word_budget:?}");

                    // Output is in document order, without duplicates.
                    assert!(sel.picks.windows(2).all(|w| w[0].index < w[1].index), "{what}");
                    let sel_idx = selected(&sel);
                    if word_budget.is_none() {
                        assert_eq!(sel_idx.len(), budget.min(n), "{what}");
                    }
                    let mut ranks: Vec<usize> = sel.picks.iter().filter(|p| p.kind == Kind::Selected).map(|p| p.rank).collect();
                    ranks.sort();
                    assert_eq!(ranks, (0..sel_idx.len()).collect::<Vec<_>>(), "{what}");
                    for pk in &sel.picks {
                        assert!(pk.group < sel.groups.len(), "{what}");
                        assert!(pk.score.is_finite(), "{what}");
                        match pk.kind {
                            Kind::Selected => assert!(doc.sentences[pk.index].candidate, "{what}: non-candidate selected"),
                            // Context comes before the pick it supports, in the same section or as its list's lead-in.
                            Kind::Context => assert!(
                                sel_idx.iter().any(|&s| s > pk.index
                                    && (s - pk.index <= 3 && doc.sentences[s].section == doc.sentences[pk.index].section
                                        || doc.sentences[s].lead_in.is_some())),
                                "{what}: stray context sentence {}",
                                pk.index
                            ),
                        }
                    }
                    if let Some(w) = word_budget {
                        let total: usize = sel.picks.iter().map(|p| doc.sentences[p.index].words).sum();
                        assert!(total <= w || sel_idx.len() == 1, "{what}: {total} words");
                    }
                    // Same input, same output.
                    let again = select(&doc, &e, &p);
                    assert_eq!(format!("{:?}", sel.picks), format!("{:?}", again.picks), "{what}");
                }
            }
        }
    }

    #[test]
    fn empty_and_zero_budget() {
        let doc = parse("", &ParseOptions::default());
        assert!(select(&doc, &[], &Params::default()).picks.is_empty());
        let doc = parse("A perfectly ordinary sentence with enough words in it.", &ParseOptions::default());
        let e = embed(&doc);
        assert!(select(&doc, &e, &Params { budget: 0, ..Default::default() }).picks.is_empty());
        assert!(select(&doc, &e, &Params { word_budget: Some(0), ..Default::default() }).picks.is_empty());
        assert_eq!(selected(&select(&doc, &e, &Params { budget: 10, ..Default::default() })), vec![0]);
    }

    #[test]
    fn central_beats_outlier() {
        // Three sentences on one topic and one off-topic: with relevance only,
        // the single pick must come from the topic cluster.
        let doc = parse(
            "The river flooded the old town after heavy rain this week. Heavy rain made the river flood the town again this week. The town council blamed the river flooding on the heavy rain. A local bakery won a national award for its sourdough bread.",
            &ParseOptions::default(),
        );
        let e = vec![unit(&[1.0, 0.1, 0.0]), unit(&[1.0, 0.0, 0.1]), unit(&[0.9, 0.1, 0.1]), unit(&[0.0, 0.0, 1.0])];
        for centrality in [Centrality::Directed, Centrality::Centroid] {
            let p = Params { budget: 1, lambda: 1.0, lead: 0.0, repair: false, centrality, ..Default::default() };
            let picks = selected(&select(&doc, &e, &p));
            assert_eq!(picks.len(), 1);
            assert_ne!(picks[0], 3, "{centrality:?} picked the outlier");
        }
        // With diversity weighted heavily, a second pick goes to the outlier
        // rather than another near-duplicate.
        let p = Params { budget: 2, lambda: 0.3, repair: false, ..Default::default() };
        assert!(selected(&select(&doc, &e, &p)).contains(&3));
    }

    #[test]
    fn sections_share_the_budget() {
        // Section A's sentences are all more central than section B's, but
        // with two equal sections and a budget of two, each gets one.
        let a = "Alpha topic sentence number one has plenty of words. Alpha topic sentence number two has plenty of words. Alpha topic sentence number three has plenty of words.";
        let b = "Beta subject line number one also has enough words. Beta subject line number two also has enough words. Beta subject line number three also has enough words.";
        let doc = parse(&format!("# Report\n\n## Part A\n\n{a}\n\n## Part B\n\n{b}\n"), &ParseOptions::default());
        let e = vec![
            unit(&[1.0, 0.0, 0.05]),
            unit(&[1.0, 0.05, 0.0]),
            unit(&[1.0, 0.02, 0.02]),
            unit(&[0.5, 1.0, 0.0]),
            unit(&[0.5, 0.9, 0.1]),
            unit(&[0.5, 1.0, 0.1]),
        ];
        let sel = select(&doc, &e, &Params { budget: 2, repair: false, ..Default::default() });
        let sections: Vec<usize> = selected(&sel).iter().map(|&i| doc.sentences[i].section).collect();
        assert_eq!(sections.len(), 2);
        assert_ne!(sections[0], sections[1], "both picks from one section: {sections:?}");
        assert!(sel.groups.iter().any(|g| g.as_deref() == Some("Part A")));
        assert!(sel.groups.iter().any(|g| g.as_deref() == Some("Part B")));
    }

    #[test]
    fn repair_brings_in_antecedent() {
        let doc = parse(
            "The committee approved a new budget for the city parks. This decision surprised many residents who expected cuts. Separately, the library extended its opening hours on weekends.",
            &ParseOptions::default(),
        );
        // The dangling sentence sits between the other two, so it is the most central.
        let e = vec![unit(&[1.0, 0.0, 0.0]), unit(&[1.0, 1.0, 1.0]), unit(&[0.0, 0.0, 1.0])];
        let base = Params { budget: 1, lambda: 1.0, dependence_penalty: 0.0, lead: 0.0, centrality: Centrality::Centroid, ..Default::default() };
        let on = select(&doc, &e, &base);
        assert_eq!(selected(&on), vec![1]);
        let ctx: Vec<(usize, Kind)> = on.picks.iter().map(|p| (p.index, p.kind)).collect();
        assert_eq!(ctx, vec![(0, Kind::Context), (1, Kind::Selected)]);
        let off = select(&doc, &e, &Params { repair: false, ..base });
        assert_eq!(off.picks.iter().map(|p| p.index).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn repair_stops_at_section_boundary() {
        let doc = parse(
            "## One\n\nThe committee approved a new budget for the city parks.\n\n## Two\n\nThis decision surprised many residents who expected cuts. Separately, the library extended its opening hours on weekends.\n",
            &ParseOptions::default(),
        );
        let e = vec![unit(&[1.0, 0.0, 0.0]), unit(&[1.0, 1.0, 1.0]), unit(&[0.0, 0.0, 1.0])];
        let p = Params { budget: 1, lambda: 1.0, dependence_penalty: 0.0, lead: 0.0, centrality: Centrality::Centroid, ..Default::default() };
        let sel = select(&doc, &e, &p);
        assert_eq!(selected(&sel), vec![1], "the dangling sentence must be the pick for this test to mean anything");
        assert!(sel.picks.iter().all(|p| p.kind == Kind::Selected), "context pulled across a heading: {:?}", sel.picks);
    }

    #[test]
    fn list_item_keeps_its_lead_in() {
        // Items long enough that the list is not folded into its lead-in.
        let item = "the party shall deliver all of the goods described in the relevant schedule on time and in full to the premises named by the other party in writing before the delivery date";
        let text = format!("The Supplier agrees to the following obligations:\n(a) {item};\n(b) {item};\n(c) {item};\n(d) {item}.\n");
        let doc = parse(&text, &ParseOptions::default());
        let lead = doc.sentences.iter().position(|s| s.lead_in.is_none()).unwrap();
        let item_idx: Vec<usize> = doc.sentences.iter().filter(|s| s.lead_in == Some(lead)).map(|s| s.index).collect();
        assert!(item_idx.len() >= 4, "{:#?}", doc.sentences);
        // The other sentences point along different axes and the last item sits
        // between them all, so it is the most central and gets picked.
        let last = *item_idx.last().unwrap();
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let e: Vec<Vec<f32>> = doc
            .sentences
            .iter()
            .filter(|s| s.candidate)
            .map(|s| if s.index == last { unit(&[1.0, 1.0, 1.0]) } else { unit(&axes[s.index % 3]) })
            .collect();
        let p = Params { budget: 1, lambda: 1.0, dependence_penalty: 0.0, lead: 0.0, centrality: Centrality::Centroid, ..Default::default() };
        let sel = select(&doc, &e, &p);
        assert_eq!(selected(&sel), vec![last]);
        assert!(sel.picks.iter().any(|p| p.index == lead && p.kind == Kind::Context), "{:?}", sel.picks);
    }

    #[test]
    fn word_cost_overrides_length() {
        // Two long sentences; compression makes the second cheap. With costs
        // supplied, a budget that fits only one uncompressed sentence fits both.
        let doc = parse(
            "The first sentence is long and has a great many words in it for this test. The second sentence is also long and has a great many words in it for this test.",
            &ParseOptions::default(),
        );
        let e = vec![unit(&[1.0, 0.0]), unit(&[0.0, 1.0])];
        let w = doc.sentences[0].words;
        let base = Params { word_budget: Some(w + 6), repair: false, ..Default::default() };
        assert_eq!(selected(&select(&doc, &e, &base)).len(), 1);
        let cheap = Params { word_cost: Some(vec![w, 6]), ..base };
        assert_eq!(selected(&select(&doc, &e, &cheap)), vec![0, 1]);
    }

    #[test]
    fn features_match_the_head_layout() {
        let doc = parse(include_str!("../samples/web-article.txt"), &ParseOptions::default());
        let e = embed(&doc);
        let sel = select(&doc, &e, &Params { want_features: true, ..Default::default() });
        assert_eq!(sel.features.len(), e.len());
        let cand: Vec<usize> = doc.sentences.iter().filter(|s| s.candidate).map(|s| s.index).collect();
        for ((si, f), (want, emb)) in sel.features.iter().zip(cand.iter().zip(&e)) {
            assert_eq!(si, want);
            assert_eq!(f.len(), crate::salience::HAND_FEATURES.len() + emb.len());
            assert_eq!(&f[crate::salience::HAND_FEATURES.len()..], emb.as_slice());
            assert!(f.iter().all(|x| x.is_finite()));
        }
        // Features are not computed unless asked for.
        assert!(select(&doc, &e, &Params::default()).features.is_empty());
    }

    #[test]
    fn salience_head_steers_selection() {
        // A head that only looks at "position" (feature 2) with negative weight
        // prefers the earliest sentence; with positive weight, the latest.
        let doc = parse(
            "Alpha beta gamma delta epsilon zeta eta theta. Completely different words appear in this one here. Yet another unrelated sentence about something else entirely.",
            &ParseOptions::default(),
        );
        let e = vec![unit(&[1.0, 0.0, 0.0]), unit(&[0.0, 1.0, 0.0]), unit(&[0.0, 0.0, 1.0])];
        let head = |w: f32| {
            let dim = crate::salience::HAND_FEATURES.len() + 3;
            let mut weights = vec![0.0; dim];
            weights[2] = w;
            let json = serde_json::json!({
                "encoder": {"hidden_size": 3, "num_hidden_layers": 1, "weights_bytes": 1},
                "features": crate::salience::HAND_FEATURES,
                "mean": vec![0.0; dim], "std": vec![1.0; dim], "weights": weights, "bias": 0.0,
            });
            std::sync::Arc::new(Head::parse(&json.to_string()).unwrap())
        };
        let p = |w: f32| Params { budget: 1, lead: 0.0, salience: Some(head(w)), salience_weight: 1.0, repair: false, ..Default::default() };
        assert_eq!(selected(&select(&doc, &e, &p(-5.0))), vec![0]);
        assert_eq!(selected(&select(&doc, &e, &p(5.0))), vec![2]);
    }
    mod props {
        //! Selection invariants on generated documents with random budgets and
        //! knobs: document order, no duplicates, only candidates, justified
        //! context, budgets held, same input same output.
        use super::*;
        use crate::testgen::document;
        use proptest::prelude::*;

        fn params() -> impl Strategy<Value = Params> {
            (
                0usize..12,
                prop::option::weighted(0.5, 0usize..300),
                0f32..=1.0,
                prop::bool::ANY,
                0f32..=1.0,
                0f32..=1.0,
                0f32..=1.0,
                0f32..=1.0,
                prop::bool::ANY,
            )
                .prop_map(|(budget, word_budget, lambda, centroid, backward, lead, section_weight, dependence_penalty, repair)| Params {
                    budget,
                    word_budget,
                    lambda,
                    centrality: if centroid { Centrality::Centroid } else { Centrality::Directed },
                    backward,
                    lead,
                    section_weight,
                    dependence_penalty,
                    repair,
                    ..Default::default()
                })
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(1000))]

            #[test]
            fn invariants(text in document(), p in params()) {
                let doc = parse(&text, &ParseOptions::default());
                let e = embed(&doc);
                let n = e.len();
                let sel = select(&doc, &e, &p);
                let what = format!(
                    "budget={} words={:?} lambda={} {:?} backward={} lead={} section={} penalty={} repair={}",
                    p.budget, p.word_budget, p.lambda, p.centrality, p.backward, p.lead, p.section_weight, p.dependence_penalty, p.repair
                );
                prop_assert!(sel.picks.windows(2).all(|w| w[0].index < w[1].index), "{}", what);
                let sel_idx = selected(&sel);
                if p.word_budget.is_none() {
                    prop_assert_eq!(sel_idx.len(), p.budget.min(n), "{}", what);
                } else if p.word_budget != Some(0) && n > 0 {
                    // Shorter than any sentence: still one sentence.
                    prop_assert!(!sel_idx.is_empty(), "{}", what);
                }
                let mut ranks: Vec<usize> = sel.picks.iter().filter(|p| p.kind == Kind::Selected).map(|p| p.rank).collect();
                ranks.sort();
                prop_assert_eq!(ranks, (0..sel_idx.len()).collect::<Vec<_>>(), "{}", what);
                for pk in &sel.picks {
                    prop_assert!(pk.group < sel.groups.len(), "{}", what);
                    prop_assert!(pk.score.is_finite(), "{}", what);
                    match pk.kind {
                        Kind::Selected => prop_assert!(doc.sentences[pk.index].candidate, "{}: non-candidate selected", what),
                        // Context comes before the pick it supports, in the same section or as its list's lead-in.
                        Kind::Context => prop_assert!(
                            sel_idx.iter().any(|&s| s > pk.index
                                && (s - pk.index <= 3 && doc.sentences[s].section == doc.sentences[pk.index].section
                                    || doc.sentences[s].lead_in.is_some())),
                            "{}: stray context sentence {}", what, pk.index
                        ),
                    }
                }
                if let Some(w) = p.word_budget {
                    let total: usize = sel.picks.iter().map(|p| doc.sentences[p.index].words).sum();
                    prop_assert!(total <= w || sel_idx.len() == 1, "{}: {} words", what, total);
                }
                let again = select(&doc, &e, &p);
                prop_assert_eq!(format!("{:?}", sel.picks), format!("{:?}", again.picks), "{}", what);
            }
        }
    }
}
