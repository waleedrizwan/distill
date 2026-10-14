//! A learned salience score: logistic regression over the frozen sentence
//! embedding plus a few position, length and dependence features.
//!
//! The weights are trained offline by `eval/train_salience.py` on oracle
//! extractive labels (CNN/DailyMail and BillSum training splits) and compiled
//! into the binary from `src/salience.json`. Scoring is one dot product per
//! sentence in a fixed order, so it is as deterministic as the rest of the
//! ranking. The weights only fit the encoder they were trained with; with any
//! other model the head is not used.

use serde::{Deserialize, Serialize};

/// Features computed by `select.rs`, in this order, followed by the embedding.
pub const HAND_FEATURES: &[&str] = &[
    "doc_centrality",
    "section_centrality",
    "position",
    "log_position",
    "section_position",
    "section_first",
    "log_words",
    "dangling",
    "points_elsewhere",
    "lead_in",
    "list_item",
    "summary_section",
];

/// Identifies the encoder the weights were trained on.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Encoder {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// Size of model.safetensors in bytes.
    pub weights_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Head {
    pub encoder: Encoder,
    /// Names of the hand features, which must match `HAND_FEATURES`.
    pub features: Vec<String>,
    /// Standardisation: x' = (x - mean) / std, per feature (hand features, then embedding).
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Head {
    /// The head compiled into this binary, if it is present and well formed.
    pub fn builtin() -> Option<Head> {
        Self::parse(include_str!("salience.json"))
    }

    pub fn parse(json: &str) -> Option<Head> {
        let h: Head = serde_json::from_str(json).ok()?;
        let dim = HAND_FEATURES.len() + h.encoder.hidden_size;
        let names_ok = h.features.iter().map(String::as_str).eq(HAND_FEATURES.iter().copied());
        let lens_ok = h.mean.len() == dim && h.std.len() == dim && h.weights.len() == dim;
        (names_ok && lens_ok && h.std.iter().all(|&s| s > 0.0)).then_some(h)
    }

    /// Log-odds that the sentence belongs in a summary.
    pub fn logit(&self, x: &[f32]) -> f32 {
        debug_assert_eq!(x.len(), self.weights.len());
        let mut z = self.bias;
        for (((&xi, &w), &m), &sd) in x.iter().zip(&self.weights).zip(&self.mean).zip(&self.std) {
            z += w * (xi - m) / sd;
        }
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(hidden: usize) -> String {
        let dim = HAND_FEATURES.len() + hidden;
        let names: Vec<String> = HAND_FEATURES.iter().map(|s| format!("{s:?}")).collect();
        let ones = vec!["1"; dim].join(",");
        let zeros = vec!["0"; dim].join(",");
        let mut w = vec!["0"; dim];
        w[0] = "2";
        format!(
            r#"{{"encoder": {{"hidden_size": {hidden}, "num_hidden_layers": 6, "weights_bytes": 1}},
                "features": [{}], "mean": [{zeros}], "std": [{ones}], "weights": [{}], "bias": -1}}"#,
            names.join(","),
            w.join(",")
        )
    }

    #[test]
    fn builtin_head_matches_default_model() {
        // The weights in src/salience.json were trained on all-MiniLM-L6-v2 as
        // pinned in fetch-model.sh; retrain them if that model changes.
        let h = Head::builtin().expect("src/salience.json is missing or malformed");
        assert_eq!(h.encoder, Encoder { hidden_size: 384, num_hidden_layers: 6, weights_bytes: 90_868_376 });
        assert!(h.weights.iter().chain(&h.mean).chain(&h.std).all(|x| x.is_finite()));
    }

    #[test]
    fn parse_and_score() {
        let h = Head::parse(&tiny(3)).expect("valid head");
        let mut x = vec![0.0; HAND_FEATURES.len() + 3];
        x[0] = 1.5;
        assert!((h.logit(&x) - 2.0).abs() < 1e-6);
        // Wrong embedding size for the stated encoder: rejected.
        assert!(Head::parse(&tiny(3).replace("\"hidden_size\": 3", "\"hidden_size\": 4")).is_none());
    }
}
