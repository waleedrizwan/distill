//! Sentence embeddings with a small BERT-family model (all-MiniLM-L6-v2 by default).
//!
//! The model is only used to turn each sentence into a vector for ranking.
//! It never produces text, so it cannot add anything to the summary.
//!
//! How a model must be used (pooling, token limit, text prefix) is read from
//! `distill.json` in the model folder, written by `fetch-model.sh`. Without it,
//! the all-MiniLM-L6-v2 settings apply.

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use serde::Deserialize;
use std::path::Path;
use tokenizers::{Tokenizer, TruncationParams};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    /// Average of the token vectors (padding excluded).
    Mean,
    /// The vector of the first ([CLS]) token.
    Cls,
}

/// Contents of `distill.json`. The defaults are all-MiniLM-L6-v2's.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub pooling: Pooling,
    /// Longer sentences are truncated to this many word-pieces.
    pub max_tokens: usize,
    /// Prepended to every sentence before tokenizing (e5 expects "query: ").
    pub prefix: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self { pooling: Pooling::Mean, max_tokens: 256, prefix: String::new() }
    }
}

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    settings: Settings,
    fingerprint: u64,
    encoder: crate::salience::Encoder,
}

impl Embedder {
    pub fn load(dir: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let config_path = dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let config: Config = serde_json::from_str(&config_text).context("parsing config.json")?;
        let raw: serde_json::Value = serde_json::from_str(&config_text)?;
        let int = |k: &str| raw.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let positions = match int("max_position_embeddings") {
            0 => 512,
            p => p,
        };

        let settings_path = dir.join("distill.json");
        let settings_text = match std::fs::read_to_string(&settings_path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", settings_path.display())),
        };
        let mut settings: Settings = if settings_text.trim().is_empty() {
            Settings::default()
        } else {
            serde_json::from_str(&settings_text).context("parsing distill.json")?
        };
        settings.max_tokens = settings.max_tokens.clamp(8, positions);

        let weights = dir.join("model.safetensors");
        let weights_len = std::fs::metadata(&weights).context("reading model.safetensors")?.len();
        // Identifies the model for the embedding cache: same config, settings and weights size.
        let fingerprint = fnv1a(
            [config_text.as_bytes(), b"\0", settings_text.as_bytes(), b"\0", &weights_len.to_le_bytes()].concat().as_slice(),
        );

        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams { max_length: settings.max_tokens, ..Default::default() }))
            .map_err(|e| anyhow!("configuring tokenizer: {e}"))?;
        tokenizer.with_padding(None);

        // Safety: the file is memory-mapped read-only and must not change while we run.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device) }
            .context("loading model.safetensors")?;
        let model = BertModel::load(vb, &config).context("building the model")?;
        let encoder = crate::salience::Encoder {
            hidden_size: int("hidden_size"),
            num_hidden_layers: int("num_hidden_layers"),
            weights_bytes: weights_len,
        };
        Ok(Self { model, tokenizer, device, settings, fingerprint, encoder })
    }

    /// Changes whenever the model, its settings or its weights file change.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// What a salience head must have been trained on to be used with this model.
    pub fn encoder(&self) -> crate::salience::Encoder {
        self.encoder.clone()
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Returns one L2-normalised vector per input text, in input order.
    pub fn embed(&self, texts: &[&str], batch_size: usize) -> Result<Vec<Vec<f32>>> {
        let mut encoded: Vec<(usize, Vec<u32>)> = Vec::with_capacity(texts.len());
        for (i, t) in texts.iter().enumerate() {
            let input = if self.settings.prefix.is_empty() {
                std::borrow::Cow::Borrowed(*t)
            } else {
                std::borrow::Cow::Owned(format!("{}{t}", self.settings.prefix))
            };
            let enc = self
                .tokenizer
                .encode(input.as_ref(), true)
                .map_err(|e| anyhow!("tokenizing: {e}"))?;
            encoded.push((i, enc.get_ids().to_vec()));
        }
        // Batch sentences of similar length together so little time goes on padding.
        // The sort is stable, so the batches are the same on every run.
        encoded.sort_by_key(|(i, ids)| (ids.len(), *i));

        let mut out: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        for chunk in encoded.chunks(batch_size.max(1)) {
            let max_len = chunk.iter().map(|(_, ids)| ids.len()).max().unwrap_or(1);
            let b = chunk.len();
            let mut ids = vec![0u32; b * max_len];
            let mut mask = vec![0u32; b * max_len];
            for (row, (_, toks)) in chunk.iter().enumerate() {
                for (col, &t) in toks.iter().enumerate() {
                    ids[row * max_len + col] = t;
                    mask[row * max_len + col] = 1;
                }
            }
            let input_ids = Tensor::from_vec(ids, (b, max_len), &self.device)?;
            let attention = Tensor::from_vec(mask, (b, max_len), &self.device)?;
            let token_types = input_ids.zeros_like()?;
            let hidden = self.model.forward(&input_ids, &token_types, Some(&attention))?;

            let pooled = match self.settings.pooling {
                // Average over real (non-padding) tokens.
                Pooling::Mean => {
                    let m = attention.to_dtype(DType::F32)?.unsqueeze(2)?;
                    let summed = hidden.broadcast_mul(&m)?.sum(1)?;
                    let counts = m.sum(1)?;
                    summed.broadcast_div(&counts)?
                }
                Pooling::Cls => hidden.i((.., 0, ..))?.contiguous()?,
            };
            let rows: Vec<Vec<f32>> = pooled.to_vec2()?;
            for ((orig, _), mut v) in chunk.iter().zip(rows) {
                normalize(&mut v);
                out[*orig] = v;
            }
        }
        Ok(out)
    }
}

/// FNV-1a, 64-bit: a fixed, dependency-free hash for cache keys.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_defaults_and_overrides() {
        let d = Settings::default();
        assert_eq!((d.pooling, d.max_tokens, d.prefix.as_str()), (Pooling::Mean, 256, ""));
        let s: Settings = serde_json::from_str(r#"{"pooling": "cls"}"#).unwrap();
        assert_eq!((s.pooling, s.max_tokens), (Pooling::Cls, 256));
        let s: Settings = serde_json::from_str(r#"{"pooling": "mean", "max_tokens": 512, "prefix": "query: "}"#).unwrap();
        assert_eq!((s.max_tokens, s.prefix.as_str()), (512, "query: "));
        assert!(serde_json::from_str::<Settings>(r#"{"pooling": "max"}"#).is_err());
    }
}
