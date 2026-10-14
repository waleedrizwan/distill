//! Sentence embeddings with a small BERT-family model (all-MiniLM-L6-v2).
//!
//! The model is only used to turn each sentence into a vector for ranking.
//! It never produces text, so it cannot add anything to the summary.

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::path::Path;
use tokenizers::{Tokenizer, TruncationParams};

/// The sentence-transformers default for MiniLM; longer sentences are truncated.
const MAX_TOKENS: usize = 256;

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn load(dir: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let config_path = dir.join("config.json");
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )
        .context("parsing config.json")?;

        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams { max_length: MAX_TOKENS, ..Default::default() }))
            .map_err(|e| anyhow!("configuring tokenizer: {e}"))?;
        tokenizer.with_padding(None);

        let weights = dir.join("model.safetensors");
        // Safety: the file is memory-mapped read-only and must not change while we run.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device) }
            .context("loading model.safetensors")?;
        let model = BertModel::load(vb, &config).context("building the model")?;
        Ok(Self { model, tokenizer, device })
    }

    /// Returns one L2-normalised vector per input text, in input order.
    pub fn embed(&self, texts: &[&str], batch_size: usize) -> Result<Vec<Vec<f32>>> {
        let mut encoded: Vec<(usize, Vec<u32>)> = Vec::with_capacity(texts.len());
        for (i, t) in texts.iter().enumerate() {
            let enc = self
                .tokenizer
                .encode(*t, true)
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

            // Mean pooling over real (non-padding) tokens.
            let m = attention.to_dtype(DType::F32)?.unsqueeze(2)?;
            let summed = hidden.broadcast_mul(&m)?.sum(1)?;
            let counts = m.sum(1)?;
            let pooled = summed.broadcast_div(&counts)?;
            let rows: Vec<Vec<f32>> = pooled.to_vec2()?;
            for ((orig, _), mut v) in chunk.iter().zip(rows) {
                normalize(&mut v);
                out[*orig] = v;
            }
        }
        Ok(out)
    }
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
