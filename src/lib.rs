//! distill as a library: `distill::distill(text)` returns a plain-language
//! summary of `text`. The pipeline's stages are modules, so the binary, the
//! unit tests and the fuzz targets (`fuzz/`) share one implementation.
//! `main.rs` is the command-line front end.

pub mod compress;
pub mod embed;
pub mod glossary;
pub mod lexical;
pub mod pipeline;
pub mod readability;
pub mod salience;
pub mod segment;
pub mod select;
pub mod split;

pub use pipeline::{distill, Distiller, Length, Options, Summary};

#[cfg(test)]
pub(crate) mod testgen;
