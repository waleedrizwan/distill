//! distill as a library: the pipeline's stages as modules, so the binary,
//! the unit tests and the fuzz targets (`fuzz/`) share one implementation.
//! `main.rs` is the command-line front end.

pub mod compress;
pub mod embed;
pub mod lexical;
pub mod salience;
pub mod segment;
pub mod select;
pub mod split;

#[cfg(test)]
pub(crate) mod testgen;
