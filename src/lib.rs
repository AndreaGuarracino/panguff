//! Panguff - A toolkit for pangenome graph analysis
//!
//! This library provides utilities for working with pangenome graphs in GFA format,
//! including finding nearest reference nodes for each node in the graph,
//! and computing pairwise path/sample similarity.

pub mod nearest;
pub mod similarity;

// Re-export key types at crate level for convenience
pub use nearest::{create_reader, find_nearest, get_ref_nodes, get_ref_positions, load_ref_paths, Graph};
pub use similarity::{compute_similarity, group_paths, write_output, GroupInfo, SimilarityIndex, SimilarityResult};
