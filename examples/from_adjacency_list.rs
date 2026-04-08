//! Build a CsMatI<f32, u32, usize> from an adjacency list and run spectral_init.
//!
//! This example answers the most common integration question: "how do I construct
//! the graph input if I'm not using umap-rs?"
//!
//! Run with: cargo run --example from_adjacency_list

use spectral_init::{SpectralInitConfig, spectral_init};
use sprs::{CsMatI, TriMatI};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Represent an undirected weighted graph as (src, dst, weight) triplets.
    // These are directed edges; we will symmetrize them below.
    let edges: &[(usize, usize, f32)] = &[
        (0, 1, 0.8),
        (0, 2, 0.5),
        (1, 3, 0.9),
        (2, 3, 0.7),
        (3, 4, 0.6),
        (1, 4, 0.4),
        (4, 5, 0.8),
        (2, 5, 0.3),
        (5, 6, 0.7),
    ];
    let n = 7usize;

    // spectral_init requires a symmetric adjacency matrix.
    // Add both (i, j) and (j, i) for each edge.
    let mut tri: TriMatI<f32, u32> = TriMatI::new((n, n));
    for &(i, j, w) in edges {
        tri.add_triplet(i, j, w);
        tri.add_triplet(j, i, w);
    }
    let graph: CsMatI<f32, u32, usize> = tri.to_csr();

    let config = SpectralInitConfig::default();
    let embedding = spectral_init(&graph, 2, 42, None, config)?;

    println!(
        "Nodes: {}, Embedding dims: {}",
        embedding.nrows(),
        embedding.ncols()
    );
    println!("Coordinates:");
    for (i, row) in embedding.outer_iter().enumerate() {
        println!("  node {i}: [{:.4}, {:.4}]", row[0], row[1]);
    }

    assert_eq!(embedding.shape(), &[n, 2]);
    Ok(())
}
