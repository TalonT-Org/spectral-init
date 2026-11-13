//! Minimal usage of spectral-init.
//!
//! Constructs a small synthetic ring graph, runs spectral initialization,
//! and prints the resulting 2-D coordinates.
//!
//! Run with: cargo run --example basic

use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{SpectralInitConfig, spectral_init};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a 10-node ring graph: every node is connected to its two neighbours.
    // spectral_init takes a view of a CsMatI<f32, u32, usize> — the type that
    // umap_rs::LearnedManifold::graph() returns.
    let n = 10usize;
    let mut tri: TriMatI<f32, u32> = TriMatI::new((n, n));
    for i in 0..n {
        let next = (i + 1) % n;
        tri.add_triplet(i, next, 1.0_f32);
        tri.add_triplet(next, i, 1.0_f32);
    }
    let graph: CsMatI<f32, u32, usize> = tri.to_csr();

    // Two embedding dimensions, seed 42. The default compute mode is
    // ComputeMode::PythonCompat, which follows umap-learn's spectral layout.
    let config = SpectralInitConfig::new(2).with_seed(42);
    let embedding = spectral_init(graph.view(), None, &config)?;

    println!("Embedding shape: {:?}", embedding.shape());
    println!("First few coordinates:");
    for i in 0..embedding.nrows().min(5) {
        println!(
            "  node {i}: [{:.4}, {:.4}]",
            embedding[[i, 0]],
            embedding[[i, 1]]
        );
    }

    assert_eq!(embedding.shape(), &[n, 2]);
    Ok(())
}
