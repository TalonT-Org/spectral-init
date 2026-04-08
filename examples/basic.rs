//! Minimal usage of spectral-init.
//!
//! Constructs a small synthetic ring graph, runs spectral initialization,
//! and prints the resulting 2-D coordinates.
//!
//! Run with: cargo run --example basic

use spectral_init::{SpectralInitConfig, spectral_init};
use sprs::{CsMatI, TriMatI};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a 10-node ring graph: every node is connected to its two neighbours.
    // spectral_init expects CsMatI<f32, u32, usize> — the same type as umap_rs::Umap::graph().
    let n = 10usize;
    let mut tri: TriMatI<f32, u32> = TriMatI::new((n, n));
    for i in 0..n {
        let next = (i + 1) % n;
        tri.add_triplet(i, next, 1.0_f32);
        tri.add_triplet(next, i, 1.0_f32);
    }
    let graph: CsMatI<f32, u32, usize> = tri.to_csr();

    // SpectralInitConfig::default() uses ComputeMode::PythonCompat, which matches
    // Python UMAP's initialization behaviour exactly.
    let config = SpectralInitConfig::default();
    let embedding = spectral_init(&graph, 2, 42, None, config)?;

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
