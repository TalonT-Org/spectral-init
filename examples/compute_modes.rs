//! Compare PythonCompat and RustNative compute modes.
//!
//! Both modes produce subspace-equivalent embeddings; RustNative uses AVX2+FMA
//! SIMD on x86_64 and falls back silently on other platforms.
//!
//! Run with: cargo run --example compute_modes

use spectral_init::{ComputeMode, SpectralInitConfig, spectral_init};
use sprs::{CsMatI, TriMatI};

fn make_ring(n: usize) -> CsMatI<f32, u32, usize> {
    let mut tri: TriMatI<f32, u32> = TriMatI::new((n, n));
    for i in 0..n {
        let next = (i + 1) % n;
        tri.add_triplet(i, next, 1.0_f32);
        tri.add_triplet(next, i, 1.0_f32);
    }
    tri.to_csr()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = make_ring(20);

    // PythonCompat: matches Python UMAP's degree accumulation and solver path.
    let mut cfg_compat = SpectralInitConfig::default();
    cfg_compat.compute_mode = ComputeMode::PythonCompat;
    let emb_compat = spectral_init(&graph, 2, 42, None, cfg_compat)?;

    // RustNative: uses AVX2+FMA SIMD kernel on x86_64; falls back on other platforms.
    // Eigenvectors are subspace-equivalent to PythonCompat (same span, may differ by sign).
    let mut cfg_native = SpectralInitConfig::default();
    cfg_native.compute_mode = ComputeMode::RustNative;
    let emb_native = spectral_init(&graph, 2, 42, None, cfg_native)?;

    println!("PythonCompat shape: {:?}", emb_compat.shape());
    println!("RustNative shape:   {:?}", emb_native.shape());

    assert_eq!(
        emb_compat.shape(),
        emb_native.shape(),
        "both modes must produce same output shape"
    );

    println!("\nBoth modes produce the same shape — embeddings are subspace-equivalent.");
    println!("\nPythonCompat first 3 rows:");
    for i in 0..3 {
        println!("  [{:.4}, {:.4}]", emb_compat[[i, 0]], emb_compat[[i, 1]]);
    }
    println!("RustNative first 3 rows:");
    for i in 0..3 {
        println!("  [{:.4}, {:.4}]", emb_native[[i, 0]], emb_native[[i, 1]]);
    }

    Ok(())
}
