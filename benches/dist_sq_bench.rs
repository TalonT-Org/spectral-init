//! Criterion microbenchmark harness for `dist_sq_*` kernel variants.
//!
//! Benchmarks the squared-Euclidean-distance kernels at d_x=50 and d_x=10.
//!
//! # groupD integration note
//! When groupD makes the SIMD kernels `pub(crate)` in `src/metrics.rs`, replace
//! the stub functions below with:
//!
//!     use spectral_init::metrics_internal::{dist_sq_avx2_looped, dist_sq_avx512_looped};
//!
//! and remove the stub definitions.
//!
//! Run with: cargo bench --bench dist_sq_bench

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// ─── Stubs (replaced by pub(crate) imports after groupD) ─────────────────────

/// Scalar fallback — mirrors the loop structure that avx2_looped will use.
/// Benchmarks this until the real looped AVX2 kernel is exposed.
fn dist_sq_avx2_looped_stub(xi: &[f64], xj: &[f64]) -> f64 {
    xi.iter().zip(xj.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
}

/// Scalar fallback — mirrors the loop structure that avx512_looped will use.
/// Benchmarks this until the real looped AVX-512 kernel is exposed.
fn dist_sq_avx512_looped_stub(xi: &[f64], xj: &[f64]) -> f64 {
    xi.iter().zip(xj.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
}

// ─── Data setup ───────────────────────────────────────────────────────────────

fn make_vectors(d: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut xi = Vec::with_capacity(d);
    let mut xj = Vec::with_capacity(d);
    for i in 0..d {
        let mut h = DefaultHasher::new();
        (seed, i as u64).hash(&mut h);
        let bits = h.finish();
        // Map to [-1, 1] deterministically
        xi.push((bits as f64) / (u64::MAX as f64) * 2.0 - 1.0);
        (seed.wrapping_add(1), i as u64).hash(&mut h);
        let bits2 = h.finish();
        xj.push((bits2 as f64) / (u64::MAX as f64) * 2.0 - 1.0);
    }
    (xi, xj)
}

// ─── Benchmark group ──────────────────────────────────────────────────────────

fn bench_dist_sq_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("dist_sq_kernels");

    for &d in &[10_usize, 50] {
        let (xi, xj) = make_vectors(d, 42);

        group.bench_with_input(
            BenchmarkId::new("avx2_looped", d),
            &d,
            |b, _| b.iter(|| dist_sq_avx2_looped_stub(black_box(&xi), black_box(&xj))),
        );

        group.bench_with_input(
            BenchmarkId::new("avx512_looped", d),
            &d,
            |b, _| b.iter(|| dist_sq_avx512_looped_stub(black_box(&xi), black_box(&xj))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dist_sq_kernels);
criterion_main!(benches);
