//! SIMD SpMV ISA baseline — Criterion benchmark harness skeleton.
//!
//! Run with: cargo bench --features testing --bench simd_spmv_exp
//!
//! This skeleton is populated incrementally by subsequent experiment groups.
//! Group B adds the five real benchmark functions; this file exists only to
//! validate that the crate compiles and Criterion wiring is correct.

use criterion::{criterion_group, criterion_main, Criterion};
use spectral_init::operator::spmv_csr;
use std::hint::black_box;

// ─── Stub helpers ────────────────────────────────────────────────────────────

/// Placeholder: returns a symmetric normalized Laplacian for an n-node ring
/// graph. Populated by groupB.
#[allow(dead_code)]
fn make_ring_lap(n: usize) -> sprs::CsMatI<f64, usize> {
    let _ = (n, spmv_csr as usize); // silence unused import warning
    unimplemented!("make_ring_lap: populated by groupB")
}

// ─── Placeholder benchmark ───────────────────────────────────────────────────

/// No-op placeholder so the harness compiles and links.
/// Replaced by real benchmark groups in groupB.
fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| black_box(())));
}

// ─── Registration ────────────────────────────────────────────────────────────

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
