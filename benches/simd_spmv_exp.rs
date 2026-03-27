//! SIMD SpMV ISA baseline — Criterion benchmark harness (groupB: CSR scaling).
//!
//! Run with: cargo bench --features testing --bench simd_spmv_exp

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use spectral_init::operator::spmv_csr;
use std::hint::black_box;

// ─── Ring Laplacian builder ───────────────────────────────────────────────────

/// Build the symmetric normalized Laplacian for an `n`-node ring graph with
/// `half` neighbours on each side (2×half off-diagonal NNZ per row).
///
/// Returns raw CSR slices `(indptr, indices, data)` in f64.
/// - Diagonal entries: 1.0 (identity term of L = I - D^{-1/2} W D^{-1/2}).
/// - Off-diagonal entries: -1/(2*half) (uniform degree d = 2*half).
///
/// Precondition: n >= 2*half + 1 (no column-index collisions on wrap-around).
fn make_ring_lap(n: usize, half: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let d = (2 * half) as f64;
    let off_val = -1.0 / d;
    let nnz_per_row = 2 * half + 1; // half left + 1 diagonal + half right

    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(n * nnz_per_row);
    let mut data = Vec::with_capacity(n * nnz_per_row);

    indptr.push(0usize);

    for i in 0..n {
        // Collect neighbour column indices (left and right), then add diagonal.
        let mut row_cols: Vec<usize> = (1..=half)
            .flat_map(|k| [(i + n - k) % n, (i + k) % n])
            .collect();
        row_cols.push(i);
        row_cols.sort_unstable();

        for &j in &row_cols {
            indices.push(j);
            data.push(if j == i { 1.0_f64 } else { off_val });
        }
        indptr.push(indptr.last().copied().unwrap() + nnz_per_row);
    }

    (indptr, indices, data)
}

// ─── Benchmark group ──────────────────────────────────────────────────────────

fn bench_spmv_csr_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_csr_scaling");
    for &n in &[200_usize, 2000, 5000, 10000, 50000] {
        let (indptr, indices, data) = make_ring_lap(n, 7);
        let x = vec![1.0_f64 / (n as f64).sqrt(); n];
        let mut y = vec![0.0_f64; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                spmv_csr(
                    black_box(&indptr),
                    black_box(&indices),
                    black_box(&data),
                    black_box(&x),
                    black_box(&mut y),
                )
            });
        });
    }
    group.finish();
}

// ─── Registration ─────────────────────────────────────────────────────────────

criterion_group!(benches, bench_spmv_csr_scaling);
criterion_main!(benches);
