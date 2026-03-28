//! Criterion benchmark baselines for spectral-init pipeline components.
//!
//! Run with: cargo bench --features testing
//!
//! Inputs: deterministic synthetic ring graphs — no fixture files required.
//! Each benchmark uses `black_box` to prevent dead-code elimination.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use spectral_init::{
    build_normalized_laplacian, compute_degrees, dense_evd, find_components, lobpcg_solve,
    rsvd_solve, spectral_init, ComputeMode, SpectralInitConfig,
};
use spectral_init::operator::{spmv_csr, CsrOperator};
use std::hint::black_box;

// ─── Synthetic Graph Helpers ──────────────────────────────────────────────────

/// Build a connected ring graph with `n` nodes where each node has `2*half`
/// undirected neighbours (half nodes to the left, half to the right, modulo n).
///
/// Returns `CsMatI<f32, u32, usize>` — the type expected by `spectral_init`.
/// Requires n >= 2*half + 1 to avoid duplicate column indices.
fn make_ring_graph(n: usize, half: usize) -> sprs::CsMatI<f32, u32, usize> {
    let nnz_per_row = 2 * half;
    let mut indptr = vec![0usize; n + 1];
    let mut indices: Vec<u32> = Vec::with_capacity(n * nnz_per_row);
    let mut data: Vec<f32> = Vec::with_capacity(n * nnz_per_row);

    for i in 0..n {
        // Collect all neighbors at distances 1..=half in both directions.
        let mut row_cols: Vec<u32> = (1..=half)
            .flat_map(|d| {
                let left = (i + n - d) % n;
                let right = (i + d) % n;
                [left as u32, right as u32]
            })
            .collect();
        row_cols.sort_unstable();
        row_cols.dedup();
        let nnz = row_cols.len();
        indices.extend_from_slice(&row_cols);
        data.extend(std::iter::repeat(1.0_f32).take(nnz));
        indptr[i + 1] = indptr[i] + nnz;
    }

    sprs::CsMatI::try_new((n, n), indptr, indices, data)
        .expect("ring graph CSR construction failed")
}

/// Build the symmetric normalized Laplacian for an `n`-node ring graph with
/// `2*half` neighbours per node. Used to provide realistic sparse-matrix inputs
/// to SpMV, dense EVD, LOBPCG, and rSVD benchmarks.
fn make_laplacian(n: usize, half: usize) -> sprs::CsMatI<f64, usize> {
    let graph = make_ring_graph(n, half);
    let (_deg, sqrt_deg) = compute_degrees(&graph, ComputeMode::PythonCompat);
    let inv_sqrt_deg: Vec<f64> = sqrt_deg
        .iter()
        .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
        .collect();
    build_normalized_laplacian(&graph, &inv_sqrt_deg)
}

// ─── Benchmark Groups ─────────────────────────────────────────────────────────

/// SpMV: single-vector CSR sparse-matrix × dense-vector product.
/// This is the Phase 3 SIMD replacement target.
fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_csr");
    for &n in &[200_usize, 2000, 5000] {
        let lap = make_laplacian(n, 2); // 4 neighbours per node
        let indptr: Vec<usize> = lap.indptr().raw_storage().to_vec();
        let indices: Vec<usize> = lap.indices().to_vec();
        let data: Vec<f64> = lap.data().to_vec();
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

/// Dense EVD: exact eigendecomposition via faer. Used for n < 2000 in production.
fn bench_dense_evd(c: &mut Criterion) {
    let lap = make_laplacian(200, 2);
    c.bench_function("dense_evd_200", |b| {
        b.iter(|| black_box(dense_evd(black_box(&lap), black_box(3))))
    });
}

/// LOBPCG: iterative eigensolver for large n. Production Level 1.
fn bench_lobpcg(c: &mut Criterion) {
    let graph = make_ring_graph(2000, 2);
    let (_, sqrt_deg) = compute_degrees(&graph, ComputeMode::PythonCompat);
    let lap = make_laplacian(2000, 2);
    let op = CsrOperator(&lap);
    c.bench_function("lobpcg_2000", |b| {
        b.iter(|| {
            black_box(lobpcg_solve(
                black_box(&op),
                black_box(3),
                black_box(42_u64),
                black_box(false),
                black_box(&sqrt_deg),
            ))
        })
    });
}

/// Randomized SVD: 2I-L trick solver. Production Level 3.
fn bench_rsvd(c: &mut Criterion) {
    let lap = make_laplacian(2000, 2);
    c.bench_function("rsvd_2000", |b| {
        b.iter(|| black_box(rsvd_solve(black_box(&lap), black_box(3), black_box(42_u64))))
    });
}

/// Laplacian construction: degrees + D^{-1/2} W D^{-1/2} build.
fn bench_laplacian_build(c: &mut Criterion) {
    let graph = make_ring_graph(2000, 2);
    let (_deg, sqrt_deg) = compute_degrees(&graph, ComputeMode::PythonCompat);
    let inv_sqrt_deg: Vec<f64> = sqrt_deg
        .iter()
        .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
        .collect();
    c.bench_function("laplacian_build_2000", |b| {
        b.iter(|| black_box(build_normalized_laplacian(black_box(&graph), black_box(&inv_sqrt_deg))))
    });
}

/// BFS connected-components labelling.
fn bench_components(c: &mut Criterion) {
    let graph = make_ring_graph(2000, 2);
    c.bench_function("components_bfs_2000", |b| {
        b.iter(|| black_box(find_components(black_box(&graph))))
    });
}

/// Full end-to-end pipeline: spectral_init() on a connected graph.
/// n=200 exercises dense EVD path; n=2000 exercises LOBPCG path.
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    for &n in &[200_usize, 2000] {
        let graph = make_ring_graph(n, 2);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    spectral_init(
                        black_box(&graph),
                        black_box(2),
                        black_box(42_u64),
                        black_box(None),
                        SpectralInitConfig::default(),
                    )
                    .expect("bench: spectral_init failed on synthetic ring graph"),
                )
            });
        });
    }
    group.finish();
}

// ─── Registration ─────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_spmv,
    bench_dense_evd,
    bench_lobpcg,
    bench_rsvd,
    bench_laplacian_build,
    bench_components,
    bench_full_pipeline,
);
criterion_main!(benches);
