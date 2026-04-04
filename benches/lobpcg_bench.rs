//! LOBPCG eigensolver microbenchmark — Criterion harness.
//!
//! Benchmarks `solve_eigenproblem_pub()` on pre-loaded CSR Laplacians, bypassing
//! `cargo nextest` overhead (process spawn ~50 s measured on the 100K MERFISH run).
//! This produces valid algorithm-level timing comparable to Python `eigsh`.
//!
//! # Run
//! ```sh
//! cargo bench --bench lobpcg_bench --features testing
//! ```
//!
//! HTML reports are written to `target/criterion/`.
//!
//! # Optional Large Fixtures
//! The 10K and 100K groups require Laplacian NPZ files in `temp/lobpcg_bench/`.
//! Generate them by exporting the normalized Laplacian for a MERFISH graph:
//!
//! ```python
//! # Python snippet — run after building the MERFISH kNN graph
//! import scipy.sparse as sp, numpy as np
//! lap = build_normalized_laplacian(graph)   # scipy CSR, float64
//! sp.save_npz("temp/lobpcg_bench/merfish_10k_laplacian.npz", lap)
//! ```
//!
//! If absent, those groups emit `[lobpcg_bench] SKIP ...` to stderr and return
//! immediately; the run still exits 0.

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray_npy::{NpzReader, ReadNpyError, ReadNpzError};
use spectral_init::solve_eigenproblem_pub;
use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

// ─── NPZ Laplacian Loader ─────────────────────────────────────────────────────

/// Load a `CsMatI<f64, usize>` from a scipy-format NPZ file.
///
/// Handles both i32 and i64 index/indptr arrays (scipy uses i32 on 32-bit
/// platforms and i64 on 64-bit platforms when the matrix is large).
///
/// Returns `None` when the file does not exist (optional fixtures).
/// Panics when the file exists but has wrong format (indicates a corrupt fixture).
fn load_laplacian(path: &Path) -> Option<sprs::CsMatI<f64, usize>> {
    if !path.exists() {
        return None;
    }

    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("cannot open laplacian {:?}: {}", path, e));
    let mut npz = NpzReader::new(file)
        .unwrap_or_else(|e| panic!("NpzReader failed on {:?}: {}", path, e));

    let data: Vec<f64> = npz
        .by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>("data")
        .unwrap_or_else(|e| panic!("'data' missing in {:?}: {}", path, e))
        .into_iter()
        .collect();

    let indices: Vec<usize> =
        match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indices") {
            Ok(arr) => arr.iter().map(|&x| x as usize).collect(),
            Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
                let arr: ndarray::Array1<i64> = npz
                    .by_name("indices")
                    .unwrap_or_else(|e| panic!("'indices' missing in {:?}: {}", path, e));
                arr.iter().map(|&x| x as usize).collect()
            }
            Err(e) => panic!("error reading 'indices' from {:?}: {}", path, e),
        };

    let indptr: Vec<usize> =
        match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indptr") {
            Ok(arr) => arr.iter().map(|&x| x as usize).collect(),
            Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
                let arr: ndarray::Array1<i64> = npz
                    .by_name("indptr")
                    .unwrap_or_else(|e| panic!("'indptr' missing in {:?}: {}", path, e));
                arr.iter().map(|&x| x as usize).collect()
            }
            Err(e) => panic!("error reading 'indptr' from {:?}: {}", path, e),
        };

    let shape: Vec<usize> =
        match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("shape") {
            Ok(arr) => arr.iter().map(|&x| x as usize).collect(),
            Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
                let arr: ndarray::Array1<i64> = npz
                    .by_name("shape")
                    .unwrap_or_else(|e| panic!("'shape' missing in {:?}: {}", path, e));
                arr.iter().map(|&x| x as usize).collect()
            }
            Err(e) => panic!("error reading 'shape' from {:?}: {}", path, e),
        };

    assert!(
        shape.len() >= 2,
        "'shape' in {:?} has {} element(s), expected 2 (must be a 2-D matrix)",
        path,
        shape.len()
    );
    let (rows, cols) = (shape[0], shape[1]);
    Some(
        sprs::CsMatI::try_new((rows, cols), indptr, indices, data)
            .unwrap_or_else(|e| panic!("CSR structure invalid in {:?}: {:?}", path, e)),
    )
}

// ─── blobs_5000 — 5,000 nodes, mandatory ─────────────────────────────────────

fn bench_lobpcg_blobs5000(c: &mut Criterion) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/blobs_5000/comp_b_laplacian.npz");

    let laplacian = load_laplacian(&path)
        .unwrap_or_else(|| panic!("mandatory fixture not found: {}", path.display()));

    let n = laplacian.rows();
    let seed = 42_u64;

    // One call outside the timed loop: primes instruction caches and surfaces
    // the solver level via the --features testing timing lines on stderr.
    let _ = solve_eigenproblem_pub(black_box(&laplacian), black_box(2), black_box(seed));
    eprintln!("[lobpcg_bench] blobs_5000 ready: n={n}");

    let mut group = c.benchmark_group("lobpcg_blobs5000");
    group.sample_size(10);            // Criterion minimum; solver takes ~300 ms
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("solve_eigenproblem", |b| {
        b.iter(|| {
            black_box(solve_eigenproblem_pub(
                black_box(&laplacian),
                black_box(2),
                black_box(seed),
            ))
        });
    });
    group.finish();
}

// ─── merfish_10k — 10,000 nodes, optional ────────────────────────────────────

fn bench_lobpcg_merfish_10k(c: &mut Criterion) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("temp/lobpcg_bench/merfish_10k_laplacian.npz");

    let Some(laplacian) = load_laplacian(&path) else {
        eprintln!(
            "[lobpcg_bench] SKIP merfish_10k: fixture not found at {}",
            path.display()
        );
        return;
    };

    let n = laplacian.rows();
    let seed = 42_u64;

    // One call outside the timed loop: primes instruction caches and surfaces
    // the solver level via the --features testing timing lines on stderr.
    let _ = solve_eigenproblem_pub(black_box(&laplacian), black_box(2), black_box(seed));
    eprintln!("[lobpcg_bench] merfish_10k ready: n={n}");

    let mut group = c.benchmark_group("lobpcg_merfish_10k");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(10));
    group.measurement_time(Duration::from_secs(300));

    group.bench_function("solve_eigenproblem", |b| {
        b.iter(|| {
            black_box(solve_eigenproblem_pub(
                black_box(&laplacian),
                black_box(2),
                black_box(seed),
            ))
        });
    });
    group.finish();
}

// ─── merfish_100k — 100,000 nodes, optional ──────────────────────────────────

fn bench_lobpcg_merfish_100k(c: &mut Criterion) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("temp/lobpcg_bench/merfish_100k_laplacian.npz");

    let Some(laplacian) = load_laplacian(&path) else {
        eprintln!(
            "[lobpcg_bench] SKIP merfish_100k: fixture not found at {}",
            path.display()
        );
        return;
    };

    let n = laplacian.rows();
    let seed = 42_u64;

    // One call outside the timed loop: primes instruction caches and surfaces
    // the solver level via the --features testing timing lines on stderr.
    let _ = solve_eigenproblem_pub(black_box(&laplacian), black_box(2), black_box(seed));
    eprintln!("[lobpcg_bench] merfish_100k ready: n={n}");

    let mut group = c.benchmark_group("lobpcg_merfish_100k");
    group.sample_size(10);
    // 100K LOBPCG may take 8–60 s per call; 600 s allows 10 samples at the high end.
    group.warm_up_time(Duration::from_secs(30));
    group.measurement_time(Duration::from_secs(600));

    group.bench_function("solve_eigenproblem", |b| {
        b.iter(|| {
            black_box(solve_eigenproblem_pub(
                black_box(&laplacian),
                black_box(2),
                black_box(seed),
            ))
        });
    });
    group.finish();
}

// ─── Registration ─────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_lobpcg_blobs5000,
    bench_lobpcg_merfish_10k,
    bench_lobpcg_merfish_100k,
);
criterion_main!(benches);
