//! RQ3: Threshold override validation — dense EVD vs LOBPCG subspace agreement.
//!
//! This test requires `--features testing` and `--test-threads=1` (env var mutation).

#[path = "../common/mod.rs"]
mod common;

use ndarray::Array2;
use ndarray_npy::write_npy;
use std::path::Path;

/// Compute the maximum principal angle between two column subspaces.
/// Returns arccos(min singular value of A^T * B) in radians.
fn max_subspace_angle(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    // M = A^T * B  (k x k matrix, k=3 in our case)
    let m = a.t().dot(b);
    // Compute singular values via eigenvalues of M^T * M
    let mtm = m.t().dot(&m);
    let k = mtm.nrows();
    let faer_mat = faer::Mat::<f64>::from_fn(k, k, |i, j| mtm[[i, j]]);
    let evd = faer_mat
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("eigendecomposition of small k×k matrix");
    let s = evd.S();
    // Singular values of M are sqrt of eigenvalues of M^T*M
    let min_sv_sq = (0..k)
        .map(|i| s.column_vector().iter().nth(i).copied().unwrap_or(0.0).max(0.0))
        .fold(f64::INFINITY, f64::min);
    let min_sv = min_sv_sq.sqrt().min(1.0); // clamp to [0, 1] for arccos
    min_sv.acos()
}

#[test]
fn rq3_threshold_override_subspace_agreement() {
    let n = 200;
    let n_components = 2; // solve_eigenproblem returns n_components+1 = 3 eigenpairs

    // Build 200-node ring Laplacian
    let laplacian = common::ring_laplacian(n);

    // ── Run 1: Force dense EVD (threshold=2000, n=200 < 2000 → Level 0) ──
    // SAFETY: single-threaded test (--test-threads=1), no concurrent env readers
    unsafe { std::env::set_var("SPECTRAL_DENSE_N_THRESHOLD", "2000"); }
    let ((eigs_dense, vecs_dense), level_dense) =
        spectral_init::solve_eigenproblem_pub(&laplacian, n_components, 42);
    assert_eq!(level_dense, 0, "expected Level 0 (dense EVD) with threshold=2000");

    // ── Run 2: Force LOBPCG (threshold=50, n=200 >= 50 → Level 1+) ──
    unsafe { std::env::set_var("SPECTRAL_DENSE_N_THRESHOLD", "50"); }
    let ((eigs_lobpcg, vecs_lobpcg), level_lobpcg) =
        spectral_init::solve_eigenproblem_pub(&laplacian, n_components, 42);
    assert!(level_lobpcg >= 1, "expected Level >= 1 (LOBPCG path) with threshold=50, got {level_lobpcg}");

    // ── Cleanup env var ──
    unsafe { std::env::remove_var("SPECTRAL_DENSE_N_THRESHOLD"); }

    // ── Save eigenvector matrices to .npy ──
    let results_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("research/2026-03-25-ci-test-optimization/results/rq3_threshold");
    std::fs::create_dir_all(&results_dir).expect("create results dir");

    write_npy(results_dir.join("dense_n200_eigvecs.npy"), &vecs_dense)
        .expect("write dense eigvecs");
    write_npy(results_dir.join("lobpcg_n200_eigvecs.npy"), &vecs_lobpcg)
        .expect("write lobpcg eigvecs");

    // ── Subspace angle ──
    let angle = max_subspace_angle(&vecs_dense, &vecs_lobpcg);
    println!("subspace_angle_rad={angle:.6e}");
    println!("dense_eigenvalues={eigs_dense:?}");
    println!("lobpcg_eigenvalues={eigs_lobpcg:?}");
    println!("dense_level={level_dense}, lobpcg_level={level_lobpcg}");

    // Write result to text file
    let result_text = format!(
        "subspace_angle_rad={angle:.6e}\ndense_level={level_dense}\nlobpcg_level={level_lobpcg}\n"
    );
    std::fs::write(results_dir.join("subspace_angle.txt"), &result_text)
        .expect("write subspace_angle.txt");

    // ── Verify .npy files were written (T7) ──
    assert!(results_dir.join("dense_n200_eigvecs.npy").exists(), "dense_n200_eigvecs.npy not written");
    assert!(results_dir.join("lobpcg_n200_eigvecs.npy").exists(), "lobpcg_n200_eigvecs.npy not written");

    assert!(
        angle < 0.01,
        "subspace angle {angle:.6e} rad exceeds 0.01 rad tolerance — \
         LOBPCG and dense EVD disagree on ring(200) eigenvectors"
    );
}
