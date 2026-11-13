//! RQ3: Threshold override validation — dense EVD vs LOBPCG subspace agreement.

use ndarray_npy::write_npy;
use spectral_init::__internal::{SolverLevel, SolverParams};
use spectral_init::ComputeMode;
use spectral_init_eval::diagnostics::max_subspace_angle;
use spectral_init_eval::{fixtures, internals};

#[test]
fn rq3_threshold_override_subspace_agreement() {
    let n = 200;
    let n_components = 2; // solve_eigenproblem returns n_components+1 = 3 eigenpairs

    // Build 200-node ring Laplacian
    let laplacian = fixtures::ring_laplacian(n);

    // ── Run 1: Force dense EVD (threshold=2000, n=200 < 2000 → Level 0) ──
    let dense_params = SolverParams {
        dense_n_threshold: 2000,
        ..SolverParams::default()
    };
    let ((eigs_dense, vecs_dense), report_dense) = internals::solve_with_unit_hint(
        &laplacian,
        n_components,
        42,
        ComputeMode::PythonCompat,
        &dense_params,
    );
    assert_eq!(
        report_dense.level,
        SolverLevel::DenseEvd,
        "expected Level 0 (dense EVD) with threshold=2000"
    );
    let level_dense = report_dense.level as u8;

    // ── Run 2: Force LOBPCG (threshold=50, n=200 >= 50 → Level 1+) ──
    let lobpcg_params = SolverParams {
        dense_n_threshold: 50,
        ..SolverParams::default()
    };
    let ((eigs_lobpcg, vecs_lobpcg), report_lobpcg) = internals::solve_with_unit_hint(
        &laplacian,
        n_components,
        42,
        ComputeMode::PythonCompat,
        &lobpcg_params,
    );
    assert!(
        report_lobpcg.level > SolverLevel::DenseEvd,
        "expected Level >= 1 (LOBPCG path) with threshold=50, got {:?}",
        report_lobpcg.level
    );
    let level_lobpcg = report_lobpcg.level as u8;

    // ── Save eigenvector matrices to .npy ──
    let results_dir = spectral_init_eval::results_dir().join("rq3_threshold");
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
    assert!(
        results_dir.join("dense_n200_eigvecs.npy").exists(),
        "dense_n200_eigvecs.npy not written"
    );
    assert!(
        results_dir.join("lobpcg_n200_eigvecs.npy").exists(),
        "lobpcg_n200_eigvecs.npy not written"
    );

    assert!(
        angle < 0.01,
        "subspace angle {angle:.6e} rad exceeds 0.01 rad tolerance — \
         LOBPCG and dense EVD disagree on ring(200) eigenvectors"
    );
}
