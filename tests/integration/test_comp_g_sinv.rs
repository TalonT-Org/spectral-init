#[path = "../common/mod.rs"]
mod common;

use spectral_init::lobpcg_sinv_solve;
use ndarray::Array1;
use ndarray_npy::NpzReader;

const N_COMPONENTS: usize = 2;

/// Return the number of connected components stored in comp_c_components.npz.
fn load_n_conn_components(dataset: &str) -> usize {
    let path = common::fixture_path(dataset, "comp_c_components.npz");
    let file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("dataset {dataset}: cannot open {path:?}: {e}"));
    let mut npz = NpzReader::new(file)
        .unwrap_or_else(|e| panic!("dataset {dataset}: NpzReader failed: {e}"));
    npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix0>("n_components")
        .unwrap_or_else(|e| panic!("dataset {dataset}: n_components key missing: {e}"))
        .into_scalar() as usize
}

/// Compute max ||L·v − λ·v|| / ||v|| over all eigenpairs.
fn max_residual(
    laplacian: &sprs::CsMat<f64>,
    eigvals: &Array1<f64>,
    eigvecs: &ndarray::Array2<f64>,
) -> f64 {
    eigvals
        .iter()
        .enumerate()
        .map(|(i, &lambda)| common::residual_spmv(laplacian, eigvecs.column(i), lambda))
        .fold(0.0_f64, f64::max)
}

fn run_sinv_test(dataset: &str) {
    let fixture_dir = common::fixture_path(dataset, "");

    // Skip disconnected datasets.
    let n_conn = load_n_conn_components(dataset);
    if n_conn > 1 {
        eprintln!("SKIP: {dataset} has {n_conn} connected components — sinv operates on single component");
        return;
    }

    let lap_path = fixture_dir.join("comp_b_laplacian.npz");
    let laplacian = common::load_sparse_csr(&lap_path);
    let n = laplacian.rows();

    // Skip if graph is too small for sinv (k >= n would trigger the None guard).
    if n <= N_COMPONENTS + 1 {
        eprintln!("SKIP: {dataset} n={n} <= N_COMPONENTS+1={}", N_COMPONENTS + 1);
        return;
    }

    // Load sqrt_deg for trivial eigenvector injection.
    let deg_path = fixture_dir.join("comp_a_degrees.npz");
    let sqrt_deg: Array1<f64> = common::load_dense(&deg_path, "sqrt_deg");
    assert_eq!(
        sqrt_deg.len(),
        n,
        "dataset={dataset}: sqrt_deg length {}, expected {n}", sqrt_deg.len()
    );

    // Load reference eigenvalues from dense EVD.
    let ref_path = fixture_dir.join("comp_d_eigensolver.npz");
    let ref_eigvals: Array1<f64> = common::load_dense(&ref_path, "eigenvalues");

    // Call shift-and-invert LOBPCG.
    let result = lobpcg_sinv_solve(&laplacian, N_COMPONENTS, 42, &sqrt_deg);
    assert!(
        result.is_some(),
        "dataset={dataset}: lobpcg_sinv_solve returned None"
    );
    let (eigvals, eigvecs) = result.unwrap();

    // T11 — eigenvalue accuracy: |λ_sinv[k] - λ_ref[k]| < 1e-8.
    let k = eigvals.len().min(ref_eigvals.len());
    for i in 0..k {
        let diff = (eigvals[i] - ref_eigvals[i]).abs();
        assert!(
            diff < 1e-8,
            "dataset={dataset}, eigenvalue[{i}]: got {}, ref {}, diff {diff:.2e}",
            eigvals[i], ref_eigvals[i]
        );
    }

    // T12 — residual quality: max_residual < 1e-8.
    let max_res = max_residual(&laplacian, &eigvals, &eigvecs);
    assert!(
        max_res < 1e-8,
        "dataset={dataset}: max_residual={max_res:.2e} >= 1e-8"
    );
}

macro_rules! make_sinv_tests {
    ($name_t11:ident, $name_t12:ident, $dataset:literal) => {
        #[test]
        fn $name_t11() {
            run_sinv_test($dataset);
        }
        // T12 is combined in run_sinv_test; a separate name variant for clarity.
        #[test]
        fn $name_t12() {
            run_sinv_test($dataset);
        }
    };
}

make_sinv_tests!(sinv_eigenvalue_accuracy_blobs_50, sinv_residual_quality_blobs_50, "blobs_50");
make_sinv_tests!(sinv_eigenvalue_accuracy_blobs_500, sinv_residual_quality_blobs_500, "blobs_500");
make_sinv_tests!(sinv_eigenvalue_accuracy_blobs_5000, sinv_residual_quality_blobs_5000, "blobs_5000");
make_sinv_tests!(
    sinv_eigenvalue_accuracy_blobs_connected_200,
    sinv_residual_quality_blobs_connected_200,
    "blobs_connected_200"
);
make_sinv_tests!(
    sinv_eigenvalue_accuracy_blobs_connected_2000,
    sinv_residual_quality_blobs_connected_2000,
    "blobs_connected_2000"
);
make_sinv_tests!(sinv_eigenvalue_accuracy_circles_300, sinv_residual_quality_circles_300, "circles_300");
make_sinv_tests!(
    sinv_eigenvalue_accuracy_disconnected_200,
    sinv_residual_quality_disconnected_200,
    "disconnected_200"
);
make_sinv_tests!(sinv_eigenvalue_accuracy_moons_200, sinv_residual_quality_moons_200, "moons_200");
make_sinv_tests!(
    sinv_eigenvalue_accuracy_near_dupes_100,
    sinv_residual_quality_near_dupes_100,
    "near_dupes_100"
);
