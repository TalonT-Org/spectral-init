use crate::{fixtures, internals};
use spectral_init::__internal::{CsrOperator, lobpcg_solve};

fn run_lobpcg_test(dataset: &str, _expected_n: usize, regularized: bool) {
    let fixture_dir = fixtures::fixture_path(dataset, "");
    let lap_path = fixture_dir.join("comp_b_laplacian.npz");

    let laplacian = fixtures::load_sparse_csr(&lap_path);
    let op = CsrOperator(&laplacian);

    let deg_path = fixture_dir.join("comp_a_degrees.npz");
    let sqrt_deg: ndarray::Array1<f64> = fixtures::load_dense(&deg_path, "sqrt_deg");
    assert_eq!(
        sqrt_deg.len(),
        laplacian.rows(),
        "sqrt_deg length must match Laplacian dimension"
    );

    let result = lobpcg_solve(&op, 2, 42, regularized, &sqrt_deg);
    assert!(
        result.is_some(),
        "lobpcg_solve (regularized={regularized}) returned None on {dataset}"
    );
    let ((eigvals, eigvecs), _) = result.unwrap();

    if !regularized {
        let ref_path = fixture_dir.join("comp_d_eigensolver.npz");
        let ref_eigvals: ndarray::Array1<f64> = fixtures::load_dense(&ref_path, "eigenvalues");

        let k = eigvals.len().min(ref_eigvals.len());
        for i in 0..k {
            assert!(
                (eigvals[i] - ref_eigvals[i]).abs() < 1e-6,
                "dataset={dataset}, eigenvalue[{i}] mismatch: got {}, expected {}",
                eigvals[i],
                ref_eigvals[i]
            );
        }

        for i in 0..eigvals.len() {
            let residual = internals::eigenpair_residual(&op, eigvecs.column(i), eigvals[i]);
            assert!(
                residual < 1e-5,
                "dataset={dataset}, residual for eigenpair {i}: {residual} >= 1e-5"
            );
        }
    } else {
        // Level 3: residuals against the unshifted Laplacian must be < 1e-5.
        for i in 0..eigvals.len() {
            let residual = internals::eigenpair_residual(&op, eigvecs.column(i), eigvals[i]);
            assert!(
                residual < 1e-5,
                "dataset={dataset}, level3 residual for eigenpair {i}: {residual} >= 1e-5"
            );
        }
    }
}

macro_rules! make_lobpcg_tests {
    ($name_l1:ident, $name_l3:ident, $dataset:literal, $n:expr) => {
        #[test]
        fn $name_l1() {
            run_lobpcg_test($dataset, $n, false);
        }
        #[test]
        fn $name_l3() {
            run_lobpcg_test($dataset, $n, true);
        }
    };
}

make_lobpcg_tests!(
    lobpcg_blobs_connected_2000_eigenvalues,
    lobpcg_blobs_connected_2000_level3,
    "blobs_connected_2000",
    2000
);
make_lobpcg_tests!(
    lobpcg_blobs_connected_200_eigenvalues,
    lobpcg_blobs_connected_200_level3,
    "blobs_connected_200",
    200
);
make_lobpcg_tests!(
    lobpcg_moons_200_eigenvalues,
    lobpcg_moons_200_level3,
    "moons_200",
    200
);
make_lobpcg_tests!(
    lobpcg_circles_300_eigenvalues,
    lobpcg_circles_300_level3,
    "circles_300",
    300
);
make_lobpcg_tests!(
    lobpcg_near_dupes_100_eigenvalues,
    lobpcg_near_dupes_100_level3,
    "near_dupes_100",
    100
);

#[test]
fn lobpcg_blobs_2000_residual_below_1e5_with_chebyshev() {
    let dataset = "blobs_connected_2000";
    let fixture_dir = fixtures::fixture_path(dataset, "");
    let lap_path = fixture_dir.join("comp_b_laplacian.npz");

    let laplacian = fixtures::load_sparse_csr(&lap_path);
    let op = CsrOperator(&laplacian);

    let deg_path = fixture_dir.join("comp_a_degrees.npz");
    let sqrt_deg: ndarray::Array1<f64> = fixtures::load_dense(&deg_path, "sqrt_deg");

    let result = lobpcg_solve(&op, 2, 42, false, &sqrt_deg);
    assert!(result.is_some(), "lobpcg_solve returned None on {dataset}");
    let ((eigvals, eigvecs), _) = result.unwrap();

    // All k=3 eigenpairs must have residual < 1e-5 (REQ-PERF-001).
    for i in 0..eigvals.len() {
        let residual = internals::eigenpair_residual(&op, eigvecs.column(i), eigvals[i]);
        assert!(
            residual < 1e-5,
            "dataset={dataset}, residual for eigenpair {i}: {residual} >= 1e-5 (Chebyshev preconditioning should achieve < 1e-5)"
        );
    }
}
