// Verifies the PSD contract: solve_eigenproblem must return eigenvalues in [0, 2]
// exactly (no tolerance margin), for all fixtures and all reachable solver levels.

#[path = "../common/mod.rs"]
mod common;

use spectral_init::{solve_eigenproblem_pub, ComputeMode};

const ALL_FIXTURES: &[&str] = &[
    "blobs_50", "blobs_500", "moons_200", "circles_300",
    "near_dupes_100", "disconnected_200", "blobs_connected_200", "blobs_connected_2000",
];

#[test]
fn test_psd_contract_all_fixtures() {
    for fixture_name in ALL_FIXTURES {
        let laplacian = common::load_sparse_csr(
            &common::fixture_path(fixture_name, "").join("comp_b_laplacian.npz"),
        );
        let ((eigenvalues, _eigenvectors), _level) = solve_eigenproblem_pub(&laplacian, 2, 42, ComputeMode::PythonCompat);
        for (i, &lambda) in eigenvalues.iter().enumerate() {
            assert!(
                lambda >= 0.0,
                "{fixture_name}: eigenvalue[{i}] = {lambda:.4e} is negative after solve_eigenproblem. \
                 The PSD contract must be enforced at the solver chain boundary."
            );
            assert!(
                lambda <= 2.0,
                "{fixture_name}: eigenvalue[{i}] = {lambda:.4e} exceeds 2.0. \
                 The normalized Laplacian eigenvalues must lie in [0, 2]."
            );
        }
    }
}
