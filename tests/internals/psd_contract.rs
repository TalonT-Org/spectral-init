// Verifies the PSD contract: solve_eigenproblem must return eigenvalues in [0, 2]
// exactly (no tolerance margin), for all fixtures and all reachable solver levels.

use crate::{fixtures, internals};
use spectral_init::__internal::SolverParams;
use spectral_init::ComputeMode;

#[test]
fn test_psd_contract_all_fixtures() {
    for fixture_name in fixtures::DATASETS {
        let laplacian = fixtures::load_sparse_csr(
            &fixtures::fixture_path(fixture_name, "").join("comp_b_laplacian.npz"),
        );
        let ((eigenvalues, _eigenvectors), _report) = internals::solve_with_unit_hint(
            &laplacian,
            2,
            42,
            ComputeMode::PythonCompat,
            &SolverParams::default(),
        );
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
