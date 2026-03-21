mod dense;
pub mod lobpcg;
mod rsvd;

// pub (not pub(crate)) so lib.rs can re-export it for integration tests.
pub use dense::dense_evd;

use ndarray::{Array1, Array2};
pub use crate::operator::LinearOperator;
use sprs::CsMatI;
use crate::SpectralError;
use crate::operator::CsrOperator;

/// Eigendecomposition result: (eigenvalues shape [k], eigenvectors shape [n, k]).
pub type EigenResult = (Array1<f64>, Array2<f64>);

/// Solver escalation chain. Tries LOBPCG → LOBPCG+reg → rSVD → dense EVD.
/// Returns `Err(SpectralError::ConvergenceFailure)` if all solvers are exhausted.
/// The `CsrOperator` for iterative solvers is constructed internally from `laplacian`.
pub(crate) fn solve(
    laplacian: &CsMatI<f64, usize>,
    n_components: usize,
    seed: u64,
) -> Result<EigenResult, SpectralError> {
    let op = CsrOperator(laplacian);

    // Level 1: LOBPCG without regularization
    if let Some(result) = lobpcg::lobpcg_solve(&op, n_components, seed, false) {
        return Ok(result);
    }

    // Level 2: LOBPCG with epsilon*I regularization
    if let Some(result) = lobpcg::lobpcg_solve(&op, n_components, seed, true) {
        return Ok(result);
    }

    // Level 3: rSVD (P2-11)
    todo!("Level 3: rsvd_solve — implement in P2-11")
}

// ─── Unit Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 6-node path-graph Laplacian in CSR format.
    ///
    /// Path graph: 0-1-2-3-4-5
    /// L = D - A where degree[0]=1, degree[1..4]=2, degree[5]=1
    fn path_graph_laplacian_6() -> CsMatI<f64, usize> {
        // Indptr: each row's non-zero count:
        //   row 0: cols 0,1      → 2 entries
        //   row 1: cols 0,1,2    → 3 entries
        //   rows 2-4 same pattern
        //   row 5: cols 4,5      → 2 entries
        CsMatI::new(
            (6, 6),
            vec![0usize, 2, 5, 8, 11, 14, 16],
            vec![
                0usize, 1, // row 0
                0, 1, 2,   // row 1
                1, 2, 3,   // row 2
                2, 3, 4,   // row 3
                3, 4, 5,   // row 4
                4, 5,      // row 5
            ],
            vec![
                1.0_f64, -1.0,        // row 0
                -1.0, 2.0, -1.0,      // row 1
                -1.0, 2.0, -1.0,      // row 2
                -1.0, 2.0, -1.0,      // row 3
                -1.0, 2.0, -1.0,      // row 4
                -1.0, 1.0,            // row 5
            ],
        )
    }

    #[test]
    fn solve_level1_success_returns_ok() {
        let laplacian = path_graph_laplacian_6();
        let result = solve(&laplacian, 2, 42);
        assert!(result.is_ok(), "solve() returned Err: {:?}", result.err());

        let (eigvals, _eigvecs) = result.unwrap();

        // All eigenvalues must be non-negative
        for (i, &v) in eigvals.iter().enumerate() {
            assert!(v >= -1e-10, "eigenvalue[{i}] is negative: {v}");
        }

        // Eigenvalues must be sorted ascending
        for i in 1..eigvals.len() {
            assert!(
                eigvals[i] >= eigvals[i - 1] - 1e-10,
                "eigenvalues not sorted: eigvals[{}]={} > eigvals[{}]={}",
                i - 1,
                eigvals[i - 1],
                i,
                eigvals[i]
            );
        }
    }
}
