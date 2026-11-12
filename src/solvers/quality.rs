//! Residual-quality thresholds and checks for the solver escalation chain.

use ndarray::{Array1, Array2};
use sprs::CsMatI;

/// Maximum acceptable max-residual from rSVD before falling to Level 5.
/// rSVD with 2 power iterations typically achieves 1e-4 to 1e-6 on well-conditioned
/// graphs; 1e-2 accepts all such results while correctly escalating pathological cases.
pub const RSVD_QUALITY_THRESHOLD: f64 = 1e-2;

/// Maximum acceptable max-residual from dense EVD (Levels 0 and 5).
/// Dense EVD via faer is numerically exact (machine precision); 1e-6 leaves
/// a generous margin while catching any pathological faer regression.
pub const DENSE_EVD_QUALITY_THRESHOLD: f64 = 1e-6;

/// Maximum acceptable max-residual from LOBPCG (Levels 1 and 3).
/// Set to 2e-5: provides ≥2× safety margin for the tightest observed fixture
/// (blobs_connected_2000 at 9.097e-6, margin = 2.20×). Tighter than rSVD (1e-2).
pub const LOBPCG_QUALITY_THRESHOLD: f64 = 2e-5;

/// Maximum acceptable max-residual from shift-and-invert LOBPCG (Level 2).
/// Sinv achieves near-exact results (like dense EVD); 1e-6 accepts all
/// well-converged results while correctly escalating pathological graphs.
pub const SINV_LOBPCG_QUALITY_THRESHOLD: f64 = 1e-6;

/// Computes the relative residual `‖L·v − λ·v‖₂ / max(‖v‖₂, ε)` for a single eigenpair.
pub fn eigenpair_residual(
    laplacian: &CsMatI<f64, usize>,
    eigenvector: &Array1<f64>,
    eigenvalue: f64,
) -> f64 {
    let lv = laplacian * eigenvector;
    let lambda_v = eigenvector.mapv(|x| eigenvalue * x);
    let diff = &lv - &lambda_v;
    let v_norm = eigenvector.dot(eigenvector).sqrt().max(1e-300);
    diff.dot(&diff).sqrt() / v_norm
}

/// Returns the maximum relative residual `‖L·v − λ·v‖₂ / ‖v‖₂` over all eigenpairs.
///
/// Delegates per-pair computation to [`eigenpair_residual`]. Returns `NaN` if any
/// individual residual is `NaN`.
pub fn max_eigenpair_residual(
    laplacian: &CsMatI<f64, usize>,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
) -> f64 {
    eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &lambda)| {
            let v = eigenvectors.column(i).to_owned();
            eigenpair_residual(laplacian, &v, lambda)
        })
        .fold(0.0_f64, |a, b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3-node path Laplacian: [[1,-1,0],[-1,2,-1],[0,-1,1]]
    fn path_laplacian_3() -> CsMatI<f64, usize> {
        CsMatI::new(
            (3, 3),
            vec![0usize, 2, 5, 7],
            vec![0usize, 1, 0, 1, 2, 1, 2],
            vec![1.0_f64, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
        )
    }

    #[test]
    fn t_const_01_solver_thresholds_have_required_values() {
        assert_eq!(DENSE_EVD_QUALITY_THRESHOLD, 1e-6_f64);
        assert_eq!(LOBPCG_QUALITY_THRESHOLD, 2e-5_f64);
        assert_eq!(SINV_LOBPCG_QUALITY_THRESHOLD, 1e-6_f64);
        assert_eq!(RSVD_QUALITY_THRESHOLD, 1e-2_f64);
    }

    #[test]
    fn t_const_02_threshold_ordering() {
        assert_eq!(DENSE_EVD_QUALITY_THRESHOLD, SINV_LOBPCG_QUALITY_THRESHOLD);
        const { assert!(SINV_LOBPCG_QUALITY_THRESHOLD < LOBPCG_QUALITY_THRESHOLD) };
        const { assert!(LOBPCG_QUALITY_THRESHOLD < RSVD_QUALITY_THRESHOLD) };
    }

    #[test]
    fn t_eresid_01_trivial_eigenvector_near_zero() {
        let laplacian = path_laplacian_3();
        let s = 1.0_f64 / 3.0_f64.sqrt();
        let v = Array1::from_vec(vec![s, s, s]);
        let residual = eigenpair_residual(&laplacian, &v, 0.0);
        assert!(
            residual < 1e-10,
            "trivial eigenvector residual={residual:.2e}, expected < 1e-10"
        );
    }

    #[test]
    fn t_eresid_02_non_eigenvector_large_residual() {
        let laplacian = path_laplacian_3();
        let v = Array1::from_vec(vec![1.0_f64, 0.0, 0.0]);
        let residual = eigenpair_residual(&laplacian, &v, 0.0);
        assert!(
            residual >= RSVD_QUALITY_THRESHOLD,
            "non-eigenvector residual={residual:.2e}, expected >= RSVD_QUALITY_THRESHOLD={RSVD_QUALITY_THRESHOLD:.2e}"
        );
    }

    #[test]
    fn t_maxresid_01_returns_worst_across_two_eigenpairs() {
        let laplacian = path_laplacian_3();
        // Trivial eigenvector at λ=0: small residual
        let s = 1.0_f64 / 3.0_f64.sqrt();
        // Non-eigenvector at λ=0: large residual
        let eigenvalues = Array1::from_vec(vec![0.0_f64, 0.0_f64]);
        let eigenvectors =
            Array2::from_shape_vec((3, 2), vec![s, 1.0_f64, s, 0.0_f64, s, 0.0_f64]).unwrap();
        let max_res = max_eigenpair_residual(&laplacian, &eigenvalues, &eigenvectors);
        // The second column [1,0,0] has a large residual; max must reflect that.
        assert!(
            max_res >= RSVD_QUALITY_THRESHOLD,
            "max_residual={max_res:.2e}, expected >= RSVD_QUALITY_THRESHOLD"
        );
    }

    #[test]
    fn t_maxresid_02_propagates_nan_from_any_eigenpair() {
        let laplacian = path_laplacian_3();
        let s = 1.0_f64 / 3.0_f64.sqrt();
        let eigenvalues = Array1::from_vec(vec![0.0_f64, 0.0_f64]);
        // The NaN pair first and last covers both operands of the fold.
        for nan_col in 0..2 {
            let mut eigenvectors = Array2::from_elem((3, 2), s);
            eigenvectors[[0, nan_col]] = f64::NAN;
            let max_res = max_eigenpair_residual(&laplacian, &eigenvalues, &eigenvectors);
            assert!(
                max_res.is_nan(),
                "NaN in column {nan_col}: max_residual={max_res:.2e}, expected NaN"
            );
        }
    }
}
