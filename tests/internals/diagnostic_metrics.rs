use crate::diagnostics::*;
use ndarray::{Array1, Array2};

#[test]
fn t_const_01_diagnostic_thresholds_have_required_values() {
    assert_eq!(DEGENERATE_GAP_THRESHOLD, 1e-6_f64);
    assert_eq!(SUBSPACE_GRAM_DET_THRESHOLD, 0.95_f64);
}

// ── Accuracy: orthogonality_error ────────────────────────────────────────

#[test]
fn t_ortho_01_identity_columns_zero_error() {
    let v = Array2::<f64>::eye(3);
    let err = orthogonality_error(&v);
    assert!(
        err < 1e-14,
        "identity orthogonality_error={err:.2e}, expected ≈ 0"
    );
}

#[test]
fn t_ortho_02_non_orthogonal_positive_error() {
    let v = Array2::from_shape_vec(
        (3, 2),
        vec![1.0_f64, 1.0_f64, 0.0_f64, 1.0_f64, 0.0_f64, 0.0_f64],
    )
    .unwrap();
    let err = orthogonality_error(&v);
    assert!(
        err > 0.0,
        "non-orthogonal matrix should have positive error, got {err:.2e}"
    );
}

// ── Accuracy: check_eigenvalue_bounds ────────────────────────────────────

#[test]
fn t_bounds_01_valid_range_and_sorted() {
    let eigs = Array1::from_vec(vec![0.0_f64, 0.5, 1.0]);
    assert_eq!(check_eigenvalue_bounds(&eigs, 1e-8), (true, true));
}

#[test]
fn t_bounds_02_negative_eigenvalue_out_of_range() {
    let eigs = Array1::from_vec(vec![-0.1_f64, 0.5]);
    assert_eq!(check_eigenvalue_bounds(&eigs, 1e-8), (false, true));
}

#[test]
fn t_bounds_03_unsorted_eigenvalues() {
    let eigs = Array1::from_vec(vec![0.5_f64, 0.1]);
    assert_eq!(check_eigenvalue_bounds(&eigs, 1e-8), (true, false));
}

// ── Accuracy: separation_ratio ───────────────────────────────────────────

#[test]
fn t_sep_01_well_separated_clusters() {
    // 4 points: cluster 0 near [0,0], cluster 1 near [10,0]
    let data = vec![0.0_f64, 0.0, 0.1, 0.0, 10.0, 0.0, 10.1, 0.0];
    let embedding = Array2::from_shape_vec((4, 2), data).unwrap();
    let labels = vec![0usize, 0, 1, 1];
    let ratio = separation_ratio(embedding.view(), &labels);
    assert!(
        ratio > 1.0,
        "well-separated clusters should have ratio > 1.0, got {ratio:.4}"
    );
}

#[test]
fn t_sep_02_overlapping_clusters() {
    // 4 points where intra-cluster spread > inter-centroid distance
    let data = vec![0.0_f64, 0.0, 3.0, 0.0, 1.0, 0.0, 4.0, 0.0];
    let embedding = Array2::from_shape_vec((4, 2), data).unwrap();
    let labels = vec![0usize, 0, 1, 1];
    let ratio = separation_ratio(embedding.view(), &labels);
    // Centroids [1.5,0] and [2.5,0]: inter=1.0, max_intra=1.5. ratio=0.667
    assert!(
        ratio <= 1.0,
        "overlapping clusters should have ratio <= 1.0, got {ratio:.4}"
    );
}

// ── Parity: eigenvalue_abs_errors ────────────────────────────────────────

#[test]
fn t_abserr_01_basic_errors() {
    let computed = Array1::from_vec(vec![1.0_f64, 2.0]);
    let reference = Array1::from_vec(vec![1.1_f64, 2.2]);
    let errors = eigenvalue_abs_errors(&computed, &reference);
    assert!(
        (errors[0] - 0.1).abs() < 1e-14,
        "errors[0]={:.6e}, expected 0.1",
        errors[0]
    );
    assert!(
        (errors[1] - 0.2).abs() < 1e-14,
        "errors[1]={:.6e}, expected 0.2",
        errors[1]
    );
}

// ── Parity: subspace_gram_det ────────────────────────────────────────────

#[test]
fn t_gram2d_01_parallel_subspace_near_one() {
    // u = r = [[1,0],[0,1],[0,0]] (orthonormal columns)
    let data = vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0];
    let u = Array2::from_shape_vec((3, 2), data.clone()).unwrap();
    let r = Array2::from_shape_vec((3, 2), data).unwrap();
    let det = subspace_gram_det(u.view(), r.view());
    assert!(
        (det - 1.0).abs() < 1e-12,
        "parallel orthonormal subspace: det={det:.6e}, expected ≈ 1.0"
    );
}

#[test]
fn t_gram2d_02_orthogonal_subspace_near_zero() {
    // u spans {e1, e2}, r spans {e3, e4}
    let u_data = vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let r_data = vec![0.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let u = Array2::from_shape_vec((4, 2), u_data).unwrap();
    let r = Array2::from_shape_vec((4, 2), r_data).unwrap();
    let det = subspace_gram_det(u.view(), r.view());
    assert!(
        det < 1e-12,
        "orthogonal subspaces: det={det:.6e}, expected ≈ 0.0"
    );
}

// ── Parity: subspace_gram_det_kd ─────────────────────────────────────────

#[test]
fn t_gramkd_01_same_subspace_near_one() {
    // k=3 identity columns in R^3
    let eye = Array2::<f64>::eye(3);
    let det = subspace_gram_det_kd(eye.view(), eye.view());
    assert!(
        (det - 1.0).abs() < 1e-10,
        "same subspace: det={det:.6e}, expected ≈ 1.0"
    );
}

#[test]
fn t_gramkd_02_orthogonal_subspaces_near_zero() {
    // computed spans {e1,e2}, reference spans {e3,e4} in R^4
    let comp_data = vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let ref_data = vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let comp = Array2::from_shape_vec((4, 2), comp_data).unwrap();
    let reff = Array2::from_shape_vec((4, 2), ref_data).unwrap();
    let det = subspace_gram_det_kd(comp.view(), reff.view());
    assert!(
        det < 1e-10,
        "orthogonal subspaces: det={det:.6e}, expected ≈ 0.0"
    );
}

// ── Parity: sign_agnostic_max_error ──────────────────────────────────────

#[test]
fn t_signerr_01_identical_arrays_zero_error() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let arr = Array2::from_shape_vec((4, 1), data).unwrap();
    let err = sign_agnostic_max_error(&arr, &arr);
    assert_eq!(err, 0.0, "identical arrays: error={err:.2e}, expected 0.0");
}

#[test]
fn t_signerr_02_negated_columns_zero_error() {
    let computed_data = vec![1.0_f32, -2.0, 3.0, -4.0];
    let reference_data = vec![-1.0_f32, 2.0, -3.0, 4.0];
    let computed = Array2::from_shape_vec((4, 1), computed_data).unwrap();
    let reference = Array2::from_shape_vec((4, 1), reference_data).unwrap();
    let err = sign_agnostic_max_error(&computed, &reference);
    assert_eq!(
        err, 0.0,
        "fully negated column: error={err:.2e}, expected 0.0"
    );
}

#[test]
fn t_signerr_03_small_perturbation_small_error() {
    let reference_data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let computed_data = vec![1.01_f32, 2.01, 3.01, 4.01];
    let reference = Array2::from_shape_vec((4, 1), reference_data).unwrap();
    let computed = Array2::from_shape_vec((4, 1), computed_data).unwrap();
    let err = sign_agnostic_max_error(&computed, &reference);
    assert!(
        err > 0.0 && err < 0.1,
        "small perturbation: error={err:.4e}, expected small positive"
    );
}

// ── Diagnostic: tolerance_margin ─────────────────────────────────────────

#[test]
fn t_tolmargin_01_basic() {
    let margin = tolerance_margin(1e-5, 5e-6);
    assert!(
        (margin - 2.0).abs() < 1e-12,
        "tolerance_margin={margin:.6e}, expected 2.0"
    );
}

#[test]
fn t_tolmargin_02_zero_worst_value_returns_infinity() {
    let margin = tolerance_margin(1e-5, 0.0);
    assert!(
        margin.is_infinite() && margin > 0.0,
        "tolerance_margin with worst_value=0.0 should be +∞, got {margin:.6e}"
    );
}

// ── Diagnostic: spectral_gap ──────────────────────────────────────────────

#[test]
fn t_specgap_01_basic() {
    let eigs = Array1::from_vec(vec![0.0_f64, 0.5, 1.0]);
    let gap = spectral_gap(&eigs);
    assert!(
        (gap - 0.5).abs() < 1e-14,
        "spectral_gap={gap:.6e}, expected 0.5"
    );
}

// ── Diagnostic: eigenvalue_condition_number ───────────────────────────────

#[test]
fn t_condnum_01_basic() {
    let eigs = Array1::from_vec(vec![0.0_f64, 0.5, 1.0]);
    let cn = eigenvalue_condition_number(&eigs);
    assert!(
        (cn - 2.0).abs() < 1e-14,
        "eigenvalue_condition_number={cn:.6e}, expected 2.0"
    );
}
