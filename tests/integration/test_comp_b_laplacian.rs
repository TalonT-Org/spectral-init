#[path = "../common/mod.rs"]
mod common;

use spectral_init::build_normalized_laplacian;

fn run_comp_b_test(dataset: &str, expected_n: usize) {
    let base = common::fixture_path(dataset, "");

    // Load inputs
    let graph = common::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));

    // Load comp_a output for sqrt_deg (already validated by comp_a tests)
    let sqrt_deg: ndarray::Array1<f64> =
        common::load_dense(&base.join("comp_a_degrees.npz"), "sqrt_deg");
    let inv_sqrt_deg: Vec<f64> = sqrt_deg
        .iter()
        .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
        .collect();

    // Call function under test
    let l = build_normalized_laplacian(&graph, &inv_sqrt_deg);

    assert_eq!(l.rows(), expected_n, "Laplacian rows");
    assert_eq!(l.cols(), expected_n, "Laplacian cols");

    // Load Python reference
    let ref_l = common::load_sparse_csr(&base.join("comp_b_laplacian.npz"));

    // Element-wise comparison of all nonzero entries in the reference
    for (&val, (row, col)) in ref_l.iter() {
        let got = l.get(row, col).copied().unwrap_or(0.0);
        assert!(
            (got - val).abs() < 1e-14,
            "L[{row},{col}]: got {got}, want {val}, diff {}",
            (got - val).abs()
        );
    }

    // Symmetry: for every nonzero (i,j), L[j,i] must match within 1e-15
    for (&val, (row, col)) in l.iter() {
        let sym = l.get(col, row).copied().unwrap_or(0.0);
        assert!(
            (val - sym).abs() < 1e-15,
            "symmetry violation L[{row},{col}]={val}, L[{col},{row}]={sym}"
        );
    }

    // Diagonal: all 1.0
    for i in 0..expected_n {
        let d = l.get(i, i).copied().unwrap_or(0.0);
        assert!(
            (d - 1.0).abs() < 1e-15,
            "L[{i},{i}] = {d}, expected 1.0"
        );
    }
}

#[test]
fn comp_b_laplacian_matches_python_blobs_connected_200() {
    run_comp_b_test("blobs_connected_200", 200);
}

#[test]
fn comp_b_laplacian_matches_python_blobs_connected_2000() {
    run_comp_b_test("blobs_connected_2000", 2000);
}

#[test]
fn comp_b_laplacian_matches_python_disconnected_200() {
    run_comp_b_test("disconnected_200", 200);
}
