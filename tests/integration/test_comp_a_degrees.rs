#[path = "../common/mod.rs"]
mod common;

use spectral_init::compute_degrees;

fn run_comp_a_test(dataset: &str, expected_n: usize) {
    let base = common::fixture_path(dataset, "");

    // Load the pruned graph (input to compute_degrees)
    let graph = common::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));

    // Call the function under test
    let (degrees, sqrt_deg) = compute_degrees(&graph);

    // Load Python reference
    let ref_path = base.join("comp_a_degrees.npz");
    let expected_deg: ndarray::Array1<f64> = common::load_dense(&ref_path, "degrees");
    let expected_sqrt: ndarray::Array1<f64> = common::load_dense(&ref_path, "sqrt_deg");

    assert_eq!(degrees.len(), expected_n, "degrees length");
    assert_eq!(sqrt_deg.len(), expected_n, "sqrt_deg length");

    // Tolerance is 1e-5: input graph weights are f32, and Python sums columns
    // (axis=0) while Rust sums rows. For symmetric graphs both give the same value,
    // but floating-point summation order differs, yielding ~1e-7 discrepancies.
    // 1e-5 is safely above f32 precision (≈1.2e-7) while tight enough to catch bugs.
    let degree_tol = 1e-5_f64;
    for (i, (&got, &want)) in degrees.iter().zip(expected_deg.iter()).enumerate() {
        assert!(
            (got - want).abs() < degree_tol,
            "degrees[{i}]: got {got}, want {want}, diff {}",
            (got - want).abs()
        );
    }
    for (i, (&got, &want)) in sqrt_deg.iter().zip(expected_sqrt.iter()).enumerate() {
        assert!(
            (got - want).abs() < degree_tol,
            "sqrt_deg[{i}]: got {got}, want {want}, diff {}",
            (got - want).abs()
        );
    }
}

#[test]
fn comp_a_degrees_matches_python_blobs_connected_200() {
    run_comp_a_test("blobs_connected_200", 200);
}

#[test]
fn comp_a_degrees_matches_python_blobs_connected_2000() {
    run_comp_a_test("blobs_connected_2000", 2000);
}

#[test]
fn comp_a_degrees_matches_python_disconnected_200() {
    run_comp_a_test("disconnected_200", 200);
}
