#[path = "../common/mod.rs"]
mod common;

use spectral_init::{compute_degrees, ComputeMode};

fn run_comp_a_test(dataset: &str, expected_n: usize, mode: ComputeMode, tol: f64) {
    let base = common::fixture_path(dataset, "");

    // Load the pruned graph (input to compute_degrees)
    let graph = common::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));

    // Call the function under test
    let (degrees, sqrt_deg) = compute_degrees(&graph, mode);

    // Load Python reference
    let ref_path = base.join("comp_a_degrees.npz");
    let expected_deg: ndarray::Array1<f64> = common::load_dense(&ref_path, "degrees");
    let expected_sqrt: ndarray::Array1<f64> = common::load_dense(&ref_path, "sqrt_deg");

    assert_eq!(degrees.len(), expected_n, "degrees length");
    assert_eq!(sqrt_deg.len(), expected_n, "sqrt_deg length");

    for (i, (&got, &want)) in degrees.iter().zip(expected_deg.iter()).enumerate() {
        let abs_tol = tol * want.abs().max(1.0);
        assert!(
            (got - want).abs() < abs_tol,
            "degrees[{i}]: got {got}, want {want}, diff {}, tol {abs_tol}",
            (got - want).abs()
        );
    }
    for (i, (&got, &want)) in sqrt_deg.iter().zip(expected_sqrt.iter()).enumerate() {
        let abs_tol = tol * want.abs().max(1.0);
        assert!(
            (got - want).abs() < abs_tol,
            "sqrt_deg[{i}]: got {got}, want {want}, diff {}, tol {abs_tol}",
            (got - want).abs()
        );
    }
}

// ── PythonCompat tests (tight tolerance 1e-15, bit-for-bit match) ────────────

macro_rules! make_comp_a_python_compat_test {
    ($name:ident, $dataset:literal, $n:expr) => {
        #[test]
        fn $name() {
            run_comp_a_test($dataset, $n, ComputeMode::PythonCompat, 1e-15);
        }
    };
}

make_comp_a_python_compat_test!(comp_a_degrees_matches_python_blobs_connected_200,  "blobs_connected_200",  200);
make_comp_a_python_compat_test!(comp_a_degrees_matches_python_blobs_connected_2000, "blobs_connected_2000", 2000);
make_comp_a_python_compat_test!(comp_a_degrees_matches_python_disconnected_200,     "disconnected_200",     200);
make_comp_a_python_compat_test!(comp_a_degrees_matches_python_moons_200,            "moons_200",            200);
make_comp_a_python_compat_test!(comp_a_degrees_matches_python_circles_300,          "circles_300",          300);
make_comp_a_python_compat_test!(comp_a_degrees_matches_python_near_dupes_100,       "near_dupes_100",       100);
make_comp_a_python_compat_test!(comp_a_degrees_matches_python_blobs_50,             "blobs_50",             50);
make_comp_a_python_compat_test!(comp_a_degrees_matches_python_blobs_500,            "blobs_500",            500);
make_comp_a_python_compat_test!(comp_a_degrees_matches_python_blobs_5000,           "blobs_5000",           5000);

// ── RustNative tests (loose tolerance 3e-7, regression guard) ────────────────

macro_rules! make_comp_a_rust_native_test {
    ($name:ident, $dataset:literal, $n:expr) => {
        #[test]
        fn $name() {
            run_comp_a_test($dataset, $n, ComputeMode::RustNative, 3e-7);
        }
    };
}

make_comp_a_rust_native_test!(comp_a_degrees_rust_native_blobs_connected_200,  "blobs_connected_200",  200);
make_comp_a_rust_native_test!(comp_a_degrees_rust_native_blobs_connected_2000, "blobs_connected_2000", 2000);
make_comp_a_rust_native_test!(comp_a_degrees_rust_native_disconnected_200,     "disconnected_200",     200);
make_comp_a_rust_native_test!(comp_a_degrees_rust_native_moons_200,            "moons_200",            200);
make_comp_a_rust_native_test!(comp_a_degrees_rust_native_circles_300,          "circles_300",          300);
make_comp_a_rust_native_test!(comp_a_degrees_rust_native_near_dupes_100,       "near_dupes_100",       100);
make_comp_a_rust_native_test!(comp_a_degrees_rust_native_blobs_50,             "blobs_50",             50);
make_comp_a_rust_native_test!(comp_a_degrees_rust_native_blobs_500,            "blobs_500",            500);
make_comp_a_rust_native_test!(comp_a_degrees_rust_native_blobs_5000,           "blobs_5000",           5000);
