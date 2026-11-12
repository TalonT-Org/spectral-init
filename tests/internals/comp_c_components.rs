use crate::fixtures;
use spectral_init::__internal::find_components;

/// Load comp_c_components.npz and compare Rust output against Python reference.
/// `expected_n_components` is the known ground-truth count for the dataset.
fn run_comp_c_test(dataset: &str, expected_n_components: usize) {
    let base = fixtures::fixture_path(dataset, "");

    // Run the function under test
    let graph = fixtures::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));
    let (labels, n_components) = find_components(graph.view());

    // Load Python reference
    let ref_path = base.join("comp_c_components.npz");
    let py_n: i32 =
        fixtures::load_dense::<i32, ndarray::Ix0>(&ref_path, "n_components").into_scalar();
    let py_labels_arr: ndarray::Array1<i32> = fixtures::load_dense(&ref_path, "labels");
    let py_labels: Vec<usize> = py_labels_arr.iter().map(|&x| x as usize).collect();

    assert_eq!(
        n_components, py_n as usize,
        "dataset {dataset}: n_components mismatch (rust={n_components}, python={py_n})"
    );
    assert_eq!(
        n_components, expected_n_components,
        "dataset {dataset}: expected {expected_n_components} components, got {n_components}"
    );
    assert_eq!(
        fixtures::partition_of(&labels),
        fixtures::partition_of(&py_labels),
        "dataset {dataset}: label partitioning does not match Python"
    );
}

// ── Per-dataset tests ─────────────────────────────────────────────────────────

#[test]
fn comp_c_components_matches_python_blobs_50() {
    run_comp_c_test("blobs_50", 3);
}

#[test]
fn comp_c_components_matches_python_blobs_500() {
    run_comp_c_test("blobs_500", 4);
}

#[test]
fn comp_c_components_matches_python_blobs_5000() {
    run_comp_c_test("blobs_5000", 2);
}

#[test]
fn comp_c_components_matches_python_blobs_connected_200() {
    run_comp_c_test("blobs_connected_200", 1);
}

#[test]
fn comp_c_components_matches_python_blobs_connected_2000() {
    run_comp_c_test("blobs_connected_2000", 1);
}

#[test]
fn comp_c_components_matches_python_circles_300() {
    run_comp_c_test("circles_300", 1);
}

#[test]
fn comp_c_components_matches_python_moons_200() {
    run_comp_c_test("moons_200", 1);
}

#[test]
fn comp_c_components_matches_python_near_dupes_100() {
    run_comp_c_test("near_dupes_100", 1);
}

#[test]
fn comp_c_components_matches_python_disconnected_200() {
    run_comp_c_test("disconnected_200", 4);
}
