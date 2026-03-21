#[path = "../common/mod.rs"]
mod common;

use ndarray_npy::NpzReader;
use spectral_init::find_components;

/// Load comp_c_components.npz and compare Rust output against Python reference.
/// `expected_n_components` is the known ground-truth count for the dataset.
fn run_comp_c_test(dataset: &str, expected_n_components: usize) {
    let base = common::fixture_path(dataset, "");

    // Run the function under test
    let graph = common::load_sparse_csr_f32_u32(&base.join("step5a_pruned.npz"));
    let (labels, n_components) = find_components(&graph);

    // Load Python reference
    let ref_path = base.join("comp_c_components.npz");
    let file = std::fs::File::open(&ref_path)
        .unwrap_or_else(|e| panic!("dataset {dataset}: cannot open {ref_path:?}: {e}"));
    let mut npz = NpzReader::new(file)
        .unwrap_or_else(|e| panic!("dataset {dataset}: NpzReader failed for {ref_path:?}: {e}"));

    let py_n: i32 = npz
        .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix0>("n_components")
        .unwrap_or_else(|e| panic!("dataset {dataset}: n_components key missing: {e}"))
        .into_scalar();
    let py_labels_arr: ndarray::Array1<i32> = npz
        .by_name("labels")
        .unwrap_or_else(|e| panic!("dataset {dataset}: labels key missing: {e}"));
    let py_labels: Vec<usize> = py_labels_arr.iter().map(|&x| x as usize).collect();

    assert_eq!(
        n_components,
        py_n as usize,
        "dataset {dataset}: n_components mismatch (rust={n_components}, python={py_n})"
    );
    assert_eq!(
        n_components,
        expected_n_components,
        "dataset {dataset}: expected {expected_n_components} components, got {n_components}"
    );
    assert_eq!(
        common::partition_of(&labels),
        common::partition_of(&py_labels),
        "dataset {dataset}: label partitioning does not match Python"
    );
}

// ── Per-dataset tests ─────────────────────────────────────────────────────────

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_c_components_matches_python_blobs_50() {
    run_comp_c_test("blobs_50", 3);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_c_components_matches_python_blobs_500() {
    run_comp_c_test("blobs_500", 4);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_c_components_matches_python_blobs_5000() {
    run_comp_c_test("blobs_5000", 2);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_c_components_matches_python_blobs_connected_200() {
    run_comp_c_test("blobs_connected_200", 1);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_c_components_matches_python_blobs_connected_2000() {
    run_comp_c_test("blobs_connected_2000", 1);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_c_components_matches_python_circles_300() {
    run_comp_c_test("circles_300", 1);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_c_components_matches_python_moons_200() {
    run_comp_c_test("moons_200", 1);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_c_components_matches_python_near_dupes_100() {
    run_comp_c_test("near_dupes_100", 1);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_c_components_matches_python_disconnected_200() {
    run_comp_c_test("disconnected_200", 4);
}

// ── All-datasets sweep ────────────────────────────────────────────────────────

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_c_components_matches_python_all_datasets() {
    let datasets: &[(&str, usize)] = &[
        ("blobs_50", 3),
        ("blobs_500", 4),
        ("blobs_5000", 2),
        ("blobs_connected_200", 1),
        ("blobs_connected_2000", 1),
        ("circles_300", 1),
        ("moons_200", 1),
        ("near_dupes_100", 1),
        ("disconnected_200", 4),
    ];
    for &(dataset, expected) in datasets {
        run_comp_c_test(dataset, expected);
    }
}
