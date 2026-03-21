#[path = "../common/mod.rs"]
mod common;

use spectral_init::select_eigenvectors;

fn run_comp_e_test(dataset: &str, expected_n: usize) {
    let base = common::fixture_path(dataset, "");

    let eigenvalues: ndarray::Array1<f64> =
        common::load_dense(&base.join("comp_d_eigensolver.npz"), "eigenvalues");
    let eigenvectors: ndarray::Array2<f64> =
        common::load_dense(&base.join("comp_d_eigensolver.npz"), "eigenvectors");

    let expected_order: ndarray::Array1<i32> =
        common::load_dense(&base.join("comp_e_selection.npz"), "order");
    let expected_embedding: ndarray::Array2<f64> =
        common::load_dense(&base.join("comp_e_selection.npz"), "embedding");

    assert_eq!(eigenvectors.shape()[0], expected_n);

    let result = select_eigenvectors(eigenvalues.as_slice().unwrap(), &eigenvectors, 2);

    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());
    let computed_order: Vec<i32> = indices[1..3].iter().map(|&i| i as i32).collect();
    assert_eq!(computed_order, expected_order.as_slice().unwrap());

    let gap = (eigenvalues[computed_order[0] as usize] - eigenvalues[computed_order[1] as usize])
        .abs();

    if gap < 1e-6 {
        // Near-degenerate pair: use 2×2 determinant subspace check
        let r0 = result.column(0);
        let r1 = result.column(1);
        let e0 = expected_embedding.column(0);
        let e1 = expected_embedding.column(1);
        let a = r0.dot(&e0);
        let b = r0.dot(&e1);
        let c = r1.dot(&e0);
        let d = r1.dot(&e1);
        let det = (a * d - b * c).abs();
        assert!(
            det > 0.99,
            "subspace mismatch for near-degenerate pair: det = {det}"
        );
    } else {
        // Well-separated pair: element-wise sign-flipped comparison
        for col in 0..2 {
            let r = result.column(col);
            let e = expected_embedding.column(col);
            let dot = r.dot(&e);
            let sign = if dot < 0.0 { -1.0 } else { 1.0 };
            for (rv, ev) in r.iter().zip(e.iter()) {
                assert!(
                    (rv - sign * ev).abs() < 1e-10,
                    "column {col} mismatch: got {rv}, expected {}", sign * ev
                );
            }
        }
    }
}

#[test]
fn comp_e_selection_matches_python_blobs_connected_200() {
    run_comp_e_test("blobs_connected_200", 200);
}

#[test]
fn comp_e_selection_matches_python_moons_200() {
    run_comp_e_test("moons_200", 200);
}

#[test]
fn comp_e_selection_matches_python_circles_300() {
    run_comp_e_test("circles_300", 300);
}

#[test]
fn comp_e_selection_matches_python_near_dupes_100() {
    run_comp_e_test("near_dupes_100", 100);
}

#[test]
fn comp_e_selection_matches_python_blobs_connected_2000() {
    run_comp_e_test("blobs_connected_2000", 2000);
}

#[test]
fn comp_e_selection_matches_python_disconnected_200() {
    run_comp_e_test("disconnected_200", 200);
}
