use ndarray::Array2;
use ndarray_npy::NpzReader;
use spectral_init::select_eigenvectors;
use std::fs::File;

#[test]
fn comp_e_selection_matches_python_blobs_connected_200() {
    let fixture_base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/blobs_connected_200");

    let mut d = NpzReader::new(
        File::open(fixture_base.join("comp_d_eigensolver.npz"))
            .expect("opening comp_d_eigensolver.npz"),
    )
    .expect("parsing comp_d_eigensolver.npz");
    let eigenvalues: ndarray::Array1<f64> = d.by_name("eigenvalues").expect("eigenvalues array");
    let eigenvectors: Array2<f64> = d.by_name("eigenvectors").expect("eigenvectors array");

    let mut e = NpzReader::new(
        File::open(fixture_base.join("comp_e_selection.npz"))
            .expect("opening comp_e_selection.npz"),
    )
    .expect("parsing comp_e_selection.npz");
    let expected_order: ndarray::Array1<i32> = e.by_name("order").expect("order array");
    let expected_embedding: Array2<f64> = e.by_name("embedding").expect("embedding array");

    let result = select_eigenvectors(eigenvalues.as_slice().unwrap(), &eigenvectors, 2);

    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());
    let computed_order: Vec<i32> = indices[1..3].iter().map(|&i| i as i32).collect();
    assert_eq!(computed_order, expected_order.as_slice().unwrap());

    for col in 0..2 {
        let r = result.column(col);
        let e = expected_embedding.column(col);
        let dot = r.dot(&e);
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        for (rv, ev) in r.iter().zip(e.iter()) {
            assert!((rv - sign * ev).abs() < 1e-10, "column {col} mismatch");
        }
    }
}
