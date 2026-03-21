mod common;

/// Run with `cargo test -- --ignored` after generating fixtures:
///   python tests/generate_fixtures.py
#[test]
#[ignore = "requires generated fixtures; run `python tests/generate_fixtures.py` first"]
fn smoke_load_comp_a_degrees_blobs_connected_200() {
    let path = common::fixture_path("blobs_connected_200", "comp_a_degrees.npz");
    assert!(
        path.exists(),
        "fixture not found at {:?}; run: python tests/generate_fixtures.py",
        path
    );

    let degrees: ndarray::Array1<f64> = common::load_dense(&path, "degrees");
    let sqrt_deg: ndarray::Array1<f64> = common::load_dense(&path, "sqrt_deg");

    // Shape: blobs_connected_200 has n=200
    assert_eq!(degrees.len(), 200, "degrees shape mismatch");
    assert_eq!(sqrt_deg.len(), 200, "sqrt_deg shape mismatch");

    // All degrees are positive (each node has at least one neighbor)
    assert!(
        degrees.iter().all(|&d| d > 0.0),
        "all degrees must be positive"
    );

    // sqrt_deg[i] == sqrt(degrees[i])
    for (&d, &s) in degrees.iter().zip(sqrt_deg.iter()) {
        approx::assert_relative_eq!(s, d.sqrt(), epsilon = 1e-12);
    }
}
