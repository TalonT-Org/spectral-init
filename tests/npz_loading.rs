mod common;

#[test]
fn smoke_load_comp_a_degrees_blobs_connected_200() {
    let path = common::fixture_path("blobs_connected_200", "comp_a_degrees.npz");
    if !path.exists() {
        // Fixture not generated yet — skip gracefully with a message
        eprintln!(
            "SKIP: fixture not found at {:?}. Generate with: python tests/generate_fixtures.py",
            path
        );
        return;
    }

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
