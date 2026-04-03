#[path = "../common/mod.rs"]
mod common;

use common::load_sparse_csr_f32_u32;
use ndarray_npy::write_npy;
use spectral_init::{spectral_init, SpectralInitConfig};
use spectral_init::{
    compute_degrees, build_normalized_laplacian, solve_eigenproblem_pub, ComputeMode,
};
use std::path::Path;
use std::time::Instant;

#[test]
#[ignore = "requires MERFISH 10K subset data"]
fn export_merfish_init_10k() {
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/visual_eval/output");

    if !output_dir.exists() {
        panic!(
            "Visual eval output directory not found: {}. Run Phase 1 data generation first.",
            output_dir.display()
        );
    }

    let graph_path = output_dir.join("merfish_10k_graph.npz");
    if !graph_path.exists() {
        panic!(
            "MERFISH graph not found: {}. Run Phase 1 data generation first.",
            graph_path.display()
        );
    }

    let graph = load_sparse_csr_f32_u32(&graph_path);
    let n = graph.rows();

    println!("\nMERFISH 10K Rust Spectral Init Export");
    println!("=====================================");
    println!("Graph: {} nodes", n);

    // Emit solver level via the testing seam so the Python sweep harness can parse it.
    // Builds the Laplacian independently of spectral_init(); the actual embedding below
    // still uses spectral_init() as the authoritative pipeline path.
    let (_degrees, sqrt_deg) = compute_degrees(&graph, ComputeMode::PythonCompat);
    let inv_sqrt_deg: Vec<f64> = sqrt_deg
        .iter()
        .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
        .collect();
    let lap = build_normalized_laplacian(&graph, &inv_sqrt_deg);
    let (_, solver_level) = solve_eigenproblem_pub(&lap, 2, 42);
    println!("SOLVER_LEVEL={}", solver_level);

    let start = Instant::now();
    let coords = spectral_init(&graph, 2, 42, None, SpectralInitConfig::default())
        .unwrap_or_else(|e| panic!("spectral_init failed on merfish_10k: {e}"));
    let elapsed = start.elapsed();

    assert_eq!(coords.nrows(), n, "coords row count {} != graph node count {}", coords.nrows(), n);
    assert_eq!(coords.ncols(), 2, "expected 2-dimensional embedding, got {} columns", coords.ncols());
    assert!(
        coords.iter().all(|x| x.is_finite()),
        "coords contain NaN or Inf values"
    );

    let out_path = output_dir.join("merfish_10k_rust_init.npy");
    write_npy(&out_path, &coords)
        .unwrap_or_else(|e| panic!("failed to write merfish_10k_rust_init.npy: {e}"));

    println!("Elapsed: {:.2}s", elapsed.as_secs_f64());
    println!("Output:  {}", out_path.display());
    println!("=====================================");
}
