#[path = "../common/mod.rs"]
mod common;

use common::load_sparse_csr_f32_u32;
use ndarray_npy::write_npy;
use spectral_init::{spectral_init, SpectralInitConfig};
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

    let start = Instant::now();
    let coords = spectral_init(&graph, 2, 42, None, SpectralInitConfig::default())
        .unwrap_or_else(|e| panic!("spectral_init failed on merfish_10k: {e}"));
    let elapsed = start.elapsed();

    let out_path = output_dir.join("merfish_10k_rust_init.npy");
    write_npy(&out_path, &coords)
        .unwrap_or_else(|e| panic!("failed to write merfish_10k_rust_init.npy: {e}"));

    println!("Elapsed: {:.2}s", elapsed.as_secs_f64());
    println!("Output:  {}", out_path.display());
    println!("=====================================");
}
