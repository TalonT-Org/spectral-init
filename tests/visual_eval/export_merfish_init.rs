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

#[test]
#[ignore = "requires MERFISH 20K subset data in temp/merfish_20k/"]
fn export_merfish_init_20k() {
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("temp/merfish_20k/output");

    if !output_dir.exists() {
        panic!(
            "Visual eval output directory not found: {}. Run Phase 1 data generation first.",
            output_dir.display()
        );
    }

    let graph_path = output_dir.join("merfish_20k_graph.npz");
    if !graph_path.exists() {
        panic!(
            "MERFISH graph not found: {}. Run Phase 1 data generation first.",
            graph_path.display()
        );
    }

    let graph = load_sparse_csr_f32_u32(&graph_path);
    let n = graph.rows();

    println!("\nMERFISH 20K Rust Spectral Init Export");
    println!("=====================================");
    println!("Graph: {} nodes", n);

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
        .unwrap_or_else(|e| panic!("spectral_init failed on merfish_20k: {e}"));
    let elapsed = start.elapsed();

    // Measure peak RSS via /proc/self/status (Linux only)
    let peak_rss_kb: u64 = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmPeak:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0);

    assert_eq!(coords.nrows(), n, "coords row count {} != graph node count {}", coords.nrows(), n);
    assert_eq!(coords.ncols(), 2, "expected 2-dimensional embedding, got {} columns", coords.ncols());
    assert!(
        coords.iter().all(|x| x.is_finite()),
        "coords contain NaN or Inf values"
    );

    let out_path = output_dir.join("merfish_20k_rust_init.npy");
    write_npy(&out_path, &coords)
        .unwrap_or_else(|e| panic!("failed to write merfish_20k_rust_init.npy: {e}"));

    // Write perf file: elapsed_s rss_kb  (format expected by run_compare in Python)
    let perf_path = output_dir.join("merfish_20k_rust_perf.txt");
    std::fs::write(&perf_path, format!("{:.4} {}\n", elapsed.as_secs_f64(), peak_rss_kb))
        .unwrap_or_else(|e| panic!("failed to write merfish_20k_rust_perf.txt: {e}"));

    println!("Elapsed: {:.2}s", elapsed.as_secs_f64());
    println!("Peak RSS: {} KB", peak_rss_kb);
    println!("Output:  {}", out_path.display());
    println!("=====================================");
}

#[test]
#[ignore = "requires MERFISH 100K subset data"]
fn export_merfish_init_100k() {
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/visual_eval/output");

    if !output_dir.exists() {
        panic!(
            "Visual eval output directory not found: {}. Run Phase 1 data generation first.",
            output_dir.display()
        );
    }

    let graph_path = output_dir.join("merfish_100k_graph.npz");
    if !graph_path.exists() {
        panic!(
            "MERFISH graph not found: {}. Run Phase 1 data generation first.",
            graph_path.display()
        );
    }

    let graph = load_sparse_csr_f32_u32(&graph_path);
    let n = graph.rows();

    println!("\nMERFISH 100K Rust Spectral Init Export");
    println!("======================================");
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
        .unwrap_or_else(|e| panic!("spectral_init failed on merfish_100k: {e}"));
    let elapsed = start.elapsed();

    assert_eq!(coords.nrows(), n, "coords row count {} != graph node count {}", coords.nrows(), n);
    assert_eq!(coords.ncols(), 2, "expected 2-dimensional embedding, got {} columns", coords.ncols());
    assert!(
        coords.iter().all(|x| x.is_finite()),
        "coords contain NaN or Inf values"
    );

    let out_path = output_dir.join("merfish_100k_rust_init.npy");
    write_npy(&out_path, &coords)
        .unwrap_or_else(|e| panic!("failed to write merfish_100k_rust_init.npy: {e}"));

    let peak_rss_kb: u64 = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmPeak:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0);

    let perf_path = output_dir.join("merfish_100k_rust_perf.txt");
    std::fs::write(&perf_path, format!("{:.4} {}\n", elapsed.as_secs_f64(), peak_rss_kb))
        .unwrap_or_else(|e| panic!("failed to write merfish_100k_rust_perf.txt: {e}"));

    println!("Elapsed: {:.2}s", elapsed.as_secs_f64());
    println!("Peak RSS: {} KB", peak_rss_kb);
    println!("Output:  {}", out_path.display());
    println!("======================================");
}

#[test]
#[ignore = "requires MERFISH 250K subset data"]
fn export_merfish_init_250k() {
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/visual_eval/output");

    if !output_dir.exists() {
        panic!(
            "Visual eval output directory not found: {}. Run Phase 1 data generation first.",
            output_dir.display()
        );
    }

    let graph_path = output_dir.join("merfish_250k_graph.npz");
    if !graph_path.exists() {
        panic!(
            "MERFISH graph not found: {}. Run Phase 1 data generation first.",
            graph_path.display()
        );
    }

    let graph = load_sparse_csr_f32_u32(&graph_path);
    let n = graph.rows();

    println!("\nMERFISH 250K Rust Spectral Init Export");
    println!("======================================");
    println!("Graph: {} nodes", n);

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
        .unwrap_or_else(|e| panic!("spectral_init failed on merfish_250k: {e}"));
    let elapsed = start.elapsed();

    assert_eq!(coords.nrows(), n, "coords row count {} != graph node count {}", coords.nrows(), n);
    assert_eq!(coords.ncols(), 2, "expected 2-dimensional embedding, got {} columns", coords.ncols());
    assert!(
        coords.iter().all(|x| x.is_finite()),
        "coords contain NaN or Inf values"
    );

    let out_path = output_dir.join("merfish_250k_rust_init.npy");
    write_npy(&out_path, &coords)
        .unwrap_or_else(|e| panic!("failed to write merfish_250k_rust_init.npy: {e}"));

    let peak_rss_kb: u64 = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmPeak:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0);

    let perf_path = output_dir.join("merfish_250k_rust_perf.txt");
    std::fs::write(&perf_path, format!("{:.4} {}\n", elapsed.as_secs_f64(), peak_rss_kb))
        .unwrap_or_else(|e| panic!("failed to write merfish_250k_rust_perf.txt: {e}"));

    println!("Elapsed: {:.2}s", elapsed.as_secs_f64());
    println!("Peak RSS: {} KB", peak_rss_kb);
    println!("Output:  {}", out_path.display());
    println!("======================================");
}
