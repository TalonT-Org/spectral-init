#[path = "../common/mod.rs"]
mod common;

use common::load_sparse_csr_f32_u32;
use ndarray_npy::write_npy;
use spectral_init::{spectral_init, SpectralInitConfig};
use std::path::Path;
use std::time::Instant;

#[test]
#[ignore = "requires Phase 1 visual eval data"]
fn export_all_rust_inits() {
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/visual_eval/output");

    if !output_dir.exists() {
        panic!(
            "Visual eval output directory not found: {}. Run the Phase 1 data generation script first.",
            output_dir.display()
        );
    }

    let graph_files: Vec<_> = std::fs::read_dir(&output_dir)
        .expect("failed to read output dir")
        .filter_map(|e| {
            e.map_err(|err| eprintln!("Warning: failed to read directory entry: {err}"))
             .ok()
        })
        .filter(|e| {
            let name = e.file_name();
            let s = name.to_string_lossy();
            s.ends_with("_graph.npz")
        })
        .collect();

    if graph_files.is_empty() {
        println!("No graph files found in {}", output_dir.display());
        return;
    }

    println!("\nVisual Eval: Rust Spectral Init Export");
    println!("======================================");
    println!("{:<25} {:>6}  {:>8}", "Dataset", "n", "Time");

    let mut failures: Vec<String> = Vec::new();

    for entry in &graph_files {
        let path = entry.path();
        let file_name = entry.file_name();
        let stem = file_name.to_string_lossy();
        let name = stem.trim_end_matches("_graph.npz");

        let graph = load_sparse_csr_f32_u32(&path);
        let n = graph.rows();

        let start = Instant::now();
        match spectral_init(&graph, 2, 42, None, SpectralInitConfig::default()) {
            Ok(coords) => {
                let elapsed = start.elapsed();
                let out_path = output_dir.join(format!("{name}_rust_init.npy"));
                if let Err(e) = write_npy(&out_path, &coords) {
                    let msg = format!("write {name}_rust_init.npy: {e}");
                    eprintln!("  ERROR {msg}");
                    failures.push(msg);
                    continue;
                }
                println!(
                    "{:<25} {:>6}  {:>7.2}s",
                    name,
                    n,
                    elapsed.as_secs_f64()
                );
            }
            Err(e) => {
                let msg = format!("spectral_init for {name}: {e}");
                eprintln!("  ERROR {msg}");
                failures.push(msg);
            }
        }
    }

    println!("======================================");
    let exported = graph_files.len() - failures.len();
    if failures.is_empty() {
        println!("All {} exports saved to {}", exported, output_dir.display());
    } else {
        println!(
            "{}/{} exports saved to {}",
            exported,
            graph_files.len(),
            output_dir.display()
        );
    }
    assert!(
        failures.is_empty(),
        "{} export(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
