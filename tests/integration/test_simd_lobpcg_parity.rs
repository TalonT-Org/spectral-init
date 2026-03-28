#[path = "../common/mod.rs"]
mod common;

use ndarray::Array2;
use serde_json::json;
use spectral_init::{
    normalize_signs_pub,
    scale_and_add_noise_pub,
    solve_eigenproblem_pub,
    DEGENERATE_GAP_THRESHOLD,
};
use std::fs;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use spectral_init::solve_eigenproblem_simd_pub;

fn results_dir() -> std::path::PathBuf {
    match std::env::var("RESULTS_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("research/2026-03-28-simd-lobpcg-near-degenerate/results"),
    }
}

#[test]
fn test_gap_sweep() {
    todo!()
}

#[test]
fn test_discrete_multi_seed() {
    todo!()
}

#[test]
fn test_solver_level_parity() {
    todo!()
}
