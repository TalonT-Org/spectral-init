#[path = "../common/mod.rs"]
mod common;

use ndarray::Ix1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use spectral_init::{normalize_signs_pub, scale_and_add_noise_pub, solve_eigenproblem_pub};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use spectral_init::operator::spmv_avx2_gather_pub;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use spectral_init::solve_eigenproblem_simd_pub;

use spectral_init::operator::spmv_csr;

const FIXTURES: &[&str] = &[
    "blobs_50",
    "near_dupes_100",
    "moons_200",
    "blobs_connected_200",
    "disconnected_200",
    "circles_300",
    "blobs_500",
    "blobs_connected_2000",
    "blobs_5000",
];

fn results_dir() -> std::path::PathBuf {
    match std::env::var("RESULTS_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("research/2026-03-27-computemode-simd-spmv-parity/results"),
    }
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    // Both exactly zero: 0 ULPs by definition.
    if a == 0.0 && b == 0.0 {
        return 0;
    }
    let ai = a.to_bits() as i64;
    let bi = b.to_bits() as i64;
    if (ai >= 0) == (bi >= 0) {
        // Same sign: direct subtraction in the ordered float integer space.
        (ai - bi).unsigned_abs()
    } else {
        // Opposite signs (non-zero): sum of distances from zero in each sign-half.
        // Handles -0.0 vs +0.0 edge case (both == 0.0 caught above, so this only
        // fires for genuine opposite-sign non-zero pairs).
        ai.unsigned_abs() + bi.unsigned_abs()
    }
}

#[test]
fn test_spmv_divergence() {
    let results_dir = results_dir();
    std::fs::create_dir_all(&results_dir).expect("cannot create RESULTS_DIR");

    let mut records: Vec<serde_json::Value> = Vec::new();

    for (fixture_idx, &fixture) in FIXTURES.iter().enumerate() {
        let lap_path = common::fixture_path(fixture, "comp_b_laplacian.npz");
        let laplacian = common::load_sparse_csr(&lap_path);
        let n = laplacian.rows();
        let nnz = laplacian.nnz();
        let indptr = laplacian.indptr().raw_storage().to_vec();
        let indices = laplacian.indices().to_vec();
        let data = laplacian.data().to_vec();

        for vec_idx in 0..5usize {
            let seed = (fixture_idx * 5 + vec_idx) as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let x: Vec<f64> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();

            let mut y_scalar = vec![0.0_f64; n];
            spmv_csr(&indptr, &indices, &data, &x, &mut y_scalar);

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            let (max_abs_diff, rms_diff, max_ulp) = {
                let mut y_avx2 = vec![0.0_f64; n];
                spmv_avx2_gather_pub(&indptr, &indices, &data, &x, &mut y_avx2);

                let mut max_abs: f64 = 0.0;
                let mut sum_sq: f64 = 0.0;
                let mut max_u: u64 = 0;
                for (s, a) in y_scalar.iter().zip(y_avx2.iter()) {
                    let diff = (s - a).abs();
                    if diff > max_abs {
                        max_abs = diff;
                    }
                    sum_sq += diff * diff;
                    let u = ulp_distance(*s, *a);
                    if u > max_u {
                        max_u = u;
                    }
                }
                let rms = (sum_sq / n as f64).sqrt();
                (max_abs, rms, max_u)
            };

            // On non-x86_64: scalar-only, divergence is zero by definition.
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            let (max_abs_diff, rms_diff, max_ulp): (f64, f64, u64) = (0.0, 0.0, 0);

            records.push(json!({
                "fixture": fixture,
                "n": n,
                "nnz": nnz,
                "vec_idx": vec_idx,
                "max_abs_diff": max_abs_diff,
                "rms_diff": rms_diff,
                "max_ulp": max_ulp,
            }));
        }
    }

    let json_str = serde_json::to_string_pretty(&records).expect("serialization failed");
    std::fs::write(results_dir.join("spmv_divergence.json"), json_str)
        .expect("cannot write spmv_divergence.json");
}

#[test]
fn test_solver_divergence() {
    let results_dir = results_dir();
    std::fs::create_dir_all(&results_dir).expect("cannot create RESULTS_DIR");
    std::fs::create_dir_all(results_dir.join("eigenvectors"))
        .expect("cannot create eigenvectors/");

    let mut records: Vec<serde_json::Value> = Vec::new();

    for &fixture in FIXTURES.iter() {
        let lap_path = common::fixture_path(fixture, "comp_b_laplacian.npz");
        let laplacian = common::load_sparse_csr(&lap_path);
        let n = laplacian.rows();

        let ((_, eigvec_scalar), solver_level_scalar) =
            solve_eigenproblem_pub(&laplacian, 2, 42);

        // x86_64 path: run SIMD solver and compare.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let record = {
            let mut eigvec_scalar = eigvec_scalar; // rebind as mut — normalize_signs_pub needs &mut
            let ((_, mut eigvec_avx2), solver_level_avx2) =
                solve_eigenproblem_simd_pub(&laplacian, 2, 42);

            normalize_signs_pub(&mut eigvec_scalar); // match production E.5
            normalize_signs_pub(&mut eigvec_avx2); // match production E.5

            let scaled_scalar = scale_and_add_noise_pub(eigvec_scalar, 42)
                .expect("scale_and_add_noise_pub failed");
            let scaled_avx2 = scale_and_add_noise_pub(eigvec_avx2, 42)
                .expect("scale_and_add_noise_pub failed");

            let f32_total = scaled_scalar.len();
            let f32_bitwise_identical = scaled_scalar
                .iter()
                .zip(scaled_avx2.iter())
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
            let f32_max_abs_diff = scaled_scalar
                .iter()
                .zip(scaled_avx2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                f32_max_abs_diff <= 4.77e-7,
                "fixture {fixture}: max_abs_diff={f32_max_abs_diff:.3e} > 4.77e-7 after normalize_signs — \
                 sign flip not resolved; SIMD kernel must be gated behind ComputeMode::RustNative"
            );

            let eigenvectors_saved = if solver_level_scalar >= 1 {
                let scalar_path = results_dir
                    .join("eigenvectors")
                    .join(format!("{}_scalar.npy", fixture));
                let avx2_path = results_dir
                    .join("eigenvectors")
                    .join(format!("{}_avx2.npy", fixture));
                ndarray_npy::write_npy(&scalar_path, &scaled_scalar)
                    .expect("write_npy scalar failed");
                ndarray_npy::write_npy(&avx2_path, &scaled_avx2)
                    .expect("write_npy avx2 failed");
                true
            } else {
                false
            };

            json!({
                "fixture": fixture,
                "n": n,
                "solver_level_scalar": solver_level_scalar,
                "solver_level_avx2": solver_level_avx2,
                "f32_bitwise_identical": f32_bitwise_identical,
                "f32_total_elements": f32_total,
                "f32_max_abs_diff": f32_max_abs_diff,
                "eigenvectors_saved": eigenvectors_saved,
            })
        };

        // Non-x86_64 path: record scalar result only; SIMD path does not exist.
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let record = {
            // Suppress unused variable: eigvec_scalar consumed but comparison skipped.
            let _ = eigvec_scalar;
            json!({
                "fixture": fixture,
                "n": n,
                "solver_level_scalar": solver_level_scalar,
                "solver_level_avx2": "not_applicable",
                "f32_bitwise_identical": null,
                "f32_total_elements": null,
                "f32_max_abs_diff": null,
                "eigenvectors_saved": false,
            })
        };

        records.push(record);
    }

    let json_str = serde_json::to_string_pretty(&records).expect("serialization failed");
    std::fs::write(results_dir.join("solver_divergence.json"), json_str)
        .expect("cannot write solver_divergence.json");
}

#[test]
fn test_spectral_gaps() {
    let results_dir = results_dir();
    std::fs::create_dir_all(&results_dir).expect("cannot create RESULTS_DIR");

    let mut records: Vec<serde_json::Value> = Vec::new();

    for &fixture in FIXTURES.iter() {
        let lap_path = common::fixture_path(fixture, "comp_b_laplacian.npz");
        let laplacian = common::load_sparse_csr(&lap_path);
        let n = laplacian.rows();

        let ev_path = common::fixture_path(fixture, "comp_d_eigensolver.npz");
        let gaps: ndarray::Array1<f64> =
            common::load_dense::<f64, Ix1>(&ev_path, "eigenvalue_gaps");

        let gaps_vec: Vec<f64> = gaps.iter().copied().collect();
        let min_gap = gaps_vec.iter().copied().fold(f64::MAX, f64::min);

        records.push(json!({
            "fixture": fixture,
            "n": n,
            "eigenvalue_gaps": gaps_vec,
            "min_gap": min_gap,
        }));
    }

    let json_str = serde_json::to_string_pretty(&records).expect("serialization failed");
    std::fs::write(results_dir.join("spectral_gaps.json"), json_str)
        .expect("cannot write spectral_gaps.json");
}
