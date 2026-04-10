//! Experiment binary: subsampled trustworthiness / Rust tradeoff study.
//!
//! Usage:
//!   tw_subsample_experiment --mode preflight --data-dir <path>
//!   tw_subsample_experiment --mode exact --x <path> --y <path> --k 15 --reps 5 --warmup 1 --output <path>
//!   tw_subsample_experiment --mode subsample --x <path> --y <path> --k 15 --m 2000 --seed 0 --reps 5 --warmup 1 --output <path>
//!   tw_subsample_experiment --mode sanity --x <path> --y <path> --k 15 --m 10000 --output <path>

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ndarray::ArrayView2;
use rayon::prelude::*;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut pargs = pico_args::Arguments::from_env();

    let mode: String = pargs.value_from_str("--mode")?;

    match mode.as_str() {
        "preflight" => {
            let data_dir: PathBuf = pargs.value_from_str("--data-dir")?;
            run_preflight(&data_dir)?;
        }
        "exact" | "subsample" | "sanity" => {
            let x_path: PathBuf = pargs.value_from_str("--x")?;
            let y_path: PathBuf = pargs.value_from_str("--y")?;
            let k: usize = pargs.opt_value_from_str("--k")?.unwrap_or(15);
            let m: Option<usize> = pargs.opt_value_from_str("--m")?;
            let seed: Option<u64> = pargs.opt_value_from_str("--seed")?;
            let reps: usize = pargs.opt_value_from_str("--reps")?.unwrap_or(5);
            let warmup: usize = pargs.opt_value_from_str("--warmup")?.unwrap_or(1);
            let output: PathBuf = pargs.value_from_str("--output")?;

            run_experiment(&mode, &x_path, &y_path, k, m, seed, reps, warmup, &output)?;
        }
        other => return Err(format!("unknown mode: {other}").into()),
    }
    Ok(())
}

// ─── Preflight ───────────────────────────────────────────────────────────────

/// Verify all four MERFISH fixture files exist, load as f64, and check shapes.
fn run_preflight(data_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let fixtures: &[(&str, [usize; 2])] = &[
        ("merfish_n10k_x.npy", [10000, 50]),
        ("merfish_n10k_y.npy", [10000, 2]),
        ("merfish_n50k_x.npy", [50000, 50]),
        ("merfish_n50k_y.npy", [50000, 2]),
    ];

    for (filename, expected_shape) in fixtures {
        let path = data_dir.join(filename);
        if !path.exists() {
            println!("PREFLIGHT FAILED: missing fixture: {}", path.display());
            std::process::exit(1);
        }
        let arr: ndarray::Array2<f64> = ndarray_npy::read_npy(&path).map_err(|e| {
            format!(
                "PREFLIGHT FAILED: cannot load {} as f64 array: {e}",
                path.display()
            )
        })?;
        let shape = arr.shape();
        if shape[0] != expected_shape[0] || shape[1] != expected_shape[1] {
            println!(
                "PREFLIGHT FAILED: {} has shape {:?}, expected {:?}",
                filename, shape, expected_shape
            );
            std::process::exit(1);
        }
    }

    println!("PREFLIGHT OK");
    Ok(())
}

// ─── Experiment dispatcher ───────────────────────────────────────────────────

fn run_experiment(
    mode: &str,
    x_path: &Path,
    y_path: &Path,
    k: usize,
    m: Option<usize>,
    seed: Option<u64>,
    reps: usize,
    warmup: usize,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let x: ndarray::Array2<f64> = ndarray_npy::read_npy(x_path)?;
    let y: ndarray::Array2<f64> = ndarray_npy::read_npy(y_path)?;
    let n = x.nrows();

    // Determinism gate: abort if Rayon produces non-deterministic results
    determinism_gate(x.view(), y.view(), k)?;

    match mode {
        "exact" => run_exact(x.view(), y.view(), n, k, reps, warmup, output)?,
        "subsample" => {
            let m = m.ok_or("--m required for subsample mode")?;
            let seed = seed.ok_or("--seed required for subsample mode")?;
            run_subsample(x.view(), y.view(), n, k, m, seed, reps, warmup, output)?;
        }
        "sanity" => {
            let m = m.ok_or("--m required for sanity mode")?;
            run_sanity(x.view(), y.view(), n, k, m, output)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn determinism_gate(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let t1 = spectral_init::trustworthiness(x, y, k);
    let t2 = spectral_init::trustworthiness(x, y, k);
    let delta = (t1 - t2).abs();
    if delta > 1e-6 {
        return Err(format!(
            "FATAL: Rayon non-determinism detected: T1={t1}, T2={t2}, |delta|={delta}"
        )
        .into());
    }
    Ok(())
}

// ─── Exact mode ──────────────────────────────────────────────────────────────

fn run_exact(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    n: usize,
    k: usize,
    reps: usize,
    warmup: usize,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Warmup
    let warmup_start = std::time::Instant::now();
    let mut t_exact = 0.0;
    for _ in 0..warmup {
        t_exact = spectral_init::trustworthiness(x, y, k);
    }
    let warmup_exact_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;

    // Timed reps
    let mut wall_exact_ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = std::time::Instant::now();
        t_exact = spectral_init::trustworthiness(x, y, k);
        wall_exact_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let json = serde_json::json!({
        "n": n,
        "m": null,
        "k": k,
        "seed": null,
        "mode": "exact",
        "t_exact": t_exact,
        "t_sub": null,
        "abs_delta_t": null,
        "wall_exact_ms": wall_exact_ms,
        "wall_sub_ms": null,
        "warmup_exact_ms": warmup_exact_ms,
        "warmup_sub_ms": null,
        "cpu_model": cpu_model(),
        "core_count": core_count(),
        "rust_version": rust_version(),
        "git_commit": git_commit(),
    });
    write_json(output, &json)
}

// ─── Subsample mode ──────────────────────────────────────────────────────────

fn run_subsample(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    n: usize,
    k: usize,
    m: usize,
    seed: u64,
    reps: usize,
    warmup: usize,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use rand::SeedableRng;

    // Generate subsampled indices
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let query_idx: Vec<usize> = rand::seq::index::sample(&mut rng, n, m).into_vec();

    // Warmup exact
    let warmup_start = std::time::Instant::now();
    let mut t_exact = 0.0;
    for _ in 0..warmup {
        t_exact = spectral_init::trustworthiness(x, y, k);
    }
    let warmup_exact_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;

    // Warmup subsample
    let warmup_start = std::time::Instant::now();
    let mut t_sub = 0.0;
    for _ in 0..warmup {
        t_sub = trustworthiness_subsample(x, y, k, &query_idx);
    }
    let warmup_sub_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;

    // Timed exact reps
    let mut wall_exact_ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = std::time::Instant::now();
        t_exact = spectral_init::trustworthiness(x, y, k);
        wall_exact_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    // Timed subsample reps
    let mut wall_sub_ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = std::time::Instant::now();
        t_sub = trustworthiness_subsample(x, y, k, &query_idx);
        wall_sub_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let abs_delta_t = (t_exact - t_sub).abs();

    let json = serde_json::json!({
        "n": n,
        "m": m,
        "k": k,
        "seed": seed,
        "mode": "subsample",
        "t_exact": t_exact,
        "t_sub": t_sub,
        "abs_delta_t": abs_delta_t,
        "wall_exact_ms": wall_exact_ms,
        "wall_sub_ms": wall_sub_ms,
        "warmup_exact_ms": warmup_exact_ms,
        "warmup_sub_ms": warmup_sub_ms,
        "cpu_model": cpu_model(),
        "core_count": core_count(),
        "rust_version": rust_version(),
        "git_commit": git_commit(),
    });
    write_json(output, &json)
}

// ─── Sanity mode ─────────────────────────────────────────────────────────────

fn run_sanity(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    n: usize,
    k: usize,
    m: usize,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // All indices, deterministic — no RNG
    let query_idx: Vec<usize> = (0..n).collect();

    let t_exact = spectral_init::trustworthiness(x, y, k);
    let t_sub = trustworthiness_subsample(x, y, k, &query_idx);
    let abs_delta_t = (t_exact - t_sub).abs();

    if abs_delta_t >= 1e-10 {
        eprintln!(
            "WARNING: sanity check failed: abs_delta_t = {abs_delta_t:.2e} >= 1e-10"
        );
    }

    let json = serde_json::json!({
        "n": n,
        "m": m,
        "k": k,
        "seed": null,
        "mode": "sanity",
        "t_exact": t_exact,
        "t_sub": t_sub,
        "abs_delta_t": abs_delta_t,
        "wall_exact_ms": null,
        "wall_sub_ms": null,
        "warmup_exact_ms": null,
        "warmup_sub_ms": null,
        "cpu_model": cpu_model(),
        "core_count": core_count(),
        "rust_version": rust_version(),
        "git_commit": git_commit(),
    });
    write_json(output, &json)
}

// ─── Subsampled trustworthiness ──────────────────────────────────────────────
//
// Exact copy of src/metrics.rs:trustworthiness() with two changes:
// 1. Outer iterator is query_idx.into_par_iter() instead of (0..n).into_par_iter()
// 2. Normalization denominator uses m instead of n

fn trustworthiness_subsample(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    k: usize,
    query_idx: &[usize],
) -> f64 {
    use std::cell::RefCell;

    let n = x.nrows();
    let m = query_idx.len();
    let d_x = x.ncols();
    let d_y = y.ncols();

    assert!(k > 0 && k < n / 2, "k must be in (0, n/2)");

    // Runtime SIMD detection — identical to library
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    #[cfg(target_arch = "x86_64")]
    let use_avx2_y = d_y == 2 && y.is_standard_layout() && is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2_y = false;

    // Validate contiguity once before the parallel loop
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    if use_avx2 && d_x >= 10 {
        assert!(
            x.is_standard_layout(),
            "x must be in C-contiguous (standard) layout for SIMD dispatch"
        );
    }

    // Thread-local scratch buffers — different names from library to avoid collisions
    thread_local! {
        static SUB_DIST_X:    RefCell<Vec<f64>>   = const { RefCell::new(Vec::new()) };
        static SUB_INDICES:   RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    thread_local! {
        static SUB_DIST_Y:    RefCell<Vec<f64>>   = const { RefCell::new(Vec::new()) };
        static SUB_INDICES_Y: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    // THE KEY CHANGE: iterate over query_idx, not 0..n
    let penalty_sum: f64 = query_idx
        .into_par_iter()
        .map(|&i| {
            let xi = x.row(i);
            let yi = y.row(i);

            SUB_DIST_X.with(|dist_x_cell| {
                SUB_INDICES.with(|indices_cell| {
                    let mut dist_x = dist_x_cell.borrow_mut();
                    let mut indices = indices_cell.borrow_mut();

                    // Phase A: X distances (identical to library)
                    dist_x.clear();
                    dist_x.resize(n, 0.0f64);
                    for j in 0..n {
                        let xj = x.row(j);
                        dist_x[j] = {
                            #[cfg(all(
                                target_arch = "x86_64",
                                target_feature = "avx2",
                                target_feature = "fma"
                            ))]
                            {
                                if use_avx2 && d_x >= 10 {
                                    let si = xi.as_slice().expect("x row must be contiguous");
                                    let sj = xj.as_slice().expect("x row must be contiguous");
                                    unsafe { spectral_init::dist_sq_avx2_looped(si, sj) }
                                } else {
                                    xi.iter()
                                        .zip(xj.iter())
                                        .map(|(&a, &b)| (a - b) * (a - b))
                                        .sum()
                                }
                            }
                            #[cfg(not(all(
                                target_arch = "x86_64",
                                target_feature = "avx2",
                                target_feature = "fma"
                            )))]
                            {
                                xi.iter()
                                    .zip(xj.iter())
                                    .map(|(&a, &b)| (a - b) * (a - b))
                                    .sum()
                            }
                        };
                    }

                    // Phase B: X partial sort + kNN set (identical to library)
                    indices.clear();
                    indices.extend(0..n);
                    indices.select_nth_unstable_by(k, |&a, &b| {
                        dist_x[a].total_cmp(&dist_x[b]).then(a.cmp(&b))
                    });
                    let knn_x_set: HashSet<usize> = indices[..=k]
                        .iter()
                        .filter(|&&m_idx| m_idx != i)
                        .copied()
                        .collect();

                    SUB_DIST_Y.with(|dy_cell| {
                        SUB_INDICES_Y.with(|iy_cell| {
                            let mut dist_y = dy_cell.borrow_mut();
                            let mut indices_y = iy_cell.borrow_mut();

                            // Phase C: Y distances (identical to library)
                            dist_y.clear();
                            dist_y.resize(n, 0.0f64);

                            #[cfg(target_arch = "x86_64")]
                            if use_avx2_y {
                                let y_flat = y
                                    .as_slice()
                                    .expect("y must be standard layout for AVX2 dispatch");
                                let yi_slice = &y_flat[i * 2..(i + 1) * 2];
                                unsafe {
                                    spectral_init::dist_sq_2d_avx2_batch(
                                        yi_slice, y_flat, n, &mut dist_y,
                                    );
                                }
                            } else {
                                for j in 0..n {
                                    let yj = y.row(j);
                                    dist_y[j] = yi
                                        .iter()
                                        .zip(yj.iter())
                                        .map(|(&a, &b)| (a - b) * (a - b))
                                        .sum();
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                for j in 0..n {
                                    let yj = y.row(j);
                                    dist_y[j] = yi
                                        .iter()
                                        .zip(yj.iter())
                                        .map(|(&a, &b)| (a - b) * (a - b))
                                        .sum();
                                }
                            }

                            dist_y[i] = f64::INFINITY; // self-exclusion

                            // Phase C': Y partial sort (identical to library)
                            indices_y.clear();
                            indices_y.extend(0..n);
                            indices_y.select_nth_unstable_by(k, |&a, &b| {
                                dist_y[a].total_cmp(&dist_y[b]).then(a.cmp(&b))
                            });

                            // Phase D: Penalty (identical to library)
                            let mut row_penalty = 0u64;
                            for &j in &indices_y[..k] {
                                if !knn_x_set.contains(&j) {
                                    let dj = dist_x[j];
                                    let rank: usize = (0..n)
                                        .filter(|&m_idx| {
                                            dist_x[m_idx] < dj
                                                || (dist_x[m_idx] == dj && m_idx < j)
                                        })
                                        .count();
                                    row_penalty += (rank - k) as u64;
                                }
                            }
                            row_penalty as f64
                        })
                    })
                })
            })
        })
        .sum();

    // THE KEY CHANGE: denominator uses m, not n
    let denom = m as f64 * k as f64 * (2 * n).saturating_sub(3 * k + 1) as f64;
    1.0 - penalty_sum * 2.0 / denom
}

// ─── JSON output and metadata ────────────────────────────────────────────────

fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(value)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.splitn(2, ':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn core_count() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

fn rust_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
