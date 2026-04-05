//! CLI binary: compare exact vs approximate trustworthiness timing.
//!
//! Usage:
//!   tw_approx_runner --x X.npy --y Y.npy [--k 15] --sample <m> --seed <u64>
//!                    --output results.json
//!
//! Requires feature: cli
//! The approximate path is a stub returning 0.0 — completed in groupB.
//! Outputs a JSON file with n, m, seed, t_exact, t_approx, delta, wall_exact_s, wall_approx_s.

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut pargs = pico_args::Arguments::from_env();

    let x_path: std::path::PathBuf = pargs.value_from_str("--x")?;
    let y_path: std::path::PathBuf = pargs.value_from_str("--y")?;
    let k: usize = pargs.opt_value_from_str("--k")?.unwrap_or(15);
    let sample: usize = pargs.value_from_str("--sample")?;
    let seed: u64 = pargs.value_from_str("--seed")?;
    let output: std::path::PathBuf = pargs.value_from_str("--output")?;

    let x: ndarray::Array2<f64> = ndarray_npy::read_npy(&x_path)
        .map_err(|e| format!("failed to load X from {}: {e}", x_path.display()))?;
    let y: ndarray::Array2<f64> = ndarray_npy::read_npy(&y_path)
        .map_err(|e| format!("failed to load Y from {}: {e}", y_path.display()))?;
    let n = x.nrows();

    // Exact T
    let t0 = std::time::Instant::now();
    let t_exact = spectral_init::trustworthiness(x.view(), y.view(), k);
    let wall_exact_s = t0.elapsed().as_secs_f64();

    // Approximate T
    let t1 = std::time::Instant::now();
    let t_approx = spectral_init::trustworthiness_approx(x.view(), y.view(), k, sample, seed);
    let wall_approx_s = t1.elapsed().as_secs_f64();

    let delta = t_approx - t_exact;

    let result = serde_json::json!({
        "n": n,
        "m": sample,
        "seed": seed,
        "t_exact": t_exact,
        "t_approx": t_approx,
        "delta": delta,
        "wall_exact_s": wall_exact_s,
        "wall_approx_s": wall_approx_s
    });
    std::fs::write(&output, serde_json::to_string_pretty(&result)?)?;
    eprintln!("Wrote {}", output.display());
    Ok(())
}
