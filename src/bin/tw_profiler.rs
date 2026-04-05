//! CLI binary: profile trustworthiness variants with warmup and timed iterations.
//!
//! Usage:
//!   tw_profiler --x X.npy --y Y.npy [--k 15] [--iters 5] [--warmup 2]
//!               --variant baseline --output results.json
//!
//! Requires features: testing, cli
//! Outputs a JSON file with variant, n, iters (per-iteration times), mean_s, std_s, warmup.

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
    let iters: usize = pargs.opt_value_from_str("--iters")?.unwrap_or(5);
    let warmup: usize = pargs.opt_value_from_str("--warmup")?.unwrap_or(2);
    let variant: String = pargs.value_from_str("--variant")?;
    let output: std::path::PathBuf = pargs.value_from_str("--output")?;

    let x: ndarray::Array2<f64> = ndarray_npy::read_npy(&x_path)
        .map_err(|e| format!("failed to load X from {}: {e}", x_path.display()))?;
    let y: ndarray::Array2<f64> = ndarray_npy::read_npy(&y_path)
        .map_err(|e| format!("failed to load Y from {}: {e}", y_path.display()))?;
    let n = x.nrows();

    let dispatch = |xv: ndarray::ArrayView2<f64>, yv: ndarray::ArrayView2<f64>, k: usize| -> f64 {
        match variant.as_str() {
            "baseline"     => spectral_init::trustworthiness(xv, yv, k),
            "thread_local" => spectral_init::trustworthiness_thread_local(xv, yv, k),
            "partial_rank" => spectral_init::trustworthiness_partial_rank(xv, yv, k),
            "avx2_kernel"  => spectral_init::trustworthiness_avx2_kernel(xv, yv, k),
            "avx512_kernel" => spectral_init::trustworthiness_avx512_kernel(xv, yv, k),
            "combined"     => spectral_init::trustworthiness_combined(xv, yv, k),
            other => panic!("unknown variant: {other}"),
        }
    };

    // Warmup iterations — discard result
    for _ in 0..warmup {
        let _ = std::hint::black_box(dispatch(x.view(), y.view(), k));
    }

    // Timed iterations
    let mut iter_times: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        let _ = std::hint::black_box(dispatch(x.view(), y.view(), k));
        iter_times.push(t0.elapsed().as_secs_f64());
    }

    let mean_s = iter_times.iter().sum::<f64>() / iters as f64;
    let variance = iter_times.iter().map(|&t| (t - mean_s).powi(2)).sum::<f64>() / iters as f64;
    let std_s = variance.sqrt();

    let result = serde_json::json!({
        "variant": variant,
        "n": n,
        "iters": iter_times,
        "mean_s": mean_s,
        "std_s": std_s,
        "warmup": warmup
    });
    std::fs::write(&output, serde_json::to_string_pretty(&result)?)?;
    eprintln!("Wrote {}", output.display());
    Ok(())
}
