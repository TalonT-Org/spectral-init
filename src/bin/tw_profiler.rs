//! CLI binary: profile trustworthiness variants with warmup and timed iterations.
//!
//! Usage:
//!   tw_profiler --x X.npy --y Y.npy [--k 15] [--iters 5] [--warmup 2]
//!               --variant baseline --output results.json
//!
//! Requires features: cli, profiling
//! Outputs a JSON file with variant, n, iters (per-iteration times), mean_s, std_s, warmup,
//! and (when profiling feature is active) step_times_ns.

/// Physical core count of the benchmark machine.
/// Host: 1 socket × 8 cores × 2-way HT = 16 logical CPUs; pin to physical cores only.
const N_THREADS: usize = 8;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(N_THREADS)
        .build_global()
        .unwrap();

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
            "baseline"      => spectral_init::trustworthiness(xv, yv, k),
            "thread_local"  => spectral_init::trustworthiness_thread_local(xv, yv, k),
            "partial_rank"  => spectral_init::trustworthiness_partial_rank(xv, yv, k),
            "avx2_kernel"   => spectral_init::trustworthiness_avx2_kernel(xv, yv, k),
            "avx512_kernel" => spectral_init::trustworthiness_avx512_kernel(xv, yv, k),
            "combined"      => spectral_init::trustworthiness_combined(xv, yv, k),
            other => panic!("unknown variant: {other}"),
        }
    };

    // Warmup iterations — discard result
    for _ in 0..warmup {
        let _ = std::hint::black_box(dispatch(x.view(), y.view(), k));
    }

    // Timed iterations
    let mut iter_times: Vec<f64> = Vec::with_capacity(iters);
    #[cfg(feature = "profiling")]
    let mut step_times: Vec<[(&'static str, u64); 6]> = Vec::with_capacity(iters);

    for _ in 0..iters {
        #[cfg(feature = "profiling")]
        spectral_init::metrics::step_timing::reset();

        let t0 = std::time::Instant::now();
        let _ = std::hint::black_box(dispatch(x.view(), y.view(), k));
        iter_times.push(t0.elapsed().as_secs_f64());

        #[cfg(feature = "profiling")]
        step_times.push(spectral_init::metrics::step_timing::read());
    }

    let mean_s = iter_times.iter().sum::<f64>() / iters as f64;
    let variance = iter_times.iter().map(|&t| (t - mean_s).powi(2)).sum::<f64>() / iters as f64;
    let std_s = variance.sqrt();

    let mut result = serde_json::json!({
        "variant": variant,
        "n": n,
        "iters": iter_times,
        "mean_s": mean_s,
        "std_s": std_s,
        "warmup": warmup
    });

    #[cfg(feature = "profiling")]
    {
        // step_times[iter][step] -> accumulated ns across all n rows
        let step_summary: Vec<serde_json::Value> = step_times.iter().map(|steps| {
            let obj: serde_json::Map<String, serde_json::Value> = steps.iter()
                .map(|&(name, ns)| (name.to_string(), serde_json::json!(ns)))
                .collect();
            serde_json::Value::Object(obj)
        }).collect();
        result["step_times_ns"] = serde_json::json!(step_summary);
    }

    std::fs::write(&output, serde_json::to_string_pretty(&result)?)?;
    eprintln!("Wrote {}", output.display());
    Ok(())
}
