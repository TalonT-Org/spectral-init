//! CLI binary: profiling harness for the trustworthiness metric.
//!
//! Usage (file mode):
//!   tw_profiler --x X.npy --y Y.npy [--variant flat_simd|kdtree] \
//!     [--output results.json] [--k 15] [--iters 5] [--warmup 2]
//!
//! Usage (in-memory mode):
//!   tw_profiler --n 1000 [--dist uniform|gauss] [--variant flat_simd|kdtree] \
//!     [--output results.json] [--k 15] [--iters 5] [--warmup 2]
//!
//! Runs the chosen trustworthiness variant multiple times (after warmup) and
//! writes structured JSON with timing statistics. Per-iteration step timing
//! arrays are captured via --stderr-capture.

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut pargs = pico_args::Arguments::from_env();

    // ── Data source args ──────────────────────────────────────────────────
    let x_path: Option<std::path::PathBuf> = pargs.opt_value_from_str("--x")?;
    let y_path: Option<std::path::PathBuf> = pargs.opt_value_from_str("--y")?;
    let n_arg:  Option<usize>              = pargs.opt_value_from_str("--n")?;
    let dist:   String = pargs.opt_value_from_str("--dist")?.unwrap_or_else(|| "uniform".to_string());

    // ── Variant + standard args ───────────────────────────────────────────
    let variant: String = pargs.opt_value_from_str("--variant")?.unwrap_or_else(|| "flat_simd".to_string());
    let output_path: Option<std::path::PathBuf> = pargs.opt_value_from_str("--output")?;
    let k: usize = pargs.opt_value_from_str("--k")?.unwrap_or(15);
    let iters: usize = pargs.opt_value_from_str("--iters")?.unwrap_or(5);
    if iters == 0 {
        return Err("--iters must be > 0".into());
    }
    let warmup: usize  = pargs.opt_value_from_str("--warmup")?.unwrap_or(2);
    let stderr_capture: Option<std::path::PathBuf> = pargs.opt_value_from_str("--stderr-capture")?;

    // ── Validate variant ──────────────────────────────────────────────────
    let use_kdtree = match variant.as_str() {
        "flat_simd" => false,
        "kdtree"    => true,
        other       => return Err(format!("unknown --variant '{other}'; expected flat_simd or kdtree").into()),
    };

    // ── Load or generate data ─────────────────────────────────────────────
    let (x, y) = match (x_path, y_path, n_arg) {
        (Some(xp), Some(yp), None) => {
            let x: ndarray::Array2<f64> = ndarray_npy::read_npy(&xp)
                .map_err(|e| format!("failed to load X from {}: {e}", xp.display()))?;
            let y: ndarray::Array2<f64> = ndarray_npy::read_npy(&yp)
                .map_err(|e| format!("failed to load Y from {}: {e}", yp.display()))?;
            (x, y)
        }
        (None, None, Some(n)) => {
            generate_data(n, &dist)?
        }
        (Some(_), Some(_), Some(_)) => {
            return Err("provide --x/--y OR --n, not both".into());
        }
        _ => {
            return Err("provide either --x/--y (file paths) or --n (in-memory generation)".into());
        }
    };

    // ── Warmup (no stderr capture yet) ────────────────────────────────────
    for _ in 0..warmup {
        let _ = std::hint::black_box(
            spectral_init::trustworthiness_inner(x.view(), y.view(), k, use_kdtree)
        );
    }

    // ── Set up stderr capture AFTER warmup ────────────────────────────────
    if let Some(ref capture_path) = stderr_capture {
        redirect_stderr(capture_path)?;
    }

    // ── Timed iterations ──────────────────────────────────────────────────
    let mut times = Vec::with_capacity(iters);
    let mut score = 0.0f64;
    for _ in 0..iters {
        let start = std::time::Instant::now();
        score = std::hint::black_box(
            spectral_init::trustworthiness_inner(x.view(), y.view(), k, use_kdtree)
        );
        times.push(start.elapsed().as_secs_f64());
    }

    let n_rows = x.nrows();
    let mean_s = times.iter().sum::<f64>() / times.len() as f64;
    let std_s = if times.len() > 1 {
        let var = times.iter().map(|&t| (t - mean_s).powi(2)).sum::<f64>()
            / (times.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    // ── Parse per-iteration step timings ─────────────────────────────────
    let step_timing = parse_step_timing(&stderr_capture);

    // ── Build JSON output ─────────────────────────────────────────────────
    let mut result = serde_json::Map::new();
    result.insert("n".into(),       serde_json::Value::from(n_rows));
    result.insert("k".into(),       serde_json::Value::from(k));
    result.insert("variant".into(), serde_json::Value::from(variant.clone()));
    result.insert("dist".into(),    serde_json::Value::from(dist.clone()));
    result.insert("iters".into(),   serde_json::json!(times));
    result.insert("mean_s".into(),  serde_json::json!(round_to(mean_s, 6)));
    result.insert("std_s".into(),   serde_json::json!(round_to(std_s, 6)));
    result.insert("warmup".into(),  serde_json::Value::from(warmup));
    result.insert("score".into(),   serde_json::json!(score));
    if !step_timing.is_empty() {
        result.insert("step_timing".into(), serde_json::json!(step_timing));
    }

    let json = serde_json::to_string_pretty(&serde_json::Value::Object(result))?;
    match output_path {
        Some(ref path) => std::fs::write(path, &json)?,
        None           => println!("{json}"),
    }

    Ok(())
}

fn generate_data(
    n: usize,
    dist: &str,
) -> Result<(ndarray::Array2<f64>, ndarray::Array2<f64>), Box<dyn std::error::Error>> {
    use rand::SeedableRng;
    use rand::Rng;

    match dist {
        "uniform" => {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
            let x = ndarray::Array2::from_shape_fn((n, 10), |_| rng.random::<f64>());
            let y = ndarray::Array2::from_shape_fn((n, 2),  |_| rng.random::<f64>());
            Ok((x, y))
        }
        "gauss" => {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(99);
            let x = ndarray::Array2::from_shape_fn((n, 10), |_| rng.random::<f64>());
            let y = gauss_mixture_y(&mut rng, n);
            Ok((x, y))
        }
        other => Err(format!("unknown --dist '{other}'; expected uniform or gauss").into()),
    }
}

fn gauss_mixture_y(rng: &mut impl rand::Rng, n: usize) -> ndarray::Array2<f64> {
    use rand_distr::{Distribution, Normal};
    let centers: [(f64, f64); 8] = [
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
        (0.0, 3.0), (1.0, 3.0), (2.0, 3.0), (3.0, 3.0),
    ];
    let sigma = 0.3f64;
    let normal = Normal::new(0.0, sigma).expect("valid normal distribution");
    let n_clusters = centers.len();
    let per = n / n_clusters;
    let remainder = n % n_clusters;

    let mut rows: Vec<[f64; 2]> = Vec::with_capacity(n);
    for (i, &(cx, cy)) in centers.iter().enumerate() {
        let count = per + if i < remainder { 1 } else { 0 };
        for _ in 0..count {
            rows.push([cx + normal.sample(rng), cy + normal.sample(rng)]);
        }
    }
    // shuffle
    use rand::seq::SliceRandom;
    rows.shuffle(rng);
    ndarray::Array2::from_shape_fn((n, 2), |(i, j)| rows[i][j])
}

fn round_to(val: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    (val * factor).round() / factor
}

#[cfg(unix)]
fn redirect_stderr(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::io::IntoRawFd;
    let file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create stderr capture file {}: {e}", path.display()))?;
    let fd = file.into_raw_fd();
    let ret = unsafe { libc::dup2(fd, 2) };
    if ret == -1 {
        unsafe { libc::close(fd) };
        return Err(format!("dup2 failed: {}", std::io::Error::last_os_error()).into());
    }
    let close_ret = unsafe { libc::close(fd) };
    if close_ret == -1 {
        eprintln!("warning: close(fd) after dup2 failed: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
fn redirect_stderr(_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    Err("--stderr-capture is only supported on Unix platforms".into())
}

fn parse_step_timing(
    stderr_capture: &Option<std::path::PathBuf>,
) -> std::collections::HashMap<String, Vec<f64>> {
    let mut timing: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    let Some(path) = stderr_capture else {
        return timing;
    };
    #[cfg(unix)]
    let _ = unsafe { libc::fsync(2) };
    let Ok(content) = std::fs::read_to_string(path) else {
        return timing;
    };
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("[timing:")
            && let Some(close) = rest.find(']')
        {
            let step = &rest[..close];
            let val_str = rest[close + 1..].trim();
            if let Ok(val) = val_str.parse::<f64>() {
                timing.entry(step.to_string()).or_default().push(val);
            }
        }
    }
    timing
}
