//! CLI binary: profiling harness for the trustworthiness metric.
//!
//! Usage:
//!   tw_profiler --x X.npy --y Y.npy --output results.json [--k 15] [--iters 5] [--warmup 2] [--stderr-capture path]
//!
//! Runs the trustworthiness function multiple times (after warmup iterations) and
//! writes structured JSON with timing statistics.

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
    let output_path: std::path::PathBuf = pargs.value_from_str("--output")?;
    let k: usize = pargs.opt_value_from_str("--k")?.unwrap_or(15);
    let iters: usize = pargs.opt_value_from_str("--iters")?.unwrap_or(5);
    let warmup: usize = pargs.opt_value_from_str("--warmup")?.unwrap_or(2);
    let stderr_capture: Option<std::path::PathBuf> = pargs.opt_value_from_str("--stderr-capture")?;

    // Set up stderr capture if requested.
    if let Some(ref capture_path) = stderr_capture {
        redirect_stderr(capture_path)?;
    }

    let x: ndarray::Array2<f64> = ndarray_npy::read_npy(&x_path)
        .map_err(|e| format!("failed to load X from {}: {e}", x_path.display()))?;
    let y: ndarray::Array2<f64> = ndarray_npy::read_npy(&y_path)
        .map_err(|e| format!("failed to load Y from {}: {e}", y_path.display()))?;

    // Warmup iterations (results discarded).
    for _ in 0..warmup {
        let _ = std::hint::black_box(spectral_init::trustworthiness(x.view(), y.view(), k));
    }

    // Timed iterations.
    let mut times = Vec::with_capacity(iters);
    let mut score = 0.0f64;
    for _ in 0..iters {
        let start = std::time::Instant::now();
        score = spectral_init::trustworthiness(x.view(), y.view(), k);
        let elapsed = start.elapsed().as_secs_f64();
        times.push(elapsed);
    }

    let n = x.nrows();
    let mean_s = times.iter().sum::<f64>() / times.len() as f64;
    let std_s = if times.len() > 1 {
        let var = times.iter().map(|&t| (t - mean_s).powi(2)).sum::<f64>() / (times.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    // Parse step_timing from captured stderr if available.
    let step_timing = parse_step_timing(&stderr_capture);

    // Build JSON output.
    let mut result = serde_json::Map::new();
    result.insert("n".into(), serde_json::Value::from(n));
    result.insert("k".into(), serde_json::Value::from(k));
    result.insert("iters".into(), serde_json::json!(times));
    result.insert("mean_s".into(), serde_json::json!(round_to(mean_s, 6)));
    result.insert("std_s".into(), serde_json::json!(round_to(std_s, 6)));
    result.insert("warmup".into(), serde_json::Value::from(warmup));
    result.insert("score".into(), serde_json::json!(score));
    if !step_timing.is_empty() {
        result.insert("step_timing".into(), serde_json::json!(step_timing));
    }

    let json = serde_json::to_string_pretty(&serde_json::Value::Object(result))?;
    std::fs::write(&output_path, &json)?;

    Ok(())
}

fn round_to(val: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    (val * factor).round() / factor
}

fn redirect_stderr(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::io::IntoRawFd;
    let file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create stderr capture file {}: {e}", path.display()))?;
    let fd = file.into_raw_fd();
    // SAFETY: dup2 is a POSIX syscall; fd is valid (just opened) and 2 is stderr.
    let ret = unsafe { libc::dup2(fd, 2) };
    if ret == -1 {
        return Err(format!("dup2 failed: {}", std::io::Error::last_os_error()).into());
    }
    // Close the original fd since stderr now owns the file descriptor.
    unsafe { libc::close(fd) };
    Ok(())
}

fn parse_step_timing(
    stderr_capture: &Option<std::path::PathBuf>,
) -> std::collections::HashMap<String, Vec<f64>> {
    let mut timing: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    let Some(path) = stderr_capture else {
        return timing;
    };
    // Flush stderr before reading the capture file.
    let _ = unsafe { libc::fsync(2) };
    let Ok(content) = std::fs::read_to_string(path) else {
        return timing;
    };
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("[timing:") {
            if let Some(close) = rest.find(']') {
                let step = &rest[..close];
                let val_str = rest[close + 1..].trim();
                if let Ok(val) = val_str.parse::<f64>() {
                    timing.entry(step.to_string()).or_default().push(val);
                }
            }
        }
    }
    timing
}
