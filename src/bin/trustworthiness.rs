//! CLI binary: compute trustworthiness score between high-dim X and embedding Y.
//!
//! Usage:
//!   trustworthiness --x X.npy --y Y.npy [--k 15]
//!
//! Prints a single f64 score to stdout (15 decimal places).
//! Exits 0 on success, 1 on error.

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
    let k: usize = pargs.value_from_str("--k").unwrap_or(15);

    let x: ndarray::Array2<f64> = ndarray_npy::read_npy(&x_path)
        .map_err(|e| format!("failed to load X from {}: {e}", x_path.display()))?;
    let y: ndarray::Array2<f64> = ndarray_npy::read_npy(&y_path)
        .map_err(|e| format!("failed to load Y from {}: {e}", y_path.display()))?;

    let score = spectral_init::metrics::trustworthiness(x.view(), y.view(), k);
    println!("{:.15}", score);
    Ok(())
}
