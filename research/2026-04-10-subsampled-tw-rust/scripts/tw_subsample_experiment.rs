//! Experiment binary: subsampled trustworthiness / Rust vs Python tradeoff study.
//!
//! Usage:
//!   tw_subsample_experiment --preflight --data-dir <path>
//!   tw_subsample_experiment [other flags to be added in groupB]
//!
//! --preflight: verify fixture files exist with correct shapes and dtype, then exit.

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut pargs = pico_args::Arguments::from_env();

    let preflight: bool = pargs.contains("--preflight");
    let data_dir: Option<std::path::PathBuf> = pargs.opt_value_from_str("--data-dir")?;

    if preflight {
        let dir = data_dir.ok_or("--preflight requires --data-dir")?;
        run_preflight(&dir)?;
        return Ok(());
    }

    // GroupB will implement the full experiment here.
    println!("tw_subsample_experiment: use --preflight to verify fixtures. Full experiment TBD.");
    Ok(())
}

/// Verify all four MERFISH fixture files exist, load as f64, and check shapes.
fn run_preflight(data_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
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
