use ndarray::Array2;
use rand::{Rng, SeedableRng};
use std::process::Command;

fn generate_fixtures(dir: &std::path::Path, n: usize, d_x: usize, d_y: usize, seed: u64) {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let x: Array2<f64> = Array2::from_shape_fn((n, d_x), |_| rng.random::<f64>());
    let y: Array2<f64> = Array2::from_shape_fn((n, d_y), |_| rng.random::<f64>());
    ndarray_npy::write_npy(dir.join("x.npy"), &x).expect("write x.npy");
    ndarray_npy::write_npy(dir.join("y.npy"), &y).expect("write y.npy");
}

#[test]
fn t_profiler_01_produces_valid_json() {
    let tmp = std::env::temp_dir().join(format!("tw_profiler_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("create temp dir");

    generate_fixtures(&tmp, 50, 10, 2, 42);

    let output_path = tmp.join("results.json");

    let status = Command::new(env!("CARGO"))
        .args([
            "run",
            "--features",
            "cli",
            "--bin",
            "tw_profiler",
            "--",
            "--x",
        ])
        .arg(tmp.join("x.npy"))
        .arg("--y")
        .arg(tmp.join("y.npy"))
        .arg("--k")
        .arg("5")
        .arg("--iters")
        .arg("2")
        .arg("--warmup")
        .arg("1")
        .arg("--output")
        .arg(&output_path)
        .status()
        .expect("failed to run tw_profiler");

    assert!(status.success(), "tw_profiler exited with {:?}", status);
    assert!(output_path.exists(), "output file not created");

    let json_str = std::fs::read_to_string(&output_path).expect("read results.json");
    let val: serde_json::Value = serde_json::from_str(&json_str).expect("parse JSON");

    assert_eq!(val["n"], 50);
    assert_eq!(val["k"], 5);
    assert_eq!(val["warmup"], 1);

    let iters = val["iters"].as_array().expect("iters should be array");
    assert_eq!(iters.len(), 2);
    for t in iters {
        assert!(t.as_f64().unwrap() > 0.0, "iter time must be > 0");
    }

    let mean = val["mean_s"].as_f64().expect("mean_s");
    assert!(mean > 0.0, "mean_s must be > 0");

    let score = val["score"].as_f64().expect("score");
    assert!(score > 0.0 && score <= 1.0, "score out of (0,1]: {score}");

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn t_profiler_02_stderr_capture_writes_file() {
    let tmp = std::env::temp_dir().join(format!("tw_profiler_stderr_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("create temp dir");

    generate_fixtures(&tmp, 50, 10, 2, 99);

    let output_path = tmp.join("results.json");
    let stderr_path = tmp.join("stderr.txt");

    let status = Command::new(env!("CARGO"))
        .args([
            "run",
            "--features",
            "cli",
            "--bin",
            "tw_profiler",
            "--",
            "--x",
        ])
        .arg(tmp.join("x.npy"))
        .arg("--y")
        .arg(tmp.join("y.npy"))
        .arg("--k")
        .arg("5")
        .arg("--iters")
        .arg("2")
        .arg("--warmup")
        .arg("1")
        .arg("--output")
        .arg(&output_path)
        .arg("--stderr-capture")
        .arg(&stderr_path)
        .status()
        .expect("failed to run tw_profiler");

    assert!(status.success(), "tw_profiler exited with {:?}", status);
    assert!(stderr_path.exists(), "stderr capture file not created");
    let meta = std::fs::metadata(&stderr_path)
        .expect("stderr capture file exists but metadata unreadable");
    assert!(
        meta.is_file(),
        "stderr capture path is not a regular file: {:?}",
        stderr_path
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
