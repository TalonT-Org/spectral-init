use spectral_init::trustworthiness;

/// Sklearn parity: load fixture and assert |rust − sklearn| < 1e-6.
/// Requires: python tests/visual_eval/generate_tw_fixture.py
#[test]
#[ignore = "requires fixture; run: python tests/visual_eval/generate_tw_fixture.py"]
fn sklearn_parity_synthetic() {
    use ndarray_npy::NpzReader;
    use std::fs::File;

    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tw_parity/tw_parity.npz");
    let f = File::open(&fixture_path)
        .unwrap_or_else(|e| panic!("could not open fixture {}: {e}", fixture_path.display()));
    let mut npz = NpzReader::new(f).expect("failed to read .npz");

    let x: ndarray::Array2<f64> = npz.by_name("X").expect("missing X in fixture");
    let y: ndarray::Array2<f64> = npz.by_name("Y").expect("missing Y in fixture");
    let k_arr: ndarray::Array0<i64> = npz.by_name("k").expect("missing k in fixture");
    let sklearn_score_arr: ndarray::Array0<f64> = npz
        .by_name("sklearn_score")
        .expect("missing sklearn_score in fixture");

    let k = *k_arr.as_slice_memory_order().unwrap().first().unwrap() as usize;
    let sklearn_score = *sklearn_score_arr
        .as_slice_memory_order()
        .unwrap()
        .first()
        .unwrap();

    assert!(
        sklearn_score > 0.0 && sklearn_score <= 1.0,
        "fixture sklearn_score out of plausible range: {sklearn_score} (possible corrupt fixture)"
    );

    let rust_score = trustworthiness(x.view(), y.view(), k);

    assert!(
        (rust_score - sklearn_score).abs() < 1e-6,
        "sklearn parity failed: rust={rust_score:.10}, sklearn={sklearn_score:.10}, diff={:.2e}",
        (rust_score - sklearn_score).abs()
    );
}

/// Sklearn parity at d_x=50: correctness gate for the x-dist SIMD experiment.
/// Requires: python research/2026-04-08-x-dist-simd-avx512/scripts/gen_tw_parity_50d.py
/// followed by: cp research/.../data/tw_parity_50d.npz tests/fixtures/tw_parity/tw_parity_50d.npz
#[test]
#[ignore = "requires fixture; run gen_tw_parity_50d.py then copy to tests/fixtures/tw_parity/"]
fn sklearn_parity_50d() {
    use ndarray_npy::NpzReader;
    use std::fs::File;

    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tw_parity/tw_parity_50d.npz");
    let f = File::open(&fixture_path)
        .unwrap_or_else(|e| panic!("could not open fixture {}: {e}", fixture_path.display()));
    let mut npz = NpzReader::new(f).expect("failed to read .npz");

    let x: ndarray::Array2<f64> = npz.by_name("X").expect("missing X in fixture");
    let y: ndarray::Array2<f64> = npz.by_name("Y").expect("missing Y in fixture");
    let k_arr: ndarray::Array0<i64> = npz.by_name("k").expect("missing k in fixture");
    let sklearn_score_arr: ndarray::Array0<f64> = npz
        .by_name("sklearn_score")
        .expect("missing sklearn_score in fixture");

    let k = *k_arr.as_slice_memory_order().unwrap().first().unwrap() as usize;
    let sklearn_score = *sklearn_score_arr
        .as_slice_memory_order()
        .unwrap()
        .first()
        .unwrap();

    assert!(
        sklearn_score > 0.0 && sklearn_score <= 1.0,
        "fixture sklearn_score out of plausible range: {sklearn_score} (possible corrupt fixture)"
    );

    let rust_score = trustworthiness(x.view(), y.view(), k);

    assert!(
        (rust_score - sklearn_score).abs() < 1e-6,
        "sklearn parity failed: rust={rust_score:.10}, sklearn={sklearn_score:.10}, diff={:.2e}",
        (rust_score - sklearn_score).abs()
    );
}

/// Records baseline correctness result to research/.../results/correctness.json.
/// Run after sklearn_parity_50d passes. Appends one newline-delimited JSON entry.
#[test]
#[ignore = "run after sklearn_parity_50d passes; writes to research/.../results/correctness.json"]
fn record_baseline_correctness() {
    use ndarray_npy::NpzReader;
    use std::fs::{File, OpenOptions};
    use std::io::Write;

    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tw_parity/tw_parity_50d.npz");
    let f = File::open(&fixture_path)
        .unwrap_or_else(|e| panic!("could not open fixture {}: {e}", fixture_path.display()));
    let mut npz = NpzReader::new(f).expect("failed to read .npz");

    let x: ndarray::Array2<f64> = npz.by_name("X").expect("missing X in fixture");
    let y: ndarray::Array2<f64> = npz.by_name("Y").expect("missing Y in fixture");
    let k_arr: ndarray::Array0<i64> = npz.by_name("k").expect("missing k in fixture");
    let sklearn_score_arr: ndarray::Array0<f64> = npz
        .by_name("sklearn_score")
        .expect("missing sklearn_score in fixture");

    let k = *k_arr.as_slice_memory_order().unwrap().first().unwrap() as usize;
    let sklearn_score = *sklearn_score_arr
        .as_slice_memory_order()
        .unwrap()
        .first()
        .unwrap();

    let rust_score = trustworthiness(x.view(), y.view(), k);
    let delta = (rust_score - sklearn_score).abs();
    let passed = delta < 1e-6;

    let record = format!(
        r#"{{"variant":"baseline","rust_score":{rust_score:.15},"sklearn_score":{sklearn_score:.15},"delta":{delta:.2e},"passed":{passed}}}"#
    );

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("research/2026-04-08-x-dist-simd-avx512/results/correctness.json");
    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .unwrap_or_else(|e| panic!("cannot open {}: {e}", out_path.display()));
    writeln!(out, "{record}").expect("failed to write correctness record");

    println!("Recorded: {record}");
    assert!(passed, "baseline failed correctness gate: delta={delta:.2e}");
}
