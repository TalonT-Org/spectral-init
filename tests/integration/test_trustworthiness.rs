use spectral_init::{
    trustworthiness,
    trustworthiness_thread_local,
    trustworthiness_partial_rank,
    trustworthiness_avx2_kernel,
    trustworthiness_combined,
};

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
    let sklearn_score_arr: ndarray::Array0<f64> =
        npz.by_name("sklearn_score").expect("missing sklearn_score in fixture");

    let k = *k_arr.as_slice_memory_order().unwrap().first().unwrap() as usize;
    let sklearn_score = *sklearn_score_arr.as_slice_memory_order().unwrap().first().unwrap();

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

/// Helper: load fixture and return (x, y, k, sklearn_score).
fn load_tw_fixture() -> (ndarray::Array2<f64>, ndarray::Array2<f64>, usize, f64) {
    use ndarray_npy::NpzReader;
    use std::fs::File;
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tw_parity/tw_parity.npz");
    let f = File::open(&fixture_path)
        .unwrap_or_else(|e| panic!("could not open fixture {}: {e}", fixture_path.display()));
    let mut npz = NpzReader::new(f).expect("failed to read .npz");
    let x: ndarray::Array2<f64> = npz.by_name("X").expect("missing X");
    let y: ndarray::Array2<f64> = npz.by_name("Y").expect("missing Y");
    let k_arr: ndarray::Array0<i64> = npz.by_name("k").expect("missing k");
    let score_arr: ndarray::Array0<f64> = npz.by_name("sklearn_score").expect("missing sklearn_score");
    let k = *k_arr.as_slice_memory_order().unwrap().first().unwrap() as usize;
    let sklearn_score = *score_arr.as_slice_memory_order().unwrap().first().unwrap();
    (x, y, k, sklearn_score)
}

#[test]
#[ignore = "requires fixture; run: python tests/visual_eval/generate_tw_fixture.py"]
fn sklearn_parity_thread_local() {
    let (x, y, k, sklearn_score) = load_tw_fixture();
    let rust_score = trustworthiness_thread_local(x.view(), y.view(), k);
    assert!(
        (rust_score - sklearn_score).abs() < 1e-6,
        "thread_local parity failed: rust={rust_score:.10}, sklearn={sklearn_score:.10}, diff={:.2e}",
        (rust_score - sklearn_score).abs()
    );
}

#[test]
#[ignore = "requires fixture; run: python tests/visual_eval/generate_tw_fixture.py"]
fn sklearn_parity_partial_rank() {
    let (x, y, k, sklearn_score) = load_tw_fixture();
    let rust_score = trustworthiness_partial_rank(x.view(), y.view(), k);
    assert!(
        (rust_score - sklearn_score).abs() < 1e-6,
        "partial_rank parity failed: rust={rust_score:.10}, sklearn={sklearn_score:.10}, diff={:.2e}",
        (rust_score - sklearn_score).abs()
    );
}

#[test]
#[ignore = "requires fixture; run: python tests/visual_eval/generate_tw_fixture.py"]
fn sklearn_parity_avx2_kernel() {
    let (x, y, k, sklearn_score) = load_tw_fixture();
    #[cfg(not(target_arch = "x86_64"))]
    {
        eprintln!("sklearn_parity_avx2_kernel: skipping on non-x86_64 host");
        return;
    }
    let rust_score = trustworthiness_avx2_kernel(x.view(), y.view(), k);
    assert!(
        (rust_score - sklearn_score).abs() < 1e-6,
        "avx2_kernel parity failed: rust={rust_score:.10}, sklearn={sklearn_score:.10}, diff={:.2e}",
        (rust_score - sklearn_score).abs()
    );
}

#[test]
#[ignore = "requires fixture; run: python tests/visual_eval/generate_tw_fixture.py"]
fn sklearn_parity_combined() {
    let (x, y, k, sklearn_score) = load_tw_fixture();
    let rust_score = trustworthiness_combined(x.view(), y.view(), k);
    assert!(
        (rust_score - sklearn_score).abs() < 1e-6,
        "combined parity failed: rust={rust_score:.10}, sklearn={sklearn_score:.10}, diff={:.2e}",
        (rust_score - sklearn_score).abs()
    );
}
