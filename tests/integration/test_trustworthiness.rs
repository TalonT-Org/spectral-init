use spectral_init::trustworthiness;

/// Random data: T must be in [0, 1].
#[test]
fn result_in_unit_interval_random() {
    use rand::{SeedableRng, Rng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let x = ndarray::Array2::from_shape_fn((30, 8), |_| rng.random::<f64>());
    let y = ndarray::Array2::from_shape_fn((30, 2), |_| rng.random::<f64>());
    let t = trustworthiness(x.view(), y.view(), 5);
    assert!(t >= 0.0 && t <= 1.0, "T out of [0,1]: {t}");
    assert!(t.is_finite(), "T is not finite: {t}");
}

/// Identity embedding: T must equal 1.0.
#[test]
fn perfect_preservation_returns_one() {
    let x = ndarray::Array2::from_shape_fn((20, 4), |(i, d)| (i * 4 + d) as f64);
    let y = x.slice(ndarray::s![.., ..2]).to_owned();
    let t = trustworthiness(x.view(), y.view(), 5);
    assert!((t - 1.0).abs() < 1e-10, "perfect preservation: T={t}");
}

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

    let rust_score = trustworthiness(x.view(), y.view(), k);

    assert!(
        (rust_score - sklearn_score).abs() < 1e-6,
        "sklearn parity failed: rust={rust_score:.10}, sklearn={sklearn_score:.10}, diff={:.2e}",
        (rust_score - sklearn_score).abs()
    );
}

/// Fixture bounds: T in [0,1] and finite.
#[test]
#[ignore = "requires fixture; run: python tests/visual_eval/generate_tw_fixture.py"]
fn result_bounds_on_fixture() {
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
    let k = *k_arr.as_slice_memory_order().unwrap().first().unwrap() as usize;

    let t = trustworthiness(x.view(), y.view(), k);
    assert!(t.is_finite(), "T is not finite: {t}");
    assert!(t >= 0.0 && t <= 1.0, "T out of [0,1]: {t}");
}
