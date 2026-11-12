//! Self-loops: diagonal graph weights take part in both the degree and the adjacency,
//! as in umap-learn. Checked against the Python reference in `tests/fixtures/selfloop`.

use crate::{diagnostics, fixtures, internals};
use ndarray::{Array1, Array2};
use spectral_init::__internal::{build_normalized_laplacian, compute_degrees, dense_evd};
use spectral_init::sprs::CsMatI;
use spectral_init::{ComputeMode, SpectralInitConfig, spectral_init};

const DATASET: &str = "selfloop";

fn graph() -> CsMatI<f32, u32, usize> {
    let graph = fixtures::load_sparse_csr_f32_u32(&fixtures::fixture_path(DATASET, "graph.npz"));
    let self_loops = (0..graph.rows())
        .filter(|&i| graph.get(i, i).is_some_and(|&w| w > 0.0))
        .count();
    assert!(self_loops >= 3, "the fixture graph must carry self-loops");
    graph
}

/// Degrees and Laplacian of the fixture graph, as the pipeline computes them.
fn degrees_and_laplacian(mode: ComputeMode) -> (Array1<f64>, CsMatI<f64, usize>) {
    let graph = graph();
    let (degrees, sqrt_deg) = compute_degrees(graph.view(), mode);
    let inv_sqrt_deg: Vec<f64> = sqrt_deg.iter().map(|&s| 1.0 / s).collect();
    (
        degrees,
        build_normalized_laplacian(graph.view(), &inv_sqrt_deg),
    )
}

#[test]
fn degrees_include_the_diagonal_and_match_python() {
    let (degrees, _) = degrees_and_laplacian(ComputeMode::PythonCompat);
    let reference: Array1<f64> = fixtures::load_dense(
        &fixtures::fixture_path(DATASET, "comp_a_degrees.npz"),
        "degrees",
    );
    assert_eq!(degrees.len(), reference.len());
    for (i, (&got, &want)) in degrees.iter().zip(reference.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-15 * want.abs().max(1.0),
            "degrees[{i}]: got {got}, want {want}"
        );
    }

    // Each degree is the full row sum, diagonal included.
    let graph = graph();
    for (i, row) in graph.outer_iterator().enumerate() {
        let row_sum: f64 = row.data().iter().map(|&w| f64::from(w)).sum();
        assert!((degrees[i] - row_sum).abs() < 1e-6, "row {i}");
    }
}

#[test]
fn laplacian_matches_python() {
    let (_, laplacian) = degrees_and_laplacian(ComputeMode::PythonCompat);
    let reference =
        fixtures::load_sparse_csr(&fixtures::fixture_path(DATASET, "comp_b_laplacian.npz"));
    let n = laplacian.rows();
    assert_eq!(reference.shape(), (n, n));
    for i in 0..n {
        for j in 0..n {
            let got = laplacian.get(i, j).copied().unwrap_or(0.0);
            let want = reference.get(i, j).copied().unwrap_or(0.0);
            assert!(
                (got - want).abs() < 1e-14,
                "L[{i},{j}]: got {got}, want {want}"
            );
        }
    }
}

#[test]
fn laplacian_equals_the_dense_formula() {
    for mode in [ComputeMode::PythonCompat, ComputeMode::RustNative] {
        let graph = graph();
        let n = graph.rows();
        let (degrees, laplacian) = degrees_and_laplacian(mode);

        // L = I − D^{-1/2} A D^{-1/2}, evaluated densely with the diagonal of A kept.
        let mut expected = Array2::<f64>::eye(n);
        for (&weight, (i, j)) in graph.iter() {
            let (i, j) = (i as usize, j as usize);
            expected[[i, j]] -= f64::from(weight) / (degrees[i].sqrt() * degrees[j].sqrt());
        }
        for i in 0..n {
            assert!(
                laplacian.get(i, i).is_some(),
                "diagonal entry {i} is not stored"
            );
            for j in 0..n {
                let got = laplacian.get(i, j).copied().unwrap_or(0.0);
                assert!(
                    (got - expected[[i, j]]).abs() < 1e-14,
                    "{mode:?} L[{i},{j}]: got {got}, want {}",
                    expected[[i, j]]
                );
            }
        }
    }
}

#[test]
fn sqrt_degree_is_the_null_vector() {
    // `PythonCompat` accumulates degrees in single precision, as umap-learn does, so its
    // null vector is exact only to f32 rounding; `RustNative` sums in double precision.
    for (mode, tolerance) in [
        (ComputeMode::PythonCompat, 1e-6),
        (ComputeMode::RustNative, 1e-14),
    ] {
        let (degrees, laplacian) = degrees_and_laplacian(mode);
        let sqrt_deg = degrees.mapv(f64::sqrt);
        let residual = &laplacian * &sqrt_deg;
        let norm = residual.dot(&residual).sqrt();
        assert!(norm < tolerance, "{mode:?}: ‖L·√d‖ = {norm:e}");
    }
}

#[test]
fn spectrum_lies_in_zero_two_and_matches_python() {
    let (_, laplacian) = degrees_and_laplacian(ComputeMode::PythonCompat);
    let n = laplacian.rows();
    let (eigenvalues, eigenvectors) = dense_evd(&laplacian, n, u64::MAX).expect("dense EVD");
    // Single-precision degrees perturb the spectrum by about one f32 rounding error.
    for (i, &lambda) in eigenvalues.iter().enumerate() {
        assert!(
            (-1e-6..=2.0 + 1e-6).contains(&lambda),
            "eigenvalue[{i}] = {lambda} outside [0, 2]"
        );
    }
    assert!(
        eigenvalues[0].abs() < 1e-6,
        "smallest eigenvalue {} is not 0",
        eigenvalues[0]
    );

    let reference_path = fixtures::fixture_path(DATASET, "comp_d_eigensolver.npz");
    let reference: Array1<f64> = fixtures::load_dense(&reference_path, "eigenvalues");
    for (i, &want) in reference.iter().enumerate() {
        assert!(
            (eigenvalues[i] - want).abs() < 1e-7,
            "eigenvalue[{i}]: got {}, want {want}",
            eigenvalues[i]
        );
    }
    for i in 0..reference.len() {
        let residual = internals::residual_spmv(&laplacian, eigenvectors.column(i), eigenvalues[i]);
        assert!(residual < 1e-12, "residual[{i}] = {residual:e}");
    }
}

#[test]
fn spectral_init_spans_the_python_embedding() {
    let graph = graph();
    let reference: Array2<f64> = fixtures::load_dense(
        &fixtures::fixture_path(DATASET, "full_spectral.npz"),
        "embedding",
    );
    for mode in [ComputeMode::PythonCompat, ComputeMode::RustNative] {
        let config = SpectralInitConfig::new(2)
            .with_seed(42)
            .with_compute_mode(mode);
        let coords = spectral_init(graph.view(), None, &config)
            .expect("spectral_init")
            .mapv(f64::from);
        assert_eq!(coords.dim(), reference.dim());
        let gram_det = diagnostics::subspace_gram_det(coords.view(), reference.view());
        assert!(
            gram_det > 0.999,
            "{mode:?}: subspace Gram determinant {gram_det}"
        );
    }
}
