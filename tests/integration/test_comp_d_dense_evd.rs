#[path = "../common/mod.rs"]
mod common;

use spectral_init::dense_evd;
use ndarray::Array1;

fn residual_spmv(
    laplacian: &sprs::CsMat<f64>,
    eigvec: ndarray::ArrayView1<f64>,
    eigval: f64,
) -> f64 {
    let n = laplacian.rows();
    let mut lv = vec![0.0f64; n];
    for (val, (row, col)) in laplacian.iter() {
        lv[row] += val * eigvec[col];
    }
    let diff_norm: f64 = lv.iter().zip(eigvec.iter())
        .map(|(&lvi, &vi)| (lvi - eigval * vi).powi(2))
        .sum::<f64>()
        .sqrt();
    let v_norm: f64 = eigvec.iter().map(|&vi| vi.powi(2)).sum::<f64>().sqrt();
    diff_norm / v_norm
}

fn run_comp_d_test(dataset: &str, expected_n: usize) {
    let base = common::fixture_path(dataset, "");

    // Load Laplacian from comp_b output
    let laplacian = common::load_sparse_csr(&base.join("comp_b_laplacian.npz"));
    assert_eq!(laplacian.rows(), expected_n);

    // Load eigensolver reference fixture
    let ref_eigenvalues: Array1<f64> =
        common::load_dense(&base.join("comp_d_eigensolver.npz"), "eigenvalues");
    let ref_eigenvectors: ndarray::Array2<f64> =
        common::load_dense(&base.join("comp_d_eigensolver.npz"), "eigenvectors");
    let eigenvalue_gaps: Array1<f64> =
        common::load_dense(&base.join("comp_d_eigensolver.npz"), "eigenvalue_gaps");
    let k_arr: Array1<i32> =
        common::load_dense(&base.join("comp_d_eigensolver.npz"), "k");
    let k = k_arr[0] as usize;

    // Call dense_evd
    let (eigenvalues, eigenvectors) = dense_evd(&laplacian, k)
        .expect("dense_evd failed");

    // Eigenvalue accuracy check
    for i in 0..k {
        let diff = (eigenvalues[i] - ref_eigenvalues[i]).abs();
        assert!(
            diff < 1e-8,
            "eigenvalue[{i}]: got {}, ref {}, diff {}",
            eigenvalues[i], ref_eigenvalues[i], diff
        );
    }

    // Eigenvector residual check (sign-independent)
    for j in 0..k {
        let r = residual_spmv(&laplacian, eigenvectors.column(j), eigenvalues[j]);
        assert!(
            r < 1e-10,
            "eigenvector residual[{j}] = {r} >= 1e-10"
        );
    }

    // Near-degenerate subspace check
    for j in 0..eigenvalue_gaps.len() {
        if eigenvalue_gaps[j] < 1e-6 {
            let u0 = eigenvectors.column(j);
            let u1 = eigenvectors.column(j + 1);
            let r0 = ref_eigenvectors.column(j);
            let r1 = ref_eigenvectors.column(j + 1);

            let a = u0.dot(&r0);
            let b = u0.dot(&r1);
            let c = u1.dot(&r0);
            let d = u1.dot(&r1);
            let det = (a * d - b * c).abs();
            assert!(
                det > 0.99,
                "subspace mismatch for near-degenerate pair at index {j}: det = {det}"
            );
        }
    }
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_d_dense_evd_blobs_connected_200() {
    run_comp_d_test("blobs_connected_200", 200);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_d_dense_evd_moons_200() {
    run_comp_d_test("moons_200", 200);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_d_dense_evd_near_dupes_100() {
    run_comp_d_test("near_dupes_100", 100);
}

#[test]
#[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
fn comp_d_dense_evd_circles_300() {
    run_comp_d_test("circles_300", 300);
}
