#[path = "../common/mod.rs"]
mod common;

/// Compute residual ||L·v - λ·v|| / ||v|| for a single eigenpair.
fn residual(laplacian: &sprs::CsMat<f64>, eigvec: ndarray::ArrayView1<f64>, eigval: f64) -> f64 {
    let lv = laplacian * &eigvec.to_owned();  // SpMV via sprs
    let diff = lv - eigval * &eigvec.to_owned();
    diff.dot(&diff).sqrt() / eigvec.dot(&eigvec).sqrt()
}

macro_rules! rsvd_test {
    ($name:ident, $dataset:expr) => {
        #[test]
        #[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
        fn $name() {
            let fixture_base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join($dataset);

            // Load Laplacian
            let laplacian = common::load_sparse_csr(&fixture_base.join("comp_b_laplacian.npz"));

            // Load reference eigenpairs from comp_d
            let mut d = ndarray_npy::NpzReader::new(
                std::fs::File::open(fixture_base.join("comp_d_eigensolver.npz")).unwrap(),
            )
            .unwrap();
            let ref_eigenvalues: ndarray::Array1<f64> = d.by_name("eigenvalues").unwrap();
            let k: i32 = d
                .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix0>("k")
                .unwrap()
                .into_scalar();
            let n_components = (k as usize) - 1;

            // Run rSVD solver
            let (eigenvalues, eigenvectors) =
                spectral_init::rsvd_solve_pub(&laplacian, n_components, 42);

            // Check eigenvalue accuracy
            let eig_slice = eigenvalues.as_slice().unwrap();
            for i in 0..n_components {
                let err = (eig_slice[i] - ref_eigenvalues[i + 1]).abs();
                assert!(
                    err < 1e-6,
                    "dataset={}, eigenvalue[{}]: rsvd={:.8}, ref={:.8}, err={:.2e} (threshold 1e-6)",
                    $dataset, i, eig_slice[i], ref_eigenvalues[i + 1], err
                );
            }

            // Check residuals
            for i in 0..n_components {
                let v = eigenvectors.column(i);
                let lambda = eig_slice[i];
                let r = residual(&laplacian, v, lambda);
                assert!(
                    r < 1e-3,
                    "dataset={}, residual[{}]={:.2e} exceeds threshold 1e-3",
                    $dataset, i, r
                );
            }
        }
    };
}

rsvd_test!(comp_11_rsvd_blobs_connected_200,  "blobs_connected_200");
rsvd_test!(comp_11_rsvd_blobs_connected_2000, "blobs_connected_2000");
rsvd_test!(comp_11_rsvd_blobs_500,            "blobs_500");
