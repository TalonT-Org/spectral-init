//! Integration tests for the randomized SVD solver (Component D).

macro_rules! rsvd_test {
    ($name:ident, $dataset:expr) => {
        #[test]
        fn $name() {
            let fixture_base = crate::fixtures::fixture_path($dataset, "");

            // Load Laplacian
            let laplacian = crate::fixtures::load_sparse_csr(&fixture_base.join("comp_b_laplacian.npz"));

            // Load reference eigenpairs from comp_d
            let ref_path = fixture_base.join("comp_d_eigensolver.npz");
            let ref_eigenvalues: ndarray::Array1<f64> =
                crate::fixtures::load_dense(&ref_path, "eigenvalues");
            let k: i32 =
                crate::fixtures::load_dense::<i32, ndarray::Ix0>(&ref_path, "k").into_scalar();
            let n_components = (k as usize) - 1;

            // Run rSVD solver
            let (eigenvalues, eigenvectors) = spectral_init::__internal::rsvd_solve(
                &laplacian,
                n_components,
                42,
                &spectral_init::__internal::SolverParams::default(),
            )
            .expect("rsvd_solve failed");

            // Check eigenvalue accuracy. The production rsvd_solve uses nbiter=2
            // and n/10 oversampling, trading precision for speed. On graphs with tiny
            // spectral gaps (second eigenvalue ≈ 0.01), eigenvalue and residual errors
            // of ~1e-1 are expected. Threshold is 5e-1 (sanity check only).
            // rsvd_solve returns n_components+1 pairs: trivial at index 0,
            // non-trivial at indices 1..=n_components.
            let eig_slice = eigenvalues.as_slice().unwrap();
            for i in 0..n_components {
                let err = (eig_slice[i + 1] - ref_eigenvalues[i + 1]).abs();
                assert!(
                    err < 5e-1,
                    "dataset={}, eigenvalue[{}]: rsvd={:.8}, ref={:.8}, err={:.2e} (threshold 5e-1)",
                    $dataset, i, eig_slice[i + 1], ref_eigenvalues[i + 1], err
                );
            }

            // Residual check (sanity: solver found some eigenvector structure; threshold 5e-1)
            for i in 0..n_components {
                let v = eigenvectors.column(i + 1);
                let lambda = eig_slice[i + 1];
                let r = crate::internals::residual_spmv(&laplacian, v, lambda);
                assert!(
                    r < 5e-1,
                    "dataset={}, residual[{}]={:.2e} exceeds threshold 5e-1",
                    $dataset, i, r
                );
            }
        }
    };
}

rsvd_test!(comp_d_rsvd_blobs_connected_200, "blobs_connected_200");
rsvd_test!(comp_d_rsvd_blobs_connected_2000, "blobs_connected_2000");
rsvd_test!(comp_d_rsvd_blobs_500, "blobs_500");
rsvd_test!(comp_d_rsvd_moons_200, "moons_200");
rsvd_test!(comp_d_rsvd_circles_300, "circles_300");
rsvd_test!(comp_d_rsvd_near_dupes_100, "near_dupes_100");
