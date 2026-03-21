// Integration tests for the operator module.
//
// The `operator`, `CsrOperator`, and `spmv_csr` items are `pub(crate)` and
// therefore not accessible from integration tests. All per-item unit tests live
// inside `src/operator.rs`. The tests here are fixture-gated and will be filled
// in once the public API crystallizes and fixtures are generated.

#[test]
#[ignore = "requires fixture generation: run tests/generate_fixtures.py first"]
fn test_eigenpair_residual_from_fixture() {
    // Load tests/fixtures/blobs_500/comp_b_laplacian.npz and
    // tests/fixtures/blobs_500/comp_d_eigensolver.npz, reconstruct the CSR
    // Laplacian, and assert residual ||L*v - λ*v|| / ||v|| < 1e-10 for each
    // eigenpair in the fixture.
    todo!("implement once fixtures are generated and public API is stable")
}

#[test]
#[ignore = "requires fixture generation: run tests/generate_fixtures.py first"]
fn test_multi_vector_matches_sequential_from_fixture() {
    // Same fixture Laplacian. Apply CsrOperator to the full eigenvector matrix
    // (all k columns at once) and compare to applying column-by-column.
    // Assert all values match within 1e-14.
    todo!("implement once fixtures are generated and public API is stable")
}
