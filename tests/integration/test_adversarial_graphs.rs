//! Adversarial synthetic graph test suite for the solver escalation chain.
//!
//! All tests are fully synthetic — no .npz fixture files required.
//! This test target requires `--features testing` (for solver-level tests).

use ndarray::{Array1, Array2, Axis, s};
use sprs::CsMatI;
use spectral_init::{spectral_init, SpectralError};

// ─── Graph builder helpers ─────────────────────────────────────────────────────

/// Complete graph K_n: all n*(n-1)/2 pairs connected, weight 1.0.
fn make_complete(n: u32) -> CsMatI<f32, u32, usize> {
    let n_usize = n as usize;
    let mut indptr = vec![0usize; n_usize + 1];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for i in 0..n_usize {
        for j in 0..n_usize {
            if j != i {
                indices.push(j as u32);
                data.push(1.0f32);
            }
        }
        indptr[i + 1] = indices.len();
    }
    CsMatI::<f32, u32, usize>::new((n_usize, n_usize), indptr, indices, data)
}

/// Barbell: K_{size} + bridge edge (weight 1.0) + K_{size}.
/// Nodes 0..size form the first clique; size..2*size form the second.
fn make_barbell(clique_size: u32) -> CsMatI<f32, u32, usize> {
    make_epsilon_bridge(clique_size, 1.0f32)
}

/// Path graph P_n: edges (i, i+1), weight 1.0.
fn make_path(n: u32) -> CsMatI<f32, u32, usize> {
    let n_usize = n as usize;
    let mut indptr = vec![0usize; n_usize + 1];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for i in 0..n_usize {
        if i > 0 {
            indices.push((i - 1) as u32);
            data.push(1.0f32);
        }
        if i + 1 < n_usize {
            indices.push((i + 1) as u32);
            data.push(1.0f32);
        }
        indptr[i + 1] = indices.len();
    }
    CsMatI::<f32, u32, usize>::new((n_usize, n_usize), indptr, indices, data)
}

/// Star S_n: hub node 0, leaf nodes 1..n, edges (0, i) weight 1.0.
fn make_star(n: u32) -> CsMatI<f32, u32, usize> {
    let n_usize = n as usize;
    let mut indptr = vec![0usize; n_usize + 1];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    // Hub row 0: connects to all leaves 1..n
    for j in 1..n_usize {
        indices.push(j as u32);
        data.push(1.0f32);
    }
    indptr[1] = indices.len();
    // Leaf rows: each connects to hub (0)
    for i in 1..n_usize {
        indices.push(0u32);
        data.push(1.0f32);
        indptr[i + 1] = indices.len();
    }
    CsMatI::<f32, u32, usize>::new((n_usize, n_usize), indptr, indices, data)
}

/// Ring C_n: edges (i, (i+1) % n), weight 1.0.
fn make_ring(n: u32) -> CsMatI<f32, u32, usize> {
    let n_usize = n as usize;
    let mut indptr = vec![0usize; n_usize + 1];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for i in 0..n_usize {
        let prev = (i + n_usize - 1) % n_usize;
        let next = (i + 1) % n_usize;
        let (lo, hi) = if prev < next { (prev, next) } else { (next, prev) };
        indices.push(lo as u32);
        data.push(1.0f32);
        indices.push(hi as u32);
        data.push(1.0f32);
        indptr[i + 1] = indices.len();
    }
    CsMatI::<f32, u32, usize>::new((n_usize, n_usize), indptr, indices, data)
}

/// Complete bipartite K_{m,n}: left nodes 0..m, right nodes m..m+n.
fn make_complete_bipartite(m: u32, n: u32) -> CsMatI<f32, u32, usize> {
    let m_usize = m as usize;
    let n_usize = n as usize;
    let total = m_usize + n_usize;
    let mut indptr = vec![0usize; total + 1];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    // Left nodes: connect to all right nodes
    for i in 0..m_usize {
        for j in m_usize..total {
            indices.push(j as u32);
            data.push(1.0f32);
        }
        indptr[i + 1] = indices.len();
    }
    // Right nodes: connect to all left nodes
    for i in m_usize..total {
        for j in 0..m_usize {
            indices.push(j as u32);
            data.push(1.0f32);
        }
        indptr[i + 1] = indices.len();
    }
    CsMatI::<f32, u32, usize>::new((total, total), indptr, indices, data)
}

/// Epsilon-bridge: K_{cluster_size} + one edge of `bridge_weight` + K_{cluster_size}.
/// Node (cluster_size-1) bridges to node cluster_size.
fn make_epsilon_bridge(cluster_size: u32, bridge_weight: f32) -> CsMatI<f32, u32, usize> {
    let cs = cluster_size as usize;
    let n = 2 * cs;
    let mut indptr = vec![0usize; n + 1];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    // First clique (rows 0..cs)
    for i in 0..cs {
        for j in 0..cs {
            if j != i {
                indices.push(j as u32);
                data.push(1.0f32);
            }
        }
        // Bridge: node cs-1 → node cs
        if i == cs - 1 {
            indices.push(cs as u32);
            data.push(bridge_weight);
        }
        indptr[i + 1] = indices.len();
    }
    // Second clique (rows cs..2*cs)
    for i in cs..n {
        // Bridge: node cs → node cs-1 (smallest index in this row, goes first)
        if i == cs {
            indices.push((cs - 1) as u32);
            data.push(bridge_weight);
        }
        for j in cs..n {
            if j != i {
                indices.push(j as u32);
                data.push(1.0f32);
            }
        }
        indptr[i + 1] = indices.len();
    }
    CsMatI::<f32, u32, usize>::new((n, n), indptr, indices, data)
}

/// Path with exponentially decaying weights: edge (i, i+1) has weight 10^(-8*i/(n-1)).
/// Range: 1.0 (first edge) to 1e-8 (last edge).
fn make_weighted_exponential_path(n: u32) -> CsMatI<f32, u32, usize> {
    let n_usize = n as usize;
    let mut indptr = vec![0usize; n_usize + 1];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for i in 0..n_usize {
        if i > 0 {
            let edge_idx = i - 1;
            let w = 10.0_f32.powf(-8.0 * edge_idx as f32 / (n_usize - 1) as f32);
            indices.push((i - 1) as u32);
            data.push(w);
        }
        if i + 1 < n_usize {
            let w = 10.0_f32.powf(-8.0 * i as f32 / (n_usize - 1) as f32);
            indices.push((i + 1) as u32);
            data.push(w);
        }
        indptr[i + 1] = indices.len();
    }
    CsMatI::<f32, u32, usize>::new((n_usize, n_usize), indptr, indices, data)
}

/// Single-edge graph: 2 nodes, 1 undirected edge, weight 1.0.
fn make_single_edge() -> CsMatI<f32, u32, usize> {
    CsMatI::<f32, u32, usize>::new(
        (2, 2),
        vec![0usize, 1, 2],
        vec![1u32, 0u32],
        vec![1.0f32, 1.0f32],
    )
}

/// Lollipop: K_{clique_size} fully connected + path P_{path_len} appended at node clique_size-1.
/// Total nodes: clique_size + path_len.
fn make_lollipop(clique_size: u32, path_len: u32) -> CsMatI<f32, u32, usize> {
    let cs = clique_size as usize;
    let pl = path_len as usize;
    let n = cs + pl;
    let mut indptr = vec![0usize; n + 1];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    // Clique nodes (0..cs)
    for i in 0..cs {
        for j in 0..cs {
            if j != i {
                indices.push(j as u32);
                data.push(1.0f32);
            }
        }
        // Node cs-1 bridges to first path node cs
        if i == cs - 1 {
            indices.push(cs as u32);
            data.push(1.0f32);
        }
        indptr[i + 1] = indices.len();
    }
    // Path nodes (cs..cs+pl): each connects to prev and next
    for i in cs..n {
        if i > 0 {
            indices.push((i - 1) as u32);
            data.push(1.0f32);
        }
        if i + 1 < n {
            indices.push((i + 1) as u32);
            data.push(1.0f32);
        }
        indptr[i + 1] = indices.len();
    }
    CsMatI::<f32, u32, usize>::new((n, n), indptr, indices, data)
}

// ─── Laplacian builder helpers (for solver-level tests) ───────────────────────

/// Well-conditioned diagonal Laplacian with eigenvalues [0, 1/(n-1), 2/(n-1), ..., 1].
/// Use for Level 0, Level 1, and Level 2 tests — LOBPCG converges trivially on this
/// input (eigengap ≈ 1/(n-1) between consecutive eigenvalues, no clustering).
fn large_diagonal_laplacian(n: usize) -> CsMatI<f64, usize> {
    let indptr: Vec<usize> = (0..=n).collect();
    let indices: Vec<usize> = (0..n).collect();
    let data: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1).max(1) as f64)
        .collect();
    CsMatI::new((n, n), indptr, indices, data)
}

/// Normalized Laplacian of P_n (path graph) in f64.
/// L = D^{-1/2}(D - A)D^{-1/2}. Tridiagonal with known closed-form entries:
///   diagonal = 1.0, endpoint off-diagonals = -1/sqrt(2), interior off-diagonals = -0.5.
/// Use ONLY for Level 3 (rSVD) tests — P_n is the worst-case LOBPCG input.
fn large_path_laplacian(n: usize) -> CsMatI<f64, usize> {
    let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt(); // ≈ 0.7071
    let mut indptr = vec![0usize; n + 1];
    let mut indices: Vec<usize> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    for i in 0..n {
        if i > 0 {
            let w = if i == n - 1 { -inv_sqrt2 } else { -0.5 };
            indices.push(i - 1);
            data.push(w);
        }
        indices.push(i);
        data.push(1.0);
        if i + 1 < n {
            let w = if i == 0 { -inv_sqrt2 } else { -0.5 };
            indices.push(i + 1);
            data.push(w);
        }
        indptr[i + 1] = indices.len();
    }
    CsMatI::new((n, n), indptr, indices, data)
}

/// 6-node unnormalized path-graph Laplacian, mirroring `path_graph_laplacian_6` in
/// `src/solvers/mod.rs` unit tests.
///
/// Intentionally unnormalized: `test_level_0_dense_evd_for_small_n` only checks that
/// `n < 2000` routes to Level 0 (dense EVD) — it does not validate eigenvector quality
/// or residuals. The escalation-level selection is purely size-based, so the
/// normalization form of the matrix is irrelevant for that test.
fn path_laplacian_6() -> CsMatI<f64, usize> {
    CsMatI::new(
        (6, 6),
        vec![0usize, 2, 5, 8, 11, 14, 16],
        vec![
            0usize, 1,   // row 0
            0, 1, 2,     // row 1
            1, 2, 3,     // row 2
            2, 3, 4,     // row 3
            3, 4, 5,     // row 4
            4, 5,        // row 5
        ],
        vec![
            1.0_f64, -1.0,       // row 0
            -1.0, 2.0, -1.0,     // row 1
            -1.0, 2.0, -1.0,     // row 2
            -1.0, 2.0, -1.0,     // row 3
            -1.0, 2.0, -1.0,     // row 4
            -1.0, 1.0,           // row 5
        ],
    )
}

// ─── Community separation helper ──────────────────────────────────────────────

/// Returns the range (max − min) of values in column `dim` among rows in `range`.
/// Used to measure intra-cluster spread along a single eigenvector dimension.
fn max_intra_1d_spread(emb: &Array2<f32>, range: std::ops::Range<usize>, dim: usize) -> f32 {
    let vals: Vec<f32> = range.map(|i| emb[[i, dim]]).collect();
    let min_v = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    max_v - min_v
}

// ─── Group 1: Correctness tests (no panic, no NaN/Inf, correct shape) ─────────

#[test]
fn test_barbell_valid_embedding() {
    let graph = make_barbell(20); // n = 40
    let result = spectral_init(&graph, 2, 42, None).expect("barbell should not fail");
    assert_eq!(result.shape(), &[40, 2]);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "barbell embedding contains NaN or Inf"
    );
}

#[test]
fn test_path_valid_embedding() {
    let graph = make_path(100); // n = 100
    let result = spectral_init(&graph, 2, 42, None).expect("path should not fail");
    assert_eq!(result.shape(), &[100, 2]);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "path embedding contains NaN or Inf"
    );
}

#[test]
fn test_star_valid_embedding() {
    let graph = make_star(50); // n = 50
    let result = spectral_init(&graph, 2, 42, None).expect("star should not fail");
    assert_eq!(result.shape(), &[50, 2]);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "star embedding contains NaN or Inf"
    );
}

#[test]
fn test_epsilon_bridge_valid_embedding() {
    let graph = make_epsilon_bridge(20, 1e-6f32); // n = 40
    let result =
        spectral_init(&graph, 2, 42, None).expect("epsilon-bridge should not fail");
    assert_eq!(result.shape(), &[40, 2]);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "epsilon-bridge embedding contains NaN or Inf"
    );
}

#[test]
fn test_complete_bipartite_valid_embedding() {
    let graph = make_complete_bipartite(10, 10); // n = 20
    let result =
        spectral_init(&graph, 2, 42, None).expect("complete bipartite should not fail");
    assert_eq!(result.shape(), &[20, 2]);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "complete bipartite embedding contains NaN or Inf"
    );
}

#[test]
fn test_ring_valid_embedding() {
    let graph = make_ring(100); // n = 100
    let result = spectral_init(&graph, 2, 42, None).expect("ring should not fail");
    assert_eq!(result.shape(), &[100, 2]);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "ring embedding contains NaN or Inf"
    );
}

#[test]
fn test_weighted_exponential_valid_embedding() {
    let graph = make_weighted_exponential_path(100); // n = 100
    let result = spectral_init(&graph, 2, 42, None)
        .expect("weighted exponential path should not fail");
    assert_eq!(result.shape(), &[100, 2]);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "weighted exponential path embedding contains NaN or Inf"
    );
}

#[test]
fn test_single_edge_returns_too_few_nodes() {
    let graph = make_single_edge(); // n = 2, n_components = 2: 2 <= 2 → TooFewNodes
    let result = spectral_init(&graph, 2, 42, None);
    assert!(
        matches!(result, Err(SpectralError::TooFewNodes { n: 2, dims: 2 })),
        "expected TooFewNodes error, got: {:?}",
        result
    );
}

#[test]
fn test_complete_k20_valid_embedding() {
    let graph = make_complete(20); // n = 20
    let result = spectral_init(&graph, 2, 42, None).expect("K_20 should not fail");
    assert_eq!(result.shape(), &[20, 2]);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "K_20 embedding contains NaN or Inf"
    );
}

#[test]
fn test_lollipop_valid_embedding() {
    let graph = make_lollipop(10, 20); // n = 30
    let result = spectral_init(&graph, 2, 42, None).expect("lollipop should not fail");
    assert_eq!(result.shape(), &[30, 2]);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "lollipop embedding contains NaN or Inf"
    );
}

// ─── Group 2: Community separation tests ──────────────────────────────────────

#[test]
fn test_barbell_separates_communities() {
    let graph = make_barbell(20); // n=40; clique_0=nodes[0..20], clique_1=nodes[20..40]
    let emb = spectral_init(&graph, 2, 42, None).unwrap();

    // The Fiedler vector (dim 0) encodes community membership: all nodes in each K_20
    // clique have nearly identical Fiedler values, so separation is measured along dim 0.
    // The second eigenvector (dim 1) lies in the degenerate intra-clique eigenspace and
    // spreads nodes within each clique — comparing 2-D pairwise distances would fail.
    let c0_x = emb.slice(s![0..20, 0]).mean_axis(Axis(0)).unwrap()[[]];
    let c1_x = emb.slice(s![20..40, 0]).mean_axis(Axis(0)).unwrap()[[]];
    let centroid_gap: f32 = (c0_x - c1_x).abs();

    let max_intra = max_intra_1d_spread(&emb, 0..20, 0)
        .max(max_intra_1d_spread(&emb, 20..40, 0));

    assert!(
        centroid_gap > max_intra,
        "barbell cliques not separated in Fiedler direction: \
         centroid_gap={centroid_gap:.3}, max_intra_dim0={max_intra:.3}"
    );
}

#[test]
fn test_epsilon_bridge_separates_communities() {
    let graph = make_epsilon_bridge(20, 1e-6f32); // n=40; same clique layout
    let emb = spectral_init(&graph, 2, 42, None).unwrap();

    // Same rationale as test_barbell_separates_communities: compare along the Fiedler
    // direction (dim 0) where the near-zero bridge weight still produces opposite-sign
    // Fiedler values for the two cliques.
    let c0_x = emb.slice(s![0..20, 0]).mean_axis(Axis(0)).unwrap()[[]];
    let c1_x = emb.slice(s![20..40, 0]).mean_axis(Axis(0)).unwrap()[[]];
    let centroid_gap: f32 = (c0_x - c1_x).abs();

    let max_intra = max_intra_1d_spread(&emb, 0..20, 0)
        .max(max_intra_1d_spread(&emb, 20..40, 0));

    assert!(
        centroid_gap > max_intra,
        "epsilon-bridge cliques not separated in Fiedler direction: \
         centroid_gap={centroid_gap:.3}, max_intra_dim0={max_intra:.3}"
    );
}

// ─── Group 3: Coordinate stability tests ──────────────────────────────────────

#[test]
fn test_star_coordinate_stability() {
    let graph = make_star(50);
    let emb = spectral_init(&graph, 2, 42, None).unwrap();
    let max_abs = emb.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 20.0,
        "star coordinate outlier: max_abs={max_abs:.2} (expected < 20.0)"
    );
}

#[test]
fn test_complete_bipartite_coordinate_stability() {
    let graph = make_complete_bipartite(10, 10);
    let emb = spectral_init(&graph, 2, 42, None).unwrap();
    let max_abs = emb.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 20.0,
        "complete bipartite coordinate outlier: max_abs={max_abs:.2} (expected < 20.0)"
    );
}

// ─── Group 4: Solver level tests (require --features testing) ─────────────────

#[test]
fn test_level_0_dense_evd_for_small_n() {
    let laplacian = path_laplacian_6(); // n=6 < 2000 → Level 0
    let (_, level) = spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
    assert_eq!(level, 0, "n=6 must use Level 0 dense EVD");
}

#[test]
fn test_level_1_lobpcg_for_large_well_conditioned_n() {
    let laplacian = large_diagonal_laplacian(2001); // n=2001 >= 2000, eigengap ≈ 0.5
    let (_, level) = spectral_init::solve_eigenproblem_pub(&laplacian, 2, 42);
    assert_eq!(
        level, 1,
        "n=2001 well-conditioned diagonal must use Level 1 LOBPCG"
    );
}

#[test]
fn test_level_2_regularized_lobpcg_produces_valid_result() {
    let laplacian = large_diagonal_laplacian(2001);
    let op = spectral_init::operator::CsrOperator(&laplacian);

    let sqrt_deg = Array1::ones(2001);
    let r1 = spectral_init::solvers::lobpcg::lobpcg_solve(&op, 2, 42, false, &sqrt_deg);
    assert!(
        r1.is_some(),
        "Level 1 (unregularized) must converge on diagonal n=2001"
    );

    let r2 = spectral_init::solvers::lobpcg::lobpcg_solve(&op, 2, 42, true, &sqrt_deg);
    assert!(
        r2.is_some(),
        "Level 2 (regularized) must converge on diagonal n=2001"
    );
    let (eigs, _) = r2.unwrap();
    assert!(
        eigs.iter().all(|v| v.is_finite()),
        "Level 2 eigenvalues must be finite"
    );
    assert!(
        eigs.iter().all(|&v| v >= -1e-6),
        "Level 2 eigenvalues must be non-negative"
    );
}

#[test]
fn test_level_3_rsvd_valid_on_large_path() {
    let laplacian = large_path_laplacian(2001);
    let (eigs, vecs) = spectral_init::rsvd_solve_pub(&laplacian, 2, 42);
    assert_eq!(eigs.len(), 3, "rsvd_solve_pub must return n_components+1=3 eigenvalues");
    assert!(
        eigs.iter().all(|&v| v >= -1e-3),
        "rSVD eigenvalues must be non-negative"
    );
    assert!(
        vecs.iter().all(|v| v.is_finite()),
        "rSVD eigenvectors contain NaN/Inf"
    );
    assert!(
        eigs[0].abs() < 0.1,
        "first rSVD eigenvalue should be near-zero, got {}",
        eigs[0]
    );
}
