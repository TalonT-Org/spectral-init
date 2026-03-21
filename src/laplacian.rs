use ndarray::Array1;
use sprs::{CsMatI, TriMat};

/// Computes per-node degree and its element-wise square root from a sparse graph.
///
/// Sums each CSR row's f32 weights with f64 accumulation to match Python UMAP's
/// implicit float64 promotion in `graph.sum()`. Returns `(degrees, sqrt_deg)`.
///
/// Zero-degree (isolated) nodes produce `degrees[i] = 0.0` and `sqrt_deg[i] = 0.0`.
pub fn compute_degrees(
    graph: &CsMatI<f32, u32, usize>,
) -> (Array1<f64>, Array1<f64>) {
    let n = graph.rows();
    let mut degrees = Array1::<f64>::zeros(n);

    for (row_idx, row_vec) in graph.outer_iterator().enumerate() {
        degrees[row_idx] = row_vec.data().iter().map(|&w| w as f64).sum();
    }

    let sqrt_deg = degrees.mapv(f64::sqrt);
    (degrees, sqrt_deg)
}

/// Builds the symmetric normalized Laplacian L = I - D^{-1/2} W D^{-1/2} in f64.
///
/// Accepts the fuzzy k-NN graph in f32/u32 format from umap-rs and upcasts weights
/// to f64 for numerical stability. `inv_sqrt_deg[i]` must equal `1/sqrt(degree[i])`
/// for connected nodes and `0.0` for isolated (zero-degree) nodes.
///
/// Returns a symmetric CSR matrix of shape `(n, n)` with:
/// - Diagonal entries exactly 1.0 for all nodes (including isolated ones)
/// - Off-diagonal entries `L[i,j] = -inv_sqrt_deg[i] * W[i,j] * inv_sqrt_deg[j]`
pub fn build_normalized_laplacian(
    graph: &CsMatI<f32, u32, usize>,
    inv_sqrt_deg: &[f64],
) -> CsMatI<f64, usize> {
    let n = graph.rows();
    // Capacity: all off-diagonal nonzeros from W, plus n diagonal entries
    let mut tri: TriMat<f64> = TriMat::with_capacity((n, n), graph.nnz() + n);

    // Off-diagonal entries: L[i,j] = -inv_sqrt_deg[i] * W[i,j] * inv_sqrt_deg[j]
    for (row_idx, row_vec) in graph.outer_iterator().enumerate() {
        for (col_idx, &val) in row_vec.iter() {
            let col = col_idx as usize;
            if row_idx != col {
                tri.add_triplet(
                    row_idx,
                    col,
                    -inv_sqrt_deg[row_idx] * (val as f64) * inv_sqrt_deg[col],
                );
            }
        }
    }

    // Diagonal entries: L[i,i] = 1.0 for all i
    // (holds exactly: k-NN graphs have no self-loops, so D^{-1/2} W D^{-1/2} diagonal is 0)
    for i in 0..n {
        tri.add_triplet(i, i, 1.0);
    }

    tri.to_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::CsMatI;

    #[test]
    fn compute_degrees_zero_degree_node() {
        // 3-node graph: only nodes 0 and 1 are connected
        // node 2 is isolated (zero degree)
        let indptr = vec![0usize, 1, 2, 2];
        let indices = vec![1u32, 0u32];
        let data = vec![2.0f32, 2.0f32];
        let graph = CsMatI::new((3, 3), indptr, indices, data);

        let (degrees, sqrt_deg) = compute_degrees(&graph);

        assert!((degrees[0] - 2.0_f64).abs() < 1e-15);
        assert!((degrees[1] - 2.0_f64).abs() < 1e-15);
        assert!((degrees[2] - 0.0_f64).abs() < 1e-15); // isolated node
        assert!((sqrt_deg[2] - 0.0_f64).abs() < 1e-15); // sqrt(0) = 0, not NaN
    }

    #[test]
    fn build_laplacian_two_node_complete_graph() {
        // 2-node graph: edge (0,1) weight 1.0 and edge (1,0) weight 1.0
        let indptr = vec![0usize, 1, 2];
        let indices = vec![1u32, 0u32];
        let data = vec![1.0f32, 1.0f32];
        let graph = CsMatI::new((2, 2), indptr, indices, data);
        let inv_sqrt_deg = vec![1.0f64, 1.0f64]; // sqrt_deg = [1.0, 1.0]

        let l = build_normalized_laplacian(&graph, &inv_sqrt_deg);

        // Diagonal
        assert!((l.get(0, 0).copied().unwrap_or(0.0) - 1.0).abs() < 1e-15);
        assert!((l.get(1, 1).copied().unwrap_or(0.0) - 1.0).abs() < 1e-15);
        // Off-diagonal
        assert!((l.get(0, 1).copied().unwrap_or(0.0) - (-1.0)).abs() < 1e-15);
        assert!((l.get(1, 0).copied().unwrap_or(0.0) - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn build_laplacian_weighted_edge() {
        // 2-node graph with weight 4.0 on both edges
        // sqrt_deg = [2.0, 2.0], inv_sqrt_deg = [0.5, 0.5]
        let indptr = vec![0usize, 1, 2];
        let indices = vec![1u32, 0u32];
        let data = vec![4.0f32, 4.0f32];
        let graph = CsMatI::new((2, 2), indptr, indices, data);
        let inv_sqrt_deg = vec![0.5f64, 0.5f64];

        let l = build_normalized_laplacian(&graph, &inv_sqrt_deg);

        // L[0,1] = -0.5 * 4.0 * 0.5 = -1.0
        assert!((l.get(0, 1).copied().unwrap_or(0.0) - (-1.0)).abs() < 1e-15);
        assert!((l.get(1, 0).copied().unwrap_or(0.0) - (-1.0)).abs() < 1e-15);
        // Diagonal must be 1.0
        assert!((l.get(0, 0).copied().unwrap_or(0.0) - 1.0).abs() < 1e-15);
        assert!((l.get(1, 1).copied().unwrap_or(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn build_laplacian_isolated_node() {
        // 3-node graph: nodes 0 and 1 connected (weight 1.0), node 2 isolated
        // inv_sqrt_deg = [1.0, 1.0, 0.0]
        let indptr = vec![0usize, 1, 2, 2];
        let indices = vec![1u32, 0u32];
        let data = vec![1.0f32, 1.0f32];
        let graph = CsMatI::new((3, 3), indptr, indices, data);
        let inv_sqrt_deg = vec![1.0f64, 1.0f64, 0.0f64];

        let l = build_normalized_laplacian(&graph, &inv_sqrt_deg);

        // Diagonal is always 1.0 including isolated node
        assert!((l.get(2, 2).copied().unwrap_or(0.0) - 1.0).abs() < 1e-15);
        // No off-diagonal entries for row/col 2 (inv_sqrt_deg[2] = 0)
        assert!((l.get(0, 2).copied().unwrap_or(0.0)).abs() < 1e-15);
        assert!((l.get(2, 0).copied().unwrap_or(0.0)).abs() < 1e-15);
        assert!((l.get(1, 2).copied().unwrap_or(0.0)).abs() < 1e-15);
        assert!((l.get(2, 1).copied().unwrap_or(0.0)).abs() < 1e-15);
        // Connected edge: L[0,1] = L[1,0] = -1.0
        assert!((l.get(0, 1).copied().unwrap_or(0.0) - (-1.0)).abs() < 1e-15);
        assert!((l.get(1, 0).copied().unwrap_or(0.0) - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn build_laplacian_symmetry() {
        // 4-node path graph (0-1-2-3) with equal weights
        // Edges: (0,1), (1,0), (1,2), (2,1), (2,3), (3,2)
        let indptr = vec![0usize, 1, 3, 5, 6];
        let indices = vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32];
        let data = vec![1.0f32; 6];
        let graph = CsMatI::new((4, 4), indptr, indices, data);
        // Node 0 and 3 have degree 1, nodes 1 and 2 have degree 2
        // inv_sqrt_deg[0] = 1/sqrt(1) = 1.0
        // inv_sqrt_deg[1] = 1/sqrt(2)
        // inv_sqrt_deg[2] = 1/sqrt(2)
        // inv_sqrt_deg[3] = 1/sqrt(1) = 1.0
        let inv_sqrt_deg = vec![
            1.0f64,
            1.0 / 2.0f64.sqrt(),
            1.0 / 2.0f64.sqrt(),
            1.0f64,
        ];

        let l = build_normalized_laplacian(&graph, &inv_sqrt_deg);

        // All diagonal entries must be exactly 1.0
        for i in 0..4 {
            let d = l.get(i, i).copied().unwrap_or(0.0);
            assert!(
                (d - 1.0).abs() < 1e-15,
                "L[{i},{i}] = {d}, expected 1.0"
            );
        }

        // Symmetry: |L[i,j] - L[j,i]| < 1e-15 for all i, j
        for i in 0..4 {
            for j in 0..4 {
                let lij = l.get(i, j).copied().unwrap_or(0.0);
                let lji = l.get(j, i).copied().unwrap_or(0.0);
                assert!(
                    (lij - lji).abs() < 1e-15,
                    "symmetry violation: L[{i},{j}]={lij}, L[{j},{i}]={lji}"
                );
            }
        }
    }
}
