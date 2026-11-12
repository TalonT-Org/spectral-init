use crate::ComputeMode;
use ndarray::Array1;
use sprs::{CsMatI, CsMatViewI, TriMat};

/// f32 column-sum scatter-add: iterates nonzeros and accumulates deg_f32[col] += val in f32,
/// then returns the raw f32 accumulator. #[inline(never)] prevents LLVM from reordering
/// f32 additions across inlining boundaries, ensuring platform-consistent order.
#[inline(never)]
pub(crate) fn column_sum_f32(graph: CsMatViewI<'_, f32, u32, usize>) -> Vec<f32> {
    let n = graph.cols();
    let mut deg_f32 = vec![0.0f32; n];
    for (val, (_row, col)) in graph.iter() {
        deg_f32[col as usize] += val;
    }
    deg_f32
}

/// Computes per-node degree and its element-wise square root from a sparse graph.
///
/// `mode` selects the accumulation algorithm:
/// - `ComputeMode::PythonCompat` (default): f32 column-sum scatter-add that matches scipy's
///   `csc_matvec` behavior — iterates nonzeros and accumulates `deg_f32[col] += val` in f32
///   before widening to f64. Produces bit-for-bit matching results against the Python reference.
/// - `ComputeMode::RustNative`: f64 row-sum — iterates CSR rows and accumulates f32 weights
///   with f64 accumulation. Numerically superior but may differ from Python by ~1 ULP.
///
/// Diagonal weights (self-loops) count towards the degree; stored zeros add nothing.
///
/// Returns `(degrees, sqrt_deg)`. Zero-degree (isolated) nodes produce
/// `degrees[i] = 0.0` and `sqrt_deg[i] = 0.0`.
pub fn compute_degrees(
    graph: CsMatViewI<'_, f32, u32, usize>,
    mode: ComputeMode,
) -> (Array1<f64>, Array1<f64>) {
    let n = graph.rows();
    let degrees: Array1<f64> = match mode {
        ComputeMode::PythonCompat => {
            let deg_f32 = column_sum_f32(graph);
            Array1::from_vec(deg_f32.iter().map(|&d| d as f64).collect())
        }
        ComputeMode::RustNative => {
            let mut degrees = Array1::<f64>::zeros(n);
            for (row_idx, row_vec) in graph.outer_iterator().enumerate() {
                degrees[row_idx] = row_vec.data().iter().map(|&w| w as f64).sum();
            }
            degrees
        }
    };
    let sqrt_deg = degrees.mapv(f64::sqrt);
    (degrees, sqrt_deg)
}

/// Builds the symmetric normalized Laplacian L = I - D^{-1/2} W D^{-1/2} in f64.
///
/// Accepts the fuzzy k-NN graph in f32/u32 format from umap-rs and upcasts weights
/// to f64 for numerical stability. `inv_sqrt_deg` must have one entry per node:
/// `1/sqrt(degree[i])` for connected nodes and `0.0` for isolated (zero-degree) nodes.
///
/// Returns a symmetric CSR matrix of shape `(n, n)` with:
/// - Diagonal entries `L[i,i] = 1 - inv_sqrt_deg[i]² * W[i,i]`, stored for every node
///   (exactly 1.0 where the graph has no self-loop)
/// - Off-diagonal entries `L[i,j] = -inv_sqrt_deg[i] * W[i,j] * inv_sqrt_deg[j]`
///
/// Stored zeros in `graph` are absent edges and produce no entry.
pub fn build_normalized_laplacian(
    graph: CsMatViewI<'_, f32, u32, usize>,
    inv_sqrt_deg: &[f64],
) -> CsMatI<f64, usize> {
    let n = graph.rows();
    debug_assert_eq!(
        inv_sqrt_deg.len(),
        n,
        "inv_sqrt_deg must have one entry per node"
    );
    // Capacity: every nonzero of W, plus n diagonal entries
    let mut tri: TriMat<f64> = TriMat::with_capacity((n, n), graph.nnz() + n);

    for (row_idx, row_vec) in graph.outer_iterator().enumerate() {
        let mut diagonal = 1.0_f64;
        for (col_idx, &val) in row_vec.iter() {
            if val == 0.0 {
                continue;
            }
            let normalized = inv_sqrt_deg[row_idx] * (val as f64) * inv_sqrt_deg[col_idx];
            if row_idx == col_idx {
                diagonal -= normalized;
            } else {
                tri.add_triplet(row_idx, col_idx, -normalized);
            }
        }
        tri.add_triplet(row_idx, row_idx, diagonal);
    }

    tri.to_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::CsMatI;

    /// Degrees, inverse square-root degrees and Laplacian of `graph`, as the pipeline
    /// computes them.
    fn laplacian_of(graph: &CsMatI<f32, u32, usize>) -> (Array1<f64>, CsMatI<f64, usize>) {
        let (degrees, sqrt_deg) = compute_degrees(graph.view(), ComputeMode::PythonCompat);
        let inv_sqrt_deg: Vec<f64> = sqrt_deg
            .iter()
            .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
            .collect();
        (
            degrees,
            build_normalized_laplacian(graph.view(), &inv_sqrt_deg),
        )
    }

    #[test]
    fn self_loops_count_in_degree_and_laplacian_diagonal() {
        // 3 nodes; edges 0-1 (2.0), 1-2 (1.0); self-loops on node 0 (1.0) and node 2 (3.0).
        //     W = [[1, 2, 0],
        //          [2, 0, 1],
        //          [0, 1, 3]]      degrees = [3, 3, 4]
        let graph = CsMatI::new(
            (3, 3),
            vec![0usize, 2, 4, 6],
            vec![0u32, 1, 0, 2, 1, 2],
            vec![1.0f32, 2.0, 2.0, 1.0, 1.0, 3.0],
        );
        let (degrees, l) = laplacian_of(&graph);
        assert_eq!(degrees.to_vec(), vec![3.0, 3.0, 4.0]);

        let entry = |i, j| l.get(i, j).copied().unwrap_or(0.0);
        assert!((entry(0, 0) - (1.0 - 1.0 / 3.0)).abs() < 1e-15);
        assert!((entry(1, 1) - 1.0).abs() < 1e-15);
        assert!((entry(2, 2) - (1.0 - 3.0 / 4.0)).abs() < 1e-15);
        assert!((entry(0, 1) + 2.0 / 3.0).abs() < 1e-15);
        assert!((entry(1, 2) + 1.0 / 12.0_f64.sqrt()).abs() < 1e-15);

        // sqrt(degree) spans the null space: L · sqrt(d) = 0.
        let sqrt_deg = degrees.mapv(f64::sqrt);
        let residual = &l * &sqrt_deg;
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-14, "(L·sqrt(d))[{i}] = {r}");
        }
    }

    #[test]
    fn every_diagonal_entry_is_stored() {
        // Node 2 has only a self-loop, so L[2,2] = 1 - w/d = 0; the entry is still stored.
        let graph = CsMatI::new(
            (3, 3),
            vec![0usize, 1, 2, 3],
            vec![1u32, 0, 2],
            vec![1.0f32, 1.0, 5.0],
        );
        let (_, l) = laplacian_of(&graph);
        for i in 0..3 {
            assert!(l.get(i, i).is_some(), "diagonal entry {i} is not stored");
        }
        assert!(l.get(2, 2).copied().unwrap().abs() < 1e-15);
    }

    #[test]
    fn stored_zeros_are_absent_edges() {
        // The same path 0-1-2, once with an explicit zero stored at (0,2) and (2,0).
        let with_zeros = CsMatI::new(
            (3, 3),
            vec![0usize, 2, 4, 6],
            vec![1u32, 2, 0, 2, 0, 1],
            vec![1.0f32, 0.0, 1.0, 1.0, 0.0, 1.0],
        );
        let canonical = CsMatI::new(
            (3, 3),
            vec![0usize, 1, 3, 4],
            vec![1u32, 0, 2, 1],
            vec![1.0f32, 1.0, 1.0, 1.0],
        );
        let (deg_zeros, l_zeros) = laplacian_of(&with_zeros);
        let (deg_canonical, l_canonical) = laplacian_of(&canonical);
        assert_eq!(deg_zeros, deg_canonical);
        assert_eq!(l_zeros, l_canonical);
    }

    #[test]
    fn compute_degrees_zero_degree_node() {
        // 3-node graph: only nodes 0 and 1 are connected
        // node 2 is isolated (zero degree)
        let indptr = vec![0usize, 1, 2, 2];
        let indices = vec![1u32, 0u32];
        let data = vec![2.0f32, 2.0f32];
        let graph = CsMatI::new((3, 3), indptr, indices, data);

        let (degrees, sqrt_deg) = compute_degrees(graph.view(), ComputeMode::PythonCompat);

        assert!((degrees[0] - 2.0_f64).abs() < 1e-15);
        assert!((degrees[1] - 2.0_f64).abs() < 1e-15);
        assert!((degrees[2] - 0.0_f64).abs() < 1e-15); // isolated node
        assert!((sqrt_deg[2] - 0.0_f64).abs() < 1e-15); // sqrt(0) = 0, not NaN
    }

    #[test]
    fn compute_degrees_python_compat_matches_column_sum() {
        // 4-node graph with asymmetric accumulation weights that reveal f32 vs f64 path.
        // Edges: (0,1,0.3), (1,0,0.3), (1,2,0.7), (2,1,0.7), (2,3,0.5), (3,2,0.5)
        // Column sums in f32:
        //   col 0: val 0.3 (from row 1)      → deg[0] = 0.3f32
        //   col 1: val 0.3 (from row 0) + 0.7 (from row 2) → deg[1] = (0.3f32 + 0.7f32)
        //   col 2: val 0.7 (from row 1) + 0.5 (from row 3) → deg[2] = (0.7f32 + 0.5f32)
        //   col 3: val 0.5 (from row 2)      → deg[3] = 0.5f32
        let indptr = vec![0usize, 1, 3, 5, 6];
        let indices = vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32];
        let data = vec![0.3f32, 0.3f32, 0.7f32, 0.7f32, 0.5f32, 0.5f32];
        let graph = CsMatI::new((4, 4), indptr, indices, data);

        // Expected column sums in f32 (manually computed):
        let expected_col_sums: Vec<f32> = {
            let mut d = vec![0.0f32; 4];
            // row 0: (col=1, val=0.3)
            d[1] += 0.3f32;
            // row 1: (col=0, val=0.3), (col=2, val=0.7)
            d[0] += 0.3f32;
            d[2] += 0.7f32;
            // row 2: (col=1, val=0.7), (col=3, val=0.5)
            d[1] += 0.7f32;
            d[3] += 0.5f32;
            // row 3: (col=2, val=0.5)
            d[2] += 0.5f32;
            d
        };

        let (deg, _) = compute_degrees(graph.view(), ComputeMode::PythonCompat);

        for i in 0..4 {
            let expected = expected_col_sums[i] as f64;
            assert!(
                (deg[i] - expected).abs() < 1e-15,
                "deg[{i}]: got {}, expected {} (f32 col-sum widened to f64)",
                deg[i],
                expected
            );
        }
    }

    #[test]
    fn compute_degrees_rust_native_uses_f64_row_sum() {
        // 4-node path graph; verify RustNative sums CSR rows in f64.
        // For integer-valued weights f32 and f64 row sums are identical.
        let indptr = vec![0usize, 1, 3, 5, 6];
        let indices = vec![1u32, 0u32, 2u32, 1u32, 3u32, 2u32];
        let data = vec![1.0f32; 6];
        let graph = CsMatI::new((4, 4), indptr, indices, data);

        let (deg, sqrt_deg) = compute_degrees(graph.view(), ComputeMode::RustNative);

        // Node 0 and 3 have degree 1, nodes 1 and 2 have degree 2
        assert!((deg[0] - 1.0_f64).abs() < 1e-15, "deg[0]={}", deg[0]);
        assert!((deg[1] - 2.0_f64).abs() < 1e-15, "deg[1]={}", deg[1]);
        assert!((deg[2] - 2.0_f64).abs() < 1e-15, "deg[2]={}", deg[2]);
        assert!((deg[3] - 1.0_f64).abs() < 1e-15, "deg[3]={}", deg[3]);
        assert!((sqrt_deg[0] - 1.0_f64).abs() < 1e-15);
        assert!((sqrt_deg[1] - 2.0_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn build_laplacian_two_node_complete_graph() {
        // 2-node graph: edge (0,1) weight 1.0 and edge (1,0) weight 1.0
        let indptr = vec![0usize, 1, 2];
        let indices = vec![1u32, 0u32];
        let data = vec![1.0f32, 1.0f32];
        let graph = CsMatI::new((2, 2), indptr, indices, data);
        let inv_sqrt_deg = vec![1.0f64, 1.0f64]; // sqrt_deg = [1.0, 1.0]

        let l = build_normalized_laplacian(graph.view(), &inv_sqrt_deg);

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

        let l = build_normalized_laplacian(graph.view(), &inv_sqrt_deg);

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

        let l = build_normalized_laplacian(graph.view(), &inv_sqrt_deg);

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
        let inv_sqrt_deg = vec![1.0f64, 1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt(), 1.0f64];

        let l = build_normalized_laplacian(graph.view(), &inv_sqrt_deg);

        // All diagonal entries must be exactly 1.0
        for i in 0..4 {
            let d = l.get(i, i).copied().unwrap_or(0.0);
            assert!((d - 1.0).abs() < 1e-15, "L[{i},{i}] = {d}, expected 1.0");
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
