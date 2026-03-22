use ndarray::{Array2, ArrayView1, ArrayView2};
use sprs::{CsMatI, TriMat};

use crate::{
    ComputeMode,
    SpectralError,
    laplacian::{build_normalized_laplacian, compute_degrees},
    selection::select_eigenvectors,
    solvers::solve_eigenproblem,
};

/// Computes spectral embedding for disconnected graphs.
/// Each component is embedded independently; component embeddings are aligned
/// into a common coordinate frame via centroid meta-embedding.
pub(crate) fn embed_disconnected(
    graph: &CsMatI<f32, u32, usize>,
    component_labels: &[usize],
    n_conn_components: usize,
    n_embedding_dims: usize,
    seed: u64,
    data: Option<ArrayView2<'_, f32>>,
    compute_mode: ComputeMode,
) -> Result<Array2<f64>, SpectralError> {
    let n = graph.rows();

    // Collect sorted node indices per component
    let component_members = collect_component_members(component_labels, n_conn_components);

    // Phase A: compute meta-embedding position for each component
    let meta_embedding = if n_conn_components > 2 * n_embedding_dims {
        match data {
            Some(d) => spectral_meta_embedding(
                &d,
                &component_members,
                n_conn_components,
                n_embedding_dims,
                seed,
                compute_mode,
            )?,
            None => {
                return Err(SpectralError::InvalidGraph(
                    "data is required for spectral meta-embedding when \
                     n_conn_components > 2 * n_embedding_dims"
                        .into(),
                ))
            }
        }
    } else {
        orthogonal_meta_embedding(n_conn_components, n_embedding_dims)
    };

    // Phase B: embed each component and assemble result
    let data_ranges = compute_data_range(&meta_embedding);
    let mut result = Array2::<f64>::zeros((n, n_embedding_dims));

    for (comp_idx, members) in component_members.iter().enumerate() {
        let meta_pos = meta_embedding.row(comp_idx);
        let comp_coords = embed_single_component(
            graph,
            members,
            n_embedding_dims,
            data_ranges[comp_idx],
            &meta_pos,
            seed,
            compute_mode,
        )?;
        for (local_i, &orig_i) in members.iter().enumerate() {
            result.row_mut(orig_i).assign(&comp_coords.row(local_i));
        }
    }

    Ok(result)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Partitions node indices by component label.
/// Returns `component_members[c]` = list of original node indices in component `c`.
fn collect_component_members(labels: &[usize], n_conn_components: usize) -> Vec<Vec<usize>> {
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_conn_components];
    for (node, &label) in labels.iter().enumerate() {
        members[label].push(node);
    }
    members
}

/// Extracts the subgraph induced by `node_indices` from the full graph,
/// re-indexing nodes to `0..node_indices.len()`.
fn extract_subgraph(
    graph: &CsMatI<f32, u32, usize>,
    node_indices: &[usize],
) -> CsMatI<f32, u32, usize> {
    let n_global = graph.rows();
    let n_comp = node_indices.len();

    // Build original-index → local-index lookup; usize::MAX means "not in this component"
    let mut lookup = vec![usize::MAX; n_global];
    for (local, &orig) in node_indices.iter().enumerate() {
        lookup[orig] = local;
    }

    let mut indptr = vec![0usize; n_comp + 1];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();

    for (local_row, &orig_row) in node_indices.iter().enumerate() {
        debug_assert!(orig_row < n_global, "node index {orig_row} out of bounds (graph has {n_global} rows)");
        if let Some(row_vec) = graph.outer_view(orig_row) {
            for (orig_col, &weight) in row_vec.iter() {
                if lookup[orig_col] != usize::MAX {
                    let local_col = u32::try_from(lookup[orig_col])
                        .expect("local index overflows u32; component size exceeds u32::MAX");
                    indices.push(local_col);
                    data.push(weight);
                }
            }
        }
        indptr[local_row + 1] = indices.len();
    }

    CsMatI::<f32, u32, usize>::new((n_comp, n_comp), indptr, indices, data)
}

/// Orthogonal meta-embedding for `n_comp <= 2 * n_embedding_dims`.
/// Row `i`: `+e_{i % n_embedding_dims}` for `i < n_embedding_dims`, `-e_{i % n_embedding_dims}` for `i >= n_embedding_dims`.
fn orthogonal_meta_embedding(n_comp: usize, n_embedding_dims: usize) -> Array2<f64> {
    debug_assert!(
        n_comp <= 2 * n_embedding_dims,
        "orthogonal_meta_embedding requires n_comp ({n_comp}) <= 2 * n_embedding_dims ({n_embedding_dims})"
    );
    let mut m = Array2::<f64>::zeros((n_comp, n_embedding_dims));
    for row in 0..n_comp {
        let d = row % n_embedding_dims;
        let sign = if row < n_embedding_dims { 1.0 } else { -1.0 };
        m[[row, d]] = sign;
    }
    m
}

/// Spectral meta-embedding using Gaussian affinity between component centroids.
/// Used when `n_comp > 2 * n_embedding_dims` (requires data).
fn spectral_meta_embedding(
    data: &ArrayView2<f32>,
    component_members: &[Vec<usize>],
    n_comp: usize,
    n_embedding_dims: usize,
    seed: u64,
    _compute_mode: ComputeMode,
) -> Result<Array2<f64>, SpectralError> {
    let n_features = data.ncols();

    // Compute per-component centroids in f64
    let mut centroids = Array2::<f64>::zeros((n_comp, n_features));
    for (c, members) in component_members.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        for &node in members {
            for f in 0..n_features {
                centroids[[c, f]] += data[[node, f]] as f64;
            }
        }
        let inv_size = 1.0 / members.len() as f64;
        for f in 0..n_features {
            centroids[[c, f]] *= inv_size;
        }
    }

    // Gaussian affinity: aff[i,j] = exp(-||centroid_i - centroid_j||^2)
    let mut aff = Array2::<f64>::zeros((n_comp, n_comp));
    for i in 0..n_comp {
        for j in 0..n_comp {
            let dist2: f64 = (0..n_features)
                .map(|f| {
                    let d = centroids[[i, f]] - centroids[[j, f]];
                    d * d
                })
                .sum();
            aff[[i, j]] = (-dist2).exp();
        }
    }

    // Normalized Laplacian of the n_comp x n_comp centroid affinity graph.
    // Degrees sum only off-diagonal entries (no self-loops) to match the
    // standard formula L = I - D^{-1/2} W D^{-1/2} where W has zero diagonal.
    let degrees: Vec<f64> = (0..n_comp)
        .map(|i| (0..n_comp).filter(|&j| j != i).map(|j| aff[[i, j]]).sum())
        .collect();
    let inv_sqrt_d: Vec<f64> = degrees
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    let mut tri = TriMat::<f64>::with_capacity((n_comp, n_comp), n_comp * n_comp);
    for i in 0..n_comp {
        tri.add_triplet(i, i, 1.0);
        for j in 0..n_comp {
            if i != j {
                tri.add_triplet(i, j, -inv_sqrt_d[i] * aff[[i, j]] * inv_sqrt_d[j]);
            }
        }
    }
    let centroid_laplacian = tri.to_csr();

    let sqrt_deg_meta = ndarray::Array1::from_iter(degrees.iter().map(|&d| d.sqrt()));
    let ((evals, evecs), _) = solve_eigenproblem(&centroid_laplacian, n_embedding_dims, seed, &sqrt_deg_meta);
    let evals_slice = evals
        .as_slice_memory_order()
        .ok_or(SpectralError::ConvergenceFailure)?;
    let selected = select_eigenvectors(evals_slice, &evecs, n_embedding_dims);

    let max_abs = selected.fold(0.0_f64, |a, &x| a.max(x.abs()));
    if max_abs == 0.0 {
        return Err(SpectralError::InvalidGraph(
            "spectral meta-embedding produced all-zero result".into(),
        ));
    }
    Ok(selected / max_abs)
}

/// Per-component data range: half the distance to the nearest other component
/// in meta-embedding space. Returns a `Vec` of length `n_comp`.
fn compute_data_range(meta_embedding: &Array2<f64>) -> Vec<f64> {
    let n_comp = meta_embedding.nrows();
    if n_comp == 1 {
        return vec![1.0];
    }
    let dim = meta_embedding.ncols();
    let mut data_ranges = vec![0.0f64; n_comp];
    for c in 0..n_comp {
        let mut min_dist = f64::INFINITY;
        for i in 0..n_comp {
            if i == c {
                continue;
            }
            let dist: f64 = (0..dim)
                .map(|d| {
                    let diff = meta_embedding[[c, d]] - meta_embedding[[i, d]];
                    diff * diff
                })
                .sum::<f64>()
                .sqrt();
            if dist > 0.0 && dist < min_dist {
                min_dist = dist;
            }
        }
        data_ranges[c] = if min_dist.is_finite() {
            min_dist / 2.0
        } else {
            1.0
        };
    }
    data_ranges
}

/// Embeds a single component using the standard spectral pipeline, scaled and
/// translated to its position in meta-embedding space.
///
/// Tiny components (`size < 2 * n_embedding_dims`) are placed directly at `meta_pos`.
fn embed_single_component(
    graph: &CsMatI<f32, u32, usize>,
    members: &[usize],
    n_embedding_dims: usize,
    data_range: f64,
    meta_pos: &ArrayView1<f64>,
    seed: u64,
    compute_mode: ComputeMode,
) -> Result<Array2<f64>, SpectralError> {
    let size = members.len();

    // Tiny component: all nodes collapse to the meta position
    if size < 2 * n_embedding_dims {
        let mut coords = Array2::<f64>::zeros((size, n_embedding_dims));
        for i in 0..size {
            for d in 0..n_embedding_dims {
                coords[[i, d]] = meta_pos[d];
            }
        }
        return Ok(coords);
    }

    // Large component: run the standard spectral pipeline on the subgraph
    let sub_graph = extract_subgraph(graph, members);
    let (_degrees, sqrt_deg) = compute_degrees(&sub_graph, compute_mode);
    let inv_sqrt_deg: Vec<f64> = sqrt_deg
        .iter()
        .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
        .collect();
    let laplacian = build_normalized_laplacian(&sub_graph, &inv_sqrt_deg);
    let ((evals, evecs), _) = solve_eigenproblem(&laplacian, n_embedding_dims, seed, &sqrt_deg);
    let evals_slice = evals
        .as_slice_memory_order()
        .ok_or(SpectralError::ConvergenceFailure)?;
    let mut coords = select_eigenvectors(evals_slice, &evecs, n_embedding_dims);

    // Scale to data_range
    let max_abs = coords.fold(0.0_f64, |a, &x| a.max(x.abs()));
    if max_abs > 0.0 {
        coords *= data_range / max_abs;
    }

    // Translate to meta position
    for i in 0..size {
        for d in 0..n_embedding_dims {
            coords[[i, d]] += meta_pos[d];
        }
    }

    Ok(coords)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::CsMatI;

    fn make_two_pair_graph() -> CsMatI<f32, u32, usize> {
        // 4-node graph: edges 0-1 and 2-3 (two isolated pairs)
        CsMatI::<f32, u32, usize>::new(
            (4, 4),
            vec![0usize, 1, 2, 3, 4],
            vec![1u32, 0u32, 3u32, 2u32],
            vec![1.0f32; 4],
        )
    }

    fn make_empty_pair_graph() -> CsMatI<f32, u32, usize> {
        // 2 isolated single nodes (no edges)
        CsMatI::<f32, u32, usize>::new((2, 2), vec![0usize, 0, 0], vec![], vec![])
    }

    fn make_3_clique_graph() -> CsMatI<f32, u32, usize> {
        // 12-node graph: 3 fully connected cliques of 4 nodes each
        // clique 0: {0,1,2,3}, clique 1: {4,5,6,7}, clique 2: {8,9,10,11}
        let n = 12usize;
        let mut indptr = vec![0usize];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for c in 0..3usize {
            let base = c * 4;
            for i in 0..4usize {
                for j in 0..4usize {
                    if i != j {
                        indices.push((base + j) as u32);
                        data.push(1.0f32);
                    }
                }
                indptr.push(indices.len());
            }
        }
        CsMatI::<f32, u32, usize>::new((n, n), indptr, indices, data)
    }

    // ── extract_subgraph ──────────────────────────────────────────────────────

    #[test]
    fn test_extract_subgraph_two_components() {
        let g = make_two_pair_graph();

        // Extract component {0, 1}
        let sub01 = extract_subgraph(&g, &[0, 1]);
        assert_eq!(sub01.rows(), 2);
        assert_eq!(sub01.cols(), 2);
        assert_eq!(sub01.nnz(), 2, "should have edges (0,1) and (1,0)");
        for (val, (r, c)) in sub01.iter() {
            assert_eq!(*val, 1.0f32, "weight at ({r},{c}) should be 1.0");
        }

        // Extract component {2, 3} – re-indexed to {0, 1}
        let sub23 = extract_subgraph(&g, &[2, 3]);
        assert_eq!(sub23.rows(), 2);
        assert_eq!(sub23.cols(), 2);
        assert_eq!(sub23.nnz(), 2);
    }

    #[test]
    fn test_extract_subgraph_single_node() {
        let g = make_two_pair_graph();
        // Node 1 has an edge to node 0, but node 0 is not in this subgraph
        let sub = extract_subgraph(&g, &[1]);
        assert_eq!(sub.rows(), 1);
        assert_eq!(sub.cols(), 1);
        assert_eq!(sub.nnz(), 0, "single-node subgraph should have no edges");
    }

    // ── orthogonal_meta_embedding ─────────────────────────────────────────────

    #[test]
    fn test_orthogonal_meta_embedding_4_comp_2_dim() {
        let m = orthogonal_meta_embedding(4, 2);
        assert_eq!(m.shape(), &[4, 2]);
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[0, 1]], 0.0);
        assert_eq!(m[[1, 0]], 0.0);
        assert_eq!(m[[1, 1]], 1.0);
        assert_eq!(m[[2, 0]], -1.0);
        assert_eq!(m[[2, 1]], 0.0);
        assert_eq!(m[[3, 0]], 0.0);
        assert_eq!(m[[3, 1]], -1.0);
    }

    #[test]
    fn test_orthogonal_meta_embedding_2_comp_2_dim() {
        let m = orthogonal_meta_embedding(2, 2);
        assert_eq!(m.shape(), &[2, 2]);
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[0, 1]], 0.0);
        assert_eq!(m[[1, 0]], 0.0);
        assert_eq!(m[[1, 1]], 1.0);
    }

    #[test]
    fn test_orthogonal_meta_embedding_1_comp_3_dim() {
        let m = orthogonal_meta_embedding(1, 3);
        assert_eq!(m.shape(), &[1, 3]);
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[0, 1]], 0.0);
        assert_eq!(m[[0, 2]], 0.0);
    }

    // ── embed_disconnected ────────────────────────────────────────────────────

    #[test]
    fn test_embed_disconnected_two_pairs_shape_and_finite() {
        let g = make_two_pair_graph();
        let labels = vec![0usize, 0, 1, 1];
        let result = embed_disconnected(&g, &labels, 2, 2, 42, None, ComputeMode::PythonCompat);
        let arr = result.expect("embed_disconnected should succeed");
        assert_eq!(arr.shape(), &[4, 2]);
        for &v in arr.iter() {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }

    #[test]
    fn test_embed_disconnected_tiny_components_placed_at_meta_pos() {
        // 2 isolated single nodes, dim=2, data=None
        // n_comp=2 ≤ 2*2=4 → orthogonal: meta_pos[0]=[1,0], meta_pos[1]=[0,1]
        // Both size=1 < 2*dim=4 → placed at meta centroid
        let g = make_empty_pair_graph();
        let labels = vec![0usize, 1];
        let result = embed_disconnected(&g, &labels, 2, 2, 42, None, ComputeMode::PythonCompat);
        let arr = result.expect("embed_disconnected should succeed for tiny components");
        assert_eq!(arr.shape(), &[2, 2]);
        let eps = 1e-10;
        assert!((arr[[0, 0]] - 1.0).abs() < eps, "node 0 x: expected 1.0, got {}", arr[[0, 0]]);
        assert!((arr[[0, 1]] - 0.0).abs() < eps, "node 0 y: expected 0.0, got {}", arr[[0, 1]]);
        assert!((arr[[1, 0]] - 0.0).abs() < eps, "node 1 x: expected 0.0, got {}", arr[[1, 0]]);
        assert!((arr[[1, 1]] - 1.0).abs() < eps, "node 1 y: expected 1.0, got {}", arr[[1, 1]]);
    }

    #[test]
    fn test_embed_disconnected_intracomp_lt_intercomp() {
        // 12-node graph: 3 components of 4 nodes each (fully connected cliques)
        // n_comp=3 ≤ 2*2=4 → orthogonal meta-embedding: meta[0]=[1,0], [0,1], [-1,0]
        // For K_n, the spectral eigenvectors have zero mean, so each component's centroid
        // equals its meta position. The min centroid-to-centroid distance (≈ sqrt(2)) is
        // strictly greater than the max intra-component spread (≤ data_range = sqrt(2)/2).
        let g = make_3_clique_graph();
        let labels: Vec<usize> = (0..12).map(|i| i / 4).collect();
        let result = embed_disconnected(&g, &labels, 3, 2, 42, None, ComputeMode::PythonCompat);
        let arr = result.expect("embed_disconnected on 3-clique graph should succeed");
        assert_eq!(arr.shape(), &[12, 2]);

        // Compute per-component centroids
        let mut centroids = [[0.0f64; 2]; 3];
        for i in 0..12 {
            let c = i / 4;
            centroids[c][0] += arr[[i, 0]] / 4.0;
            centroids[c][1] += arr[[i, 1]] / 4.0;
        }

        // Max distance from each node to its component centroid (intra-component spread)
        let mut max_spread = 0.0f64;
        for i in 0..12 {
            let c = i / 4;
            let dx = arr[[i, 0]] - centroids[c][0];
            let dy = arr[[i, 1]] - centroids[c][1];
            max_spread = max_spread.max((dx * dx + dy * dy).sqrt());
        }

        // Min distance between component centroids
        let mut min_centroid_dist = f64::INFINITY;
        for a in 0..3 {
            for b in (a + 1)..3 {
                let dx = centroids[a][0] - centroids[b][0];
                let dy = centroids[a][1] - centroids[b][1];
                min_centroid_dist = min_centroid_dist.min((dx * dx + dy * dy).sqrt());
            }
        }

        assert!(
            min_centroid_dist > max_spread,
            "min centroid-to-centroid dist {min_centroid_dist} should be > max intra-comp spread {max_spread}"
        );
    }
}
