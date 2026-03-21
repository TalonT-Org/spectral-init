use ndarray::Array1;
use sprs::CsMatI;

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
/// Accepts the fuzzy k-NN graph in f32/u32 format from umap-rs and upcasts.
pub(crate) fn build_normalized_laplacian(
    graph: &CsMatI<f32, u32, usize>,
) -> (CsMatI<f64, usize>, Array1<f64>) {
    todo!("build_normalized_laplacian: construct L and return degree vector D")
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
}
