use sprs::CsMatI;

/// BFS connected-component labelling on a sparse graph.
/// Returns a Vec of length n where `result[i]` is the component index for node i,
/// and the number of distinct components.
pub(crate) fn find_components(graph: &CsMatI<f32, u32, usize>) -> (Vec<usize>, usize) {
    todo!("find_components: BFS on sprs CSR graph")
}
