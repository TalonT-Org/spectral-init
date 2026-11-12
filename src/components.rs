use sprs::CsMatViewI;
use std::collections::VecDeque;

/// BFS connected-component labelling on a sparse graph in CSR storage.
/// Returns a Vec of length n where `result[i]` is the component index for node i,
/// and the number of distinct components. Stored zeros are absent edges.
pub fn find_components(graph: CsMatViewI<'_, f32, u32, usize>) -> (Vec<usize>, usize) {
    debug_assert!(
        graph.is_csr(),
        "find_components requires CSR storage; a CSC matrix would silently traverse \
         in-neighbors instead of out-neighbors"
    );

    let n = graph.rows();
    let mut labels = vec![usize::MAX; n];
    let mut n_components = 0usize;

    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        labels[start] = n_components;
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            // `node` is a row index below `n`, so the row always exists.
            let Some(row) = graph.outer_view(node) else {
                continue;
            };
            for (neighbor, &weight) in row.iter() {
                // Column indices of a square matrix are below `n`.
                if weight != 0.0 && labels[neighbor] == usize::MAX {
                    labels[neighbor] = n_components;
                    queue.push_back(neighbor);
                }
            }
        }

        n_components += 1;
    }

    (labels, n_components)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::CsMatI;

    #[test]
    fn test_empty_graph() {
        let g = CsMatI::<f32, u32, usize>::new((0, 0), vec![0usize], vec![], vec![]);
        let (labels, n) = find_components(g.view());
        assert_eq!(n, 0);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_single_node_no_edges() {
        let g = CsMatI::<f32, u32, usize>::new((1, 1), vec![0usize, 0], vec![], vec![]);
        let (labels, n) = find_components(g.view());
        assert_eq!(n, 1);
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn test_two_isolated_nodes() {
        let g = CsMatI::<f32, u32, usize>::new((2, 2), vec![0usize, 0, 0], vec![], vec![]);
        let (labels, n) = find_components(g.view());
        assert_eq!(n, 2);
        assert_ne!(labels[0], labels[1]);
    }

    #[test]
    fn test_two_connected_nodes() {
        // Symmetric: 0→1, 1→0
        let g = CsMatI::<f32, u32, usize>::new(
            (2, 2),
            vec![0usize, 1, 2],
            vec![1u32, 0u32],
            vec![1.0f32, 1.0f32],
        );
        let (labels, n) = find_components(g.view());
        assert_eq!(n, 1);
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn test_path_graph_3() {
        // 0-1-2 chain: 0→1, 1→0, 1→2, 2→1
        let g = CsMatI::<f32, u32, usize>::new(
            (3, 3),
            vec![0usize, 1, 3, 4],
            vec![1u32, 0u32, 2u32, 1u32],
            vec![1.0f32; 4],
        );
        let (labels, n) = find_components(g.view());
        assert_eq!(n, 1);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn test_three_isolated_nodes() {
        let g = CsMatI::<f32, u32, usize>::new((3, 3), vec![0usize, 0, 0, 0], vec![], vec![]);
        let (labels, n) = find_components(g.view());
        assert_eq!(n, 3);
        assert_ne!(labels[0], labels[1]);
        assert_ne!(labels[1], labels[2]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_stored_zero_does_not_connect() {
        // 0-1 joined by a real edge; 1-2 joined only by a stored zero.
        let g = CsMatI::<f32, u32, usize>::new(
            (3, 3),
            vec![0usize, 1, 3, 4],
            vec![1u32, 0u32, 2u32, 1u32],
            vec![1.0f32, 1.0, 0.0, 0.0],
        );
        let (labels, n) = find_components(g.view());
        assert_eq!(n, 2);
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[1], labels[2]);
    }

    #[test]
    fn test_two_components_four_nodes() {
        // Edges: 0-1 and 2-3, no cross edges
        let g = CsMatI::<f32, u32, usize>::new(
            (4, 4),
            vec![0usize, 1, 2, 3, 4],
            vec![1u32, 0u32, 3u32, 2u32],
            vec![1.0f32; 4],
        );
        let (labels, n) = find_components(g.view());
        assert_eq!(n, 2);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }
}
