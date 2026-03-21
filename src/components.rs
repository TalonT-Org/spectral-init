use sprs::CsMatI;
use std::collections::VecDeque;

/// BFS connected-component labelling on a sparse graph.
/// Returns a Vec of length n where `result[i]` is the component index for node i,
/// and the number of distinct components.
pub(crate) fn find_components(graph: &CsMatI<f32, u32, usize>) -> (Vec<usize>, usize) {
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
            for &col in graph
                .outer_view(node)
                .expect("node index in bounds")
                .indices()
            {
                let neighbor = col as usize;
                if labels[neighbor] == usize::MAX {
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
    use std::path::{Path, PathBuf};

    fn fixture_path(dataset: &str, filename: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(dataset)
            .join(filename)
    }

    fn load_pruned_as_csr_f32(dataset: &str) -> CsMatI<f32, u32, usize> {
        use ndarray::Array1;
        use ndarray_npy::{NpzReader, ReadNpyError, ReadNpzError};

        let path = fixture_path(dataset, "step5a_pruned.npz");
        let file = std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("cannot open fixture {:?}: {}", path, e));
        let mut npz = NpzReader::new(file)
            .unwrap_or_else(|e| panic!("cannot open NpzReader for {:?}: {}", path, e));

        let data: Vec<f32> = match npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix1>("data") {
            Ok(arr) => arr.into_iter().collect(),
            Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
                let arr: Array1<f64> = npz
                    .by_name("data")
                    .unwrap_or_else(|e| panic!("data key not found in {:?}: {}", path, e));
                arr.iter().map(|&x| x as f32).collect()
            }
            Err(e) => panic!("error reading 'data' from {:?}: {}", path, e),
        };

        let indices: Vec<u32> =
            match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indices") {
                Ok(arr) => arr.iter().map(|&x| x as u32).collect(),
                Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
                    let arr: Array1<i64> = npz
                        .by_name("indices")
                        .unwrap_or_else(|e| panic!("indices key not found in {:?}: {}", path, e));
                    arr.iter().map(|&x| x as u32).collect()
                }
                Err(e) => panic!("error reading 'indices' from {:?}: {}", path, e),
            };

        let indptr: Vec<usize> =
            match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indptr") {
                Ok(arr) => arr.iter().map(|&x| x as usize).collect(),
                Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
                    let arr: Array1<i64> = npz
                        .by_name("indptr")
                        .unwrap_or_else(|e| panic!("indptr key not found in {:?}: {}", path, e));
                    arr.iter().map(|&x| x as usize).collect()
                }
                Err(e) => panic!("error reading 'indptr' from {:?}: {}", path, e),
            };

        let shape_arr: Vec<usize> =
            match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("shape") {
                Ok(arr) => arr.iter().map(|&x| x as usize).collect(),
                Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
                    let arr: Array1<i64> = npz
                        .by_name("shape")
                        .unwrap_or_else(|e| panic!("shape key not found in {:?}: {}", path, e));
                    arr.iter().map(|&x| x as usize).collect()
                }
                Err(e) => panic!("error reading 'shape' from {:?}: {}", path, e),
            };

        assert!(
            shape_arr.len() >= 2,
            "shape key in {:?} has {} elements, expected 2",
            path,
            shape_arr.len()
        );

        CsMatI::try_new((shape_arr[0], shape_arr[1]), indptr, indices, data)
            .unwrap_or_else(|e| panic!("fixture CSR structure invalid in {:?}: {:?}", path, e))
    }

    fn partition_of(
        labels: &[usize],
    ) -> std::collections::BTreeSet<std::collections::BTreeSet<usize>> {
        let mut map: std::collections::BTreeMap<usize, std::collections::BTreeSet<usize>> =
            std::collections::BTreeMap::new();
        for (node, &label) in labels.iter().enumerate() {
            map.entry(label).or_default().insert(node);
        }
        map.into_values().collect()
    }

    // ── Pure unit tests ──────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = CsMatI::<f32, u32, usize>::new((0, 0), vec![0usize], vec![], vec![]);
        let (labels, n) = find_components(&g);
        assert_eq!(n, 0);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_single_node_no_edges() {
        let g = CsMatI::<f32, u32, usize>::new((1, 1), vec![0usize, 0], vec![], vec![]);
        let (labels, n) = find_components(&g);
        assert_eq!(n, 1);
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn test_two_isolated_nodes() {
        let g = CsMatI::<f32, u32, usize>::new((2, 2), vec![0usize, 0, 0], vec![], vec![]);
        let (labels, n) = find_components(&g);
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
        let (labels, n) = find_components(&g);
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
        let (labels, n) = find_components(&g);
        assert_eq!(n, 1);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn test_three_isolated_nodes() {
        let g =
            CsMatI::<f32, u32, usize>::new((3, 3), vec![0usize, 0, 0, 0], vec![], vec![]);
        let (labels, n) = find_components(&g);
        assert_eq!(n, 3);
        assert_ne!(labels[0], labels[1]);
        assert_ne!(labels[1], labels[2]);
        assert_ne!(labels[0], labels[2]);
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
        let (labels, n) = find_components(&g);
        assert_eq!(n, 2);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    // ── Fixture-gated tests ──────────────────────────────────────────────────

    #[test]
    #[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
    fn test_components_blobs_connected_200() {
        use ndarray_npy::NpzReader;

        let graph = load_pruned_as_csr_f32("blobs_connected_200");
        let (labels, n_components) = find_components(&graph);

        let path = fixture_path("blobs_connected_200", "comp_c_components.npz");
        let mut npz = NpzReader::new(std::fs::File::open(&path).unwrap()).unwrap();
        let py_n: i32 = npz
            .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix0>("n_components")
            .unwrap()
            .into_scalar();
        let py_labels_arr: ndarray::Array1<i32> = npz.by_name("labels").unwrap();
        let py_labels: Vec<usize> = py_labels_arr.iter().map(|&x| x as usize).collect();

        assert_eq!(n_components, py_n as usize, "n_components mismatch");
        assert_eq!(n_components, 1, "expected 1 component");
        assert_eq!(partition_of(&labels), partition_of(&py_labels));
    }

    #[test]
    #[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
    fn test_components_disconnected_200() {
        use ndarray_npy::NpzReader;

        let graph = load_pruned_as_csr_f32("disconnected_200");
        let (labels, n_components) = find_components(&graph);

        let path = fixture_path("disconnected_200", "comp_c_components.npz");
        let mut npz = NpzReader::new(std::fs::File::open(&path).unwrap()).unwrap();
        let py_n: i32 = npz
            .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix0>("n_components")
            .unwrap()
            .into_scalar();
        let py_labels_arr: ndarray::Array1<i32> = npz.by_name("labels").unwrap();
        let py_labels: Vec<usize> = py_labels_arr.iter().map(|&x| x as usize).collect();

        assert_eq!(n_components, py_n as usize, "n_components mismatch");
        assert_eq!(n_components, 4, "expected 4 components");
        assert_eq!(partition_of(&labels), partition_of(&py_labels));
    }

    #[test]
    #[ignore = "requires generated .npz fixtures; run: python tests/generate_fixtures.py"]
    fn test_components_all_datasets() {
        use ndarray_npy::NpzReader;

        let datasets = [
            "blobs_50",
            "blobs_500",
            "blobs_5000",
            "blobs_connected_200",
            "blobs_connected_2000",
            "circles_300",
            "moons_200",
            "near_dupes_100",
            "disconnected_200",
        ];

        for dataset in &datasets {
            let graph = load_pruned_as_csr_f32(dataset);
            let (labels, n_components) = find_components(&graph);

            let path = fixture_path(dataset, "comp_c_components.npz");
            let mut npz = NpzReader::new(
                std::fs::File::open(&path).unwrap_or_else(|e| {
                    panic!("dataset {}: cannot open {:?}: {}", dataset, path, e)
                }),
            )
            .unwrap_or_else(|e| panic!("dataset {}: cannot parse {:?}: {}", dataset, path, e));

            let py_n: i32 = npz
                .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix0>("n_components")
                .unwrap_or_else(|e| {
                    panic!("dataset {}: n_components not found: {}", dataset, e)
                })
                .into_scalar();
            let py_labels_arr: ndarray::Array1<i32> = npz
                .by_name("labels")
                .unwrap_or_else(|e| panic!("dataset {}: labels not found: {}", dataset, e));
            let py_labels: Vec<usize> = py_labels_arr.iter().map(|&x| x as usize).collect();

            assert_eq!(
                n_components,
                py_n as usize,
                "dataset {}: n_components mismatch (rust={}, python={})",
                dataset,
                n_components,
                py_n
            );
            assert_eq!(
                partition_of(&labels),
                partition_of(&py_labels),
                "dataset {}: label partitioning mismatch",
                dataset
            );
        }
    }
}
