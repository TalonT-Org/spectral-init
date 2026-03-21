use ndarray::{Array2, Axis};

/// Selects `n_components` eigenvectors, skipping the trivial zero-eigenvalue
/// eigenvector(s). Sorts eigenvalues ascending, skips index 0 (trivial), and
/// returns the next `n_components` columns from the eigenvectors matrix.
///
/// # Panics
///
/// Panics if `eigenvalues.len() <= n_components` (need at least `n_components + 1`
/// eigenvalues to skip the trivial one and still return `n_components` columns).
pub fn select_eigenvectors(
    eigenvalues: &[f64],
    eigenvectors: &Array2<f64>,
    n_components: usize,
) -> Array2<f64> {
    assert!(
        eigenvalues.len() > n_components,
        "need at least n_components+1 eigenvalues, got {}",
        eigenvalues.len()
    );
    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    eigenvectors.select(Axis(1), &indices[1..1 + n_components])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn assert_col_eq(result: &Array2<f64>, result_col: usize, source: &Array2<f64>, source_col: usize) {
        let r = result.column(result_col);
        let s = source.column(source_col);
        assert_eq!(r.len(), s.len(), "column length mismatch");
        for (i, (a, b)) in r.iter().zip(s.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "col {result_col}[{i}]: {a} != {b}");
        }
    }

    #[test]
    fn selects_correct_columns_sorted_input() {
        // eigenvalues already in ascending order: [0.001, 0.4, 1.2]
        // index 0 is trivial (smallest), should select indices 1 and 2
        let eigenvalues = vec![0.001_f64, 0.4, 1.2];
        let eigenvectors = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let result = select_eigenvectors(&eigenvalues, &eigenvectors, 2);
        assert_eq!(result.shape(), &[3, 2]);
        // Columns 1 and 2 selected (trivial column 0 skipped)
        assert_col_eq(&result, 0, &eigenvectors, 1);
        assert_col_eq(&result, 1, &eigenvectors, 2);
    }

    #[test]
    fn sorts_before_skipping_trivial() {
        // eigenvalues out of order: [1.2, 0.001, 0.5]
        // after sort: index 1 (0.001) is trivial, should select original indices 2 (0.5) and 0 (1.2)
        let eigenvalues = vec![1.2_f64, 0.001, 0.5];
        let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
        let eigenvectors = Array2::from_shape_vec((3, 3), data).unwrap();
        let result = select_eigenvectors(&eigenvalues, &eigenvectors, 2);
        assert_eq!(result.shape(), &[3, 2]);
        // sorted order: trivial=col1, then col2 (0.5), then col0 (1.2)
        // selected = [col2, col0]
        assert_col_eq(&result, 0, &eigenvectors, 2);
        assert_col_eq(&result, 1, &eigenvectors, 0);
    }

    #[test]
    fn selects_single_nontrivial_eigenvector() {
        let eigenvalues = vec![0.0_f64, 0.3, 0.8];
        let eigenvectors = Array2::eye(3);
        let result = select_eigenvectors(&eigenvalues, &eigenvectors, 1);
        assert_eq!(result.shape(), &[3, 1]);
        // Only column 1 selected (index 0 is trivial)
        assert_col_eq(&result, 0, &eigenvectors, 1);
    }
}
