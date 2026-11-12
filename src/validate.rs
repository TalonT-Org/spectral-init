//! Boundary validation of the caller's configuration, graph and data.

use crate::laplacian::column_sum_f32;
use crate::{SpectralError, SpectralInitConfig};
use ndarray::ArrayView2;
use sprs::CsMatViewI;

/// Largest relative difference between `W[i,j]` and `W[j,i]` that still counts as
/// symmetric. A fuzzy union `A + Aᵀ − A∘Aᵀ` evaluated in single precision can leave the
/// two halves of an edge one rounding error apart (1.2e-7 relative); the tolerance
/// admits a few of those.
const SYMMETRY_REL_TOL: f64 = 1e-6;

/// Validates the inputs of `spectral_init` and returns the graph as a CSR view.
///
/// Checks run on the matrix as the caller supplied it, so reported coordinates are in
/// the caller's orientation, in this order of precedence: `InvalidParameter`,
/// `NotSquare`, `TooFewNodes`, `NonFiniteWeight`, `NegativeWeight`, `NotSymmetric`,
/// `DegreeOverflow`, `DataShapeMismatch`, `NonFiniteData`.
///
/// Stored zeros count as absent edges. A CSC view of a symmetric matrix is returned as
/// the CSR view of its transpose.
pub(crate) fn validate<'a>(
    graph: CsMatViewI<'a, f32, u32, usize>,
    data: Option<ArrayView2<'_, f32>>,
    config: &SpectralInitConfig,
) -> Result<CsMatViewI<'a, f32, u32, usize>, SpectralError> {
    let n_components = config.n_components();
    if n_components == 0 {
        return Err(SpectralError::InvalidParameter {
            name: "n_components",
            reason: "must be at least 1",
        });
    }

    let (rows, cols) = graph.shape();
    if rows != cols {
        return Err(SpectralError::NotSquare { rows, cols });
    }
    let n = rows;
    if n <= n_components {
        return Err(SpectralError::TooFewNodes { n, n_components });
    }

    check_weights(graph)?;
    check_symmetry(graph)?;

    let graph = if graph.is_csr() {
        graph
    } else {
        graph.transpose_into()
    };

    check_degrees(graph)?;
    if let Some(data) = data {
        check_data(data, n)?;
    }
    Ok(graph)
}

/// Every stored weight must be finite and non-negative. A non-finite weight anywhere
/// takes precedence over a negative one.
fn check_weights(graph: CsMatViewI<'_, f32, u32, usize>) -> Result<(), SpectralError> {
    let mut first_negative = None;
    for (&weight, (row, col)) in graph.iter() {
        if !weight.is_finite() {
            return Err(SpectralError::NonFiniteWeight {
                row: row as usize,
                col: col as usize,
            });
        }
        if weight < 0.0 && first_negative.is_none() {
            first_negative = Some((row as usize, col as usize));
        }
    }
    match first_negative {
        Some((row, col)) => Err(SpectralError::NegativeWeight { row, col }),
        None => Ok(()),
    }
}

/// Every non-zero off-diagonal entry must have a mirror entry of the same weight, within
/// `SYMMETRY_REL_TOL`. A missing or zero mirror entry fails the comparison, so the
/// pattern is checked along with the values.
fn check_symmetry(graph: CsMatViewI<'_, f32, u32, usize>) -> Result<(), SpectralError> {
    for (&weight, (row, col)) in graph.iter() {
        let (row, col) = (row as usize, col as usize);
        if weight == 0.0 || row == col {
            continue;
        }
        let weight = f64::from(weight);
        let mirrored = graph.get(col, row).map_or(0.0, |&w| f64::from(w));
        if (weight - mirrored).abs() > SYMMETRY_REL_TOL * weight.max(mirrored) {
            return Err(SpectralError::NotSymmetric { row, col });
        }
    }
    Ok(())
}

/// Every node's degree must stay finite in the single-precision accumulation the
/// `PythonCompat` degree computation uses.
fn check_degrees(graph: CsMatViewI<'_, f32, u32, usize>) -> Result<(), SpectralError> {
    match column_sum_f32(graph)
        .iter()
        .position(|degree| !degree.is_finite())
    {
        Some(node) => Err(SpectralError::DegreeOverflow { node }),
        None => Ok(()),
    }
}

/// `data` must have one row per node and only finite values.
fn check_data(data: ArrayView2<'_, f32>, n: usize) -> Result<(), SpectralError> {
    if data.nrows() != n {
        return Err(SpectralError::DataShapeMismatch {
            expected_rows: n,
            actual_rows: data.nrows(),
        });
    }
    match data.indexed_iter().find(|(_, value)| !value.is_finite()) {
        Some(((row, col), _)) => Err(SpectralError::NonFiniteData { row, col }),
        None => Ok(()),
    }
}
