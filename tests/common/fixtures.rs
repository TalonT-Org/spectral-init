//! Fixture loaders and synthetic graph builders shared by the test binaries.

#![allow(dead_code)]

use ndarray::{Array, Array1, Dimension};
use ndarray_npy::{NpzReader, ReadNpyError, ReadNpzError, ReadableElement};
use sprs::{CsMat, CsMatI};
use std::path::{Path, PathBuf};

/// Directory holding the Python-generated reference fixtures.
///
/// Located by walking up from the including package's manifest directory to the first
/// ancestor that contains `tests/fixtures`, so every workspace member finds the same data.
pub fn fixtures_dir() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .map(|dir| dir.join("tests").join("fixtures"))
        .find(|candidate| candidate.is_dir())
        .unwrap_or_else(|| panic!("no tests/fixtures directory at or above {manifest_dir:?}"))
}

pub fn fixture_path(dataset: &str, filename: &str) -> PathBuf {
    fixtures_dir().join(dataset).join(filename)
}

pub fn load_dense<T, D>(path: &Path, key: &str) -> Array<T, D>
where
    T: ReadableElement,
    D: Dimension,
{
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("cannot open fixture {:?}: {}", path, e));
    let mut npz = NpzReader::new(file)
        .unwrap_or_else(|e| panic!("cannot open NpzReader for {:?}: {}", path, e));
    npz.by_name(key)
        .unwrap_or_else(|e| panic!("key {:?} not found in {:?}: {}", key, path, e))
}

/// Load a scipy-saved CSR matrix as f64 data with usize indices.
pub fn load_sparse_csr(path: &Path) -> CsMat<f64> {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("cannot open fixture {:?}: {}", path, e));
    let mut npz = NpzReader::new(file)
        .unwrap_or_else(|e| panic!("cannot open NpzReader for {:?}: {}", path, e));

    // data: try f64 first (comp_b_laplacian), then f32 (step3/4/5a membership)
    let data: Vec<f64> = match npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>("data") {
        Ok(arr) => arr.into_iter().collect(),
        Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
            let arr: Array1<f32> = npz
                .by_name("data")
                .unwrap_or_else(|e| panic!("data key not found in {:?}: {}", path, e));
            arr.iter().map(|&x| x as f64).collect()
        }
        Err(e) => panic!("error reading 'data' from {:?}: {}", path, e),
    };

    // indices: try i32 first, then i64
    let indices: Vec<usize> = match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indices")
    {
        Ok(arr) => arr.iter().map(|&x| x as usize).collect(),
        Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
            let arr: Array1<i64> = npz
                .by_name("indices")
                .unwrap_or_else(|e| panic!("indices key not found in {:?}: {}", path, e));
            arr.iter().map(|&x| x as usize).collect()
        }
        Err(e) => panic!("error reading 'indices' from {:?}: {}", path, e),
    };

    // indptr: try i32 first, then i64
    let indptr: Vec<usize> = match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indptr") {
        Ok(arr) => arr.iter().map(|&x| x as usize).collect(),
        Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
            let arr: Array1<i64> = npz
                .by_name("indptr")
                .unwrap_or_else(|e| panic!("indptr key not found in {:?}: {}", path, e));
            arr.iter().map(|&x| x as usize).collect()
        }
        Err(e) => panic!("error reading 'indptr' from {:?}: {}", path, e),
    };

    // shape: try i32 first (most common), then i64 (SciPy on some 64-bit platforms)
    let shape_arr: Vec<usize> = match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("shape")
    {
        Ok(arr) => arr.iter().map(|&x| x as usize).collect(),
        Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
            let arr: Array1<i64> = npz
                .by_name("shape")
                .unwrap_or_else(|e| panic!("shape key not found in {:?}: {}", path, e));
            arr.iter().map(|&x| x as usize).collect()
        }
        Err(e) => panic!("error reading 'shape' from {:?}: {}", path, e),
    };
    let (rows, cols) = csr_shape(path, &shape_arr);

    // Assumes CSR format. ndarray-npy cannot read the |S3 "format" key (b"csr"), so the
    // storage order is not verified at runtime. Fixtures must be saved as CSR; if saved
    // as CSC the resulting CsMat will silently hold CSC data under a CSR label.

    CsMat::try_new((rows, cols), indptr, indices, data)
        .unwrap_or_else(|e| panic!("fixture CSR structure invalid in {:?}: {:?}", path, e))
}

/// Load a scipy-saved CSR matrix with f32 data and u32 column indices.
/// Python scipy typically stores indices as int32 (i32), which are cast to u32 here.
pub fn load_sparse_csr_f32_u32(path: &Path) -> CsMatI<f32, u32, usize> {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("cannot open fixture {:?}: {}", path, e));
    let mut npz = NpzReader::new(file)
        .unwrap_or_else(|e| panic!("cannot open NpzReader for {:?}: {}", path, e));

    // data: f32
    let data: Vec<f32> = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix1>("data")
        .unwrap_or_else(|e| panic!("data key not found in {:?}: {}", path, e))
        .into_iter()
        .collect();

    // indices: i32 → u32
    let indices: Vec<u32> = match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indices") {
        Ok(arr) => arr.iter().map(|&x| x as u32).collect(),
        Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
            let arr: Array1<i64> = npz
                .by_name("indices")
                .unwrap_or_else(|e| panic!("indices key not found in {:?}: {}", path, e));
            arr.iter().map(|&x| x as u32).collect()
        }
        Err(e) => panic!("error reading 'indices' from {:?}: {}", path, e),
    };

    // indptr: i32 → usize
    let indptr: Vec<usize> = match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indptr") {
        Ok(arr) => arr.iter().map(|&x| x as usize).collect(),
        Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
            let arr: Array1<i64> = npz
                .by_name("indptr")
                .unwrap_or_else(|e| panic!("indptr key not found in {:?}: {}", path, e));
            arr.iter().map(|&x| x as usize).collect()
        }
        Err(e) => panic!("error reading 'indptr' from {:?}: {}", path, e),
    };

    // shape: i32 → usize
    let shape_arr: Vec<usize> = match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("shape")
    {
        Ok(arr) => arr.iter().map(|&x| x as usize).collect(),
        Err(ReadNpzError::Npy(ReadNpyError::WrongDescriptor(_))) => {
            let arr: Array1<i64> = npz
                .by_name("shape")
                .unwrap_or_else(|e| panic!("shape key not found in {:?}: {}", path, e));
            arr.iter().map(|&x| x as usize).collect()
        }
        Err(e) => panic!("error reading 'shape' from {:?}: {}", path, e),
    };

    let (rows, cols) = csr_shape(path, &shape_arr);
    CsMatI::try_new((rows, cols), indptr, indices, data)
        .unwrap_or_else(|e| panic!("fixture CSR structure invalid in {:?}: {:?}", path, e))
}

/// `(rows, cols)` from a scipy CSR archive's `shape` key, which must hold two entries.
fn csr_shape(path: &Path, shape: &[usize]) -> (usize, usize) {
    assert_eq!(
        shape.len(),
        2,
        "shape key in {:?} has {} elements, expected 2",
        path,
        shape.len()
    );
    (shape[0], shape[1])
}

/// Convert a label vector to a set-of-sets, permitting label-invariant comparison.
/// Two label assignments are equivalent iff they produce the same partition.
pub fn partition_of(
    labels: &[usize],
) -> std::collections::BTreeSet<std::collections::BTreeSet<usize>> {
    let mut map: std::collections::BTreeMap<usize, std::collections::BTreeSet<usize>> =
        std::collections::BTreeMap::new();
    for (node, &label) in labels.iter().enumerate() {
        map.entry(label).or_default().insert(node);
    }
    map.into_values().collect()
}

/// Normalized Laplacian of C_n (ring graph) in f64.
/// Every node has degree 2. L = D^{-1/2}(D - A)D^{-1/2}:
///   diagonal = 1.0, off-diagonals = -0.5 (= -1/(sqrt(2)*sqrt(2))).
pub fn ring_laplacian(n: usize) -> sprs::CsMatI<f64, usize> {
    let mut indptr = vec![0usize; n + 1];
    let mut indices: Vec<usize> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    for i in 0..n {
        let left = if i == 0 { n - 1 } else { i - 1 };
        let right = if i + 1 == n { 0 } else { i + 1 };
        let mut entries = [(left, -0.5_f64), (i, 1.0_f64), (right, -0.5_f64)];
        entries.sort_unstable_by_key(|&(col, _)| col);
        for (col, val) in entries {
            indices.push(col);
            data.push(val);
        }
        indptr[i + 1] = indices.len();
    }
    sprs::CsMatI::new((n, n), indptr, indices, data)
}

/// sqrt_deg vector for C_n (ring graph): all entries are sqrt(2).
pub fn ring_sqrt_deg(n: usize) -> ndarray::Array1<f64> {
    ndarray::Array1::from_elem(n, 2.0_f64.sqrt())
}

/// Ring graph C_n as adjacency matrix: edges (i, (i±1) % n), weight 1.0.
/// Returns CsMatI<f32, u32, usize> suitable for spectral_init().
pub fn make_ring(n: u32) -> CsMatI<f32, u32, usize> {
    let n_usize = n as usize;
    let mut indptr = vec![0usize; n_usize + 1];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for i in 0..n_usize {
        let prev = (i + n_usize - 1) % n_usize;
        let next = (i + 1) % n_usize;
        let (lo, hi) = if prev < next {
            (prev, next)
        } else {
            (next, prev)
        };
        indices.push(lo as u32);
        data.push(1.0f32);
        indices.push(hi as u32);
        data.push(1.0f32);
        indptr[i + 1] = indices.len();
    }
    CsMatI::<f32, u32, usize>::new((n_usize, n_usize), indptr, indices, data)
}

/// Path graph P_n as adjacency matrix: edges (i, i+1), weight 1.0.
/// Returns CsMatI<f32, u32, usize> suitable for spectral_init().
pub fn make_path(n: u32) -> CsMatI<f32, u32, usize> {
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

/// Normalized Laplacian of P_n (path graph) in f64.
/// L = D^{-1/2}(D - A)D^{-1/2}. Tridiagonal with known closed-form entries:
///   diagonal = 1.0, endpoint off-diagonals = -1/sqrt(2), interior off-diagonals = -0.5.
pub fn path_laplacian(n: usize) -> CsMatI<f64, usize> {
    let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
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

/// sqrt_deg vector for P_n (path graph): endpoints = 1.0, interior = sqrt(2).
pub fn path_sqrt_deg(n: usize) -> Array1<f64> {
    let mut v = vec![2.0_f64.sqrt(); n];
    v[0] = 1.0;
    v[n - 1] = 1.0;
    Array1::from_vec(v)
}
