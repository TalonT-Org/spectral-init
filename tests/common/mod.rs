#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub fn fixture_path(dataset: &str, filename: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dataset)
        .join(filename)
}

use ndarray::{Array, Array1, Dimension};
use ndarray_npy::{NpzReader, ReadNpyError, ReadNpzError, ReadableElement};

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

use sprs::CsMat;

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
    let indices: Vec<usize> =
        match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>("indices") {
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

    // shape: 1D array of 2 i32 elements
    let shape_arr: Array1<i32> = npz
        .by_name("shape")
        .unwrap_or_else(|e| panic!("shape key not found in {:?}: {}", path, e));
    let rows = shape_arr[0] as usize;
    let cols = shape_arr[1] as usize;

    // Note: "format" key is a |S3 byte string (b"csr") — ndarray-npy cannot read it, skip it.

    CsMat::try_new((rows, cols), indptr, indices, data)
        .unwrap_or_else(|e| panic!("fixture CSR structure invalid in {:?}: {:?}", path, e))
}

pub fn load_metadata(path: &Path) -> serde_json::Value {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read metadata {:?}: {}", path, e));
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("invalid JSON in {:?}: {}", path, e))
}
