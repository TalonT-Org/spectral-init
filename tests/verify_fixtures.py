#!/usr/bin/env python3
"""
verify_fixtures.py — Comprehensive validation of generated spectral-init fixtures.

Usage:
    python tests/verify_fixtures.py [--output-dir DIR] [--datasets NAME ...]

Exit code 0 if all checks pass, 1 if any fail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse


# ---------------------------------------------------------------------------
# Per-file check functions — each returns a list of error strings (empty = pass)
# ---------------------------------------------------------------------------

def check_step0(d: Any, n: int) -> list[str]:
    """step0_raw_data.npz: X (f64, n×d), n_samples (int32), n_features (int32)."""
    errors = []
    if "X" not in d.files:
        errors.append("missing key 'X'")
    else:
        if d["X"].dtype != np.float64:
            errors.append(f"X dtype {d['X'].dtype} != float64")
        if d["X"].shape[0] != n:
            errors.append(f"X.shape[0] {d['X'].shape[0]} != n={n}")
        if not np.all(np.isfinite(d["X"])):
            errors.append("X contains non-finite values")
    if "n_samples" not in d.files:
        errors.append("missing key 'n_samples'")
    elif int(d["n_samples"]) != n:
        errors.append(f"n_samples {d['n_samples']} != {n}")
    return errors


def check_step1(d: Any, n: int, k: int) -> list[str]:
    """step1_knn.npz: indices (int32, n×k), dists (f32, n×k), n_neighbors (int32)."""
    errors = []
    idx = d["knn_indices"]
    dists = d["knn_dists"]
    if idx.dtype != np.int32:
        errors.append(f"knn_indices dtype {idx.dtype} != int32")
    if idx.shape != (n, k):
        errors.append(f"knn_indices shape {idx.shape} != ({n}, {k})")
    if dists.dtype != np.float32:
        errors.append(f"knn_dists dtype {dists.dtype} != float32")
    if dists.shape != (n, k):
        errors.append(f"knn_dists shape {dists.shape} != ({n}, {k})")
    if not (np.all(idx >= 0) and np.all(idx < n)):
        errors.append("knn_indices contains out-of-range values")
    if not np.all(dists >= 0):
        errors.append("knn_dists contains negative values")
    if int(d["n_neighbors"]) != k:
        errors.append(f"n_neighbors {d['n_neighbors']} != {k}")
    return errors


def check_step2(d: Any, n: int, k: int) -> list[str]:
    """step2_smooth_knn.npz: sigmas (f32, n), rhos (f32, n), n_neighbors (int32)."""
    errors = []
    if d["sigmas"].dtype != np.float32:
        errors.append(f"sigmas dtype {d['sigmas'].dtype} != float32")
    if d["sigmas"].shape != (n,):
        errors.append(f"sigmas shape {d['sigmas'].shape} != ({n},)")
    if d["rhos"].dtype != np.float32:
        errors.append(f"rhos dtype {d['rhos'].dtype} != float32")
    if d["rhos"].shape != (n,):
        errors.append(f"rhos shape {d['rhos'].shape} != ({n},)")
    if not np.all(d["sigmas"] > 0):
        errors.append("sigmas contains non-positive values")
    if int(d["n_neighbors"]) != k:
        errors.append(f"n_neighbors {d['n_neighbors']} != {k}")
    return errors


def check_step3(A: scipy.sparse.spmatrix, n: int) -> list[str]:
    """step3_membership.npz: COO, shape (n,n), values in (0,1], NOT symmetric."""
    errors = []
    if A.shape != (n, n):
        errors.append(f"shape {A.shape} != ({n}, {n})")
    if A.nnz == 0:
        errors.append("matrix is empty (nnz == 0)")
    data = A.data if hasattr(A, "data") else A.tocsr().data
    if not np.all(data > 0):
        errors.append("contains non-positive values")
    if not np.all(data <= 1.0 + 1e-10):
        errors.append(f"values exceed 1.0 (max={data.max():.6f})")
    diff = (A - A.T).tocsr()
    if diff.nnz == 0:
        errors.append("matrix is symmetric (must be directed/asymmetric)")
    return errors


def check_step4(A: scipy.sparse.spmatrix, n: int) -> list[str]:
    """step4_symmetrized.npz: CSR, shape (n,n), values in (0,1], symmetric."""
    errors = []
    if A.shape != (n, n):
        errors.append(f"shape {A.shape} != ({n}, {n})")
    data = A.data if hasattr(A, "data") else A.tocsr().data
    if not np.all(data > 0):
        errors.append("contains non-positive values")
    if not np.all(data <= 1.0 + 1e-10):
        errors.append(f"values exceed 1.0 (max={data.max():.6f})")
    diff = abs(A - A.T)
    if diff.max() >= 1e-10:
        errors.append(f"not symmetric: max|A-A^T|={diff.max():.2e}")
    return errors


def check_step5a(A5: scipy.sparse.spmatrix, A4: scipy.sparse.spmatrix, n: int) -> list[str]:
    """step5a_pruned.npz: fewer nnz than step4, symmetric, min nonzero >= threshold."""
    errors = []
    if A5.shape != (n, n):
        errors.append(f"shape {A5.shape} != ({n}, {n})")
    if A5.nnz >= A4.nnz:
        errors.append(f"nnz {A5.nnz} not < step4 nnz {A4.nnz} (no edges pruned)")
    diff = abs(A5 - A5.T)
    if diff.max() >= 1e-10:
        errors.append(f"not symmetric: max|A-A^T|={diff.max():.2e}")
    if A5.nnz > 0:
        n_epochs = 500 if n <= 10000 else 200
        threshold = A4.tocsr().data.max() / float(n_epochs)
        if A5.data.min() < threshold - 1e-12:
            errors.append(f"min nonzero {A5.data.min():.6f} below threshold {threshold:.6f}")
    return errors


def check_comp_a(d: Any, n: int) -> list[str]:
    """comp_a_degrees.npz: degrees (f64, n), sqrt_deg (f64, n), consistency."""
    errors = []
    deg = d["degrees"]
    sqrt_deg = d["sqrt_deg"]
    if deg.dtype != np.float64:
        errors.append(f"degrees dtype {deg.dtype} != float64")
    if deg.shape != (n,):
        errors.append(f"degrees shape {deg.shape} != ({n},)")
    if not np.all(deg >= 0):
        errors.append("degrees contains negative values")
    if not np.allclose(sqrt_deg, np.sqrt(deg), atol=1e-14):
        errors.append("sqrt_deg != sqrt(degrees)")
    return errors


def check_comp_b(L: scipy.sparse.spmatrix, n: int) -> list[str]:
    """comp_b_laplacian.npz: f64, diagonal=1, symmetric, PSD, eigenvalues in [0,2]."""
    errors = []
    if L.shape != (n, n):
        errors.append(f"shape {L.shape} != ({n}, {n})")
    if L.dtype != np.float64:
        errors.append(f"dtype {L.dtype} != float64")
    diag = np.asarray(L.diagonal())
    if not np.allclose(diag, 1.0, atol=1e-12):
        errors.append(f"diagonal not all 1.0 (max_err={np.abs(diag - 1.0).max():.2e})")
    diff = abs(L - L.T)
    if diff.max() >= 1e-14:
        errors.append(f"not symmetric: max|L-L^T|={diff.max():.2e}")
    # PSD check via eigenvalues (only feasible for small n; skip for large graphs)
    if n <= 1000:
        eigvals = np.linalg.eigvalsh(L.toarray())
        if eigvals.min() < -1e-10:
            errors.append(f"not PSD: min eigenvalue={eigvals.min():.2e}")
        if eigvals.max() > 2.0 + 1e-10:
            errors.append(f"eigenvalue > 2: max={eigvals.max():.6f}")
    return errors


def check_comp_c(d: Any, n: int) -> list[str]:
    """comp_c_components.npz: contiguous integer labels, n_components >= 1."""
    errors = []
    labels = d["labels"]
    n_comp = int(d["n_components"])
    if labels.dtype != np.int32:
        errors.append(f"labels dtype {labels.dtype} != int32")
    if labels.shape != (n,):
        errors.append(f"labels shape {labels.shape} != ({n},)")
    if n_comp < 1:
        errors.append(f"n_components={n_comp} < 1")
    if labels.min() < 0:
        errors.append(f"labels contain negative values (min={labels.min()})")
    if labels.max() >= n_comp:
        errors.append(f"labels.max()={labels.max()} >= n_components={n_comp}")
    # All component IDs 0..n_comp-1 must appear
    present = set(labels.tolist())
    expected = set(range(n_comp))
    if present != expected:
        missing = expected - present
        errors.append(f"component IDs not contiguous: missing {missing}")
    return errors


def check_comp_d(d: Any) -> list[str]:
    """comp_d_eigensolver.npz: eigenvalues in [0,2], non-decreasing, residuals < 1e-4,
    eigenvectors orthonormal."""
    errors = []
    lam = d["eigenvalues"]
    V = d["eigenvectors"]
    res = d["residuals"]
    k = int(d["k"])

    if lam.shape != (k,):
        errors.append(f"eigenvalues shape {lam.shape} != ({k},)")
    if not np.all(lam >= -1e-12):
        errors.append(f"eigenvalues below 0 (min={lam.min():.2e})")
    if not np.all(lam <= 2.0 + 1e-12):
        errors.append(f"eigenvalues above 2 (max={lam.max():.6f})")
    if not np.all(np.diff(lam) >= -1e-12):
        errors.append(f"eigenvalues not monotonically non-decreasing")
    if not np.all(res < 1e-4):
        errors.append(f"residuals >= 1e-4 (max={res.max():.2e})")
    VTV = V.T @ V
    if not np.allclose(VTV, np.eye(VTV.shape[0]), atol=1e-10):
        off_diag = np.abs(VTV - np.eye(VTV.shape[0])).max()
        errors.append(f"eigenvectors not orthonormal (max_off_diag={off_diag:.2e})")
    return errors


def check_comp_e(d: Any, d_d: Any, n: int, dim: int) -> list[str]:
    """comp_e_selection.npz: embedding shape (n, dim), trivial index excluded from order,
    columns match corresponding eigenvectors from comp_d."""
    errors = []
    emb = d["embedding"]
    order = d["order"]
    if emb.shape != (n, dim):
        errors.append(f"embedding shape {emb.shape} != ({n}, {dim})")
    # trivial eigenvector index must not be in order
    trivial_col = int(np.argsort(d_d["eigenvalues"])[0])
    if trivial_col in order.tolist():
        errors.append(f"trivial eigenvector index {trivial_col} included in order")
    # Cross-step: embedding columns must equal eigenvectors[:, order]
    V = d_d["eigenvectors"]
    for i, col in enumerate(order.tolist()):
        if not (np.allclose(emb[:, i], V[:, col], atol=1e-14) or
                np.allclose(emb[:, i], -V[:, col], atol=1e-14)):
            errors.append(f"comp_e column {i} != ±comp_d eigenvectors[:, {col}]")
            break  # one is enough to flag the issue
    return errors


def check_comp_f(d: Any, n: int, dim: int) -> list[str]:
    """comp_f_scaling.npz: pre_noise max=10, final=pre_noise+noise, f32 dtypes."""
    errors = []
    pre = d["pre_noise"]
    noise = d["noise"]
    final = d["final"]
    if pre.dtype != np.float32:
        errors.append(f"pre_noise dtype {pre.dtype} != float32")
    if noise.dtype != np.float32:
        errors.append(f"noise dtype {noise.dtype} != float32")
    if final.dtype != np.float32:
        errors.append(f"final dtype {final.dtype} != float32")
    if d["expansion"].dtype != np.float64:
        errors.append(f"expansion dtype {d['expansion'].dtype} != float64")
    max_abs = float(np.abs(pre).max())
    if abs(max_abs - 10.0) > 1e-5:
        errors.append(f"pre_noise max abs = {max_abs:.6f} != 10.0 (tol 1e-5)")
    if not np.allclose(final, pre + noise, atol=1e-7):
        errors.append("final != pre_noise + noise")
    return errors


def check_full_spectral(d: Any, n: int, dim: int) -> list[str]:
    """full_spectral.npz: embedding (f64, n×dim), finite, not all zeros."""
    errors = []
    emb = d["embedding"]
    if emb.dtype != np.float64:
        errors.append(f"embedding dtype {emb.dtype} != float64")
    if emb.shape != (n, dim):
        errors.append(f"embedding shape {emb.shape} != ({n}, {dim})")
    if not np.all(np.isfinite(emb)):
        errors.append("embedding contains non-finite values")
    if np.all(emb == 0):
        errors.append("embedding is all zeros")
    return errors


def check_full_umap_e2e(d: Any, n: int, dim: int) -> list[str]:
    """full_umap_e2e.npz: embedding (f32, n×dim), finite, not all zeros."""
    errors = []
    emb = d["embedding"]
    if emb.dtype != np.float32:
        errors.append(f"embedding dtype {emb.dtype} != float32")
    if emb.shape != (n, dim):
        errors.append(f"embedding shape {emb.shape} != ({n}, {dim})")
    if not np.all(np.isfinite(emb)):
        errors.append("embedding contains non-finite values")
    if np.all(emb == 0):
        errors.append("embedding is all zeros")
    return errors


def cross_check_full_spectral_vs_comp_f(
    dataset_dir: Path, n: int, dim: int, n_conn_components: int
) -> list[str]:
    """
    For connected datasets (n_conn_components == 1): verify that full_spectral["embedding"]
    matches comp_f_scaling["pre_noise"] up to sign per column (both scaled to max abs 10).

    Sign normalization: for each column i, if dot(v_full[:,i], v_comp[:,i]) < 0, flip v_full.
    Uses atol=0.01 (generous: covers f32 precision, different ARPACK v0 random state, and any
    trivial residual differences between ARPACK runs).

    For disconnected datasets, UMAP's per-component stitching produces a different result
    than our single-graph eigensolver chain; cross-check is skipped.
    """
    if n_conn_components != 1:
        return []  # skip: different algorithm branch for multi-component graphs

    errors = []
    full_emb = np.load(dataset_dir / "full_spectral.npz", allow_pickle=False)["embedding"]
    pre_noise = np.load(dataset_dir / "comp_f_scaling.npz", allow_pickle=False)["pre_noise"]
    full_f32 = full_emb.astype(np.float32)

    for col in range(dim):
        v_full = full_f32[:, col]
        v_comp = pre_noise[:, col]
        if np.dot(v_full.astype(np.float64), v_comp.astype(np.float64)) < 0:
            v_full = -v_full
        if not np.allclose(v_full, v_comp, atol=0.01):
            max_diff = float(np.abs(v_full - v_comp).max())
            errors.append(
                f"full_spectral col {col} != comp_f pre_noise col {col} "
                f"(max_diff={max_diff:.4f} > atol=0.01)"
            )
    return errors


def verify_exact_pipeline_files(
    dataset_dir: Path,
    n_samples: int,
    n_neighbors: int,
    n_components: int = 2,
) -> list[str]:
    """
    Validate the five _exact.npz fixture variants if they are present.
    Skips silently when the files are absent (large datasets, or approx-only run).
    """
    failures: list[str] = []
    if not (dataset_dir / "step1_knn_exact.npz").exists():
        return failures  # exact pipeline was not generated; skip

    def run(label: str, fn, *args):
        for e in fn(*args):
            failures.append(f"  [{label}] {e}")

    n, k = n_samples, n_neighbors

    d1e = np.load(dataset_dir / "step1_knn_exact.npz", allow_pickle=False)
    run("step1_knn_exact", check_step1, d1e, n, k)

    d2e = np.load(dataset_dir / "step2_smooth_knn_exact.npz", allow_pickle=False)
    run("step2_smooth_knn_exact", check_step2, d2e, n, k)

    A3e = scipy.sparse.load_npz(str(dataset_dir / "step3_membership_exact.npz"))
    run("step3_membership_exact", check_step3, A3e, n)

    A4e = scipy.sparse.load_npz(str(dataset_dir / "step4_symmetrized_exact.npz"))
    run("step4_symmetrized_exact", check_step4, A4e, n)

    A5e = scipy.sparse.load_npz(str(dataset_dir / "step5a_pruned_exact.npz"))
    run("step5a_pruned_exact", check_step5a, A5e, A4e, n)

    return failures


# ---------------------------------------------------------------------------
# Per-dataset orchestrator
# ---------------------------------------------------------------------------

def verify_dataset(
    dataset_dir: Path,
    n_samples: int,
    n_neighbors: int,
    n_components: int = 2,
) -> list[str]:
    """
    Run all checks for one dataset directory. Returns list of failure strings.
    Each failure string is prefixed with the fixture filename.
    """
    failures: list[str] = []

    def run(label: str, fn, *args):
        errs = fn(*args)
        for e in errs:
            failures.append(f"  [{label}] {e}")

    n, k, dim = n_samples, n_neighbors, n_components

    # Dense fixtures
    d0 = np.load(dataset_dir / "step0_raw_data.npz", allow_pickle=False)
    run("step0_raw_data", check_step0, d0, n)

    d1 = np.load(dataset_dir / "step1_knn.npz", allow_pickle=False)
    run("step1_knn", check_step1, d1, n, k)

    d2 = np.load(dataset_dir / "step2_smooth_knn.npz", allow_pickle=False)
    run("step2_smooth_knn", check_step2, d2, n, k)

    # Sparse fixtures
    A3 = scipy.sparse.load_npz(str(dataset_dir / "step3_membership.npz"))
    run("step3_membership", check_step3, A3, n)

    A4 = scipy.sparse.load_npz(str(dataset_dir / "step4_symmetrized.npz"))
    run("step4_symmetrized", check_step4, A4, n)

    A5 = scipy.sparse.load_npz(str(dataset_dir / "step5a_pruned.npz"))
    run("step5a_pruned", check_step5a, A5, A4, n)

    # Laplacian steps
    da = np.load(dataset_dir / "comp_a_degrees.npz", allow_pickle=False)
    run("comp_a_degrees", check_comp_a, da, n)

    L = scipy.sparse.load_npz(str(dataset_dir / "comp_b_laplacian.npz"))
    run("comp_b_laplacian", check_comp_b, L, n)

    dc = np.load(dataset_dir / "comp_c_components.npz", allow_pickle=False)
    run("comp_c_components", check_comp_c, dc, n)

    # Eigensolver and selection
    dd = np.load(dataset_dir / "comp_d_eigensolver.npz", allow_pickle=False)
    run("comp_d_eigensolver", check_comp_d, dd)

    de = np.load(dataset_dir / "comp_e_selection.npz", allow_pickle=False)
    run("comp_e_selection", check_comp_e, de, dd, n, dim)

    df = np.load(dataset_dir / "comp_f_scaling.npz", allow_pickle=False)
    run("comp_f_scaling", check_comp_f, df, n, dim)

    # Full-pipeline references
    dfs = np.load(dataset_dir / "full_spectral.npz", allow_pickle=False)
    run("full_spectral", check_full_spectral, dfs, n, dim)

    dfu = np.load(dataset_dir / "full_umap_e2e.npz", allow_pickle=False)
    run("full_umap_e2e", check_full_umap_e2e, dfu, n, dim)

    # Cross-step consistency
    n_conn = int(dc["n_components"])
    cross = cross_check_full_spectral_vs_comp_f(dataset_dir, n, dim, n_conn)
    for e in cross:
        failures.append(f"  [cross_check] {e}")

    # Exact-distance pipeline (skipped silently if files absent)
    exact_failures = verify_exact_pipeline_files(dataset_dir, n_samples, n_neighbors, n_components)
    failures.extend(exact_failures)

    return failures


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    output_dir: "Path | str | None" = None,
    dataset_names: "list[str] | None" = None,
) -> bool:
    """
    Verify all (or selected) datasets in output_dir.
    Returns True if all checks pass, False otherwise.
    Prints per-check PASS/FAIL summary to stdout.
    """
    if output_dir is None:
        output_dir = Path("tests/fixtures")
    output_dir = Path(output_dir)

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest.json not found in {output_dir}", file=sys.stderr)
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    datasets = manifest["datasets"]
    if dataset_names:
        name_set = set(dataset_names)
        datasets = [d for d in datasets if d["name"] in name_set]

    all_pass = True
    for entry in datasets:
        name = entry["name"]
        n = entry["shape"][0]
        k = entry.get("n_neighbors", 15)
        dataset_dir = output_dir / name
        print(f"\n[{name}] Verifying ({n} samples, k={k})...")
        failures = verify_dataset(dataset_dir, n_samples=n, n_neighbors=k)
        if failures:
            all_pass = False
            print(f"  FAIL ({len(failures)} error(s)):")
            for msg in failures:
                print(msg)
        else:
            print(f"  PASS — all checks OK")

    print()
    if all_pass:
        print("verify_fixtures: ALL PASS")
    else:
        print("verify_fixtures: FAILURES DETECTED", file=sys.stderr)
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify spectral-init test fixtures"
    )
    parser.add_argument(
        "--output-dir", default="tests/fixtures",
        help="Root fixture directory (default: tests/fixtures)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Dataset names to verify; omit to verify all",
    )
    args = parser.parse_args()
    ok = main(output_dir=args.output_dir, dataset_names=args.datasets)
    sys.exit(0 if ok else 1)
