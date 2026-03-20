#!/usr/bin/env python3
"""
generate_fixtures.py — Fixture orchestrator for spectral-init tests.

Usage:
    python tests/generate_fixtures.py [--output-dir DIR] [--datasets NAME ...] [--verify]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from numpy.random import RandomState

sys.path.insert(0, str(Path(__file__).parent))
from fixture_utils import DATASETS, get_env_metadata, save_dense, save_metadata


# ---------------------------------------------------------------------------
# Pipeline step functions — steps 0–2
# ---------------------------------------------------------------------------

def generate_step0_raw_data(X: np.ndarray, outdir: Path) -> None:
    """Save dataset X as float64 with shape scalars to step0_raw_data.npz."""
    n_samples, n_features = X.shape
    np.savez(
        outdir / "step0_raw_data",
        X=X.astype(np.float64),
        n_samples=np.int32(n_samples),
        n_features=np.int32(n_features),
    )


def generate_step1_knn(
    X: np.ndarray, outdir: Path, params: dict, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run UMAP nearest_neighbors and save step1_knn.npz.
    Returns (knn_indices, knn_dists) for downstream steps.
    """
    from umap.umap_ import nearest_neighbors

    random_state = RandomState(params["seed"])
    knn_indices, knn_dists, _ = nearest_neighbors(
        X,
        n_neighbors,
        metric="euclidean",
        metric_kwds={},
        angular=False,
        random_state=random_state,
        n_jobs=1,
    )
    np.savez(
        outdir / "step1_knn",
        knn_indices=knn_indices.astype(np.int32),
        knn_dists=knn_dists.astype(np.float32),
        n_neighbors=np.int32(n_neighbors),
    )
    return knn_indices, knn_dists


def generate_step2_smooth_knn(
    knn_dists: np.ndarray, outdir: Path, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run UMAP smooth_knn_dist and save step2_smooth_knn.npz.
    Returns (sigmas, rhos) for downstream steps.
    """
    from umap.umap_ import smooth_knn_dist

    sigmas, rhos = smooth_knn_dist(knn_dists.astype(np.float32), float(n_neighbors))
    np.savez(
        outdir / "step2_smooth_knn",
        sigmas=sigmas.astype(np.float32),
        rhos=rhos.astype(np.float32),
        n_neighbors=np.int32(n_neighbors),
    )
    return sigmas, rhos


def generate_step3_membership(
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    sigmas: np.ndarray,
    rhos: np.ndarray,
    outdir: Path,
    n: int,
) -> scipy.sparse.coo_matrix:
    """
    Compute directed membership-strength graph and save step3_membership.npz.

    Calls compute_membership_strengths with C-contiguous float32 inputs.
    Returns the COO matrix for use by generate_step4_symmetrized.
    """
    from umap.umap_ import compute_membership_strengths

    rows, cols, vals, _ = compute_membership_strengths(
        knn_indices.astype(np.int32),
        np.ascontiguousarray(knn_dists.astype(np.float32)),
        np.ascontiguousarray(sigmas.astype(np.float32)),
        np.ascontiguousarray(rhos.astype(np.float32)),
    )
    directed = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
    scipy.sparse.save_npz(str(outdir / "step3_membership"), directed)
    return directed


def generate_step4_symmetrized(
    directed: scipy.sparse.coo_matrix,
    outdir: Path,
) -> scipy.sparse.csr_matrix:
    """
    Compute symmetric fuzzy union A + A^T - A∘A^T and save step4_symmetrized.npz.

    Uses element-wise multiply (.multiply) for the A∘A^T term (NOT matrix multiply).
    Returns the symmetric CSR matrix for use by generate_step5a_prune.
    """
    A = directed.tocsr()
    symmetric = A + A.T - A.multiply(A.T)
    symmetric.sum_duplicates()
    symmetric.eliminate_zeros()
    scipy.sparse.save_npz(str(outdir / "step4_symmetrized"), symmetric)
    return symmetric


def generate_step5a_prune(
    symmetric: scipy.sparse.csr_matrix,
    outdir: Path,
    n: int,
) -> scipy.sparse.csr_matrix:
    """
    Prune edges below threshold = max(data) / n_epochs and save step5a_pruned.npz.

    n_epochs: 500 for n <= 10000, 200 for larger datasets.
    Returns the pruned CSR matrix for downstream use.
    """
    n_epochs = 500 if n <= 10000 else 200
    graph = symmetric.copy()
    threshold = graph.data.max() / float(n_epochs)
    graph.data[graph.data < threshold] = 0.0
    graph.eliminate_zeros()
    scipy.sparse.save_npz(str(outdir / "step5a_pruned"), graph)
    return graph


# ---------------------------------------------------------------------------
# Laplacian construction steps
# ---------------------------------------------------------------------------

def generate_comp_a_degrees(
    pruned_graph: scipy.sparse.csr_matrix,
    outdir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute degree vector and its square root from the pruned graph.

    degrees = graph.sum(axis=0), squeezed to 1D, cast to f64.
    sqrt_deg = sqrt(degrees).
    Zero-degree nodes (isolated after pruning) are handled naturally: sqrt(0) = 0.
    """
    degrees = np.asarray(pruned_graph.sum(axis=0), dtype=np.float64).squeeze()
    sqrt_deg = np.sqrt(degrees)
    np.savez(
        outdir / "comp_a_degrees",
        degrees=degrees,
        sqrt_deg=sqrt_deg,
    )
    return degrees, sqrt_deg


def generate_comp_b_laplacian(
    pruned_graph: scipy.sparse.csr_matrix,
    sqrt_deg: np.ndarray,
    outdir: Path,
) -> scipy.sparse.csr_matrix:
    """
    Build symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2} in f64.

    Zero-degree nodes have inv_sqrt_deg = 0 (not inf), so their rows/columns
    in D^{-1/2} A D^{-1/2} are zero, and L[i,i] = 1.0 for all i.
    """
    n = pruned_graph.shape[0]
    inv_sqrt_deg = np.where(sqrt_deg > 0.0, 1.0 / sqrt_deg, 0.0)
    D = scipy.sparse.diags(inv_sqrt_deg, dtype=np.float64, format="csr")
    A = pruned_graph.astype(np.float64)
    I = scipy.sparse.eye(n, dtype=np.float64, format="csr")
    L = I - D @ A @ D
    L.sum_duplicates()
    L.eliminate_zeros()
    scipy.sparse.save_npz(str(outdir / "comp_b_laplacian"), L)
    return L


def generate_comp_c_components(
    pruned_graph: scipy.sparse.csr_matrix,
    outdir: Path,
) -> tuple[int, np.ndarray]:
    """
    Find connected components of the undirected pruned graph.

    Uses scipy.sparse.csgraph.connected_components with directed=False.
    Saves n_components (int32 scalar) and labels (int32 array of length n).
    """
    n_components, labels = scipy.sparse.csgraph.connected_components(
        pruned_graph, directed=False
    )
    np.savez(
        outdir / "comp_c_components",
        n_components=np.int32(n_components),
        labels=labels.astype(np.int32),
    )
    return int(n_components), labels


def generate_comp_d_eigensolver(
    L: scipy.sparse.csr_matrix,
    dim: int,
    outdir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the dim+1 smallest eigenpairs of the normalized Laplacian via ARPACK.

    Uses Python UMAP's exact parameters:
      k   = dim + 1  (extra for the trivial zero eigenvector)
      ncv = max(2*k + 1, int(sqrt(n)))  (number of Lanczos vectors)
      tol = 1e-4, v0 = ones(n), maxiter = n * 5

    eigsh returns eigenvalues in ascending order.
    Residuals are ||L·v − λ·v|| / ||v|| per eigenpair.
    """
    n = L.shape[0]
    k = dim + 1
    ncv = max(2 * k + 1, int(np.sqrt(n)))
    v0 = np.ones(n, dtype=np.float64)

    try:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            L, k=k, which="SM", ncv=ncv, tol=1e-4, v0=v0, maxiter=n * 5
        )
    except scipy.sparse.linalg.ArpackNoConvergence:
        # Dense fallback for graphs with degenerate near-zero eigenvalues
        # (e.g. disconnected graphs where #components >= k).
        L_dense = L.toarray()
        all_eigenvalues, all_eigenvectors = np.linalg.eigh(L_dense)
        eigenvalues = all_eigenvalues[:k]
        eigenvectors = all_eigenvectors[:, :k]

    eigenvalues = np.maximum(eigenvalues, 0.0)

    residuals = np.array([
        np.linalg.norm(L @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i])
        / np.linalg.norm(eigenvectors[:, i])
        for i in range(k)
    ])

    np.savez(
        outdir / "comp_d_eigensolver",
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        residuals=residuals,
        k=np.int32(k),
    )
    return eigenvalues, eigenvectors


def generate_comp_e_selection(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    dim: int,
    outdir: Path,
) -> np.ndarray:
    """
    Select the dim non-trivial eigenvectors by sorting eigenvalues and skipping index 0.

    order = argsort(eigenvalues)[1 : dim+1]   — skips the trivial (smallest) eigenvector
    embedding = eigenvectors[:, order]         — shape (n, dim)
    """
    order = np.argsort(eigenvalues)[1 : dim + 1]
    embedding = eigenvectors[:, order]

    np.savez(
        outdir / "comp_e_selection",
        order=order.astype(np.int32),
        embedding=embedding,
    )
    return embedding


def generate_comp_f_scaling(
    embedding: np.ndarray,
    outdir: Path,
) -> np.ndarray:
    """
    Scale embedding to max absolute value 10, cast to f32, add seeded Gaussian noise.

    expansion = 10.0 / max(|embedding|)
    pre_noise = (embedding * expansion).astype(f32)
    noise     = RandomState(42).normal(scale=0.0001, size=pre_noise.shape).astype(f32)
    final     = pre_noise + noise

    RandomState(42) is re-created from scratch each call so output is byte-identical
    across runs on the same platform.
    """
    expansion = 10.0 / np.abs(embedding).max()
    pre_noise = (embedding * expansion).astype(np.float32)
    noise = np.random.RandomState(42).normal(
        scale=0.0001, size=pre_noise.shape
    ).astype(np.float32)
    final = pre_noise + noise

    np.savez(
        outdir / "comp_f_scaling",
        pre_noise=pre_noise,
        noise=noise,
        final=final,
        expansion=np.float64(expansion),
    )
    return final


# ---------------------------------------------------------------------------
# Per-dataset orchestrator
# ---------------------------------------------------------------------------

def generate_all_for_dataset(
    name: str,
    gen_fn: Callable[..., tuple[np.ndarray, dict]],
    kwargs: dict,
    outdir: Path,
    verify: bool = False,
    n_neighbors: int = 15,
    n_components: int = 2,
) -> dict:
    """Run all pipeline steps for a single dataset, return manifest entry."""
    dataset_dir = outdir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name}] Generating dataset...")
    X, params = gen_fn(**kwargs)
    print(f"  shape: {X.shape}, dtype: {X.dtype}")

    # Step 0: raw data (replaces the former X.npz save)
    generate_step0_raw_data(X, dataset_dir)
    print(f"  step0_raw_data.npz written")

    # Step 1: nearest neighbors (triggers numba JIT on first call)
    knn_indices, knn_dists = generate_step1_knn(X, dataset_dir, params, n_neighbors)
    print(f"  step1_knn.npz written")

    # Step 2: smooth KNN distances
    sigmas, rhos = generate_step2_smooth_knn(knn_dists, dataset_dir, n_neighbors)
    print(f"  step2_smooth_knn.npz written")

    # Write meta.json
    meta = {
        "dataset": name,
        "params": params,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": get_env_metadata(),
    }
    save_metadata(dataset_dir / "meta.json", meta)
    print(f"  meta.json written")

    # Steps 3–5a: membership strengths, fuzzy union, edge pruning
    n = X.shape[0]

    directed = generate_step3_membership(knn_indices, knn_dists, sigmas, rhos, dataset_dir, n)
    print(f"  step3_membership.npz written")

    symmetric = generate_step4_symmetrized(directed, dataset_dir)
    print(f"  step4_symmetrized.npz written")

    pruned = generate_step5a_prune(symmetric, dataset_dir, n)
    print(f"  step5a_pruned.npz written")

    degrees, sqrt_deg = generate_comp_a_degrees(pruned, dataset_dir)
    print(f"  comp_a_degrees.npz written")

    L = generate_comp_b_laplacian(pruned, sqrt_deg, dataset_dir)
    print(f"  comp_b_laplacian.npz written")

    n_conn_components, labels = generate_comp_c_components(pruned, dataset_dir)
    print(f"  comp_c_components.npz written ({n_conn_components} component(s))")

    eigenvalues, eigenvectors = generate_comp_d_eigensolver(L, n_components, dataset_dir)
    print(f"  comp_d_eigensolver.npz written")

    embedding = generate_comp_e_selection(eigenvalues, eigenvectors, n_components, dataset_dir)
    print(f"  comp_e_selection.npz written")

    generate_comp_f_scaling(embedding, dataset_dir)
    print(f"  comp_f_scaling.npz written")

    return {"name": name, "shape": list(X.shape), "params": params, "n_neighbors": n_neighbors}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate spectral-init test fixtures"
    )
    parser.add_argument(
        "--output-dir", default="tests/fixtures",
        help="Root output directory (default: tests/fixtures)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Dataset names to generate; omit to generate all",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run post-generation constraint checks",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=15,
        help="Number of nearest neighbors for KNN steps (default: 15)",
    )
    parser.add_argument(
        "--n-components", type=int, default=2,
        help="Number of UMAP embedding dimensions (default: 2)",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected = DATASETS
    if args.datasets:
        name_set = set(args.datasets)
        selected = [(n, f, k) for n, f, k in DATASETS if n in name_set]
        if not selected:
            print(f"ERROR: No datasets match: {args.datasets}", file=sys.stderr)
            sys.exit(1)

    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [],
    }

    for name, gen_fn, kwargs in selected:
        entry = generate_all_for_dataset(
            name, gen_fn, kwargs, outdir,
            verify=args.verify, n_neighbors=args.n_neighbors,
            n_components=args.n_components,
        )
        manifest["datasets"].append(entry)

    manifest_path = outdir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nmanifest.json → {manifest_path}")
    print(f"Done. {len(manifest['datasets'])} dataset(s) generated.")


if __name__ == "__main__":
    main()
