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
# Downstream pipeline stubs (implemented in later issues)
# ---------------------------------------------------------------------------

def step_compute_laplacian(pruned_graph: scipy.sparse.csr_matrix, name: str, outdir: Path, params: dict) -> None:
    """Compute symmetric normalized Laplacian. (stub — implemented in later issue)"""
    print("  [step_compute_laplacian] not implemented")


def step_compute_eigenvectors(laplacian: None, name: str, outdir: Path, params: dict) -> None:
    """Compute leading eigenvectors via solver escalation chain. (stub)"""
    print("  [step_compute_eigenvectors] not implemented")


def step_compute_embedding(eigenvectors: None, name: str, outdir: Path, params: dict) -> None:
    """Scale and noise-inject eigenvectors to UMAP embedding. (stub)"""
    print("  [step_compute_embedding] not implemented")


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

    lap   = step_compute_laplacian(pruned, name, dataset_dir, params)
    eigs  = step_compute_eigenvectors(lap, name, dataset_dir, params)
    _emb  = step_compute_embedding(eigs, name, dataset_dir, params)

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
        )
        manifest["datasets"].append(entry)

    manifest_path = outdir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nmanifest.json → {manifest_path}")
    print(f"Done. {len(manifest['datasets'])} dataset(s) generated.")


if __name__ == "__main__":
    main()
