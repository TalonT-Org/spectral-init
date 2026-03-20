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

sys.path.insert(0, str(Path(__file__).parent))
from fixture_utils import DATASETS, get_env_metadata, save_dense, save_metadata


# ---------------------------------------------------------------------------
# Pipeline stub functions — each returns None and prints "not implemented"
# ---------------------------------------------------------------------------

def step_build_knn_graph(X: np.ndarray, name: str, outdir: Path, params: dict) -> None:
    """Build k-NN graph from X. (stub — implemented in later issue)"""
    print("  [step_build_knn_graph] not implemented")


def step_compute_laplacian(knn_graph: None, name: str, outdir: Path, params: dict) -> None:
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
) -> dict:
    """Run all pipeline steps for a single dataset, return manifest entry."""
    dataset_dir = outdir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name}] Generating dataset...")
    X, params = gen_fn(**kwargs)
    print(f"  shape: {X.shape}, dtype: {X.dtype}")

    # Save raw points
    save_dense(dataset_dir / "X.npz", X=X)

    # Write meta.json
    meta = {
        "dataset": name,
        "params": params,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": get_env_metadata(),
    }
    save_metadata(dataset_dir / "meta.json", meta)
    print(f"  meta.json written")

    # Pipeline stubs — future issues will fill these in
    knn   = step_build_knn_graph(X, name, dataset_dir, params)
    lap   = step_compute_laplacian(knn, name, dataset_dir, params)
    eigs  = step_compute_eigenvectors(lap, name, dataset_dir, params)
    _emb  = step_compute_embedding(eigs, name, dataset_dir, params)

    return {"name": name, "shape": list(X.shape), "params": params}


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
        entry = generate_all_for_dataset(name, gen_fn, kwargs, outdir, verify=args.verify)
        manifest["datasets"].append(entry)

    manifest_path = outdir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nmanifest.json → {manifest_path}")
    print(f"Done. {len(manifest['datasets'])} dataset(s) generated.")


if __name__ == "__main__":
    main()
