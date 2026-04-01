#!/usr/bin/env python3
"""merfish_preprocessing_sweep.py — Preprocessing parameter sweep on 10K MERFISH subset."""
from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata

DATA_DIR = Path(__file__).parent / "merfish_data"
OUTPUT_DIR = Path(__file__).parent / "output"

NORMALIZATIONS = ["normalize_total+log1p", "log2"]
PCA_DIMS = [10, 20, 30, 50]
METRICS = ["euclidean", "cosine"]
N_NEIGHBORS = 15
SCALE_MAX_VALUE = 10
TRUSTWORTHINESS_THRESHOLD = 0.95


def apply_normalization(expression: np.ndarray, norm: str) -> "anndata.AnnData":
    """Normalize expression matrix according to the specified strategy.

    Parameters
    ----------
    expression:
        float32 array storing log2(count+1) values as extracted from the H5AD file.
    norm:
        Normalization variant: ``"normalize_total+log1p"`` or ``"log2"``.

    Returns
    -------
    AnnData with ``.X`` containing the normalized expression matrix.
    """
    import anndata
    import scanpy as sc

    if norm == "normalize_total+log1p":
        # Back-transform log2(count+1) → raw counts, then apply Zhuang lab pipeline
        raw_counts = np.exp2(expression) - 1.0  # 2^x − 1
        adata = anndata.AnnData(X=raw_counts.astype(np.float32))
        sc.pp.normalize_total(adata, target_sum=1000)
        sc.pp.log1p(adata)
    elif norm == "log2":
        # Use stored log2(count+1) values directly — no further normalization
        adata = anndata.AnnData(X=expression.astype(np.float32))
    else:
        raise ValueError(f"Unknown normalization: {norm!r}")
    return adata


def run_config(
    expression: np.ndarray,
    labels: np.ndarray,
    norm: str,
    n_pcs: int,
    metric: str,
) -> dict:
    """Run a single preprocessing configuration and return measured metrics.

    Parameters
    ----------
    expression:
        float32 (n, 1122) log2(count+1) expression matrix.
    labels:
        int32 (n,) cell-type label array.
    norm:
        Normalization variant (``"normalize_total+log1p"`` or ``"log2"``).
    n_pcs:
        Number of PCA components.
    metric:
        Distance metric for kNN and UMAP (``"euclidean"`` or ``"cosine"``).

    Returns
    -------
    Dict with keys: normalization, n_pcs, metric, spectral_gap, condition_number,
    n_components, trustworthiness, silhouette, wall_time_s.
    """
    import scanpy as sc
    import umap as umap_lib
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import silhouette_score
    from scipy.sparse import eye, diags
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.csgraph import connected_components

    t0 = time.perf_counter()

    adata = apply_normalization(expression, norm)
    sc.pp.scale(adata, max_value=SCALE_MAX_VALUE)
    sc.tl.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=n_pcs, metric=metric)
    X_pca = adata.obsm["X_pca"].astype(np.float64)

    # Normalized Laplacian eigenvalues (same construction as run_baseline)
    graph = adata.obsp["connectivities"]
    degree = np.array(graph.sum(axis=1)).flatten()
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degree, 1e-10)))
    L = eye(graph.shape[0]) - D_inv_sqrt @ graph @ D_inv_sqrt
    try:
        eigenvalues, _ = eigsh(L, k=10, which="SM")
    except Exception as exc:
        raise RuntimeError(
            f"eigsh failed for norm={norm}, n_pcs={n_pcs}, metric={metric}"
        ) from exc
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))

    spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) >= 2 else 0.0
    condition_number = (
        float(eigenvalues[-1] / eigenvalues[1])
        if len(eigenvalues) >= 2 and eigenvalues[1] > 0
        else float("inf")
    )
    n_conn, _ = connected_components(graph, directed=False)

    # UMAP
    mapper = umap_lib.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=0.1,
        n_components=2,
        metric=metric,
        random_state=42,
        n_jobs=1,
    ).fit(X_pca)
    embedding = mapper.embedding_

    tw = trustworthiness(X_pca, embedding, n_neighbors=N_NEIGHBORS)
    sil = silhouette_score(embedding, labels)
    wall_time = time.perf_counter() - t0

    return {
        "normalization": norm,
        "n_pcs": n_pcs,
        "metric": metric,
        "spectral_gap": spectral_gap,
        "condition_number": condition_number,
        "n_components": n_conn,
        "trustworthiness": tw,
        "silhouette": sil,
        "wall_time_s": wall_time,
    }


def run_sweep(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Run all 16 preprocessing configurations and return results as a DataFrame."""
    expression = np.load(data_dir / "merfish_10k_expression.npz")["arr_0"].astype(np.float32)
    labels = np.load(data_dir / "merfish_10k_labels.npz")["arr_0"].astype(np.int32)

    configs = list(itertools.product(NORMALIZATIONS, PCA_DIMS, METRICS))
    results = []
    for i, (norm, n_pcs, metric) in enumerate(configs, 1):
        print(f"[{i:2d}/16] norm={norm!s:<25s} n_pcs={n_pcs:2d}  metric={metric} ...", flush=True)
        row = run_config(expression, labels, norm, n_pcs, metric)
        results.append(row)
        print(
            f"         spectral_gap={row['spectral_gap']:.4f}  tw={row['trustworthiness']:.4f}  "
            f"sil={row['silhouette']:.4f}  t={row['wall_time_s']:.1f}s",
            flush=True,
        )
    return pd.DataFrame(results)


def select_winner(df: pd.DataFrame) -> pd.Series:
    """Select the winning configuration from sweep results.

    Selects the configuration with the highest spectral gap among those with
    trustworthiness > ``TRUSTWORTHINESS_THRESHOLD``. Falls back to the
    highest-trustworthiness configuration if none qualify.
    """
    qualified = df[df["trustworthiness"] > TRUSTWORTHINESS_THRESHOLD]
    if qualified.empty:
        print(
            f"[WARN] No config achieved trustworthiness > {TRUSTWORTHINESS_THRESHOLD:.2f}; "
            "selecting config with highest trustworthiness instead."
        )
        qualified = df.nlargest(1, "trustworthiness")
    return qualified.loc[qualified["spectral_gap"].idxmax()]


def print_summary(df: pd.DataFrame, winner: pd.Series) -> None:
    """Print ranked sweep results and winning configuration details."""
    ranked = df.sort_values("spectral_gap", ascending=False).reset_index(drop=True)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print("\n=== MERFISH Preprocessing Parameter Sweep — Ranked by Spectral Gap ===")
    print(ranked.to_string(index=True))
    print("\n=== WINNING CONFIGURATION ===")
    print(f"  normalization : {winner['normalization']}")
    print(f"  n_pcs         : {int(winner['n_pcs'])}")
    print(f"  metric        : {winner['metric']}")
    print(f"  spectral_gap  : {winner['spectral_gap']:.4f}")
    print(f"  condition_num : {winner['condition_number']:.2f}")
    print(f"  n_components  : {int(winner['n_components'])}")
    print(f"  trustworthness: {winner['trustworthiness']:.4f}")
    print(f"  silhouette    : {winner['silhouette']:.4f}")
    print(f"  wall_time_s   : {winner['wall_time_s']:.1f}s")
    # Report diff vs defaults
    defaults = {"normalization": "normalize_total+log1p", "n_pcs": 30, "metric": "euclidean"}
    diffs = {k: (defaults[k], winner[k]) for k in defaults if defaults[k] != winner[k]}
    if diffs:
        print("\n  NOTE: Winner differs from current defaults in generate_merfish_comparisons.py:")
        for k, (old, new) in diffs.items():
            print(f"    {k}: {old!r} → {new!r}")
        print("  → Update preprocess_merfish() with the winning parameters.")
    else:
        print("\n  Current defaults already match the winning configuration. No update needed.")


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[merfish_preprocessing_sweep] Starting 16-config parameter sweep...")
    t_total = time.perf_counter()
    df = run_sweep()
    csv_path = output_dir / "merfish_preprocessing_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    winner = select_winner(df)
    print_summary(df, winner)
    print(f"\n[merfish_preprocessing_sweep] Total time: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
