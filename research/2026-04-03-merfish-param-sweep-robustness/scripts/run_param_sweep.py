#!/usr/bin/env python3
"""run_param_sweep.py — 56-run UMAP parameter sweep (Phase 3)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.spatial
import umap as umap_lib
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

# Import metric helpers from visual_eval (same pattern as merfish_preprocessing_sweep.py)
sys.path.insert(0, str(Path(__file__).parents[3] / "tests" / "visual_eval"))
from global_metrics import random_triplet_accuracy, knn_preservation  # noqa: E402
from spatial_metrics import spatial_neighbor_agreement, compute_spatial_metrics  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[3]
DATA_DIR = Path(__file__).parents[1] / "data"
RESULTS_DIR = Path(__file__).parents[1] / "results"
GRAPH_NPZ = PROJECT_ROOT / "tests" / "visual_eval" / "output" / "merfish_10k_graph.npz"
RUST_INIT_NPY = PROJECT_ROOT / "tests" / "visual_eval" / "output" / "merfish_10k_rust_init.npy"

# ── Sweep constants ──────────────────────────────────────────────────────────
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_METRIC = "euclidean"
RANDOM_STATE = 42
N_EPOCHS = 200

N_NEIGHBORS_VALUES = [5, 10, 15, 30, 50, 100]
MIN_DIST_VALUES = [0.0, 0.01, 0.1, 0.25, 0.5, 0.8]
METRIC_VALUES = ["euclidean", "cosine"]

INIT_METHODS = ["rust_spectral", "python_spectral", "pca", "random"]
CSV_COLUMNS = [
    "param_swept", "param_value", "init_method",
    "trustworthiness", "triplet_accuracy", "knn_preservation",
    "sna", "morans_i_max",
    "procrustes_rust_vs_python", "procrustes_vs_default",
    "solver_level", "wall_time_s",
]


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_pca = np.load(DATA_DIR / "merfish_10k_pca.npy").astype(np.float64)
    spatial = np.load(DATA_DIR / "merfish_10k_spatial.npz")["arr_0"]
    labels = np.load(DATA_DIR / "merfish_10k_labels.npz")["arr_0"]
    return X_pca, spatial, labels


def build_umap_graph(
    X_pca: np.ndarray,
    n_neighbors: int,
    metric: str,
) -> scipy.sparse.csr_matrix:
    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=RANDOM_STATE,
    ).fit(X_pca)
    return reducer.graph_


def save_graph_for_rust(graph: scipy.sparse.csr_matrix, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csr = graph.tocsr()
    np.savez(
        output_dir / "merfish_10k_graph.npz",
        data=csr.data.astype(np.float32),
        indices=csr.indices.astype(np.int32),
        indptr=csr.indptr.astype(np.int32),
        shape=np.array(csr.shape, dtype=np.int64),
    )


def run_rust_export(
    project_root: Path,
) -> tuple[np.ndarray | None, int | None]:
    cmd = [
        "cargo", "nextest", "run",
        "--test", "export_merfish_init",
        "--run-ignored", "only",
        "--features", "testing",
        "--success-output", "immediate",
        "--", "export_merfish_init_10k",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"[WARN] cargo nextest failed (rc={result.returncode}):\n{result.stderr[:500]}",
                flush=True,
            )
            return None, None

        solver_level: int | None = None
        for line in (result.stdout + result.stderr).splitlines():
            stripped = line.strip()
            if stripped.startswith("SOLVER_LEVEL="):
                try:
                    solver_level = int(stripped.split("=", 1)[1].strip())
                except ValueError:
                    pass

        rust_init = np.load(
            project_root / "tests" / "visual_eval" / "output" / "merfish_10k_rust_init.npy"
        )
        return rust_init, solver_level

    except Exception as exc:
        print(f"[WARN] run_rust_export raised: {exc}", flush=True)
        return None, None


def get_python_spectral_init(
    X_pca: np.ndarray,
    n_neighbors: int,
    metric: str,
) -> np.ndarray:
    reducer = umap_lib.UMAP(
        init="spectral",
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=RANDOM_STATE,
        n_epochs=0,
    ).fit(X_pca)
    return reducer.embedding_


def run_umap_from_init(
    X_pca: np.ndarray,
    init_coords: np.ndarray | str,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> np.ndarray:
    return umap_lib.UMAP(
        init=init_coords,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=RANDOM_STATE,
        n_epochs=N_EPOCHS,
    ).fit_transform(X_pca)


def compute_metrics(
    X_pca: np.ndarray,
    spatial: np.ndarray,
    labels: np.ndarray,
    embedding: np.ndarray,
    n_neighbors_k: int,
) -> dict:
    tw = sklearn_trustworthiness(X_pca, embedding, n_neighbors=n_neighbors_k)
    ta = random_triplet_accuracy(X_pca, embedding)
    kp = knn_preservation(X_pca, embedding, k=15)
    sna = spatial_neighbor_agreement(spatial, embedding, k=15)
    spatial_m = compute_spatial_metrics(spatial, embedding, labels)
    return {
        "trustworthiness": tw,
        "triplet_accuracy": ta,
        "knn_preservation": kp,
        "sna": sna,
        "morans_i_max": spatial_m["morans_i_max"],
    }


def run_config(
    X_pca: np.ndarray,
    spatial: np.ndarray,
    labels: np.ndarray,
    params: dict,
    default_embeddings: dict[str, np.ndarray],
) -> list[dict]:
    param_swept = params["param_swept"]
    param_value = params["param_value"]
    n_neighbors = params["n_neighbors"]
    min_dist = params["min_dist"]
    metric = params["metric"]
    n_neighbors_k = n_neighbors if param_swept == "n_neighbors" else DEFAULT_N_NEIGHBORS

    print(
        f"  Building graph: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}",
        flush=True,
    )
    graph = build_umap_graph(X_pca, n_neighbors, metric)
    save_graph_for_rust(graph, GRAPH_NPZ.parent)

    rust_init, solver_level = run_rust_export(PROJECT_ROOT)
    py_init = get_python_spectral_init(X_pca, n_neighbors, metric)

    rows = []
    for init_method in INIT_METHODS:
        t0 = time.perf_counter()

        if init_method == "rust_spectral":
            if rust_init is None:
                # Rust export failed — record NaN metrics for this row
                row = {c: float("nan") for c in CSV_COLUMNS}
                row.update(
                    param_swept=param_swept,
                    param_value=param_value,
                    init_method=init_method,
                    solver_level=None,
                    wall_time_s=0.0,
                )
                rows.append(row)
                continue
            init_coords: np.ndarray | str = rust_init
        elif init_method == "python_spectral":
            init_coords = py_init
        elif init_method == "pca":
            init_coords = "pca"
        else:  # random
            init_coords = "random"

        embedding = run_umap_from_init(X_pca, init_coords, n_neighbors, min_dist, metric)
        metrics = compute_metrics(X_pca, spatial, labels, embedding, n_neighbors_k)

        procrustes_rust_vs_python = float("nan")
        if init_method == "rust_spectral":
            _, _, procrustes_rust_vs_python = scipy.spatial.procrustes(py_init, embedding)

        default_emb = default_embeddings[init_method]
        _, _, procrustes_vs_default = scipy.spatial.procrustes(default_emb, embedding)

        wall_time_s = time.perf_counter() - t0
        rows.append({
            "param_swept": param_swept,
            "param_value": param_value,
            "init_method": init_method,
            **metrics,
            "procrustes_rust_vs_python": procrustes_rust_vs_python,
            "procrustes_vs_default": procrustes_vs_default,
            "solver_level": solver_level if init_method == "rust_spectral" else float("nan"),
            "wall_time_s": wall_time_s,
        })

    return rows


def _build_sweep_configs() -> list[dict]:
    configs = []
    for n in N_NEIGHBORS_VALUES:
        configs.append({
            "param_swept": "n_neighbors", "param_value": n,
            "n_neighbors": n, "min_dist": DEFAULT_MIN_DIST, "metric": DEFAULT_METRIC,
        })
    for d in MIN_DIST_VALUES:
        configs.append({
            "param_swept": "min_dist", "param_value": d,
            "n_neighbors": DEFAULT_N_NEIGHBORS, "min_dist": d, "metric": DEFAULT_METRIC,
        })
    for m in METRIC_VALUES:
        configs.append({
            "param_swept": "metric", "param_value": m,
            "n_neighbors": DEFAULT_N_NEIGHBORS, "min_dist": DEFAULT_MIN_DIST, "metric": m,
        })
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="MERFISH UMAP parameter sweep")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--configs", type=int, metavar="N",
        help="Run only the first N sweep configurations",
    )
    group.add_argument(
        "--dry-run", action="store_true",
        help="Run only the first 1 configuration (alias for --configs 1)",
    )
    args = parser.parse_args()
    n_configs: int | None = 1 if args.dry_run else args.configs

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[run_param_sweep] Loading data...", flush=True)
    X_pca, spatial, labels = load_data()

    print("[run_param_sweep] Running default config for reference embeddings...", flush=True)
    default_graph = build_umap_graph(X_pca, DEFAULT_N_NEIGHBORS, DEFAULT_METRIC)
    save_graph_for_rust(default_graph, GRAPH_NPZ.parent)
    rust_init_def, _ = run_rust_export(PROJECT_ROOT)
    py_init_def = get_python_spectral_init(X_pca, DEFAULT_N_NEIGHBORS, DEFAULT_METRIC)

    default_embeddings: dict[str, np.ndarray] = {}
    for init_method in INIT_METHODS:
        if init_method == "rust_spectral":
            ic: np.ndarray | str = rust_init_def if rust_init_def is not None else "random"
        elif init_method == "python_spectral":
            ic = py_init_def
        elif init_method == "pca":
            ic = "pca"
        else:
            ic = "random"
        default_embeddings[init_method] = run_umap_from_init(
            X_pca, ic, DEFAULT_N_NEIGHBORS, DEFAULT_MIN_DIST, DEFAULT_METRIC
        )

    sweep_configs = _build_sweep_configs()
    if n_configs is not None:
        sweep_configs = sweep_configs[:n_configs]

    all_rows: list[dict] = []
    total = len(sweep_configs)
    for i, params in enumerate(sweep_configs, 1):
        print(
            f"[{i:2d}/{total}] param_swept={params['param_swept']!s:<12s} "
            f"value={params['param_value']}",
            flush=True,
        )
        rows = run_config(X_pca, spatial, labels, params, default_embeddings)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
    csv_path = RESULTS_DIR / "results_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[run_param_sweep] Saved {len(df)} rows → {csv_path}", flush=True)


if __name__ == "__main__":
    main()
