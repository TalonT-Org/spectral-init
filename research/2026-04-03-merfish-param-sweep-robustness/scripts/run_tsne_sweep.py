#!/usr/bin/env python3
"""run_tsne_sweep.py — 5-run t-SNE perplexity sweep (Phase 4)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openTSNE import TSNE
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

sys.path.insert(0, str(Path(__file__).parents[3] / "tests" / "visual_eval"))
from global_metrics import random_triplet_accuracy, knn_preservation  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parents[1] / "data"
RESULTS_DIR = Path(__file__).parents[1] / "results"

# ── Sweep constants ──────────────────────────────────────────────────────────
PERPLEXITIES = [5, 15, 30, 50, 100]
RANDOM_STATE = 42
CSV_COLUMNS = [
    "perplexity", "trustworthiness", "triplet_accuracy",
    "knn_preservation", "wall_time_s",
]


def load_data() -> tuple[np.ndarray, np.ndarray]:
    X_pca = np.load(DATA_DIR / "merfish_10k_pca.npy").astype(np.float64)
    spatial = np.load(DATA_DIR / "merfish_10k_spatial.npz")["arr_0"]
    return X_pca, spatial


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[run_tsne_sweep] Loading data...", flush=True)
    X_pca, _spatial = load_data()

    all_rows: list[dict] = []
    total = len(PERPLEXITIES)
    for i, perplexity in enumerate(PERPLEXITIES, 1):
        print(f"[{i}/{total}] perplexity={perplexity}", flush=True)
        t0 = time.perf_counter()

        tsne = TSNE(
            perplexity=perplexity,
            initialization="pca",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        emb = np.array(tsne.fit(X_pca))

        wall_time_s = time.perf_counter() - t0

        tw = sklearn_trustworthiness(X_pca, emb, n_neighbors=15)
        ta = random_triplet_accuracy(X_pca, emb)
        kp = knn_preservation(X_pca, emb, k=15)

        all_rows.append({
            "perplexity": perplexity,
            "trustworthiness": tw,
            "triplet_accuracy": ta,
            "knn_preservation": kp,
            "wall_time_s": wall_time_s,
        })

    df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
    csv_path = RESULTS_DIR / "results_tsne.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[run_tsne_sweep] Saved {len(df)} rows → {csv_path}", flush=True)


if __name__ == "__main__":
    main()
