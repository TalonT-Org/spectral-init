# Data for the MERFISH parameter-sweep experiment

The scripts in `../scripts/` (`run_param_sweep.py`, `run_tsne_sweep.py`) read
three files from this directory. They are not stored here; copy or link them
from the locations below before running the scripts.

| File expected here | Source | Regenerate with |
|---|---|---|
| `merfish_10k_labels.npz` | `tests/visual_eval/merfish_data/merfish_10k_labels.npz` (tracked) | `python tests/visual_eval/generate_merfish_subset.py` |
| `merfish_10k_spatial.npz` | `tests/visual_eval/merfish_data/merfish_10k_spatial.npz` (tracked) | `python tests/visual_eval/generate_merfish_subset.py` |
| `merfish_10k_pca.npy` | `tests/visual_eval/output/merfish_10k_pca.npy` (untracked, generated) | `python tests/visual_eval/generate_merfish_comparisons.py` |

From the repository root:

```sh
cp tests/visual_eval/merfish_data/merfish_10k_labels.npz \
   tests/visual_eval/merfish_data/merfish_10k_spatial.npz \
   tests/visual_eval/output/merfish_10k_pca.npy \
   research/2026-04-03-merfish-param-sweep-robustness/data/
```

The experiment scripts reproduce at the `research-baseline-pre-0.1` tag; see
`research/README.md`.
