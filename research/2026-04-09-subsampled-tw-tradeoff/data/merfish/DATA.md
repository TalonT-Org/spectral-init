# MERFISH data for the subsampled-trustworthiness experiment

The scripts in `../../scripts/` (`compute_exact.py`, `run_subsampling.py`,
`verify_normalization.py`, through `utils.load_npy_pair`) load
`merfish_n{N}_x.npy` and `merfish_n{N}_y.npy` from this directory.

`merfish_n20000_{x,y}.npy` are stored here. The 10 000- and 50 000-cell pairs
are not: they are the arrays of an earlier experiment and have to be copied or
linked from it before the scripts are run.

| File expected here | Source (tracked) | Regenerate with |
|---|---|---|
| `merfish_n10000_x.npy` | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy` | `research/2026-04-05-tw-perf-rerun-clean/scripts/prepare_merfish.py` |
| `merfish_n10000_y.npy` | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy` | `research/2026-04-05-tw-perf-rerun-clean/scripts/prepare_merfish.py` |
| `merfish_n50000_x.npy` | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_x.npy` | `research/2026-04-05-tw-perf-rerun-clean/scripts/prepare_merfish_50k.py` |
| `merfish_n50000_y.npy` | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n50k_y.npy` | `research/2026-04-05-tw-perf-rerun-clean/scripts/prepare_merfish_50k.py` |

From the repository root:

```sh
src=research/2026-04-05-tw-perf-rerun-clean/data/merfish
dst=research/2026-04-09-subsampled-tw-tradeoff/data/merfish
cp "$src/merfish_n10k_x.npy" "$dst/merfish_n10000_x.npy"
cp "$src/merfish_n10k_y.npy" "$dst/merfish_n10000_y.npy"
cp "$src/merfish_n50k_x.npy" "$dst/merfish_n50000_x.npy"
cp "$src/merfish_n50k_y.npy" "$dst/merfish_n50000_y.npy"
```

The experiment scripts reproduce at the `research-baseline-pre-0.1` tag; see
`research/README.md`.
