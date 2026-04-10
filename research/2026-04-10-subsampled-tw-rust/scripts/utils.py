"""Shared constants for subsampled-tw-rust experiment.

Usage:
    micromamba run -n subsampled-tw-rust python scripts/analyze_results.py
"""

from pathlib import Path

EXPROOT = Path(__file__).resolve().parent.parent

# -- Experiment constants -----------------------------------------------------
K = 15
SEEDS = list(range(10))

M_VALUES_10K = [500, 1000, 2000, 3000, 5000, 7500, 10000]
M_VALUES_50K = [1000, 2000, 5000, 10000, 20000, 35000, 50000]

# -- Python reference values (from PR #260) -----------------------------------
PYTHON_SPEEDUP_10K = {500: 18.2, 1000: 9.1, 2000: 4.1, 5000: 1.7}
PYTHON_MEAN_DELTA_T_10K_M2000 = 0.00165

# -- Derived ------------------------------------------------------------------
N_LABEL = {10000: "n10k", 50000: "n50k"}
M_VALUES = {10000: M_VALUES_10K, 50000: M_VALUES_50K}
