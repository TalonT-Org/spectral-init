"""Tests for analyze_results.py and utils.py.

Usage:
    micromamba run -n subsampled-tw-rust python scripts/test_analyze_results.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def make_trial_json(mode, n, m, k, seed, t_exact, t_sub,
                    wall_exact_ms, wall_sub_ms):
    """Generate a synthetic trial JSON dict matching the binary's output schema."""
    abs_delta_t = abs(t_exact - t_sub) if t_sub is not None else None
    return {
        "n": n,
        "m": m,
        "k": k,
        "seed": seed,
        "mode": mode,
        "t_exact": t_exact,
        "t_sub": t_sub,
        "abs_delta_t": abs_delta_t,
        "wall_exact_ms": wall_exact_ms,
        "wall_sub_ms": wall_sub_ms,
        "warmup_exact_ms": 100.0 if wall_exact_ms else None,
        "warmup_sub_ms": 50.0 if wall_sub_ms else None,
        "cpu_model": "test",
        "core_count": 4,
        "rust_version": "test",
        "git_commit": "test",
    }


def write_trial(raw_dir, filename, record):
    """Write a single trial JSON file."""
    (raw_dir / filename).write_text(json.dumps(record, indent=2))


def test_utils_constants():
    """T2: Verify utils.py constants match specification."""
    from utils import (
        K, SEEDS, M_VALUES_10K, M_VALUES_50K,
        PYTHON_SPEEDUP_10K, PYTHON_MEAN_DELTA_T_10K_M2000,
    )
    assert K == 15, f"K={K}, expected 15"
    assert len(SEEDS) == 10, f"len(SEEDS)={len(SEEDS)}, expected 10"
    assert len(M_VALUES_10K) == 7, f"len(M_VALUES_10K)={len(M_VALUES_10K)}"
    assert M_VALUES_10K[0] == 500, f"M_VALUES_10K[0]={M_VALUES_10K[0]}"
    assert M_VALUES_10K[-1] == 10000, f"M_VALUES_10K[-1]={M_VALUES_10K[-1]}"
    assert len(M_VALUES_50K) == 7, f"len(M_VALUES_50K)={len(M_VALUES_50K)}"
    assert set(PYTHON_SPEEDUP_10K.keys()) == {500, 1000, 2000, 5000}, \
        f"PYTHON_SPEEDUP_10K keys={set(PYTHON_SPEEDUP_10K.keys())}"
    assert PYTHON_MEAN_DELTA_T_10K_M2000 == 0.00165, \
        f"PYTHON_MEAN_DELTA_T_10K_M2000={PYTHON_MEAN_DELTA_T_10K_M2000}"
    print("PASS: test_utils_constants")


def test_full_verdicts():
    """T1: Synthetic data producing real verdicts on all hypotheses."""
    import analyze_results

    tmpdir = Path(tempfile.mkdtemp(prefix="tw_test_full_"))
    try:
        raw_dir = tmpdir / "results" / "raw"
        raw_dir.mkdir(parents=True)
        analysis_dir = tmpdir / "results" / "analysis"

        # --- Sanity trials (H6 should PASS) ---
        write_trial(raw_dir, "sanity_n10000.json", make_trial_json(
            mode="sanity", n=10000, m=10000, k=15, seed=None,
            t_exact=0.95, t_sub=0.95,  # abs_delta_t = 0 < 1e-10
            wall_exact_ms=None, wall_sub_ms=None,
        ))
        write_trial(raw_dir, "sanity_n50000.json", make_trial_json(
            mode="sanity", n=50000, m=50000, k=15, seed=None,
            t_exact=0.96, t_sub=0.96,
            wall_exact_ms=None, wall_sub_ms=None,
        ))

        # --- Exact baselines ---
        write_trial(raw_dir, "exact_n10000.json", make_trial_json(
            mode="exact", n=10000, m=None, k=15, seed=None,
            t_exact=0.95, t_sub=None,
            wall_exact_ms=[3500.0, 3600.0, 3550.0, 3580.0, 3520.0],
            wall_sub_ms=None,
        ))
        write_trial(raw_dir, "exact_n50000.json", make_trial_json(
            mode="exact", n=50000, m=None, k=15, seed=None,
            t_exact=0.96, t_sub=None,
            wall_exact_ms=[91000.0, 91500.0, 91200.0, 91300.0, 91100.0],
            wall_sub_ms=None,
        ))

        # --- Subsample trials for H1 (n=10K, m=2000): 10 seeds ---
        # abs_delta_t ~ 0.002 (well within 0.01 threshold)
        for seed in range(10):
            delta = 0.001 + seed * 0.0002  # range [0.001, 0.0028]
            t_sub = 0.95 - delta
            write_trial(raw_dir, f"trial_n10000_m2000_s{seed}.json", make_trial_json(
                mode="subsample", n=10000, m=2000, k=15, seed=seed,
                t_exact=0.95, t_sub=t_sub,
                wall_exact_ms=[3500.0, 3600.0, 3550.0, 3580.0, 3520.0],
                wall_sub_ms=[850.0, 870.0, 860.0, 855.0, 865.0],
            ))

        # --- Subsample trials for H2/H3: additional m-values for n=10K ---
        # Speedup ~ n/m, variance decays with m
        m_speedup_map_10k = {
            500: (170.0, 0.005),     # fast subsample, larger error
            1000: (350.0, 0.003),
            3000: (1200.0, 0.0015),
            5000: (2000.0, 0.001),
            7500: (2800.0, 0.0008),
            10000: (3500.0, 0.0003),  # m=n, almost exact
        }
        for m_val, (wall_sub, base_delta) in m_speedup_map_10k.items():
            for seed in range(10):
                delta = base_delta + seed * 0.0001
                t_sub = 0.95 - delta
                write_trial(raw_dir, f"trial_n10000_m{m_val}_s{seed}.json", make_trial_json(
                    mode="subsample", n=10000, m=m_val, k=15, seed=seed,
                    t_exact=0.95, t_sub=t_sub,
                    wall_exact_ms=[3500.0, 3600.0, 3550.0, 3580.0, 3520.0],
                    wall_sub_ms=[wall_sub, wall_sub * 1.02, wall_sub * 0.99,
                                 wall_sub * 1.01, wall_sub * 0.98],
                ))

        # --- Subsample trials for H5 (n=50K, m=2000): 10 seeds ---
        for seed in range(10):
            delta = 0.001 + seed * 0.0002
            t_sub = 0.96 - delta
            write_trial(raw_dir, f"trial_n50000_m2000_s{seed}.json", make_trial_json(
                mode="subsample", n=50000, m=2000, k=15, seed=seed,
                t_exact=0.96, t_sub=t_sub,
                wall_exact_ms=[91000.0, 91500.0, 91200.0, 91300.0, 91100.0],
                wall_sub_ms=[3600.0, 3650.0, 3620.0, 3610.0, 3640.0],
            ))

        # --- Additional n=50K m-values for H2 stratum ---
        m_speedup_map_50k = {
            1000: (1800.0, 0.004),
            5000: (4500.0, 0.002),
            10000: (9000.0, 0.001),
            20000: (18000.0, 0.0005),
            35000: (45000.0, 0.0002),
            50000: (91000.0, 0.00005),
        }
        for m_val, (wall_sub, base_delta) in m_speedup_map_50k.items():
            for seed in range(10):
                delta = base_delta + seed * 0.0001
                t_sub = 0.96 - delta
                write_trial(raw_dir, f"trial_n50000_m{m_val}_s{seed}.json", make_trial_json(
                    mode="subsample", n=50000, m=m_val, k=15, seed=seed,
                    t_exact=0.96, t_sub=t_sub,
                    wall_exact_ms=[91000.0, 91500.0, 91200.0, 91300.0, 91100.0],
                    wall_sub_ms=[wall_sub, wall_sub * 1.02, wall_sub * 0.99,
                                 wall_sub * 1.01, wall_sub * 0.98],
                ))

        # Run analysis
        verdicts = analyze_results.main(exproot=tmpdir)

        # Assert verdicts.json exists and is valid
        verdicts_path = analysis_dir / "verdicts.json"
        assert verdicts_path.exists(), "verdicts.json not created"
        loaded = json.loads(verdicts_path.read_text())
        assert "hypotheses" in loaded
        for hkey in ["H1", "H2", "H3", "H4", "H5", "H6"]:
            assert hkey in loaded["hypotheses"], f"Missing hypothesis {hkey}"
            assert "verdict" in loaded["hypotheses"][hkey], f"No verdict in {hkey}"

        # H6 should PASS (sanity data is deterministically exact)
        assert loaded["hypotheses"]["H6"]["verdict"] == "PASS", \
            f"H6 verdict={loaded['hypotheses']['H6']['verdict']}, expected PASS"

        # H1 should PASS (synthetic data well within threshold)
        assert loaded["hypotheses"]["H1"]["verdict"] == "PASS", \
            f"H1 verdict={loaded['hypotheses']['H1']['verdict']}, expected PASS"

        # H1 should have documented fields
        h1 = loaded["hypotheses"]["H1"]
        for field in ["t_statistic", "p_value", "ci_upper_97_5",
                      "mean_abs_delta_T", "max_abs_delta_T", "n_seeds"]:
            assert field in h1, f"H1 missing field: {field}"

        # summary.md exists and is non-empty
        summary_path = analysis_dir / "summary.md"
        assert summary_path.exists(), "summary.md not created"
        assert len(summary_path.read_text()) > 0, "summary.md is empty"

        # Three PNG plots exist
        for png_name in ["error_vs_m.png", "speedup_vs_m.png", "variance_decay.png"]:
            png_path = analysis_dir / png_name
            assert png_path.exists(), f"{png_name} not created"

        print("PASS: test_full_verdicts")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_insufficient_data():
    """T1 (INSUFFICIENT_DATA path): With only 2 trials, H1-H5 produce INSUFFICIENT_DATA."""
    import analyze_results

    tmpdir = Path(tempfile.mkdtemp(prefix="tw_test_insuf_"))
    try:
        raw_dir = tmpdir / "results" / "raw"
        raw_dir.mkdir(parents=True)

        # 1 sanity trial (H6 can produce a real verdict from just 1)
        write_trial(raw_dir, "sanity_n10000.json", make_trial_json(
            mode="sanity", n=10000, m=10000, k=15, seed=None,
            t_exact=0.95, t_sub=0.95,
            wall_exact_ms=None, wall_sub_ms=None,
        ))

        # Only 2 subsample trials (below minimum of 3 for t-test)
        for seed in range(2):
            write_trial(raw_dir, f"trial_n10000_m2000_s{seed}.json", make_trial_json(
                mode="subsample", n=10000, m=2000, k=15, seed=seed,
                t_exact=0.95, t_sub=0.948,
                wall_exact_ms=[3500.0, 3600.0, 3550.0, 3580.0, 3520.0],
                wall_sub_ms=[850.0, 870.0, 860.0, 855.0, 865.0],
            ))

        verdicts = analyze_results.main(exproot=tmpdir)
        h = verdicts["hypotheses"]

        # H1 should be INSUFFICIENT_DATA (only 2 trials at target cell)
        assert h["H1"]["verdict"] == "INSUFFICIENT_DATA", \
            f"H1 verdict={h['H1']['verdict']}, expected INSUFFICIENT_DATA"
        assert "reason" in h["H1"], "H1 INSUFFICIENT_DATA missing reason"

        # H5 should be INSUFFICIENT_DATA (no n=50K trials)
        assert h["H5"]["verdict"] == "INSUFFICIENT_DATA", \
            f"H5 verdict={h['H5']['verdict']}, expected INSUFFICIENT_DATA"

        # H2 should be INSUFFICIENT_DATA (only 1 m-value)
        assert h["H2"]["verdict"] in ("INSUFFICIENT_DATA", "FAIL"), \
            f"H2 verdict={h['H2']['verdict']}"

        # H6 should still produce a real verdict (PASS from the sanity trial)
        assert h["H6"]["verdict"] == "PASS", \
            f"H6 verdict={h['H6']['verdict']}, expected PASS"

        print("PASS: test_insufficient_data")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_shell_syntax():
    """T3: Verify run_experiment.sh passes bash -n syntax check."""
    import subprocess
    script_path = Path(__file__).parent / "run_experiment.sh"
    assert script_path.exists(), f"run_experiment.sh not found at {script_path}"
    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, \
        f"bash -n failed: {result.stderr}"
    print("PASS: test_shell_syntax")


if __name__ == "__main__":
    test_utils_constants()
    test_full_verdicts()
    test_insufficient_data()
    test_shell_syntax()
    print("\nAll tests passed.")
