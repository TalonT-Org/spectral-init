#!/usr/bin/env python3
"""Analyze rSVD oversampling coverage experiment results."""

import csv
import os
import sys

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

BUG_FILE_MAP = {
    "B1": "dense.rs",
    "B2": "mod.rs",
    "B3": "scaling.rs",
    "B4": "multi_component.rs",
    "B5": "laplacian.rs",
    "B6": "lobpcg.rs",
    "B7": "rsvd.rs",
    "B8": "sinv.rs",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)
    results_dir = sys.argv[1]
    analyze_rq1(results_dir)
    print()
    analyze_rq2(results_dir)


def analyze_rq1(results_dir):
    path = os.path.join(results_dir, "rq1_oversampling_sweep.csv")
    if not os.path.exists(path):
        print("[RQ1] WARNING: rq1_oversampling_sweep.csv not found — skipping RQ1.")
        return

    print("=" * 72)
    print("RQ1: Oversampling Sweep Summary")
    print("=" * 72)

    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "fixture":      row["fixture"],
                "n":            int(row["n"]),
                "p":            int(row["p"]),
                "residual":     float(row["residual"]),
                "wall_time_s":  float(row["wall_time_s"]),
                "passes":       row["passes"].strip().lower() == "true",
            })

    fixtures = {}
    for r in rows:
        fixtures.setdefault(r["fixture"], []).append(r)

    hdr = f"{'fixture':<28} {'n':>6} {'min_p':>6} {'residual':>12} {'t_p200_s':>10} {'t_min_s':>8} {'speedup':>18}"
    print(hdr)
    print("-" * len(hdr))

    min_p_per_fixture = {}
    for name, frows in fixtures.items():
        passing = [r for r in frows if r["passes"]]
        if not passing:
            print(f"{name:<28}  (no passing p found)")
            continue
        best = min(passing, key=lambda r: r["p"])
        min_p_per_fixture[name] = best["p"]
        row_200 = next((r for r in frows if r["p"] == 200), None)
        if row_200:
            speedup_str = f"{row_200['wall_time_s'] / best['wall_time_s']:.2f}x"
            t200_str    = f"{row_200['wall_time_s']:.3f}"
        else:
            speedup_str = "N/A (p=200 absent)"
            t200_str    = "N/A"
        print(
            f"{name:<28} {best['n']:>6} {best['p']:>6} "
            f"{best['residual']:>12.3e} {t200_str:>10} "
            f"{best['wall_time_s']:>8.3f} {speedup_str:>18}"
        )

    blob_key = "blobs_connected_2000"
    if blob_key in fixtures:
        print()
        print(f"Quality/time tradeoff curve — {blob_key}:")
        print(f"  {'p':>6}  {'residual':>12}  {'wall_time_s':>12}  {'passes':>6}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*6}")
        for r in sorted(fixtures[blob_key], key=lambda r: r["p"]):
            print(f"  {r['p']:>6}  {r['residual']:>12.3e}  {r['wall_time_s']:>12.3f}  {str(r['passes']):>6}")

    if min_p_per_fixture:
        print()
        max_min_p = max(min_p_per_fixture.values())
        if max_min_p <= 10:
            verdict = "H1 CONFIRMED — p≤10 sufficient for all fixtures"
        elif max_min_p <= 30:
            verdict = "H2 CONFIRMED — p≤30 sufficient for all fixtures"
        else:
            verdict = "H0 CONFIRMED — at least one fixture requires p≥100"
        print(f"DECISION: {verdict}")

    quick_path = os.path.join(results_dir, "rq1_quick_scan.csv")
    if os.path.exists(quick_path):
        _print_quick_scan_comparison(rows, quick_path)


def _print_quick_scan_comparison(prod_rows, quick_path):
    quick_rows = []
    with open(quick_path, newline="") as f:
        for row in csv.DictReader(f):
            quick_rows.append({
                "fixture":  row["fixture"],
                "p":        int(row["p"]),
                "residual": float(row["residual"]),
            })
    quick_map = {(r["fixture"], r["p"]): r["residual"] for r in quick_rows}
    print()
    print("Power-iteration benefit (production nbiter=2 vs accurate nbiter=6):")
    print(f"  {'fixture':<28} {'p':>6}  {'prod_res':>12}  {'acc_res':>12}  {'ratio':>8}")
    print(f"  {'-'*28} {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}")
    for r in sorted(prod_rows, key=lambda r: (r["fixture"], r["p"])):
        key = (r["fixture"], r["p"])
        if key in quick_map:
            acc = quick_map[key]
            ratio = r["residual"] / acc if acc > 0 else float("inf")
            print(f"  {r['fixture']:<28} {r['p']:>6}  {r['residual']:>12.3e}  {acc:>12.3e}  {ratio:>8.2f}")


def analyze_rq2(results_dir):
    path = os.path.join(results_dir, "rq2_mutation_matrix.csv")
    if not os.path.exists(path):
        print("[RQ2] WARNING: rq2_mutation_matrix.csv not found — skipping RQ2.")
        return

    print("=" * 72)
    print("RQ2: Mutation Detection Matrix")
    print("=" * 72)

    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "bug_id":             row["bug_id"],
                "target_file":        BUG_FILE_MAP.get(row["bug_id"], "unknown"),
                "detected":           row["detected"].strip().lower() == "true",
                "suite_wall_time_s":  float(row["suite_wall_time_s"]),
            })

    print(f"{'bug_id':<8} {'target_file':<22} {'detected':>8} {'suite_wall_time_s':>18}")
    print(f"{'-'*8} {'-'*22} {'-'*8} {'-'*18}")
    for r in rows:
        print(
            f"{r['bug_id']:<8} {r['target_file']:<22} "
            f"{'YES' if r['detected'] else 'NO':>8} "
            f"{r['suite_wall_time_s']:>18.3f}"
        )

    n_detected = sum(1 for r in rows if r["detected"])
    total_time = sum(r["suite_wall_time_s"] for r in rows)
    total_bugs = len(rows)
    print()
    print(f"Detection rate: {n_detected}/{total_bugs}")
    print(f"Total suite wall time: {total_time:.1f}s")
    print()
    if n_detected >= 6 and total_time <= 30.0:
        verdict = "H1 CONFIRMED — promote invariant tests to permanent suite"
    else:
        verdict = "H0 CONFIRMED — supplement with chain-level tests"
    print(f"DECISION: {verdict}")


if __name__ == "__main__":
    main()
