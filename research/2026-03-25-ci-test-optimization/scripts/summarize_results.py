#!/usr/bin/env python3
"""Summarize all CSV results from the CI test optimization experiment.

Usage (run from the experiment root):
    python3 scripts/summarize_results.py

Reads results/**/*.csv. Prints a WARNING for each expected RQ directory
that contains no CSV files. Outputs a markdown table per CSV file; when
a CSV is empty (header only), prints a 'no data' skeleton row.
"""
import csv
import sys
from pathlib import Path

EXPECTED_RQS = [
    "rq1_opt_levels",
    "rq2_min_set",
    "rq3_threshold",
    "rq4_property_tests",
]

RESULTS_DIR = Path("results")


def _print_table(fieldnames, rows):
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join("---" for _ in fieldnames) + " |"
    print(header)
    print(sep)
    if not rows:
        no_data = ["no data"] + [""] * (len(fieldnames) - 1)
        print("| " + " | ".join(no_data) + " |")
    else:
        for row in rows:
            print("| " + " | ".join(str(row.get(f, "")) for f in fieldnames) + " |")


def summarize_rq(rq_name):
    rq_dir = RESULTS_DIR / rq_name
    csvs = sorted(rq_dir.glob("*.csv")) if rq_dir.exists() else []

    print(f"\n## {rq_name}")

    if not csvs:
        print(f"WARNING: no CSV files found in {rq_dir}", file=sys.stderr)
        print(f"WARNING: no CSV files found in {rq_dir}")
        _print_table(["file", "(no data)"], [])
        return

    for csv_path in csvs:
        print(f"\n### {csv_path.name}")
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
        except Exception as exc:
            print(f"WARNING: could not read {csv_path}: {exc}")
            continue
        if not fieldnames:
            _print_table(["(no data)"], [])
        else:
            _print_table(fieldnames, rows)


def main():
    if not RESULTS_DIR.exists():
        print(
            "WARNING: results/ directory not found. Run from the experiment root "
            f"(research/2026-03-25-ci-test-optimization/).",
            file=sys.stderr,
        )
    for rq in EXPECTED_RQS:
        summarize_rq(rq)


if __name__ == "__main__":
    main()
