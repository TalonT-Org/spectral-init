#!/usr/bin/env python3
"""Download and validate MERFISH Zhuang-ABCA-1 dataset for visual evaluation.

Data is stored at a fixed absolute path so that worktrees and other
working directories all reference the same location — no redundant downloads.

Usage:
    python tests/visual_eval/download_merfish.py          # download all files
    python tests/visual_eval/download_merfish.py --check   # validate only, no download

Data location:
    /home/talon/projects/spectral-init/data/merfish-abca1/

Files downloaded:
    Zhuang-ABCA-1-log2.h5ad   2,128,478,610 bytes  Expression matrix (log2)
    cell_metadata.csv            661,035,596 bytes  Cell metadata (2,846,908 cells)
    gene.csv                          84,677 bytes  Gene panel (1,122 genes)
    ccf_coordinates.csv          220,766,326 bytes  CCF 3D coordinates (2,616,328 cells)

Source: Allen Brain Cell Atlas (CC BY 4.0)
        https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical data location — absolute path, shared across worktrees
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/talon/projects/spectral-init/data/merfish-abca1")

S3_BASE = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"

# (filename, s3_path, expected_size_bytes)
FILES = [
    (
        "Zhuang-ABCA-1-log2.h5ad",
        f"{S3_BASE}/expression_matrices/Zhuang-ABCA-1/20230830/Zhuang-ABCA-1-log2.h5ad",
        2_128_478_610,
    ),
    (
        "cell_metadata.csv",
        f"{S3_BASE}/metadata/Zhuang-ABCA-1/20241115/cell_metadata.csv",
        661_035_596,
    ),
    (
        "gene.csv",
        f"{S3_BASE}/metadata/Zhuang-ABCA-1/20241115/gene.csv",
        84_677,
    ),
    (
        "ccf_coordinates.csv",
        f"{S3_BASE}/metadata/Zhuang-ABCA-1-CCF/20230830/ccf_coordinates.csv",
        220_766_326,
    ),
]

# Expected row counts (excluding header) for CSV validation
EXPECTED_ROWS = {
    "cell_metadata.csv": 2_846_908,
    "gene.csv": 1_122,
    "ccf_coordinates.csv": 2_616_328,
}


def _count_lines(path: Path) -> int:
    """Count lines in a file (fast, no CSV parsing)."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def _download(url: str, dest: Path) -> None:
    """Download a file with progress reporting."""
    print(f"  Downloading {dest.name} ...")
    print(f"    URL: {url}")

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8 * 1024 * 1024  # 8 MB chunks

        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 / total
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    print(
                        f"\r    {mb:.0f} / {total_mb:.0f} MB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
        print()


def validate(quiet: bool = False) -> bool:
    """Validate all files exist with correct sizes and row counts.

    Returns True if all checks pass.
    """
    ok = True
    for filename, _url, expected_size in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            if not quiet:
                print(f"  MISSING: {path}")
            ok = False
            continue

        actual_size = path.stat().st_size
        if actual_size != expected_size:
            if not quiet:
                print(
                    f"  SIZE MISMATCH: {filename} "
                    f"(got {actual_size}, expected {expected_size})"
                )
            ok = False
            continue

        # CSV row count validation
        if filename in EXPECTED_ROWS:
            line_count = _count_lines(path)
            expected_lines = EXPECTED_ROWS[filename] + 1  # +1 for header
            if line_count != expected_lines:
                if not quiet:
                    print(
                        f"  ROW COUNT MISMATCH: {filename} "
                        f"(got {line_count - 1} data rows, "
                        f"expected {EXPECTED_ROWS[filename]})"
                    )
                ok = False
                continue

        if not quiet:
            size_mb = actual_size / (1024 * 1024)
            print(f"  OK: {filename} ({size_mb:.1f} MB)")

    return ok


def download_all() -> None:
    """Download any missing or corrupt files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url, expected_size in FILES:
        path = DATA_DIR / filename

        # Skip if already downloaded and correct size
        if path.exists() and path.stat().st_size == expected_size:
            size_mb = expected_size / (1024 * 1024)
            print(f"  SKIP (already exists): {filename} ({size_mb:.1f} MB)")
            continue

        _download(url, path)

        # Verify size after download
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            print(
                f"  ERROR: {filename} size mismatch after download "
                f"(got {actual_size}, expected {expected_size})"
            )
            sys.exit(1)


def main() -> None:
    check_only = "--check" in sys.argv

    print(f"MERFISH Zhuang-ABCA-1 Data {'Validation' if check_only else 'Download'}")
    print(f"Location: {DATA_DIR}")
    print()

    if check_only:
        if validate():
            print("\nAll files validated.")
        else:
            print("\nValidation FAILED.")
            sys.exit(1)
    else:
        download_all()
        print()
        print("Validating...")
        if validate():
            print("\nAll files downloaded and validated.")
        else:
            print("\nValidation FAILED after download.")
            sys.exit(1)


if __name__ == "__main__":
    main()
