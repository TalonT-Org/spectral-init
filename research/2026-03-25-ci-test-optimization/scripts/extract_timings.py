#!/usr/bin/env python3
"""Extract per-testcase timings from a JUnit XML report.

Usage:
    python3 extract_timings.py <junit_xml_path> [output_csv_path]

Outputs CSV with columns: suite,test_name,classname,time_s,status
sorted by time_s descending. When the input is empty or unparseable,
outputs only the CSV header row (exit 0).
"""
import csv
import sys
import xml.etree.ElementTree as ET

FIELDNAMES = ["suite", "test_name", "classname", "time_s", "status"]


def _status(tc_elem):
    if tc_elem.find("failure") is not None:
        return "failed"
    if tc_elem.find("error") is not None:
        return "error"
    if tc_elem.find("skipped") is not None:
        return "skipped"
    return "passed"


def extract(xml_path):
    rows = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return rows
    root = tree.getroot()
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = root.findall("testsuite")
    for suite in suites:
        suite_name = suite.get("name", "")
        for tc in suite.findall("testcase"):
            rows.append({
                "suite": suite_name,
                "test_name": tc.get("name", ""),
                "classname": tc.get("classname", ""),
                "time_s": float(tc.get("time", "0") or "0"),
                "status": _status(tc),
            })
    rows.sort(key=lambda r: r["time_s"], reverse=True)
    return rows


def write_csv(rows, out):
    writer = csv.DictWriter(out, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <junit_xml_path> [output_csv_path]",
              file=sys.stderr)
        sys.exit(1)
    rows = extract(sys.argv[1])
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", newline="") as f:
            write_csv(rows, f)
    else:
        write_csv(rows, sys.stdout)


if __name__ == "__main__":
    main()
