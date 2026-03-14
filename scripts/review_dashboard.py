#!/usr/bin/env python3
"""
review_dashboard.py — CLI tool to review research agent progress.

Reads the evidence ledger and experiment logs, prints a formatted summary.

Usage:
    python scripts/review_dashboard.py [--since N]
"""
import argparse
import glob
import json
import os
import re
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Research agent review dashboard")
    parser.add_argument("--since", type=int, default=0,
                        help="Show only the last N cycles (0 = all)")
    parser.add_argument("--project-dir", default=".",
                        help="Project root directory")
    return parser.parse_args()


def read_evidence_ledger(project_dir):
    """Read and parse the evidence ledger."""
    path = os.path.join(project_dir, "evidence_ledger.md")
    if not os.path.exists(path):
        return "(evidence ledger not found)"
    with open(path) as f:
        return f.read()


def read_latest_status(project_dir):
    """Read latest_status.json."""
    path = os.path.join(project_dir, "experiment_log", "latest_status.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_experiment_logs(project_dir, since=0):
    """List experiment log files, optionally filtered to last N."""
    pattern = os.path.join(project_dir, "experiment_log", "exp_*.md")
    files = sorted(glob.glob(pattern))
    if since > 0:
        files = files[-since:]
    return files


def extract_table(content):
    """Extract markdown table rows from evidence ledger."""
    lines = content.split("\n")
    table_lines = []
    in_table = False
    for line in lines:
        if "|" in line and "---" not in line:
            if "Claim" in line:
                in_table = True
            if in_table:
                table_lines.append(line)
        elif in_table and "|" not in line:
            break
    return table_lines


def print_header(text):
    """Print a section header."""
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_progress_bar(label, value, max_val=5):
    """Print a progress bar."""
    filled = "\u25a0" * value
    empty = "\u25a1" * (max_val - value)
    print(f"  {filled}{empty}  {label}")


def scan_for_flags(log_content):
    """Check if an experiment log has flags for human review."""
    keywords = ["needs human review", "unexpected", "contradiction",
                "flag for human", "error", "FAILED", "limitation"]
    found = []
    for kw in keywords:
        if kw.lower() in log_content.lower():
            found.append(kw)
    return found


def main():
    args = parse_args()
    project_dir = args.project_dir

    # Header
    print_header("RESEARCH AGENT DASHBOARD")

    # Latest status
    status = read_latest_status(project_dir)
    if status:
        print(f"\n  Last cycle:     {status.get('cycle', '?')}")
        print(f"  Timestamp:      {status.get('timestamp', '?')}")
        print(f"  Type:           {status.get('type', '?')}")
        print(f"  Last template:  {status.get('template', '?')}")
        print(f"  Key finding:    {status.get('key_finding', '?')}")
        print(f"  Next planned:   {status.get('next_planned', '?')}")

        if "progress" in status:
            print()
            print("  Progress:")
            for claim, info in status["progress"].items():
                bar_val = info.get("bar", 0)
                claim_display = claim.replace("_", " ").title()
                status_str = info.get("status", "?")
                print_progress_bar(f"{claim_display} ({status_str})", bar_val)
    else:
        print("\n  No cycles completed yet.")

    # Evidence ledger summary
    print_header("EVIDENCE SUMMARY")
    ledger = read_evidence_ledger(project_dir)
    # Print the summary section (up to first ---)
    summary = ledger.split("---")[0] if "---" in ledger else ledger
    # Remove the top-level header
    summary_lines = summary.strip().split("\n")
    for line in summary_lines:
        if line.strip():
            print(f"  {line}")

    # Experiment logs
    log_files = list_experiment_logs(project_dir, args.since)
    label = f"EXPERIMENT LOGS (last {args.since})" if args.since else "ALL EXPERIMENT LOGS"
    print_header(label)

    if not log_files:
        print("\n  No experiment logs found.")
    else:
        for log_file in reversed(log_files):
            basename = os.path.basename(log_file)
            with open(log_file) as f:
                content = f.read()

            # Extract first line as title
            first_line = content.strip().split("\n")[0] if content.strip() else "(empty)"
            flags = scan_for_flags(content)

            flag_str = ""
            if flags:
                flag_str = f"  [!] FLAGS: {', '.join(flags)}"

            print(f"\n  {basename}: {first_line}{flag_str}")

    # Flags summary
    print_header("FLAGS FOR HUMAN REVIEW")
    all_logs = list_experiment_logs(project_dir, 0)
    any_flags = False
    for log_file in all_logs:
        with open(log_file) as f:
            content = f.read()
        flags = scan_for_flags(content)
        if flags:
            print(f"  [!] {os.path.basename(log_file)}: {', '.join(flags)}")
            any_flags = True
    if not any_flags:
        print("  (none)")

    print()


if __name__ == "__main__":
    main()
