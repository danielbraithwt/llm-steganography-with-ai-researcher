#!/usr/bin/env python3
"""Smoke test for exp_022 — run 3 problems, 4 conditions."""
import sys
sys.path.insert(0, '.')

# Monkey-patch the config
import scripts.exp_022_geometric_dissociation as exp
exp.NUM_PROBLEMS = 30  # try 30, stop after 3 valid
exp.CONDITIONS = [
    ("dir_kv_late",       "late",  "kv", "direction", 0.0),
    ("mag_kv_10_late",    "late",  "kv", "magnitude", 1.0),
    ("dir_kv_early",      "early", "kv", "direction", 0.0),
    ("mag_kv_10_early",   "early", "kv", "magnitude", 1.0),
]

# Override time limit to be generous for smoke test
import time
_orig_main = exp.main

def smoke_main():
    # Just run with short time limit
    exp.main()

if __name__ == "__main__":
    smoke_main()
