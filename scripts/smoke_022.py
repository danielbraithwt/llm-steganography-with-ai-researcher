#!/usr/bin/env python3
"""Smoke test for exp_022 - runs 5 problems with 3 conditions."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import exp_022_geometric_dissociation as exp

exp.NUM_PROBLEMS = 5
exp.RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "results", "exp_022_smoke")
exp.CONDITIONS = [
    ("dir_kv_late",    "late",  "kv", "direction", 0.0),
    ("mag_kv_10_late", "late",  "kv", "magnitude", 1.0),
    ("mag_kv_14_late", "late",  "kv", "magnitude", 1.414),
]

exp.main()
