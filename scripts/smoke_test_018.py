#!/usr/bin/env python3
"""Smoke test for exp_018: verify PGD + per-position analysis works on 2 problems."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import exp_018_pgd_position_control as exp

# Override config for smoke test
exp.NUM_PROBLEMS = 8   # Try 8, expect 2-3 valid (Qwen-Base ~50% accuracy)
exp.PGD_STEPS = 10     # Fast PGD
exp.RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "results", "smoke_018")
os.makedirs(exp.RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    exp.main()
