#!/usr/bin/env python3
"""Smoke test for exp_003: run on 3 problems, 2 fractions to verify correctness."""

import sys
import os

# Override config for smoke test
import scripts.exp_003_pruning_sweep as exp003

exp003.NUM_PROBLEMS = 3
exp003.PRUNE_FRACTIONS = [0.50, 0.90]
exp003.NOISE_FRACTION = 0.90
exp003.RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "results", "exp_003_smoke")
os.makedirs(exp003.RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    exp003.main()
