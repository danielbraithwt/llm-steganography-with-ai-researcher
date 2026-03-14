#!/usr/bin/env python3
"""Smoke test for exp_016 — run 2 problems to verify pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Monkey-patch config before importing
import exp_016_positional_confound as exp
exp.NUM_PROBLEMS = 3
exp.START_IDX = 65
exp.NOISE_FRACTIONS = [0.03]

if __name__ == "__main__":
    exp.RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "smoke_016")
    os.makedirs(exp.RESULTS_DIR, exist_ok=True)
    exp.main()
