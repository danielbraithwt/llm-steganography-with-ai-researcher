#!/usr/bin/env python3
"""Smoke test for exp_001 — run 2 problems only to catch bugs."""
import sys
import os

# Override config for smoke test
sys.path.insert(0, os.path.dirname(__file__))
import exp_001_double_dissociation as exp
exp.NUM_PROBLEMS = 2
exp.RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_001_smoke")
os.makedirs(exp.RESULTS_DIR, exist_ok=True)

exp.main()
