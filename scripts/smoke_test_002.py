#!/usr/bin/env python3
"""Smoke test for exp_002 — run 3 problems to verify fixes."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import exp_002_double_dissociation as exp

exp.NUM_PROBLEMS = 3
exp.RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_002_smoke")
os.makedirs(exp.RESULTS_DIR, exist_ok=True)

exp.main()
