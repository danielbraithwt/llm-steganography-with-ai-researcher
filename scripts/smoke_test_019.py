#!/usr/bin/env python3
"""Smoke test for exp_019 with 3 problems."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import exp_019_double_dissociation as exp
exp.NUM_PROBLEMS = 15
exp.NOISE_FRACTIONS = [0.05]
exp.RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_019_smoke")
exp.main()
