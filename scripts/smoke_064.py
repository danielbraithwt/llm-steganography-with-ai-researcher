#!/usr/bin/env python3
"""Smoke test for exp 064 — 2 problems on Qwen only."""
import sys
sys.path.insert(0, '.')
import scripts.exp_064_per_head_spectral as exp

exp.N_PROBLEMS = 2
exp.MODELS = ['Qwen/Qwen3-4B-Base']
exp.TIME_BUDGET = 300
exp.RESULTS_DIR = '/tmp/exp_064_smoke'

exp.main()
