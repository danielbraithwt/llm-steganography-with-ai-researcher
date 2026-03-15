#!/usr/bin/env python3
"""Quick smoke test for exp_059 — 2 problems on Qwen only."""
import sys
sys.path.insert(0, '.')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Monkey-patch the config for smoke test
import scripts.exp_059_large_n_chain_length as exp059
exp059.MODELS = [("Qwen/Qwen3-4B-Base", 2)]
exp059.TIME_BUDGET = 300
exp059.RESULTS_DIR = "/tmp/exp059_smoke"

exp059.main()
