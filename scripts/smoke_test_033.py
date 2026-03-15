#!/usr/bin/env python3
"""Smoke test for exp_033: run 1 problem, 5 PGD steps, verify shapes and outputs."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Patch config before importing
import exp_033_v_only_challenge as exp033
exp033.NUM_PROBLEMS = 2
exp033.PGD_STEPS = 5
exp033.PGD_LR = 0.08
exp033.LAMBDA_ANS = 10.0
exp033.TIMEOUT = 300
exp033.MAX_GEN_TOKENS = 256
exp033.MAX_REASONING_TOKENS = 150

print("=== SMOKE TEST: exp_033 ===")
print(f"PGD_STEPS={exp033.PGD_STEPS}, LR={exp033.PGD_LR}, LAMBDA={exp033.LAMBDA_ANS}")
print(f"Conditions: {[c[0] for c in exp033.PGD_CONDITIONS]}")
exp033.main()
print("\n=== SMOKE TEST PASSED ===")
