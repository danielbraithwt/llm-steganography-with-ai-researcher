#!/usr/bin/env python3
"""Smoke test for exp_028 — run 3 problems with all conditions."""
import sys
sys.path.insert(0, '.')

# Monkey-patch to run just 3 problems with short timeout
import scripts.exp_028_position_kv_interaction as exp
exp.NUM_PROBLEMS = 30  # attempt 30, should get 3 valid quickly
exp.RESULTS_DIR = "/tmp/smoke_028"

# Override time limit
import time
_orig_main = exp.main

def patched_main():
    import builtins
    _orig_time = time.time
    _start = _orig_time()

    # Run with very short time limit by reducing the check
    old_conditions = exp.CONDITIONS
    # Test just 3 conditions for speed: K-early, V-early, K-late
    exp.CONDITIONS = [
        ("dir_k_early",  "early", 0.05, "k"),
        ("dir_v_early",  "early", 0.05, "v"),
        ("dir_k_late",   "late",  0.05, "k"),
    ]
    exp.main()

if __name__ == "__main__":
    patched_main()
