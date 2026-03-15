#!/usr/bin/env python3
"""Smoke test for exp_058 — test imports, selection methods, and profile analysis."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import exp_058_sink_analysis_budget_sweep as exp

print("Module imported OK")
print(f"BUDGETS: {exp.BUDGETS}")
print(f"BASE_SELECTIONS: {exp.BASE_SELECTIONS}")
print(f"MODIFIED_SELECTIONS: {exp.MODIFIED_SELECTIONS}")

conditions = []
for budget in exp.BUDGETS:
    for sel in exp.BASE_SELECTIONS:
        conditions.append(("mask", sel, budget))
for budget in exp.MODIFIED_BUDGETS:
    for sel in exp.MODIFIED_SELECTIONS:
        conditions.append(("mask", sel, budget))
print(f"Total conditions: {len(conditions)}")

reasoning_len = 100
prompt_len = 500
k_norm = np.random.randn(600)
true_h2o_all = np.random.randn(100)
true_h2o_late = np.random.randn(100)

for sel in exp.BASE_SELECTIONS + exp.MODIFIED_SELECTIONS:
    keep_pos, evict_pos, keep_idx = exp.select_positions(
        sel, 0.33, reasoning_len, prompt_len, k_norm, true_h2o_all, true_h2o_late)
    profile = exp.analyze_position_profile(keep_idx, reasoning_len)
    n_keep = len(keep_pos)
    n_evict = len(evict_pos)
    assert n_keep + n_evict == reasoning_len, f"Mismatch: {n_keep} + {n_evict} != {reasoning_len}"
    print(f"  {sel:20s}: n_keep={n_keep}, n_evict={n_evict}, "
          f"sink_frac={profile['sink_frac']*100:.1f}%, mean_pos={profile['mean_pos']:.3f}")

print("\nVerifying sink_excluded_h2o excludes sinks...")
keep_pos, _, keep_idx = exp.select_positions(
    "sink_excluded_h2o", 0.33, reasoning_len, prompt_len,
    k_norm, true_h2o_all, true_h2o_late)
sinks_in_selection = sum(1 for i in keep_idx if i < exp.N_SINKS)
print(f"  Sinks in selection: {sinks_in_selection} (should be 0)")
assert sinks_in_selection == 0, "Sink-excluded H2O should not select sinks!"

print("\nAll selection methods work correctly!")
print("Smoke test PASSED")
