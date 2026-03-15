#!/usr/bin/env python3
"""Smoke test for exp 061 spectral metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from exp_061_effective_rank import compute_spectral_metrics

np.random.seed(42)

# Test 1: Low-rank matrix (3 dominant dimensions out of 128)
A = np.random.randn(50, 3) @ np.random.randn(3, 128) + 0.01 * np.random.randn(50, 128)
M = torch.tensor(A, dtype=torch.float32)
metrics = compute_spectral_metrics(M)
print("Low-rank test:")
print(f"  Effective rank: {metrics['effective_rank']:.2f}")
print(f"  Normalized eff rank: {metrics['normalized_eff_rank']:.4f}")
print(f"  Top-1 energy: {metrics['top1_energy']:.4f}")
print(f"  Top-5 energy: {metrics['top5_energy']:.4f}")
print(f"  Spectral gap: {metrics['spectral_gap']:.2f}")
print(f"  Decay rate: {metrics['decay_rate']:.4f}")
print(f"  n_svs: {metrics['n_singular_values']}")

# Test 2: High-rank matrix (all dimensions used)
B = np.random.randn(50, 128)
M2 = torch.tensor(B, dtype=torch.float32)
metrics2 = compute_spectral_metrics(M2)
print("\nHigh-rank test:")
print(f"  Effective rank: {metrics2['effective_rank']:.2f}")
print(f"  Normalized eff rank: {metrics2['normalized_eff_rank']:.4f}")
print(f"  Top-1 energy: {metrics2['top1_energy']:.4f}")
print(f"  Top-5 energy: {metrics2['top5_energy']:.4f}")
print(f"  Spectral gap: {metrics2['spectral_gap']:.2f}")

print("\nMetrics function works correctly.")
assert metrics['effective_rank'] < metrics2['effective_rank'], "Low-rank should have lower eff rank"
print(f"Low-rank eff_rank={metrics['effective_rank']:.1f} << High-rank eff_rank={metrics2['effective_rank']:.1f}")
print("PASS")
