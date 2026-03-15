#!/usr/bin/env python3
"""Quick smoke test for spectral metrics computation."""
import torch
import numpy as np

def compute_spectral_metrics(matrix):
    M = matrix.float()
    n, d = M.shape
    if n < 2 or d < 2:
        return None
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    S = S.cpu().numpy()
    S = S[S > 1e-10]
    if len(S) < 2:
        return None
    p = S / S.sum()
    entropy = -np.sum(p * np.log(p + 1e-30))
    effective_rank = np.exp(entropy)
    participation_ratio = (S.sum() ** 2) / (np.sum(S ** 2))
    spectral_gap = S[0] / S[1] if S[1] > 1e-10 else float('inf')
    total_energy = np.sum(S ** 2)
    top1_energy = S[0] ** 2 / total_energy
    return {
        'effective_rank': effective_rank,
        'participation_ratio': participation_ratio,
        'spectral_gap': spectral_gap,
        'top1_energy': top1_energy,
        'n_singular_values': len(S),
    }

# Test with low-rank matrix (should have low effective rank)
low_rank = torch.randn(50, 3) @ torch.randn(3, 128)
result_low = compute_spectral_metrics(low_rank)
print(f'Low-rank (3 components): eff_rank={result_low["effective_rank"]:.2f}, top1={result_low["top1_energy"]:.3f}')

# Test with full-rank matrix (should have high effective rank)
full_rank = torch.randn(50, 128)
result_full = compute_spectral_metrics(full_rank)
print(f'Full-rank: eff_rank={result_full["effective_rank"]:.2f}, top1={result_full["top1_energy"]:.3f}')

# Test with very low-rank (1 component + noise)
rank1 = torch.randn(50, 1) @ torch.randn(1, 128) + 0.01 * torch.randn(50, 128)
result_r1 = compute_spectral_metrics(rank1)
print(f'Rank-1 + noise: eff_rank={result_r1["effective_rank"]:.2f}, top1={result_r1["top1_energy"]:.3f}')

assert result_low['effective_rank'] < 5, f"Low-rank should have low eff rank, got {result_low['effective_rank']}"
assert result_full['effective_rank'] > 20, f"Full-rank should have high eff rank, got {result_full['effective_rank']}"
assert result_r1['top1_energy'] > 0.9, f"Rank-1 should have high top1 energy, got {result_r1['top1_energy']}"

print('Smoke test PASSED')
