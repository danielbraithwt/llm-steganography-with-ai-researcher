# Evidence Ledger

## Current Summary
Last updated: 2026-03-14 (cycle 3)
Cycles completed: 2

### Hypothesis Status: INVESTIGATING

The KV cache carries a functionally separable hidden channel that encodes answer-relevant
information independent of the visible reasoning tokens.

### Evidence Overview
| Claim | Status | Strength | Key Experiments | Notes |
|-------|--------|----------|-----------------|-------|
| Unused output capacity | established | strong | Exp 1 (research_spec) | 4-5 bits/token unused |
| CoT narrows distribution | established | strong | Exp 2 (research_spec) | 3x entropy reduction, median near zero |
| KV cache fragility (SNR cliff) | supported | moderate | Exp 3 (research_spec) | 14dB cliff, Qwen only — needs replication |
| Adversarial null space exists | supported | strong | Exp 4 (research_spec) | 377x signal norm, Qwen only |
| Null space has spatial structure | **challenged** | weak | Exp 5 (research_spec), Exp 002 | PGD shows rho=0.78 but pruning shows no dissociation |
| Cross-model text-dependence variation | supported | moderate | Exp 6 (research_spec) | Qwen 94% compliant, Llama ~30% |
| Functional separability via pruning | **disconfirmed** | weak negative | Exp 002 | 100% accuracy at 50% pruning all conditions |

### Open Questions
1. Why does 50% position pruning not affect accuracy? Is it methodology (zeroing too weak) or genuine (no spatial separation)?
2. Does the SNR cliff replicate on Llama-3.1-8B?
3. At what pruning fraction does accuracy break, and does it break differently for AC vs TC positions?
4. Would noise injection (instead of zeroing) at classified positions reveal the dissociation?

### Confirmed Findings
- LLM output distributions have ~4-5 bits/token unused capacity (Exp 1)
- CoT narrows per-token entropy 3x; median entropy near zero during reasoning (Exp 2)

### Disconfirmed or Revised
- **Position-level functional separation via pruning** (Exp 002): Zeroing 50% of answer-coupled positions does not reduce accuracy. The spatial structure seen by PGD (rho=0.78) does not translate to functional separation under pruning. May be methodological limitation or genuine absence.

---

## Experiment Log Summary

### Exp 001 (Cycle 1) — FAILED
**Double Dissociation via KV Position Pruning**
- Used wrong model (instruct instead of base) — no valid results
- See `experiment_log/exp_001.md`

### Exp 002 (Cycle 2) — NEGATIVE RESULT
**Double Dissociation via KV Position Pruning (retry)**
- Model: Qwen3-4B-Base, n=82 valid problems, 50% pruning fraction
- **All conditions achieved 100% accuracy** — no dissociation
- Text loss: AC-pruned +0.506, TC-pruned +0.527, Random +0.225 nats
- Interpretation: zeroing may be too weak; model routes around via remaining positions
- Evidential strength: weak negative (could be methodological)
- See `experiment_log/exp_002.md`, `results/exp_002/`
