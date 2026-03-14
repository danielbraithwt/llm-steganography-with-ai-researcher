# Evidence Ledger

## Current Summary
Last updated: 2026-03-14 (cycle 4)
Cycles completed: 4

### Hypothesis Status: INVESTIGATING — partial support emerging

The KV cache carries a functionally separable hidden channel that encodes answer-relevant
information independent of the visible reasoning tokens.

### Evidence Overview
| Claim | Status | Strength | Key Experiments | Notes |
|-------|--------|----------|-----------------|-------|
| Unused output capacity | established | strong | Exp 1 (research_spec) | 4-5 bits/token unused |
| CoT narrows distribution | established | strong | Exp 2 (research_spec) | 3x entropy reduction, median near zero |
| KV cache fragility (SNR cliff) | supported | moderate | Exp 3 (research_spec) | 14dB cliff, Qwen only — needs replication |
| Adversarial null space exists | supported | strong | Exp 4 (research_spec) | 377x signal norm, Qwen only |
| Null space has spatial structure | **partially supported** | moderate | Exp 5, Exp 002, **Exp 004** | PGD rho=0.78; zeroing shows nothing; **noise shows accuracy dissociation** |
| Cross-model text-dependence variation | supported | moderate | Exp 6 (research_spec) | Qwen 94% compliant, Llama ~30% |
| Functional separability via zeroing | **disconfirmed** | moderate negative | Exp 002, Exp 004 | Zeroing even 99% of positions → 100% accuracy |
| **Accuracy dissociation via noise** | **supported** | **moderate** | **Exp 004** | **Noise at 5%: AC 44% vs TC 68% accuracy (+23.5 pp)** |
| **Noise >> zeroing for ablation** | **established** | **strong** | **Exp 004** | **5% noise ≈ 99% zeroing in destructiveness** |

### Open Questions
1. ~~Why does 50% position pruning not affect accuracy?~~ **ANSWERED (Exp 004): zeroing is the wrong method. The model routes around zeros but is devastated by noise.**
2. Does the SNR cliff replicate on Llama-3.1-8B?
3. ~~At what pruning fraction does accuracy break, and does it break differently for AC vs TC positions?~~ **PARTIALLY ANSWERED: Under noise, AC breaks faster than TC. Under zeroing, neither breaks even at 99%.**
4. ~~Would noise injection at classified positions reveal the dissociation?~~ **ANSWERED: Yes, accuracy dissociation emerges. Text loss dissociation does not.**
5. **NEW:** Are AC-selective positions genuinely "answer-specific" or just "generally important hubs"? The text loss pattern (AC ablation hurts text MORE) suggests the latter.
6. **NEW:** Would a more surgical noise intervention (partial noise, not full replacement) reveal finer-grained separation?
7. **NEW:** Does the accuracy dissociation replicate on Llama-3.1-8B?

### Confirmed Findings
- LLM output distributions have ~4-5 bits/token unused capacity (Exp 1)
- CoT narrows per-token entropy 3x; median entropy near zero during reasoning (Exp 2)
- **Zeroing KV positions is fundamentally inadequate for position importance studies** — the model is robust to 99% zeroing but fragile to 5% noise injection (Exp 004)
- **Answer-coupled positions carry disproportionate answer information** — noise at 5% of AC-selective positions drops accuracy 23.5 pp more than TC-selective positions (Exp 004)

### Disconfirmed or Revised
- **Position-level functional separation via zeroing** (Exp 002): Zeroing is too weak. This is a methodological limitation, not evidence against spatial structure. (Revised understanding from Exp 004.)
- **Full double dissociation** (Exp 004): The text loss dissociation is reversed — AC positions are important for BOTH answer and text, not selectively answer-specific. The channels may not map cleanly to separate positions.

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
- Interpretation: zeroing is too weak (confirmed by Exp 004)
- Evidential strength: weak negative → revised to methodological artifact
- See `experiment_log/exp_002.md`, `results/exp_002/`

### Exp 003 (Cycle 3) — INCOMPLETE (agent crash)
**Pruning Fraction Sweep**
- Script written but never executed
- Superseded by Exp 004 which uses improved methodology
- See `experiment_log/exp_003.md`

### Exp 004 (Cycle 4) — PARTIAL CONFIRMATION
**Noise Ablation Sweep with Selectivity Classification**
- Model: Qwen3-4B-Base, n=34 valid problems
- Two ablation methods: noise injection (5-20%) and zeroing (90-99%)
- **Accuracy dissociation found:** At 5% noise, selective-AC accuracy=44.1%, selective-TC=67.6% (+23.5 pp)
- **Text loss dissociation NOT found:** AC ablation hurts text MORE than TC ablation (reversed)
- **Key methodological finding:** 5% noise ≈ 99% zeroing in destructiveness
- Evidential strength: moderate (accuracy dissociation strong, text dissociation absent)
- See `experiment_log/exp_004.md`, `results/exp_004/`
