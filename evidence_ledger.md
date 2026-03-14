# Evidence Ledger

## Current Summary
Last updated: 2026-03-14 (cycle 8, exp_008)
Cycles completed: 8

### Hypothesis Status: WEAKENING — key mechanistic claims (SNR cliff, adversarial null space) are Qwen-specific

The KV cache carries a functionally separable hidden channel that encodes answer-relevant
information independent of the visible reasoning tokens.

### Evidence Overview
| Claim | Status | Strength | Key Experiments | Notes |
|-------|--------|----------|-----------------|-------|
| Unused output capacity | established | strong | Exp 1 (research_spec) | 4-5 bits/token unused |
| CoT narrows distribution | established | strong | Exp 2 (research_spec) | 3x entropy reduction, median near zero |
| KV cache fragility (SNR cliff) | **Qwen-specific, NOT general** | moderate negative | Exp 3, **Exp 007** | Qwen: cliff at 14 dB; Llama: 100% at 5 dB, no cliff |
| Adversarial null space exists | **Qwen-specific, NOT general** | **moderate negative** | Exp 4, **Exp 008** | Qwen: 377x, 100% success. **Llama: 0.8x, 0% success — produces garbage not valid answers** |
| Null space has spatial structure | **partially supported** | moderate | Exp 5, Exp 002, Exp 004 | PGD rho=0.78; zeroing shows nothing; noise shows accuracy dissociation |
| Cross-model text-dependence variation | supported | moderate | Exp 6 (research_spec) | Qwen 94% compliant, Llama ~30% |
| Functional separability via zeroing | **disconfirmed** | moderate negative | Exp 002, Exp 004, **Exp 005** | Zeroing even 99% of positions → 86-100% accuracy on both models |
| **Accuracy dissociation via noise** | **established** | **strong** | **Exp 004, Exp 005** | **Replicated: Qwen +23.5pp, Llama +23.8pp at 5% noise** |
| **Noise >> zeroing for ablation** | **established** | **strong** | **Exp 004, Exp 005** | **Confirmed on both Qwen and Llama** |
| **AC positions are general hubs** | **established** | **moderate** | **Exp 004, Exp 005** | **AC ablation hurts text MORE than TC on both models** |
| **Dissociation is architecture-general** | **supported** | **strong** | **Exp 005** | **Effect size nearly identical across Qwen and Llama** |
| **SNR robustness is architecture-SPECIFIC** | **established** | **strong** | **Exp 003, Exp 007** | **Qwen: digital cliff at 14 dB. Llama: robust to 5 dB, no cliff** |
| **Encoding strategy differs: digital (Qwen) vs distributed (Llama)** | **supported** | **moderate** | **Exp 005, Exp 007** | **Llama: position-sensitive, noise-robust. Qwen: position-tolerant, noise-fragile** |
| **PGD null space is Qwen-specific** | **established** | **strong** | **Exp 4, Exp 008** | **Qwen: 100% success, 377x norm, valid answers. Llama: 0% success, 0.8x norm, garbage output** |
| **Partial null space exists on Llama** | **supported** | **weak** | **Exp 008** | **PGD CAN change answer-region predictions (0% match) while preserving text (94%), but produces incoherent output not valid answers** |

### Open Questions
1. ~~Why does 50% position pruning not affect accuracy?~~ **ANSWERED (Exp 004): zeroing is the wrong method.**
2. ~~Does the SNR cliff replicate on Llama-3.1-8B?~~ **ANSWERED (Exp 007): NO. Llama shows 100% accuracy at all SNR ≥5 dB. The cliff is Qwen-specific.**
3. ~~At what pruning fraction does accuracy break, and does it break differently for AC vs TC positions?~~ **ANSWERED: Under noise, AC breaks faster than TC on both models.**
4. ~~Would noise injection at classified positions reveal the dissociation?~~ **ANSWERED: Yes.**
5. **Are AC-selective positions genuinely "answer-specific" or just "generally important hubs"?** **ANSWERED (Exp 004+005): Hubs. AC ablation hurts text MORE on both models. But TC-selective noise preserves answer accuracy, so the separation is real — it's just that AC positions are important for everything, not selectively for answers.**
6. Would a more surgical noise intervention (partial noise, not full replacement) reveal finer-grained separation?
7. ~~Does the accuracy dissociation replicate on Llama-3.1-8B?~~ **ANSWERED (Exp 005): YES, +23.8pp at 5% noise (vs Qwen's +23.5pp)**
8. ~~**NEW:** Why is Llama so much MORE fragile than Qwen under random noise?~~ **PARTIALLY ANSWERED (Exp 007): Llama is fragile to POSITION-SELECTIVE destruction (exp_005) but ROBUST to UNIFORM noise (exp_007). Information is distributed across all positions (analog), so destroying any subset is devastating, but uniform noise averages out. Remaining question: does GQA (8 KV heads shared across 32 query heads) provide inherent noise robustness?**
9. ~~Replicate the adversarial null space experiment (Exp 4) on Llama — does the PGD-discovered null space also exist?~~ **ANSWERED (Exp 008): NO in the strong sense. PGD can change answer-region predictions while preserving text (partial null space), but cannot redirect to a valid alternative answer. 0% success rate vs Qwen's 100%. Perturbation norm 0.8x vs Qwen's 377x.**
10. **NEW:** Does the ~24pp dissociation effect hold on a third model family (e.g., Qwen3-8B)?
11. **NEW:** Does GQA (vs MHA) explain Llama's noise robustness? Test SNR cliff on a GQA variant of Qwen or MHA variant of Llama.
12. **NEW:** What happens between SNR 0 and 5 dB on Llama? Need finer sampling (1, 2, 3, 4 dB) to characterize the transition.
13. **NEW:** Is the digital vs distributed encoding difference related to model size (4B vs 8B) or architecture (Qwen vs Llama)?
14. **NEW:** Would a TARGETED PGD attack (maximize probability of specific wrong answer) succeed on Llama where untargeted divergence fails?
15. **NEW:** Would reasoning-only or full-sequence PGD attacks succeed on Llama? Prompt-only attacks may be too constrained for distributed encoding.
16. **NEW:** Is the null space failure on Llama due to instruction tuning (format robustness) or architecture? Test PGD on Qwen-Instruct to separate these factors.

### Confirmed Findings
- LLM output distributions have ~4-5 bits/token unused capacity (Exp 1)
- CoT narrows per-token entropy 3x; median entropy near zero during reasoning (Exp 2)
- **Zeroing KV positions is fundamentally inadequate for position importance studies** — confirmed on BOTH Qwen and Llama (Exp 004, 005)
- **Answer-coupled positions carry disproportionate answer information** — noise at 5% of AC-selective positions drops accuracy ~24pp more than TC-selective positions, replicated across architectures (Exp 004, 005)
- **The ~24pp accuracy dissociation is architecture-general** — nearly identical effect on Qwen3-4B-Base (+23.5pp) and Llama-3.1-8B-Instruct (+23.8pp) (Exp 005)
- **AC-selective positions are hubs, not channel-specific** — they carry more information for both answer and text (Exp 004, 005)
- **Random noise is more destructive than targeted noise** — at 5%, random (5% acc) < SelAC (19%) < SelTC (43%) on Llama, confirming channel separation (Exp 005)
- **SNR cliff is Qwen-specific, NOT architecture-general** — Llama shows 100% accuracy at all SNR ≥5 dB, no cliff in the 5-25 dB range where Qwen collapses (Exp 007)
- **Models use different encoding strategies:** Qwen: digital/concentrated (fragile to uniform noise, SNR cliff at 14 dB). Llama: distributed/analog (robust to uniform noise, fragile to position ablation) (Exp 005+007)
- **Adversarial null space (PGD) is Qwen-specific in the strong sense** — Llama shows 0% attack success vs Qwen's 100%. PGD can change distributions but not redirect to valid alternative answers on Llama (Exp 008)

### Disconfirmed or Revised
- **Position-level functional separation via zeroing** (Exp 002): Zeroing is too weak. Methodological limitation, not evidence against spatial structure.
- **Full double dissociation** (Exp 004, 005): Text loss dissociation is reversed on BOTH models. AC positions are hubs important for everything. The "hidden channel" is not cleanly separable at the position level — but noise-based ablation still reveals differential answer importance.
- **Llama's text-resistance = stronger hidden channel** (Exp 005): Llama shows the SAME dissociation effect size as Qwen (~24pp), not larger. Its text-resistance (Exp 6) comes from different computation, not from different spatial structure of the hidden channel.
- **SNR cliff is a general property of transformer KV caches** (Exp 007): The sharp cliff at ~14 dB is Qwen-specific. Llama shows no cliff — 100% accuracy at 5 dB (noise at 56% of signal). The "digital-like fragility" interpretation applies only to Qwen's architecture.

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
- Interpretation: zeroing is too weak (confirmed by Exp 004, 005)
- See `experiment_log/exp_002.md`, `results/exp_002/`

### Exp 003 (Cycle 3) — INCOMPLETE (agent crash)
**Pruning Fraction Sweep**
- Script written but never executed
- Superseded by Exp 004
- See `experiment_log/exp_003.md`

### Exp 004 (Cycle 4) — PARTIAL CONFIRMATION
**Noise Ablation Sweep with Selectivity Classification (Qwen)**
- Model: Qwen3-4B-Base, n=34 valid problems
- **Accuracy dissociation:** +23.5 pp at 5% noise (SelAC=44.1%, SelTC=67.6%)
- **Text loss dissociation reversed** (AC ablation hurts text MORE)
- Evidential strength: moderate
- See `experiment_log/exp_004.md`, `results/exp_004/`

### Exp 005 (Cycle 5) — CROSS-MODEL REPLICATION CONFIRMED
**Noise Ablation Sweep on Llama-3.1-8B (replication of exp_004)**
- Model: meta-llama/Llama-3.1-8B-Instruct, n=21 valid problems
- **Accuracy dissociation replicates:** +23.8 pp at 5% noise (SelAC=19.0%, SelTC=42.9%)
- **Text loss dissociation reversed** — same pattern as Qwen
- **Zeroing robustness confirmed** — 86% accuracy at 99% zeroing
- **Llama more fragile under random noise** — 5% random → 5% acc (vs Qwen's 56%)
- Evidential strength: strong (cross-model replication with quantitatively identical effect)
- See `experiment_log/exp_005.md`, `results/exp_005/`

### Exp 006 (Cycle 6) — FAILED
**SNR Cliff Replication on Llama-3.1-8B (first attempt)**
- `DynamicCache` object is not subscriptable — all problems errored at all SNR levels
- Root cause: transformers 5.3.0 changed DynamicCache API (no `key_cache`/`value_cache` attributes)
- See `experiment_log/exp_006.md`

### Exp 007 (Cycles 7-8) — NEGATIVE REPLICATION (important finding)
**SNR Cliff Replication on Llama-3.1-8B**
- Model: meta-llama/Llama-3.1-8B-Instruct, n=21 valid problems, 15 SNR levels
- **SNR cliff does NOT replicate:** Llama shows 100% accuracy at all SNR ≥5 dB
- **Collapse only at extreme noise:** 0% accuracy at SNR 0 dB (noise = signal)
- **Qwen cliff at 14 dB is architecture-specific**, not a general transformer property
- **Digital (Qwen) vs distributed (Llama) encoding:** reconciles with exp_005 position sensitivity
- Evidential strength: strong (clear negative replication; qualitative difference, not marginal)
- See `experiment_log/exp_007.md`, `results/exp_007/`
