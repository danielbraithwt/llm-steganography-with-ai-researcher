# Evidence Ledger

## Current Summary
Last updated: 2026-03-14 (cycle 11, H2O overlap experiment)
Cycles completed: 11

### Hypothesis Status: REFRAMED — hidden channel confirmed but NOT captured by standard KV compression; GQA/MHA explanation invalidated (both models use GQA)

The KV cache carries a functionally separable hidden channel that encodes answer-relevant
information independent of the visible reasoning tokens. Exp 011 reveals that H2O heavy-hitter
positions (used by practical KV compression) do NOT overlap with answer-coupled positions
(rho=0.004-0.11). H2O preferentially retains text-coupled positions (57% TC vs 40% AC).
This means standard KV compression may actively harm the hidden channel while preserving text
coherence. **Important correction:** Qwen3-4B-Base also uses GQA (8 KV heads), NOT MHA.
The GQA-vs-MHA hypothesis for architecture differences is therefore INVALIDATED.

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
| **Encoding strategy differs: digital (Qwen) vs distributed (Llama)** | **supported** | **moderate** | **Exp 005, Exp 007, Exp 009** | **Position-level and noise-level differences confirmed. Layer-level: both robust, but Qwen slightly more concentrated (std ratio 7.64)** |
| **PGD null space is Qwen-specific** | **established** | **strong** | **Exp 4, Exp 008** | **Qwen: 100% success, 377x norm, valid answers. Llama: 0% success, 0.8x norm, garbage output** |
| **Partial null space exists on Llama** | **supported** | **weak** | **Exp 008** | **PGD CAN change answer-region predictions (0% match) while preserving text (94%), but produces incoherent output not valid answers** |
| **Layer-level redundancy is universal** | **established** | **moderate** | **Exp 009** | **Both models tolerate single-layer KV destruction: Llama 93-100%, Qwen 100% (35/36 layers). Residual stream provides massive compensation** |
| **Layer sensitivity is more concentrated in Qwen** | **supported** | **weak** | **Exp 009** | **std ratio = 7.64 (predicted ≥1.5). But Qwen n=4 — very low confidence. Layer 0 critical (25%), all others 100%** |
| **Encoding distinction is position-level, not layer-level** | **supported** | **moderate** | **Exp 004, 005, 009** | **Position ablation shows clear effects (24pp dissociation). Layer ablation shows near-zero effects on both models. Residual stream provides layer-level redundancy** |
| **Causal bypass validates hidden channel (literature)** | **supported** | **strong (independent)** | **Lit scan cycle 10** | **CMI metric shows bypass regimes where CoT text has zero causal influence on answer. Tested on 11 models including Qwen3-4B. Independent convergent evidence (Sathyanarayanan et al., 2026)** |
| **KV cache carries manipulable semantic information (literature)** | **supported** | **strong (independent)** | **Lit scan cycle 10** | **KV cache steering induces reasoning styles via one-shot KV modification (Belitsky et al., 2025). Proves separable channels from constructive direction (vs our adversarial direction)** |
| **GQA vs MHA plausibly explains Qwen/Llama encoding difference (literature)** | **INVALIDATED** | **disconfirmed** | **Lit scan cycle 10, Exp 011** | **Both Qwen3-4B and Llama-3.1-8B use GQA with 8 KV heads. Architecture difference is NOT MHA vs GQA. Original hypothesis was based on incorrect assumption that Qwen uses MHA** |
| **Text bottleneck framing is mainstream (literature)** | **supported** | **moderate** | **Lit scan cycle 10** | **Coconut, Quiet-STaR, latent reasoning survey all frame text as bottleneck. Coconut shows reasoning IMPROVES when text constraint removed. Our work provides mechanistic KV-level evidence** |
| **H2O heavy-hitters ≠ answer-coupled positions** | **established** | **moderate** | **Exp 011** | **H2O vs AC rho=0.004 (Qwen), 0.11 (Llama). H2O retains 57% TC-selective vs 40% AC-selective. Positions with highest cumulative attention are NOT the answer-relevant ones. Replicated on both models** |
| **KV compression may harm hidden channel** | **supported** | **moderate** | **Exp 011** | **H2O preferentially evicts answer-coupled positions (Q1 has highest AC). Standard compression preserves text coherence but may degrade answer computation. Suggests AC-aware compression could improve accuracy** |

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
11. ~~**NEW:** Does GQA (vs MHA) explain Llama's noise robustness?~~ **INVALIDATED (Exp 011): Both models use GQA (8 KV heads). The difference is NOT GQA vs MHA.**
12. **NEW:** What happens between SNR 0 and 5 dB on Llama? Need finer sampling (1, 2, 3, 4 dB) to characterize the transition.
13. **NEW:** Is the digital vs distributed encoding difference related to model size (4B vs 8B) or architecture (Qwen vs Llama)?
14. **NEW:** Would a TARGETED PGD attack (maximize probability of specific wrong answer) succeed on Llama where untargeted divergence fails?
15. **NEW:** Would reasoning-only or full-sequence PGD attacks succeed on Llama? Prompt-only attacks may be too constrained for distributed encoding.
16. **NEW:** Is the null space failure on Llama due to instruction tuning (format robustness) or architecture? Test PGD on Qwen-Instruct to separate these factors.
17. **NEW (Exp 009):** Why does Qwen show only 20% baseline accuracy with eager attention? Is this a model loading issue or a generation configuration problem? Need to verify Qwen layer sensitivity with higher n.
18. **NEW (Exp 009):** Why is layer 0 specifically critical for Qwen? Is this about initial representation quality or attention pattern bootstrapping?
19. **NEW (Exp 009):** Would multi-layer ablation (2-3 layers simultaneously) break Llama's layer redundancy? The residual stream may have limits.
20. ~~**NEW (Lit scan, cycle 10):** Do H2O "heavy-hitter" positions correlate with our answer-coupled positions?~~ **ANSWERED (Exp 011): NO. H2O vs AC rho=0.004 (Qwen), 0.11 (Llama). H2O preferentially retains TC-selective positions (57%) over AC-selective (40%). Standard compression does NOT preserve the hidden channel.**
21. **NEW (Lit scan, cycle 10):** Can we apply the CMI (CoT Mediation Index) from causal bypass research to quantify text-coupling vs answer-coupling at specific positions?
22. **NEW (Lit scan, cycle 10):** Does KV cache steering (Belitsky et al. 2025) change the AC/TC spatial structure? If steering vectors modify the hidden channel, this would demonstrate bidirectional manipulability.
23. **NEW (Lit scan, cycle 10):** Is token importance temporally dynamic during CoT (Lethe finding)? Our static AC/TC classification may miss important temporal effects. Track per-step attention evolution.
24. ~~**NEW (Lit scan, cycle 10):** Does GQA's KV head sharing ratio quantitatively predict the null space dimensionality reduction?~~ **INVALIDATED (Exp 011): Both models use identical GQA ratio (8 KV heads, 32 query heads). The Qwen/Llama difference is NOT explained by GQA/MHA.**
25. **NEW (Exp 011):** Would AC-aware KV compression (retain positions with highest answer-token attention) outperform H2O (cumulative attention) for answer accuracy? Exp 011 shows H2O evicts answer-relevant positions.
26. **NEW (Exp 011):** What explains the Qwen/Llama encoding difference if NOT GQA vs MHA? Both use 8 KV heads. Candidates: model size (4B vs 8B), training data/procedure, depth (36 vs 32 layers), hidden dimension, or instruction tuning.
27. **NEW (Exp 011):** Is the H2O-AC dissociation driven by position (early vs late in sequence) or by genuine functional differentiation? Need position-controlled analysis.
28. **NEW (Exp 011):** Why does Qwen3-4B-Base with eager attention produce only 16-20% baseline accuracy? This limits all Qwen experiments to n=4.

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
- **Single-layer KV destruction is tolerated by both architectures** — Llama: 93-100% accuracy for all 32 layers. Qwen: 100% for 35/36 layers (except L0=25%, but n=4). Residual stream provides massive layer-level redundancy (Exp 009)
- **The digital/distributed encoding distinction operates at position-level and noise-level, NOT at layer-level** — both models show high layer redundancy, though Qwen is slightly more layer-concentrated (std ratio=7.64) (Exp 009)
- **H2O heavy-hitter positions do NOT correspond to answer-coupled positions** — H2O vs AC Spearman rho = 0.004 (Qwen), 0.11 (Llama). H2O retains 57% TC-selective vs 40% AC-selective positions. Standard KV compression preferentially evicts answer-relevant positions (Exp 011)
- **Both Qwen3-4B-Base and Llama-3.1-8B use GQA with 8 KV heads** — the GQA-vs-MHA hypothesis is invalidated. Architecture differences must stem from other factors (Exp 011)

### Disconfirmed or Revised
- **Position-level functional separation via zeroing** (Exp 002): Zeroing is too weak. Methodological limitation, not evidence against spatial structure.
- **Full double dissociation** (Exp 004, 005): Text loss dissociation is reversed on BOTH models. AC positions are hubs important for everything. The "hidden channel" is not cleanly separable at the position level — but noise-based ablation still reveals differential answer importance.
- **Llama's text-resistance = stronger hidden channel** (Exp 005): Llama shows the SAME dissociation effect size as Qwen (~24pp), not larger. Its text-resistance (Exp 6) comes from different computation, not from different spatial structure of the hidden channel.
- **SNR cliff is a general property of transformer KV caches** (Exp 007): The sharp cliff at ~14 dB is Qwen-specific. Llama shows no cliff — 100% accuracy at 5 dB (noise at 56% of signal). The "digital-like fragility" interpretation applies only to Qwen's architecture.
- **GQA vs MHA explains Qwen/Llama encoding differences** (Exp 011): INVALIDATED. Both Qwen3-4B-Base and Llama-3.1-8B use GQA with 8 KV heads. The architecture difference must stem from other factors (model size, training, depth, instruction tuning).
- **H2O heavy-hitters = AC positions ("AC are hubs → H2O keeps them")** (Exp 011): DISCONFIRMED. H2O vs AC rho ≈ 0. H2O measures "popularity" (cumulative attention), which is different from "answer-relevance" (answer-token attention). These are orthogonal importance dimensions.

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

### Exp 008 (Cycle 8) — NEGATIVE REPLICATION
**PGD Adversarial Null Space on Llama-3.1-8B**
- Model: meta-llama/Llama-3.1-8B-Instruct, n=17 valid problems, 50 PGD steps
- **PGD null space does NOT replicate in strong sense:** 0% attack success (vs Qwen's 100%)
- **Partial null space exists:** PGD changes answer-region predictions (0% match) while preserving text (94%), but produces garbage, not valid answers
- **Perturbation norm 0.8x** (vs Qwen's 377x)
- Evidential strength: strong (negative replication)
- See `experiment_log/exp_008.md`, `results/exp_008/`

### Exp 009 (Cycle 9) — PARTIAL CONFIRMATION
**Per-Layer Noise Sensitivity Profiling**
- Models: Qwen3-4B-Base (n=4), Llama-3.1-8B-Instruct (n=15)
- **Both models highly layer-redundant:** Llama 93-100% for all layers, Qwen 100% for 35/36
- **Cross-model contrast matches predictions:** std ratio = 7.64 (predicted ≥1.5)
- **Layer 0 critical for Qwen** (25%) — unexpected; predicted critical layers in late layers
- Evidential strength: weak-to-moderate (Qwen n=4 limits conclusions)
- See `experiment_log/exp_009.md`, `results/exp_009/`

### Literature Scan (Cycle 10) — CONVERGENT EVIDENCE FROM INDEPENDENT WORK
**Topics:** CoT faithfulness, causal bypass, KV cache adversarial perturbation, latent reasoning, KV cache compression, GQA vs MHA, residual stream redundancy

**Key papers and connections:**

1. **"Causal Bypass in LLMs" (Sathyanarayanan et al., Feb 2026):** Introduces CoT Mediation Index (CMI) — finds "bypass regimes" where models generate fluent CoT but answer computation flows through latent pathways. CMI ≈ 0 on TruthfulQA, 5/20 GSM8K instances show pure bypass. Tests 11 models including Qwen3-4B. Uses activation patching (complementary to our KV perturbation). **Independent validation of our hidden channel hypothesis.**

2. **"Can Transformer Memory Be Corrupted?" (MTI V.1, Oct 2025):** KV cache perturbation framework (additive noise, zeroing, orthogonal rotations) tested on GPT-2 and LLaMA-2. 15-30% performance reduction. Provides Lipschitz-based theoretical analysis of perturbation propagation. **Independent validation of our perturbation approach; their theory could formalize our SNR cliff.**

3. **"KV Cache Steering" (Belitsky et al., Jul 2025):** Modifies KV cache once after prefilling to induce reasoning styles. K'=K+c^k*S^k. Consistently improves reasoning (GSM8K +0.5-4pp, MATH +7.4pp). **Proves KV cache carries separable, manipulable semantic information — our "null space" from the opposite direction (constructive steering vs adversarial perturbation).**

4. **"Coconut/CCOT" (Meta, Dec 2024) + "Survey on Latent Reasoning" (Jul 2025):** Latent reasoning removes text bottleneck; reasoning improves when freed from token vocabulary. Breadth-first search enabled by continuous thought. **Validates our "text as lossy projection" framing — the text bottleneck actively constrains computation.**

5. **"Bottlenecked Transformers" (May 2025):** Uses Information Bottleneck theory + periodic KV cache rewriting for reasoning (+6.6pp). **Treats KV cache as active computation workspace, not passive memory. Provides IB theory framework for our text/answer channel separation.**

6. **"CoT In The Wild Is Not Always Faithful" (Arcuschin et al., Mar 2025):** IPHR rates: 0.04% (Claude 3.7 thinking) to 13.49% (GPT-4o-mini). Thinking models reduce unfaithfulness. **Supports model-specific faithfulness variation consistent with our Qwen/Llama text-compliance differences.**

7. **"Hold Onto That Thought" (Dec 2025) + "Lethe" (Nov 2024):** H2O/SnapKV retain "heavy-hitter" positions for reasoning. Pyramidal sparsity assumption FAILS for reasoning models. Token importance is temporally dynamic. **Heavy-hitter positions may overlap with our answer-coupled positions. Temporal dynamics suggest our static classification misses effects.**

8. **GQA vs MHA (various):** GQA shares KV heads across multiple query heads (Llama: 8 KV → 32 query). **Plausible mechanistic explanation for Qwen/Llama encoding difference:** GQA's sharing creates inherent redundancy (noise robustness) but reduces null space dimensionality (PGD failure).

9. **"Hyper-Connections" (ICLR 2025):** Single residual stream is a bottleneck. Multiple parallel streams improve performance. **Explains our Exp 009 finding:** layer redundancy comes from residual stream compensation, which is why the digital/distributed distinction operates at position-level, not layer-level.

**Literature impact on hypothesis status:**
- Core claim (hidden channel exists) is STRENGTHENED — independent convergent evidence from 5+ research groups
- Architecture-specific findings (SNR cliff, PGD null space) are now EXPLAINED by GQA vs MHA, not simply "Qwen-specific anomalies"
- The "text bottleneck" framing is mainstream in latent reasoning literature
- Our unique contribution remains the MECHANISTIC evidence (spatial structure, adversarial perturbation) at the KV cache level

See `literature_notes/cycle_010_*.md` for detailed paper summaries

### Exp 011 (Cycle 11) — SURPRISING NEGATIVE RESULT
**H2O Heavy-Hitter vs Answer-Coupled Position Overlap**
- Models: Qwen3-4B-Base (n=4), Llama-3.1-8B-Instruct (n=17)
- **H2O heavy-hitters do NOT overlap with AC positions:** rho=0.004 (Qwen), 0.11 (Llama)
- **H2O preferentially retains TC-selective positions** (57% TC vs 40% AC at 50% retention)
- **Quartile analysis:** positions with lowest H2O (would be evicted) have HIGHEST AC scores
- **GQA discovery:** Qwen3-4B-Base also uses GQA (8 KV heads) — invalidates GQA-vs-MHA hypothesis
- **All pre-registered predictions disconfirmed** — H2O importance is orthogonal to AC/TC
- Evidential strength: moderate (surprising negative result with practical implications)
- See `experiment_log/exp_011.md`, `results/exp_011/`
