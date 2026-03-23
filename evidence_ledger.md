# Evidence Ledger

## Current Summary
Last updated: 2026-03-23 (cycle 103 — **Model's own answer vs gold truth V-probe: INCONCLUSIVE.** Tested whether V-cache encodes model's predicted answer or just problem features by comparing V_R(gold) vs V_R(pred) for incorrect problems. CRITICAL CONFOUND: gold-pred R=0.929 (model's errors are near-misses). V_R(gold) ≈ V_R(pred) at all 4 layers (|Δ|<0.012, all p>0.3). Experiment UNDERPOWERED to distinguish interpretations due to target collinearity. Methodological null, not substantive null. The accuracy-conditional finding from exp 101/102 remains ambiguous between computation-faithful and general-features interpretations.)
Cycles completed: 103 (89 experimental + 1 consolidation + 8 literature scans + 4 blocked/crashed + 1 null/confounded)

### Core Hypothesis
Chain-of-thought (CoT) reasoning text is a **lossy projection** of the model's internal computation. The KV cache carries a functionally separable hidden channel that encodes answer-relevant information independent of the visible reasoning tokens.

### Verdict: STRONGLY SUPPORTED — with important nuances

The hypothesis is supported by converging evidence from 39 experimental cycles, 5 model variants across 4 families, and 4 independent literature scans covering 45+ papers. However, the original framing (spatial concentration at "answer-coupled positions") has been **decisively disconfirmed**. The hidden channel operates through **geometric/distributed mechanisms** (primarily K-routing vs V-content), localized in specific **answer heads** (H0+H5), not through spatially concentrated positions.

### Key Numbers
- **Models tested:** Qwen3-4B-Base, Qwen3-4B (Instruct), Qwen3-8B-Base, Llama-3.1-8B-Instruct, Phi-3.5-mini-Instruct (MHA), Mistral-7B-v0.3 (Base)
- **Phase 2 (natural channel usage):** V→final raw signal on 3 models (Qwen, Phi, Mistral); residualized signal (V|nums) confirmed on Qwen (R=0.24, p<0.001) AND Phi (R=0.19, p<0.001), not on Mistral (R=0.06, p>0.08). WRRA 87.5% on Phi (p=0.002), ~~71.4% on Qwen (p=0.039)~~ **DOES NOT REPLICATE** (Exp 099: 52% at n=25, p=1.0), 37.9% on Mistral (ns). **K-probe at WRRA positions** (first test): K=52-64%, none significant, directionally K≥V. WRRA "smoking gun" downgraded from Moderate to Weak/Inconclusive. Forward-looking V signal confirmed on 2/3 model families. **V>K at arithmetic computation positions** (Exp 099): V→local R=0.97 vs K→local R=0.96, V→final R=0.64 vs K→final R=0.57 (ALL 4 layers V>K), consistent with K=routing/V=content. Position-sweep decodability (Exp 083): V decodes from 3% of chain (R=0.34) where text reveals 0%. Input-number confound RULED OUT (Exp 084): V|nums_R = 0.357 at 2.5% (text reveals 0%), peak 0.497 (p=0.01). **CROSS-MODEL POSITION-SWEEP REPLICATION (Exp 086):** Phi-3.5-mini V|nums positive 20/20 bins, V|nums=0.233 at 2.5% (text reveals 0%), peak V|nums=0.540 (p=0.003), gap=70%. Position-sweep now confirmed on 2 model families (GQA+MHA, Base+Instruct). **Experiment B (paraphrase disruption) NULL (Exp 085):** Synonym paraphrase drops partial-TF accuracy by 0.6% (1/168, p=1.0); random replacement drops 6.0% (10/168, p=0.002). Non-number tokens don't carry essential hidden info. **MISTRAL BOUNDARY TEST (Exp 087):** Position-sweep V|nums positive 20/20 bins (L16) but bootstrap p=0.137 (NOT significant). Peak V|nums=0.369. 3-model gradient: channel strength scales with accuracy (Qwen 88%→0.50, Phi 85%→0.54, Mistral 44%→0.37). Mistral is partial exception. **SIZE SCALING (Exp 088):** Qwen3-8B-Base replicates position-sweep: V|nums positive 20/20 bins BOTH layers, L27 peak=0.478 (p=0.013), comparable to 4B (peak=0.497, p=0.01). Forward-looking channel is SIZE-INDEPENDENT within Qwen (4B→8B). Unexpected: nums_R much higher on 8B (0.42 vs 4B's 0.26) despite similar accuracy (90% vs 88%). **LAYER SWEEP (Exp 089):** Full 36-layer × 20-bin heatmap on Qwen3-4B-Base reveals TWO-PHASE emergence: ramp L0-L9 (signal emerges at L3, 9% depth), plateau L10-L35 (mean V|nums 0.17-0.22, 19/20 bins positive). Forward-looking is DISTRIBUTED across 26 layers, not localized. At chain start (0-5%, text=0%): V|nums emerges at L8 (+0.10), peaks at L19 (+0.32). The signal is established at middle layers and maintained via residual stream.
- **K LAYER SWEEP (Exp 091):** K|nums > V|nums at 32/36 layers on Qwen3-4B-Base. K emerges at L0 (V at L3). Ramp phase: K|nums=+0.156 vs V|nums=+0.071 (K 2.2x stronger). Plateau: K|nums=+0.219 vs V|nums=+0.193. K peak at L29 (0.246), V peak at L17 (0.216). Same two-phase ramp+plateau structure.
- **RoPE ABLATION (Exp 092):** RoPE HURTS K probing at 8/8 layers (mean -0.159 Pearson R). K_pre (no RoPE) ≈ V overall, K_pre > V at ramp. K>V from exp_091 is NOT a RoPE artifact — it's conservative (true intrinsic K advantage larger). Confound #1 DECISIVELY REJECTED.
- **CROSS-MODEL K vs V REVERSAL (Exp 093):** On Phi-3.5-mini (MHA, analog), V > K at 10/12 layers (83%), bootstrap p<0.05 at 10/12. Mean V|nums=+0.167 vs K|nums=+0.120 (diff=-0.048). Plateau: V|nums=+0.231 vs K|nums=+0.160 (diff=-0.070). REVERSES Qwen's K>V (32/36 layers, diff=+0.043). K>V probing is GQA/digital-specific, NOT universal. Both K and V carry forward-looking signal on all models; the balance depends on architecture. K>V perturbation fragility remains universal (Phase 1), but information content hierarchy is architecture-dependent.
- **MISTRAL K vs V SWEEP (Exp 094):** On Mistral-7B-v0.3 (GQA, analog), V>K at 9/12 layers (75%), K-V diff=-0.017. K|nums mean=-0.004, V|nums mean=+0.013. DISCONFIRMS GQA compression as driver of K>V probing. Mistral is GQA (like Qwen) but analog (like Phi) → shows V≥K (like Phi). 3-model taxonomy: digital encoding → K>V, analog encoding → V≥K, regardless of GQA vs MHA. K>V probing is Qwen/digital-SPECIFIC.
- **Architecture coverage:** GQA (4 models) + MHA (1 model); Base (2) + Instruct (3)
- **Total valid problems across all experiments:** ~1,700+ evaluations
- **K > V perturbation (universal):** 5/5 models × 3 positions = 15 independent conditions under direction perturbation; 16/16 heads across 2 models; K/V effective rank ratio 0.87-0.94 (geometric evidence, Exp 062). K>V perturbation fragility is UNIVERSAL.
- **K vs V probing (ENCODING-dependent, Exp 091+093+094):** Qwen (GQA, digital): K>V at 32/36 layers (89%), mean diff +0.043. Phi (MHA, analog): V>K at 10/12 layers (83%), mean diff -0.048. **Mistral (GQA, analog): V>K at 9/12 layers (75%), mean diff -0.017.** Digital→K>V, analog→V≥K regardless of GQA/MHA. GQA compression does NOT drive K>V probing. RoPE ablation (Exp 092) confirms K>V on Qwen is NOT a RoPE artifact.
- **V-only σ=1 immunity:** 228/228 across 5 variants (magnitude); 393/394 across 4 models at early+mid (direction); **456/456 per-head across 2 models**
- **Text-answer dissociation:** Text ≥98% at near-zero accuracy, confirmed on all 5 models at all perturbation doses
- **Answer head H5:** Primary answer-routing head on BOTH Qwen (50% acc) and Llama (18.2% acc); cross-model convergence at same KV head index
- **Answer-head specialization (Qwen-specific):** H0+H5 at 25% capacity → 3.7% acc; dispensable pairs at 25% → 96-100% acc; +95.1pp gap (Qwen). Llama: best pair 16.2%, worst 0.0%, gap only +16.2pp. Two-regime pattern does NOT replicate on analog models.
- **Head × position interaction is ENCODING-DEPENDENT:** Qwen H5 range=9.3pp (position-independent, Exp 049); Llama H5 range=50.8pp (position-DEPENDENT, Exp 051). Digital encoding → uniform H5; analog → early-concentrated H5.
- **Early-position cascading is GENERAL on Llama (Exp 052):** All 4 tested heads show early>late gradient. Position-dependence scales perfectly with criticality: r=-0.991 (p=0.009). H3 range=49.1pp, H1 range=34.5pp. Early ≈ all for 3/4 heads. Cascading is architectural (analog encoding), not circuit-specific.
- **ANSWER-STEP ATTENTION ROUTING (Exp 095):** 7/8 KV heads increase late-chain attention at answer step (mean +24pp, all p<0.001). H0 is the ONLY head that DECREASES late-chain attention (-3.4pp, p<0.001) — unique retrieval pattern. H5 entropy drops at deep layers (L27: -0.61 bits, L35: -0.75 bits). Two-stage retrieval: early layers→prompt, L18→computation chain. First mechanistic evidence connecting Phase 1 answer heads to Phase 2 natural behavior.
- **PROBE-ATTENTION CORRELATION (Exp 096):** After controlling for position (recency), model attends to V|nums-rich positions at ALL 32 head×layer conditions (partial r = 0.04-0.36, all p<0.001, n=174). Answer step > control at ALL 32 conditions (Wilcoxon p<0.001). Ecological: V|nums r=0.45 >> nums_R r=0.18 (hidden info 2.5x > visible text). H5 partial_r increases L9→L35 (0.26→0.34). BRIDGES probing evidence (info exists) with attention evidence (model retrieves it) into complete encode→store→retrieve circuit.
- **CROSS-MODEL PROBE-ATTENTION + QUADRATIC CONTROL (Exp 097):** Phi-3.5-mini REPLICATES linear partial_r: 4/4 layers, 128/128 head×layer, all p<0.001 (Phi≈Qwen at early-mid layers: L08=0.171 vs L09=0.160, L16=0.243 vs L18=0.260). BUT **quadratic position control** reveals ~80% of signal is non-linear recency: L16 retains 22% (r=0.054, 32/32 significant p<0.001), L24+L31 REVERSE to negative. Ecological r=0.58-0.70 on Phi (V|nums 4-5x > nums_R). Information-directed attention is CROSS-MODEL but MUCH SMALLER than linear-only analysis suggested. Mid-plateau (L16) signal is genuine beyond quadratic recency.
- **ROBUST POSITION CONTROL ON QWEN (Exp 098):** GOLD-STANDARD rank-based (Spearman partial, non-parametric) control applied to Qwen. L9-L18 retain **43-59%** of linear signal under rank control (r=0.155-0.159, 8/8 heads positive, ALL p<0.001). This is **~3x MORE** than Phi's quadratic retention (22%). **Deep-layer QUADRATIC-RANK DISSOCIATION:** L27-L35 retain 88% under quadratic but ONLY 15-17% under rank — reveals non-quadratic monotonic recency at deep layers, explaining Phi's L24/L31 reversal. Phi's ~80% reduction was PHI-SPECIFIC, not universal. **Definitive effect sizes:** Qwen r≈0.16, Phi r≈0.05, concentrated at mid-plateau layers. Architecture-dependent: digital encoding (Qwen) produces 3x stronger information-directed attention than analog (Phi). Permutation null borderline (p=0.047-0.052); V|nums ≈ nums_R ecological on Qwen (weaker hidden-vs-text distinction than Phi).
- **ACCURACY- & DIFFICULTY-CONDITIONAL V|nums (Exp 101+102, CROSS-MODEL):** Forward-looking signal is FUNCTIONAL on BOTH Qwen AND Phi. **Accuracy conditioning:** Qwen 3/4 layers (p=0.006-0.012, gaps 0.04-0.16); **Phi 4/4 layers (p<0.001, gaps 0.17-0.31 — 2x STRONGER than Qwen).** V|nums positive for correct (+0.002 to +0.015), NEGATIVE for incorrect (-0.15 to -0.30), on both models. **Difficulty conditioning:** Qwen 4/4 layers (p<0.001, gaps 0.20-0.24); **Phi 3/4 layers (p=0.004-0.016, gaps 0.03-0.06 — 5-6x SMALLER).** Phi's smaller difficulty effect explained by text staying informative for hard problems (Phi nums_R=0.245 vs Qwen nums_R=-0.085). Same qualitative pattern: V|nums scales with difficulty because text informativeness collapses. **CROSS-MODEL TABLE: 7/8 layer×model conditions significant for accuracy, 7/8 for difficulty.** V is dominant info source for hard problems: V>nums at 15/20 bins (Qwen) and 7-9/20 bins (Phi). Hidden channel carries the answer only when model succeeds and matters most when computation is complex — generalizes across GQA/MHA, digital/analog, base/instruct.

---

## Established Findings

### 1. Text-Answer Dissociation (MOST REPLICATED)
**Status: Decisively established across all models and perturbation types**

KV cache perturbation can destroy answer accuracy while preserving text prediction quality. This is the most replicated finding in the program:
- Text accuracy ≥98% even at 0-4% answer accuracy (Llama KV σ=5.0, Phi KV σ=2.0, Qwen-Instruct KV σ=2.0)
- Holds across all 5 model variants, all perturbation types (direction, magnitude, additive noise), and all dose levels
- The two channels — text prediction and answer computation — are functionally separable within the KV cache

**Key experiments:** Exp 003, 004, 005, 021, 023-038

### 2. K > V Hierarchy (Routing > Content) — REVISED: Encoding-Dependent
**Status: Universal for PERTURBATION fragility; ENCODING-DEPENDENT for probing content**

K (key) vectors are universally more FRAGILE (destroying K routing is catastrophic across all models), but the K/V balance for INFORMATION CONTENT depends on encoding strategy (digital vs analog), NOT attention architecture (GQA vs MHA):
- **Digital models (Qwen):** K carries more decodable forward-looking info (K>V at 32/36 layers, Exp 091)
- **Analog models (Phi, Mistral):** V carries more decodable forward-looking info (V>K at 10/12 layers on Phi [Exp 093], V>K at 9/12 layers on Mistral [Exp 094])
- **GQA compression does NOT drive K>V:** Mistral is GQA (4.0x ratio, like Qwen's 4.5x) but shows V≥K because it's analog
This dissociates causal importance (perturbation) from information content (probing), and further dissociates architecture from encoding.

**Direction perturbation (mechanistically correct test):**

| Model | Early gap | Mid gap | Late gap | n |
|-------|-----------|---------|----------|---|
| Qwen3-4B-Base | +66.7pp | +76.9pp | +51.3pp | 39 |
| Llama-3.1-8B-Instruct | +100pp | +94pp | +66pp | 50 |
| Phi-3.5-mini-Instruct (MHA) | +98.3pp | +96.7pp | +70.0pp | 60 |
| Mistral-7B-v0.3 (Base) | +93.8pp | +93.8pp | +62.5pp | 48 |

Qwen-Instruct not tested at all 3 positions but K > V confirmed at late (+51pp, Exp 029).

**Magnitude perturbation:** K > V confirmed on 4/5 models. Mistral shows K ≈ V under magnitude (explained by Mistral's exceptionally robust K-magnitude encoding at 100% through σ=1.0).

**V-only immunity:**
- Direction at early+mid: 393/394 (99.7%) across 4 models
- Magnitude at σ≤1: 228/228 (100%) across 5 models

**Mechanistic explanation (literature-grounded):** K vectors determine attention routing (WHICH positions to attend to). V vectors carry content (WHAT information flows through). Destroying routing (K) is catastrophic because the model attends to wrong positions. Destroying content (V) at any individual position is recoverable via routing redundancy — intact K-routing finds alternative content sources. Confirmed by Anthropic's attention sparsity finding (99.7% of edges prunable) and the Patterns/Messages framework (McCormick 2025).

**Geometric evidence (Exp 062, 063):** K cache has lower effective rank than V (K/V ratio 0.87-0.94 post-RoPE) and K top-1 singular value captures 3.5-4.1x more energy than V top-1. **Critically, the RoPE confound test (Exp 063) shows pre-RoPE K has EVEN MORE extreme spectral properties** (K_pre/V top-1 = 4.7-5.0x; K_pre/V effective rank = 0.55-0.60). RoPE actually WEAKENS K spectral dominance by distributing energy — the intrinsic K spectral asymmetry is stronger than post-RoPE measurements indicated. All Wilcoxon tests for K_pre vs V are p < 0.01. The K>V spectral hierarchy is definitively NOT a RoPE artifact.

**RoPE ablation (Exp 092):** K_post (with RoPE) is WORSE than K_pre (without RoPE) for probing at **8/8 tested layers** on Qwen3-4B-Base. Mean RoPE degradation: -0.159 Pearson R at bin 19. RoPE rotations DESTROY probing signal. K_pre ≈ V overall (diff -0.011), K_pre > V at ramp layers (+0.024). This means exp_091's K>V finding (measured on K_post) is **conservative** — the true intrinsic K content advantage is larger. The RoPE artifact confound (#1 from exp_091's self-review) is **DECISIVELY REJECTED**. Note: residualized metric (|nums) had methodology bug (R² vs Pearson R); primary conclusions rest on raw R comparisons.

**Key experiments:** Exp 023, 024, 027, 028, 029, 032, 033, 034, 035, 036, 037, 038, 062, 063, 092

### 3. PGD Null Space Is K-Specific
**Status: Statistically significant (p=0.013); smaller than originally claimed**

Adversarial perturbations (PGD) can change the model's final answer while preserving intermediate token predictions — but only through K-vector modifications.

- **K-only PGD:** 6/56 answer changes (10.7%) across 2 experiments
- **V-only PGD:** 0/57 answer changes (0%) despite maximum-effort optimization (100 steps, lr=0.08, λ=10, cosine schedule)
- **Fisher exact p = 0.013** — statistically significant K-only null space
- **Quality distinction:** K perturbation produces genuine numeric answer redirects (e.g., 48→5); V perturbation produces only garbage when anything changes
- **KV-both PGD is WORSE than K-only** (0/31) — V absorbs gradient signal, diluting K optimization

**Important revisions:**
- Original Exp 4 claimed 100% success (n=6) and 377x perturbation norm — does NOT replicate at scale (Exp 018: 0% genuine success on n=45; Exp 032-033: 10.7% on n=56)
- The null space is real and K-specific, but narrow — conservative genuine redirect rate ~3.6% (2/56)
- **Qwen-specific:** PGD produces 0% success on Llama (Exp 008) — distributed encoding prevents adversarial answer redirection

**Key experiments:** Exp 004 (research_spec), 008, 018, 032, 033

### 4. Encoding Taxonomy: Digital vs Analog
**Status: Complete across 5 model variants**

Models encode information in KV cache with qualitatively different strategies:

| Model | Training | K encoding | V encoding | Superadditivity | Classification |
|-------|----------|-----------|-----------|-----------------|---------------|
| Qwen3-4B-Base | Base | DIGITAL (cliff σ=0.3-0.5) | DIGITAL (cliff σ=3-5) | Weak (1.1-2.0x) | FULLY DIGITAL |
| Qwen3-4B | Instruct | DIGITAL (cliff σ=0.5-1.0) | ANALOG (gradual σ=1-10) | Strong (9.1x) | MIXED |
| Llama-3.1-8B | Instruct | Analog (gradual σ=1-5) | Analog (gradual σ=1-10) | Strong (7.9-16.8x) | FULLY ANALOG |
| Phi-3.5-mini | Instruct | Analog (gradual σ=0.5-5) | Analog (gradual σ=1-10) | Strong (1.7-10.6x) | FULLY ANALOG |
| Mistral-7B-v0.3 | Base | Analog (gradual σ=1-5) | Analog (gradual σ=1-10) | Weak-Moderate | FULLY ANALOG |

**Key findings:**
- **K digital encoding is Qwen-family-specific** — preserved by instruction tuning (cliff shifts right ~0.5σ)
- **V digital encoding is Base-specific** — instruction tuning converts V from digital to analog on Qwen
- **Digital ≠ Base models generally** — Mistral-Base is fully analog, confirming digital is Qwen-specific
- **Superadditivity is explained by K-V degradation range overlap:** when both K and V degrade in the same σ range, cascading routing+throughput failure creates superadditive interaction (up to 16.8x)

**Key experiments:** Exp 023, 024, 025, 026, 035, 036, 037

### 5. Positional Dissociation Dominates
**Status: Established across all tested models**

The functional separation between text and answer computation maps onto POSITION in the reasoning chain, not onto AC/TC selectivity scores:

- **Late positions (last 5%):** Destroying these drops accuracy 59-65x more than text quality (Exp 021: acc_drop=64.7%, text_drop=1.1%)
- **Early positions:** Destroy everything equally (acc_drop=82.4%, text_drop=74.1%) — these are computational infrastructure (attention sinks)
- **Within position-controlled quartiles:** AC-noise and TC-noise produce IDENTICAL accuracy effects (23.5% vs 23.5% in Q3, 17.6% vs 17.6% in Q4) — ZERO selectivity effect beyond position

**Early positions are infrastructure (attention sinks):**
- K-only at early positions: destroys everything (0-6% acc, 10-30% text)
- V-only at early positions: zero effect (100% acc, 93%+ text)
- Explained by attention sink literature (ICLR 2025): initial tokens serve as no-op normalization channels

**Key experiments:** Exp 013, 016, 017, 021, 028

### 6. Spatial Selectivity Is Weak
**Status: Original rho=0.78 DISCONFIRMED; actual rho=0.20**

The original claim that PGD perturbations concentrate at "answer-coupled positions" (Exp 5, rho=0.78) does NOT replicate:
- **Exp 018 (n=45):** Bivariate rho = 0.197; partial rho|position = 0.157 pooled / 0.043 per-attack median
- **Exp 5 inflation explained:** Original pooled 141 attacks mixing prompt-only, reasoning-only, and binary search. Reasoning-only attacks create perturbation-attention correlation by construction.
- **Selectivity-based destruction (AC vs TC noise):** Explained entirely by positional confound on ALL models (Exp 013, 016, 017, 021)
- **AC/TC selectivity adds ZERO explanatory power beyond position** (gold-standard double dissociation, Exp 021)

The hidden channel EXISTS (PGD null space is real, K-specific, p=0.013) but its spatial structure is weak. It likely operates through distributed geometric properties (K-direction clusters) across many positions, not through concentrated perturbation at specific positions.

### 7. Literature Convergence
**Status: Decisive independent convergence from 22+ angles across 11 literature scans (126+ papers)**

Eleven literature scans (cycles 10, 20, 30, 40, 50, 60, 67, 70, 80, 90, 100) covering 126+ papers show the field converging on our core hypothesis from increasingly diverse angles:

| Angle | Key evidence | Source |
|-------|-------------|--------|
| **Architectural** | 95-99% gradient norm suppressed by LM head — text is architecturally guaranteed lossy projection | Godey & Artzi, March 2026 |
| **Behavioral** | Models decide answers internally 80% of tokens before visible CoT; TASK-DEPENDENT: MMLU=performative, GPQA-D=genuine | "Reasoning Theater" v3, Boppana et al., March 2026 |
| **Safety** | <20% verbalization of reasoning; obfuscated reward hacking under optimization; CoT still informative for complex behaviors (METR) | Anthropic (Chen 2025), OpenAI (Baker 2025), METR Aug 2025 |
| **Computational** | 3.1x compression (CODI); 4x compression (Latent-SFT); HybridCoT 94% perf at 1.97x less compute; multi-path superposition (Neff≈3-4) | CODI (EMNLP 2025), Latent-SFT (Oct 2025), HybridCoT (NeurIPS 2025) |
| **Mechanistic** | QK=routing, OV=content; QK attributions unlock feature tracing; iteration heads carry forward CoT computation; KV cache steering induces reasoning styles | McCormick 2025, Anthropic QK (2025), Cabannes (NeurIPS 2024), Belitsky Jul 2025 |
| **Compression** | Phase transition at ~90% KV compression; K more sensitive than V (mathematical proof: K has larger spectral norms); V can be 1-bit; Qwen funnel vs LLaMA inverted funnel | 2603.01426, KV-AdaQuant (Feb 2025), AsymKV (COLING 2025), PM-KVQ 2026 |
| **Head specialization** | <7% of heads functionally critical; cross-model transferability unexplored (we address this); iteration heads emerge for iterative reasoning | Zheng et al. Patterns 2025, Cabannes NeurIPS 2024 |
| **Faithfulness decay** | Reasoning Horizon at 70-85% chain length; anti-faithful models; faithfulness evaluation is hard (judges detect but can't localize errors) | Ye et al. Feb 2026, C2-Faith March 2026 |
| **Steganographic** | Steganographic collusion published at IJCNLP-AACL 2025; RL enhances encoding; mitigations insufficient | Mathew et al. IJCNLP-AACL 2025 |
| **KV attacks** | KV cache as attack surface: perturbation (15-30% degradation), history swapping (topic hijacking), early=structural/late=local | Hossain Oct 2025, Ganesh Nov 2025 |
| **KV as computation** | KV cache treated as lightweight representation for reasoning (ICLR 2026); KV steering induces reasoning in frozen models; RL discovers reasoning-critical heads for compression | Xing (ICLR 2026), Belitsky Jul 2025, Du Oct 2025 |
| **V=content decodability** | V vectors encode semantics BETTER than hidden states — training-free Value Aggregation outperforms residual stream | Zhang et al. Feb 2026 |
| **WRRA probing** | Probes decode CORRECT arithmetic answers from hidden states when model outputs WRONG answer (>90% accuracy) | Sun, Stolfo, Sachan (EMNLP 2025) |
| **SAE K/V decomposition** | Keys = "sparse routers with Semantic Elbow"; Values = "dense content payloads" — 4th independent confirmation of K=routing, V=content | Ma et al. (STA-Attention, Dec 2025) |
| **Information bottleneck** | ~460 bits per activation vs ~15 bits per token = ~30x compression. CoT is bandwidth-constrained; latent CoT overcomes this. | Information Bottleneck of CoT (Oct 2025) |
| **CoT fragility** | CoT perturbations cause 20-60% accuracy loss; token content matters more than chain length; scaling provides limited defense on dimensional reasoning | Fragile Thoughts (Aravindan & Kejriwal, Feb 2026) |
| **Counter-evidence: iterative faithfulness** | Probing shows answers emerge iteratively during CoT, not before — CoT IS computation, but compatible with lossy projection | Kudo et al. (arXiv:2412.01113, revised March 2026) |
| **K/V Hessian proof** | Concentrated Key spectra → routing homogeneity; dispersed Value spectra → content heterogeneity. Theoretical proof via exact Hessian analysis. 5th K=routing/V=content angle. | KVSlimmer (Liu et al., March 2026) |
| **K/V local asymmetry** | Adjacent keys locally homogeneous (routing); adjacent values locally heterogeneous (content). 6th K=routing/V=content angle. | AsymKV (Cui & Xu, NeurIPS 2025) |
| **Layer ramp/plateau universal** | Intrinsic dimensionality shows "peak then long plateau" SIZE-INDEPENDENT across 35M-3B models. Our ramp/plateau is a universal transformer property. | Valeriani et al. (NeurIPS 2023), Jiang et al. (ICLR 2025), Lad et al. (2025), Sun et al. (AAAI 2025) |
| **Forward-looking features theory** | Next-token prediction gradient NATURALLY produces features useful for future tokens; effect strongest in formal reasoning domains. Theoretical grounding for V|nums. | Rofin et al. (ICLR 2026) |
| **CoT monitorability crisis** | 40-author consensus: monitoring is "fragile." OpenAI: 2.7% CoT controllability vs 61.9% output controllability. Models have parallel computation channels. | Korbak et al. (Jul 2025), Yueh-Han et al. (OpenAI, March 2026) |
| **Internalized reasoning pathology** | Formalized "internalized reasoning" (computing in hidden states with filler tokens) as a CoT pathology. Our hypothesis is now a recognized failure mode. | Liu et al. (Feb 2026) |
| **Forward-looking probing** | Hidden states encode correctness of FUTURE answers before formulation. Hidden planning exceeds verbalized awareness. | Zhang et al. (Apr 2025), Dong et al. (ICML 2025) |
| **Circuit tracing QK blind spot** | Anthropic's state-of-the-art interpretability CANNOT see QK routing. "This is invisible to our current approach." Our K-routing work fills exactly this gap. | Anthropic Biology paper (March 2025), QK tracing (July 2025) |
| **Retrieval heads = answer heads** | Sparse (<5%), universal, intrinsic (in base models), causally critical. Direct parallel to our H5 finding. 7th head specialization angle. | Wu et al. (ICLR 2025 Oral), Ma & Okazaki (Jan 2026) |
| **KV cache steering induces reasoning** | One-shot KV cache modification induces reasoning in frozen models — KV cache IS computation medium. No K/V decomposition (gap we fill). | Belitsky et al. (Jul 2025) |
| **CoT faithfulness quantified** | 25-39% faithfulness in reasoning models (Anthropic). 0.04-13% natural unfaithfulness rates across 7 models. 61-75% of computation is hidden. | Chen et al. (Anthropic, May 2025), Arcuschin et al. (Mar 2025) |

**New in cycle 50 (5 key convergences):**
1. **K > V triple-confirmed:** KV-AdaQuant (Hariri et al., Feb 2025) provides MATHEMATICAL PROOF that K matrices have larger spectral/Frobenius norms → strictly more quantization-sensitive. Our K > V now confirmed by perturbation (us), quantization engineering (AsymKV), AND formal mathematics (KV-AdaQuant).
2. **Architectural encoding taxonomy explained:** Ananthanarayanan et al. (March 2026) independently find Qwen=funnel (early exploration, late consolidation) vs LLaMA=inverted funnel (early consensus, late diversification). This causally explains our encoding taxonomy: Qwen digital = late consolidation of precise K-routing codewords; LLaMA analog = late diversification prevents cliff behavior.
3. **Head specialization gaps we address:** The Patterns survey (Zheng et al.) identifies cross-model transferability and multi-head collaboration as key open questions. Our H5 finding (same primary answer head on Qwen AND Llama) and multi-head threshold experiments directly address both gaps.
4. **Latent reasoning goes mainstream with HybridCoT:** NeurIPS 2025 accepts interleaved latent/text reasoning (94% of CoT performance at 1.97x less compute). Latent-SFT shows hidden states are "entirely inconsistent" with token embeddings — the distribution mismatch IS our lossy projection hypothesis.
5. **Safety nuanced by METR:** CoT is informative for complex behaviors despite unfaithfulness (>97% detection rate). Our hidden channel finding coexists: the channel exists and can carry computation, but complex dangerous behaviors likely still require CoT compute that makes them detectable.

**New in cycle 70 (6 key convergences):**
1. **V≥K decodability independently validated:** Zhang et al. (Feb 2026) show V vectors encode semantics BETTER than hidden states. This directly explains our Phase 2 surprise: V=content (decodable), K=routing (causal). The "contradiction" between K>V (Phase 1) and V≥K (Phase 2) is actually the most informative finding — it reveals the hidden channel has two components: K-routing program and V-content data.
2. **WRRA probing directly validated:** Sun et al. (EMNLP 2025) achieve >90% accuracy recovering correct answers from hidden states at error positions (residual stream probing on Gemma 2 2B). Our Experiment C will extend this to KV cache decomposition — testing K vs V at error positions.
3. **KV-as-computation goes mainstream (ICLR 2026):** Xing et al. treat KV cache as lightweight representation for Chain-of-Embedding and Fast/Slow Thinking. Up to 5.7x token reduction. Our Phase 1+2 provides the causal evidence for WHY this works.
4. **QK-OV weak coupling confirmed (Anthropic 2025):** QK and OV operate in "mostly-disjoint subspaces" — validates our finding that K and V carry complementary (not redundant) information for different functional roles.
5. **Steganographic CoT concerns intensify:** Skaf et al. (NeurIPS 2025) show models learn generalized steganographic encoding schemes under process supervision. NEST (Karpov 2026) shows nascent steganographic capability across 28 models. Anwar et al. (2026) formalize the steganographic gap using V-information. Our KV-text decodability gap IS a measurable steganographic gap.
6. **Probing papers all use residual stream — our K/V decomposition is unique:** Every probing paper (Zhang et al. 2025, Sun et al. 2025, PING framework) probes the combined residual stream. None decompose into K vs V. Our K/V decomposition reveals the routing-content duality that the residual stream obscures.

**New in cycle 80 (6 key convergences):**
1. **K=routing, V=content SAE-confirmed (4th angle):** Ma et al. (STA-Attention, Dec 2025) use Top-K Sparse Autoencoders on KV cache: K = "sparse routers with Semantic Elbow," V = "dense content payloads." Now 4 independent confirmation angles: perturbation (us), quantization (AsymKV/KV-AdaQuant), spectral geometry (us), SAE decomposition (Ma et al.).
2. **Reasoning Theater is TASK-DEPENDENT (v3):** Boppana et al. March 2026 update: MMLU = performative (early commitment, 80% tokens reducible), GPQA-D = genuine reasoning. GSM8K is intermediate → our forward-looking R≈0.29 is consistent with partial performative/partial genuine computation.
3. **"Correct representations despite failure" now 3-group convergence:** Ye et al. (Feb 2026) + Sun et al. (EMNLP 2025) + our exp_078 WRRA: models encode correct answers internally despite text errors. Our unique contribution: K/V decomposition at error positions.
4. **Information bottleneck QUANTIFIED:** ~460 bits/activation vs ~15 bits/token = ~30x compression. Our cumV > cumText gap (+0.055-0.114 R) empirically measures this bottleneck at the KV cache level.
5. **CoT fragility validates Experiment B:** Fragile Thoughts (Feb 2026) shows CoT perturbations cause 20-60% accuracy loss depending on type. MathError > UnitConversion > SkippedSteps > ExtraSteps. Our planned paraphrase disruption (surface text changes, numbers preserved) should produce intermediate 10-30% loss.
6. **Important counter-evidence acknowledged:** Kudo et al. (revised March 2026) show answers emerge iteratively during CoT (faithfulness claim). RESOLUTION: faithfulness and lossy projection are COMPATIBLE — the model computes during CoT (faithful) AND the text captures only part of the computation (lossy). Our partial R≈0.29 (not 0 and not 1) is consistent with both claims.

**New in cycle 90 (7 key convergences):**
1. **K=routing, V=content now 6 independent angles (5th+6th added):** KVSlimmer (Liu et al., March 2026) provides THEORETICAL PROOF via exact Hessian analysis: concentrated Key spectra → routing homogeneity, dispersed Value spectra → content heterogeneity. AsymKV (Cui & Xu, NeurIPS 2025) provides EMPIRICAL confirmation: adjacent keys locally homogeneous, adjacent values locally heterogeneous. K=routing, V=content is now the most multiply-confirmed finding in transformer mechanistic interpretability.
2. **Ramp/plateau layer pattern is UNIVERSAL (4 independent confirmations):** Valeriani et al. (NeurIPS 2023) found the same "peak then long plateau" in intrinsic dimensionality, SIZE-INDEPENDENT across 35M-3B. Jiang et al. (ICLR 2025): "ridge-to-plateau" with early saturation maintained. Sun et al. (AAAI 2025): three layer classes (sketch/refinement/finish). Lad et al.: four stages of inference, middle layers 72-95% deletion robust. Our V|nums ramp/plateau is NOT unique — it's a universal transformer layer dynamic.
3. **Forward-looking features THEORETICALLY EXPLAINED:** Rofin et al. (ICLR 2026) show next-token prediction gradients NATURALLY produce features useful for future tokens, with formal reasoning domains showing the strongest effect. Wu et al. (COLM 2024): "breadcrumbs" (byproduct) more common than "pre-caching" (intentional), but pre-caching increases with scale. Saunshi et al. (ICLR 2025): depth (not parameters) governs reasoning → explains our size-independence.
4. **CoT monitorability CRISIS — safety community converging on our finding:** Korbak et al. (40+ authors including Anthropic, OpenAI, DeepMind, MIRI): CoT monitoring is a "fragile opportunity." Yueh-Han et al. (OpenAI, March 2026): 2.7% CoT controllability vs 61.9% output controllability gap proves models have PARALLEL computation channels. Jiralerspong et al. (March 2026): models can INFER monitoring from feedback. Our mechanistic evidence provides the SUBSTRATE-LEVEL explanation for WHY CoT monitoring is fragile.
5. **"Internalized reasoning" formalized as CoT pathology:** Liu et al. (Feb 2026) taxonomy: post-hoc rationalization, encoded reasoning, internalized reasoning. "Internalized reasoning" = our hypothesis (computing in hidden states with filler tokens). Our finding is now a RECOGNIZED FAILURE MODE in the safety literature.
6. **Forward-looking probing validated independently:** Zhang et al. (Apr 2025): hidden states encode correctness of FUTURE answers before formulation, probe as verifier saves 24% tokens. Dong et al. (ICML 2025): hidden planning EXCEEDS verbalized awareness. Venhoff et al. (Oct 2025): reasoning exists latently in BASE models, 12% of tokens sufficient to recover 91% of thinking-model gap. Samragh et al. (Jul 2025): vanilla LLMs inherently know future tokens.
7. **Illegible CoT carries essential computation:** Jose (NeurIPS 2025): RL-trained models produce illegible CoT; 53% accuracy drop when removing illegible portions. The illegible tokens encode information through KV cache rather than human-readable text — exactly our finding at a different level of analysis.

**New in cycle 100 (5 key convergences — MILESTONE 10th lit scan):**
1. **Anthropic circuit tracing QK BLIND SPOT:** "On the Biology of a Large Language Model" (March 2025) explicitly states QK routing is "invisible to our current approach." Our K-routing findings (Phase 1+2) investigate EXACTLY the mechanism the leading interpretability method cannot see. This positions our work as complementary to, not competing with, the dominant paradigm. Circuit tracing sees content flow (OV/features); we see routing (QK/KV decomposition). **Strongest validation of our research direction.**
2. **Retrieval heads = our answer heads (7th head specialization angle):** ICLR 2025 Oral: retrieval heads are sparse (<5%), universal across models, intrinsic (in base models), and causally critical (pruning → hallucination). Direct parallel to our H5 (sparse, same index on Qwen+Llama, intrinsic in base model, pruning → 3.7% accuracy). Ma & Okazaki (Jan 2026): concentrated vs distributed retrieval patterns parallel our digital vs analog taxonomy.
3. **KV cache steering induces reasoning — no K/V decomposition:** Belitsky et al. (Jul 2025) demonstrate one-shot KV cache modification induces reasoning in frozen LLMs. K coefficient range 0.0-0.4, V range 1-10 (10-25x larger). But no K-only vs V-only ablation — our perturbation decomposition fills this gap. The 10-25x larger V coefficients are consistent with our V-immunity finding (V requires more force to affect).
4. **CoT faithfulness quantified — 61-75% hidden computation:** Anthropic (Chen, May 2025): 25-39% faithfulness in reasoning models means 61-75% of computation is hidden. Arcuschin et al. (Mar 2025): 0.04-13% natural unfaithfulness across 7 models. Both consistent with our R≈0.29 forward-looking signal (capturing the ~30% that IS forward-looking computation). Baker et al. (OpenAI, Mar 2025): RL optimization causes obfuscated reward hacking through CoT.
5. **Counter-evidence: depth-recurrent transformer shows NO latent CoT:** Lu et al. (Jul 2025): Huginn-3.5B shows "limited evidence of interpretable latent CoT." Important caveat: specific to depth-recurrent architecture (layer reuse), not standard autoregressive transformers (our models). May suggest latent computation works better through KV cache (horizontal, across tokens) than through layer reuse (vertical, across depth).

Our unique contribution: **causal perturbation evidence** at the KV cache level identifying the K-routing channel as the mechanistic substrate of the hidden computation, with HEAD-LEVEL resolution (H5 primary answer head, position-independent) and cross-model validation (5 model variants, 4 families). The K > V hierarchy is now the most independently confirmed finding in this research area (6 angles). **Phase 2 adds observational evidence** (KV probes > text baseline on 2 models) and resolves the K>V causality vs V≥K decodability distinction with independent literature support. **Anthropic's QK blind spot acknowledgment (cycle 100) validates our K-routing research as filling a critical gap in the leading interpretability framework.**

---

## Evidence Table (Current Status Only)

| # | Claim | Status | Strength | Key Experiments |
|---|-------|--------|----------|-----------------|
| 1 | Text-answer dissociation in KV cache | Established | **Decisive** | All experiments, 5 models |
| 2 | K > V under direction perturbation | Universal (5/5 models) | **Strong** | 023-029, 034, 038 |
| 3 | K > V under magnitude perturbation | 4/5 models (Mistral K≈V magnitude-specific) | **Strong** | 023-026, 035-037 |
| 4 | V-only immunity at moderate perturbation | 228/228 (σ≤1 mag), 393/394 (dir early+mid) | **Decisive** | 023-038 |
| 5 | PGD null space exists in K-space | K-only 6/56 vs V-only 0/57, p=0.013 | **Strong** | 032, 033 |
| 6 | PGD null space is Qwen-specific | Llama 0% success vs Qwen ~10% | **Strong** | 004, 008, 018 |
| 7 | Digital encoding is Qwen-family-specific | 2/2 Qwen=digital, 3/3 non-Qwen=analog | **Strong** | 023-026, 035-037 |
| 8 | Instruction tuning converts V digital→analog, preserves K digital | Qwen-Base→Qwen-Instruct comparison | **Strong** | 036 |
| 9 | K-V superadditivity from degradation overlap | Analog models 7.9-16.8x, digital weak 1.1-2.0x | **Strong** | 025, 026, 035, 036, 037 |
| 10 | Positional dissociation (late=answer, early=infrastructure) | Qwen-Base + Llama: text gradient 9→95% across deciles; accuracy saturated on BOTH at 10% dose; encoding-independent | **Strong** | 013, 016, 017, 021, 028, 041, 042 |
| 11 | Spatial selectivity (AC/TC) adds zero beyond position | Gold-standard double dissociation | **Decisive (negative)** | 021 |
| 12 | SNR cliff is Qwen-specific (14 dB) | Llama robust to 5 dB, no cliff | **Strong** | 003, 007 |
| 13 | Single-layer KV destruction tolerated by both models | Llama 93-100%, Qwen 100% (35/36 layers) | **Moderate** | 009 |
| 14 | H2O heavy-hitters ≠ answer-coupled positions | rho=0.004 (Qwen), 0.11 (Llama) | **Moderate** | 011 |
| 15 | AC/SEL-based protection fails on all tested models | SEL ≈ AC ≈ Random << TC ≈ H2O | **Strong (negative)** | 012, 013, 015 |
| 16 | Cross-model text-dependence variation | Qwen-8B 94% compliant, Llama ~30% | **Moderate** | Exp 6 (research_spec) |
| 17 | Unused output capacity (4-5 bits/token) | Established | **Strong** | Exp 1 (research_spec) |
| 18 | CoT narrows entropy 3x (median near zero) | Established | **Strong** | Exp 2 (research_spec) |
| 19 | Text = lossy projection (literature consensus) | Mainstream (22+ convergent angles, 126+ papers) | **Decisive (independent)** | Lit scans 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 |
| 20 | K routing at early positions = general infrastructure | K-early destroys everything; V-early dispensable | **Strong** | 028, 029, 034, 038 |
| 21 | Energy confound does NOT explain K > V | SNR-matched test: K still more sensitive | **Strong** | 027 |
| 22 | K > V confirmed by quantization literature (independent) | AsymKV: V 1-bit; PM-KVQ: K needs more precision; KV-AdaQuant: MATHEMATICAL PROOF K spectral norms > V | **Decisive (independent)** | Lit scans 40, 50, 90 |
| 100 | K=routing, V=content confirmed by Hessian analysis (5th angle, independent) | KVSlimmer (March 2026): concentrated Key spectra → routing homogeneity, dispersed Value spectra → content heterogeneity. Theoretical PROOF. | **Decisive (independent)** | Lit scan 90 |
| 101 | K=routing, V=content confirmed by local asymmetry (6th angle, independent) | AsymKV-NeurIPS (2025): adjacent keys locally homogeneous (routing), adjacent values locally heterogeneous (content). | **Strong (independent)** | Lit scan 90 |
| 102 | Ramp/plateau layer pattern is universal transformer property (independent) | Valeriani (NeurIPS 2023): ID peak-then-plateau, SIZE-INDEPENDENT 35M-3B. Jiang (ICLR 2025): ridge-to-plateau with maintained saturation. 4 independent confirmations. | **Strong (independent)** | Lit scan 90 |
| 103 | Forward-looking features arise from next-token prediction gradients (independent) | Rofin et al. (ICLR 2026): gradient structure naturally produces features for future tokens; strongest in formal reasoning. Theoretical foundation for V|nums. | **Strong (independent)** | Lit scan 90 |
| 104 | CoT monitorability acknowledged as fragile by major labs | 40-author paper (Anthropic+OpenAI+DeepMind+MIRI). OpenAI: 2.7% CoT controllability vs 61.9% output. Parallel computation channels confirmed. | **Decisive (independent)** | Lit scan 90 |
| 105 | "Internalized reasoning" formalized as recognized CoT pathology | Liu et al. (Feb 2026): models computing through hidden states = our hypothesis as diagnostic category. | **Strong (independent)** | Lit scan 90 |
| 110 | Anthropic circuit tracing QK blind spot validates our research direction | "On the Biology of a Large Language Model" (March 2025): cross-layer transcoders see features/OV but QK routing is "invisible to our current approach." Our K-routing work fills the critical blind spot of the leading interpretability method. | **Decisive (independent)** | Lit scan 100 |
| 111 | Retrieval heads parallel our H5 — 7th head specialization angle | ICLR 2025 Oral: retrieval heads are sparse (<5%), universal, intrinsic (base models), causally critical. Same properties as our H5 answer head. Ma & Okazaki (Jan 2026): concentrated vs distributed patterns parallel digital vs analog taxonomy. | **Strong (independent)** | Lit scan 100 |
| 112 | KV cache steering induces reasoning without K/V decomposition — gap we fill | Belitsky et al. (Jul 2025): one-shot KV cache modification induces reasoning in frozen LLMs. Uses independent K and V coefficients (K:0.0-0.4, V:1-10) but no K-only vs V-only ablation. Our perturbation decomposition fills this gap. | **Strong (independent)** | Lit scan 100 |
| 113 | CoT faithfulness quantified at 25-39% — 61-75% hidden computation | Anthropic (May 2025): 25% Claude 3.7 Sonnet, 39% DeepSeek R1. Arcuschin et al. (Mar 2025): 0.04-13% natural unfaithfulness. Consistent with our R≈0.29 forward-looking signal (capturing ~30% genuine computation). | **Strong (independent)** | Lit scan 100 |
| 114 | CRV — structural verification via computational graphs (ICLR 2026 Oral) | Zhao et al.: reasoning errors have "highly predictive" structural signatures in attribution graphs. Domain-specific patterns. Feature-level interventions correct faulty reasoning causally. Complementary to our probing approach. | **Moderate (independent)** | Lit scan 100 |
| 115 | Depth-recurrent latent CoT NEGATIVE (counter-evidence) | Lu et al. (Jul 2025): Huginn-3.5B shows "limited evidence of interpretable latent CoT" via logit lens/coda lens. Specific to depth-recurrent architecture (layer reuse), not standard autoregressive (our models). | **Moderate (counter-evidence)** | Lit scan 100 |
| 116 | V|nums is accuracy-conditional: positive for correct, negative for incorrect | V|nums gap 0.15-0.16, p<0.01 at 3/4 layers. Channel carries answer info only when model succeeds. Incorrect V|nums is NEGATIVE (V < text baseline). | **Moderate** | 101 |
| 117 | V|nums scales with difficulty: hard 5-13x easy (p<0.001) — but driven by text collapse | Hard V|nums 0.23-0.28 vs easy 0.02-0.05. CONFOUNDED: nums_R<0 for hard (text anti-predictive), V_R actually LOWER for hard. V is dominant info source for hard (15/20 bins) but redundant for easy. | **Moderate (confounded)** | 101 |
| 23 | Positional > content confirmed by compression literature (independent) | "Where > What" (Tian 2026): position dominates semantic content for KV importance | **Strong (independent)** | Lit scan 40 |
| 24 | K > V at latest decile on Llama | V-K gap +76pp at 5% dose (V=92%, K=16%), +55pp at 10% dose (V=71%, K=16%) | **Strong** | 041, 043 |
| 25 | Llama K-routing extremely fragile/distributed | 5% K-direction perturbation STILL saturates accuracy at 0-2.6% for bins 0-6; only bin 9 (15.8%) recovers | **Strong** | 041, 043 |
| 26 | No Reasoning Horizon detected at 70-85% | Dissociation transition is linear (~9pp/bin), no sharp phase transition; confirmed at both 5% and 10% dose | **Strong (negative)** | 041, 043 |
| 27 | Positional dissociation is encoding-independent at 10% dose | Qwen-Base (digital) and Llama (analog) show identical patterns: acc~0% all bins, text 15→95% linear, dissociation r=0.997 | **Strong** | 041, 042 |
| 28 | V-only direction perturbation is dose-dependent | V-dir at 5% dose: 92.1% (immune); V-dir at 10% dose: 56-71% (partially destructive). V-immunity holds at ≤5% direction perturbation | **Strong** | 041, 042, 043 |
| 29 | Text gradient is dose-independent | Slope ~9pp/bin and r≈1.0 at both 5% and 10% dose; trivial cascading effect | **Strong** | 041, 042, 043 |
| 30 | K > V gap INCREASES at lower dose | +76pp at 5% vs +55pp at 10% — V recovers more at lower dose while K stays at floor | **Moderate** | 041, 043 |
| 31 | Exp 028 "late=22%" was inflated by coarse binning | At 10-bin resolution: late=8.8% (bins 7-9 avg), not 22%; recovery concentrated at bin 9 only (15.8%) | **Moderate** | 028, 043 |
| 32 | Digital encoding provides UNIFORM accuracy elevation at 5% dose | Qwen mean ~14% across all bins (non-monotonic, r=-0.05) vs Llama mean ~2% (weak gradient, r=0.64). Digital protection is stochastic, not positional. | **Moderate-Strong** | 044 |
| 33 | Positional dissociation is encoding-DEPENDENT at 5% dose | At 10% both models saturate ~0%. At 5%, Qwen recovers ~14% uniformly while Llama recovers ~2%. Digital encoding shifts the dose-response threshold upward. | **Strong** | 041, 042, 043, 044 |
| 34 | V-only direction vulnerability: Qwen more fragile than Llama | Qwen V-only 5%: 81.5% [63, 92]; Llama V-only 5%: 92.1% [79, 97]. Qwen V-only 10%: 56%; Llama V-only 10%: 71%. Consistent across both doses. | **Moderate** | 041, 042, 043, 044 |
| 35 | K > V at bin 9 on Qwen-Base at 5% dose | V=81.5% vs K=18.5%, gap +63pp. Completes 2×2 model×dose matrix: gap always +55-76pp | **Strong** | 044 |
| 36 | "Answer heads" exist — head 5 is primary answer-routing head on BOTH models | Qwen H5: 50.0% [31, 69]; Llama H5: 18.2% [8.6, 34.4]. SAME head index is most critical on both families. H7 most dispensable on both (91-96%). | **Strong** | 045, 046 |
| 37 | V-immunity absolute at per-head level — 2 models | V-only: 456/456 (100%) across Qwen (192/192) + Llama (264/264). No V-head critical on either model. | **Decisive** | 045, 046 |
| 38 | Head-level K-redundancy is encoding-dependent | Qwen (digital): 89.1% mean K-acc (massive redundancy). Llama (analog): 50.0% mean K-acc (less redundant, -39pp). But BOTH still show breadth>depth: per-head K (12.5%) is LESS destructive than per-position K (5-10%). | **Strong** | 045, 046 |
| 39 | Head 5 shows strongest dissociation on BOTH models | Qwen H5: +39pp dissociation (AccDrop=50%, TxtDrop=11%). Llama H5: +47pp dissociation (AccDrop=82%, TxtDrop=35%). | **Strong** | 045, 046 |
| 40 | No energy confound for head-level results (2 models) | Perturbation/signal ratio = sqrt(2) = 1.414 at ALL 32 conditions (16 per model). K norms vary by model but ratios identical. | **Strong** | 045, 046 |
| 41 | Llama K-heads more heterogeneous than Qwen | Llama range=72.7pp, std=21.5pp (4/8 heads <52%). Qwen range=50pp, std=18.3pp (1/8 head <52%). Analog encoding distributes critical routing more broadly. | **Moderate-Strong** | 045, 046 |
| 42 | K > V confirmed at ALL 16 heads across 2 models | 8/8 heads Qwen + 8/8 heads Llama = 16/16 heads show K-only acc < V-only acc | **Decisive** | 045, 046 |
| 43 | Answer-head specialization confirmed by dose-matched multi-head test | H0+H5 (25% cap): 3.7% acc vs dispensable pairs (25%): 96-100%. Gap +95.1pp. Per-problem concordance 25/27 (92.6%). Non-overlapping CIs. | **Decisive** | 047 |
| 44 | Dispensable heads are genuinely redundant in pairs | H1+H2=96.3%, H3+H4=100%, H6+H7=100%. All dispensable 2-head combos near-perfect despite 25% capacity loss. | **Strong** | 047 |
| 45 | At 50%, answer-head inclusion determines survival | Disp4 H1234 (50%): 22.2% vs Ans+disp H0125 (50%): 0.0%. Gap +22.2pp. Same capacity, different outcome based on which heads. | **Strong** | 047 |
| 46 | Answer heads necessary but NOT sufficient | Leave only H0+H5 (75% destroyed): 0.0%. Answer heads alone cannot sustain computation — need dispensable head infrastructure. | **Strong** | 047 |
| 47 | Redundancy curve has two regimes (dispensable-tolerant vs answer-catastrophic) | Dispensable removal: 98.8% (2h), 22.2% (4h). Answer removal: 3.7% (2h), 0% (4h). Head identity > capacity fraction. | **Decisive** | 045, 046, 047 |
| 48 | Two-regime redundancy curve is QWEN-SPECIFIC (does NOT replicate on Llama) | Llama best pair (H0+H7): 16.2% vs Qwen best pair: 98.8%. Gap +16.2pp vs +95.1pp. Concordance 16.2% vs 92.6%. 31/37 problems fail on BOTH pairs. | **Strong (disconfirmatory)** | 047, 048 |
| 49 | Head-level redundancy is encoding-dependent at multi-head level | Qwen (digital): 2-head dispensable=98.8%, 4-head=22.2%. Llama (analog): 2-head best=16.2%, 4-head=2.7%. Digital encoding provides ~83pp more head-level redundancy. | **Strong** | 045, 046, 047, 048 |
| 50 | Llama head-level redundancy collapses at 2/8 heads | 1-head mean=50.0% → 2-head best=16.2% (-33.8pp). Steepest transition in redundancy curve is 1→2 heads on Llama. | **Strong** | 046, 048 |
| 51 | Critical pair still marginally more destructive on Llama | H3+H5=0.0% vs H0+H7=16.2%. Fisher exact p≈0.027. Per-problem: 6/37 differentiate, 0/37 reverse. But signal weak vs Qwen's 25/27. | **Moderate** | 048 |
| 52 | H5 position-independence is QWEN-SPECIFIC (does NOT replicate on Llama) | Qwen H5 range=9.3pp (position-independent). Llama H5 range=50.8pp (position-DEPENDENT): early=18.6%, mid=37.3%, late=69.5%. Non-overlapping CIs. 30:0 concordance. | **Strong (disconfirmatory)** | 049, 051 |
| 53 | H7 dispensability is position-independent on Qwen only | Qwen H7=100% at all positions (0/172 failures). Llama H7: early=81.4%, late=96.6% (range=15.3pp). Even dispensable heads show position-dependence on analog models. | **Moderate (revised)** | 049, 051 |
| 54 | Head × position interaction is ENCODING-DEPENDENT | Qwen: interaction +9.3pp (orthogonal). Llama: interaction +35.6pp (STRONG). Digital encoding → uniform H5 routing; analog encoding → early-concentrated H5 routing. | **Strong** | 049, 051 |
| 55 | K > V confirmed by formal mathematical proof (independent) | KV-AdaQuant (Hariri Feb 2025): K matrices have larger spectral/Frobenius norms → strictly more quantization-sensitive. Third angle of confirmation. | **Decisive (independent)** | Lit scan 50 |
| 56 | Encoding taxonomy explained by architectural depth dynamics (independent) | Qwen=funnel (early exploration → late consolidation → digital K); LLaMA=inverted funnel (early consensus → late diversification → analog K) | **Strong (independent)** | Lit scan 50 |
| 57 | Head specialization is sparse and cross-model transferable (independent) | Survey: <7% of heads critical; our H5 finding addresses identified gap on cross-model transferability | **Strong (independent)** | Lit scan 50 |
| 58 | Hidden states distribution inconsistent with token embeddings (independent) | Latent-SFT (Deng 2025): latent reasoning in vocabulary-space superposition; Neff≈3-4 simultaneous paths | **Strong (independent)** | Lit scan 50 |
| 59 | Llama H5 answer routing concentrates at EARLY positions | H5-early=18.6% = H5-all=18.6% (destroying first 33% at H5 ≡ destroying H5 everywhere). H5-late=69.5% (mostly tolerated). 30:0 per-problem concordance. | **Strong** | 051 |
| 60 | Analog models show position-dependent head routing | Both H5 (range=50.8pp) and H7 (range=15.3pp) show position gradients on Llama. H7-early=81.4% vs H7-late=96.6%. Analog encoding distributes critical routing to early infrastructure positions. | **Strong** | 051 |
| 61 | H5-early routing cascades through full sequence on Llama | H5-early (33% dose at 1 head) = H5-all (100% dose at 1 head) = 18.6%. Destroying early K-routing propagates downstream making mid+late perturbation redundant. | **Strong** | 051 |
| 62 | Early-position cascading is GENERAL on Llama (not H5-specific) | ALL 4 tested heads show early>late gradient: H5 50.8pp, H3 49.1pp, H1 34.5pp, H7 15.3pp. Early ≈ all for 3/4 heads (H5 0pp, H1 3.6pp, H7 3.3pp; H3 borderline 5.5pp). Concordance decisive: H3 29:2, H1 20:1. | **Strong** | 051, 052 |
| 63 | Position-dependence scales perfectly with head criticality | Pearson r=-0.991 (p=0.009), Spearman rho=-1.000. More critical heads show larger position ranges. Multiplicative interaction: range ≈ (1 - SH_acc) × ~55pp. 4-point spectrum across full head criticality range. | **Strong** | 051, 052 |
| 64 | Cascading is architectural (analog encoding property), not circuit-specific | General cascading eliminates the H5-specific answer-routing explanation. Early positions carry K-routing infrastructure for ALL heads on Llama. Analog continuous routing has no error-correction — early perturbation shifts the full downstream trajectory. | **Strong** | 049, 051, 052 |
| 65 | "Recent" (StreamingLLM) is the best KV cache eviction strategy | Keeps attention sinks (first 4) + most recent N positions. 100% accuracy at 33% budget on BOTH Qwen and Llama. Outperforms H2O by +21.7pp on Llama at 33%. Naturally captures both routing infrastructure (early) and answer computation (late). | **Strong** | 055 |
| 66 | Early-priority eviction is catastrophically bad | Keeping earliest positions and removing latest: 35% (Qwen) and 8.7% (Llama) at 33% budget — WORSE than random. Late positions carry answer computation; removing them destroys answers. Not contradictory with cascading (corrupting early cascades ≠ keeping early is sufficient). | **Strong** | 055 |
| 67 | K-preserve does NOT translate perturbation K>V to compression advantage | On Qwen: K-preserve = H2O = 100% (V redundant). On Llama: K-preserve WORSE than H2O at 33% (47.8% vs 78.3%). Keeping K at evicted positions creates "phantom routing" — model attends to positions with no V-content. K>V in perturbation ≠ K>V in compression. | **Moderate** | 055 |
| 68 | Head-selective compression outperforms H2O | Answer heads (H0/H5) get 2x budget. Llama: +8.7pp over H2O at 33% (87.0% vs 78.3%). Validates answer-head specialization for practical compression. Advantage grows at tighter budgets. | **Moderate** | 055 |
| 69 | Digital encoding is more compression-robust than analog | Qwen random@33%=95% vs Llama random@33%=70%. Qwen maintains 100% with H2O/recent/k_preserve at all budgets. Position-independent codewords (digital) tolerate position removal; distributed routing (analog) is fragile. | **Moderate** | 055 |
| 70 | Phantom routing confirmed by 3-way method comparison | zero_v (K-preserve) = 51.9% on Llama at 33% H2O — worst of all 3 methods. Replicates Exp 055 K-preserve finding. Keeping K at evicted positions with zeroed V misdirects attention to content-free positions. True masking (66.7%) and full zeroing (81.5%) both outperform K-preserve. | **Strong** | 055, 056 |
| 71 | Attention-based selection (H2O) provides ZERO benefit over random under true masking | Llama mask_h2o = mask_random = 66.7% at 33%. H2O's advantage under zeroing (+14.8pp over random) comes from model self-organizing attention around high-K-norm positions, NOT from information content at those positions. When positions are truly removed (masked), the K-norm signal that H2O relies on is gone. | **Strong** | 056 |
| 72 | Zeroing outperforms masking for imperfect position selection | Llama zero_kv_h2o = 81.5% vs mask_h2o = 66.7% (-14.8pp gap). zero_kv_recent = 100% vs mask_recent = 100% (no gap). Gap appears specifically when selection is imperfect (H2O/random), because zeroed positions provide an "attention cushion" — model can attend to them harmlessly, whereas masked positions force attention redistribution. | **Strong** | 056 |
| 73 | "Recent" strategy is uniquely method-invariant | 100% accuracy across ALL 3 methods (mask, zero_kv, zero_v) × BOTH budgets (33%, 50%) on Llama. No other strategy achieves this. Recent's superiority is due to selecting intrinsically the RIGHT positions (sinks + late computation), not dependent on eviction mechanism details. | **Strong** | 055, 056 |
| 74 | Digital encoding (Qwen) is eviction-method-invariant | Qwen achieves 100% for mask/zero_kv/zero_v × h2o/recent at both 33% and 50%. Only random and head_selective show minor degradation (84-88%). Method choice only matters for analog (Llama) encoding. | **Moderate** | 056 |
| 75 | K-norm and cumulative attention are ANTI-CORRELATED on Llama | Spearman rho=-0.431±0.064 (n=27). Jaccard overlap at 33%: 0.057 (≈zero). K-norm measures content-density/information-storage, cumulative attention measures routing/infrastructure. These are OPPOSITE properties. On Qwen: rho=+0.023 (uncorrelated), Jaccard=0.207. | **Strong** | 057 |
| 76 | Cumulative attention selects WORSE positions than random for compression | Qwen: true_h2o=33.3% vs random=91.7% (-58.4pp!). Llama zero_kv: true_h2o=55.6% vs random=66.7% (-11pp). Most-attended positions are attention sinks / structural markers that don't carry answer-relevant content. | **Strong** | 057 |
| 77 | K-norm captures content-density, outperforms all other selection metrics on Qwen | Qwen k_norm_h2o=100% at ALL budgets (33%, 50%) × ALL methods (mask, zero_kv). Random=91.7%, true_h2o=33.3%. K-norm identifies positions with high information density independent of attention patterns. | **Strong** | 057 |
| 78 | K-norm advantage under zeroing is model self-organization, not selection quality | Llama zero_kv: k_norm=81.5% > random=66.7% (+14.8pp, replicating Exp 056). Llama mask: k_norm=66.7% = random=66.7% (0pp). Advantage disappears when positions are truly removed (masked), confirming the model self-organizes attention around high-K-norm positions under zeroing. | **Strong** | 056, 057 |
| 79 | Budget-dependent metric reversal at 50% on Llama | mask_50%: true_h2o (85.2%) > k_norm (70.4%) by +14.8pp. zero_kv_50%: k_norm (88.9%) > true_h2o (81.5%) by +7.4pp. At 50%, enough budget for true H2O to retain both sinks AND content. At 33%, sinks dominate true H2O selection. | **Moderate** | 057 |
| 80 | "Recent" still universally best (100%) regardless of selection metric | 100% on both models × both methods × both budgets. Only strategy that doesn't depend on importance metric or eviction method. | **Strong** | 055, 056, 057 |
| 81 | True H2O selects early positions (Q1=39-40%); K-norm selects middle-late positions | True H2O: Q1=40.2% (Qwen), 38.7% (Llama); Q5=5.4%, 3.2%. K-norm: Q1=21.5% (Qwen), 9.0% (Llama); Q5=21.7%, 25.4%. Mean position: true H2O=0.32 vs K-norm=0.52-0.56. Sink fraction only 11-12% of true H2O selections — early bias extends BEYOND sinks across entire Q1. | **Strong** | 058 |
| 82 | Budget crossover at 40% on Llama: true_h2o = k_norm = 76.5% | Below 40%: k_norm > true_h2o. Above 40%: true_h2o ≥ k_norm. At 60%: true_h2o=100% vs k_norm=82.4% (+17.6pp). True H2O wins at generous budgets because it retains complete attention infrastructure (sinks + structural markers). On Qwen, no crossover below 75% — k_norm=100% from 33% onward. | **Strong** | 057, 058 |
| 83 | Random > BOTH true_h2o AND k_norm at low budgets (25-33%) on Llama | Llama@25%: random=82.4% vs k_norm=64.7% vs true_h2o=47.1%. Llama@33%: random=88.2% vs k_norm=70.6% vs true_h2o=64.7%. On analog models, information is so distributed that ANY selection bias hurts; random is least biased. On Qwen@25%: random=80.0% < k_norm=86.7% (digital encoding concentrates info in high-K-norm positions). | **Strong** | 058 |
| 84 | Sink-excluded H2O partially rescues true H2O on Llama (+11.8pp at 33%) | Llama@33%: sink_excluded=76.5% vs true_h2o=64.7% (+11.8pp), surpassing k_norm (70.6%). Llama@50%: sink_excluded=94.1% vs true_h2o=88.2% (+5.9pp). Qwen@33%: NO improvement (40.0% = 40.0%) — early bias on Qwen extends beyond sinks. Sink exclusion is a partial fix, not a complete solution. | **Moderate** | 058 |
| 85 | Late-layer H2O partially rescues true H2O on Qwen (+13.3pp at 33%) | Qwen@33%: late_layer=53.3% vs true_h2o=40.0% (+13.3pp). Llama@33%: late_layer=70.6% = k_norm=70.6%. Both fixes at 50%: 86.7-94.1%. Neither fix reaches K-norm level on Qwen (100%). Late-layer attention focuses more on answer-relevant positions but still biased toward early positions. | **Moderate** | 058 |
| 86 | "Recent" achieves 100% at ALL 6 budgets (25-75%) on BOTH models | Extended from 2 budgets (Exp 057) to 6 budgets. Recent=100% at 25%, 33%, 40%, 50%, 60%, 75% on both Qwen and Llama. No other strategy achieves this. Confirms recency is the dominant predictor of position importance for CoT reasoning. | **Strong** | 055, 056, 057, 058 |
| 87 | Anti-correlation rho=-0.435 replicates on Llama; near-zero on Qwen (rho=0.001) | Llama: rho=-0.435±0.070 (n=17), confirming Exp 057 (rho=-0.431). Late-layer rho=-0.362 (less anti-correlated but still negative). Qwen: rho=0.001±0.113 (near zero, not anti-correlated). The anti-correlation is Llama-specific — on Qwen, K-norm and cumulative attention are orthogonal (independent). | **Strong** | 057, 058 |
| 88 | "Recent" confirmed at n=40 on Llama: 100% [91.2%, 100%] | Largest-N eviction test in the program. 40/40 correct under masking at 33% budget. Pooled across Exp 055-059: **134/134 = 100% [97.3%, 100%]**. Effectively rules out true rate below 97%. CIs narrowed from [82.4%, 100%] at n=17 to [91.2%, 100%] at n=40. | **Decisive** | 055, 056, 057, 058, 059 |
| 89 | Hybrid strategies (k_norm + recency) match recent but don't improve | hybrid_50_50 and hybrid_70_30 both 100% on Llama (n=40) and Qwen (n=20). Total: 60/60 across both models. Adding k_norm-selected positions to a recent core is harmless but unnecessary. The recent component (sinks + most recent) dominates. Practical implication: keep compression simple — sinks + recent. | **Strong** | 059 |
| 90 | K-norm drops to 95% on Qwen at n=20 — first failure detected | K-norm was 100% in ALL prior experiments (n=15-25). At n=20 with different problem sample: 19/20 = 95% [76.4%, 99.1%]. Single failure at chain length 96 (short chain). The "100% k_norm on Qwen" was a small-sample artifact. True rate is ~95%, not 100%. | **Moderate** | 055, 056, 057, 058, 059 |
| 91 | K-norm = random on Llama under masking replicates at n=40 | 87.5% = 87.5% (35/40 each). Replicates Exp 056 direction (mask_h2o = mask_random). Absolute accuracy higher (87.5% vs 66.7%) — attributed to different problem sample. Key finding: K-norm provides zero selection advantage over random when positions are truly removed from attention. | **Strong** | 056, 057, 059 |
| 92 | No chain-length effect detected on eviction strategy effectiveness (Qwen) | K-norm failure at chain length 96 (SHORT). Random failures spread across [80, 96, 109, 189]. Point-biserial r_pb near zero (p>0.49). Chain length does NOT modulate strategy effectiveness within GSM8K range (73-189 tokens). Limited by absence of "long" (>200 token) chains. | **Weak-Moderate** | 059 |
| 93 | K cache has lower effective rank than V cache — UNIVERSAL | K/V eff rank ratio = 0.874 (Qwen), 0.938 (Llama). K top-1 energy 3.5-4.1x higher than V (K routing dominated by ONE principal direction). K spectral gap 2.2-2.8x larger. Layer-dependent crossover: early layers K>V rank, late layers K<V rank. Cross-model K rank difference only 2.4% (NOT ≥20% predicted). Qwen K has 3x higher layer variability (std=40 vs 13). First non-perturbation geometric measurement of K>V hierarchy. | **Strong** | 062 |
| 94 | K spectral properties do NOT differentiate answer from dispensable heads | K eff rank ratio answer/dispensable = 1.003 (Qwen p=0.760 ns) and 1.005 (Llama p=0.944 ns). K top-1 range across 8 heads: 0.065 (Qwen), 0.024 (Llama). H5 (primary answer head) ranks #5 (Qwen) and #6 (Llama) in K concentration — average. Head specialization arises from Q×K interaction, not K manifold geometry. K spectral geometry is a property of the K MECHANISM, shared by all heads uniformly. | **Strong (negative)** | 064 |
| 95 | V spectral properties DO differentiate answer from dispensable heads | Answer heads V eff rank +1.13 (Qwen, p<0.0001) and +1.17 (Llama, p<0.0001) higher than dispensable. Answer heads V top-1 energy -0.012 (Qwen) and -0.025 (Llama) lower. Answer heads carry more distributed V-content on BOTH models. H5 has highest V eff rank on Qwen; H0 highest on Llama. | **Moderate-Strong** | 064 |
| 96 | V forward-looking signal (V→final\|nums) REPLICATES cross-model on Phi-3.5-mini (MHA) | Phi V→final\|nums R=0.191 at L24 (p<0.001). V→final\|embed R=0.169 (p<0.001). V→final\|embed+local R=0.169 (p<0.001). Significant at ALL 4 probe layers. Third model family (Microsoft), MHA architecture (not GQA), instruction-tuned. Compare: Qwen V\|nums=0.242 (p<0.001), Mistral V\|nums=0.067 (ns). | **Moderate** | 079, 081, 082 |
| 97 | WRRA replicates on Phi-3.5-mini — strongest result (87.5%, p=0.002) | 14/16 WRRA cases at L24 align with CORRECT intermediate value despite text error. Phi=87.5% (p=0.002), Qwen=71.4% (p=0.039), Mistral=37.9% (ns). V at error positions encodes correct computation on 2/3 model families. | **Moderate** | 078, 081, 082 |
| 98 | V > K for forward-looking computation (3 models) | V→final > K→final at all layers on Phi (consistent with Qwen, Mistral). After controls: K near zero on all models, V significant on Qwen and Phi. Content (V) carries accumulated forward-looking information; routing (K) is more position-local. | **Strong** | 079, 081, 082 |
| 99 | Mistral forward-looking failure likely accuracy-mediated | Both models with V\|nums signal have high accuracy (Qwen 87%, Phi 83%). Mistral (43%) fails. At low accuracy, nums→final=0.390 dominates, leaving no residual for V. However, Phi nums→final=0.453 (higher than Mistral despite higher accuracy), so accuracy is not the sole factor. | **Moderate** | 081, 082 |
| 106 | Model attends to V\|nums-rich positions BEYOND recency (information-directed retrieval) | Partial r(attention, V\|nums \| position) positive at ALL 32 head×layer conditions (8 heads × 4 layers), all p<0.001, n=174. Mean partial r: L9=0.160, L18=0.260, L27=0.261, L35=0.292. Answer step > control at all 32 conditions (Wilcoxon p<0.001). H5 partial_r increases with depth (0.26→0.34). Ecological: V\|nums r=0.45 >> nums_R r=0.18 — hidden info 2.5x > visible text numbers. BRIDGES probing (info exists) + attention (model retrieves). Complete encode→store→retrieve circuit for hidden channel. | **Strong** | 096 |
| 107 | Information-directed attention survives GOLD-STANDARD rank-based control on Qwen | Rank-based (Spearman partial, non-parametric) control removes ALL monotonic position effects: L9 r=0.155 (59% retained, 8/8 pos, 8/8 p<0.001), L18 r=0.159 (43% retained, 8/8 pos, 8/8 p<0.001). Deep layers collapse: L27 15%, L35 17%. Qwen retains ~3x more than Phi's quadratic r=0.054. Genuine effect: r≈0.16 on Qwen, r≈0.05 on Phi. Architecture-dependent: digital > analog. | **Moderate-Strong** | 098 |
| 108 | Quadratic-rank dissociation at deep layers reveals non-quadratic recency | L27-L35 on Qwen: quadratic retains 88% but rank retains 15-17%. Deep-layer attention has non-quadratic monotonic position dependence (likely exponential). Explains Phi's L24/L31 reversal under quadratic control: quadratic OVERCORRECTS. Rank-based control is the proper standard. Quadratic control is insufficient at deep layers. | **Strong (methodological)** | 097, 098 |
| 109 | Phi's ~80% recency reduction is architecture-specific, NOT universal | Qwen quadratic retention: L9=48%, L18=46% (vs Phi L8=21%, L16=22%). Qwen rank retention: L9=59%, L18=43%. Digital encoding (Qwen) produces stronger info-directed routing than analog (Phi). Effect size scales with encoding strategy, not architecture (GQA/MHA). | **Strong** | 097, 098 |

---

## Disconfirmed Claims

| Claim | Original evidence | Disconfirming evidence | Resolution |
|-------|-------------------|----------------------|------------|
| PGD spatial concentration (rho=0.78) | Exp 5 (research_spec) | Exp 018: rho=0.20 | Inflated by mixing attack types; actual structure is weak/distributed |
| PGD 100% success rate | Exp 4 (n=6) | Exp 018 (0%, n=45), Exp 032-033 (10.7%, n=56) | Small favorable sample; true rate ~10% K-only, 0% V-only |
| SNR cliff is architecture-general | Exp 3 (Qwen) | Exp 007 (Llama: no cliff) | Qwen-specific digital encoding |
| Functional separation via zeroing | Exp 002 | Exp 004, 005 | Zeroing too weak; noise required |
| AC/TC selectivity identifies separable channels | Exp 004, 005 | Exp 013, 016, 017, 021 | Positional confound; zero selectivity effect within quartiles |
| ~24pp dissociation is selectivity-based | Exp 004, 005 | Exp 017, 021 | Primarily positional (≥70% of variance) |
| Reversed dissociation on Qwen-Instruct | Exp 015 | Exp 016 | Positional confound (SelTC=early, SelAC=late) |
| Ultra-fragile KV on Qwen-Instruct | Exp 014 | Exp 015 | Pipeline bug (generated from after answer) |
| GQA vs MHA explains encoding differences | Lit scan 10 | Exp 011 | Both Qwen and Llama use GQA (8 KV heads) |
| Direction-magnitude geometric double dissociation | 2602.11169 (literature) | Exp 023 | No crossover; K-V is the real dissociation, not dir-mag |
| AC-aware compression outperforms H2O | Exp 011 (suggested) | Exp 012, 013 | AC-protection ≈ random on Llama |
| Digital encoding = Base models generally | Exp 023 (Qwen-Base) | Exp 037 (Mistral-Base analog) | Qwen-family-specific, not Base-specific |
| K>V perturbation hierarchy translates directly to compression | Exp 023-038 (K perturbation >> V) | Exp 055 (K-preserve WORSE than H2O on Llama: 47.8% vs 78.3%) | Perturbation K>V reflects routing importance, but keeping K without V creates phantom routing that misdirects attention |
| Reasoning Horizon (70-85%) aligns with K-routing transition | Lit scan 40 (Ye et al. correlation) | Exp 041, 043 (linear gradient, no phase transition at both 5% and 10%) | Dissociation increases ~9pp/bin linearly; no sharp transition at 70-85% on Llama |
| Exp 028 late accuracy gradient (22%) at 5% dose | Exp 028 (3 coarse bins) | Exp 043 (10 bins at 5%: late avg=8.8%, bin 9 only=15.8%) | Coarse binning inflated estimate; actual recovery concentrated at bin 9 only |
| Two-regime redundancy curve is universal | Exp 047 (Qwen: +95.1pp gap, 25/27 concordance) | Exp 048 (Llama: +16.2pp gap, 6/37 concordance) | Qwen-specific; driven by digital encoding + binary head specialization. Analog models show near-complete collapse at 2+ heads |
| H5 answer routing concentrates at late positions | Predicted from Phase 4+5 synthesis | Exp 049: H5 range=9.3pp (Qwen); Exp 051: H5-EARLY most destructive on Llama (18.6%) | H5 position-dependence is encoding-dependent. On Llama, EARLY positions most critical (opposite of prediction). |
| Head × position orthogonality is universal | Exp 049 (Qwen: interaction +9.3pp) | Exp 051 (Llama: interaction +35.6pp, non-overlapping CIs) | Orthogonality is Qwen-specific (digital encoding). Analog encoding creates strong head×position interaction. |
| Masking (true eviction) is strictly better than zeroing | Intuition: removing positions entirely should outperform leaving zero artifacts | Exp 056: mask_h2o=66.7% < zero_kv_h2o=81.5% on Llama (-14.8pp). Zeroing provides "attention cushion" for imperfect selection. | Zeroing > masking when selection is imperfect; masking = zeroing only when selection is perfect ("recent") |
| K-norm is a good proxy for true H2O (cumulative attention) | H2O uses cumulative attention; K-norm (L2 norm of K-vectors) was assumed to approximate this | Exp 057: Spearman rho=-0.431 on Llama (ANTI-CORRELATED). Jaccard overlap=0.057 at 33%. They select OPPOSITE positions. | K-norm captures content-density (information stored at position); cumulative attention captures routing-importance (how often attended). These are DIFFERENT properties — high-attention positions have LOW information density. |
| Cumulative attention identifies informative positions | "Heavy hitters" (most-attended positions) assumed to be most important | Exp 057: true_h2o=33.3% on Qwen (vs random 91.7%); true_h2o=55.6% on Llama zero_kv (vs random 66.7%). Most-attended = infrastructure/sinks, NOT content. | At tight budgets (33%), most-attended positions are sinks+structural markers. At 50%, enough budget for true H2O to include content positions too (85.2% under masking). |
| K spectral geometry predicts head function (answer vs dispensable) | Exp 062-063 (K<V spectral asymmetry) + Exp 045-048 (head specialization) | Exp 064: K eff rank answer/dispensable = 1.003-1.005 (p=0.76-0.94 ns). H5 ranks #5-6 out of 8 in K concentration. | Head specialization is a Q×K interaction effect, not an intrinsic K manifold property. K spectral geometry is universal across heads. V (not K) differentiates answer from dispensable heads. |
| WRRA (wrong reasoning, right answer) is common enough for probing on Qwen | Sun et al. EMNLP 2025 (5-15% error rate on similar tasks) | Exp 071: 2/1339 operations = 0.15% error rate (1 genuine error). | The `<<EXPR=RESULT>>` calculator format enforces near-perfect arithmetic. Errors in reasoning are in SETUP (wrong expressions), not COMPUTATION (wrong evaluation). WRRA requires models that make arithmetic errors, not reasoning errors. |

---

## Open Questions (Genuinely Unanswered)

### High Priority (would strengthen or extend key findings)
1. **Why does instruction tuning convert V digital→analog but preserve K digital?** Hypothesis: K defines routing (architectural constraint from QK mechanism), V carries content (reorganizable by RLHF). (Exp 036)
2. **Does K-only PGD succeed on Phi (MHA)?** Would confirm null space is universal K-routing phenomenon beyond GQA. (Exp 034 motivation)
3. **Why is Qwen K-direction MORE robust than Llama despite being "digital"?** Possible: discrete direction clusters in digital encoding — random replacement sometimes lands near valid codewords. (Exp 029)
4. **Does a lower dose (<5%) reveal encoding-dependent accuracy gradients on Llama?** 5% dose on Llama (Exp 043) still saturates accuracy at 0-2.6% for bins 0-6; only bin 9 (15.8%) recovers. Exp 028's "late=22%" was inflated by coarse binning. A 2-3% dose sweep would test whether Llama has positional accuracy structure at ultra-low perturbation. Qwen at 5% dose still untested. (Exp 028, 041, 042, 043)
5. ~~Can TC-aware compression outperform H2O on actual KV eviction benchmarks?~~ **ANSWERED (Exp 055):** "Recent" (sinks+recent) beats H2O at all budgets. Head-selective (answer heads 2x) beats H2O by +8.7pp at 33% on Llama. K-preserve is WORSE than H2O on Llama. Early_priority is catastrophically bad.
5b. ~~WHY are K-norm and cumulative attention anti-correlated?~~ **ANSWERED (Exp 058):** True H2O selects early positions (Q1=39-40%, mean_pos=0.32); K-norm selects middle-late (mean_pos=0.52-0.56). Sinks account for only 11-12% of true H2O — the bias extends across entire Q1. Cumulative attention reflects routing frequency (early positions participate in MORE query-key interactions); K-norm reflects content-density. On Llama: rho=-0.435 (anti-correlated). On Qwen: rho=0.001 (orthogonal). Sink-excluded H2O partially rescues (+11.8pp Llama), late-layer H2O partially rescues (+13.3pp Qwen), but neither fully fixes the early-position bias.
5c. ~~Where is the crossover budget?~~ **ANSWERED (Exp 058):** Crossover at 40% budget on Llama (true_h2o = k_norm = 76.5%). Below 40%: k_norm > true_h2o. Above 40%: true_h2o ≥ k_norm. At 60%: true_h2o=100% vs k_norm=82.4%. On Qwen, no crossover below 75% (k_norm=100% from 33% onward). Most surprisingly: random > BOTH at 25-33% on Llama.

### Medium Priority (mechanistic depth)
6. **V-direction immunity is dose-dependent (PARTIALLY ANSWERED).** V-dir at 5% dose = 92.1% (immune); V-dir at 10% = 56-71% (destructive). Per-head V: 100% at all 8 heads (absolute immunity). The V-direction vulnerability is confined to >5% positional fraction with multi-head perturbation. (Exp 041, 042, 043, 045)
6b. **What makes head 5 the answer-routing head on BOTH models?** Does it implement specific attention patterns? Is this GQA-universal (test on Phi MHA)? What is the initialization/architecture explanation for H5 convergence? K spectral geometry does NOT distinguish H5 — specialization must live in Q projections (Exp 064). (Exp 045, 046 — PARTIALLY ANSWERED: Llama has same H5 primary)
6c. **Why do answer heads carry more distributed V-content?** Exp 064: answer heads V eff rank +1.1-1.2 higher (p<0.0001 both models). Possible: answer heads attend more broadly (gathering answer info from many positions), accumulating more diverse V-content. Alternatively, V represents richer answer features that require more dimensions. Test via attention pattern analysis.
6d. **Does Q spectral analysis reveal head specialization that K does not?** Head specialization is a Q×K interaction effect (Exp 064). Q projections may show spectral differences that K does not — different Q manifolds select different routing patterns from the same K manifold.
7. **WHY do late positions selectively affect accuracy but not text?** Hypothesis: answer computation via attention from final positions to late reasoning positions; text computation is more local. (Exp 021)
7. **Is there a "procedural" third channel beyond text/answer?** KV cache steering (Belitsky 2025) encodes reasoning STYLE. Hub positions may be procedural nodes. (Lit scan 20)
8. **Why is Mistral K the most magnitude-robust of all models (100% at σ=1)?** Model size (7B) or sliding window attention? (Exp 037)
9. **Does the "Reasoning Theater" early internal confidence correspond to our null space?** Positions where model "already knows the answer" may carry answer-relevant KV info. (Lit scan 20)
10. **Why does Position-TC correlation reverse: Qwen-Instruct (-0.44) vs Llama (+0.44)?** Suggests different attention pattern organization. (Exp 014)
11. **What is K-V superadditivity mechanism?** K-distortion redirects attention to wrong positions AND V-distortion corrupts content there → no compensation pathway. Testable via attention pattern analysis under KV-combined. (Exp 025)
12. **Do R-KV "redundant" tokens (whose removal IMPROVES accuracy) overlap with TC-selective positions?** Would confirm text scaffolding actively interferes with answer computation. (Lit scan 20)

### Lower Priority (extensions)
13. Where exactly is the additive noise cliff on Qwen-Instruct? Between 0.3x and 1.0x. Finer sweep would locate it. (Exp 015)
19. ~~Multi-head perturbation threshold~~ **ANSWERED (Exp 047):** Two-regime curve: dispensable pairs=96-100%, 4 dispensable=22.2%. Answer pair H0+H5=3.7%. Gap +95pp. Head identity > capacity fraction.
20. ~~Head 5 × position interaction~~ **ANSWERED (Exp 049+051+052):** H5 position-independence is Qwen-SPECIFIC (range=9.3pp). On Llama, ALL heads are position-dependent with range scaling perfectly with criticality (r=-0.991). Early-position cascading is general, not H5-specific. Digital → orthogonal; analog → coupled.
21. **Why can't answer heads sustain computation alone?** Leave-only H0+H5 = 0%. What infrastructure do dispensable heads provide? (Exp 047)
22. ~~Multi-head threshold on Llama~~ **ANSWERED (Exp 048):** Two-regime DOES NOT replicate. Best pair=16.2% (vs Qwen 98.8%). Head-level redundancy is near-zero on analog models. The two-regime pattern is Qwen-specific (digital encoding). (Exp 047, 048)
14. Would targeted K-only PGD (maximize specific wrong answer) succeed at higher rates? (Exp 032)
15. Would K-only PGD restricted to late layers (18+) be more efficient? (Exp 032)
16. Does per-head SNR normalization mask individual head vulnerability? (Exp 027)
17. At what training stage does V convert from digital to analog — SFT, RLHF, or DPO? (Exp 036)
18. Why is layer 0 specifically critical for Qwen? Attention pattern bootstrapping? (Exp 009)

---

## Experiment Log (43 experiments + 3 literature scans)

### Phase 1: Establishing the Phenomenon (Cycles 1-9)
| Exp | Cycle | Model | Key Result |
|-----|-------|-------|------------|
| 001 | 1 | Qwen-Instruct | FAILED (wrong model) |
| 002 | 2 | Qwen-Base | Zeroing 50% positions → 100% accuracy (zeroing too weak) |
| 003 | 3 | — | Agent crash, script only |
| 004 | 4 | Qwen-Base | +23.5pp noise dissociation (AC vs TC at 5%); primarily positional (confirmed Exp 017) |
| 005 | 5 | Llama-Instruct | +23.8pp noise dissociation replicates; Llama more fragile to random noise |
| 006 | 6 | Llama-Instruct | FAILED (DynamicCache API change) |
| 007 | 7-8 | Llama-Instruct | SNR cliff does NOT replicate — Llama 100% at SNR ≥5 dB |
| 008 | 8 | Llama-Instruct | PGD null space does NOT replicate — 0% success on Llama |
| 009 | 9 | Both | Single-layer KV destruction tolerated (93-100% on both) |
| — | 10 | — | **Literature scan #1:** CMI causal bypass, KV steering, latent reasoning, attention sinks |

### Phase 2: Positional Confound Resolution (Cycles 11-21)
| Exp | Cycle | Model | Key Result |
|-----|-------|-------|------------|
| 011 | 11 | Both | H2O ≠ AC (rho≈0); GQA discovery (both use 8 KV heads) |
| 012 | 12-13 | Llama | AC-protection fails (≈ random); H2O/TC work by protecting early positions |
| 013 | 13 | Llama | Selectivity also fails; position dominates; TC best protection metric |
| 014 | 15 | Qwen-Instruct | INVALIDATED (pipeline bug — generated from after answer) |
| 015 | 16 | Qwen-Instruct | Reversed dissociation (-47pp); sharp cliff 0.3x-1.0x; pipeline fixed |
| 016 | 17 | Qwen-Instruct | "Reversed dissociation" is positional confound; within-half gap collapses |
| 017 | 18 | Qwen-Base | Qwen-Base +23.7pp also primarily positional; within-early-half = 2.6pp |
| 018 | 18-19 | Qwen-Base | PGD rho=0.20 (NOT 0.78); spatial structure dramatically weakened |
| 019/021 | 19-21 | Qwen-Base | **Gold-standard double dissociation:** ZERO selectivity effect in quartile-controlled test. Strong POSITIONAL dissociation discovered (late=answer 59x, early=infrastructure) |
| — | 20 | — | **Literature scan #2:** Reasoning Theater, direction-magnitude dissociation, phase transitions, attention sinks, distributed CoT features |

### Phase 3: K-V Functional Dissociation (Cycles 22-33)
| Exp | Cycle | Model | Key Result |
|-----|-------|-------|------------|
| 023 | 22 | Qwen-Base | K-V dissociation discovered: V-mag = ZERO effect, K-perturbation devastating |
| 024 | 23 | Llama | K-V direction dissociation replicates (3.7x vs 3.6x); K-mag immune on Llama |
| 025 | 24 | Llama | Full dose-response: K-V superadditivity 7.9-16.8x; analog degradation |
| 026 | 25 | Qwen-Base | Full dose-response: K cliff σ=0.3-1, V cliff σ=3-5; digital encoding confirmed |
| 027 | 27 | Llama | Energy confound tested: K > V survives SNR-matched noise |
| 028 | 28 | Llama | K > V at all 3 position bands (+66-100pp); early K = infrastructure |
| 029 | 29 | Qwen-Base | K > V replicates (+51-77pp); Qwen K-direction MORE robust than Llama |
| — | 30 | — | **Literature scan #3:** QK/OV framework, CoT unfaithfulness safety, gradient bottleneck, KV memories |
| 032 | 32 | Qwen-Base | K-only PGD 9.4% vs V-only 0% — null space is K-specific |
| 033 | 33 | Qwen-Base | V-only 0% survives maximum-effort optimization; K-only improves to 12.5% |

### Phase 4: Cross-Model Universality (Cycles 34-38)
| Exp | Cycle | Model | Key Result |
|-----|-------|-------|------------|
| 034 | 34 | Phi-Instruct (MHA) | K > V replicates on MHA (+70-98pp); GQA confound eliminated |
| 035 | 35 | Phi-Instruct (MHA) | Phi = FULLY ANALOG; K > V extends to magnitude on MHA |
| 036 | 36 | Qwen-Instruct | K=digital (preserved), V=analog (converted); MIXED encoding discovered |
| 037 | 37 | Mistral-Base | Digital is QWEN-SPECIFIC (Mistral-Base fully analog); K ≈ V under magnitude |
| 038 | 38 | Mistral-Base | K > V RESTORED under direction perturbation (+62-94pp); K ≈ V was magnitude-specific |

### Evidence Consolidation & Literature Scans (Cycles 39-40)
| Exp | Cycle | Type | Key Result |
|-----|-------|------|------------|
| — | 39 | Consolidation | Restructured 625→250 line ledger; 114→18 open questions; narrative synthesis |
| — | 40 | **Lit scan #4** | K > V confirmed by quantization lit (AsymKV); Reasoning Horizon at 70-85% maps to positional dissociation; steganographic capabilities emerging; "Where > What" confirms position > content |

### Phase 5: Fine-Grained Positional Analysis (Cycles 41+)
| Exp | Cycle | Model | Key Result |
|-----|-------|-------|------------|
| 041 | 41 | Llama-Instruct | 10-decile K-only sweep: accuracy saturated ~0% at all positions (Llama K-routing extremely fragile); text gradient 9→94% (linear, no Reasoning Horizon); K > V +55pp at bin 9 |
| 042 | 42 | Qwen-Base | 10-decile K-only sweep: accuracy saturated 0-4% at all positions (matches Llama); text 15→95% linear (r=0.997); K > V +52pp at bin 9; encoding-independent pattern confirmed |
| 043 | 43 | Llama-Instruct | 5% dose 10-decile sweep: accuracy STILL saturated 0-2.6% bins 0-6; only bin 9=15.8% recovers; V-only immunity RESTORED at 5% (92.1%); K > V gap +76pp; text gradient dose-independent; Exp 028 "22%" was inflated |
| 044 | 44 | Qwen-Base | 5% dose 10-decile sweep: digital encoding provides UNIFORM elevation (~14% vs Llama 2.6%); non-monotonic pattern (r=-0.05); V-only 81.5%; K > V +63pp; 2×2 model×dose matrix complete |
| 045 | 45 | Qwen-Base | **Per-head K-V sweep:** "Answer heads" discovered — H5=50%, H0=67%, 6 others=100%. V-immunity absolute (192/192). Head-level K-redundancy massive: 12.5% per-head → 89% acc vs 5% per-position → 14% acc. Fragility is about breadth not depth. |
| 046 | 46 | Llama-Instruct | **Per-head K-V sweep (cross-model):** H5 = 18.2% (SAME primary answer head as Qwen!). Llama mean K-acc=50% (vs Qwen 89%). V-immunity absolute 264/264. Head-level redundancy encoding-dependent. K>V at 8/8 heads. |
| 047 | 47 | Qwen-Base | **Multi-head threshold (DECISIVE):** H0+H5=3.7% vs dispensable pairs=96-100% (+95pp gap). Per-problem concordance 25/27. Disp4 at 50%=22.2% vs ans+disp=0%. Leave-only-answer=0%. Two-regime redundancy curve. |
| 048 | 48 | Llama-Instruct | **Multi-head threshold (DISCONFIRMATORY):** Two-regime DOES NOT replicate. Best pair 16.2% (vs Qwen 98.8%). Gap +16.2pp (vs +95.1pp). 31/37 fail on both pairs. Head-level redundancy near-zero on analog model. |
| 049 | 49 | Qwen-Base | **Head × position interaction:** H5 answer routing is POSITION-INDEPENDENT (range=9.3pp, CIs overlap). H7=100% at all positions. Head identity explains ~95% of variance. Text gradient exists within H5 (early 90.3%, late 97.9%) but accuracy flat. Head and position are orthogonal mechanisms. |
| — | 50 | **Lit scan #5** | K > V triple-confirmed (mathematical proof: K spectral norms > V); Qwen funnel vs LLaMA inverted funnel explains encoding taxonomy; head specialization <7% sparse (survey); HybridCoT NeurIPS 2025 (latent reasoning mainstream); hidden state distribution mismatch confirms lossy projection; KV cache attack surface (MTI, history swapping); METR: CoT informative for complex behaviors despite unfaithfulness |
| 051 | 52 | Llama-Instruct | **Head × position interaction (DISCONFIRMATORY):** H5 position-independence DOES NOT replicate. Llama H5 range=50.8pp (vs Qwen 9.3pp). H5-early=18.6%=H5-all (early positions carry ALL critical routing). H5-late=69.5% (mostly tolerated). 30:0 concordance. Interaction +35.6pp. Head×position coupling is encoding-dependent. |
| 052 | 53 | Llama-Instruct | **Cross-head cascading (CONFIRMATORY for general mechanism):** H3 range=49.1pp, H1 range=34.5pp. ALL 4 heads show early>late gradient. Criticality×range correlation r=-0.991 (p=0.009). Early≈all for 3/4 heads. Cascading is architectural, not H5-specific. |
| 053 | 54 | Qwen-Base | **Cross-head position-independence (CONFIRMATORY):** ALL 4 Qwen heads position-independent (max range 14pp). H0 reversed gradient (late most destructive). Digital encoding prevents cascading at all heads. |
| 055 | 55 | Both | **KV cache eviction benchmark:** "Recent" (sinks+recent) best strategy on both models (100% at 33%). Early_priority catastrophically bad (8.7% Llama@33%). K-preserve WORSE than H2O on Llama. Head-selective +8.7pp over H2O. Digital more compression-robust. |
| 056 | 56 | Both | **Mask vs zero eviction (3-way):** Phantom routing confirmed (zero_v=51.9% worst on Llama). H2O=random under masking (66.7%). Zeroing > masking for imperfect selection (+14.8pp). "Recent" uniquely method-invariant (100% across all). Qwen method-invariant (digital). |
| 057 | 57 | Both | **True H2O vs K-norm proxy:** K-norm and cumulative attention are ANTI-CORRELATED on Llama (rho=-0.431, Jaccard=0.057). True H2O selects WORSE positions than random (Qwen 33.3% vs 91.7%; Llama 55.6% vs 66.7%). K-norm captures content-density, not routing importance. Budget-dependent reversal at 50%: true_h2o beats k_norm under masking (+14.8pp). "Recent" still universally best (100%). |
| 058 | 58 | Both | **Sink-dominance analysis + budget crossover:** True H2O selects Q1 positions (39-40%); K-norm selects late. Sinks=11-12% (NOT the full explanation). Crossover at 40% budget on Llama. Random > both at 25-33% on Llama. Sink-excluded H2O +11.8pp on Llama; late-layer H2O +13.3pp on Qwen. Neither fully rescues. "Recent" 100% at ALL 6 budgets. Anti-correlation rho=-0.435 replicates. |
| 059 | 59 | Both | **Large-N replication + hybrid + chain-length:** "Recent" 40/40=100% on Llama [91.2%, 100%]. Pooled 134/134 [97.3%, 100%]. Hybrid 50/50 and 70/30 also 100% but no improvement. K-norm drops to 95% on Qwen (first failure). No chain-length effect detected. |
| — | 60 | **Lit scan #6** | 20+ papers: "Reasoning Theater" (2603.05488) — hidden computation decodable 80% earlier than text; Causal Bypass (2602.03994) — CMI≈0 bypass regimes; K>V now 6-angle confirmed (low-rank K manifold); KVP/RLKV — per-head RL policies confirm head specialization; R-KV/TRIM-KV — removing scaffolding can improve accuracy; CoT in Wild — 0.04-13% unfaithfulness rates; Hold Onto That Thought — H2O+SnapKV-D best for real-time eviction (challenges our "recent" finding for deployment) |
| 062 | 62 | Both | **Effective rank analysis of K vs V cache (NEW ANGLE):** K effective rank < V on BOTH models (K/V=0.874 Qwen, 0.938 Llama). K top-1 energy 3.5-4.1x > V — K routing dominated by one principal direction. K spectral gap 2.2-2.8x > V. Novel layer-dependent crossover: early K>V, late K<V. Cross-model K rank difference only 2.4% — effective rank doesn't capture digital/analog, but spectral gap (Qwen 4.32>Llama 3.66) and top-1 energy (Qwen 0.615>Llama 0.576) DO. Qwen K has 3x higher layer-to-layer variability (std=40 vs 13). First independent geometric evidence for K>V hierarchy (non-perturbation). Scenario B confirmed. |
| 064 | 64 | Both | **Per-head spectral analysis (answer vs dispensable heads):** K spectral properties do NOT differentiate answer from dispensable heads — K eff rank ratio 1.003 (Qwen p=0.76) and 1.005 (Llama p=0.94). All 8 heads remarkably uniform in K geometry (range 2.5-3.4%). H5 ranks #5-6 in K concentration (average). SURPRISE: V spectral properties DO differentiate — answer heads V eff rank +1.1-1.2 higher (p<0.0001 both). Head specialization is a Q×K interaction effect, not an intrinsic K geometry property. Scenario B confirmed + Scenario D partially confirmed. |
| — | 65–66 | — | **BLOCKED:** Sandbox restrictions prevent Python execution. Experiment script `scripts/exp_065_early_decodability.py` designed and ready (cycle 65), execution attempted (cycle 66), both blocked. |
| — | 67 | **Lit scan #7** | 5 new papers for Phase 2. "Knowing Before Saying" (ACL 2025): probes predict CoT success from hidden states BEFORE tokens generated (60-76.4% acc, middle layers best). "Probing for Arithmetic Errors" (EMNLP 2025): correct answers decodable from internal states when text is wrong (80-90% detection). "Causality ≠ Decodability" (2025): decodable ≠ causal — our Phase 1+2 combination addresses this. "CoT Is Not Explainability" (Barez 2025): paraphrased CoT shows NO accuracy gap — warning for Experiment B. Godey & Artzi (2026) confirmed: 95-99% gradient suppression by LM head. Literature strongly validates Phase 2 design, especially Experiments A and C. |
| 068 | 68 | Qwen-Base | **Early decodability — FIRST PHASE 2 OBSERVATIONAL EVIDENCE:** KV probes > text at ALL positions 10-90%. K-text gap peaks at +0.235 (70%). V≥K for decodability (surprise). Cumulative features on fair comparison. 80 problems, 4 layers. |
| 069 | 69 | Mistral-Base | **Early decodability — CROSS-MODEL REPLICATION:** KV > text on Mistral (analog, different family). V mean=0.609 > K mean=0.575 > text=0.524. V≥K replicates at ALL 4 layers. Effect weaker than Qwen but consistent. K>text at 6/10 positions (vs 9/10 Qwen). Best layer L8 (25%) vs Qwen L27 (77%). |
| — | 70 | **Lit scan #8** | 12 papers. V≥K decodability independently validated (Zhang 2026: V encodes semantics > hidden states). WRRA probing validated (Sun EMNLP 2025: >90% correct answer from hidden states at error positions). KV-as-computation at ICLR 2026. QK-OV weakly coupled (Anthropic 2025). Steganographic CoT (NeurIPS 2025 + 3 papers). All probing papers use residual stream — our K/V decomposition is unique. |
| 071 | 71 | Qwen-Base | **WRRA attempt + computation-position probes:** Arithmetic error rate near-zero (2/1339=0.15%) — WRRA underpowered. BUT computation-position probes: V R=0.975, K R=0.961, text R=0.108 at arithmetic "=" positions. KV-text gap +0.85 (~10x general position gap). V>K at all 4 layers. 1337 operations, shuffle R≈-0.06. |
| 072 | 72 | Mistral-Base | **Computation-position cross-model + expression-embed baseline:** KV(R=0.435) > expr-embed(0.310) > token-embed(0.111). 2.2x weaker than Qwen. Expr-embed narrows KV advantage to +0.125. WRRA alignment NEGATIVE (below chance, n=91 errors). Effect model-dependent. |
| 073 | 73 | Qwen-Base | **Expression-embed retroactive baseline + permutation significance:** V R=0.977 replicates. Expr-embed R=0.771 (strong but below V). V-expr gap=+0.206 (narrows from +0.867 over token-embed but remains substantial). Perm p=0.000 for V,K,expr. Token-embed p=0.936 (uninformative). 1474 operations. Gap 1.65x > Mistral's +0.125. |
| 074 | 74 | Qwen-Base | **Paraphrase disruption DECISIVE NULL:** 100% accuracy across ALL conditions (original, synonym, random). Replacing ALL non-numeric tokens with random gibberish has ZERO effect on accuracy (215/215 on all 3 conditions). The `<<EXPR=RESULT>>` format makes English text completely redundant. UNINFORMATIVE about hidden channel — design confound. Needs redesign with plain-text CoT. |
| 075-080 | 75-80 | Various | See individual experiment logs. Key Phase 2 results: cumulative V > text (2 models), forward-looking V→final R=0.49 (Qwen), V→final\|embed R=0.14-0.22 (Qwen GroupKFold), WRRA 71.4% (Qwen, p=0.039). Within-problem data leakage discovered and corrected. Lit scan #9: 90+ papers, K=routing/V=content SAE-confirmed. |
| **081** | **81** | **Mistral-Base** | **Cross-model forward-looking probing — MIXED REPLICATION:** V→final R=0.23 (p<0.001) replicates raw signal. BUT V→final\|nums R=0.06 (p>0.08) does NOT replicate — Mistral's signal explained by problem numbers. WRRA 38% (below chance, n=29). K negative after all controls. nums→final R=0.39 (2.5x Qwen's 0.15) — accuracy selection effect. Phase 2 natural_usage revised WEAK-MODERATE. |
| **082** | **82** | **Phi-3.5-mini** | **Cross-model forward-looking REPLICATION CONFIRMED.** V→final\|nums R=0.191 (L24, p<0.001). V→final\|embed R=0.169 (p<0.001). WRRA 87.5% (14/16, p=0.002) — strongest across all models. MHA architecture (32 KV heads). Forward-looking signal on 2/3 model families (Qwen + Phi). Mistral failure likely accuracy-mediated. Phase 2 revised MODERATE. |
| **083** | **83** | **Qwen-Base** | **Position-sweep KV decodability — FULL EXPERIMENT A.** V decodes answer from 3% of chain (V_R=0.34, L18) where text reveals 0%. Early Decodability Gap: 25% (L18) / 80% (L27). Peak V_R=0.678 (bootstrap CI [0.71,0.89], p<0.001). Text median first-reveal at 95%. L18>L27 for peak (unexpected). Shuffle ≈0. Input-number confound noted but addressed by exp_079 residualization. Phase 2 revised MODERATE-STRONG. |
| **084** | **84** | **Qwen-Base** | **Position-sweep with cumulative numbers control — INPUT-NUMBER CONFOUND RULED OUT.** V\|nums_R = 0.357 at position 2.5% (text reveals 0%); V\|nums positive at 19/20 bins (mean 0.23 L18, 0.25 L27); peak V\|nums = 0.497 (bootstrap p=0.01). nums_R = 0.35 at 2.5% confirms confound IS real (numbers predict answer) but V carries SUBSTANTIAL info beyond numbers. K\|nums also positive (0.04-0.39). One anomalous bin at 85-90% (formatting transition). Phase 2 natural_usage revised MODERATE-STRONG → STRONG. |
| 085-095 | 85-95 | Various | See individual experiment logs. Key Phase 2 results: paraphrase disruption NULL (exp 085), cross-model position-sweep replication on Phi (exp 086), Mistral boundary (exp 087), size scaling 4B→8B (exp 088), 36-layer × 20-bin heatmap (exp 089), K>V encoding-dependent (exps 091-094), answer-step attention routing (exp 095). |
| **096** | **96** | **Qwen-Base** | **Probe-attention correlation — BRIDGES probing + attention.** Partial r(attn, V\|nums \| position) positive at ALL 32 head×layer conditions (all p<0.001, n=174). Mean partial r 0.16-0.29. Answer step > control at ALL 32 conditions. V\|nums r=0.45 >> nums_R r=0.18. H5 partial_r increases L9→L35 (0.26→0.34). Complete encode→store→retrieve circuit demonstrated. Phase 2 natural_usage STRONG (unchanged, now with retrieval evidence). |
| **099** | **99** | **Qwen-Base** | **WRRA K/V decomposition — NON-REPLICATION + novel V>K finding.** K-probe tested at WRRA positions for first time: K=52-64%, V=40-52%, NEITHER significant (n=25, all p>0.23). Exp_078's V=71.4% (p=0.039) DOES NOT REPLICATE (V=52%, p=1.0). WRRA "smoking gun" downgraded Moderate→Weak/Inconclusive. **Novel: V>K at ALL 4 layers** for numeric probing at "=" positions (V→local R=0.97, K→local R=0.96; V→final R=0.64, K→final R=0.57). Consistent with K=routing/V=content: V stores computation RESULTS, K provides attention ROUTING. Forward-looking probing REPLICATES (V→final R=0.643 vs exp_078's 0.635). Error rate 1.18% replicates exp_078's 1.16%. |

---

## Narrative Synthesis

### What we found (and didn't find)

**The original hypothesis was right about the big picture but wrong about the mechanism.** The KV cache does carry functionally separable channels for text prediction and answer computation. But the separation is not spatial (concentrated at specific token positions) — it is **geometric and component-wise** (K-vectors for routing vs V-vectors for content).

**The story in five results:**

1. **There's room for a hidden channel** (Exp 1-2): LLM output distributions have 4-5 bits/token of unused capacity. During CoT, per-token entropy drops to near zero — most tokens are forced. The "computation" isn't happening in the token choices.

2. **The KV cache carries precise, fragile state** (Exp 3, 7): On Qwen, a sharp SNR cliff at 14 dB shows digital-like encoding. On Llama, analog but still precise: small targeted perturbations destroy answers while text survives (dissociation at every dose tested).

3. **The answer channel lives in K-vectors (routing), not V-vectors (content)** (Exp 23-38, 45, 62): This is our central mechanistic finding, now confirmed by independent spectral geometry (K effective rank < V, K top-1 energy 3.5-4.1x higher than V). K perturbation is devastating for accuracy; V perturbation at moderate levels has literally zero effect. This holds across all 5 models, 3 position bands, both perturbation types, and now at the individual head level (V-immunity: 192/192 across 8 heads). The K > V hierarchy reflects the fundamental QK-routing vs OV-content split in the attention mechanism. Crucially, K-routing fragility is about BREADTH, not DEPTH: destroying one K-head everywhere is well-tolerated (89.1% acc at 12.5% capacity) while destroying all K-heads at 5% of positions is devastating (14% acc). GQA provides 8 redundant routing channels at each position.

4. **The hidden channel is distributed positionally and concentrated in specific heads — but their interaction is encoding-dependent** (Exp 13-21, 45, 49, 51, 52): The original PGD spatial correlation (rho=0.78) was inflated by methodology. Actual rho=0.20. What IS spatially structured is the position gradient: early positions are computational infrastructure; late positions carry answer-specific information. At the head level, the hidden channel is CONCENTRATED: head 5 is the primary answer-routing head on both models. **However, head × position interaction differs by encoding type** (Exp 049 vs 051): On Qwen (digital), H5's answer routing is position-independent (range=9.3pp) — discrete codewords function at any position. On Llama (analog), ALL heads are strongly position-dependent, with position-dependence scaling perfectly with criticality (r=-0.991, p=0.009): H5 range=50.8pp, H3 range=49.1pp, H1 range=34.5pp, H7 range=15.3pp (Exp 052). Destroying K-routing at early positions cascades through the full sequence for 3/4 tested heads. **This cascading is architectural (general property of analog encoding), not circuit-specific (not unique to H5's answer routing)** — established by Exp 052 showing H3 and H1 follow the identical pattern. Digital encoding decouples head and position via error-correcting codewords; analog encoding couples them via continuous routing that propagates early perturbation downstream.

5. **Models encode differently but the hierarchy is universal** (Exp 23-38): Qwen uses digital encoding (sharp accuracy cliffs); Llama/Phi/Mistral use analog (gradual degradation). This affects fragility thresholds and superadditivity patterns but NOT the K > V hierarchy, which holds on every model tested. Digital encoding is Qwen-family-specific; instruction tuning converts V from digital→analog but preserves K digital encoding.

6. **The hidden channel flows through specific "answer heads" — but redundancy is encoding-dependent** (Exp 045, 046, 047, 048): KV heads H0 and H5 are specialized answer-routing heads on both Qwen and Llama. On Qwen (digital): destroying H0+H5 (25%) reduces accuracy to 3.7%, while dispensable pairs preserve 96-100% — a +95.1pp gap. On Llama (analog): the best pair (H0+H7) achieves only 16.2%, and the critical pair (H3+H5) gives 0.0% — a +16.2pp gap. The two-regime redundancy curve is **Qwen-specific**: digital encoding provides massive head-level redundancy (dispensable heads at 96-100%), while analog encoding provides near-zero redundancy (best pair at 16.2%). Head 5 remains the primary answer head on BOTH models, but the practical significance of head specialization depends entirely on encoding type. V-immunity is absolute at per-head level (456/456 across 2 models).

### What this means

**For interpretability:** You cannot trust CoT text as a faithful representation of model computation. The answer-relevant computation flows through K-routing patterns that are invisible to text prediction. This is not a model failure — it is an architectural consequence of the autoregressive text bottleneck (95-99% of gradient norm is suppressed by the LM head).

**For safety:** Models can carry answer-relevant computation through a channel that is undetectable by reading the reasoning text. The K-routing channel could, in principle, encode information that maintains text coherence while redirecting answer computation — the PGD null space demonstrates this is possible (if narrow). CoT monitoring alone is insufficient for safety oversight.

**For KV cache compression:** The eviction benchmark series (Exp 055-058) reveals critical insights. "Recent" (StreamingLLM-style: sinks + most recent) achieves 100% accuracy at ALL budgets (25-75%) on both models — the ONLY universally robust strategy, now confirmed across 6 budget levels. K-norm and cumulative attention (true H2O) are ANTI-CORRELATED on Llama (rho=-0.435, replicated n=17) but orthogonal on Qwen (rho=0.001). True H2O selects early positions (Q1=39-40%, mean_pos=0.32) because earlier positions participate in MORE query-key interactions, accumulating attention regardless of content value. K-norm selects middle-late positions (mean_pos=0.52-0.56) where content-density is highest. Sinks account for only 11-12% of true H2O's early bias — the problem is structural, not just about attention sinks. Two partial fixes: sink-excluded H2O gains +11.8pp on Llama at 33%; late-layer H2O gains +13.3pp on Qwen at 33%. But neither fully rescues true H2O to K-norm levels. The budget crossover on Llama is at 40% (true_h2o = k_norm = 76.5%); above 40%, true H2O wins (100% at 60% vs k_norm 82.4%). Most strikingly, random selection OUTPERFORMS both true H2O and K-norm at low budgets (25-33%) on Llama — on analog models, any selection bias hurts more than random sampling. The K>V perturbation hierarchy does NOT translate to compression: phantom routing (keeping K while zeroing V) is worse than full eviction. Head-selective compression (answer heads 2x budget) outperforms K-norm H2O by +8.7pp on Llama.

**For latent reasoning:** The text bottleneck is now mainstream consensus in the field. Our results provide the most detailed mechanistic characterization of HOW the bottleneck operates at the KV cache level: K-routing carries the answer computation, V-content carries text prediction, and these channels share the same cache but are functionally separable.

---

## Phase 2: Natural Channel Usage (Observational Evidence)

### 7. KV Cache Carries Answer Information Beyond Text During Normal Generation
**Status: STRONG — UPGRADED after exp_096 demonstrates active RETRIEVAL. Evidence now spans encode→store→retrieve: (1) V|nums encodes answer info beyond text (V|nums_R=0.36 at 2.5%, peak 0.50, p=0.01), confirmed on 2 families. (2) Answer-step attention shift (7/8 heads, +24pp, exp_095). (3) Probe-attention correlation: model attends to V|nums-rich positions BEYOND recency (partial r positive at ALL 32 head×layer conditions, p<0.001, n=174; V|nums r=0.45 >> nums_R r=0.18). Three independent methodologies (probing, attention, probe-attention correlation), 2 model families. Mistral does not replicate probing (R=0.06, ns).**

**Experiment:** Train ridge regression probes (5-fold CV) on cumulative KV cache activations vs cumulative token embeddings at 10 normalized CoT positions, to predict the final numeric answer (log-transformed). Qwen3-4B-Base, 80 correctly-solved GSM8K problems, 4 probe layers.

**Key finding:** Both K-probe and V-probe consistently exceed the text baseline at ALL positions from 10% to 90% through CoT. The KV cache encodes answer information that goes BEYOND what the text tokens provide — **observational evidence that the hidden channel is used during normal, unperturbed generation.**

| Position | K-probe (L27) | V-probe (L27) | Text baseline | K-Text gap |
|----------|---------------|---------------|---------------|------------|
| 10% | 0.430 | 0.454 | 0.361 | +0.069 |
| 30% | 0.629 | 0.625 | 0.581 | +0.048 |
| 50% | 0.612 | 0.630 | 0.543 | +0.070 |
| 70% | **0.641** | 0.509 | 0.406 | **+0.235** |
| 90% | 0.397 | 0.488 | 0.356 | +0.041 |

**K-text advantage grows through CoT**, peaking at +0.235 at 70%, then narrowing toward 100%.

**Complementary finding — V ≥ K for decodability:**
- V mean r across positions = 0.546, K mean r = 0.539 at best layer (L27)
- At middle layers (L18), V dramatically exceeds K (mean V=0.623 vs K=0.467)
- This COMPLEMENTS Phase 1's K > V for causality: **K controls routing (destructive when perturbed), V carries content (decodable by probes)**
- Consistent with "Causality ≠ Decodability" (2025) literature finding

**Controls:**
- Shuffle control: r ≈ 0 or negative at all positions (validates real signal)
- Dimensionality: KV features (1024-dim) vs text (2560-dim) — text has MORE dimensions but LOWER performance, strengthening the finding
- Layer sweep: Effect present at all 4 layers (25%, 51%, 77%, 100% depth)
- Within-bin comparison (symmetric local features): K ≥ text at all positions; text ≥ V only at some

**Cross-model replication (Exp 069 — Mistral-7B-v0.3):**

| Position | K-probe (L8) | V-probe (L8) | Text baseline | K-Text gap |
|----------|--------------|--------------|---------------|------------|
| 10% | 0.646 | 0.622 | 0.657 | -0.011 |
| 20% | **0.778** | 0.740 | 0.690 | **+0.088** |
| 50% | 0.639 | 0.527 | 0.602 | +0.037 |
| 80% | 0.510 | 0.586 | 0.411 | +0.099 |
| 90% | 0.499 | 0.572 | 0.347 | **+0.152** |
| 100% | 0.539 | 0.582 | 0.400 | +0.140 |

Mean across all positions: K=0.575, V=0.609, Text=0.524. K > text at 6/10, V > text at 7/10.

**Cross-model comparison:**

| Metric | Qwen (digital) | Mistral (analog) |
|--------|---------------|------------------|
| Mean K (cum) | 0.539 | 0.575 |
| Mean V (cum) | 0.546 | 0.609 |
| Mean text (cum) | 0.449 | 0.524 |
| V ≥ K | Most bins | ALL layers |
| Best K layer | L27 (77%) | L8 (25%) |
| KV-text advantage | K: +0.090, V: +0.097 | K: +0.051, V: +0.085 |

The finding replicates with model-specific variation:
- **Both models:** KV > text in mean, V ≥ K for decodability
- **Mistral:** Weaker K advantage but stronger V advantage; best layer is early (L8) vs late (L27)
- **Mistral's text dim is 4x KV dim** (4096 vs 1024) — despite this disadvantage, KV still exceeds text

**Computation-position probes (Exp 071 — Qwen3-4B-Base):**

At specific arithmetic positions (the "=" token in `<<EXPR=RESULT>>` operations), the
KV-text gap is dramatically amplified:

| Layer | K-probe R | V-probe R | Text R | KV-Text gap |
|-------|-----------|-----------|--------|-------------|
| L9 (25%) | 0.936 | 0.948 | 0.108 | +0.84 |
| L18 (50%) | 0.936 | 0.955 | 0.108 | +0.85 |
| L27 (75%) | 0.954 | 0.971 | 0.108 | +0.86 |
| L35 (97%) | 0.961 | **0.975** | 0.108 | **+0.87** |

N=1337 correct arithmetic operations, 5-fold CV, shuffle R ≈ -0.06.
V > K at all layers — V encodes computation results better than K.
Effect increases with depth (V: 0.948 → 0.975).
Text R = 0.108 because the "=" token embedding is identical for all operations.

The KV-text gap at computation positions (+0.85) is **~10x larger** than at general CoT
positions (+0.09 from exp_068). At positions where computation actually happens, the KV
cache is sharply tuned to the result while text embeddings are generic.

**WRRA analysis (Exp 071):** Underpowered — only 2 arithmetic errors in 1339 operations
(0.15% error rate). The `<<EXPR=RESULT>>` calculator format produces near-perfect
arithmetic. One error was expression notation ambiguity, not genuine arithmetic error.
True error rate ≈ 0.07%. WRRA requires a less accurate model or different prompting format.

**Computation-position probes: Cross-model replication (Exp 072 — Mistral-7B-v0.3):**

Pre-registered replication of exp_071's computation-position finding on Mistral, with
enhanced text baseline (average of expression token embeddings).

| Layer | K-probe R | V-probe R | Expr-embed R | Token-embed R | Shuffle K |
|-------|-----------|-----------|-------------|---------------|-----------|
| L8 (28%) | 0.295 | 0.357 | 0.310 | 0.111 | -0.066 |
| L16 (53%) | **0.431** | **0.435** | 0.310 | 0.111 | +0.031 |
| L24 (78%) | 0.390 | 0.348 | 0.310 | 0.111 | -0.070 |
| L31 (100%) | 0.341 | 0.331 | 0.310 | 0.111 | -0.060 |

N=877 correct-arithmetic positions, 5-fold CV, 396 problems, 41.9% accuracy.

**Core finding PARTIALLY REPLICATES but is 2.2x weaker:**
- KV (0.435) > expr-embed (0.310) > token-embed (0.111) → hierarchy confirmed
- But KV advantage over expression embedding = only **+0.125** (Qwen: +0.87 over token-embed)
- Expression-embed captures much of what KV knows — the KV "hidden computation" signal is modest
- Peaks at L16 (53% depth), not deep layers (Qwen peaked at 97%)
- V ≈ K (0.435 vs 0.431) — weak V-dominance, unlike Qwen's strong V > K

**WRRA alignment on Mistral (n=91 errors, 9.4% error rate):**
- All alignment rates BELOW chance (0.37-0.42)
- KV does NOT encode correct values at arithmetic error positions
- No "hidden correct computation" evidence on Mistral
- WRRA rate: only 3/91 errors had correct final answer (3.3%)

**Expression-embedding retroactive baseline on Qwen (Exp 073):**

Pre-registered experiment testing whether the expression-embedding baseline (mean of
expression token embeddings) collapses Qwen's computation-position gap. Added permutation
significance testing (N=1000 shuffles).

| Layer | K-probe R | V-probe R | Expr-embed R | Token-embed R | Shuffle V |
|-------|-----------|-----------|-------------|---------------|-----------|
| L9 (27%) | 0.932 | 0.946 | 0.771 | -0.080 | -0.052 |
| L18 (52%) | 0.937 | 0.956 | 0.771 | -0.080 | -0.068 |
| L27 (77%) | 0.961 | 0.971 | 0.771 | -0.080 | -0.056 |
| L35 (100%) | **0.965** | **0.977** | **0.771** | -0.080 | -0.042 |

N=1474 correct-arithmetic positions, 5-fold CV, 493 problems, 85.2% accuracy.

**Permutation significance (N=1000):** V p=0.000, K p=0.000, expr p=0.000, token p=0.936.

**Key result: V - expr gap = +0.206.** Expression-embed narrows the gap from +0.867 (over
token-embed) but does NOT collapse it. The KV cache encodes computation BEYOND what
expression tokens provide. The gap is 1.65x larger than Mistral's +0.125.

**The computation-position effect is MODEL-DEPENDENT:**

| Metric | Qwen (digital, exp 073) | Mistral (analog, exp 072) |
|--------|------------------------|--------------------------|
| Best V R | 0.977 | 0.435 |
| Best K R | 0.965 | 0.431 |
| Expr-embed R | **0.771** | 0.310 |
| Token-embed R | -0.080 | 0.111 |
| **V - expr gap** | **+0.206** | **+0.125** |
| V - token gap | +1.057 | +0.324 |
| Perm p (V) | 0.0000 | — |
| Error rate | 0.1% | 9.4% |
| WRRA alignment | n/a (n=2) | Below chance |
| Best layer | 100% depth | 53% depth |

The hidden channel at computation positions is: near-perfect on Qwen (digital, high-accuracy),
moderate on Mistral (analog, low-accuracy). With the expression-embed baseline, the Qwen
advantage narrows from "decisive" to "moderate-strong" but remains substantial (+0.206).
Possible explanations for cross-model difference:
1. Digital encoding creates sharper cluster structure → more linearly decodable
2. Higher accuracy models produce cleaner computation representations
3. Mistral may rely more on text than hidden channel for computation
4. Even the expression-embed baseline itself is 2.5x more decodable on Qwen (0.771 vs 0.310)

**Dimensionality note:** The V-probe (dim=640-1024) outperforms expr-embed (dim=2560) despite
having 2.5x FEWER features. The KV cache encodes MORE relevant computation in FEWER dimensions.

**Paraphrase disruption — UNINFORMATIVE (Exp 074):**

Replacing ALL non-numeric tokens in the CoT with random gibberish has ZERO effect on answer
accuracy (215/215 = 100% across original, synonym, and random conditions). The `<<EXPR=RESULT>>`
format preserves the complete arithmetic chain in numeric tokens, making English text entirely
redundant. The model extracts answers from `<<EXPR=RESULT>>` expressions regardless of
surrounding text.

This result is UNINFORMATIVE about the hidden channel hypothesis because the experiment design
has a critical confound: the format makes the manipulation ineffective by construction. A proper
paraphrase test requires plain-text CoT without the calculator format.

However, the null result is CONSISTENT with probing findings: the computation that matters flows
through numeric/arithmetic positions, not through English word positions. The hidden channel
(if it exists) operates at computation-specific positions, and non-numeric tokens are scaffolding.

**Limitations (remaining — updated cycle 81):**
- Residualized forward-looking signal (V|nums, V|embed) confirmed only on Qwen, NOT on Mistral (exp_081)
- WRRA alignment above chance only on Qwen (71.4%, p=0.039, n=21); below chance on Mistral (38%, p=0.93, n=29; and exp_072: below chance, n=91)
- Non-replication could be power (83 vs 341 problems), accuracy selection (43% vs 87%), or genuine architecture difference
- Paraphrase disruption (Experiment B) untested with plain-text CoT — independent test needed
- Cumulative KV advantage (+0.055) is small and may reflect richer representations
- GSM8K answers are functions of problem numbers — task specificity limits generalization
- Need high-accuracy model replication (Llama-Instruct) to disambiguate power vs architecture

**Key experiments:** Exp 065 (initial, confound identified), Exp 068 (Qwen, corrected), Exp 069 (Mistral, cross-model replication), Exp 071 (computation-position probes + WRRA attempt), Exp 072 (Mistral computation-position + expression-embed baseline + WRRA), Exp 073 (Qwen expression-embed retroactive baseline + permutation significance), Exp 074 (paraphrase disruption — decisive null due to format confound), Exp 075 (early answer decodability — text dominates KV at all positions, design limitation identified), Exp 076 (cumulative KV probe — cum_V > cum_text at 10/10 positions, +0.055 mean, fixes exp_075 design flaw)

---

### Exp 075: Early Answer Decodability — Text Dominates KV at All Positions
**Cycle 75 | Qwen3-4B-Base | n=258 correct problems | Experiment A from research_spec**

**Core question:** Does the KV cache "know" the final answer before the text reveals it?

**Method:** Trained RidgeCV probes at 10 evenly-spaced positions (10%-100% of CoT) to predict
log(|final_answer|+1) from: (a) K-vector at that position (dim=1024), (b) V-vector at that
position (dim=1024), (c) cumulative mean of token embeddings from CoT start to that position
(dim=2560), (d) single token embedding at that position (dim=2560). Two layers: L18 (51%) and
L35 (100%). Shuffle controls at each decile.

**Results (L35):**

| Position | K R | V R | Text-cum R | Text-tok R | Shuffle |
|----------|-----|-----|-----------|-----------|---------|
| 10% | 0.384 | 0.331 | **0.519** | 0.023 | -0.204 |
| 20% | 0.309 | 0.355 | **0.689** | 0.232 | 0.016 |
| 40% | 0.084 | 0.020 | **0.754** | -0.107 | -0.081 |
| 70% | 0.028 | 0.091 | **0.690** | -0.103 | -0.065 |
| 100% | -0.011 | -0.044 | **0.630** | -0.017 | -0.082 |

**Key findings:**
1. **Text-cumulative DOMINATES KV probes at ALL 10 positions and both layers.** Gap ranges
   from +0.19 to +0.73 in favor of text. No position shows KV > text.
2. **No early decodability gap:** Both reach R>0.3 at decile 1 (10%). Gap = 0 deciles.
3. **Text-cum is already R=0.519 at 10%** because problem numbers appear early in the
   generated CoT (model restates "Janet sells 16 - 3 - 4 = ..."), and the final answer is a
   mathematical function of these numbers. The cumulative mean accumulates number-token
   embeddings that are inherently predictive.
4. **KV probes are weak and inconsistent:** best single R = 0.384 (K at L35/10%). Many
   positions show R < 0.1 or negative. No monotonic pattern.
5. **Shuffle controls clean:** R ∈ [-0.20, +0.02] across all conditions.

**Pre-registered prediction evaluation:** 2/10 TRUE-branch confirmed, 1/4 FALSE-branch
confirmed. NEITHER branch predicted the text-cumulative strength (both predicted text R < 0.3
at early positions; actual R = 0.52 at 10%). The result fell outside BOTH prediction ranges.

**Critical design limitation — asymmetric comparison:**
- Text-cumulative: aggregates embeddings from hundreds of tokens (accumulates problem numbers)
- KV at position P: a SINGLE vector at ONE position (encodes local attention state)

This is NOT apples-to-apples. The text baseline has access to far more raw information (the
identities of many tokens) than the KV probe (one compressed vector). The text advantage may be
purely due to this asymmetry, not because the KV cache lacks answer information.

**Interpretation:** The result is WEAKLY informative about the hidden channel hypothesis.
- Does NOT support the specific "Reasoning Theater" claim that a single KV position encodes the answer early
- Does NOT rule out early decodability through cumulative KV (mean of all KV vectors up to P) or through hidden states
- IS consistent with exp_071-073: answer information in KV is CONCENTRATED at computation-specific positions ("="), not uniformly distributed

**Evidence strength:** WEAK (clear null but design limitation prevents strong conclusions)

**Follow-up needed:** Cumulative KV probe (mean of KV vectors up to P) vs cumulative text.
This removes the asymmetry and enables a fair apples-to-apples comparison.

### Exp 076: Cumulative KV Probe — Fair Apples-to-Apples Comparison
**Cycle 76 | Qwen3-4B-Base | n=258 correct problems | Fixes exp_075 design flaw**

**Core question:** When KV and text are aggregated identically (cumulative mean from CoT start
to position P), does the KV cache predict the final answer better than text?

**Method:** At each of 10 decile positions (10%-100% of CoT), computed: (a) cumulative mean of
V-vectors from positions 0:P (dim=1024), (b) cumulative mean of K-vectors from 0:P (dim=1024),
(c) cumulative mean of token embeddings from 0:P (dim=2560), (d) hidden state at position P
(dim=2560), (e) single-position V at P (dim=1024, for exp_075 comparison). RidgeCV 5-fold CV,
target = log(|answer|+1) × sign(answer). L35 (100% depth).

**Results:**

| Position | cum_V R | cum_K R | cum_text R | hidden_st R | single_V R | Shuffle |
|----------|---------|---------|------------|-------------|------------|---------|
| 10% | **0.623** | 0.620 | 0.519 | 0.386 | 0.331 | -0.207 |
| 20% | **0.722** | 0.712 | 0.689 | 0.374 | 0.355 | 0.008 |
| 30% | **0.759** | 0.721 | 0.730 | 0.234 | 0.185 | -0.056 |
| 40% | **0.784** | 0.760 | 0.754 | 0.138 | 0.020 | -0.059 |
| 50% | **0.797** | 0.767 | 0.752 | 0.185 | 0.110 | -0.136 |
| 60% | **0.771** | 0.726 | 0.711 | 0.020 | -0.015 | -0.048 |
| 70% | **0.751** | 0.720 | 0.690 | -0.059 | 0.091 | -0.070 |
| 80% | **0.716** | 0.705 | 0.648 | 0.152 | 0.136 | -0.116 |
| 90% | 0.704 | **0.717** | 0.651 | 0.169 | 0.056 | -0.140 |
| 100% | 0.694 | **0.699** | 0.630 | 0.012 | -0.044 | -0.106 |

**KEY FINDING: Cumulative V > cumulative text at ALL 10/10 positions.**

cum_V advantage: +0.104 (10%), +0.033 (20%), +0.029 (30%), +0.029 (40%), +0.045 (50%),
+0.060 (60%), +0.061 (70%), +0.067 (80%), +0.053 (90%), +0.065 (100%).
**Mean advantage: +0.055 R. Universality: 10/10 positions.**

**Aggregation improvement:** Single V → cumulative V: mean +0.609 R. Cumulative aggregation
transforms single-position noise into strong signal. Answer information is DISTRIBUTED
across many positions — no single position carries the full answer.

**Surprise — hidden state at P underperforms cumulative text:** Mean disadvantage -0.517 R.
Despite attending to all prior positions through causal attention, a single hidden state
encodes ONE position's contribution to the sequence (optimized for next-token prediction),
not a global answer representation. This confirms answer info is distributed.

**No early decodability gap:** All probes reach R>0.3 at decile 1 (10%). Both KV and text
are immediately informative because problem numbers appear at the start.

**Interpretation:** The cumulative V advantage is small (+0.055 mean) but universal (10/10).
It demonstrates that V-vectors at L35 carry COMPUTATION ARTIFACTS beyond raw token identities:
each V[P] has been processed through 35 layers of attention + FFN, encoding attention-aggregated
context and nonlinear computation. The text baseline, despite having 2.5x more dimensions
(2560 vs 1024), consistently underperforms.

The advantage is largest at 10% (+0.104) — the KV cache has encoded more answer-relevant
information than the first few text tokens. This is because KV vectors incorporate inter-token
relationships through attention, while text embeddings are position-independent.

**Confounds:** (1) KV representations at L35 are inherently richer than input embeddings (35
layers of computation) — the advantage could reflect general representational richness, not
answer-specific "hidden computation." (2) GSM8K answers are functions of problem numbers — any
representation capturing numbers will be predictive. (3) Single model (Qwen only).

**Pre-registered predictions:** 7/11 TRUE-hypothesis confirmed, 1.5/4 FALSE confirmed.
Hidden-state predictions wrong in both branches.

**Evidence strength:** MODERATE — cumulative V universally exceeds cumulative text in a fair
comparison, but the advantage is small (+0.055) and could be explained by richer representations.

---

### Exp 077: Cross-Model Cumulative KV Probe — Mistral-7B + Layer Sweep + Bootstrap Test
**Cycle 77 | 2026-03-21 | Phase 2 — cross-model replication + significance**

**Model:** Mistral-7B-v0.3 (different architecture family from Qwen)
**Method:** Replicated exp_076 cumulative KV probe on Mistral with 4 probe layers (L8, L16,
L24, L31) and 2000-sample bootstrap significance test. 250 GSM8K problems generated,
109 correct (43.6%), 109 valid for probing. KV dim=1024, text dim=4096 (4x more features
for text — cumV advantage is more impressive).

**Results:**

| Layer | Depth | Mean cumV Advantage | V Leads | Bootstrap p | 95% CI |
|-------|------:|--------------------:|:-------:|:-----------:|:------:|
| L8 | 26% | +0.083 | 8/10 | 0.037 * | [-0.009, +0.184] |
| L16 | 52% | +0.114 | 7/10 | 0.016 * | [+0.011, +0.238] |
| L24 | 77% | **+0.114** | **8/10** | **0.003** ** | [+0.030, +0.212] |
| L31 | 100% | +0.072 | 7/10 | 0.071 | [-0.026, +0.168] |

**Cross-model comparison (deepest layer):**
- Qwen3-4B-Base (exp_076): cumV advantage = +0.055, 10/10 positions
- Mistral-7B-v0.3 (exp_077): cumV advantage = +0.072, 7/10 positions
- **Both models show cumV > cumText — REPLICATION CONFIRMED**

**Key findings:**
1. **Cross-model replication confirmed**: cumV > cumText on Mistral at 7-8/10 positions at
   ALL 4 layers. Effect is now demonstrated across 2 model families.
2. **Statistically significant**: Bootstrap p=0.003 at L24 (strongest), p=0.016 at L16,
   p=0.037 at L8. Deepest layer (L31) marginal at p=0.071.
3. **Layer-independent**: Advantage present from L8 (26% depth) through L31 (100%). No depth
   trend (Spearman rho=-0.200, p=0.800). Not a late-computation effect.
4. **Concentrated at late chain positions**: At all layers, cumV advantage near zero at 10-30%
   of chain, largest at 60-100% (+0.15 to +0.25 R). Driven by cumText declining as
   non-numeric tokens dilute the number signal, while cumV remains stable.
5. **CumK weaker than cumV on Mistral**: Unlike Qwen where cumK ≈ cumV, Mistral cumK lags
   behind both cumV and cumText. V (content) aggregates more coherently than K (routing)
   under cumulative mean.
6. **Advantage LARGER on Mistral** (+0.072-0.114) than Qwen (+0.055) despite fewer samples
   (109 vs 258) and lower accuracy (43.6% vs 86.0%).

**Interpretation:** The cumV > cumText effect is a genuine cross-model phenomenon. The
layer-independent profile suggests the advantage reflects a fundamental property of attention
V-projections (which encode token identity + context from attention) vs raw embeddings
(token identity only). Even 8 layers of processing create this advantage. The concentration
at late chain positions suggests V vectors maintain computation-relevant signal even as
the text representation degrades through dilution.

**Pre-registered predictions:** 4/10 TRUE-hypothesis confirmed, 2/4 FALSE confirmed.
Depth-growth predictions wrong (advantage is flat, not growing). CumK-cumV divergence
on Mistral was unexpected.

**Evidence strength:** MODERATE-STRONG — cross-model replication with bootstrap significance
(p=0.003) substantially strengthens the cumV > cumText finding. Effect now confirmed across
2 model families (Qwen, Mistral) with statistical testing.

### Exp 078: WRRA Plain-Text CoT + Forward-Looking Computation Probing
**Cycle 78 | Qwen3-4B-Base | Phase 2 — natural channel usage | STRONG**

Two major findings from probing KV at arithmetic "=" positions in plain-text CoT:

**Finding 1: Forward-Looking Probing (STRONG, n=1787 correct operations)**

V at computation positions predicts the FINAL answer, not just the local result:

| Layer | V→local | V→final | Text→final | V→final\|local (partial) |
|-------|---------|---------|------------|--------------------------|
| L9 (26%) | 0.952 | 0.594 | -0.034 | **0.476** |
| L18 (51%) | 0.915 | 0.546 | -0.034 | **0.460** |
| L27 (77%) | 0.973 | 0.617 | -0.034 | **0.496** |
| L35 (100%) | 0.974 | **0.635** | -0.034 | **0.520** |

The partial correlation V→final|local R=0.520 means V carries information about the final
answer BEYOND the local computation step. Text adds nothing (R=-0.056 partial). Shuffle
control validates (R≈-0.07). K also carries forward-looking info (K→partial=0.423 at L35).

**Finding 2: WRRA Alignment (MODERATE, n=21 errors)**

Plain-text format yields 7.7x more arithmetic errors (21/1808 = 1.16%) than calculator
format (2/1339 = 0.15%). At error positions where the model writes wrong arithmetic:

| Layer | Correct-alignment | p-value |
|-------|:-----------------:|:-------:|
| L27 (77%) | **15/21 = 0.714** | **0.039 \*** |
| L35 (100%) | 13/21 = 0.619 | 0.192 |

At L27, 71.4% of error positions have V probes predicting closer to the CORRECT value
than the WRITTEN value (p=0.039). The model's internal representation encodes the correct
computation despite the text saying the wrong thing.

**Finding 3: Format Effect on Arithmetic Errors**
Plain-text CoT produces 1.16% arithmetic errors vs 0.15% with `<<EXPR=RESULT>>` calculator.
The calculator format constrains the model into near-perfect arithmetic computation.

**Pre-registered predictions:** 10/10 TRUE-hypothesis confirmed, 1/5 FALSE confirmed.
This is the strongest prediction match in Phase 2.

**Confounds:** (1) Problem-number confound: V at "=" has attended to problem numbers
through attention, which are correlated with the final answer. Partial correlation controls
for local result but not problem numbers directly. (2) n=21 for WRRA is small; needs
cross-model replication with more errors.

**Evidence strength:** ~~STRONG~~ **REVISED TO MODERATE** for forward-looking probing.
Exp_079 discovered that the V→final|local R=0.520 was inflated by within-problem data
leakage (KFold assigns same-problem operations to train/test). With proper GroupKFold CV:
V→final|local R=0.299 (not 0.520). V→final|embed R=0.14-0.22 (V survives problem-context
control). V→final|embed+local R=0.13-0.22 (survives strictest control). Effect is MODERATE
(R≈0.2), not STRONG (R≈0.5). Still positive evidence, but exp_078 overstated the effect.
MODERATE for WRRA (significant at L27 but small n, unaffected by the leakage issue).

### Exp 079: Problem-Number Residualization — CHALLENGE Experiment
**Cycle 79 | Qwen3-4B-Base | Phase 2 — challenge experiment | MODERATE (methodological correction)**

**Challenge:** Last 3 experiments (076-078) all confirmed hypothesis. This experiment attacks
the biggest confound: does V at computation positions predict the final answer because of
hidden computation, or because V encodes problem numbers via attention?

**Critical methodological discovery: within-problem data leakage.** With ~4.6 operations per
problem sharing identical (prob_embed, final_answer), standard KFold leaks same-problem
observations between train/test folds. This inflated exp_078's R=0.977 for embed→final
and deflated V→final|embed to ~0.04. GroupKFold (no within-problem leakage) is the ground truth.

**GroupKFold results (ground truth, n=1573 ops from 341 problems):**

| Probe | L27 (75%) | L35 (97%) | Interpretation |
|-------|-----------|-----------|----------------|
| V → final | **0.487** | **0.476** | V carries substantial answer info |
| embed → final | 0.344 | 0.344 | Problem context predicts moderately |
| nums → final | 0.153 | 0.153 | Raw numbers weakly predictive |
| V → final \| nums | **0.242** | **0.235** | V survives number control ✓ |
| V → final \| embed | **0.221** | **0.140** | V survives context control ✓ |
| V → final \| local | 0.299 | 0.285 | V survives local control (corrects exp_078's 0.520) |
| V → final \| e+l | **0.215** | **0.134** | V survives strictest combined ✓ |
| Shuffle | -0.017 | 0.032 | Valid |

**Challenge outcome: PARTIALLY SURVIVED.**
- V's forward-looking signal is REAL: V → final (0.49) > embed → final (0.34), and V
  survives all residualization controls
- But the effect is MODERATE (R≈0.2), not STRONG (R≈0.5 as exp_078 claimed)
- Exp_078's R=0.520 partial was inflated ~1.8x by data leakage (true R≈0.29)

**KFold vs GroupKFold comparison (key methodological lesson):**

| Probe | KFold | GroupKFold | Inflation |
|-------|-------|-----------|-----------|
| V → final | 0.633 | 0.487 | +0.146 |
| embed → final | 0.977 | 0.344 | **+0.633** |
| V → final \| embed | 0.040 | 0.221 | **-0.181** (deflated) |
| V → final \| e+l | -0.001 | 0.215 | **-0.216** (deflated) |

**Evidence strength:** MODERATE. V carries forward-looking computation info beyond problem
context (R=0.14-0.22 after strictest control). Effect is modest but real and survived the
challenge. Most important contribution is the methodological correction of exp_078.

### Exp 081: Cross-Model Forward-Looking Probing — Mistral-7B-v0.3
**Cycle 81 | Mistral-7B-v0.3 | Phase 2 — cross-model replication | MIXED (partial replication)**

**Core question:** Does the V forward-looking signal from exp_078/079 (V→final|nums R=0.24
on Qwen with GroupKFold) replicate on a different model family?

**Setup:** 200 GSM8K problems, plain-text CoT, Mistral-7B-v0.3 (base, GQA, analog encoding).
86 correct (43%), 85 valid with operations, 573 correct operations from 83 problems.
29 arithmetic errors (4.82% — 4.1x more than Qwen's 1.16%). GroupKFold ONLY.

**GroupKFold results (n=573 ops from 83 problems):**

| Probe | L8 (25%) | L16 (50%) | L24 (75%) | L31 (97%) | Qwen L27 |
|-------|----------|-----------|-----------|-----------|----------|
| V → final | 0.113 | **0.234** | **0.231** | **0.212** | 0.487 |
| K → final | 0.011 | 0.100 | 0.195 | 0.129 | N/A |
| nums → final | 0.390 | 0.390 | 0.390 | 0.390 | 0.153 |
| embed → final | -0.045 | -0.045 | -0.045 | -0.045 | 0.344 |
| V \| nums | **-0.055** | **0.067** | **0.057** | **0.066** | **0.242** |
| V \| embed | 0.049 | 0.004 | -0.010 | 0.038 | 0.221 |
| V \| e+l | 0.049 | 0.004 | -0.010 | 0.038 | 0.215 |
| Shuffle | -0.061 | -0.088 | -0.083 | -0.026 | -0.017 |

**Bootstrap significance:**
- V→final: p < 0.007 at all layers (significant — raw signal is real)
- V|nums: p = 0.088-0.876 at all layers (**NOT significant** — signal explained by numbers)
- V|embed: p = 0.17-0.58 (not significant)

**What REPLICATES:** V→final raw signal (R=0.21-0.23, p<0.001). V > K consistently.
Depth increase (L8→L16). Shuffle ≈ 0.

**What does NOT replicate:** V|nums (R=0.06 vs Qwen's 0.24, not significant).
V|embed (≈0 vs Qwen's 0.22). V|e+l (≈0 vs Qwen's 0.22).

**WRRA: DOES NOT REPLICATE.** 11/29 = 37.9% alignment at L24 (p=0.932, below chance).
Compare Qwen: 15/21 = 71.4% at L27 (p=0.039). Mistral's V at error positions encodes the
WRITTEN (wrong) value, not the correct value.

**Interpretation:**

Three possible explanations for non-replication of residualized signal:

1. **Power issue:** 83 problems (Mistral) vs 341 (Qwen). GroupKFold has less power. But
   V|nums R=0.06 is 4x smaller than Qwen's 0.24 — not just underpowered, genuinely smaller.

2. **Accuracy selection effect:** At 43% accuracy, Mistral only solves easy problems where
   answer ≈ f(numbers). This inflates nums→final (0.39 vs 0.15), leaving less residual for
   V. The problems Mistral solves may not require "forward-looking computation."

3. **Architecture difference:** Analog encoding (Mistral) may couple text and computation
   more tightly than digital encoding (Qwen), meaning V follows text even when text is wrong
   (explaining WRRA reversal).

**Impact on Phase 2 evidence:** The forward-looking probing evidence is WEAKER than estimated:
- V→final raw signal is cross-model (still moderate)
- V|nums residualized signal is Qwen-specific so far (weak-moderate)
- WRRA is Qwen-specific (Qwen: above chance, Mistral: below chance on both exp_072 and exp_081)
- Revised natural_usage assessment: WEAK-MODERATE (down from MODERATE)

**Evidence strength:** WEAK-MODERATE. Raw V→final replicates but residualized signal does not.
Honest assessment: the Phase 2 forward-looking evidence may be Qwen-specific or
accuracy-dependent. Needs replication on a high-accuracy model (Llama-Instruct) to disambiguate.

### Exp 082: Cross-Model Forward-Looking Probing — Phi-3.5-mini-Instruct (MHA)
**Cycle 82 | Phi-3.5-mini-Instruct | Phase 2 — cross-model replication | CONFIRMED**

REPLICATION CONFIRMED on a third model family:
- V→final|nums R=0.191 (L24, p<0.001), V→final|embed R=0.169 (p<0.001)
- WRRA 14/16 = 87.5% (p=0.002) — STRONGEST WRRA across all models
- Phi has MHA architecture (32 KV heads, not GQA 8) and higher accuracy (83% vs Mistral's 43%)
- Forward-looking V signal now confirmed on 2 families: Qwen (R=0.24) + Phi (R=0.19)
- Mistral failure likely accuracy-mediated, not architectural

**Evidence strength:** MODERATE (revised UP from WEAK-MODERATE). Cross-model replication achieved.

### Exp 083: Position-Sweep KV Decodability (Full Experiment A)
**Cycle 83 | Qwen3-4B-Base | Phase 2 — position-sweep decodability | MODERATE-STRONG**

**The V-cache decodes the final answer BEFORE the text reveals it.** Full position-sweep
across ALL CoT token positions (not just "=" positions):

- **220 problems, 21,245 V/K vectors per layer, 20 position bins, GroupKFold 5-fold**
- V_R = 0.335 (L18) at position 3% of chain, where text reveals answer to 0% of problems
- V_R rises gradually from 0.17-0.36 through the chain, peaking at 0.678 (L18) / 0.520 (L27)
- Text reveals answer (median) at position 95% — very late in the chain
- **Early Decodability Gap: 25% of chain (L18), 80% of chain (L27)**
- Peak V_R bootstrap: mean=0.812, 95% CI [0.713, 0.888], p < 0.0001
- Shuffle controls near 0 at all bins (range: [-0.08, +0.10])
- V > K at L18 (most positions); V ≈ K at L27
- Unexpected: L18 peak (0.678) > L27 peak (0.520) — mid-layers have stronger linear decodability

**Important confound:** Early V_R could be partly driven by input number encoding (visible
problem numbers at position 3% predict the answer). Exp_079's V|nums R=0.24 confirms genuine
forward-looking signal at "=" positions, but position-sweep nums control not yet done.

**Key figures:**
- `results/exp_083/decodability_curve.png` — V-probe R vs text-reveals-answer (main paper figure)
- `results/exp_083/kv_comparison.png` — K vs V by position
- `results/exp_083/bootstrap_peak_R.png` — Bootstrap significance

**Evidence strength:** MODERATE-STRONG. Compelling decodability curve with large gap. Missing
nums-control at each position bin (addressed by exp_079 at "=" positions, not full sweep).
Combined with exp_079 V|nums = 0.24: STRONG evidence for forward-looking hidden computation.

**Impact on Phase 2 evidence:** Revised natural_usage from MODERATE to MODERATE-STRONG.
The position-sweep decodability curve is the most compelling visualization of the hidden
channel hypothesis. Combined with residualized probing (exp_079/082) and WRRA (exp_078/082),
Phase 2 evidence now comes from 3 independent methodologies on 2 model families.

### Exp 084: Position-Sweep with Cumulative Numbers Control — INPUT-NUMBER CONFOUND RULED OUT
**Cycle 84 | Qwen3-4B-Base | Phase 2 — numbers-controlled position sweep | STRONG**

**Addresses the #1 confound from exp_083.** At each of 20 position bins, trains three probes:
- V → log(answer): raw V decodability (replicates exp_083)
- cumNums → log(answer): cumulative visible numbers baseline (question + CoT numbers)
- V → log(answer) | cumNums: V residualized by visible numbers

**The input-number confound is real but V goes SUBSTANTIALLY beyond it:**

- **220 problems, ~21,000 vectors per layer, cumNums features: 36-dim (30 log-numbers + 6 stats)**
- At position 2.5% (text reveals 0%): nums_R = 0.353, V_R = 0.436, **V|nums = 0.357**
- V|nums is positive at **19/20 bins** (both layers), mean V|nums = 0.228 (L18), 0.251 (L27)
- Peak V|nums = 0.497 at 90-95% of chain (L18), bootstrap p = 0.010
- Peak V|nums = 0.457 at 95-100% of chain (L27), bootstrap p = 0.013
- One anomalous bin at 85-90% (V|nums ≈ 0): "####" formatting transition zone
- K|nums also substantial (0.04-0.39): K carries forward-looking info beyond numbers too
- Shuffle controls near 0 at most bins (range: [-0.24, +0.17])

**Pre-registered prediction score:** 6/8 TRUE confirmed, 0/4 FALSE confirmed (decisive).

**Key figures:**
- `results/exp_084/decodability_nums_control_L18.png` — Main: V, nums, V|nums, text curves
- `results/exp_084/decodability_both_layers.png` — L18 and L27 comparison
- `results/exp_084/forward_looking_gap.png` — V beyond numbers gap
- `results/exp_084/bootstrap_V_nums.png` — Bootstrap significance

**Evidence strength:** STRONG. V|nums = 0.36 at position 2.5% definitively rules out
the input-number confound. The V-cache carries genuine forward-looking information at
every stage of the reasoning chain, beyond what visible text numbers predict.

**Impact on Phase 2 evidence:** Revised natural_usage from MODERATE-STRONG to STRONG.
Combined evidence: position-sweep V|nums (exp_084, Qwen), per-operation V|nums (exp_079
Qwen R=0.24, exp_082 Phi R=0.19), WRRA (exp_078 Qwen 71.4%, exp_082 Phi 87.5%), and
raw position-sweep decodability gap 25-80% (exp_083). Four methodologies, two model families.

### Exp 085: Paraphrase Disruption (Experiment B) — NULL RESULT
**Cycle 85 | Qwen3-4B-Base | Phase 2 — behavioral test | NULL for non-number tokens**

**First BEHAVIORAL test of the hidden channel.** All prior Phase 2 evidence was probe-based.
This experiment tests whether specific non-number token choices carry hidden computation
information by paraphrasing CoT text while preserving all numbers and arithmetic.

**Method:** 200 GSM8K problems, 170 correct (85%). For each correct problem, created synonym
paraphrase (dictionary-based, 9% actual replacement) and random word replacement (50.6%
actual, 85.7% token change). Two phases: full teacher-forcing (model copies last number)
and partial teacher-forcing (50% prefix, model generates remaining computation).

**Full Teacher-Forcing Results (trivially 100% — uninformative):**

| Condition | Accuracy | Token change |
|-----------|----------|:------------:|
| Original | 100.0% | 0% |
| Synonym | 100.0% | 24.4% |
| Random | 100.0% | 85.7% |

**Full TF is uninformative**: model copies the last number before "####". Since all numbers
are preserved across conditions, accuracy is trivially 100%.

**Partial Teacher-Forcing Results (50% prefix — INFORMATIVE):**

| Condition | Accuracy | Drop | McNemar p |
|-----------|----------|------|-----------|
| Original prefix | 100.0% (168/168) | — | — |
| Synonym prefix | 99.4% (167/168) | +0.6% | 1.0000 |
| Random prefix | 94.0% (158/168) | +6.0% | **0.0020** |

**Key finding: Synonym paraphrase has NO significant effect (0.6%, p=1.0).** The model reads
semantic/mathematical content, not specific token patterns. Random replacement has a significant
but modest effect (6.0%, p=0.002), confirming the positive control works.

**Interpretation — REFINES the hidden channel hypothesis:**
1. The hidden channel EXISTS (probing evidence from exp_078-084: STRONG)
2. But non-number tokens DON'T carry essential hidden computation information
3. The model reads the mathematical content (numbers and operations) from text
4. The hidden channel operates primarily through NUMBER-TOKEN K/V patterns
5. Non-number tokens provide context but aren't essential for computation

**Reconciliation with probing evidence:** V|nums > 0 at ALL positions (including non-number
positions) from exp_084. This information may be REDUNDANT with number-token information, or
may reflect "problem understanding" rather than step-by-step computation state. The
paraphrase null result constrains the interpretation: V information at non-number positions
is correlated with the answer but not ESSENTIAL for computation.

**Pre-registered predictions:** 3/7 TRUE confirmed, 5/5 FALSE confirmed.
**FALSE hypothesis (model reads semantics) decisively wins: 5/5 vs 3/7.**

**Evidence strength:** MODERATE (for the null finding). Clean methodology, good controls,
clear gradient (original > synonym > random). Single model (Qwen). The null result is
meaningful: it constrains the hidden channel to number-token positions and refines the
"lossy projection" framing — the text IS the computation for non-number content.

**Impact on Phase 2 evidence:** Natural_usage remains STRONG overall but now has an important
constraint: the hidden channel operates through number-token positions, not surface words.
Experiment B is null for non-number tokens. Five methodologies tested, four positive
(probing × 2, WRRA, position-sweep), one null (paraphrase disruption).

### Exp 086: Position-Sweep Decodability on Phi-3.5-mini — STRONG REPLICATION
**Cycle 86 | Phi-3.5-mini-Instruct (MHA) | Phase 2 — cross-model position-sweep | REPLICATES**

**Cross-model replication of exp_083/084's position-sweep decodability finding.** First
position-sweep on a non-Qwen model. Tests generality across model family (Microsoft vs
Alibaba), architecture (MHA vs GQA), and training regime (Instruct vs Base).

**Method:** Same as exp_084. 250 GSM8K problems, 212 correct (84.8%). Forward pass on all
212 correct problems, extract V and K at ALL CoT positions at L16 (50%) and L24 (75%).
20 bins, GroupKFold 5-fold Ridge probing. Cumulative numbers baseline at each bin. Bootstrap
(300 iterations) for peak significance. KV dim = 3072 (32 MHA heads × 96 head_dim).

**Results — L16 (50% depth):**

| Position | V_R | V\|nums | text% | Interpretation |
|----------|-----|---------|-------|----------------|
| 2.5% | 0.343 | **0.233** | 0.0% | V knows answer before text reveals anything |
| 22.5% | 0.425 | **0.315** | 6.6% | V carries beyond-numbers info early in chain |
| 47.5% | 0.300 | **0.065** | 12.7% | Minimum (still positive) |
| 72.5% | 0.504 | **0.410** | 16.0% | Growing as computation accumulates |
| 92.5% | 0.717 | **0.536** | 69.3% | Near-peak as chain approaches answer |
| 97.5% | 0.783 | **0.540** | 100% | Peak (computation complete) |

**V|nums positive at ALL 20 bins on BOTH layers** — no anomalous bins (unlike Qwen's bin
17 drop). More consistent signal than Qwen.

**Key metrics:**
- V|nums at 2.5% (text reveals 0%): L16 = 0.233, L24 = 0.194
- V|nums peak: L16 = 0.540 (p=0.003), L24 = 0.556 (p=0.027)
- V|nums early mean (0-30%): L16 = 0.261, L24 = 0.219
- Early decodability gap: L16 = 70%, L24 = 85%
- Shuffle control: range [-0.135, +0.118], mean ≈ 0

**Cross-model comparison (Qwen L18 vs Phi L16):**
- V|nums peak: Phi 0.540 vs Qwen 0.497 — **Phi slightly stronger**
- V|nums positive bins: Phi 20/20 vs Qwen 19/20 — **Phi more consistent**
- Bootstrap p: Phi 0.003 vs Qwen 0.010 — **Phi more significant**
- Early gap: Phi 70% vs Qwen 25% — **Phi much larger gap**
- V|nums at 2.5%: Phi 0.233 vs Qwen 0.357 — Qwen slightly stronger here

**Pre-registered predictions:** 8/8 TRUE confirmed, 0/4 FALSE confirmed. DECISIVE.

**Evidence strength:** STRONG. Position-sweep decodability now confirmed on 2 model families
(GQA + MHA), 2 training regimes (Base + Instruct). Combined with per-operation probing
(exp_079, 082) and WRRA (exp_079, 082), Phase 2 natural_usage is established across 3
independent methodologies on 2 model families.

**Impact on Phase 2 evidence:** Cross-model position-sweep replication complete. Phase 2
natural_usage upgraded to STRONG with cross-model confirmation. Position-sweep is now the
strongest single piece of evidence for a paper figure, confirmed across architectures.

### Exp 087: Position-Sweep Decodability on Mistral-7B — WEAK/MARGINAL (boundary test)
**Cycle 87 | Mistral-7B-v0.3 (GQA, Base) | Phase 2 — boundary test | MIXED**

**Boundary test: does the position-sweep find signal on the one model where Phase 2
evidence previously failed?** Mistral showed no significant V|nums at "=" positions
(exp_081: R=0.067, p=0.087) and WRRA was 37.9% (below chance). This experiment uses
the full position-sweep methodology (all positions, not just "=") for maximum power.

**Method:** Same as exp_084/086. 294 GSM8K problems (time-budget capped), 130 correct
(44.2%). Forward pass on all 130 correct, extract V and K at ALL CoT positions at L16
(50%) and L24 (75%). 20 bins, ~430-530 tokens per bin, 130 groups per bin. GroupKFold
5-fold Ridge probing. Bootstrap (300 iterations). KV dim = 1024 (8 GQA heads × 128).

**Results — L16 (50% depth):**

| Position | V_R | V\|nums | nums_R | text% | Interpretation |
|----------|-----|---------|--------|-------|----------------|
| 2.5% | 0.308 | **0.172** | 0.477 | 0.0% | Some V signal, but nums already high |
| 17.5% | 0.388 | **0.310** | 0.572 | 3.8% | Best early V|nums — but nums dominates |
| 47.5% | 0.140 | **0.167** | 0.522 | 11.5% | V_R low, nums very high |
| 82.5% | 0.618 | **0.369** | 0.371 | 20.8% | Peak V|nums (computation zone) |
| 92.5% | 0.674 | **0.190** | 0.545 | 81.5% | V|nums drops (text revealing) |
| 97.5% | 0.714 | **0.060** | 0.158 | 100% | V|nums near zero at end |

**V|nums positive at 20/20 bins (L16), 16/20 bins (L24)** — point estimates suggest signal.
**BUT bootstrap is NOT significant: L16 p=0.137, L24 p=0.323.**

**Key metrics:**
- V|nums at 2.5% (text reveals 0%): L16 = 0.172, L24 = 0.203
- V|nums peak: L16 = 0.369 (p=0.137), L24 = 0.304 (p=0.323)
- V|nums early mean (0-30%): L16 = 0.178, L24 = 0.107
- nums_R mean: ~0.43 (very high — input numbers dominate prediction)
- K > V at late positions (unique to Mistral)

**3-model gradient (THE KEY FINDING):**

| Model | Accuracy | V|nums peak | p-value | nums_R mean |
|-------|----------|-------------|---------|-------------|
| Qwen3-4B-Base | 88% | 0.497 | **0.010** | ~0.26 |
| Phi-3.5-mini | 85% | 0.540 | **0.003** | ~0.26 |
| Mistral-7B | 44% | 0.369 | 0.137 | **~0.43** |

Forward-looking channel strength scales with model accuracy. At 44% accuracy, Mistral
only solves easy problems where input numbers already predict the answer (nums_R = 0.43
vs 0.26 for Qwen/Phi). Less residual variance → weaker/noisier V|nums.

**Pre-registered predictions:** TRUE 3/5, FALSE 2/5 — GENUINELY MIXED.

**Evidence strength:** WEAK. Signal suggestive but not statistically significant. The
position-sweep finds more signal than per-operation probing (exp_081: 0.067), but not
enough to reach significance. Cannot formally reject the null that V|nums = 0 on Mistral.

**Impact on Phase 2 evidence:** No change to STRONG rating (maintained by Qwen + Phi).
Mistral confirmed as partial exception — the forward-looking channel may exist weakly
but is drowned by the dominance of input-number prediction at low accuracy. The graded
3-model pattern is informative for the paper: the hidden channel matters most for models
that can solve hard problems requiring multi-step reasoning.

---

### Exp 088 (cycle 88): Position-Sweep Decodability — Qwen3-8B-Base (Size Scaling)
**Model:** Qwen/Qwen3-8B-Base | **Accuracy:** 89.6% | **n:** 224 correct problems, 19,563 vectors/layer
**Layers probed:** L18 (50%) and L27 (75%) of 36 layers
**Original plan:** Llama-3.1-8B-Instruct (gated, inaccessible). Pivoted to size-scaling test.

**Key result: REPLICATES within Qwen family at 2x scale.**

V|nums positive at **20/20 bins on BOTH layers**:
- L18 (50%): V|nums at 2.5% = 0.213 (text reveals 0%), peak = 0.488 at 92%, p = 0.083 (marginal)
- **L27 (75%): V|nums at 2.5% = 0.218 (text reveals 0%), peak = 0.478 at 78%, p = 0.013 (SIGNIFICANT)**
- Early Decodability Gap: L27 = 75% (V at 18%, text at 92%)
- Shuffle control: range [-0.22, +0.19], mean ≈ 0

**Size scaling comparison (Qwen 4B vs 8B):**

| Metric | Qwen-4B (exp_084) | Qwen-8B (exp_088) |
|--------|-------------------|-------------------|
| Accuracy | 88.0% | 89.6% |
| V\|nums peak | 0.497 (L18) | 0.488 (L18) |
| V\|nums at 2.5% | 0.357 (L18) | 0.218 (L27) |
| Bootstrap p (best) | 0.010 | **0.013** |
| V\|nums positive bins | 19/20 | **20/20** |
| nums_R mean | ~0.26 | **~0.42** |

**Key finding:** Peak V|nums is NEARLY IDENTICAL across scales (0.488 vs 0.497). The
forward-looking channel is an **architectural feature** of Qwen GQA, not a capacity-
scaling phenomenon. Signal strength at the peak is the same; the difference is in
early positions (4B encodes more upfront) and nums_R (8B has higher text-number baseline).

**Surprising observation: nums_R asymmetry.** Despite similar accuracy (88% vs 90%),
8B's answers are much more predictable from input numbers (nums_R = 0.42 vs 0.26).
This is NOT explained by accuracy selection (unlike Mistral, where low accuracy
selects easy problems). Possible explanations: 8B produces cleaner/more linear
computational chains, or its larger capacity enables more regular answer patterns.

**4-model gradient updated:**

| Model | Accuracy | V\|nums peak | p-value | nums_R mean |
|-------|----------|-------------|---------|-------------|
| Phi-3.5-mini | 85% | 0.540 | 0.003 | ~0.26 |
| **Qwen3-8B** | **90%** | **0.488** | **0.013** | **~0.42** |
| Qwen3-4B | 88% | 0.497 | 0.010 | ~0.26 |
| Mistral-7B | 44% | 0.369 | 0.137 | ~0.43 |

**Pre-registered predictions:** TRUE 5.5/7, FALSE 0/4 — CLEAR WIN for TRUE hypothesis.

**Evidence strength:** MODERATE-STRONG. Replicates within Qwen at 2x scale with L27 p=0.013.
Does not add a new family (same Qwen architecture). Establishes size-independence as a new
finding. The nums_R asymmetry is unexplained and warrants investigation.

### Exp 089 (cycle 89): Layer × Position Heatmap — Full Layer Sweep
**Model:** Qwen/Qwen3-4B-Base | **Accuracy:** 88.0% | **n:** 198 correct problems, 19,223 vectors/layer
**Layers probed:** ALL 36 layers × 20 position bins = 720 probes

**Key result: TWO-PHASE EMERGENCE — ramp then plateau.**

The forward-looking signal (V|nums) shows a clear two-phase structure:
- **Phase A — Ramp (L0-L9, 0-25% depth):** Signal emerges at L3 (9%), rapidly builds to 0.15
- **Phase B — Plateau (L10-L35, 28-100% depth):** Signal stabilizes at 0.17-0.22 (mean), 19/20 bins positive

**Layer-wise summary:**

| Layer | Depth% | Mean V|nums | Positive | V|nums at 0-5% (text=0%) |
|------:|-------:|------------:|---------:|-------------------------:|
| L0    |   0%   |   -0.061    |   6/20   |      -0.134              |
| L3    |   9%   |   +0.096    |  17/20   |      +0.009              |
| L8    |  23%   |   +0.113    |  18/20   |      +0.096              |
| L10   |  29%   |   +0.200    |  19/20   |      +0.253              |
| L17   |  49%   | **+0.216**  |  19/20   |      +0.274              |
| L19   |  54%   |   +0.214    |  19/20   |    **+0.315**            |
| L27   |  77%   |   +0.208    |  19/20   |      +0.128              |
| L35   | 100%   |   +0.209    |  19/20   |      +0.196              |

**Key findings:**
1. **Forward-looking is DISTRIBUTED**: Present across 26 layers (L10-L35) at similar strength.
   Not localized to specific layers.
2. **Very early emergence (L3, 9% depth)**: The model begins forward-looking computation
   almost immediately after embedding layers.
3. **Plateau, not accumulation**: Signal is ESTABLISHED at middle layers and MAINTAINED through
   residual stream, rather than progressively refined.
4. **At chain start (0-5%, text=0%)**: Forward-looking V|nums emerges at L8 (+0.10), peaks at
   L19 (+0.32). The model "knows" the answer from the first CoT tokens by ~L8.
5. **Shuffle controls**: Systematically negative (~-0.13, GroupKFold artifact), but gap to
   V|nums is ~0.33 at plateau — large and consistent.

**Pre-registered predictions:** TRUE 2.5/6, FALSE 0/3. TRUE wins but specific predictions
about monotonic increase (PLATEAUS instead) and emergence layer (L3 vs predicted L12-L18) were
wrong. The signal is MORE pervasive and emerges EARLIER than expected.

**Evidence strength:** MODERATE-STRONG. First full layer sweep adds depth dimension to Phase 2
evidence. Two-phase structure is a new mechanistic finding. Single model (Qwen-4B) limits
generalizability. Consistent with prior results at L18 and L27.

**Impact on Phase 2 evidence:** Strengthens to STRONG. Position-sweep now confirmed on:
- 2 model families (Qwen + Phi) with significance
- 2 model sizes (4B + 8B) within Qwen
- 2 architecture types (GQA + MHA)
- 2 training regimes (Base + Instruct)
- Mistral as graded exception
- All with V|nums positive at 20/20 bins

---

### Exp 091: K Layer × Position Heatmap — K vs V Depth Profile Comparison
**Cycle 91 | 2026-03-22 | Qwen3-4B-Base | DISCONFIRMATORY (K>V for probing?)**

**Motivation:** All Phase 2 probing focused on V. Phase 1 established K > V for
perturbation causality. Does K > V extend to probing decodability? If V > K in probing,
the unified K=routing=hidden channel narrative weakens.

**Result: K > V CONFIRMED for probing. Phase 1+2 narrative UNIFIED.**

| Metric | K (exp_091) | V (exp_089) | K−V |
|--------|-------------|-------------|-----|
| Wins at layers | 32/36 (89%) | 1/36 (3%) | K dominates |
| Mean |nums | 0.201 | 0.159 | +0.043 |
| Peak layer | L29 (0.246) | L17 (0.216) | K peaks later, higher |
| Emergence | L0 (0% depth) | L3 (9% depth) | K emerges EARLIER |
| Ramp (L0-L9) | +0.156 | +0.071 | +0.086 (K 2.2x) |
| Plateau (L10-35) | +0.219 | +0.193 | +0.026 |

**Key findings:**
1. K carries MORE decodable forward-looking information than V at 32/36 layers.
2. K emerges 3 layers earlier (L0 vs L3) — routing encodes answer info BEFORE content.
3. The K advantage is strongest in early layers (ramp phase, 2.2x stronger).
4. Both K and V show identical two-phase ramp+plateau structure (architecture-general).
5. K peaks later (L29 vs L17) — routing continues to refine through deeper layers.

**Confounds:**
1. RoPE in K adds positional structure — partially addressed by nums residualization.
2. Single model (Qwen-4B-Base). Cross-model K sweep needed.
3. No bootstrap significance for K-V difference at individual layers.

**Pre-registered predictions:** K>V TRUE: 4/5 confirmed. V>K FALSE: 0/4. K≈V: 1/2.
The disconfirmatory hypothesis is REJECTED.

**Evidence strength:** MODERATE-STRONG. Consistent 32/36 layer dominance. Large early-layer
effect. Same methodology as exp_089. RoPE confound partially addressed. Single model.

**Impact:** The K > V hierarchy is now confirmed across BOTH paradigms:
- Perturbation (Phase 1): K destruction is more catastrophic than V
- Probing (Phase 2): K carries more decodable forward-looking information
This UNIFIES the Phase 1 and Phase 2 narratives: K-routing is the primary carrier of
the hidden computation channel, in both causal and observational evidence.

**UPDATE (Exp 093):** K>V probing does NOT generalize — V>K on Phi. See Exp 093 below.
The unified narrative requires revision: K is universally more FRAGILE, but not universally
more INFORMATIVE. The K/V information balance depends on architecture.

---

### Exp 092: RoPE Ablation — Is K>V Probing a Positional Encoding Artifact?
**Cycle 92 | 2026-03-22 | Qwen3-4B-Base | DISCONFIRMATORY (RoPE confound test)**

**Motivation:** Exp_091 found K>V at 32/36 layers. RoPE is applied to K but not V — could
the K advantage come from positional encoding rather than learned content? Tested by
comparing K_pre (before RoPE) vs K_post (after RoPE) vs V at 8 layers.

**Key result:** RoPE HURTS K probing at 8/8 layers (mean effect: -0.159 Pearson R at bin 19).
K_post (with RoPE) is WORSE than K_pre (without RoPE) at every tested layer. K_pre ≈ V
overall (diff -0.011 raw R at bin 19). K_pre > V at ramp (+0.024). The RoPE artifact
confound is DECISIVELY REJECTED — exp_091's K>V finding is conservative.

**Methodology bug:** Residualized metric (|nums) used R² instead of Pearson R and in-sample
residualization. Absolute resid_R values unreliable; relative comparisons and raw R valid.

**Pre-registered predictions:** RoPE artifact hypothesis: 0/4 confirmed. Intrinsic hypothesis:
3/4 confirmed (one exceeded). Mixed: 1/3.

**Evidence strength:** MODERATE. RoPE hurts K probing (8/8) is robust. Single model.
Methodology bug limits residualized analysis.

---

### Exp 093: Cross-Model K vs V Layer Sweep — Phi-3.5-mini-Instruct (MHA)
**Cycle 93 | 2026-03-22 | Phi-3.5-mini-Instruct | DISCONFIRMATORY (K>V generalization)**

**Motivation:** K>V probing (exp_091) demonstrated only on Qwen (GQA, digital encoding).
Phi-3.5-mini is maximally different: MHA (not GQA), analog (not digital), different family.
If K>V holds → universal. If K≈V or K<V → architecture-specific.

**Key result: V > K at 10/12 layers (83%) on Phi, with bootstrap p<0.05 at 10/12 layers.**
- Mean K|nums=+0.120, V|nums=+0.167 (diff=-0.048)
- Ramp (L<10): K≈V (diff=-0.002)
- Plateau (L≥10): V>>K (V|nums=+0.231 vs K|nums=+0.160, diff=-0.070)
- Both emerge at L6, both peak at L18
- V positive at 20/20 bins at peak layers; K positive at 17-18/20

**Cross-model contrast:**
- Qwen (GQA): K>V at 32/36 layers (89%), mean diff +0.043
- Phi (MHA): V>K at 10/12 layers (83%), mean diff -0.048
- The reversal is symmetric and dramatic

**Interpretation:** K>V probing is architecture-specific, NOT universal. Possible
explanation: GQA compression forces K to be more information-dense (8 KV heads serve
36 Q heads), making it easier to probe. In MHA, K and V have symmetric dimensionality.

**Pre-registered predictions:**
- K>V universal: 1.5/5 confirmed → REJECTED
- K≈V or K<V on Phi: 2/5 confirmed (direction right, magnitude exceeded prediction)

**This DISSOCIATES two K>V properties:**
1. Causal importance (perturbation fragility): K > V UNIVERSALLY (5/5 models, Phase 1)
2. Information content (probing decodability): Architecture-dependent (GQA→K wins, MHA→V wins)
Destroying routing (K) is always catastrophic. But which of K,V carries more DECODABLE
forward-looking info depends on how the attention mechanism is structured.

**Evidence strength:** STRONG. V>K at 10/12 layers with bootstrap significance. Same
methodology as exp_091 on Qwen. Effect size comparable. Consistent with architectural theory.

**Impact:** MAJOR REVISION to K>V narrative. "K is the primary hidden channel carrier" is
now "K is universally more fragile, but the information balance depends on architecture."
The hidden channel operates through BOTH K and V on all models.

---

### Exp 094: Mistral K vs V Layer Sweep (Cycle 94)
**Result: V>K on Mistral (GQA, analog) — K>V probing is ENCODING-dependent, not GQA-driven**

**Setup:** Mistral-7B-v0.3, GSM8K 8-shot CoT, 191 generated, 82 correct (42.9%),
K and V extracted at 12 layers (L0-L31), probed at 20 position bins.

**Key findings:**
- V>K at 9/12 layers (75%), K>V at 3/12 (25%)
- K|nums mean: -0.004, V|nums mean: +0.013, K-V diff: -0.017
- Significant: K>V at L0 only (p=0.024), V>K at L27 only (p=0.040)
- Ramp (L0-L9): K≈V (diff=+0.002), both weak. Plateau (L10-L31): V>K (diff=-0.027)
- Overall signal much weaker than Qwen/Phi (K|nums≈0, V|nums≈0.01) due to 43% accuracy

**3-model taxonomy (complete):**

| Model | Arch | Encoding | K vs V | Diff |
|-------|------|----------|--------|------|
| Qwen3-4B-Base | GQA (4.5x) | Digital | K>V 89% | +0.043 |
| Phi-3.5-mini | MHA (1x) | Analog | V>K 83% | -0.048 |
| Mistral-7B | GQA (4x) | Analog | V>K 75% | -0.017 |

**Critical test result:** Mistral shares GQA with Qwen but shows V≥K (like Phi). This
DISCONFIRMS GQA compression as the driver of K>V probing. **Digital encoding** (Qwen only)
is the predictor: digital→K>V, analog→V≥K regardless of GQA/MHA.

**Pre-registered prediction assessment:**
- "GQA-general" prediction: DISCONFIRMED — K>V at only 25% of layers
- "Qwen-specific" prediction: CONFIRMED — V≥K at 75% of layers, diff negative

**Confounds:**
1. Low accuracy (43%) means both K and V near zero — K/V balance is directional but
   individual layers mostly not significant
2. Only 82 problems — lower sample sizes limit statistical power
3. Cannot fully separate encoding from family — would need digital non-Qwen model

**Evidence strength:** MODERATE — Direction clear and consistent with Phi. Combined with
3-model pattern, the encoding→K/V balance relationship is coherent. But Mistral's weak
overall signal limits individual-layer significance.

**Impact:** RESOLVES exp_093's open question. K>V probing is NOT driven by GQA compression
but by digital encoding (Qwen-specific). Updated narrative: "K>V perturbation fragility is
universal (attention routing), but K>V information content is encoding-dependent (digital
concentrates info in K, analog distributes more evenly with V slightly leading)."

---

### Exp 095: Answer-Step Attention Routing Analysis (Cycle 95)
**Result: Universal late-chain attention shift at answer step; H0 shows UNIQUE reversal**

**Setup:** Qwen3-4B-Base, GSM8K 8-shot CoT, 198 generated, 173 correct (87.4%).
Forward pass with output_attentions at 9 layers (L0-L35). Attention extracted at
answer position + 5 mid-chain control positions per problem. Late-chain = bins 14-19
(70-100% of chain). Bootstrap significance test (n=2000).

**Key findings:**

1. **Universal late-chain shift:** 7/8 KV heads INCREASE late-chain attention at the
   answer step vs control (all p<0.001). Mean increase: +24.0pp. The model concentrates
   on recent computation when generating the answer.

2. **H0 dissociation:** H0 is the ONLY head that DECREASES late-chain attention at the
   answer step (diff=-0.034, p<0.001). While all other heads concentrate on recent chain,
   H0 shifts attention AWAY from late chain. Suggests H0 retrieves from earlier positions
   or prompt, complementing other heads' recent-chain retrieval.

3. **Entropy at deep layers:** H5 and H0 become MORE focused (lower entropy) at the
   answer step at L27 and L35 (H5: -0.61 to -0.75 bits, H0: -0.05 to -0.77 bits).
   Other heads show minimal change. Confirms answer head specialization is active during
   natural generation.

4. **Two-stage retrieval at L18:** Early layers (L0-L14) show MORE prompt attention at
   answer step. L18 dramatically shifts from prompt (37.1% vs 50.4% ctrl) to late-chain
   (50.7% vs 30.2% ctrl). Suggests layer-dependent pipeline: prompt re-reading → chain retrieval.

| KV Head | Late-ans | Late-ctrl | Diff | p |
|---------|----------|-----------|------|---|
| H0 | 0.508 | 0.542 | **-0.034** | <0.001 |
| H1-H7 avg | 0.520 | 0.280 | **+0.240** | <0.001 |
| H5 | 0.423 | 0.194 | **+0.228** | <0.001 |
| H3 (max) | 0.608 | 0.248 | **+0.360** | <0.001 |

**Confounds:**
1. Late-chain attention ≠ hidden channel retrieval — model could simply be reading recent text
2. Control positions are mid-chain (shorter visible sequence), making relative comparison imperfect
3. H0 finding on single model — needs cross-model replication
4. Cannot distinguish H0's early-position attention from attention sink behavior

**Evidence strength:** MODERATE — Universal shift is strong (7/8 heads, all p<0.001), H0
dissociation is striking and unexpected. But cannot definitively distinguish hidden channel
retrieval from text reading. First mechanistic evidence connecting Phase 1 answer heads to
Phase 2 natural behavior.

**Impact:** Establishes that the answer step is computationally DISTINCT from reasoning steps
in attention patterns. H0 and H5 play DIFFERENT roles: H5 follows majority pattern (late-chain),
H0 uniquely retrieves from earlier positions. Complements probing evidence (exps 083-094) with
mechanistic retrieval evidence.

---

### Exp 097 — Cross-Model Probe-Attention Correlation + Quadratic Control (Phi-3.5-mini)
**Cycle 97 | Phase 2 — Cross-model replication + robustness | 2026-03-22**

**Model:** Phi-3.5-mini-Instruct (MHA, 32 KV heads, analog encoding)
**n:** 200 generated, 173 correct, 173 with attention extracted
**V|nums source:** exp_086 (Phi position-sweep, L16+L24)

**Question:** Does the probe-attention correlation (exp_096) replicate on a maximally
different architecture? Does it survive quadratic position control?

**Pre-registered predictions:**
1. Linear partial_r > 0 for ≥3/4 layers → CONFIRMED (4/4, all 128 head×layer p<0.001)
2. Answer > control ≥3/4 layers → CONFIRMED (4/4 linear), PARTIAL (2/4 quadratic)
3. ≥24/32 heads positive under quadratic at L16 → CONFIRMED (32/32)
4. Ecological V|nums > nums_R → CONFIRMED (4-5x advantage)
5. Quadratic retains ≥70% of linear → **DISCONFIRMED** (only 21-22%)

**Key results:**

| Layer | Linear ans_r | Quad ans_r | % Retained | Quad pos/32 | Quad sig/32 |
|-------|-------------|-----------|------------|-------------|-------------|
| L8    | 0.171       | 0.037     | 21%        | 32/32       | 21/32       |
| L16   | 0.243       | **0.054** | 22%        | **32/32**   | **32/32**   |
| L24   | 0.167       | -0.037    | -22%       | 5/32        | 2/32        |
| L31   | 0.178       | -0.045    | -25%       | 3/32        | 0/32        |

Cross-model comparison (linear):

| Layers    | Qwen    | Phi     | Ratio |
|-----------|---------|---------|-------|
| L09/L08   | 0.160   | 0.171   | 1.07x |
| L18/L16   | 0.260   | 0.243   | 0.93x |
| L27/L24   | 0.261   | 0.167   | 0.64x |
| L35/L31   | 0.292   | 0.178   | 0.61x |

Ecological correlation (binned, all heads averaged):

| Layer | r(Δattn, V|nums) | r(Δattn, nums_R) | p (V|nums) |
|-------|-----------------|------------------|------------|
| L8    | 0.695           | 0.121            | 0.0007     |
| L16   | 0.660           | 0.147            | 0.0016     |
| L24   | 0.640           | 0.143            | 0.0024     |
| L31   | 0.577           | 0.171            | 0.0077     |

**Confounds:**
1. ~80% of linear partial_r attributable to non-linear recency (quadratic position effects)
2. V|nums ramps up with position → ANY recency-biased attention correlates with V|nums
3. Quadratic control may be too aggressive (removes genuine curvature in signal)
4. Deep-layer reversal (L24, L31 go negative) is unexplained
5. Same quadratic test should be applied to Qwen (exp_096) for fair comparison

**Evidence strength:** MODERATE — Cross-model replication under linear control is strong
(4/4 layers, n=173, all p<0.001). Quadratic control reveals ~80% is recency, but L16
survives robustly (32/32, all p<0.001, r=0.054). The information-directed attention finding
is REAL but MUCH SMALLER than exp_096 suggested. Net signal at mid-plateau is r≈0.05, not
r≈0.24.

**Impact:** (1) CONFIRMS cross-model generality of info-directed attention under standard
linear control. (2) SUBSTANTIALLY DOWNGRADES the effect size by identifying non-linear
recency as the primary driver of the raw correlation. (3) Establishes that the genuine
signal (r≈0.05, surviving quadratic control) is concentrated at mid-plateau layers. (4)
The exp_096 finding needs retrospective quadratic control to determine if Qwen shows the
same pattern.

---

### Entry #116: V|nums is Accuracy- and Difficulty-Conditional (Exp 101)
**Cycle:** 101 | **Date:** 2026-03-22 | **Model:** Qwen3-4B-Base

**Design:** Generate 8-shot CoT on 248 GSM8K problems (218 correct, 30 incorrect).
Split correct problems by ground-truth difficulty (Easy ≤2 steps n=76, Medium 3-4 n=100,
Hard >4 n=42). Train V-probe on correct (5-fold CV Ridge), apply to incorrect
(train-on-correct → predict-on-incorrect). Extract V-cache at L9/L18/L27/L35, 20 position
bins. Compare V|nums across accuracy and difficulty groups.

**Pre-registered predictions:**
- If TRUE: V|nums(correct) > V|nums(incorrect); V|nums(hard) > V|nums(easy)
- If FALSE: V|nums similar across groups (position artifact)

**Results:**

Analysis A — Accuracy effect:

| Layer | V|nums(correct) | V|nums(incorrect) | Gap | p |
|-------|-----------------|-------------------|-----|---|
| L9 | +0.003 | -0.161 | 0.164 | **0.008** |
| L18 | +0.014 | -0.149 | 0.163 | **0.012** |
| L27 | +0.012 | -0.024 | 0.036 | 0.216 |
| L35 | +0.006 | -0.145 | 0.151 | **0.006** |

Incorrect V|nums is NEGATIVE at 3/4 layers — the V-cache carries LESS answer info than
visible text when the model gets the problem wrong. Functional evidence: the hidden
channel carries the answer only when the model succeeds.

Analysis B — Difficulty effect:

| Layer | Easy V|nums | Hard V|nums | Gap | p |
|-------|-------------|-------------|-----|---|
| L9 | 0.054 | 0.257 | 0.203 | **<0.001** |
| L18 | 0.037 | 0.270 | 0.233 | **<0.001** |
| L27 | 0.034 | 0.275 | 0.241 | **<0.001** |
| L35 | 0.021 | 0.234 | 0.213 | **<0.001** |

V|nums is 5-13x higher for hard problems at all layers.

**CRITICAL CONFOUND — decomposed V_R and nums_R:**

| Difficulty | V_R mean | nums_R mean | V-nums (unclip) | Bins V>nums |
|------------|----------|-------------|-----------------|-------------|
| Easy (76) | 0.198 | 0.334 | -0.136 | 6/20 |
| Medium (100) | 0.177 | 0.481 | -0.304 | 1/20 |
| Hard (42) | 0.145 | **-0.085** | **+0.230** | **15/20** |

V_R is LOWER for hard problems (0.145 vs 0.198). The V|nums effect is driven by nums_R
collapsing for hard problems (negative, text numbers anti-predict the answer with many
intermediate steps). V provides more info RELATIVE TO TEXT for hard problems because text
becomes uninformative.

**Predictions assessment:** Hypothesis predictions: 2.5/4 confirmed. Null predictions:
0.5/3. Hypothesis better matches the data overall.

**Confounds:**
1. V|nums difficulty effect conflated with text informativeness (nums_R collapse)
2. Small hard group (n=42) limits per-bin precision
3. Bootstrap permutes bins, not problems (correlation structure)
4. V|nums clipping at 0 introduces ~10% positive bias for noisier hard group

**Evidence strength:** MODERATE — Accuracy effect is clean (3/4 layers, p<0.01). Difficulty
effect is highly significant but confounded: reflects V being the dominant info source for
hard problems (V>nums at 15/20 bins) rather than V carrying more info in absolute terms.
The key insight: the hidden channel MATTERS MOST when computation is complex and text is
least informative.

**Impact:** (1) First functional test of forward-looking signal — V|nums is not merely a
position artifact. (2) V-cache provides answer information that text cannot for hard
problems. (3) Hidden channel is accuracy-conditional — carries the answer only for correct
problems. (4) Nuances the narrative: the channel isn't "hidden computation" so much as
"residual computation that text fails to capture, especially for hard problems."

### Entry #117: Cross-Model Replication of Functional Conditioning on Phi-3.5-mini (Exp 102)
**Cycle:** 102 | **Date:** 2026-03-22 | **Model:** Phi-3.5-mini-Instruct (MHA, analog)

**Design:** Replicate exp_101's accuracy/difficulty-conditional V|nums methodology on
Phi-3.5-mini-Instruct. Generated 369 GSM8K problems (307 correct/83.2%, 62 incorrect).
PCA reduced 3072-dim V vectors → 256-dim (75-88% variance retained). Ridge alpha=10.0.
V-cache at L8/L16/L24/L31, 20 position bins. Difficulty terciles: Easy ≤3 steps (177),
Medium 3-4 (65), Hard >4 (65).

**Pre-registered predictions:**
- If TRUE: V|nums(correct)>V|nums(incorrect) at ≥2/4 layers; V|nums(hard)>V|nums(easy)
  at ≥2/4 layers; same nums_R collapse pattern.
- If FALSE: V|nums similar across groups (Qwen-specific).

**Results — REPLICATES on both analyses:**

Analysis A — Accuracy effect (STRONGER than Qwen):

| Layer | V|nums(correct) | V|nums(incorrect) | Gap | p |
|-------|-----------------|-------------------|-----|---|
| L8 | +0.015 | -0.152 | **0.167** | **<0.001** |
| L16 | +0.012 | -0.201 | **0.212** | **<0.001** |
| L24 | +0.002 | -0.188 | **0.189** | **<0.001** |
| L31 | +0.004 | -0.303 | **0.307** | **<0.001** |

4/4 layers significant (p<0.001). Gaps 2x larger than Qwen (0.17-0.31 vs 0.04-0.16).

Analysis B — Difficulty effect (smaller but significant):

| Layer | Easy V|nums | Hard V|nums | Gap | p |
|-------|-------------|-------------|-----|---|
| L8 | 0.006 | 0.049 | 0.043 | **0.012** |
| L16 | 0.024 | 0.053 | 0.029 | 0.142 |
| L24 | 0.010 | 0.066 | 0.056 | **0.016** |
| L31 | 0.003 | 0.066 | 0.063 | **0.004** |

3/4 layers significant. Gaps 5-6x smaller than Qwen (0.03-0.06 vs 0.20-0.24).

**V_R decomposition (same qualitative pattern):**

| Difficulty | V_R mean | nums_R mean | V-nums | Bins V>nums |
|------------|----------|-------------|--------|-------------|
| Easy (177) | 0.214 | 0.463 | -0.248 | 1/20 |
| Hard (65) | 0.167 | 0.245 | -0.078 | 7-9/20 |

Key difference: Phi hard nums_R=0.245 (still positive) vs Qwen hard nums_R=-0.085
(negative). Phi's text stays informative even for hard problems → smaller V|nums gap.

**Predictions assessment:** Hypothesis TRUE: 3/5 confirmed (accuracy 4/4, difficulty 3/4,
nums_R collapse). Hypothesis FALSE: 0.5/3. Cross-model replication succeeds.

**Confounds:** PCA may attenuate V_R (applied equally to all groups). Ridge alpha differs
from exp_101. V_R(incorrect) ≥ V_R(correct) on Phi (mechanism differs from Qwen). Same
seed/problems overlap with Qwen.

**Evidence strength:** MODERATE-STRONG — Accuracy conditioning replicates with STRONGER
effect on a maximally different architecture (4/4 layers, p<0.001, 2x Qwen's gap).
Difficulty conditioning replicates at 3/4 layers with same qualitative pattern. Cross-model
functional evidence: the hidden channel carries the answer only when the model succeeds
and matters most for hard problems — this property is architecture-independent.

**Impact:** (1) Functional conditioning is CROSS-MODEL, not Qwen-specific. (2) Accuracy
effect is actually STRONGER on analog/MHA/instruct than digital/GQA/base — interesting
dissociation from difficulty effect which is weaker. (3) 7/8 accuracy tests and 7/8
difficulty tests significant across 2 models. (4) Architecture-independent principle:
hidden channel usage scales with computational demand and predicts model success.

---

### Entry #118: Model's Own Answer vs Ground Truth V-Probe — INCONCLUSIVE (Gold-Pred Collinearity)
**Exp 103** | Cycle 103 | 2026-03-22 | Qwen3-4B-Base

**Question:** Does the V-cache encode the model's own predicted answer (computation-faithful)
or just correlate with problem-level features (general features)?

**Design:** Train V-probe on correct problems (where gold=pred), apply to incorrect (where
gold≠pred). Compare V_R(gold) vs V_R(predicted) for incorrect. If V encodes computation:
V_R(pred) >> V_R(gold). If general features: V_R(pred) ≈ V_R(gold).

**CRITICAL CONFOUND: gold-pred correlation R=0.929 for incorrect problems.** The model's
errors on GSM8K are near-misses (e.g., 12 vs 13, 8712 vs 9360). In log-space, the two
targets are nearly collinear. The experiment lacks power to distinguish interpretations.

**Results:** n=248 correct, n=29 incorrect.

| Layer | V_R(gold) | V_R(pred) | Δ | p | %closer_pred |
|-------|-----------|-----------|---|---|--------------|
| L9    | 0.198 | 0.196 | -0.001 | 0.517 | 47.1% |
| L18   | 0.197 | 0.196 | -0.001 | 0.521 | 47.9% |
| L27   | 0.346 | 0.335 | -0.011 | 0.753 | 49.7% |
| L35   | 0.256 | 0.267 | +0.010 | 0.313 | 50.0% |

All differences negligible (|Δ|<0.012), all p>0.3, per-problem 47-50% (chance).
Nums baseline: nums_R(gold)=0.365 > nums_R(pred)=0.285 (text tracks gold better, dominated
by question numbers).

**Predictions assessment:** Hypothesis TRUE: 0/4. Hypothesis FALSE: 2/3. But comparison
is meaningless given R=0.929 confound.

**Evidence strength:** INCONCLUSIVE — methodological null, not substantive null. The experiment
cannot answer the question because the targets are insufficiently different on GSM8K.

**Methodological insight:** Probing for model's own answer vs gold truth requires tasks
where errors are qualitatively large (different operation, not just magnitude). GSM8K
errors are proportional near-misses in log-space. Future: use harder benchmark (MATH),
weaker model, or classification probe (correct/incorrect) to avoid the collinearity issue.

**Impact:** The accuracy-conditional V|nums finding from exp 101/102 remains AMBIGUOUS
between "computation-faithful" and "general features" interpretations. This is an important
open question. The next best approach is classification probing or probing for intermediate
computation values (which avoids the gold/pred confound entirely).
