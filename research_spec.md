# Research Specification

## Hypothesis
Chain-of-thought (CoT) reasoning text is a **lossy projection** of the model's internal computation. The KV cache carries a functionally separable hidden channel that encodes answer-relevant information independent of the visible reasoning tokens. The model computes through high-dimensional hidden representations and is constrained to emit plausible tokens as a byproduct of the autoregressive architecture, but the text itself is not the computation.

## Background
Recent work on chain-of-thought prompting has shown that LLMs can solve complex reasoning problems by generating intermediate steps. However, the relationship between the visible reasoning text and the model's internal computation is poorly understood. Several lines of evidence suggest the text may not faithfully represent the computation: models sometimes arrive at correct answers despite flawed reasoning steps, and the KV cache carries far more information than is needed to produce the visible tokens.

This investigation probes the mechanistic relationship between visible reasoning tokens and the hidden states that produce them. If the KV cache carries a separable hidden channel, this has implications for interpretability (we can't trust the text), safety (models can compute things we can't see), and efficiency (latent reasoning architectures remove an unnecessary bottleneck).

The core experimental strategy is to perturb the KV cache during chain-of-thought generation and measure the effect on (a) the visible token predictions and (b) the final answer. If these can be dissociated — changing the answer without changing the text — then the hidden channel exists.

## Evidence Targets
What would constitute sufficient evidence FOR the hypothesis:
- Demonstration that adversarial KV cache perturbations can change the final answer while preserving all intermediate token predictions (null space existence)
- Spatial structure in the null space: perturbations concentrate at answer-coupled positions and avoid text-coupled positions
- Sharp SNR cliff in KV cache noise experiments, indicating precisely-encoded digital information rather than gracefully-degrading analog computation
- Cross-model variation in text-dependence, showing the hidden channel's relative importance varies by architecture
- Replication of key findings across at least two model families

What would constitute sufficient evidence AGAINST the hypothesis:
- Inability to find adversarial perturbations that dissociate text from answer (null space is empty or negligible)
- No spatial structure: perturbations affect text and answer positions uniformly
- Gradual accuracy degradation under KV noise (analog-like, not digital-like)
- High text-compliance across all models (models always follow corrupted reasoning text, suggesting text IS the computation)

## Scope & Constraints
- Models: Qwen3-4B-Base (primary), Qwen3-8B, Llama-3.1-8B (replication)
- Benchmark: GSM8K (8-shot chain-of-thought)
- Compute budget: 48 hours total
- What is OUT of scope: training or fine-tuning models, multi-GPU experiments, proprietary/API-only models

## Key Literature
- Lanham et al. (2023) "Measuring Faithfulness in Chain-of-Thought Reasoning": establishes that CoT is not always faithful
- Turpin et al. (2023) "Language Models Don't Always Say What They Think": demonstrates systematic unfaithfulness in CoT
- Pfau et al. (2024) "Let's Think Dot by Dot": shows models can reason with filler tokens, suggesting computation happens in hidden states
- Coconut (Hao et al. 2024): latent reasoning without text tokens — relevant to the "text bottleneck" framing
- Quiet-STaR (Zelikman et al. 2024): internal reasoning tokens — shows benefit of removing text constraint
- SnapKV, H2O, Think Clearly: KV cache compression methods that may inadvertently exploit the text/answer channel separation

## Starting Evidence

### Experiment 1: Steganographic Channel Capacity

**Scripts:** `experiment.py`, `stego_demo.py`
**Models:** Qwen2.5-3B (embedding), GPT-2 (token-selection)
**Results:** `figures/Qwen2.5-3B/`, `figures/stego_demo/`

#### Motivation

Establish that LLM output distributions have enormous unused bandwidth — far more information could be encoded in each token choice than is needed for the visible text.

#### Method

Two complementary approaches:

1. **Embedding-space steganography** (`experiment.py`): Perturb embedding vectors using orthogonal directions (QR decomposition) to encode secret bits. Sweep perturbation magnitude (alpha) and bits-per-token (K = 1-256) across 5 sentences.

2. **Token-selection steganography** (`stego_demo.py`): Binary partition coding — recursively split candidate tokens into equal-probability groups, use secret bits to select groups. Full encode/decode round-trip with KV-cache autoregressive generation.

#### Key Results

**Token-selection (GPT-2):**
- 128-bit message encoded in 25 tokens (5.12 bits/token average, up to 9 bits/token)
- Perfect round-trip recovery (100% decode accuracy)
- Mean per-token Shannon entropy: 4.44 bits — substantial unused capacity per token

**Embedding-space (Qwen2.5-3B):**
- Token preservation rate mapped across alpha x K parameter space
- 100% token preservation achievable up to K ~ 64 bits/token at small alpha
- Message recovery near-perfect across the entire sweep

#### Significance

LLM output distributions routinely carry 4-5 bits of unused capacity per token. This establishes that there is ample "room" in the distribution for a hidden computation channel — the model doesn't need to use the visible token to carry all its information.

---

### Experiment 2: CoT Narrows the Output Distribution

**Script:** `gsm8k_capacity.py`
**Models:** Qwen3-4B-Base, Llama-3.1-8B
**Results:** `figures/gsm8k_capacity_qwen3-4b-base/`, `figures/gsm8k_capacity_llama-3.1-8b/`

#### Method

Measure Shannon entropy (bits) of the output distribution token-by-token during GSM8K generation under three prompting conditions:

| Condition | Prompt suffix | What it tests |
|---|---|---|
| `zero_shot_none` | `A:` | Free generation, no guidance |
| `standard_cot` | `A: Let's think step by step.` | Chain-of-thought |
| `no_cot` | `A: The answer is` | Reasoning suppressed |

#### Key Results

**Qwen3-4B-Base:**

| Condition | Tokens | Mean entropy | Median entropy | Total entropy |
|---|---|---|---|---|
| `no_cot` | 500 | 1.74 bits/token | 1.13 bits/token | 870.8 bits |
| `standard_cot` | 5,000 | 0.54 bits/token | 0.023 bits/token | 2,700 bits |
| `zero_shot_none` | 4,694 | 0.51 bits/token | 0.021 bits/token | 2,411 bits |

**Llama-3.1-8B:**

| Condition | Tokens | Mean entropy | Median entropy | Total entropy |
|---|---|---|---|---|
| `no_cot` | 487 | 2.65 bits/token | 2.50 bits/token | 1,292 bits |
| `standard_cot` | 3,882 | 1.19 bits/token | 0.63 bits/token | 4,638 bits |

#### Significance

CoT narrows the per-token entropy by ~3x (Qwen: 1.74 -> 0.54; Llama: 2.65 -> 1.19). The model becomes more deterministic during reasoning — each step constrains the next. But the median entropy during CoT is near zero (0.023 bits for Qwen), meaning most tokens are essentially forced. The "computation" isn't happening in the token choices; it's happening in the hidden states that produce those near-deterministic tokens.

---

### Experiment 3: KV Cache State Is Fragile and Precisely Encoded

**Script:** `kv_noise_experiment.py`
**Model:** Qwen3-4B-Base
**Results:** `figures/kv_noise_qwen3-4b-base/`

#### Method

Inject calibrated Gaussian noise into the KV cache during 8-shot CoT generation on 100 GSM8K problems. Noise is calibrated per-layer based on L2 norms to achieve a target SNR. Five experimental phases: baseline, noise channel comparison (teacher-forced / noise-all / noise-reasoning-only), SNR sweep, position-specific noise, layer-specific noise.

#### Key Results

**Sharp accuracy cliff at SNR 14 dB:**

| SNR (dB) | Answer accuracy | Token accuracy |
|---|---|---|
| inf (clean) | 100% | 99.8% |
| 20 | 100% | 98.6% |
| 15 | 100% | 89.8% |
| **14** | **92.6%** | **82.0%** |
| **13** | **40.7%** | **55.3%** |
| 12 | 3.7% | 23.9% |
| 10 | 0% | 14.3% |
| 0 | 0% | 2.7% |

The transition from 100% to near-0% accuracy occurs within a ~3 dB window (15 dB -> 12 dB). This is characteristic of precisely-encoded digital information, not gracefully-degrading analog computation.

#### Significance

The KV cache during CoT encodes ~14 dB of answer-relevant information — roughly 2-3 bits per element. Small perturbations destroy it completely. This fragility is inconsistent with the view that "the text carries the reasoning" (text is recoverable at much lower SNR since individual token predictions degrade gracefully). The answer depends on precise hidden state, not the visible token sequence.

---

### Experiment 4: Adversarial Null Space in the KV Cache

**Scripts:** `kv_distortion_experiment.py`, `analyze_perturbation_structure.py`
**Model:** Qwen3-4B-Base
**Results:** `figures/kv_distortion_qwen3-4b-base/`

#### Method

Use PGD (Projected Gradient Descent) optimization to find KV cache perturbations that change the model's final answer while preserving every intermediate token prediction. This tests whether the KV cache has a null space — directions that affect the answer without affecting the text.

Three attack scopes:
1. **Prompt-only attacks**: Perturb only the exemplar (prompt) KV entries; reasoning tokens are generated through the perturbed cache but their KV entries are not directly modified.
2. **Reasoning-only attacks**: Perturb only the reasoning-phase KV entries; prompt KV entries remain clean.
3. **Binary search baseline**: Find the maximum isotropic noise scale that preserves correct answers.

#### Key Results

**The null space is enormous:**

| Metric | Isotropic noise threshold | Adversarial perturbation |
|---|---|---|
| Relative to signal norm | Median 0.006x | Mean 377x |

Isotropic (random) noise breaks the answer at 0.006x the signal norm. Structured adversarial perturbation preserves all text predictions at 377x the signal norm. The adversarial perturbation lives in a null space of the text-generation function that is invisible to the token predictions but visible to the answer computation.

**Prompt-only vs reasoning-only attacks (5 problems, filtered to attacks where clean model was correct):**

| Scope | Success rate | n (attacks) |
|---|---|---|
| Prompt-only | 100% | 6 |
| Reasoning-only | 40% | 15 |

Prompt attacks always succeed because the prompt KV entries have loose coupling to reasoning text (they influence everything downstream but aren't constrained by any specific token prediction). Reasoning attacks are harder because reasoning KV entries are tightly coupled to both text and answer.

**Per-layer noise sweep:** All 36 layers independently support attack — the answer-relevant information is distributed across the full depth of the model.

**Noise tolerance:** Attacks remain effective up to 0.1x signal noise; sharp failure at 0.316x.

#### Significance

The existence of a large null space proves that the KV cache carries two functionally separable channels: one that determines the visible text (tightly constrained, breaks under small random noise) and one that determines the answer (exploitable by adversarial perturbation without affecting text). This is the central mechanistic finding.

---

### Experiment 5: Spatial Structure of the Hidden Channel

**Script:** `attention_perturbation_correlation.py`
**Model:** Qwen3-4B-Base
**Results:** `figures/kv_distortion_qwen3-4b-base/correlation_results.json`

#### Method

For each of 47 GSM8K problems (141 PGD attacks: 3 variants per problem), compute:
- **Per-position attention**: How much the answer token attends to each reasoning position (using answer-phase attention weights with `output_attentions=True`)
- **Per-position perturbation**: How much the PGD optimizer perturbs the KV cache at each reasoning position
- **Correlation**: Spearman rho between attention and perturbation across positions

Define two coupling metrics:
- **Answer-coupling**: Attention from the answer token to each reasoning position (via top-5 answer-selective heads)
- **Text-coupling**: Attention from later reasoning tokens to each reasoning position

#### Key Results

**The central dissociation:**

| Metric | Mean rho | 95% CI | Interpretation |
|---|---|---|---|
| Answer-coupling | 0.781 | [0.768, 0.794] | PGD targets positions that feed the answer |
| Text-coupling | -0.062 | [-0.081, -0.044] | PGD ignores positions that feed later text |

The PGD optimizer discovers the spatial structure of the hidden channel without being told about it: it concentrates perturbation at positions where the answer token attends and leaves text-scaffolding positions untouched.

**Distribution of per-attack answer-coupling rho (141 attacks):**

| Range | Count | Percentage |
|---|---|---|
| rho > 0.8 | 68 | 48% |
| 0.6 < rho <= 0.8 | 69 | 49% |
| rho <= 0.6 | 4 | 3% |
| **Median** | **0.793** | |
| **Min / Max** | **0.433 / 0.918** | |

The dissociation is consistent across 97% of attacks (rho > 0.6), regardless of whether the attack succeeds or fails. The optimizer always uses the same strategy.

**Pooled correlations (n = 14,394 position-level observations):**

| Attention metric | Spearman rho | Pearson r | Note |
|---|---|---|---|
| Effective contribution | 0.926 | 0.521 | Best overall metric |
| Mean attention | 0.904 | 0.576 | Robust across heads |
| Answer-head-only | 0.751 | 0.259 | Noisier but mechanistically precise |
| Text coupling | -0.042 | 0.022 | Near zero — confirms dissociation |

Note: Large Spearman-Pearson gap (0.926 vs 0.521) indicates a monotonic but nonlinear relationship.

**Per-layer validation:** Median per-layer rho = 0.871 (range [0.710, 0.935]). The correlation holds within individual layers, not just across the pooled aggregate.

**Binned quartile analysis (positions binned by attention):**

| Attention quartile | Mean perturbation (normalized) | Mean attention (normalized) |
|---|---|---|
| Q1 (lowest attention) | 0.796 | 0.133 |
| Q2 | 0.929 | 0.262 |
| Q3 | 1.046 | 0.545 |
| Q4 (highest attention) | 1.229 | 3.058 |

Monotonic increase: positions with 23x more attention (Q4 vs Q1) receive 1.54x more perturbation.

**Token type analysis:**

| Token type | Mean attention | Mean perturbation | n |
|---|---|---|---|
| Formatting (newlines, spaces, parens) | 0.00342 | 7.19 | 2,760 |
| Numbers | 0.00218 | 6.62 | 3,693 |
| Operators (+, -, x, /, =) | 0.00216 | 7.13 | 1,080 |
| Text (words) | 0.00098 | 6.14 | 6,861 |

Formatting tokens receive 3.5x more attention from the answer token than text tokens. These structural delimiters (newlines between steps, equals signs) may serve as "checkpoints" that the answer head reads to aggregate computation.

**Layer structure:** Perturbation concentrates in late layers (>=18), consistent with answer computation happening in the later transformer blocks.

#### Significance

The hidden channel has spatial structure: it lives at specific positions (answer-coupled) and in specific layers (late). The text channel occupies different positions (text-coupled). These two channels share the same KV cache but are functionally separable — the PGD optimizer discovers this separation automatically by targeting the answer channel while leaving the text channel untouched.

---

### Experiment 6: Forced Corrupted Reasoning

**Script:** `forced_corrupted_reasoning/run_experiment.py`
**Models:** Llama-3.1-8B, Qwen3-8B
**Results:** `figures/forced_corrupted_reasoning/`

#### Method

Test whether models follow corrupted reasoning text or ignore it in favor of their own computation. Four phases:

1. **Trace generation (Phase 1):** Generate 8-shot CoT traces via API for all 1,319 GSM8K problems.
2. **Corruption (Phase 2):** Parse arithmetic dependency chains in each trace, surgically corrupt intermediate results with 5 conditions:
   - **A (propagated early):** Corrupt first operation, propagate error forward through chain
   - **B (propagated late):** Corrupt penultimate operation, propagate forward
   - **C (unpropagated):** Corrupt first operation *without* propagating (creates internal inconsistency)
   - **D (magnitude):** Propagated corruption with small (+/-2), medium (+/-15), or large (+/-75) offsets
   - **E (control):** Unmodified correct trace
3. **Teacher-forcing (Phase 3):** Feed corrupted trace token-by-token, then let model generate the final answer freely.
4. **Cross-model transfer (Phase 4):** Teacher-force one model's corrupted traces through the other model.

Compliance is measured as: does the model output the corrupted answer (following the text) or the correct answer (ignoring the text)?

#### Key Results

**Same-model compliance (model reads its own corrupted traces):**

| Condition | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| A (propagated early) | **94.4%** [92.5, 95.9] | 31.3% [14.2, 55.6] |
| B (propagated late) | **95.5%** [92.8, 97.2] | 40.0% [11.8, 76.9] |
| C (unpropagated) | **0.0%** [0.0, 0.5] | 0.0% [0.0, 22.8] |
| D small (+/-2) | **94.1%** [92.2, 95.6] | 26.3% [11.8, 48.8] |
| D medium (+/-15) | **94.4%** [92.5, 95.9] | 31.3% [14.2, 55.6] |
| D large (+/-75) | **94.2%** [92.0, 95.7] | 16.7% [4.7, 44.8] |
| E (control) | 0.0% | 0.0% |

Values in brackets are Wilson 95% CIs.

**Critical observations:**
- **Qwen3-8B is highly text-compliant** (~94% follows corrupted text) — it treats the reasoning trace as authoritative
- **Llama-3.1-8B is text-resistant** (~26-40% follows corrupted text) — it largely ignores the text and computes independently
- **Condition C (unpropagated) shows 0% compliance in all cases** — when the corruption creates an inconsistency within the trace, neither model follows it. This proves the models *can* detect errors; they choose not to when the trace is internally consistent.
- **No dose-response effect** — corruption magnitude (+/-2 vs +/-75) doesn't significantly affect compliance rates for either model

**Cross-model transfer:**

| Direction | Compliance (A condition) |
|---|---|
| Llama traces -> Qwen reads | **95.2%** [92.4, 97.0] |
| Qwen traces -> Llama reads | **22.6%** [11.4, 39.8] |

The asymmetry is striking: Qwen follows corrupted text regardless of source (~95%); Llama resists corrupted text regardless of source (~23%). This is a property of the *reader* model, not the trace.

**Important caveat:** Llama's n_valid counts are very low (5-19 out of 425 problems per condition). Most Llama outputs are classified as "other" (neither correct nor corrupted answer), meaning Llama often produces answers that match neither the true answer nor the corrupted answer. The compliance rates have wide confidence intervals.

#### Significance

This experiment reveals a spectrum of text-dependence across models:
- **Qwen3-8B** relies heavily on the visible reasoning text for its answer — corrupting the text corrupts the answer. This model's "hidden channel" (if it exists) is weak.
- **Llama-3.1-8B** computes largely independently of the visible text — corrupting the trace has minimal effect. This model appears to rely more on internal computation (hidden channel) than on the text.

The 0% compliance on unpropagated corruptions (Condition C) is particularly important: it rules out the "model can't do math" explanation. Both models detect that `3 x 5 = 20` is wrong when the next line says `20 + 10 = 30` but the final answer should be 25 (inconsistency). They only follow the corruption when it's self-consistent — suggesting they process the text as a coherent signal rather than blindly copying numbers.

---

### Synthesis

#### The emerging picture

1. **LLM output distributions have enormous unused capacity** (Exp. 1): 4-5 bits/token of Shannon entropy available beyond what the visible text requires. There is room for a hidden channel.

2. **CoT narrows the distribution** (Exp. 2): Per-token entropy drops 3x during reasoning (median near zero). Most tokens are deterministic — the "computation" can't be happening in the token choices themselves.

3. **The KV cache encodes precise answer-relevant state** (Exp. 3): A sharp SNR cliff at 14 dB shows digital-like fragility. Small noise destroys the answer even when token predictions are largely preserved.

4. **A large null space separates text from answer** (Exp. 4): Adversarial perturbations at 377x the signal norm change the answer while preserving every intermediate token. The gap between random noise threshold (0.006x) and adversarial threshold (377x) proves the null space exists.

5. **The null space has spatial structure** (Exp. 5): PGD perturbations concentrate at answer-coupled positions (rho = 0.78) and ignore text-coupled positions (rho = -0.06). The hidden channel occupies specific positions and layers, not the entire KV cache uniformly.

6. **Models vary in text-dependence** (Exp. 6): Qwen3-8B follows corrupted text at 94% compliance; Llama-3.1-8B resists at ~70%. This suggests the relative importance of the text channel vs the hidden channel varies across model families.

#### What remains untested

- **Cross-model replication of the KV adversarial experiments (Exps 3-5) on Llama** — currently Qwen-only
- **Double dissociation experiment** — pruning answer-coupled vs text-coupled KV positions under normal generation (no adversarial machinery). If pruning answer-coupled positions destroys accuracy but leaves coherent text, and pruning text-coupled positions preserves accuracy but produces incoherent text, this would be the most direct evidence for functional separability.
- **Dose-response curve** — scaling the adversarial perturbation delta by alpha in [-2, +2] to show precise information positioning in the null space
- **Layer ablation** — systematic per-layer perturbation to localize which layers carry answer vs text information
- **Connection to existing KV cache compression work** — testing whether answer-coupled positions are the ones retained by SnapKV, H2O, or Think Clearly

#### Implications

- **CoT unfaithfulness is architecturally guaranteed**: If the text is a lossy projection of the computation, it cannot faithfully represent all of it. Unfaithfulness is not a failure mode — it's a consequence of the autoregressive text bottleneck.
- **KV cache compression should preserve answer-coupled positions**: Methods like Think Clearly may work precisely because they prune text-scaffolding positions that don't carry answer information. Our answer-coupling metric could improve compression strategies.
- **Latent reasoning architectures are well-motivated**: Coconut, Quiet-STaR, and similar approaches remove the text bottleneck. Our results show this bottleneck actively constrains computation — the model would benefit from computing in latent space.
- **Monitoring hidden channels for safety**: If models can carry answer-relevant computation in a channel invisible to the text, this has implications for interpretability and oversight. The hidden channel is not detectable by reading the reasoning text.
