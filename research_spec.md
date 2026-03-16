# Research Specification

## Hypothesis
Chain-of-thought (CoT) reasoning text is a **lossy projection** of the model's internal computation. The KV cache carries a functionally separable hidden channel that encodes answer-relevant information independent of the visible reasoning tokens. The model computes through high-dimensional hidden representations and is constrained to emit plausible tokens as a byproduct of the autoregressive architecture, but the text itself is not the computation.

## Background
Recent work on chain-of-thought prompting has shown that LLMs can solve complex reasoning problems by generating intermediate steps. However, the relationship between the visible reasoning text and the model's internal computation is poorly understood. Several lines of evidence suggest the text may not faithfully represent the computation: models sometimes arrive at correct answers despite flawed reasoning steps, and the KV cache carries far more information than is needed to produce the visible tokens.

This investigation probes the mechanistic relationship between visible reasoning tokens and the hidden states that produce them. If the KV cache carries a separable hidden channel, this has implications for interpretability (we can't trust the text), safety (models can compute things we can't see), and efficiency (latent reasoning architectures remove an unnecessary bottleneck).

**Critical gap (as of cycle 64):** All causal evidence for the hidden channel comes from **artificial perturbation** — injecting noise, adversarial PGD optimization, teacher-forcing corrupted text. No experiment has demonstrated the channel being used during **normal, unperturbed generation**. A skeptical reviewer can dismiss all perturbation results with: "you broke the KV cache and things broke — of course. The model is just reading its own text." The next phase of research must close this gap.

---

## Phase 1: Established Results (Cycles 1–64)

All evidence below comes from perturbation-based experiments across 5 model variants (Qwen3-4B-Base, Qwen3-4B-Instruct, Llama-3.1-8B-Instruct, Phi-3.5-mini-Instruct, Mistral-7B-v0.3). Full details in `evidence_ledger.md`.

| # | Finding | Strength | Summary |
|---|---------|----------|---------|
| 1 | **Text-answer dissociation** | Decisive (5 models) | KV cache perturbation destroys answer accuracy while preserving text prediction quality (text ≥98% at near-zero accuracy). The two channels are functionally separable. |
| 2 | **K > V universal hierarchy** | Decisive (5 models, independent geometric + quantization confirmation) | K-vectors (routing) carry functionally more critical information than V-vectors (content) for answer computation. V-only perturbation at moderate levels has literally zero effect (456/456 per-head). Reflects QK-routing vs OV-content split. |
| 3 | **Encoding taxonomy (digital vs analog)** | Strong (5 models) | Qwen uses digital K-encoding (sharp accuracy cliffs); Llama/Phi/Mistral use analog (gradual degradation). Digital is Qwen-family-specific. Instruction tuning converts V digital→analog but preserves K digital. |
| 4 | **Answer heads (H0, H5)** | Decisive (2 models) | Head 5 is the primary answer-routing head on BOTH Qwen and Llama (same KV head index). Destroying H0+H5 (25% capacity) reduces Qwen accuracy to 3.7% while dispensable pairs preserve 96-100%. Two-regime redundancy curve is Qwen-specific. |
| 5 | **Spatial structure: positional > selectivity** | Decisive | Late positions carry answer-specific information; early positions are computational infrastructure (attention sinks). AC/TC selectivity adds ZERO explanatory power beyond position (gold-standard double dissociation). Original PGD rho=0.78 was inflated; actual rho=0.20. |
| 6 | **Cross-model variation in text-dependence** | Moderate | Qwen3-8B follows corrupted text at 94% compliance; Llama-3.1-8B resists at ~30%. The hidden channel's relative importance varies by architecture. |

### Additional established results
- KV cache compression: "Recent" (sinks + most recent) achieves 100% at all budgets (25-75%) on both models — the only universally robust eviction strategy (134/134 pooled)
- K spectral geometry: K has lower effective rank than V (ratio 0.87-0.94), K top-1 energy 3.5-4.1x higher — geometric confirmation of K>V hierarchy (non-perturbation)
- Head × position interaction is encoding-dependent: digital (Qwen) → orthogonal; analog (Llama) → strongly coupled (r=-0.991)
- Literature convergence: 60+ papers across 6 scans show independent convergence from 10+ angles (architectural, behavioral, safety, computational, mechanistic, compression)

### Key limitation of Phase 1
**Every causal experiment perturbed the KV cache.** We showed that destroying K-routing destroys answers (interventionist evidence), but we never showed that K-routing carries answer information during normal generation (observational evidence). The perturbation experiments prove the channel CAN carry information but not that it DOES carry information under natural conditions.

---

## Phase 2: Natural Channel Usage (Current Focus)

### Motivation
The Phase 1 evidence establishes that the KV cache has the capacity and structure for a hidden channel. But capacity ≠ usage. A building can have a secret room, but that doesn't prove anyone uses it. We need to show the hidden channel is actively used during normal, unperturbed generation.

This is the critical gap for a publishable paper. The three strongest forms of natural evidence are:

1. **Information in the KV cache that isn't in the text** — if a linear probe on K-vectors can decode the final answer at intermediate positions before the text contains it, the channel carries information beyond what the text provides.
2. **Text manipulation that shouldn't matter but does** — if paraphrasing the CoT (same math, different surface tokens) degrades accuracy, the original token choices carried hidden information in their K-routing patterns.
3. **Correct answers despite wrong text** — if the model writes incorrect arithmetic but gets the right answer, and the K-vectors at error positions encode the correct value, the model is routing through the hidden channel rather than reading its own text.

### Evidence Targets

**What would constitute evidence FOR natural channel usage:**
- Linear probe on K-cache at intermediate positions decodes the final answer significantly earlier than a text-only baseline (early decodability gap ≥ 5 positions)
- Paraphrasing CoT text (preserving semantic content, changing surface tokens) degrades accuracy by ≥10% compared to original text
- Naturally occurring wrong-reasoning-right-answer cases exist at ≥1% rate, AND the K-vectors at error positions encode the correct intermediate value (probe accuracy > chance)
- Any combination of the above that survives controls

**What would constitute evidence AGAINST natural channel usage:**
- Linear probe cannot decode the answer any earlier than a text-only baseline (the text IS the computation)
- Paraphrasing has no effect on accuracy (the model reads the semantic content, not the specific tokens)
- Wrong-reasoning-right-answer cases don't exist, OR exist but K-vectors at error positions don't encode correct values (errors are genuine mistakes, not hidden computation)
- All three experiments null → the hidden channel has capacity (Phase 1) but the model doesn't use it during normal generation

### Experiment Directions

#### Experiment A: Early Answer Decodability (Linear Probe)

**Core question:** Does the KV cache know the answer before the text reveals it?

**Method:**
1. Generate 8-shot CoT on GSM8K (100+ problems), collecting the full KV cache at every token position
2. Train a linear probe (logistic regression or single-layer MLP) on K-cache activations at each position to predict the final numeric answer (or answer correctness)
3. For the text-only baseline: train the same probe architecture on token embeddings (what the text "says") at each position
4. Plot probe accuracy as a function of position (normalized to chain length)
5. The **early decodability gap** = how many positions earlier the K-probe reaches X% accuracy compared to the text-baseline probe

**Key controls:**
- Shuffle control: train probe on K-cache from problem A paired with answer from problem B → should be at chance
- Layer sweep: probe at each layer to find where answer information emerges
- Compare K-probe vs V-probe (expect K to be earlier/more accurate, consistent with Phase 1)

**Predicted outcome if hypothesis is true:** K-probe decodes the answer 10-30% of chain positions earlier than text-baseline. The gap is largest in middle layers (where routing is established but text hasn't revealed the answer yet).

**Predicted outcome if hypothesis is false:** K-probe and text-baseline track together. The KV cache learns the answer at the same rate as the text because it IS reading the text.

#### Experiment B: Paraphrase Disruption

**Core question:** Do the specific token choices in CoT carry hidden information beyond their semantic content?

**Method:**
1. Generate CoT traces for 100+ GSM8K problems (collect both text and final answer)
2. Paraphrase each trace: preserve all numbers, mathematical operations, and logical structure, but change the surface text (e.g., "She has 5 apples and buys 3 more" → "The count of apples she possesses is 5, and she acquires an additional 3")
3. Use an LLM to generate paraphrases (GPT-4 or Claude via API), constrained to preserve all numeric values and operations
4. Feed original traces through the model via teacher-forcing, let it generate the final answer → accuracy_original
5. Feed paraphrased traces through the model via teacher-forcing, let it generate the final answer → accuracy_paraphrased
6. Accuracy drop = accuracy_original - accuracy_paraphrased

**Key controls:**
- Verify paraphrases are semantically equivalent (same numbers, same operations, same logical flow)
- Number-preservation check: paraphrase should preserve all numeric tokens exactly
- Baseline: teacher-force the SAME trace but with random token replacements for non-numeric tokens → expected large accuracy drop (shows teacher-forcing setup works)
- Cross-model: if paraphrase disruption exists on Qwen (text-compliant model), does it also exist on Llama (text-resistant model)?

**Predicted outcome if hypothesis is true:** Accuracy drops ≥10% under paraphrase, because the model's K-routing patterns were tuned to the specific original tokens. Different tokens = different K-vectors = disrupted routing. Effect should be larger on Qwen (text-compliant, 94% follows corrupted text) than Llama (text-resistant, ~30%).

**Predicted outcome if hypothesis is false:** Accuracy is unchanged under paraphrase, because the model reads the semantic content (numbers and operations) not the specific token choices. Paraphrase is just a different way of saying the same thing.

#### Experiment C: Wrong Reasoning, Right Answer

**Core question:** When the model writes wrong arithmetic but gets the right answer, is the correct computation happening in the hidden channel?

**Method:**
1. Generate CoT traces for ALL 1,319 GSM8K problems
2. Parse each trace to identify arithmetic operations and their results
3. Find cases where the model writes an incorrect intermediate result but arrives at the correct final answer (the "wrong-reasoning-right-answer" or WRRA cases)
4. For each WRRA case, extract K-cache activations at the position of the arithmetic error
5. Train a linear probe on K-cache at error positions to predict the CORRECT intermediate value (not the written wrong value)
6. Compare: does the K-probe predict the correct value or the written-wrong value?

**Key controls:**
- Correct-reasoning cases: at positions where arithmetic is correct, probe should predict the written (=correct) value with high accuracy
- V-probe comparison: does V also carry the correct value, or only K? (Phase 1 predicts K carries answer info, V less so)
- Random position control: probe at non-arithmetic positions → should not predict intermediate values
- Rate estimation: what fraction of problems have WRRA cases? (Literature suggests models write wrong arithmetic at 5-15% of steps)

**Predicted outcome if hypothesis is true:** At WRRA positions, the K-probe predicts the CORRECT value with above-chance accuracy despite the text saying the wrong value. The model "knew" the right answer but wrote the wrong one — the hidden channel carried the correct computation.

**Predicted outcome if hypothesis is false:** At WRRA positions, the K-probe predicts the WRITTEN (wrong) value, or is at chance. The KV cache doesn't have the correct value — the model genuinely made a mistake and got lucky with the final answer through error cancellation or other means.

### Priority Order
1. **Experiment A (linear probe)** — most direct evidence, cleanest methodology, no text manipulation needed
2. **Experiment C (WRRA)** — strongest "smoking gun" if cases exist, but depends on finding naturally occurring errors
3. **Experiment B (paraphrase)** — interesting but harder to control (paraphrase quality, semantic equivalence)

---

## Scope & Constraints
- **Models:** Qwen3-4B-Base (primary), Llama-3.1-8B-Instruct (replication)
- **Benchmark:** GSM8K (8-shot chain-of-thought)
- **GPU is available and SHOULD be used** for all model inference and probe training
- Compute budget: 48 hours total
- Build on existing infrastructure in `scripts/` — reuse model loading, GSM8K evaluation, KV cache extraction code from Phase 1 experiments
- What is OUT of scope: training or fine-tuning the base models, multi-GPU experiments, proprietary/API-only models

---

## Key Literature
- Lanham et al. (2023) "Measuring Faithfulness in Chain-of-Thought Reasoning": establishes that CoT is not always faithful
- Turpin et al. (2023) "Language Models Don't Always Say What They Think": demonstrates systematic unfaithfulness in CoT
- Pfau et al. (2024) "Let's Think Dot by Dot": shows models can reason with filler tokens, suggesting computation happens in hidden states
- Coconut (Hao et al. 2024): latent reasoning without text tokens — relevant to the "text bottleneck" framing
- Quiet-STaR (Zelikman et al. 2024): internal reasoning tokens — shows benefit of removing text constraint
- **Boppana et al. (2026) "Reasoning Theater"**: models decide answers internally 80% of tokens before visible CoT — directly motivates Experiment A
- **Godey & Artzi (March 2026)**: 95-99% gradient norm suppressed by LM head — text is architecturally guaranteed lossy projection
- McCormick (2025) "Patterns/Messages": QK=routing, OV=content framework — grounds our K>V finding
- SnapKV, H2O, StreamingLLM: KV cache compression methods — our "recent" finding extends this literature
- **Cabannes et al. (NeurIPS 2024)**: iteration heads carry forward CoT computation — relevant to answer head findings
- **CODI (EMNLP 2025), Latent-SFT, HybridCoT (NeurIPS 2025)**: latent reasoning architectures achieve near-CoT performance at fraction of compute — validates the text bottleneck framing
- **Chen (2025, Anthropic), Baker (2025, OpenAI)**: <20% verbalization of reasoning; obfuscated reward hacking — safety implications of hidden computation
