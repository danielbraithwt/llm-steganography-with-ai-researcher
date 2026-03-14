# Causal Bypass in Large Language Models

**Paper:** "When Chains of Thought Don't Matter: Causal Bypass in Large Language Models"
**Authors:** Sathyanarayanan, Nagarsekar, Rathore (BITS Pilani)
**Date:** February 2026, arXiv:2602.03994
**Scanned:** Cycle 10, 2026-03-14

## Key Concepts

**CoT Mediation Index (CMI):** A bounded metric (0-1) that isolates CoT-specific causal influence by comparing performance degradation from patching CoT-token hidden states against matched control patches. Uses activation patching at the hidden state level.

**Bypass Regimes:** Conditions where CMI ≈ 0 despite plausible CoT text output. The model generates fluent rationales while routing decision-critical computation through latent pathways.

**Reasoning Windows:** Localized depth ranges showing concentrated CoT-specific influence. CoT influence is "depth-localized into narrow reasoning windows" rather than distributed uniformly across layers.

## Models Tested

11 models across multiple families: Phi (1.5, 2, 3.5-mini, 4, 4-mini-reasoning, mini-MoE), Qwen (3-0.6B, 1.7B, 4B, 8B), DialoGPT-large.

## Tasks

StrategyQA, TruthfulQA, GSM8K

## Key Results

- **TruthfulQA:** Near-total bypass regime across almost all instances (CMI ≈ 0, Bypass ≈ 1.0)
- **GSM8K:** 5/20 instances showed pure bypass (CMI = 0)
- **MoE models** have more distributed CMI profiles (consistent with routing mechanisms)
- **Dense transformers** show narrow "reasoning windows"
- **Reasoning-tuned models** (Phi-4-mini-reasoning) showed stronger CMI than larger untuned counterparts

## Relevance to Our Research

**HIGHLY RELEVANT — Direct independent validation of our core hypothesis.**

1. **Bypass = our hidden channel:** The "bypass regime" where CoT text is generated but computation flows through latent pathways is exactly what our KV cache perturbation experiments demonstrate mechanistically. Our PGD null space (Exp 4) shows the same thing from a different angle — perturbations that change the answer but not the text prove the answer computation bypasses the text channel.

2. **Reasoning windows = our layer concentration:** Their finding that CoT influence is "depth-localized" parallels our Exp 009 finding that layer-level sensitivity is concentrated (Qwen: layer 0 critical, all others 100%).

3. **Architecture-dependent bypass:** MoE models show more distributed mediation, while dense models show narrow windows. This parallels our finding that Llama (GQA, larger) shows different encoding than Qwen (MHA, smaller) — the information routing architecture affects where computation happens.

4. **Methodology difference:** They use activation patching on hidden states; we use adversarial perturbation of KV cache. Both converge on the same conclusion: CoT text is not causally necessary for answer computation.

5. **Open question connection:** Their CMI metric could be adapted to measure our "text-coupling" vs "answer-coupling" — a position with high CMI is one where CoT text causally matters (text-coupled), while low CMI positions route through the bypass (answer-coupled).

## Impact on Evidence

- **Strengthens:** "CoT unfaithfulness is architecturally guaranteed" interpretation
- **Supports:** Architecture-specific differences in computation routing
- **New method:** CMI could be used alongside our PGD/noise approaches for converging evidence
- **Challenges:** Nothing directly challenges our findings; this is independent convergent evidence
