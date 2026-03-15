# Mechanistic Evidence for CoT Faithfulness Decay

**Cycle:** 40
**Date:** 2026-03-15
**Papers reviewed:**
- Ye, Loffgren, Kotadia & Wong (February 2026). "Mechanistic Evidence for Faithfulness Decay in Chain-of-Thought Reasoning." arXiv:2602.11201
- Chen, Plaat & van Stein (July 2025). "How does Chain of Thought Think? Mechanistic Interpretability of Chain-of-Thought Reasoning with Sparse Autoencoding." arXiv:2507.22928
- Arcuschin, Janiak, Krzyzanowski, Rajamanoharan, Nanda & Conmy (March 2025, revised June 2025). "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful." arXiv:2503.08679

---

## 1. Reasoning Horizon and Faithfulness Decay (Ye et al., February 2026)

### Key Innovation: The Reasoning Horizon (k*)

A consistent **Reasoning Horizon** exists at **70-85% of chain length**, beyond which reasoning tokens have little or negative causal effect on the final answer. This was measured using a new metric: **Normalized Logit Difference Decay (NLDD)**.

NLDD quantifies step-level faithfulness by corrupting individual reasoning steps and measuring confidence drops:
- **Positive NLDD** = model causally relies on that reasoning step
- **Negative NLDD** = corrupting the step paradoxically IMPROVES performance

### Model-Specific Faithfulness

| Model | NLDD (PrOntoQA) | NLDD (GSM8K) | Regime |
|-------|-----------------|---------------|--------|
| DeepSeek | +84.3% | +96.7% | **Faithful** — genuinely depends on reasoning chain |
| Llama | +high | +high | **Faithful** |
| Gemma | **-52.5%** | — | **Anti-faithful** — reasoning HURTS performance |

**Critical finding:** Gemma achieves 99% accuracy on PrOntoQA while having NEGATIVE faithfulness. Corrupting the reasoning chain IMPROVES Gemma's answers. The model has already computed the answer internally and the reasoning text actively interferes.

### The Mapping Gap

Models can encode task-relevant information in hidden states without utilizing it for predictions. Gemma achieves 82% linear probe accuracy on stack-depth information but 0% task accuracy — "the task-relevant structure is encoded but not utilized."

### Position-Specific Importance

Beyond the reasoning horizon, RSA (Representational Similarity Analysis) remains stable while NLDD drops sharply. The model maintains geometric consistency internally without using those representations for output — "a representational echo without causal force."

### Relevance to Our Research

**STRONG CONVERGENCE with our positional findings:**
- The Reasoning Horizon at 70-85% maps directly onto our finding that **late positions (last 5%) carry answer-specific information** while earlier positions serve as infrastructure
- Gemma's anti-faithfulness (negative NLDD) is the behavioral analogue of our **text-answer dissociation**: the model computes answers through hidden states independent of the visible text
- The "Mapping Gap" (information encoded but not used) parallels our **null space**: the KV cache carries information invisible to text predictions
- The "representational echo" concept matches our finding that **V-only perturbation has zero effect** — the representations persist but don't causally influence the answer when routing (K) is intact

**New insight for our framework:** The Reasoning Horizon suggests that our "late positions carry answer information" finding may be even more specific — positions beyond 70-85% of the chain may be where the model transitions from reasoning to answer retrieval. This is testable: apply our K/V perturbation at the reasoning horizon vs. before/after.

---

## 2. Mechanistic Interpretability of CoT via Sparse Autoencoders (Chen et al., July 2025)

### Key Findings

Using SAEs + activation patching on Pythia models (70M and 2.8B):
- Transferring CoT-reasoning features into a non-CoT run significantly improved answer confidence in 2.8B (from 1.2 to 4.3) but **NOT in 70M**
- Larger models show higher activation sparsity and feature interpretability — more modular computation
- Useful CoT information is **distributed widely** rather than concentrated in top features

### Critical Finding: Scale Threshold

There exists a model-size threshold below which CoT doesn't create genuine mechanistic features. Above this threshold, CoT induces interpretable internal structures that causally influence answers.

### Relevance

1. **Distributed CoT features** align with our finding that the hidden channel is **distributed** (not spatially concentrated, rho=0.20). The answer-relevant computation is spread across many features/positions.
2. The **scale threshold** suggests our experiments on 4B-8B models are in the regime where CoT creates genuine features. Smaller models might not show the same text-answer dissociation.
3. Feature patching from CoT→noCoT is conceptually related to our K-only PGD: both identify which internal representations causally determine the answer independent of the text.

---

## 3. CoT Unfaithfulness In The Wild (Arcuschin et al., March 2025)

### Key Innovation

Demonstrates CoT unfaithfulness under **natural conditions** (no artificial biases or hint injection). Models produce contradictory answers to paired questions while generating plausible-sounding justifications for both.

### Unfaithfulness Rates

| Model | Unfaithfulness rate |
|-------|-------------------|
| GPT-4o-mini | 13% |
| Haiku 3.5 | 7% |
| Gemini 2.5 Flash | 2.17% |
| ChatGPT-4o | 0.49% |
| DeepSeek R1 | 0.37% |
| Gemini 2.5 Pro | 0.14% |
| Sonnet 3.7 (thinking) | 0.04% |

### Relevance

1. Unfaithfulness occurs naturally, not just under adversarial conditions — consistent with our view that unfaithfulness is architecturally guaranteed (LM head gradient bottleneck)
2. The variation across models (13% to 0.04%) parallels our cross-model variation in text-dependence (Qwen 94% compliant vs Llama ~30%)
3. Models with extended thinking (Sonnet 3.7) show lowest unfaithfulness — more tokens of reasoning may better align text with computation

---

## Synthesis: Mechanistic Faithfulness Landscape

These three papers collectively establish:

1. **Faithfulness decays with chain position** — a Reasoning Horizon at 70-85% beyond which text tokens don't influence the answer (Ye et al.)
2. **Anti-faithfulness exists** — some models compute answers that are HURT by their own reasoning text (Gemma's -52.5% NLDD)
3. **Faithfulness is scale-dependent** — CoT creates genuine mechanistic features only above a model-size threshold (Chen et al.)
4. **Unfaithfulness occurs naturally** — no adversarial setup needed; rates vary 300x across models (Arcuschin et al.)

**Our unique contribution:** These papers study faithfulness at the behavioral/feature level. Our KV cache perturbation experiments identify the MECHANISTIC SUBSTRATE: the K-routing channel carries answer computation through directions invisible to text prediction. The Reasoning Horizon likely corresponds to the transition point where answer-coupled K-routing becomes dominant over text-coupled patterns.
