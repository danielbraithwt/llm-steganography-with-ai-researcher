# Literature Notes: Latent Reasoning Advances
**Cycle 50 — Literature Scan #5 — 2026-03-15**

## 1. Reasoning Beyond Language: A Comprehensive Survey on Latent CoT
**Chen et al. (May 2025, revised Nov 2025)**
**arXiv:** 2505.16782

Comprehensive taxonomy of latent chain-of-thought methods:
- **Token-wise horizontal approaches:** Compress reasoning steps into fewer latent tokens
- **Layer-wise vertical strategies:** Use depth/recurrence for computation without text
- Key insight: Latent reasoning decouples reasoning from explicit language generation, enabling "richer cognitive representations and faster inference"
- Applicable to "abstract reasoning tasks beyond language"

**Relevance:** The field now recognizes the text bottleneck as a fundamental limitation. Our mechanistic evidence (K-routing channel as the substrate of hidden computation) provides the "where" to complement the field's "what" (latent reasoning exists).

## 2. HybridCoT: Interleaving Latent and Text Chain-of-Thought
**Meta/Facebook Research (NeurIPS 2025)**
**OpenReview:** NRGRrHmq1H

Key innovation — interleaves latent tokens with text tokens:
- **94% of text-only CoT performance** with **1.97x less inference compute**
- Keeps critical text tokens (math operations) in context while compressing semantic reasoning to latent space
- In-context text-to-token distillation for explicit supervision
- Iterative parallelized latent rollout for training efficiency
- Surpasses LightThinker (1.36x) and StreamLLM (1.26x)

**Relevance — MAJOR:** HybridCoT's design implicitly validates our text/answer channel separation. Their approach keeps "critical text tokens" (math operations = our text-coupled positions) while compressing "semantic reasoning" (= our answer-coupled latent channel). The interleaving mirrors the KV cache's dual-channel structure: some positions carry text-critical information, others carry computation. HybridCoT works precisely because these channels are separable.

## 3. Latent Reasoning as Vocabulary-Space Superposition (Latent-SFT)
**Deng et al. (Oct 2025)**
**arXiv:** 2510.15522

Defines latent tokens as superpositions of vocabulary embeddings:
- z = Σ αᵢeᵢ (probability-weighted sum over vocabulary)
- **4x compression** of reasoning chains with comparable performance
- Key findings:
  - **Distribution mismatch:** Hidden states in LLM are "entirely inconsistent" with token embedding distribution — model operates in different space than text
  - **Low-rank structure:** Token embeddings occupy only a low-dimensional subspace despite high nominal dimensionality
  - **Multi-path superposition:** Latent reasoning maintains ~3-4 alternative reasoning paths simultaneously (Neff ≈ 3-4)
  - Soft embeddings outperform hidden-state representations

**Relevance — CRITICAL:** The distribution mismatch finding directly supports our hypothesis. If hidden states operate in a fundamentally different space than token embeddings, then the text (which is projected from token embeddings) CANNOT faithfully represent the computation (which happens in hidden states). This is the "lossy projection" our hypothesis describes. The multi-path superposition finding (3-4 simultaneous paths) also explains why per-token entropy is near zero during CoT: the model maintains multiple internal reasoning paths even when the text commits to a single deterministic sequence.

## 4. Beyond Speedup: KV Cache for Sampling and Reasoning
**Xing et al. (Jan 2026)**
**arXiv:** 2601.20326

Proposes using KV cache as reusable representation substrate:
- **Chain-of-Embedding:** KV-derived representations competitive with dedicated embeddings
- **Fast/slow thinking switching:** Dynamic adjustment of computation depth via KV manipulation
- **5.7x token generation reduction** with minimal accuracy loss

**Relevance:** Further evidence that the KV cache carries rich computational information beyond its text-generation role. The fast/slow thinking switching via KV manipulation directly demonstrates that reasoning depth is encoded in the KV cache.

## 5. KV Cache Steering for Controlling Frozen LLMs
**Belitsky et al. (July 2025)**
**arXiv:** 2507.08799

One-shot KV cache intervention induces reasoning styles:
- Steering vectors derived from contrastive CoT vs non-CoT prompts
- Applied as one-shot intervention to KV cache at inference time
- Induces stepwise, causal, analogical reasoning styles
- Improved both qualitative reasoning structure and quantitative performance
- More efficient than continuous activation steering (reduced latency, better stability)

**Relevance:** KV cache steering demonstrates that reasoning-relevant information is encoded in the KV cache (not just in the text). The ability to change reasoning STYLE via KV manipulation without changing text mirrors our PGD finding — the KV cache carries information separable from the text channel. This is essentially a constructive (beneficial) version of our adversarial (PGD) manipulation.

---

## Synthesis: Latent Reasoning Goes Mainstream

The field has moved decisively toward accepting that text is not the computation:

| Approach | Compression | Key Innovation | Venue |
|----------|------------|----------------|-------|
| HybridCoT | 1.97x compute | Interleaved latent/text tokens | NeurIPS 2025 |
| Latent-SFT | 4x chain length | Vocabulary-space superposition | arXiv Oct 2025 |
| KV-Embedding | 5.7x tokens | KV cache as representation substrate | arXiv Jan 2026 |
| KV Steering | N/A | Reasoning style via KV manipulation | arXiv Jul 2025 |

**Our unique contribution in this landscape:** Everyone knows latent reasoning exists and can be exploited. But we provide the mechanistic answer to WHERE it lives: the K-routing channel of the KV cache, concentrated in specific "answer heads" (H5 primary), operating position-independently throughout the reasoning chain. The practical implication: latent reasoning architectures should preserve K-routing fidelity while allowing V-content and text-scaffolding positions to be compressed.
