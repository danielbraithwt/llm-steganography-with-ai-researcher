# Latent Reasoning and KV Cache as Computation Medium

**Papers:**
1. "Training LLMs to Reason in Continuous Latent Space" (Coconut) — Hao et al. (Meta, Dec 2024, arXiv:2412.06769)
2. "A Survey on Latent Reasoning" — (July 2025, arXiv:2507.06203)
3. "Bottlenecked Transformers: Periodic KV Cache Consolidation for Generalised Reasoning" (May 2025, arXiv:2505.16950)
4. "KV Cache Steering for Controlling Frozen LLMs" — Belitsky et al. (July 2025, arXiv:2507.08799)
5. "Beyond Speedup: Utilizing KV Cache for Sampling and Reasoning" — Xing et al. (Jan 2026, arXiv:2601.20326)

**Scanned:** Cycle 10, 2026-03-14

## Key Findings

### Coconut (Meta, 2024)
- Trains LLMs to reason in continuous latent space by feeding last hidden state directly as next input embedding
- Eliminates the "text bottleneck" — reasoning happens in continuous space, not constrained to token vocabulary
- Enables **breadth-first search** reasoning (continuous thoughts encode multiple alternatives) vs CoT's depth-first
- Outperforms CoT on logical reasoning tasks requiring planning/search
- Multi-stage curriculum: gradually replaces language steps with latent states

### Survey on Latent Reasoning (2025)
Categorizes approaches into:
1. **Activation-based recurrence** — using internal activations for iterative processing
2. **Hidden state propagation** — leveraging continuous representations across layers
3. **Fine-tuning strategies** — compressing or internalizing explicit reasoning traces

Key insight: "Latent reasoning tackles the bottleneck by performing multi-step inference entirely in the model's continuous hidden state, eliminating token-level supervision." Natural language has "limited expressive bandwidth" compared to continuous hidden states.

### Bottlenecked Transformers (2025)
- Uses **Information Bottleneck theory**: generalization emerges from balancing input compression with predictive information retention
- Introduces Cache Processor: auxiliary transformer performing "periodic, non-causal, in-place KV rewrites" at reasoning step boundaries
- Consolidation mimics neuroscience: stabilizing new traces + reconsolidating old ones
- +6.6pp on selected math reasoning tasks
- Directly modifies KV cache during reasoning — showing the cache IS a computation medium

### KV Cache Steering (Belitsky et al. 2025)
- Extracts steering vectors from contrastive prompts (with/without reasoning)
- Modifies KV cache once after prefilling: K' = K + c^k * S^k, V' = V + c^v * S^v
- **One-shot intervention** (vs activation steering's per-step intervention)
- Consistently improves reasoning: GSM8K +0.5-4pp, ARC-Challenge +1-5pp, MATH +7.4pp
- **Style transfer experiments:** Cache steering induces distinct reasoning styles — showing semantic information is encoded and manipulable in KV representations
- Tested on: SmolLM2-360M through Llama-3.1-70B

### KV Cache for Reasoning (Xing et al. 2026)
- Uses KV cache as lightweight representation for task difficulty scoring
- KV cache encodes "sufficient contextual information for local comparisons"
- Enables adaptive fast/slow thinking switching: 5.7x token reduction with minimal accuracy loss
- KV representations work best for "relative ordering within a small candidate set"

## Relevance to Our Research

**EXTREMELY RELEVANT — These papers validate our core framing from multiple directions.**

1. **Coconut validates the "text bottleneck" hypothesis:** Our research shows the KV cache carries information beyond what the text projects — Coconut explicitly removes the text bottleneck and shows reasoning IMPROVES. This is the strongest theoretical validation of our hypothesis: if text constrains reasoning, then removing the constraint should help — and it does.

2. **KV Cache Steering proves separable information channels:** Belitsky et al. show that you can modify KV cache representations to change reasoning STYLE without changing the problem. This is our "null space" from a different angle — they modify the KV cache to change behavior while preserving the task, we modify it to change the answer while preserving the text.

3. **Bottlenecked Transformers use the KV cache AS a computation medium:** The Cache Processor that rewrites KV entries during reasoning directly treats the KV cache as a workspace for computation, not just a memory buffer. This aligns with our evidence that answer-relevant information is actively encoded at specific KV positions.

4. **Information Bottleneck theory provides our theoretical framework:** The Bottlenecked Transformers paper's use of IB theory (compression + relevance) maps directly onto our findings: the "text channel" is a compressed projection (high compression, limited bandwidth), while the "hidden channel" retains predictive information not captured in the text.

5. **Latent reasoning survey confirms our research sits at a frontier:** Our experimental approach (probing the KV cache to understand internal computation) is at the intersection of multiple active research directions. The survey's taxonomy positions our work alongside Coconut, Quiet-STaR, and other latent reasoning approaches.

6. **Task difficulty from KV cache (Xing et al.):** The finding that KV cache representations encode sufficient information for difficulty classification supports our claim that the KV cache carries task-relevant information beyond what's visible in the text.

## Impact on Evidence

- **Strongly validates:** The "text is a lossy projection" framing is now mainstream in the latent reasoning literature
- **Theoretical grounding:** Information Bottleneck theory provides the formal framework we need
- **Extends our contribution:** Our work uniquely provides MECHANISTIC evidence (spatial structure, adversarial null space) for what Coconut/Quiet-STaR show from a training perspective
- **New experiment idea:** Apply KV cache steering vectors to induce/suppress reasoning and measure how this affects our AC/TC position structure
