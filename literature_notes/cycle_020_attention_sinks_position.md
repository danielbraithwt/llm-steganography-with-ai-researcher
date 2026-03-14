# Attention Sinks and Position-Dependent Encoding

**Cycle:** 20
**Date:** 2026-03-14
**Papers reviewed:**
- (2025). "When Attention Sink Emerges." ICLR 2025.
- (2024/2025). "Efficient Streaming Language Models with Attention Sinks." ICLR 2024 + follow-up work.
- Belitsky et al. (2025). "KV Cache Steering for Controlling Frozen LLMs." arXiv:2507.08799

---

## 1. Attention Sinks (ICLR 2025)

### Key Finding
Attention sinks are the emergence of strong attention scores toward initial tokens, regardless of semantic content. These serve as a **"no-op" channel** that restricts mixing and preserves local sensitivity.

### Mechanism
- Initial tokens receive disproportionate attention as a mathematical necessity of softmax normalization
- The sink absorbs "excess" attention probability that would otherwise distribute randomly
- Removing or misplacing sink tokens causes **severe performance degradation**
- Magnitude scales with context length, model depth, and data packing

### Relevance to Our Research
**DIRECTLY explains several of our findings:**

1. **Why early positions are critical on ALL models (Q48)**: Our exps 013, 016, 017 all show POS_EARLY << POS_LATE. Destroying early positions removes attention sinks, disrupting the attention routing infrastructure that ALL subsequent computation depends on. This is NOT about answer-specific information at early positions — it's about structural requirements of softmax attention.

2. **Why layer 0 is uniquely critical on Qwen (Q18)**: Layer 0 processes the raw input and generates the initial KV cache that subsequent layers build upon. If layer 0 KV is destroyed, the attention sinks are removed from the very foundation, and all subsequent layers lose their routing infrastructure.

3. **Why position dominates answer accuracy universally**: The early-position sensitivity is an architectural property of transformers (softmax + causal masking), not a learned property of specific models. This explains why it appears on ALL models regardless of training regime (Qwen-Base, Qwen-Instruct, Llama-Instruct).

4. **The positional confound in our experiments**: When we noise "SelAC" positions (which happen to be early on some models and late on others), the effect is primarily driven by whether we're destroying attention sinks (early positions) or not. The selectivity label is secondary to the positional role.

### Implications for Experimental Design
Any future experiment comparing position importance must control for the attention sink effect. Specific recommendations:
- Exclude the first N tokens (where N is the number of attention sinks, typically 2-4) from any AC/TC/selectivity analysis
- When doing position-controlled noise injection, ensure both conditions are above the attention sink range
- The attention sink tokens should be treated as infrastructure, not as part of the text/answer channel analysis

---

## 2. KV Cache Steering — Updated Understanding (Belitsky et al., 2025)

### Key Finding (Update from Cycle 10)
One-shot KV cache intervention outperforms continuous activation steering in:
- Inference latency
- Hyperparameter stability
- Ease of API integration
- Works across model scales (small to large)

### New Understanding
Cache steering can induce **controllable transfer of reasoning styles** (stepwise, causal, analogical). This demonstrates that the KV cache doesn't just carry information — it carries **procedural instructions** for how to reason.

### Relevance
1. **Constructive complement to our adversarial work**: We show the KV cache has a null space where adversarial perturbation changes answers. They show the KV cache has a "steering space" where constructive perturbation changes reasoning style. Together: the KV cache encodes both WHAT (answer) and HOW (reasoning procedure) in separable components.

2. **"Procedural" channel**: Beyond our text/answer dichotomy, there may be a third channel: procedural/meta-reasoning information that controls the reasoning approach. This could explain why some positions are "hubs" — they may carry procedural instructions, not just text or answer information.

3. **One-shot intervention works**: This means a single modification to the KV cache persists through the entire generation. Consistent with our finding that prompt-only PGD attacks can change the answer — the perturbation cascades forward through the computation.

---

## Synthesis: Position as Infrastructure

The literature on attention sinks provides a crucial reframing of our position-dominance finding:

**Early positions serve as computational infrastructure (attention sinks, routing channels), not as content carriers.** Their importance for answer accuracy is a side effect of their structural role — destroy the infrastructure and everything built on it collapses.

This means:
1. The "position dominates" finding is NOT evidence against functional separability — it's evidence of a **third category** beyond text and answer: **infrastructure**.
2. Our selectivity analysis should exclude infrastructure positions (sinks) to reveal genuine text/answer separation in the remaining positions.
3. The within-late-half selectivity effect on Qwen-Base (42.1pp gap in exp 017) may represent genuine functional separation AFTER controlling for infrastructure effects.
4. Future experiments should explicitly partition positions into: (a) infrastructure/sinks (first few tokens), (b) reasoning positions, and (c) answer-relevant positions.
