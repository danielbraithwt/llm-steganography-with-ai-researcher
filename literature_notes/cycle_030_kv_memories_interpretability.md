# KV Memories, Interpretability, and Mechanistic Advances

**Cycle:** 30
**Date:** 2026-03-14
**Papers reviewed:**
- Ye et al. (NeurIPS 2025). "Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders." OpenReview.
- Arora et al. (May 2025, revised January 2026). "Mechanistic Evaluation of Transformers and State Space Models." arXiv:2505.15105
- Anthropic (2025). "Sparse Attention Post-Training for Mechanistic Interpretability." arXiv:2512.05865

---

## 1. KV Memories ≈ SAE Features (Ye et al., NeurIPS 2025)

### Key Finding

Feed-forward (FF) layers, when treated as "key-value memories" (keys = first linear layer weights, values = second linear layer weights), are **nearly as interpretable as sparse autoencoder features**. SAEs showed only "an observable but minimal improvement in some aspects." In certain evaluation dimensions, vanilla FF layers actually outperformed SAEs.

### Critical Detail

Features discovered by SAEs and FFs **diverged** — they extract different representations. This means there isn't one "true" feature decomposition; different methods reveal different structures.

### Relevance to Our Research

1. **Validates K-V as organizational principle:** The fact that FF layers' K-V structure is interpretable validates the idea that K-V organization is a fundamental computational principle in transformers, not just an architectural convenience. Our KV cache perturbation experiments probe the ATTENTION K-V cache (a different mechanism from FF K-V memories), but the shared K-V organizational principle suggests deep structural parallels.

2. **FF keys as routing, FF values as content:** In FF layers, the key (first linear) determines WHETHER a "memory" is activated (routing), and the value (second linear) determines WHAT information is transmitted (content). This is structurally parallel to attention K-V: attention K determines WHERE to attend (routing), attention V determines WHAT flows through (content). Our K > V finding may extend to FF layers as well.

3. **Implications for our null space:** If FF K-V memories are interpretable, the null space in attention KV cache might also be interpretable. The directions that PGD finds to change the answer while preserving text might correspond to specific interpretable features — not just arbitrary high-dimensional directions.

---

## 2. Mechanistic Evaluation of Architectures (Arora et al., 2025-2026)

### Key Findings

- Transformers and the Based SSM learn to store key-value associations using induction, where "association is computed and stored at the value before retrieval"
- A single attention head can implement a "position-independent and generalizing notion of association"
- Mamba (SSM) implements induction NOT via its SSM but via short convolutions — different architectures achieve same behavior through different mechanisms

### Relevance

The finding that "association is computed and stored at the value before retrieval" adds nuance to our K > V finding. In induction heads, the VALUE stores the associated information while the KEY enables retrieval. This means V content IS important for the specific information being recalled — but K routing determines WHETHER it gets recalled at all. Our finding that K perturbation is more destructive than V perturbation for ANSWER accuracy is consistent: the answer depends on retrieving the RIGHT information (K routing) more than on the precise content at any single position (V content, which has redundancy across positions).

---

## 3. Sparse Attention for Interpretability (Anthropic, December 2025)

### Key Finding

Post-training method that makes transformer attention sparse **without sacrificing performance** — attention connectivity can be reduced to approximately **0.3% of its edges** while retaining original pretraining loss.

### Relevance

If 99.7% of attention edges can be pruned without loss, this means the vast majority of attention routing is redundant. This connects to our V-immunity finding: if most attention connections are dispensable, then perturbing V content at any given position has minimal effect because the model can route around it. K perturbation is more damaging because it disrupts the REMAINING 0.3% of critical routing pathways.

This also suggests that the "answer-coupled" positions we identified may be among the 0.3% of critical attention connections that survive sparsification.

---

## Synthesis: Toward a Unified K-V Framework

The literature is converging on a picture where K-V separation is a deep organizational principle of transformer computation:

1. **FF layers:** K = activation gating (routing), V = information storage (content). Nearly as interpretable as SAE features. (NeurIPS 2025)
2. **Attention layers:** K = attention pattern computation (routing), V = weighted information aggregation (content). QK routing > OV content for answer accuracy. (Our experiments)
3. **Induction heads:** K enables retrieval, V stores association. (Arora et al.)
4. **Sparse attention:** 99.7% of routing connections are redundant, but the remaining 0.3% are critical. (Anthropic)

Our K > V finding is a specific instantiation of a general principle: **routing decisions are more critical than content transmission** for downstream task performance, because routing errors compound (wrong information is propagated) while content errors are dampened by redundancy (the right positions are still attended, and their aggregate preserves most of the needed information).
