# Literature Notes: K-V Asymmetry Confirmations and Head Specialization Updates
**Cycle 60 — Literature Scan #6 — 2026-03-15**

## K-V Asymmetry: New Confirmations

### 1. NQKV: KV Cache Quantization Based on Normal Distribution Characteristics
**arXiv:** 2505.16210 (May 2025)

Quantization approach leveraging distributional properties of K vs V:
- Keys have higher magnitudes of outliers in certain channels; values don't show this pattern
- Key cache generally exhibits HIGHER NORMS than value cache
- Matrices with larger norms → more prone to quantization errors → need more bits

**Relevance:** Fourth independent confirmation angle for K > V. Our perturbation experiments, AsymKV quantization, KV-AdaQuant mathematical proof, and now distributional analysis ALL converge on the same finding: K requires more precision than V.

### 2. KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization
**OpenReview (2026)**

Nearly lossless 3.25-bit mixed precision:
- Theoretical analysis of "inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors"
- Studies "why key cache is generally more important than value cache for quantization error reduction"
- 21.25% inference throughput improvement on Llama-3.1-8B-Instruct

**Relevance:** Their theoretical analysis of WHY K is more important complements our experimental finding. Our mechanistic explanation (K determines routing, V determines content; routing is non-redundant, content is redundant) provides the causal mechanism behind their quantization sensitivity finding.

### 3. Hardware-Efficient Attention for Fast Decoding
**arXiv:** 2505.21487 (May 2025)

Singular-value analysis of KV cache:
- **Key cache lives in a LOW-RANK SUBSPACE** — almost all variance captured by few principal directions
- Keys are "highly redundant" in terms of singular value decay
- Steep spectral decay means K-vectors cluster in a low-dimensional manifold

**Relevance — MECHANISTIC INSIGHT:** The low-rank structure of the key cache EXPLAINS our digital encoding finding on Qwen. If K-vectors occupy a low-dimensional manifold, then small perturbations that push keys off this manifold are catastrophic (cliff behavior = digital encoding). Random K-direction replacement sometimes lands near valid codewords because the manifold is low-rank. This also explains why K-routing is fragile — the routing information lives on a thin manifold, and any perturbation off this manifold destroys routing accuracy.

**Counter-observation:** If keys are "highly redundant" (low-rank), how is K MORE important than V? Resolution: low-rank means the information is CONCENTRATED — few dimensions carry everything. Perturbation of those few critical dimensions is devastating precisely BECAUSE redundancy is low in the non-trivial directions.

## Head Specialization Updates

### 4. Head Pursuit: Probing Attention Specialization in Multimodal Transformers
**Authors:** (October 2025)
**arXiv:** 2510.21518

SOMP-based head specialization analysis:
- Attention heads specialize for specific semantic or visual attributes
- Individual heads in Mistral-7B specialize for politics, nationality, temporal concepts, numerical reasoning
- Top-attended tokens cluster coherently around each head's semantic category

**Relevance:** The semantic specialization pattern extends our functional specialization finding. Our H5 "answer-routing head" likely specializes for numerical/mathematical reasoning — it routes answer-relevant arithmetic results to the final position. Their clustering methodology could be applied to characterize WHAT H5 attends to beyond attention weight analysis.

### 5. Cross-Model Head Transfer Patterns
**Emergent research theme (2025-2026)**

Key observations from across the surveyed literature:
- Datasets with related semantics share specialized heads (MNIST/SVHN overlap, EuroSAT/RESISC45 overlap)
- Interventions on similar-dataset-sharing heads cause correlated performance degradation
- Early heads focus on general/peripheral features; deeper heads specialize in task-relevant signals

**Relevance:** The early=general, deep=specialized pattern exactly matches our layer structure findings: early positions/layers serve infrastructure (general text scaffolding), late layers serve answer computation (task-specific). The cross-dataset head sharing suggests our H5 convergence across Qwen and Llama reflects genuine functional conservation, not coincidence.

### 6. RLKV: Which Heads Matter for Reasoning? RL-Guided KV Cache Compression
**arXiv:** 2510.08525 (October 2025)

RL identifies reasoning-critical heads for per-head cache allocation:
- Full cache for critical heads, aggressive compression for dispensable heads
- 20-50% cache reduction, up to 1.21x speedup
- **"A fraction of heads proves essential for reasoning"**

**Relevance — INDEPENDENT REPLICATION:** This is the cleanest independent confirmation of our Exp 045-048 findings:
1. Head specialization exists (some critical, most dispensable) — confirmed
2. Per-head cache allocation is more effective than uniform compression — confirmed
3. RL discovers the specialization automatically — matches our observation that PGD discovers answer-coupled positions automatically (Exp 4-5)

Their result also validates our practical recommendation from Exp 055: "head-selective 2x" (allocating more cache budget to critical heads) outperforms uniform H2O. RLKV provides the learned version of our principled allocation.

### 7. LLMs Know What to Drop: Self-Attention Guided KV Cache Eviction
**arXiv:** 2503.08879 (March 2025)

- Attention in diverse long-context tasks exhibits sparsity
- LLMs implicitly "know" which tokens can be dropped at the HEAD LEVEL after pre-filling
- Head-level eviction decisions are informed by self-attention patterns

**Relevance:** The finding that eviction decisions should be made at HEAD level (not position level) is another confirmation that head identity matters more than position for compression quality. This aligns with our Exp 047 finding that head identity > capacity fraction: dispensable heads at 25% capacity retain 96-100%, while answer heads at the same 25% capacity drop to 3.7%.

---

## Synthesis: K-V Asymmetry Is Now a Field Consensus

The K > V asymmetry is now confirmed from SIX independent angles:

| Angle | Source | Year | Method |
|-------|--------|------|--------|
| 1. Perturbation | Our Exp 023-038, 045-046 | 2026 | K-direction devastating, V-direction immune |
| 2. Quantization (empirical) | AsymKV (COLING) | 2025 | V can be 1-bit quantized |
| 3. Mathematical proof | KV-AdaQuant | 2025 | K has larger spectral/Frobenius norms |
| 4. Compression physics | Ananthanarayanan et al. | 2026 | Phase transitions driven by routing accessibility |
| 5. Distributional analysis | NQKV | 2025 | K has higher-magnitude outliers |
| 6. Low-rank structure | Hardware-Efficient Attention | 2025 | K lives on low-rank manifold (concentrated info) |

**New mechanistic insight from low-rank finding:** K > V is because K information is CONCENTRATED on a low-rank manifold. The few critical dimensions carry all routing information. This concentration makes K fragile to perturbation (digital-like) but also highly compressible (if you know the manifold). V information is more distributed (high-rank), making it robust to individual perturbation (analog-like) but harder to compress efficiently.

This connects beautifully to our encoding taxonomy:
- **Digital K (Qwen):** Very low-rank K manifold → sharp cliff when perturbed off manifold
- **Analog K (Llama):** Higher-rank K manifold → gradual degradation as perturbation grows
- **V universally analog:** High-rank content representation → always gradual

The field is building toward a unified understanding of KV cache organization that matches our experimental findings almost exactly. Our contribution is the causal/mechanistic explanation (routing vs content) that grounds the engineering observations.
