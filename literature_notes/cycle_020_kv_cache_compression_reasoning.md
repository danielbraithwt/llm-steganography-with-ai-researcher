# KV Cache Compression, Reasoning, and Phase Transitions

**Cycle:** 20
**Date:** 2026-03-14
**Papers reviewed:**
- (2026). "Understanding the Physics of Key-Value Cache Compression for LLMs through Attention Dynamics." arXiv:2603.01426
- (2025/2026). "R-KV: Redundancy-aware KV Cache Compression for Reasoning Models." NeurIPS 2025.
- (2026). "LongFlow: Efficient KV Cache Compression for Reasoning Models." arXiv:2603.11504
- (2025/2026). "Which Heads Matter for Reasoning? RL-Guided KV Cache Compression." OpenReview.
- (2025). "Can Transformer Memory Be Corrupted? (MTI V.1)" arXiv:2510.17098
- Feng et al. (2025). "Identify Critical KV Cache in LLM Inference from an Output Perturbation Perspective." arXiv:2502.03805

---

## 1. Phase Transitions in KV Cache Compression (2603.01426, March 2026)

### Key Finding
KV cache compression induces **structural phase transitions in semantic accessibility**, not gradual degradation. A "safety cliff" near **90% compression** where hallucination rates spike dramatically.

### Critical Insights
- **Retaining tokens ≠ utilizing them**: Concept vectors may stay decodable even as hallucination spikes. Information presence doesn't guarantee functional access.
- **Global Eviction Ratio (GER)**: When answer-critical tokens are erased across ALL attention heads simultaneously, ALL routing pathways are blocked → catastrophic failure.
- **Representational Rigidity**: Even when tokens survive, excessive head-level consensus prevents flexible re-routing → alternative failure mode.
- **Differential vulnerability**: "Relational and hierarchical categories degrade earlier than atomic entity categories." Distributed, process-level representations fragment before concrete nouns.
- **Multi-hop reasoning collapses structurally before complete token loss**: Routing pathways are more fragile than token presence.

### Relevance to Our Research
**CRITICAL — independent discovery of our SNR cliff phenomenon.**

1. Their "phase transition in semantic accessibility" at ~90% compression maps directly onto our SNR cliff at 14 dB on Qwen. Both show that KV cache information encoding has **digital-like fragility** — performance is maintained until a threshold, then collapses sharply.

2. The **two failure modes** (representational erasure vs representational rigidity) provide a framework for understanding why different noise types produce different effects in our experiments:
   - Random noise → representational erasure (isotropic destruction)
   - PGD adversarial → avoids both modes for text, targets answer-specific pathways

3. Their finding that **distributed, relational information fragments before localized, entity information** is exactly what we observe: answer computation (which requires multi-hop reasoning across the chain) is fragile, while individual text token predictions (local) are robust. The hidden channel carries the fragile relational/computational information.

4. The GER concept (simultaneous erasure across all heads) explains why our single-layer destruction (exp 009) has so little effect — the information has redundant routing pathways through other layers. Only simultaneous multi-layer or multi-head destruction breaks through.

---

## 2. R-KV: Redundancy-Aware Compression (NeurIPS 2025)

### Key Finding
At 16% (Llama-8B) or 33% (Qwen-14B) cache, R-KV reaches **105% of FullKV accuracy** — removing redundant tokens IMPROVES reasoning.

### Relevance
This is striking and directly relevant:
- Removing "text-scaffolding" tokens may reduce interference with answer computation
- Consistent with our finding that TC-selective noise preserves answer accuracy
- Suggests that some KV entries actively HURT reasoning (by introducing noise into the answer computation pathway)
- Potential connection: the tokens R-KV identifies as "redundant" may overlap with our TC-selective positions

---

## 3. LongFlow: KV Cache Compression for Reasoning Models (March 2026)

### Key Finding
"LongFlowScore" (L1-norm of attention weight × value vector) identifies critical tokens. 80% compression with ≤1.3% degradation on reasoning benchmarks (GSM8K, MATH, AIME).

### Key Insight
"The current query contains sufficient information to effectively estimate the importance of all historical tokens." This means importance is query-dependent — the SAME token may be important for one query (e.g., the answer) and unimportant for another (e.g., intermediate text).

### Relevance
This query-dependent importance directly supports our AC/TC framework:
- Answer token query → AC importance
- Text token query → TC importance
- Different queries access different subsets of the KV cache
- The functional separability is not a static property of positions but a dynamic property of how different queries access the cache

---

## 4. Which Heads Matter for Reasoning? (RLKV)

### Key Finding
Only a **small fraction of attention heads** is essential for reasoning. RL-guided allocation gives full cache to reasoning-critical heads and compressed cache to others. 20-50% reduction with near-lossless performance.

### Relevance
This head-level heterogeneity connects to our research:
- Our AC metric uses "top-5 answer-selective heads" — RLKV validates that reasoning-critical heads exist
- Standard compression methods (H2O) don't account for head-level specialization
- This could explain why H2O preferentially evicts AC positions (exp 011) — H2O uses cumulative attention across ALL heads, but answer computation may concentrate in a few specialized heads whose signal gets diluted in the aggregate

---

## 5. MTI V.1: Adversarial KV Cache Corruption (October 2025)

### Key Finding
Framework for corrupting cached keys via three mechanisms:
- **MTI-Gaussian**: Additive Gaussian noise (σ ∈ {0.01, 0.05, 0.1, 0.2})
- **MTI-Zeroing**: Complete key erasure
- **MTI-Rotation**: Norm-preserving orthogonal transformation

KL divergence averaging ~0.26 under corruption (vs <0.05 natural). 15-30% accuracy drops across benchmarks. Extractive QA most vulnerable (90.7% collapse); agentic pipelines most resilient.

### Relevance
**Complementary but less specific than our work:**
- MTI explores general corruption; we explore answer-specific corruption
- MTI doesn't separate text from answer effects
- Their finding that different tasks have different vulnerability profiles is consistent with our model-dependent encoding differences
- MTI-Rotation (norm-preserving, structured corruption) is closest to our PGD approach — both find that structured perturbation is more effective than random noise

---

## 6. Critical KV Cache via Output Perturbation (February 2025)

### Key Finding
Beyond attention weights, **value states and pretrained parameter matrices** are crucial for KV entry criticality. Perturbation-constrained selection reduces compression loss by >50%.

### Relevance
- Their emphasis on VALUE states (not just keys or attention weights) validates our methodology of perturbing BOTH K and V in the KV cache
- Their perturbation-constrained approach (minimize output change) is the DUAL of our PGD approach (minimize text change, maximize answer change)
- Could their "critical KV entries" correspond to our AC positions? Worth testing.

---

## Synthesis: Phase Transitions and the Hidden Channel

The KV cache compression literature reveals a consistent picture:

1. **Information is encoded digitally, not analog** — sharp phase transitions at specific compression thresholds (matching our SNR cliff)
2. **Relational/computational information is more fragile than local/entity information** — consistent with answer channel fragility
3. **Removing redundant tokens can IMPROVE reasoning** — consistent with the text channel containing noise relative to the answer channel
4. **Head-level specialization exists for reasoning** — only some heads matter for reasoning, and standard compression doesn't respect this
5. **Query-dependent importance** — the same position may be critical for one function (answer) and dispensable for another (text), supporting our AC/TC framework at a conceptual level even though the spatial concentration is weaker than originally claimed
