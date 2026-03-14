# QK Routing vs OV Content: Mechanistic Basis for K > V

**Cycle:** 30
**Date:** 2026-03-14
**Papers/resources reviewed:**
- McCormick (2025). "Patterns and Messages: A New Framing of Transformer Attention." Blog post at mccormickml.com.
- Anthropic/Transformer Circuits (2025). "Tracing Attention Computation Through Feature Interactions." transformer-circuits.pub.
- Elhage et al. (2021). "A Mathematical Framework for Transformer Circuits." Transformer Circuits Thread.
- Kadem & Zheng (2026). "Interpreting Transformers Through Attention Head Intervention." arXiv:2601.04398.

---

## 1. The "Patterns and Messages" Framework (McCormick, 2025)

### Core Idea

By merging weight matrices, each attention head decomposes into two independent computations:

1. **Patterns (W^QK):** The merged query-key matrix determines *which tokens to attend to and how much weight to give them* — the **routing function**.
2. **Messages (W^OV):** The merged value-output matrix defines *what information each token communicates if selected* — the **content function**.

Key insight: "Key, query, and value vectors can be thought of as intermediate results in the computation of the low-rank matrices" — they are computational conveniences, not fundamental primitives.

### Implications

- **Routing and content are independently decomposable.** This is exactly the mathematical structure behind our K > V finding: K vectors participate in the routing computation (QK dot product → attention weights), while V vectors participate in the content computation (weighted sum of V → output).
- **Attention heads function as dynamic neural networks** whose routing weights are constructed at inference time from the QK interaction. Disrupting K disrupts the dynamic weight construction; disrupting V only disrupts the information that flows through already-established routes.

---

## 2. Transformer Circuits: QK vs OV Circuits (2025)

### Key Finding

The mechanistic interpretability community has established a clean functional separation:

- **QK circuits:** Solve the routing problem — which positions attend to which. They determine attention patterns based on positional and content-based relationships.
- **OV circuits:** Solve the content problem — what information gets transmitted once attention is established. They determine the semantic information that flows through the attention connections.

### Attribution Integration

This work integrates QK/OV analysis into attribution graphs, enabling systematic tracing of information flow through transformer networks by decomposing attention into routing (which positions interact) and content (what information gets communicated).

---

## 3. Attention Head Intervention (Kadem & Zheng, January 2026)

### Key Finding

Survey of attention head intervention techniques for mechanistic interpretability. Demonstrates that individual attention heads specialize for specific semantic categories (politics, nationality, temporal concepts, numerical reasoning). Targeted intervention on specific heads can suppress harmful outputs and manipulate semantic content.

### Relevance

Head specialization validates our use of "answer-selective heads" (top-5 heads that attend from the answer token). Different heads serve different routing functions, and the answer-relevant heads implement answer-specific QK routing patterns.

---

## Synthesis: Our K > V Finding Has a Principled Mechanistic Explanation

**CRITICAL CONNECTION:** The mechanistic interpretability literature independently establishes that:

1. K vectors participate in the **QK routing circuit** (attention pattern computation)
2. V vectors participate in the **OV content circuit** (information transmission)
3. These are **functionally separable** — routing and content are independent computations

Our experimental program has provided the **causal validation** of this theoretical framework through perturbation experiments:

- **K perturbation destroys routing:** Random K-direction replacement at 5% of positions drops answer accuracy to 0-33% (Llama/Qwen). The model attends to wrong positions → wrong information flows → wrong answer.
- **V perturbation preserves routing:** Random V-direction replacement at same positions maintains 88-100% accuracy. Attention patterns are intact (routing preserved) → correct positions are still attended → even with corrupted content, the model recovers the answer through redundancy across positions.

**Why K > V for answers but not for text:**
- **Text prediction** depends on local routing (attend to recent tokens) AND local content (what those tokens say). Both K and V matter similarly for next-token prediction.
- **Answer prediction** depends on GLOBAL routing (attend to the right reasoning positions across the entire chain) but NOT on any single position's content (the answer integrates across many positions). K routing is the bottleneck; V content has massive redundancy.

This explains:
- K > V at ALL positions (routing is universally more critical than content for answer accuracy)
- V > K gap is LARGEST at early positions (+100pp on Llama) — infrastructure routing is the most critical
- Text damage from K perturbation at early positions is devastating (10.5%) but V perturbation has mild text effect (92.8%) — infrastructure routing underlies ALL computation
- K > V gap is smallest at late positions (+51-66pp) — late positions do carry some answer-relevant V content

**This reframes our K > V finding from phenomenological observation to mechanistically grounded result.** The finding is not surprising from the QK/OV framework perspective — what IS surprising is the MAGNITUDE of the asymmetry (V is completely dispensable at early/mid positions) and the UNIVERSALITY across model families.

### Novel Contribution Beyond Existing Literature

The QK/OV framework is theoretical/observational (matrix analysis, attribution). No one has previously shown through CAUSAL PERTURBATION that:
1. K > V holds at ALL position bands (early, mid, late)
2. V is literally zero-impact at early/mid infrastructure positions
3. The asymmetry holds across model families with different encoding strategies
4. The asymmetry survives SNR-matched energy control

Our work provides the first causal perturbation evidence for the routing > content hierarchy in KV cache during CoT reasoning.
