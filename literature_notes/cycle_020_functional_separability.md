# Functional Separability in Transformer Representations

**Cycle:** 20
**Date:** 2026-03-14
**Papers reviewed:**
- (2026). "Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis." arXiv:2602.11169
- Facebook/Meta (2025/2026). "Verifying Chain-of-Thought Reasoning via Its Computational Graph (CRV)." ICLR 2026 Oral. arXiv:2510.09312
- (2025). "How does Chain of Thought Think?" arXiv:2507.22928

---

## 1. Direction-Magnitude Double Dissociation (2602.11169, February 2026)

### Key Finding
A striking cross-over dissociation in transformer hidden states:
- **Angular (direction) perturbations** damage language modeling loss **42.9x** more than magnitude perturbations
- **Magnitude perturbations** damage syntactic processing (subject-verb agreement) — **20.4% accuracy drop** vs 1.6% for angular changes

### Method
L2-matched perturbation analysis: angular and magnitude perturbations are constrained to produce identical Euclidean displacements, isolating the functional role of each geometric component.

### Functional Pathways
- **Direction** → attentional routing and general language modeling capacity. Angular damage flows through **attention pathways** (28.4% loss recovery via attention repair).
- **Magnitude** → processing intensity for fine-grained syntactic judgments. Magnitude damage flows through **LayerNorm pathways** (29.9% recovery via LayerNorm repair).

### Architecture Dependence
- LayerNorm architectures show the dissociation
- RMSNorm architectures exhibit different patterns
- Tested on Pythia-family models

### Relevance to Our Research
**CRITICAL — independent validation of functional separability in transformer representations.**

1. This paper demonstrates that THE SAME hidden state vector encodes functionally separable signals in different geometric components. This is exactly what our null space finding claims: the KV cache encodes text-prediction and answer-computation information in separable subspaces.

2. Their direction/magnitude split maps loosely onto our text/answer split:
   - Direction (semantic routing) ↔ text prediction (which token comes next)
   - Magnitude (processing intensity) ↔ answer computation (accumulating numerical results)

3. The architecture dependence (LayerNorm vs RMSNorm) could explain why Qwen and Llama show different encoding strategies — they may use different normalization schemes that affect how functional separability manifests.

4. **Methodological lesson**: Their L2-matched design controls for total perturbation magnitude. Our PGD approach implicitly does something similar (the optimizer can freely choose direction, revealing which directions matter for which function). But our destruction experiments DON'T control for perturbation type — we use isotropic noise, which conflates direction and magnitude effects.

5. **Open question**: Would direction-only vs magnitude-only KV cache perturbation reveal cleaner text/answer dissociation than our current isotropic noise approach?

---

## 2. Circuit-based Reasoning Verification — CRV (ICLR 2026 Oral)

### Key Finding
Attribution graphs of correct vs incorrect CoT steps have **distinct structural fingerprints**. These signatures are:
- **Highly predictive** of reasoning errors
- **Domain-specific** — failures in different reasoning tasks manifest as different computational patterns
- **Causally valid** — targeted interventions on individual transcoder features successfully correct faulty reasoning

### Method
Four-stage pipeline on interpretable LLMs with transcoders (sparse autoencoders):
1. Build computational graph from attribution
2. Extract structural features from the graph
3. Train classifier on these features
4. Use analysis to guide targeted interventions

### Relevance to Our Research
**Strong support for structured internal computation.**

1. CRV demonstrates that correct reasoning has a specific computational fingerprint that differs from incorrect reasoning. This means the model's internal computation IS structured and interpretable — not a black box. Our KV cache perturbation experiments manipulate this computational structure.

2. The domain-specificity finding is important: different reasoning tasks use different computational circuits. This could explain why our selectivity metrics (AC/TC) work differently across tasks and models — different computational circuits may be active.

3. Their causal intervention (correcting faulty reasoning by editing specific features) is the constructive complement to our adversarial approach (disrupting correct reasoning by editing the KV cache).

4. **Future direction**: Could we apply CRV-style attribution graph analysis to characterize what the PGD null space actually is? If the null space directions correspond to specific transcoder features, this would provide a mechanistic interpretation.

---

## 3. How Does Chain of Thought Think? (2507.22928, July 2025)

### Key Finding
First **feature-level causal study of CoT faithfulness** using sparse autoencoders + activation patching on Pythia models (70M and 2.8B) solving GSM8K.

### Key Results
- **Scale-dependent**: Swapping CoT-reasoning features into a noCoT run raises answer log-probabilities significantly in 2.8B but not in 70M. CoT benefits emerge at larger scales.
- **Internal reorganization**: CoT prompting induces higher activation sparsity and feature interpretability — more modular internal computation.
- **Distributed**: Useful CoT information is widely distributed across the network, not concentrated in top-performing patches.
- **Confidence**: CoT improves answer confidence from 1.2 to 4.3 in correct generation.

### Relevance to Our Research
**Moderate support for our distributed computation findings.**

1. The "widely distributed" finding aligns with our observation that the null space is not concentrated at a few positions (rho=0.20, not 0.78). CoT information is distributed, which is consistent with a high-dimensional null space that operates through many small changes rather than large targeted perturbations.

2. The scale-dependence finding suggests that our 4B Qwen model may be at or near the threshold where CoT features become meaningful. Larger models might show stronger hidden channels.

3. The modular computation finding (higher sparsity during CoT) suggests that CoT DOES reorganize internal representations — it's not just performative text. This is consistent with our finding that the KV cache structure during CoT is fundamentally different from non-CoT computation.

---

## Synthesis: Functional Separability Is Real But Distributed

These three papers, from independent research groups using different methods, converge on a key insight:

**Transformer representations encode multiple functionally separable signals, but the separation is distributed and geometric rather than localized.**

- Direction vs magnitude (2602.11169)
- Correct vs incorrect reasoning circuits (CRV)
- CoT vs non-CoT feature activation patterns (2507.22928)

This reframes our findings: the PGD null space is likely a distributed geometric phenomenon (consistent with rho=0.20), not a localized spatial one (inconsistent with the original rho=0.78). The functional separability exists, but it operates through the geometry of the representation space (direction, magnitude, feature activation patterns) rather than through spatial concentration at specific token positions.
