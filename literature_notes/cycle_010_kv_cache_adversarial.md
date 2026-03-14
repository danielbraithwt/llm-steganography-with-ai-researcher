# KV Cache Adversarial Perturbation (MTI Framework)

**Paper:** "Can Transformer Memory Be Corrupted?"
**Authors:** (MTI V.1 framework authors, see arXiv:2510.17098)
**Date:** October 2025
**Scanned:** Cycle 10, 2026-03-14

## Key Concepts

**Malicious Token Injection (MTI):** A modular framework for systematically perturbing cached key vectors at selected layers and timesteps. Perturbation methods include additive Gaussian noise, zeroing operations, and orthogonal rotations.

**Theoretical grounding:** Quantifies how perturbations propagate through attention via Frobenius norm of corruption and softmax Lipschitz dynamics. Provides mathematical framework for understanding KV cache vulnerability.

## Models Tested

GPT-2, LLaMA-2/7B

## Key Results

- Small perturbations to cached keys can systematically bias attention maps
- Alter downstream token predictions without modifying input or model parameters
- 15-30% performance reduction across classification, QA, and summarization
- Vulnerabilities propagate silently through the inference pipeline
- RAG systems and agentic reasoning pipelines destabilized

## Relevance to Our Research

**HIGHLY RELEVANT — Independent validation of KV cache perturbation approach.**

1. **Confirms KV cache vulnerability:** Their framework demonstrates exactly what our experiments show — the KV cache is an exploitable attack surface where small perturbations have large effects on downstream computation.

2. **Framing difference:** They frame this as a security vulnerability (attack surface); we frame it as a mechanistic insight (hidden channel). Same phenomenon, different perspective. Their security framing could be important for our "implications" section.

3. **Method overlap:** Their additive Gaussian noise is equivalent to our SNR sweep (Exp 3/007). Their orthogonal rotations are conceptually related to our PGD null space search (Exp 4/008).

4. **Key gap in their work:** They don't distinguish between text-coupled and answer-coupled perturbation effects. Our spatial structure finding (Exp 5: rho=0.78 for answer-coupling) would extend their framework — not all perturbations are equal; the effect depends on which positions/layers are targeted.

5. **Cross-model comparison:** They test GPT-2 and LLaMA-2, but don't compare vulnerability profiles across architectures in the way we compare Qwen vs Llama encoding strategies.

## Impact on Evidence

- **Strengthens:** KV cache as carrier of separable computation (not just text generation)
- **Extends our work:** Their Lipschitz analysis could provide theoretical grounding for our SNR cliff finding
- **Cite in:** Discussion of KV cache vulnerability and the security implications of hidden channels
