# Architecture-Specific Encoding and Residual Stream Redundancy

**Papers:**
1. GQA: "Training Generalized Multi-Query Transformer Models" (Ainslie et al., arXiv:2305.13245, 2023)
2. "Hyper-Connections" (ICLR 2025, ByteDance)
3. "MUDDFormer: Breaking Residual Bottlenecks in Transformers" (2025, arXiv:2502.12170)
4. Various GQA vs MHA comparisons (2024-2025)

**Scanned:** Cycle 10, 2026-03-14

## GQA vs MHA: Information Encoding Implications

### Architecture Difference
- **MHA (Multi-Head Attention):** H query heads, H key heads, H value heads (all independent)
- **GQA (Grouped Query Attention):** H query heads, G key heads, G value heads (G < H, each KV head shared across H/G query heads)
- **MQA (Multi-Query Attention):** H query heads, 1 key head, 1 value head (maximum sharing)

### Performance Comparison
- GQA achieves 30-40% faster inference than MHA with near-equivalent accuracy
- Retains more complex pattern capture than MQA
- Trade-off: more groups (closer to MHA) = higher quality but slower
- Llama-2/3 uses GQA; Qwen uses MHA (in earlier versions) or GQA (in later versions)

### Implications for Our Research

**Our open question #11:** "Does GQA (vs MHA) explain Llama's noise robustness?"

**Hypothesis:** GQA's sharing of KV heads creates inherent redundancy — perturbation of one KV head affects multiple query heads, so the model has fewer independent information channels but each channel is shared/redundant. This could explain:

1. **Llama's noise robustness (Exp 007):** With 8 KV heads shared across 32 query heads (4x sharing ratio), uniform noise averages out across the shared channels. Each query head sees the same perturbed KV, so there's no divergence between query groups — the perturbation is "consistent" across the model.

2. **Llama's PGD null space failure (Exp 008):** GQA reduces the dimensionality of the KV cache (fewer independent KV heads), which shrinks the potential null space. With MHA (each query head has its own KV head), there are more independent directions to exploit. With GQA, perturbation at any KV head affects 4 query heads simultaneously, making it harder to find directions that change the answer without changing the text.

3. **Llama's distributed encoding (Exp 005, 007):** GQA forces information to be distributed across fewer, shared KV heads. This naturally produces a "distributed" encoding strategy because the model can't concentrate information at specific KV heads — everything is shared.

**This could be the mechanistic explanation for the Qwen/Llama encoding difference.**

## Residual Stream Redundancy

### Hyper-Connections (ICLR 2025)
- Problem: single residual stream is a bottleneck as models scale to hundreds of layers
- Solution: expand to multiple parallel streams that interact
- Shows that residual stream capacity limits information throughput
- Halves training tokens needed for equivalent performance

### MUDDFormer (2025)
- Demonstrates enhanced depth utilization via dynamic dense connections
- Standard transformer shows "diminishing returns" from depth — later layers contribute less
- Fixing this through richer connections improves reasoning

### Implications for Exp 009 (Layer Redundancy)

Our finding that single-layer ablation barely affects accuracy on BOTH models is consistent with:
1. **Residual stream compensation:** Each layer adds to a residual stream, so ablating one layer leaves N-1 layers contributing through residual connections
2. **Diminishing returns from depth:** If later layers contribute marginally (MUDDFormer's finding), destroying any single one has negligible effect
3. **Layer-level redundancy ≠ position-level redundancy:** The residual stream provides layer redundancy (each layer compensates for others), but doesn't provide position redundancy (each position's contribution is independent)

This explains our key finding: **the digital/distributed encoding distinction operates at position-level and noise-level, NOT at layer-level** (Exp 004, 005, 009). The residual stream masks layer-level differences but doesn't affect position-level differences.

## Impact on Evidence

- **Partially answers:** Open question #11 (GQA vs MHA and noise robustness) — GQA sharing ratio is a plausible mechanism
- **Explains:** Layer-level redundancy via residual stream compensation (Exp 009)
- **Suggests experiment:** Test PGD/SNR on a GQA version of Qwen or MHA version of Llama to isolate the attention mechanism's effect
- **Theoretical grounding:** The residual stream as an information-routing bottleneck (Hyper-Connections) provides a formal framework for understanding why layer ablation has minimal effect
