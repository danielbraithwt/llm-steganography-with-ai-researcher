# KV Cache Compression Effects on Reasoning

**Papers:**
1. "Hold Onto That Thought: Assessing KV Cache Compression On Reasoning" (arXiv:2512.12008, Dec 2025)
2. "Lethe: Layer- and Time-Adaptive KV Cache Pruning" (arXiv:2511.06029, Nov 2024)
3. "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling" (arXiv:2406.02069, Jun 2024)

**Scanned:** Cycle 10, 2026-03-14

## Key Findings

### Hold Onto That Thought
- H2O and decoding-enabled SnapKV are the dominant compression strategies for reasoning models
- "Heavy-hitter" tracking (attention-based importance) works best for reasoning traces
- Eviction at low budgets can paradoxically produce LONGER reasoning traces (more tokens)
- No singular strategy fits all for non-reasoning models (e.g., Llama-3.1-8B-Instruct)
- Specifically addresses reasoning tasks requiring long decoding (including GSM8K)

### Lethe — Challenges Pyramidal Assumption
- **Critical finding:** Pyramidal sparsity assumption DOES NOT hold for reasoning models
- LLaMA-8B: "Both early and late layers are sparse, while mid-layers are dense" — NOT pyramidal
- Qwen-7B: "Varied trends" with unpredictable fluctuations across layers
- Token relevance is **temporally dynamic** — "tokens that were once crucial may quickly become obsolete"
- Up to 91.7% KV cache memory reduction while maintaining accuracy (Math500: 85.4% on Qwen-7B vs 86.4% full cache)

### PyramidKV
- Original claim: Attention follows pyramidal funneling (lower layers dense, upper layers sparse)
- Works well for general tasks but challenged by reasoning-specific patterns
- Key insight: Attention sparsity increases in later layers for standard tasks, allowing smaller caches in upper layers

## Relevance to Our Research

**DIRECTLY RELEVANT — Multiple connections to our findings.**

1. **Heavy-hitter positions = our answer-coupled positions?** The "heavy-hitter" positions that H2O retains (high cumulative attention) may overlap with our answer-coupled positions. Both are positions that receive disproportionate attention. If H2O preferentially retains answer-coupled positions, this explains why it works for reasoning — it preserves the hidden channel.

2. **Pyramidal assumption failure for reasoning:** Lethe's finding that sparsity patterns differ for reasoning tasks connects to our Exp 009 finding — layer-level importance is NOT uniform or predictably pyramidal for reasoning. Both Qwen and Llama showed high layer redundancy (one layer destroyed ≈ no effect), which Lethe's finding of mid-layer density helps explain.

3. **Temporal dynamics of token importance:** Lethe's finding that token importance changes during generation connects to our position-level findings. During CoT, the "answer-coupled" positions may shift as new reasoning steps are generated. This could explain why our position classification (based on final attention) captures only part of the picture.

4. **Eviction produces longer traces:** The finding that KV cache eviction causes longer reasoning traces is fascinating — it suggests the model compensates for lost information by generating more text. This is consistent with our "text as lossy projection" hypothesis — when the hidden channel is degraded (positions evicted), the model needs more text to compensate.

## Impact on Evidence

- **Supports:** Position-level importance hierarchy for reasoning (heavy-hitters matter most)
- **Extends:** Our answer-coupling metric could inform better compression strategies
- **Challenges:** Temporal dynamics suggest our static position classification may miss important effects
- **New experiment idea:** Test whether H2O's retained positions correlate with our answer-coupled positions
