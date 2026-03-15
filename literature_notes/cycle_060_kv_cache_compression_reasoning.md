# Literature Notes: KV Cache Compression for Reasoning Models
**Cycle 60 — Literature Scan #6 — 2026-03-15**

## 1. Hold Onto That Thought: Assessing KV Cache Compression On Reasoning
**Authors:** (December 2025)
**arXiv:** 2512.12008

Comprehensive benchmark of KV cache compression methods on reasoning tasks (GSM8K, MATH500) with thousands-of-token reasoning traces:

- **H2O and SnapKV-D (decoding-enabled) are dominant** for reasoning models
- **Critical insight:** "No singular strategy fits all — performance is heavily influenced by dataset type" for non-reasoning models like Llama-3.1-8B-Instruct
- Heavy-hitter tracking (H2O) indicates *which* tokens appear most frequently matters more than simple recency for complex reasoning chains
- **Cache-inference tradeoff:** Tight eviction budgets paradoxically produce LONGER reasoning traces (model compensates for lost context by reasoning more)

**Relevance to our work — IMPORTANT:**
Their finding that H2O beats recency for reasoning is OPPOSITE to our finding that "recent" (sinks + most recent) beats everything at 100% (134/134 pooled). Key difference: they test during DECODING (ongoing generation), while we test post-hoc masking of completed traces. During decoding, heavy-hitter tracking identifies genuinely reused positions; our experiments test whether the final answer can be extracted from a compressed static trace. This is a critical methodological distinction — our "recent" universality may be GSM8K-specific or post-hoc-masking-specific. Real-time eviction during generation is a harder problem.

## 2. The Pitfalls of KV Cache Compression
**Authors:** (October 2025)
**arXiv:** 2510.00231

Identifies critical failure modes in KV cache compression:

- **Selective amnesia:** Certain instructions degrade much more rapidly with compression, effectively becoming invisible to the LLM
- Three contributing factors: compression method, instruction ordering, KV eviction bias
- Standard benchmarks MASK real-world problems — aggregate metrics look fine while specific instructions are lost
- **Security implication:** System prompt leakage vulnerability when compression loses instruction-enforcement

**Relevance:** The "selective amnesia" concept maps to our text/answer channel separation. If eviction preferentially removes answer-coupled positions (as our K-norm analysis suggests), the model loses answer computation while text appears fine — exactly our core finding. Their warning about aggregate metrics hiding per-instruction failures validates our per-problem analysis approach.

## 3. R-KV: Redundancy-aware KV Cache Compression for Reasoning Models
**Authors:** Cai et al. (NeurIPS 2025)
**arXiv:** 2505.24133

Three-component approach for reasoning model compression:
1. **Attention-based importance scoring** — identifies frequently-attended positions
2. **Dynamic redundancy scoring** — identifies repetitive tokens through real-time key vector analysis
3. **Joint eviction** — combines importance and redundancy

**Key result:** 90% memory savings, 6.6x throughput improvement, retaining near-full performance with 10-34% of KV cache. Even EXCEEDS baseline at 16% cache (!).

**Relevance — MAJOR:** The "exceeds baseline" finding at 16% resonates with our compression results showing that selective retention can improve accuracy — removing text-scaffolding positions that actively interfere with answer computation. Their "redundancy scoring via key vector analysis" is mechanistically related to our K-norm approach — both use K-vector properties to identify dispensable positions. The finding that reasoning models produce "excessively long outputs" with redundant tokens aligns with our observation that most CoT tokens are near-deterministic (median entropy ~0.023 bits).

## 4. ForesightKV: Optimizing KV Cache Eviction for Reasoning Models
**Authors:** (February 2026)
**arXiv:** 2602.03203

Training-based framework that learns optimal eviction:
- **Golden Eviction Algorithm:** Uses FUTURE attention scores as ground truth (privileged information during training)
- **Supervised + RL:** Pairwise ranking loss + GRPO for language modeling
- Outperforms prior methods at 50% cache budget on AIME2024/2025

**Relevance:** The use of future attention as ground truth addresses our methodological concern about using past attention (cumulative H2O) for eviction. Our Exp 057-058 showed true H2O (past attention) anti-correlates with K-norm and selects WRONG positions. ForesightKV's future-attention approach might resolve this by selecting positions that will be important for FUTURE computation, not positions that were historically popular.

## 5. Learning to Evict from Key-Value Cache (KV Policy / KVP)
**Authors:** (February 2026)
**arXiv:** 2602.10238

Per-head RL agents learn specialized eviction policies:
- **Key innovation:** Per-head agents trained using only K and V vectors (query-independent)
- Does NOT simply rediscover recency or attention-sink patterns
- Recovers "complex, non-local structure of future attention despite using only past keys and values"
- Outperforms H2O, StreamingLLM, SnapKV, TOVA, K-Norm, and random on RULER and OASST2-4k
- **Per-head specialization confirmed:** "The relative performance of different strategies varies significantly across heads"

**Relevance — CRITICAL:**
- Per-head policy specialization directly confirms our finding that different heads serve different functions (H0+H5 answer-critical vs dispensable heads). A single eviction heuristic cannot capture head-level functional diversity.
- The fact that learned policies DON'T rediscover simple heuristics validates our finding that K-norm and H2O select fundamentally different (anti-correlated) positions. Simple heuristics capture only one dimension of token importance.
- Query-independent eviction using only K and V vectors parallels our K-norm approach — the K-vector alone carries information about a position's future utility.

## 6. TRIM-KV: Cache What Lasts — Token Retention for Memory-Bounded KV Cache
**Authors:** (December 2025)
**arXiv:** 2512.03324

Lightweight learned retention gate:
- Retention scores naturally recover sink tokens, sliding windows, and gist compression WITHOUT explicit design
- **Surpasses full-cache models in some settings** — selective retention acts as regularization
- Retention scores align with human intuition and reveal **layer- and head-specific roles**

**Relevance:** The automatic recovery of sink + recency patterns confirms these are genuine computational features, not artifacts of heuristic design. But their improvement OVER full-cache is the real finding — removing certain positions (presumably text-scaffolding) can IMPROVE performance. This directly supports our hypothesis that text-scaffolding positions carry noise that degrades answer computation.

## 7. DefensiveKV: Taming the Fragility of KV Cache Eviction
**Authors:** (October 2025)
**arXiv:** 2510.13334

Identifies that existing scoring-aggregation frameworks assume token importance is STABLE over time:
- This "stability assumption" is inherently fragile
- Mean aggregation vulnerable to extreme cases
- **DefensiveKV:** Worst-case risk control via two-step linear-time approach
- 2.3x and 4.3x quality loss reduction vs strongest baseline at 20% cache

**Relevance:** The fragility finding resonates with our SNR cliff — precise encoding means small perturbations can cause catastrophic failure. Their defensive approach (managing worst-case) parallels our finding that "recent" works because it provides a structural guarantee (always keeps the computational endpoint), not because it scores individual positions optimally.

## 8. RLKV: Which Heads Matter for Reasoning?
**Authors:** (October 2025)
**arXiv:** 2510.08525

RL-guided per-head KV cache allocation:
- Full KV cache to reasoning-critical heads, aggressive compression of others
- 20-50% cache reduction with minimal performance loss, up to 1.21x speedup
- **"A fraction of heads proves essential for reasoning"** — head specialization confirmed independently

**Relevance — DIRECT CONFIRMATION:**
This is an independent confirmation of our core finding that specific attention heads (H0+H5 in our work) are answer-critical while others are dispensable. Their per-head allocation strategy (full cache for critical, compressed for others) is exactly the "head-selective 2x" strategy we tested in Exp 055, which beat uniform H2O by +8.7pp on Llama.

---

## Synthesis: The Compression Landscape Validates Our Framework

The KV cache compression literature has rapidly converged on several findings that align with our experimental program:

| Finding | Independent Source | Our Finding |
|---------|-------------------|-------------|
| Per-head functional specialization | KVP, RLKV, TRIM-KV | H0+H5 answer-critical, 6/8 dispensable (Exp 045-048) |
| Simple heuristics are suboptimal | KVP, ForesightKV | True H2O anti-correlated with K-norm (Exp 057) |
| Removing some tokens IMPROVES performance | R-KV, TRIM-KV | Text-scaffolding removal implicit in null space |
| Eviction fragility under worst-case | DefensiveKV, Pitfalls | SNR cliff at 14 dB (research_spec Exp 3) |
| Decoding vs post-hoc differ | Hold Onto That Thought | Our "recent" result is post-hoc masking specific |
| Future attention ≠ past attention | ForesightKV | True H2O (past) anti-correlated with K-norm (Exp 057) |

**Gap our work fills:** The literature treats compression as an engineering optimization problem. Our contribution is the MECHANISTIC explanation: K-routing vs V-content functional separation, digital vs analog encoding taxonomy, and the text/answer channel dissociation. The practical implication — preserve K-routing fidelity in critical heads — provides principled guidance for the engineering approaches above.
