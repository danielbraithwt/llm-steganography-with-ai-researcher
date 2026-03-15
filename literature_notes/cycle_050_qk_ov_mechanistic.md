# Literature Notes: QK/OV Mechanistic Interpretability and Architectural Differences
**Cycle 50 — Literature Scan #5 — 2026-03-15**

## 1. Tracing Attention Computation Through Feature Interactions (QK Attributions)
**Anthropic / Transformer Circuits (2025)**
**URL:** transformer-circuits.pub/2025/attention-qk/index.html

New method: QK attributions describe attention head scores as bilinear functions of feature activations:
- Represents a "significant qualitative improvement" over previous attribution methods
- Unlocks analyses previously impossible
- Key insight: Features interact THROUGH attention — QK determines routing, OV determines content transfer
- Demonstrates that transformers use sophisticated routing where different heads specialize in routing specific feature types

**Relevance:** Directly supports our K-routing > V-content framework. The bilinear QK attribution method could be applied to understand exactly WHAT features H5 routes during answer computation.

## 2. Circuits Updates — July 2025
**Anthropic / Transformer Circuits**
**URL:** transformer-circuits.pub/2025/july-update/index.html

Feature-based framework for understanding attention heads:
- **Copy heads:** Implement X → "saying X" mapping via positive eigenvalues of W_U W_OV W_E
- **Previous token heads:** Convert X → "prev(X)" — track what appeared one token back
- **Induction heads:** Combine copying (OV) with pattern matching (QK) — "if query is X, attend to keys preceded by X"
- Each head understood via two matrices: OV circuit (read/write) and QK circuit (where to attend)
- Feature-based language replaces eigenvalue analysis

**Relevance:** The copy → previous token → induction head hierarchy shows increasing QK routing sophistication. Our H5 answer-routing head likely implements a more specialized version of induction-like pattern matching, where the QK circuit routes answer-relevant information to the final position. The OV circuit determines what information flows through — and our V-immunity finding suggests the OV (content) channel has high redundancy while QK (routing) has low redundancy.

## 3. Physics of KV Cache Compression — Architectural Differences
**Ananthanarayanan et al. (March 2026)**
**arXiv:** 2603.01426

Identifies fundamentally different compression dynamics by architecture:

| Architecture | Depth Pattern | Description |
|-------------|--------------|-------------|
| **LLaMA** | Inverted funnel | Early-layer consensus + late diversification |
| **Qwen** | Funnel | Early exploration + late consolidation |

**Relevance — MAJOR:**
This is the first independent characterization of architectural differences in KV organization that maps directly to our encoding taxonomy:

- **Qwen funnel (early exploration → late consolidation):** The model explores broadly early, then consolidates into precise K-routing patterns in late layers. This explains DIGITAL encoding: late-stage K-routing is precise (cliff behavior) because the consolidation phase creates clean routing "codewords." It also explains why early positions are infrastructure — they support the broad exploration phase.

- **LLaMA inverted funnel (early consensus → late diversification):** The model reaches approximate consensus early, then diversifies routing in late layers for nuanced output. This explains ANALOG encoding: distributed routing across late layers means no single routing pattern is precise enough to create a cliff — degradation is gradual because disrupting any one late-layer routing channel just shifts the diversification pattern.

This provides a causal explanation for our encoding taxonomy that we couldn't derive from perturbation experiments alone.

---

## Synthesis: QK/OV Framework Is the Mechanistic Foundation

Our entire K > V finding program can now be grounded in the QK/OV framework:

1. **K-perturbation is devastating** because K determines the QK routing matrix — which positions attend to which. Disrupting routing means the model reads wrong positions.

2. **V-perturbation is recoverable** because V determines the OV content matrix — what information flows through each position. If one V is corrupted, the model can route AROUND it via intact K-routing to find the same information at other positions (redundancy via intact routing).

3. **H5 is the answer-routing head** because it implements a specialized QK circuit that routes answer-relevant features to the final token position. The OV circuit of H5 transfers answer content, but its criticality is in ROUTING (QK), not CONTENT (OV).

4. **Architectural differences (Qwen funnel vs LLaMA inverted funnel)** explain encoding differences: digital encoding arises from late-stage routing consolidation (Qwen), analog from late-stage routing diversification (LLaMA).

The convergence between our perturbation-based experimental findings and the field's theoretical/architectural understanding of QK/OV circuits is now complete.
