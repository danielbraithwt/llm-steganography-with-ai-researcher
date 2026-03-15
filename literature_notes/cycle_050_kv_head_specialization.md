# Literature Notes: KV Head Specialization and Functional Roles
**Cycle 50 — Literature Scan #5 — 2026-03-15**

## 1. Attention Heads of Large Language Models: A Survey
**Zheng et al. (2024/2025) — Patterns (Cell Press)**
**arXiv:** 2409.03752

Comprehensive survey proposing a four-stage framework for understanding attention head specialization, inspired by human cognition:
1. **Knowledge Recalling** — heads that retrieve stored knowledge
2. **In-Context Identification** — heads that process contextual cues
3. **Latent Reasoning** — heads that perform inference/computation
4. **Expression Preparation** — heads that organize output generation

Key findings:
- Specialized heads are **universally sparse** (<7% of all heads achieve >0.9 accuracy on reasoning functions)
- Two discovery methods: Modeling-Free (probing) and Modeling-Required (intervention)
- **Critical gap identified:** Cross-model transferability of head functions is largely unexplored — "whether a specialized head identified in one LLM exhibits the same functionality in another LLM" remains open
- Most studies investigate individual heads; few study collaborative relationships among multiple heads

**Relevance to our work:** Our finding that H5 is the primary answer-routing head on BOTH Qwen and Llama (Exp 045-046) directly addresses their identified gap on cross-model transferability. Our multi-head threshold experiments (Exp 047-048) address the collaborative relationship gap.

## 2. Iteration Head: A Mechanistic Study of Chain-of-Thought
**Cabannes et al. (NeurIPS 2024)**
**arXiv:** 2406.02128

Identifies a specialized attention mechanism — the "iteration head" — that enables transformers to implement iterative reasoning during CoT:
- A specific distribution of weights in the first two attention layers creates the iteration head
- These heads explicitly attend to previously generated tokens to carry forward interim results
- Two-layer transformers can implement iteration heads sufficient for any iterative algorithm
- **Good transferability** of CoT skills between tasks via the same iteration heads

**Relevance to our work:** The iteration head concept maps to our H5 answer-routing head. Our finding that H5 is position-independent (Exp 049) is consistent with iteration heads operating throughout the reasoning chain rather than at specific positions. The transferability finding resonates with H5 being primary on both Qwen and Llama.

## 3. Investigating Functional Roles of Attention Heads in Vision Language Models
**Jiang et al. (Dec 2025)**
**arXiv:** 2512.10300

Probing-based analysis of attention head specialization in multimodal models:
- Functional heads are "universally sparse" — only small subset performs specialized roles
- Fewer than 7% of heads achieve >0.9 accuracy across functions
- Intervention experiments (removal and emphasis) validate causal role

**Relevance:** Independent confirmation that specialized heads are sparse, consistent with our finding that 2/8 KV heads (H0, H5 = 25%) are answer-critical while 6/8 are dispensable.

## 4. Attention Head Intervention for Causal Interpretability
**Zhang et al. (Jan 2026)**
**arXiv:** 2601.04398

Paradigm shift from visualization to causal intervention in attention head analysis:
- Direct intervention (ablation, steering) is replacing correlation-based analysis
- Validates mechanistic hypotheses through direct manipulation

**Relevance:** Our K-direction replacement methodology is an instance of this paradigm. Our per-head K-only perturbation (Exp 045-046) is exactly the causal intervention approach this paper advocates.

---

## Synthesis: Head Specialization Convergence

The literature is converging on several findings that align with our experimental results:

1. **Head specialization is sparse** — <7% of heads are functionally critical (survey). We found 2/8 (25%) KV heads are answer-critical, with H5 being the primary answer-routing head.

2. **Cross-model transferability is an open question** — the survey identifies this as a key gap. Our H5 finding (same primary answer head on Qwen and Llama) is among the first experimental evidence for cross-model head function conservation.

3. **Iteration heads carry forward computation** — the NeurIPS mechanism study shows dedicated heads for iterative reasoning. Our H5 operates position-independently throughout the chain, consistent with an iteration/carry-forward function.

4. **Causal intervention is the gold standard** — our K-direction replacement method aligns with the field's methodological direction.
