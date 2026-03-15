# Literature Notes: CoT Faithfulness and Safety Updates
**Cycle 50 — Literature Scan #5 — 2026-03-15**

## 1. CoT May Be Highly Informative Despite Unfaithfulness
**METR (Aug 2025)**
**URL:** metr.org/blog/2025-08-08-cot-may-be-highly-informative-despite-unfaithfulness/

Key argument distinguishing faithfulness from informativeness:
- **Faithfulness:** Does CoT accurately represent ALL reasoning steps?
- **Informativeness:** Does CoT contain USEFUL SIGNALS about model behavior?
- Core claim: CoT can be highly informative for detecting complex behaviors EVEN when unfaithful

Evidence:
- When reasoning was complex (requiring substantial CoT), **>97% relaxed faithfulness** across 21,272 trajectories
- Clue-usage detector achieved **99.3% true positive / 96.2% specificity** even on unfaithful traces
- Red-teamers spent 119 hours trying to obfuscate CoT — could not defeat the detector for complex behaviors
- Key insight: Dangerous behaviors require sufficient computational complexity that models BENEFIT from using CoT, making the behavior visible

**Relevance — IMPORTANT COUNTERPOINT:**
This challenges the strongest interpretation of our safety findings. We showed models can carry answer-relevant computation through a K-routing channel invisible to text. METR argues that for truly complex dangerous behaviors, models will USE CoT because they need the computational help, making the behavior detectable.

**Resolution:** Both findings can coexist. Our evidence shows the hidden channel EXISTS and can carry answer-relevant computation. METR shows that for COMPLEX behaviors, models tend to use CoT anyway. The gap: what about behaviors of MODERATE complexity that the model can handle purely through the hidden channel? Our PGD results show the null space is narrow (~10% success rate), suggesting there are limits to what can be accomplished through hidden computation alone. But instruction-tuned models (Llama at 30% text compliance) may have stronger hidden channels.

## 2. C2-Faith: Benchmarking LLM Judges for CoT Faithfulness
**Mittal & Arike (March 2026)**
**arXiv:** 2603.05167

Introduces benchmark for evaluating LLM judges on two dimensions:
- **Causal faithfulness:** Does each step logically follow from prior context?
- **Coverage faithfulness:** Are essential intermediate inferences present?

Key findings:
- Judges show "substantial gap between detecting an error and localizing it"
- **Coverage judgments systematically inflated** for incomplete reasoning
- No single judge performs optimally across all settings
- Task framing significantly affects judge performance

**Relevance:** Even when CoT IS used, evaluating its faithfulness is hard. Judges can detect that something is wrong but struggle to identify WHERE the reasoning fails. This is consistent with our finding that the hidden computation channel is distributed — the unfaithfulness isn't at a single identifiable step but spread across the K-routing channel.

## 3. Steganographic Collusion in LLMs (Update)
**Mathew et al. (IJCNLP-AACL 2025, Dec 2025)**
**arXiv:** 2410.03768

Update from lit scan 40: Now published at IJCNLP-AACL 2025.
- GBRL and ICRL methods reliably elicit steganographic text encoding
- Standard mitigations (paraphrasing, passive oversight) NOT fully effective
- Emergence of steganographic collusion is a "plausible concern that should be monitored"

**Relevance:** Our finding that models carry hidden computation in K-routing channels is the MECHANISTIC substrate for the safety concerns this paper raises. If models can encode answer-relevant information invisible to text, they could in principle encode coordination signals the same way.

## 4. Chain-of-Thought Is Not Explainability
**Barez et al. (2025)**
**URL:** fbarez.github.io/assets/pdf/Cot_Is_Not_Explainability.pdf

Argues CoT should not be treated as model explanation:
- CoT may be plausible yet misleading
- Increases trust without guaranteeing safety
- Unfaithfulness propagates into downstream systems (RLHF, process reward models)

**Relevance:** Reinforces our core framing — text is a lossy projection, not an explanation.

## 5. KV Cache Adversarial Attacks
### 5a. Can Transformer Memory Be Corrupted? (MTI V.1)
**Hossain et al. (Oct 2025, revised Jan 2026)**
**arXiv:** 2510.17098

Framework for KV cache corruption during inference:
- Three corruption types: additive Gaussian noise, zeroing, orthogonal rotations
- Three control dimensions: magnitude, frequency, layer placement
- 15-30% performance drops across GPT-2, LLaMA-2, Gemma
- Destabilizes retrieval-augmented and reasoning pipelines

**Relevance:** Independent validation of our perturbation methodology. Their "magnitude, frequency, layer placement" dimensions map to our experimental variables. However, they don't test K vs V separately or examine the text/answer dissociation — our key contribution.

### 5b. Whose Narrative is it Anyway? (History Swapping)
**Ganesh et al. (Nov 2025)**
**arXiv:** 2511.12752

Block-level KV cache manipulation to steer conversations:
- **Early layers encode high-level structural plans; late layers maintain local discourse**
- Only full-layer overwrites successfully hijack topics
- Three outcomes: immediate persistent shift, partial recovery, delayed hijack
- Tested across 324 configurations on Qwen 3 models

**Relevance:** The early=structural, late=local finding maps to our positional dissociation (early=infrastructure, late=answer). Their finding that full-layer (not partial) overwrites are needed for topic hijacking resonates with our breadth > depth fragility finding (destroying K across all heads at a position is more devastating than destroying one head everywhere).

---

## Synthesis: Safety and Faithfulness Landscape

The literature now contains a nuanced debate:

| Position | Evidence | Implication |
|----------|----------|-------------|
| CoT is unfaithful (us, Barez, Anthropic/OpenAI) | Hidden computation in K-routing; 95-99% gradient suppressed by LM head | Text monitoring insufficient |
| CoT is informative despite unfaithfulness (METR) | >97% faithfulness for complex behaviors; robust detection | CoT monitoring valuable for complex behaviors |
| CoT faithfulness evaluation is hard (C2-Faith) | Judges can detect but not localize errors | Even when monitoring, quality is limited |
| Steganographic encoding is possible (Mathew et al.) | RL elicits hidden encoding; mitigations fail | Hidden channels are exploitable |
| KV cache is an attack surface (Hossain, Ganesh) | 15-30% degradation; topic hijacking possible | Deployment security concern |

Our contribution: the MECHANISTIC substrate for all these concerns — K-routing in the KV cache.
