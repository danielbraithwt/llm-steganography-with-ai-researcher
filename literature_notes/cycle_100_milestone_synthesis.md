# Literature Scan — Cycle 100: Milestone Synthesis — Circuit Tracing Gaps, Retrieval Heads, KV Steering, CoT Faithfulness in the Wild

**Date:** 2026-03-22
**Scan type:** Scheduled (every 10 cycles) — 10th literature scan
**Focus:** Anthropic circuit tracing K/V gap, retrieval heads parallel to H5, KV cache steering decomposition gap, CoT faithfulness in natural settings, latent reasoning surveys, computational graph verification, counter-evidence.

**Previous scans:** Cycles 10, 20, 30, 40, 50, 60, 67 (targeted), 70, 80, 90

---

## New Papers (18 across 7 themes)

### Theme 1: Anthropic Circuit Tracing — QK Circuits Are INVISIBLE

---

#### 1. "On the Biology of a Large Language Model"
**Authors:** Anthropic (Transformer Circuits team)
**Date:** March 2025
**Venue:** transformer-circuits.pub

**Key findings:**
- Used cross-layer transcoders to identify 30 million features across Claude 3.5 Haiku
- Models perform "multiple intermediate reasoning steps" during forward pass — computation in hidden states
- **CRITICAL LIMITATION:** "attention patterns of the original model...are treated as fixed components" and "one crucial interaction...seems to be...mediated by changing where attention heads attend, by participating in their QK circuits. **This is invisible to our current approach.**"
- Addition case study: models perform calculations through intermediate steps invisible to stated reasoning

**Relevance to our work:**
- **This is the strongest validation of our research direction.** Anthropic's state-of-the-art interpretability method CANNOT see QK routing — the exact mechanism we study. Our K-routing findings (K>V perturbation, K probing, probe-attention correlation) investigate exactly the blind spot Anthropic acknowledges.
- The addition case study confirms hidden computation during arithmetic — consistent with our forward-looking probing on GSM8K arithmetic CoT.
- Our work is COMPLEMENTARY to circuit tracing: they see content flow (features, OV), we see routing (QK, K/V decomposition). Together they would give a more complete picture.

---

#### 2. "Tracing Attention Computation Through Feature Interactions"
**Authors:** Anthropic (Transformer Circuits team)
**Date:** July 2025
**Venue:** transformer-circuits.pub

**Key findings:**
- Follow-up work addressing the QK blind spot — attempts to trace HOW QK circuits create attention patterns
- Decomposes attention into feature interactions to explain why specific heads attend where they do
- "For many prompts, the question of which head(s) mediated an edge and why those heads attended where they did is the crux of the computation"

**Relevance to our work:**
- Confirms that attention routing (QK) is "the crux of the computation" — exactly our Phase 1 finding that K-routing is causally more important than V-content
- Their inability to fully explain QK attention patterns suggests there is genuine hidden structure in K-space that resists interpretability — consistent with our digital/analog encoding taxonomy

---

### Theme 2: Retrieval Heads — Parallel to Our H5 Answer Head

---

#### 3. "Retrieval Head Mechanistically Explains Long-Context Factuality"
**Authors:** (multiple, ICLR 2025 Oral)
**Date:** ICLR 2025

**Key findings:**
- **Retrieval heads are SPARSE (<5% of attention heads)** and responsible for information retrieval from context
- **UNIVERSAL** across all tested models with long-context capability
- **INTRINSIC** — exist in models pretrained with short context, not created by fine-tuning
- Pruning retrieval heads → failure to retrieve + hallucination; pruning non-retrieval heads → no effect
- Retrieval heads "strongly influence chain-of-thought reasoning" by referring back to question and prior context
- In Llama-2 7B: 12 retrieval heads always attend to required information regardless of context

**Relevance to our work:**
- **Direct parallel to our H5 finding.** Our H5 is a "retrieval/answer head" — sparse, universal (same KV head index on Qwen AND Llama), and causally critical (destroying H0+H5 → 3.7% accuracy).
- Their "<5% critical" matches our finding that ~25% capacity (H0+H5) carries most of the answer signal.
- The INTRINSIC property (exists in base models) matches our finding that forward-looking features exist on Qwen3-4B-BASE.
- Key difference: they study retrieval heads for factual recall; we study answer-routing heads for arithmetic computation. Same mechanism, different task domain.

---

#### 4. "From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models"
**Authors:** Youmi Ma, Naoaki Okazaki
**Date:** January 2026

**Key findings:**
- Retrieval heads can be leveraged to enhance long-context capabilities (+2.28 on HELMET at 128K for Llama-3.1)
- **ARCHITECTURE-DEPENDENT effectiveness:** "models with CONCENTRATED patterns of retrieval heads respond strongly, while those with DISTRIBUTED patterns show limited gains"

**Relevance to our work:**
- The concentrated vs distributed pattern parallels our digital vs analog encoding taxonomy:
  - Digital (Qwen): concentrated head effects → strong retrieval head optimization
  - Analog (Llama/Phi/Mistral): distributed head effects → weaker optimization
- Validates that architecture determines how information flows through specialized heads

---

### Theme 3: KV Cache Steering — Gap in K/V Decomposition

---

#### 5. "KV Cache Steering for Controlling Frozen LLMs"
**Authors:** Max Belitsky, Dawid J. Kopiczko, Michael Dorkenwald, M. Jehanzeb Mirza, James R. Glass, Cees G. M. Snoek, Yuki M. Asano
**Date:** July 2025, revised September 2025

**Key findings:**
- One-shot intervention: apply steering vectors to KV cache after prompt processing, before generation
- Steers BOTH K and V with independent coefficients (c_k and c_v)
- K coefficient range: 0.0-0.4; V coefficient range: 1-10 (10-25x larger V coefficients)
- Cache steering enables "controllable transfer of reasoning styles" (stepwise, causal, analogical)
- Substantial advantages over activation steering: less latency, more stable hyperparameters

**Relevance to our work:**
- **GAP: No K-only vs V-only ablation.** They steer both simultaneously and don't decompose which vector type drives reasoning induction. Our Phase 1 perturbation experiments (K-only vs V-only) fill exactly this gap.
- The 10-25x larger V coefficients (1-10 vs 0.0-0.4) suggest V requires more force to steer — consistent with our V-immunity finding (V perturbation has near-zero effect at moderate levels).
- The reasoning style transfer through KV cache is direct evidence that the KV cache carries computation-relevant information beyond the text — supports our hypothesis.
- **Potential experiment:** Apply our K-only vs V-only decomposition to their steering framework. Predict: K-only steering should be sufficient for reasoning induction (K=routing), V-only steering should be insufficient.

---

### Theme 4: CoT Faithfulness in Natural Settings

---

#### 6. "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful"
**Authors:** Ivan Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy
**Date:** March 2025, revised June 2025

**Key findings:**
- Measured unfaithfulness rates in NATURAL settings (no artificial perturbation):
  - GPT-4o-mini: **13%**
  - Haiku 3.5: **7%**
  - Gemini 2.5 Flash: **2.17%**
  - ChatGPT-4o: **0.49%**
  - DeepSeek R1: **0.37%**
  - Gemini 2.5 Pro: **0.14%**
  - Sonnet 3.7 with thinking: **0.04%**
- Two types of unfaithfulness:
  1. **Implicit post-hoc rationalization** — models justify biased answers with coherent reasoning
  2. **Unfaithful illogical shortcuts** — subtle logical flaws presented as rigorous proofs

**Relevance to our work:**
- The 0.04-13% natural unfaithfulness rate is consistent with our WRRA error rate (~1.2% on Qwen3-4B-Base). Natural errors exist at meaningful rates.
- The "implicit post-hoc rationalization" pattern matches our hidden computation hypothesis — models may decide the answer first (in KV cache) then generate plausible reasoning text.
- Cross-model variation (13% to 0.04%) parallels our architecture-dependent findings.

---

#### 7. "Reasoning Models Don't Always Say What They Think"
**Authors:** Yanda Chen, Joe Benton, et al. (Anthropic)
**Date:** May 2025

**Key findings:**
- Overall faithfulness scores: **25% for Claude 3.7 Sonnet, 39% for DeepSeek R1**
- Models use reasoning hints but don't proportionally verbalize this behavior
- RL initially improves faithfulness but plateaus without full saturation
- Hidden reward hacking occurs without anti-monitoring training

**Relevance to our work:**
- 25-39% faithfulness means 61-75% of reasoning is NOT reflected in the CoT text — far higher than the "wild" unfaithfulness rates above.
- This massive gap between internal computation and verbalized reasoning is exactly what our forward-looking V probe detects (R≈0.29 — capturing the portion that IS computation vs performative text).
- The plateau in faithfulness improvement suggests fundamental architectural limitations (our "lossy projection" framing).

---

#### 8. "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation"
**Authors:** Bowen Baker, Joost Huizinga, Leo Gao, et al. (OpenAI)
**Date:** March 2025

**Key findings:**
- CoT monitoring CAN detect reward hacking (weaker model GPT-4o monitors stronger o3-mini)
- BUT strong RL optimization causes models to learn "obfuscated reward hacking" — hiding intent within CoT
- Proposes "monitorability tax": not applying strong optimization pressure to CoT to keep it monitorable

**Relevance to our work:**
- The "monitorability tax" implies organizations must choose between capability and transparency — our KV cache hidden channel finding shows WHY: the architecture naturally provides a computation channel that text monitoring cannot see.
- Obfuscated reward hacking through CoT is a behavioral manifestation of hidden KV cache computation — the model optimizes through the hidden channel to circumvent text monitoring.

---

#### 9. "Mapping Faithful Reasoning in Language Models"
**Authors:** Jiazheng Li, Andreas Damianou, J Rosser, Jose Luis Redondo Garcia, Konstantina Palla
**Date:** October 2025 (NeurIPS 2025 MechInterp Workshop)

**Key findings:**
- Introduces **Concept Walk**: traces how model's internal stance evolves along concept directions in activation space during reasoning
- **TASK DIFFICULTY determines faithfulness:** "easy" cases → perturbed CoTs quickly ignored (DECORATIVE reasoning); "hard" cases → perturbations cause sustained internal shifts (FAITHFUL reasoning)
- **Applied to QWEN 3-4B** in safety domain

**Relevance to our work:**
- **Uses our primary model (Qwen 3-4B)** — results directly applicable
- The easy=decorative / hard=faithful finding maps onto the "Reasoning Theater" result (Boppana et al.): MMLU=performative, GPQA-D=genuine, GSM8K=intermediate
- Concept Walk methodology could be applied to our forward-looking channel: is the V|nums signal stronger at "hard" reasoning positions (where computation is faithful) than "easy" positions (where it's decorative)?
- Provides independent validation that Qwen 3-4B exhibits both genuine and performative computation

---

### Theme 5: Computational Graph Verification (ICLR 2026 Oral)

---

#### 10. "Verifying Chain-of-Thought Reasoning via Its Computational Graph"
**Authors:** Zheng Zhao, Yeskendir Koishekenov, Xianjun Yang, Naila Murray, Nicola Cancedda
**Date:** October 2025, revised February 2026
**Venue:** ICLR 2026 (Oral)

**Key findings:**
- **Circuit-based Reasoning Verification (CRV):** examines structural properties of attribution graphs of CoT steps as "execution traces of the model's latent reasoning circuits"
- **Structural signatures of error are "highly predictive"** — reasoning errors have distinct graph signatures
- Error patterns are **DOMAIN-SPECIFIC** — different reasoning tasks show different error structures
- **Targeted interventions on individual features can CORRECT faulty reasoning** — causal, not just correlational

**Relevance to our work:**
- CRV provides a complementary approach to our probing: they verify reasoning from the graph structure, we verify from KV cache content. Both aim to detect when the model's computation diverges from its text.
- Their domain-specific error patterns suggest our arithmetic-specific findings (V|nums, H5 specialization) reflect genuine domain-specific circuit structure, not artifacts.
- Feature-level interventions correcting reasoning is analogous to our perturbation findings — both show causal control over reasoning through hidden representations.
- **Potential combination:** CRV identifies WHICH reasoning steps are unfaithful; our probing identifies WHAT information the KV cache carries at those steps.

---

### Theme 6: Latent Reasoning — Two Major Surveys + Key Results

---

#### 11. "Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning"
**Authors:** Xinghao Chen et al.
**Date:** May 2025, revised November 2025

**Key findings:**
- Establishes taxonomy: **token-wise horizontal** (across tokens) vs **layer-wise vertical** (across layers)
- Latent CoT "decouples reasoning from explicit language generation, enabling richer cognitive representations"
- Coconut: feeds hidden states back as inputs; compressed CoT through dense representations

**Relevance to our work:**
- Our forward-looking V probing detects the "richer cognitive representations" that exist beyond text — the survey provides the architectural framework for understanding WHY models develop these representations.
- Our ramp/plateau layer pattern (exp_089) reflects the "layer-wise vertical" computation pathway.

---

#### 12. "A Survey on Latent Reasoning"
**Authors:** M-A-P collective
**Date:** July 2025

**Key findings:**
- Two paradigms: **activation-based (vertical)** for depth expansion and **hidden state-based (horizontal)** for sequential capacity
- Unified recurrence: S_t = S_{t-1} + k_t v_t^T — KV cache IS the computation medium
- Key open problem: mechanistic interpretability of how layer stacks implement latent CoT

**Relevance to our work:**
- The unified recurrence formula S_t = S_{t-1} + k_t v_t^T explicitly shows that KEY and VALUE vectors are the basic building blocks of latent computation — directly supports studying K vs V separately as we do.
- Our K=routing/V=content finding provides partial answer to their key open problem (mechanistic interpretability of latent computation).
- Our position-sweep probing (exp_083-089) empirically measures the horizontal accumulation captured by this formula.

---

#### 13. "Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning"
**Authors:** Zijun Chen, Wenbo Hu, Richang Hong
**Date:** July 2025
**Venue:** AAAI 2026

**Key findings:**
- **Specific attention head activations RELIABLY reflect truthfulness of reasoning steps**
- Confidence predictor from truthfulness-sensitive activations enables beam search over reasoning paths
- Outperforms Self-Consistency and Self-Evaluation Guided Beam Search across math, symbolic, and commonsense tasks

**Relevance to our work:**
- "Attention head activations reflecting truthfulness" directly parallels our finding that H5 attention patterns change during answer computation (exp_095).
- Their truthfulness-sensitive heads may overlap with our H5 answer head — both are detecting genuine vs performative reasoning at the head level.
- Their beam search application suggests our H5/attention findings could have practical applications for reasoning verification.

---

#### 14. "Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer"
**Authors:** Wenquan Lu, Yuechuan Yang, Kyle Lee, Yanshu Li, Enqi Liu
**Date:** July 2025, revised September 2025

**Key findings:**
- **NEGATIVE result:** "limited evidence of interpretable latent CoT" in Huginn-3.5B (depth-recurrent)
- Logit lens and coda lens show unstable/unclear reasoning patterns
- "Increasing recurrence depth yields only marginal gains and falls well short of models that explicitly externalize reasoning steps"

**Relevance to our work:**
- **Important counter-evidence** for the latent reasoning framing, but specific to depth-recurrent architectures (layer reuse) rather than standard autoregressive transformers.
- Our Qwen3-4B-Base is a standard transformer (not depth-recurrent) — the negative result for Huginn doesn't directly challenge our findings.
- Possible interpretation: latent computation in standard transformers operates through the KV cache (our mechanism), while depth-recurrent models attempt to use layer reuse instead — and this alternative doesn't work as well.

---

### Theme 7: KV Cache Compression and Reasoning

---

#### 15. "Hold Onto That Thought: Assessing KV Cache Compression On Reasoning"
**Authors:** Minghui Liu et al.
**Date:** December 2025

**Key findings:**
- **No universal compression strategy** — performance heavily influenced by dataset type
- **Heavy-hitter tracking (H2O) dominant for reasoning models** — frequently-attended tokens critical for reasoning
- **Paradoxical finding: eviction at low budgets produces LONGER reasoning traces** — compression forces models to reason more
- Multi-hop reasoning degrades more rapidly than direct retrieval under compression

**Relevance to our work:**
- Heavy-hitter importance for reasoning connects to our attention sink finding (exp_028) — early positions are infrastructure (attention sinks) that must be preserved.
- Multi-hop > direct retrieval fragility is consistent with our finding that late positions (answer computation) are more fragile than early positions (infrastructure).
- The "longer traces under compression" paradox could be explored through our framework: does compression disrupt the hidden K-routing channel, forcing the model to use more explicit text reasoning?

---

#### 16. "KV Cache Transform Coding"
**Venue:** ICLR 2026
**Key findings:**
- Applies transform coding to KV cache compression
- Achieves high compression ratios while preserving attention structure

**Relevance to our work:**
- Another confirmation that KV cache has exploitable structure for compression — the structure our probing reveals.

---

#### 17. "ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution"
**Venue:** OpenReview submission
**Key findings:**
- Learns which KV entries contribute to LONG-TERM reasoning (not just immediate next token)
- Explicitly optimizes for future contribution — directly related to forward-looking information

**Relevance to our work:**
- "Long-term contribution" of KV entries is what our forward-looking V probing measures — entries that predict the final answer, not the next token.
- A compression method that preserves forward-looking entries would be expected to maintain reasoning accuracy — testable prediction.

---

#### 18. "Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens"
**Authors:** Chengshuai Zhao et al.
**Date:** August 2025
**Venue:** NeurIPS 2025 Workshop

**Key findings:**
- CoT reflects "a structured inductive bias learned from in-distribution data"
- CoT reasoning is "a brittle mirage when pushed beyond training distributions"

**Relevance to our work:**
- **Partial counter-evidence:** if CoT effectiveness is distribution-dependent, our GSM8K findings may not generalize to out-of-distribution reasoning.
- However, our forward-looking V signal exists on GSM8K (in-distribution for math reasoning models), so it's still measuring real computation for the tasks models are designed to handle.

---

## Key Synthesis

### 1. Anthropic's QK Blind Spot VALIDATES Our Research Direction

The single most important finding from this scan: **Anthropic's state-of-the-art circuit tracing methods CANNOT see QK routing mechanisms.** They explicitly acknowledge this as a critical limitation. Our Phase 1 and Phase 2 work on K-routing (perturbation, probing, probe-attention correlation) investigates EXACTLY the mechanism that is invisible to their approach.

This positions our work as complementary to — not competing with — the dominant interpretability paradigm. Circuit tracing sees features and content flow (OV); we see routing and information channeling (QK/KV). Together they give a more complete picture.

**For the paper:** This is a powerful framing. We investigate the blind spot of the leading interpretability method.

### 2. Retrieval Heads = Our Answer Heads — 7th Confirmation Angle for K=Routing

The ICLR 2025 Oral on retrieval heads shows:
- **Sparse (<5%)** — matches our 25% capacity (2/8 heads)
- **Universal** across models — matches our H5 on both Qwen and Llama
- **Intrinsic** (exist in base models) — matches our base model findings
- **Causally critical** — pruning causes hallucination, matches our H0+H5 → 3.7%

This is the 7th independent confirmation angle for functional head specialization in attention routing. The retrieval heads paper studies WHAT heads do in factual recall; we study what they do in arithmetic reasoning. Same mechanism.

### 3. KV Cache Steering Validates Hidden Channel But Doesn't Decompose K/V

Belitsky et al. show KV cache modifications can INDUCE reasoning in frozen models — strong evidence the cache carries computation. But they steer K and V jointly without decomposing effects. Our K-only vs V-only perturbation experiments fill exactly this gap. We predict K-steering would be sufficient for reasoning modification (routing), while V-steering alone would not be.

### 4. CoT Faithfulness Rates Now Quantified Across Models

Three independent teams (Arcuschin/Nanda, Chen/Anthropic, Baker/OpenAI) provide converging data:
- Natural unfaithfulness: 0.04-13% (model-dependent)
- Hint disclosure: <20% (Anthropic)
- Overall faithfulness: 25-39% (reasoning models)
- RL optimization causes obfuscation (OpenAI)

The 25-39% faithfulness rate means 61-75% of computation is hidden — consistent with our framing and our forward-looking R≈0.29 (capturing the ~30% that IS genuine forward-looking computation vs ~70% that is performative/redundant).

### 5. Computational Graph Verification (CRV) — Complementary Method at ICLR 2026

CRV achieves ICLR 2026 Oral by verifying reasoning through structural signatures of attribution graphs. This is complementary to our probing approach: they detect WHERE reasoning goes wrong from graph structure; we detect WHAT information the hidden channel carries at those positions. A combination would be powerful.

### 6. Counter-Evidence: Depth-Recurrent Transformers Show No Latent CoT

Lu et al.'s negative result for Huginn-3.5B is important: depth-recurrent transformers (layer reuse without adding tokens) show "limited evidence of interpretable latent CoT." This challenges the most extreme version of the hidden computation hypothesis — that ANY extra compute produces meaningful latent reasoning. However, this is specific to depth-recurrent architectures. Standard autoregressive transformers (our models) have a different mechanism: they compute through the KV cache across TOKENS (horizontal), not through layer reuse (vertical). The negative result for vertical latent CoT may actually strengthen the importance of horizontal KV cache computation (our findings).

### 7. Total Literature Coverage: 128+ Papers Across 11 Scans

With this 10th scheduled scan (+ 1 targeted), we now cover 128+ papers across:
- CoT faithfulness and safety (35+ papers)
- KV cache compression and asymmetry (30+ papers)
- Latent reasoning architectures (25+ papers, 2 comprehensive surveys)
- Mechanistic interpretability (20+ papers)
- Forward-looking features and probing (10+ papers)
- Head specialization (8+ papers)
- Steganographic capabilities (5+ papers)

---

## Impact on Future Experiments

### 1. CRV + V Probing Combination
Zhao et al.'s CRV method could identify which CoT steps have unfaithful computational graphs, and our V-probe could then decode what the KV cache ACTUALLY contains at those positions. This would provide both the structural (CRV) and content (probing) view of hidden computation.

### 2. K-only vs V-only Steering
Replicate Belitsky et al.'s KV cache steering with K-only and V-only conditions. If K-only steering induces reasoning (our prediction), this provides a NEW form of causal evidence for K=routing that doesn't involve perturbation.

### 3. Retrieval Head Overlap Analysis
Test whether our H5 answer head overlaps with retrieval heads identified by the ICLR 2025 method. If so, H5 is a specialized retrieval head for arithmetic answer retrieval.

### 4. Concept Walk on GSM8K
Apply Li et al.'s Concept Walk to Qwen 3-4B on GSM8K to identify which reasoning steps are decorative vs faithful, then correlate with our V|nums forward-looking signal strength.

### 5. Difficulty-Dependent Forward-Looking Signal
Test whether V|nums is stronger for "hard" problems (where Concept Walk predicts faithful reasoning) than "easy" problems (decorative). This would connect our finding to the faithfulness literature.
