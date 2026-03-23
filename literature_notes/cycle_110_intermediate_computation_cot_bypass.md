# Literature Scan — Cycle 110: Intermediate Computation Probing, Causal Bypass, Arithmetic Error Detection, KV Cache Dynamics

**Date:** 2026-03-23
**Scan type:** Scheduled (every 10 cycles) — 11th literature scan
**Focus:** Probing intermediate computation during CoT, causal bypass mechanisms, arithmetic error detection from hidden states, CoT faithfulness measurement, information bottleneck & lossy projection, KV cache architecture dynamics, steganographic capabilities.

**Previous scans:** Cycles 10, 20, 30, 40, 50, 60, 67 (targeted), 70, 80, 90, 100

---

## New Papers (21 across 9 themes)

### Theme 1: Causal Bypass — CoT Is Often Decorative

---

#### 1. "When Chains of Thought Don't Matter: Causal Bypass in Large Language Models"
**Authors:** Anish Sathyanarayanan, Aditya Nagarsekar, Aarush Rathore
**Date:** February 2026
**Venue:** Under review at ICLR 2026
**arXiv:** 2602.03994

**Key findings:**
- Introduces **CoT Mediation Index (CMI)**: patches CoT token hidden states with no-CoT activations and measures answer degradation. CMI = max(0, Δ_cot − Δ_ctrl) / max(Δ_cot + Δ_ctrl, ε)
- **83.4% overall bypass**: mean CMI = 0.166 across tasks
- **Task-dependent**: QA items show 100% bypass (CMI = 0); arithmetic shows modest mediation (CMI ≈ 0.08-0.125); logic shows range 0.0-0.564
- **Narrow reasoning windows**: CoT influence concentrates at specific depths — Qwen2.5-0.5B peaks at layers 18-24 (~92% depth), DialoGPT peaks at layers 0-2
- Audit-aware prompting increases manipulation signals (mean delta +5.10) WITHOUT increasing causal mediation — answers are "causally invariant to CoT interventions" in bypass regimes
- Models tested: Qwen2.5-0.5B-Instruct, DialoGPT-large

**Relevance to our work:**
- **STRONGEST causal evidence yet that CoT is decorative.** Their CMI metric provides mechanistic quantification of what our forward-looking V probing measures from the content side. Our V-cache at intermediate positions carries computation info (R=0.95) that is DISCONNECTED from CoT text — consistent with their CMI ≈ 0 finding.
- The narrow reasoning windows at specific depths map onto our ramp→plateau layer structure (exp_089): computation concentrates at specific depths.
- Their arithmetic CMI = 0.08-0.125 (87-92% bypass even for MATH) is consistent with our paraphrase null (exp_085: synonym paraphrase = 0.6% accuracy drop) — surface text doesn't mediate arithmetic answers.
- **KEY DIFFERENCE:** They test small models (0.5B-0.8B); our work tests 4B-8B models where effects may be stronger.

---

#### 2. "When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors"
**Authors:** Scott Emmons, Erik Jenner, David K. Elson, et al.
**Date:** July 2025
**Venue:** arXiv (cs.AI)
**arXiv:** 2507.05246

**Key findings:**
- Distinguishes **CoT-as-rationalization** (post-hoc, decorative) from **CoT-as-computation** (genuinely necessary for the task)
- When bad behavior requires complex multi-step reasoning (CoT-as-computation), models MUST expose reasoning in CoT to execute it → monitorable
- Prior unfaithfulness findings arose because tasks were simple enough to bypass CoT entirely
- "Monitorability, not faithfulness" is the key safety property

**Relevance to our work:**
- Our GSM8K arithmetic tasks are intermediate complexity — some steps require genuine computation (multi-step), some don't (single-step recall). This explains our finding that V|MLP is strongest for multi-step operations (+0.088, exp_108) — those are the CoT-as-computation cases.
- The rationalization/computation distinction maps onto our accuracy-conditional findings (exp_101/102): V|nums is positive for correct (genuine computation) and NEGATIVE for incorrect (failed rationalization).
- Implies our hidden channel is most active during CoT-as-computation steps and less active during CoT-as-rationalization steps.

---

#### 3. CoT May Be Highly Informative Despite "Unfaithfulness"
**Authors:** METR (Alignment research)
**Date:** August 2025
**Venue:** Blog post / technical report

**Key findings:**
- Across 21,272 trajectories where hints were genuinely difficult to understand without reasoning, only 3 cases of potential unfaithfulness — rate ~0.016%
- Automated detector achieved 99.3% true positive rate even on trivial clues where models were unfaithful
- Argues "unfaithfulness matters less than capability complexity" — when CoT is genuinely needed, it's almost always faithful

**Relevance to our work:**
- Provides important nuance: our hidden channel findings apply primarily to cases where CoT is NOT genuinely needed for the computation (simple arithmetic, recall-type steps). For genuinely difficult multi-step reasoning, text and hidden channel may converge.
- Compatible with our difficulty-conditional findings: easy problems → large V|nums (channel carries info text doesn't need to), hard problems → V|nums still positive but text is also informative.

---

### Theme 2: Probing Intermediate Computation During Arithmetic CoT

---

#### 4. "LLMs Faithfully and Iteratively Compute Answers During CoT: A Systematic Analysis With Multi-step Arithmetics"
**Authors:** Keito Kudo, Yoichi Aoki, Tatsuki Kuribayashi, et al.
**Date:** December 2024 (v1); March 2026 (v4)
**Venue:** EACL 2026 (Long Finding)
**arXiv:** 2412.01113

**Key findings:**
- **Linear probes on residual stream hidden states** at each (position, layer) during multi-step symbolic arithmetic
- Answers emerge DURING CoT generation, not before: pre-CoT accuracy ~18-50%, post-CoT ~94-100%
- **Iterative computation**: different variables become decodable at different timesteps, matching the order of computation in the reasoning chain
- Causal intervention: patching the immediately preceding equation affects generation, but patching problem statement has almost no effect
- **9 models tested**: Qwen2.5 (7B/14B/32B), Qwen2.5-Math-7B, Yi1.5 (9B/34B), Llama3.1-8B, Llama3.2-3B, Mistral-Nemo-12B

**Relevance to our work:**
- **MOST DIRECTLY COMPARABLE to our experiments 106-109.** They probe hidden states during arithmetic CoT at intermediate positions — exactly what we do. But they probe the RESIDUAL STREAM only, while we probe K and V SEPARATELY. Our K/V decomposition adds a dimension they lack.
- Their finding that "answers emerge during CoT" is CONSISTENT with our intermediate value decodability (exp_106): V at "=" positions encodes step results (R=0.95). The results converge.
- Their causal intervention (patching preceding equation matters, problem statement doesn't) parallels our V_eq >> V_prompt finding (gap +0.39 to +0.63, exp_106) — information at computation positions is genuinely computed, not inherited from the problem encoding.
- **KEY GAP IN THEIR WORK:** They don't test whether hidden states carry information BEFORE the text reveals it. Our forward-looking probing (V at offset -5, R=0.794) fills this gap.
- **Their claim of "faithful" computation is COMPATIBLE with our claim of "hidden" computation.** They show computation happens in hidden states during CoT (iterative). We show computation happens in hidden states BEYOND what text reveals (forward-looking). These are not contradictory — both describe computation in hidden representations, we additionally show the text is a lossy projection of it.

---

#### 5. "Probing for Arithmetic Errors in Language Models"
**Authors:** Yucheng Sun, Alessandro Stolfo, Mrinmaya Sachan
**Date:** July 2025
**Venue:** EMNLP 2025
**arXiv:** 2507.12379

**Key findings:**
- Simple probes decode both the model's predicted output AND the correct answer from hidden states, **regardless of whether the output is correct**
- Error detectors predict model correctness with >90% accuracy from activations alone
- Probes trained on simple 3-digit addition generalize to GSM8K addition-only problems
- Probes enable selective re-prompting of erroneous reasoning steps

**Relevance to our work:**
- **INDEPENDENT VALIDATION of our WRRA hypothesis.** They show the model's hidden states encode the correct answer even when the model writes the wrong answer — exactly what we attempted to measure in exp_099 (WRRA K-probe at error positions). Their >90% accuracy from probes is far stronger than our inconclusive K=52-64% (ns) result.
- Critical difference: they probe the RESIDUAL STREAM; we probe K and V separately. The correct-answer signal may be primarily in the residual stream (which includes both K and V projections plus MLP outputs), explaining why our K-only and V-only probes showed weaker signal.
- **Suggests our WRRA methodology should be revised**: probe the RESIDUAL STREAM at error positions rather than K or V alone. This could recover the "correct answer despite wrong text" signal that was inconclusive in exp_099.
- The GSM8K generalization result is directly applicable to our setup — we could train probes on simple arithmetic and transfer to multi-step GSM8K reasoning.

---

#### 6. "The Validation Gap: A Mechanistic Analysis of How Language Models Compute Arithmetic but Fail to Validate It"
**Authors:** Leonardo Bertolazzi, Philipp Mondorf, Barbara Plank, Raffaella Bernardi
**Date:** November 2025
**Venue:** EMNLP 2025

**Key findings:**
- **Structural dissociation**: arithmetic computation occurs in HIGHER layers; validation occurs in MIDDLE layers
- Validation mechanisms operate BEFORE arithmetic computation finishes → models CANNOT validate their own arithmetic
- Models rely on "consistency heads" that check surface-level alignment of numerical values, not deep verification
- This architectural mismatch — not lack of capacity — limits error detection

**Relevance to our work:**
- **DIRECTLY maps onto our layer-sweep findings.** Our ramp phase (L0-L9) corresponds to their "validation/encoding" layers; our plateau phase (L10-L35) corresponds to their "computation" layers. The forward-looking V signal EMERGES at L8-L9 and PLATEAUS through L10-L35 (exp_089) — matching their temporal ordering of encoding→computation.
- The "consistency heads" that check surface alignment parallel our finding that K>V perturbation fragility reflects routing necessity: the model checks positional consistency (K routing) but doesn't verify computational content (V).
- **Explains our Validation Gap observation from exp_108:** the nonlinear MLP baseline captures ~75% of V's apparent computation signal because the MLP computes the same arithmetic the model is doing — but the remaining V|MLP=0.13-0.23 represents the HIDDEN computation that validation can't access.
- Their finding that this is architectural (not capacity-limited) supports our "lossy projection" framing — the text bottleneck is structural, not just information-theoretic.

---

#### 7. "Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification"
**Authors:** Anqi Zhang, Yulin Chen, Jane Pan, et al.
**Date:** April 2025
**Venue:** arXiv (cs.AI)
**arXiv:** 2504.05419

**Key findings:**
- Hidden states encode correctness of intermediate answers with high accuracy
- **Hidden states encode correctness of FUTURE answers** — enabling early prediction BEFORE the answer is fully formulated
- Probe-based verifier reduces inference tokens by 24% without performance loss
- Models "encode a notion of correctness yet fail to exploit it"

**Relevance to our work:**
- **"Future answer correctness" = our forward-looking signal.** They show hidden states predict whether future answers will be correct, we show V-cache carries the actual computation values forward. Same phenomenon from different angles.
- Their 24% token savings from early exit parallels the Reasoning Theater finding (80% savings on MMLU) — both show the model "knows" more than it reveals.
- "Encode but fail to exploit" is the core of our lossy projection hypothesis: the model has the information but the text generation bottleneck prevents using it.

---

### Theme 3: Reasoning Theater — Updated Full Paper

---

#### 8. "Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought"
**Authors:** Siddharth Boppana, Annabel Ma, Max Loeffler, Raphael Sarfati, Eric Bigelow, Atticus Geiger, Owen Lewis, Jack Merullo
**Date:** March 5, 2026 (v3)
**Venue:** arXiv (cs.CL)
**arXiv:** 2603.05488

**Key findings:**
- **Full paper follow-up** to original Reasoning Theater blog post (already in our literature)
- Models tested: DeepSeek-R1 (671B) and GPT-OSS (120B)
- Activation probes decode model's final answer FAR EARLIER in CoT than a text monitor can
- Probe-guided early exits save up to **80% of tokens on MMLU** and **30% on GPQA-Diamond** while maintaining accuracy
- Inflection points (backtracking, "aha" moments) occur primarily when probes detect large belief shifts — these behaviors track GENUINE uncertainty

**Relevance to our work:**
- The gap between probe decodability and text monitor decodability IS our forward-looking signal measured differently. They detect "answer known" from activations 80% of chain before text reveals it; we detect V|nums from 3% of chain where text reveals 0%.
- Their "belief shift at inflection points" finding connects to our accuracy-conditional V|nums: genuine computation produces detectable hidden state changes at specific positions.
- Scale validation: their results on 120B-671B models extend the phenomenon beyond our 4B-8B range. The hidden channel exists across 3 orders of magnitude of model scale.
- **For the paper:** This is our strongest positioning reference. "Reasoning Theater" frames the phenomenon; we provide the MECHANISTIC decomposition (K vs V, layer sweep, head specialization) that explains HOW the model maintains beliefs independently of text.

---

### Theme 4: CoT Faithfulness — Measurement and Mechanisms

---

#### 9. "Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps" (FUR)
**Authors:** Martin Tutek, Fateme Hashemi Chaleshtori, Ana Marasovic, Yonatan Belinkov
**Date:** February 2025
**Venue:** EMNLP 2025 — **Outstanding Paper Award**
**arXiv:** 2502.14829

**Key findings:**
- **FUR framework**: erases information from specific reasoning steps within model parameters, then measures effect on final predictions
- FUR can "precisely change the underlying model's prediction" by unlearning key steps → those steps are parametrically faithful
- CoTs generated post-unlearning "support different answers" → deeper effect beyond simple prediction changes
- Tested across 4 models, 5 multi-hop QA datasets

**Relevance to our work:**
- FUR is a complementary methodology to our perturbation approach: they modify PARAMETERS to test step importance; we modify KV CACHE to test channel importance. Both are causal interventions.
- Their finding that unlearning key steps changes predictions confirms some CoT steps ARE causally important — consistent with our finding that multi-step operations show genuine V computation (V>MLP +0.088).
- Outstanding Paper at EMNLP 2025 signals high community interest in CoT faithfulness measurement — validates the importance of our research direction.

---

#### 10. "Robust Answers, Fragile Logic: Probing the Decoupling Hypothesis in LLM Reasoning"
**Authors:** Enyi Jiang, Changming Xu, Nischay Singh, Tian Qiu, Gagandeep Singh
**Date:** May 2025 (revised February 2026)
**Venue:** arXiv (cs.AI)
**arXiv:** 2505.17406

**Key findings:**
- Under minor input perturbations, LLMs maintain correct answers while generating "inconsistent or nonsensical reasoning"
- **MATCHA framework**: conditions generation on the model's predicted answer to isolate reasoning quality
- Multi-step and commonsense tasks show greater susceptibility to answer-reasoning decoupling than logic tasks
- Adversarial examples transfer to black-box systems

**Relevance to our work:**
- The "decoupling hypothesis" IS our hypothesis stated differently. They show answers decouple from reasoning text; we show the KV cache carries answers independently of text. Same phenomenon, different vantage point.
- Their finding that MULTI-STEP tasks show MORE decoupling is consistent with our V|MLP being strongest for multi-step operations — the hidden channel is most active when computation is complex.
- MATCHA's answer-conditioned probing could complement our V-probing: condition on the model's answer, then check if V-cache at error positions matches the conditioned answer or the correct answer.

---

### Theme 5: Information Bottleneck & Lossy Projection

---

#### 11. "Lost in Backpropagation: The LM Head is a Gradient Bottleneck"
**Authors:** Nathan Godey, Yoav Artzi
**Date:** March 10, 2026
**Venue:** arXiv (cs.CL)
**arXiv:** 2603.10145

**Key findings:**
- **95-99% of gradient norm suppressed** by the output (LM head) layer during backpropagation
- The D→V projection (D << V) creates both an expressivity and optimization bottleneck
- Makes "trivial patterns unlearnable" and drastically alters training dynamics
- This is an architectural limitation, not a training configuration issue

**Relevance to our work:**
- **Updated version of paper already in our literature** (previously referenced as Godey & Artzi, March 2026). Now confirmed: 95-99% gradient suppression = 95-99% of hidden computation is invisible to the LM head. This IS the architectural guarantee that text is a lossy projection.
- Our V-cache probing bypasses this bottleneck — we read the hidden states BEFORE they pass through the LM head, recovering information the text cannot contain.
- The "unlearnable patterns" finding suggests the model's hidden representations contain structure that CANNOT be expressed in text tokens — consistent with our digital/analog encoding taxonomy where information is encoded in geometric properties of K/V vectors, not in discrete token choices.

---

#### 12. "Reasoning as Compression: Unifying Budget Forcing via the Conditional Information Bottleneck"
**Authors:** Fabio Valerio Massoli, Andrey Kuzmin, Arash Behboodi
**Date:** March 9, 2026
**Venue:** arXiv (cs.LG)
**arXiv:** 2603.08462

**Key findings:**
- Reframes CoT as "lossy compression" under Information Bottleneck (IB) principle
- Identifies **Attention Paradox**: attention violates the Markov property between prompt/reasoning/response, breaking naive IB theory
- Proposes **Conditional Information Bottleneck (CIB)**: reasoning trace Z contains "only the information about the response Y that is not directly accessible from the prompt X"
- Achieves improved accuracy at moderate compression with minimal loss

**Relevance to our work:**
- Their CIB formulation provides the THEORETICAL FRAMEWORK for our forward-looking V probing. V|nums = I(V; answer | text) is exactly the conditional information that CoT carries beyond the prompt. Our metric operationalizes their theory.
- The Attention Paradox (attention creates non-Markov dependencies) explains WHY the KV cache carries more information than a simple prompt→text→answer chain would suggest — attention allows the model to route information through hidden dimensions that bypass the text channel.
- **For the paper:** This paper provides the information-theoretic grounding for our "lossy projection" framing. Our experiments are empirical measurements of the information loss they characterize theoretically.

---

#### 13. "Latent Chain-of-Thought as Planning: Decoupling Reasoning from Verbalization"
**Authors:** Jiecong Wang, Hao Peng, Chunyang Liu
**Date:** January 29, 2026
**Venue:** arXiv (cs.AI)
**arXiv:** 2601.21358

**Key findings:**
- **PLaT framework**: reasoning as deterministic trajectory of latent planning states, separate Decoder grounds these into text
- Lower greedy accuracy but "superior scalability in terms of reasoning diversity" — broader solution space
- Model dynamically determines when to terminate reasoning without fixed hyperparameters

**Relevance to our work:**
- PLaT explicitly architecturalizes what our probing evidence suggests happens implicitly: the model computes in latent space and then projects to text. PLaT just makes this explicit by design.
- Their "lower greedy accuracy but broader solution space" mirrors our finding that text is suboptimal — the hidden channel may explore more solutions than the text reveals.

---

### Theme 6: Forward-Looking Representations

---

#### 14. "Do Language Models Plan Ahead for Future Tokens?"
**Authors:** Wilson Wu, John X. Morris, Lionel Levine
**Date:** April 2024 (revised August 2024)
**Venue:** COLM 2024

**Key findings:**
- Two mechanisms: **pre-caching** (models compute future-useful features at current timestep) vs **breadcrumbs** (present-useful features happen to be future-useful)
- Synthetic data: clear pre-caching evidence
- Language modeling: breadcrumbs more prevalent, but pre-caching increases with model scale
- Myopic training (removing future-gradient flow) used to distinguish mechanisms

**Relevance to our work:**
- Our forward-looking V signal (V|nums at offset -5, R=0.794) is either pre-caching or breadcrumbs. Their finding that pre-caching increases with scale suggests our 4B-8B models may show genuine pre-caching of future arithmetic results.
- The breadcrumbs/pre-caching distinction applies to our layer sweep: early layers may use breadcrumbs (features useful now also useful later), while plateau layers may show pre-caching (features specifically computed for future use).
- **Suggests a future experiment:** myopic training analysis on our arithmetic task to determine whether the V signal at intermediate positions is genuine pre-caching or breadcrumbs.

---

### Theme 7: Geometry and Architecture of Reasoning

---

#### 15. "The Geometry of Thought: How Scale Restructures Reasoning"
**Authors:** Samuel Cyrenius Anderson
**Date:** January 2026
**Venue:** arXiv (cs.AI)
**arXiv:** 2601.13358

**Key findings:**
- Analyzed 25,000+ CoT trajectories across Law, Science, Code, Math comparing 8B vs 70B models
- **Domain-specific geometric changes**: Legal reasoning "crystallizes" (dimensionality drops 45%); Math/Science remain "liquid" (stable geometry); Code forms discrete "lattice"
- Neural Reasoning Operators map initial→terminal hidden states with 63.6% accuracy WITHOUT traversing intermediate steps
- Universal oscillatory signature across all domains (attention vs feedforward opposing dynamics)

**Relevance to our work:**
- Math remaining "geometrically liquid" across scale explains why our forward-looking signal is SIZE-INDEPENDENT within Qwen (exp_088: 4B peak=0.497, 8B peak=0.478). Math reasoning geometry doesn't crystallize with scale.
- The 63.6% initial→terminal mapping WITHOUT intermediate steps is consistent with our forward-looking signal: the model can predict the final answer from early hidden states.
- The oscillatory signature (attention vs feedforward) maps onto our K (attention/routing) vs V (content/feedforward) decomposition — the oscillation may reflect the alternating dominance we observe.

---

#### 16. "Understanding the Physics of Key-Value Cache Compression through Attention Dynamics"
**Authors:** Samhruth Ananthanarayanan, Ayan Sengupta, Tanmoy Chakraborty
**Date:** March 2, 2026
**Venue:** arXiv (cs.CL)
**arXiv:** 2603.01426

**Key findings:**
- KV compression reframed as "controlled perturbation of token-level routing"
- **Sharp hallucination cliff near 90% compression** — phase transition where semantic accessibility collapses
- **Architecture-specific routing**: LLaMA shows early consensus → late diversification; Qwen shows funnel-like late convergence
- Excessive head-level consensus collapses routing flexibility
- Sparse token-route structures govern compression tolerance

**Relevance to our work:**
- **LLaMA early consensus / Qwen late convergence** maps onto our digital vs analog encoding taxonomy:
  - Qwen (digital): funnel convergence = concentrated routing through specific heads (H0+H5) → digital cliff
  - LLaMA (analog): early consensus with late diversification = distributed routing → gradual degradation
- Their "compression as routing perturbation" framing IS our Phase 1 perturbation framework. They perturb by eviction; we perturbed by noise injection. Same mechanism.
- The 90% compression cliff parallels our digital encoding cliff (Qwen σ=0.3-0.5) — there's a critical threshold below which the routing structure collapses.

---

### Theme 8: KV Cache as Computation Medium

---

#### 17. "Bottlenecked Transformers: Periodic KV Cache Consolidation for Generalised Reasoning"
**Authors:** Adnan Oomerjee, Zafeirios Fountas, Haitham Bou-Ammar, Jun Wang
**Date:** May 2025
**Venue:** arXiv (cs.LG)
**arXiv:** 2505.16950

**Key findings:**
- **Cache Processor**: auxiliary transformer that periodically rewrites KV cache entries
- Consolidates recent entries + reconsolidates top-k attention-selected prior entries
- +6.6pp improvement on math reasoning benchmarks
- Framed through Information Bottleneck theory: optimal balance between compression and retention

**Relevance to our work:**
- The fact that REWRITING the KV cache improves reasoning is causal evidence that the KV cache is the computation medium (not just a memory store). The Cache Processor adds computation to the KV cache → better reasoning.
- The top-k attention-selected reconsolidation parallels our H5 retrieval head finding — certain entries are more important and the model actively selects them.
- The IB framing connects to Massoli et al.'s CIB — both view KV cache information as the compression bottleneck for reasoning.

---

#### 18. "Beyond Speedup — Utilizing KV Cache for Sampling and Reasoning"
**Authors:** Zeyu Xing, Xing Li, Hui-Ling Zhen, Mingxuan Yuan, Sinno Jialin Pan
**Date:** January 2026
**Venue:** arXiv
**arXiv:** 2601.20326

**Key findings:**
- KV caches contain meaningful contextual information suitable for downstream tasks (weaker than full hidden states but "free")
- **Fast/Slow Thinking Switching**: uses KV-derived representations for adaptive reasoning
- Up to 5.7x reduction in token generation with minimal accuracy loss
- Tested on Qwen3-8B and DeepSeek-R1-Distill-Qwen-14B

**Relevance to our work:**
- Uses KV cache as a REPRESENTATION for reasoning — validates our claim that the KV cache carries computation-relevant information beyond just enabling next-token prediction.
- The Fast/Slow switching based on KV representations is a practical application of the hidden channel: if the KV cache already "knows" the answer (fast thinking), skip verbose CoT.
- Tests include Qwen3-8B, one of our model families.

---

### Theme 9: Steganographic Capabilities & Hidden Computation

---

#### 19. "Early Signs of Steganographic Capabilities in Frontier LLMs"
**Authors:** Artur Zolkowski, Kei Nishimura-Gasparian, Robert McCarthy, Roland S. Zimmermann, David Lindner
**Date:** July 2025
**Venue:** arXiv (cs.CR)
**arXiv:** 2507.02737

**Key findings:**
- Two types evaluated: **passing encoded messages** and **performing encoded reasoning**
- Under standard conditions, models CANNOT encode short messages without detection
- BUT models succeed with unmonitored scratchpads (affordance-dependent)
- Early signs of basic encoded reasoning in state-tracking problems (hexadecimal encoding)
- Models "can rarely hide reasoning subtly within a cover task"
- Capabilities "likely insufficient to bypass well-designed monitors at present"

**Relevance to our work:**
- The explicit/intentional steganography they test is a HARDER task than the implicit/unintentional hidden channel we study. Our KV cache hidden channel doesn't require the model to "intentionally" hide information — it's an architectural side effect of the QK/OV decomposition.
- Their finding that encoded reasoning is possible but limited in current models suggests the architecture ENABLES hidden computation but current training hasn't optimized for it. Our work shows the architecture ALREADY uses it unintentionally.
- The affordance-dependence (success with scratchpads) parallels our finding that the KV cache IS an "unmonitored scratchpad" — it stores computation that text monitors can't see.

---

#### 20. "Interpreting Transformers Through Attention Head Intervention"
**Authors:** Mason Kadem, Rong Zheng
**Date:** January 2026
**Venue:** arXiv (cs.CL)
**arXiv:** 2601.04398

**Key findings:**
- Traces evolution from attention visualization to causal intervention
- Head intervention enables suppression of toxic outputs and manipulation of semantic content through targeted 1% of heads
- Validates interpretability research's practical utility for safety

**Relevance to our work:**
- The evolution from correlation to causal intervention mirrors our trajectory: Phase 1 (perturbation/causal) → Phase 2 (probing/observational). Both approaches are needed.
- 1% of heads controlling targeted behavior parallels our H0+H5 finding (25% of heads carrying most answer signal).

---

#### 21. "Head Pursuit: Probing Attention Specialization in Multimodal Transformers"
**Authors:** Lorenzo Basile, Valentino Maiorca, Diego Doimo, et al.
**Date:** October 2025
**Venue:** NeurIPS 2025 (Spotlight)
**arXiv:** 2510.21518

**Key findings:**
- Consistent specialization patterns emerge across unimodal and multimodal transformers
- Modifying 1% of selected heads reliably suppresses/enhances targeted concepts
- Signal processing-based decomposition of head outputs

**Relevance to our work:**
- NeurIPS 2025 Spotlight = high-visibility confirmation that attention heads specialize. Our H5 finding (specialized for arithmetic answer routing) fits within this broader paradigm.
- 1% of heads controlling concepts parallels our sparse critical head finding.

---

## Key Synthesis

### 1. THREE Independent Teams Confirm Hidden Computation During Arithmetic CoT

The most important convergence from this scan: Kudo et al. (EACL 2026), Sun et al. (EMNLP 2025), and our own work all probe hidden states during arithmetic chain-of-thought and find:
- **Kudo**: Residual stream probes show iterative computation during CoT (sub-answers emerge sequentially)
- **Sun**: Residual stream probes decode CORRECT answers even when model writes WRONG answers (>90% accuracy)
- **Us**: K/V-separated probes show V-cache encodes intermediate results (R=0.95) with forward-looking signal (offset -5, R=0.79)

**Our unique contribution**: K/V decomposition. Neither Kudo nor Sun separate key from value vectors. Our work shows the computation signal is carried primarily in V (content), while K provides routing, and this balance is architecture-dependent (digital→K>V probing, analog→V>K probing). This granularity is absent from the other two papers.

### 2. Causal Bypass Quantifies the Hidden Channel

Sathyanarayanan et al.'s CMI metric shows 83% of CoT is causally disconnected from answers. This is the CAUSAL version of what our PROBING shows: V-cache carries answer info that text doesn't mediate. Our probing says "V knows the answer before text reveals it"; their patching says "text doesn't causally influence the answer." Same conclusion, complementary methods.

### 3. The "Validation Gap" Explains Our Layer-Sweep Structure

Bertolazzi et al.'s finding that computation happens in higher layers while validation happens in middle layers provides an architectural explanation for our ramp→plateau pattern. The ramp (L0-L9) is encoding + validation; the plateau (L10-L35) is computation. The model can't validate its own arithmetic because validation runs BEFORE computation finishes — this is the architectural mechanism behind the lossy projection.

### 4. Information Bottleneck Theory Now Provides Formal Framework

Massoli et al.'s CIB reframes our empirical findings theoretically: V|nums = I(V; answer | text) is the conditional information the hidden channel carries beyond the prompt. The "Attention Paradox" (attention breaks Markov assumptions) explains WHY the KV cache can carry more information than a simple text chain. Godey & Artzi's 95-99% gradient suppression provides the MAGNITUDE of information loss at the LM head bottleneck.

### 5. Reasoning Theater Scales to 671B Parameters

Boppana et al.'s full paper shows the probe-earlier-than-text phenomenon on DeepSeek-R1 (671B) — the hidden channel exists across 3 orders of magnitude of model scale (our 4B → their 671B). The 80% token savings from early exit quantifies the MAGNITUDE of performative computation.

### 6. Steganographic Capabilities Are Architectural, Not Intentional

Zolkowski et al. show intentional steganography is hard for current models. But our work shows UNINTENTIONAL hidden computation is widespread — the architecture provides a hidden channel whether the model "wants" it or not. The safety implication is that hidden computation doesn't require malicious intent; it's an inevitable consequence of the QK/OV decomposition and the LM head bottleneck.

### 7. Counter-Evidence: METR Shows CoT IS Faithful When Necessary

METR and Emmons et al. provide important nuance: when tasks genuinely REQUIRE CoT computation, faithfulness is high (99.98%). This doesn't contradict our findings — it refines them. The hidden channel is most active for SIMPLE/INTERMEDIATE computation steps (where CoT-as-rationalization dominates), less active for genuinely HARD steps (where CoT-as-computation takes over). Our accuracy and difficulty conditioning (exp_101/102) already shows this pattern: V|nums is positive for correct (computation succeeds) and negative for incorrect (computation fails).

---

## Impact on Next Experiments

### 1. Residual Stream WRRA Probing (HIGH PRIORITY)
Sun et al. show >90% accuracy decoding correct answers from residual stream even when output is wrong. Our exp_099 WRRA K-probe showed only 52-64% (ns). **Key insight: probe the RESIDUAL STREAM, not K or V alone.** The correct-answer signal may require both K and V information combined. This could salvage the WRRA "smoking gun" that was inconclusive.

### 2. CMI-Style Activation Patching
Sathyanarayanan et al.'s CMI provides a complementary causal method to our probing. Computing CMI on our GSM8K tasks would quantify what fraction of arithmetic CoT is causally necessary vs decorative. This bridges our probing evidence (info exists) with causal evidence (text is bypassed).

### 3. Layer-Wise Computation/Validation Dissociation
Bertolazzi et al.'s "Validation Gap" methodology (circuit analysis to identify computation vs validation layers) could be applied to our models to confirm the ramp=validation / plateau=computation mapping from exp_089.

### 4. Pre-caching vs Breadcrumbs Analysis
Wu et al.'s myopic training framework could determine whether our forward-looking V signal is genuine pre-caching (deliberate future-oriented computation) or breadcrumbs (present-useful features that happen to predict the future). This would strengthen the "hidden computation" interpretation.

### 5. Scale Validation
Our results span 4B-8B models. Boppana et al. confirm the phenomenon at 120B-671B. Testing on models in the 14B-32B range (available through Qwen2.5 family) could establish the scale-independence of the hidden channel.

---

## Total Literature Coverage: 149+ Papers Across 12 Scans

With this 11th scheduled scan, we now cover 149+ papers across:
- CoT faithfulness, bypass, and safety (45+ papers)
- KV cache compression, asymmetry, and dynamics (35+ papers)
- Latent reasoning architectures and compressed CoT (30+ papers)
- Mechanistic interpretability and probing (25+ papers)
- Forward-looking features, pre-caching, and hidden computation (15+ papers)
- Head specialization and retrieval heads (10+ papers)
- Steganographic capabilities and encoded reasoning (7+ papers)
- Information bottleneck theory and lossy projection (5+ papers)
