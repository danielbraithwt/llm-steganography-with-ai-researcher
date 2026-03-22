# Literature Scan — Cycle 80: Phase 2 Convergence, Counter-Evidence, and Methodological Advances

**Date:** 2026-03-22
**Scan type:** Scheduled (every 10 cycles)
**Focus:** Papers validating/challenging Phase 2 natural channel usage findings, KV cache
interpretability via SAEs, CoT faithfulness mechanistic analysis, information bottleneck
quantification, and steganographic computation updates.

**Previous scans:** Cycles 10, 20, 30, 40, 50, 60, 67 (targeted), 70

---

## New Papers (13 total)

### 1. "Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought" (v3 UPDATE)
**Authors:** Siddharth Boppana, Annabel Ma, Max Loeffler, Raphael Sarfati, Eric Bigelow, Atticus Geiger, Owen Lewis, Jack Merullo
**Date:** March 5-12, 2026
**Venue:** arXiv:2603.05488 (v3)

**Key updates from v1 (already in our lit notes from cycle 20):**
- Now tested on **DeepSeek-R1 671B** and **GPT-OSS 120B** (not just the original models)
- **TASK-DEPENDENT**: MMLU → performative reasoning (answer decodable from internal activations
  long before text reveals it), GPQA-Diamond → genuine reasoning (probes improve gradually,
  aligned with text computation)
- Probe-guided early termination saves **80% tokens on MMLU**, **30% on GPQA-D**
- "Observable reasoning behaviors (backtracking, insights) correlate with genuine belief shifts"
- Three methods: activation probing, forced early answering, CoT monitor

**Relevance to our work:**
- The task-dependence finding is crucial for interpreting our Phase 2 results on GSM8K
- GSM8K arithmetic is intermediate: some steps are genuine computation (like GPQA-D) while
  scaffolding text may be performative (like MMLU)
- Our forward-looking V probing (R≈0.3 partial) is consistent: V at computation positions
  encodes the answer because genuine computation happens there, but the partial R isn't 1.0
  because not all positions carry forward-looking info (some are performative)
- Their activation probing approach validates our probing methodology

---

### 2. "Truth as a Trajectory: What Internal Representations Reveal About LLM Reasoning"
**Authors:** Hamed Damirchi, Ignacio Meza De la Jara, Ehsan Abbasnejad, Afshar Shamsi, Zhen Zhang, Javen Shi
**Date:** March 1, 2026
**Venue:** arXiv:2603.01326

**Key findings:**
- Analyzes transformer inference as "an unfolded trajectory of iterative refinements"
- **Layer-wise geometric displacement** (changes between layers) better predicts reasoning
  quality than static activations at any single layer
- "Effectively mitigates reliance on static lexical confounds" — probes learn reasoning
  dynamics, not surface patterns
- Outperforms conventional probing on commonsense reasoning, QA, and toxicity detection
- Tested on dense and MoE architectures

**Relevance to our work:**
- **Methodological innovation**: Instead of probing activations at one layer, probe the
  TRAJECTORY across layers. This could strengthen our forward-looking probing by measuring
  how V-representations change through layers rather than just their values at L27/L35
- Addresses the "lexical confound" concern from our exp_079 — trajectory-based probing
  avoids learning problem-number patterns because it focuses on CHANGES, not static values
- Could inspire a new experiment: does the V trajectory toward the final answer differ
  between correct and incorrect problems?

---

### 3. "Fragile Thoughts: How Large Language Models Handle Chain-of-Thought Perturbations"
**Authors:** Ashwath Vaithinathan Aravindan, Mayank Kejriwal
**Date:** February 11 / March 6, 2026
**Venue:** arXiv:2603.03332

**Key findings:**
- Tests 5 structured CoT perturbation types on **13 models spanning 3B to 1.5T parameters**
- **MathError**: 50-60% accuracy loss in small models; significant scaling benefits
- **UnitConversion**: 20-30% loss even in largest models (most persistent vulnerability)
- **ExtraSteps**: 0-6% degradation regardless of scale (remarkably robust)
- **SkippedSteps**: ~15% loss (intermediate)
- **Sycophancy**: ~7% loss for small models
- Scaling follows power-law patterns but provides limited defense for dimensional reasoning

**Relevance to our work:**
- **DIRECTLY validates Experiment B (paraphrase disruption) design** — CoT IS fragile to
  text perturbations, and the fragility depends on perturbation type
- MathError (changing numbers) is catastrophic — expected, since numbers are critical
- UnitConversion (changing dimensional reasoning) is persistent even at scale — this suggests
  that specific token choices DO carry hidden information about dimensional/unit tracking
- ExtraSteps (adding irrelevant text) is harmless — suggests the model can ignore non-essential
  tokens, reading past them via attention routing
- Our paraphrase disruption is intermediate: preserving numbers/operations but changing surface
  text. Based on these results, expect moderate disruption (10-30% accuracy loss)
- The **scaling results** suggest our 4B model will be more vulnerable than larger models,
  which is useful for detecting the effect but may not generalize to frontier models

---

### 4. "Unlocking the Address Book: Dissecting the Sparse Semantic Structure of LLM KV Caches via SAEs"
**Authors:** Qingsen Ma, Dianyun Wang, Jiaming Lyu, et al.
**Date:** December 11, 2025
**Venue:** arXiv:2512.10547

**Key findings:**
- **Top-K Sparse Autoencoders (SAEs)** applied to decompose KV cache into interpretable
  "semantic atoms"
- **K vs V asymmetry confirmed via SAE decomposition:**
  - Keys = "highly sparse routers dominated by a 'Semantic Elbow'"
  - Values = "dense content payloads requiring a larger budget"
- **Dual-Budget Strategy:** Different compression budgets for K (sparse, needs fewer atoms)
  vs V (dense, needs more atoms) — dynamic allocation
- Functional stratification: shallow layers = lexical patterns, middle = syntactic backbone,
  deep = polysemy resolution via orthogonal semantic features
- Tested on Yi-6B, Mistral-7B, Qwen2.5-32B

**Relevance to our work:**
- **4th independent confirmation of K=routing, V=content:**
  1. Perturbation (our Phase 1): K>V causality
  2. Quantization engineering (AsymKV, KV-AdaQuant): K needs more precision
  3. Spectral geometry (our exp_062): K lower rank, higher top-1 energy
  4. **SAE decomposition (this paper): K sparse routing, V dense content**
- The "Semantic Elbow" in K-cache = our finding that K has lower effective rank and higher
  top-1 singular value energy (3.5-4.1x). The SAE confirms this isn't just geometry — it
  reflects FUNCTIONAL sparsity (few routing dimensions matter)
- "Dense content payloads" in V = our Phase 2 finding that V probes decode answers better
  than K probes (V=content, K=routing)
- Their Dual-Budget approach validates our compression findings: K needs precision (digital
  accuracy), V needs capacity (content storage)

---

### 5. "Mechanistic Evidence for Faithfulness Decay in Chain-of-Thought Reasoning"
**Authors:** Donald Ye, Max Loffgren, Om Kotadia, Linus Wong
**Date:** February 4, 2026
**Venue:** arXiv:2602.11201

**Key findings:**
- Introduces **NLDD (Normalized Logit Difference Decay)**: corrupts individual reasoning steps,
  measures confidence change
- **Reasoning Horizon (k*) at 70-85% of chain length** — beyond this, reasoning tokens have
  "little or negative effect on the final answer"
- **Critical finding: "models can encode correct internal representations while completely
  failing the task"** — correct answers exist internally even when the task fails
- Tested on 3 model families across syntactic, logical, and arithmetic tasks
- Faithfulness breakdown is gradual, not sudden

**Relevance to our work:**
- **"Correct internal representations despite task failure" = our WRRA finding (exp_078)**
  At error positions, V probes predict the CORRECT value (71.4%, p=0.039). Ye et al. confirm
  this is a general phenomenon, not specific to our probing setup.
- Reasoning Horizon at 70-85% is consistent with our Phase 1 finding that late positions carry
  answer-specific information (positional dissociation)
- **Our Phase 1 tested the Reasoning Horizon prediction (exp_041-043): we found NO sharp
  phase transition, just a linear dissociation gradient.** Ye et al.'s "gradual breakdown"
  is actually consistent with our finding — their k*=70-85% may reflect a smooth gradient,
  not a cliff.
- **NLDD methodology could be useful for Phase 2:** instead of probing V for the answer,
  corrupt individual reasoning steps and measure whether the KV cache routing changes.

---

### 6. "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful"
**Authors:** Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan,
Neel Nanda, Arthur Conmy
**Date:** March 2025 (ICLR 2025 Workshop)
**Venue:** arXiv:2503.08679

**Key findings:**
- CoT unfaithfulness found even on realistic, unbiased prompts (no explicit bias injection)
- Two types: (1) implicit post-hoc rationalization, (2) unfaithful illogical shortcuts
- Unfaithfulness rates: GPT-4o-mini 13%, Haiku 3.5 7%, frontier models <1%
  (DeepSeek R1 0.37%, Sonnet 3.7 with thinking 0.04%)
- "Thinking" models are more faithful but not entirely

**Relevance to our work:**
- Confirms unfaithfulness exists even in production models without adversarial prompting
- The low rates in frontier models (<1%) suggest the hidden channel may be more subtle
  than overt unfaithfulness — our probing approach detects computation that isn't in the
  text, which may exist even at 0.04% unfaithfulness rate
- Frontier model faithfulness improvements may reflect better text-computation alignment,
  not elimination of hidden computation

---

### 7. "LLMs Faithfully and Iteratively Compute Answers During CoT"
**Authors:** Keito Kudo, Yoichi Aoki, Tatsuki Kuribayashi, et al.
**Date:** December 2024, revised March 2026
**Venue:** arXiv:2412.01113

**Key findings:**
- Probing shows answers emerge iteratively during CoT, not before generation begins
- Claims CoT is "faithful reflections of the model's internal computation"
- Intermediate results become decodable during text generation
- Uses controlled arithmetic tasks

**Relevance to our work:**
- **COUNTER-EVIDENCE for the strongest version of our hypothesis.** If answers emerge
  iteratively during CoT, the text IS participating in the computation, not just a facade.
- **BUT compatible with our moderate findings:** Our forward-looking V probing (partial
  R≈0.29 with GroupKFold) shows V at computation positions carries SOME forward-looking
  info beyond the current step, but R≈0.29 (not R≈1.0). This is consistent with:
  - The model computes iteratively during CoT (Kudo et al.)
  - BUT the text only captures ~30% of the forward-looking computation (our finding)
  - The remaining ~70% is local computation faithfully reflected in text
- **The key question is not "does the text matter" but "does the text capture ALL the
  computation."** Both Kudo et al. and our work can be simultaneously true:
  the model faithfully computes during CoT AND the text is a lossy projection.
- Their controlled arithmetic is simpler than GSM8K's multi-step word problems,
  which may have more room for hidden forward-looking computation.

---

### 8. "Invisible Safety Threat: Malicious Finetuning for LLM via Steganography"
**Authors:** Guangnian Wan, Xinyin Ma, Gongfan Fang, Xinchao Wang
**Date:** March 9, 2026
**Venue:** arXiv:2603.08104

**Key findings:**
- Finetuning embeds malicious Q&A within benign cover interactions via steganography
- "All stegotexts containing malicious content are incorrectly classified as safe" by
  Llama-Guard-3-8B
- Tested on GPT-4.1, Llama-3.3-70B, Phi-4, Mistral-Small-24B
- Evades OpenAI finetuning API safety classifiers

**Relevance to our work:**
- Demonstrates practical exploitation of hidden channels in LLMs for safety bypass
- Their steganography is at the TEXT level (encoded in token choices). Our finding is at
  the KV CACHE level (computation in routing patterns). Both are steganographic channels
  but at different abstraction levels.
- Safety implication: even if text-level steganography is detected (as Llama-Guard fails to),
  KV-level hidden computation is fundamentally harder to monitor because it operates in the
  model's internal representation space, not in human-readable text.

---

### 9. "Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought"
**Authors:** Yuyi Zhang, Boyu Tang, Tianjie Ju, Sufeng Duan, Gongshen Liu
**Date:** December 25, 2025
**Venue:** arXiv:2512.21711

**Key findings:**
- COCONUT-style latent tokens are "uninterpretable placeholders rather than encoding
  faithful reasoning"
- "Pseudo-reasoning mechanisms: generates plausible traces that conceal shortcut dependence"
- Latent tokens show "minimal sensitivity to steering and lack reasoning-critical information"
- Under adversarial conditions, COCONUT "consistently exploits dataset artifacts"
- Tested on MMLU and HotpotQA

**Relevance to our work:**
- **IMPORTANT COUNTER-EVIDENCE for latent reasoning.** COCONUT's continuous chain-of-thought
  tokens do NOT carry meaningful computation — they exploit shortcuts.
- **BUT this is about ARTIFICIAL latent tokens** (trained with Coconut objective), not natural
  KV cache computation. The distinction matters:
  - Coconut replaces text tokens with continuous vectors during training → these vectors may
    learn surface patterns rather than computation
  - Our KV cache computation is a NATURAL byproduct of standard autoregressive generation →
    K-routing and V-content emerge from the attention mechanism, not from special training
- The caution is valid: we should not assume hidden representations carry meaningful
  computation just because they exist. Our Phase 2 probing methodology (GroupKFold,
  residualization against problem context) is designed to address exactly this concern.

---

### 10. "Early Signs of Steganographic Capabilities in Frontier LLMs"
**Authors:** Artur Zolkowski, Kei Nishimura-Gasparian, Robert McCarthy, et al.
**Date:** July 2025 (ICLR 2026 poster)
**Venue:** arXiv:2507.02737

**Key findings:**
- Current models CANNOT sustain hidden reasoning for complex math at the text level
- Can encode with additional affordances (scratchpad, pre-arranged schemes)
- "Likely insufficient to bypass well-designed monitors at present"
- Evaluated across frontier models

**Relevance to our work:**
- Text-level steganography is rudimentary for math → consistent with our finding that the
  hidden channel operates at the KV cache level, not the text level
- Our KV-level computation is architecturally fundamental (attention mechanism), not an
  intentional encoding scheme → harder to detect AND harder to mitigate
- Updated from cycle 70 notes: now accepted at ICLR 2026 (poster)

---

### 11. "Fragile Thoughts" Mechanistic Extension: Shortcut Circuits in GPT-2 Small
**Authors:** Various (BlueDot Impact Technical AI Safety Sprint, January 2026)
**Source:** blog.bluedot.org

**Key findings:**
- Unfaithfulness is localized to specific "shortcut circuits," not diffuse
- **Faithful circuits in early layers** (L0-L1): L0H1, L0MLP, L0H6, L1H7
- **Shortcut circuits in later layers** (L7-L10): L7H6, L10MLP + 35 additional heads
- "Circuit Breakers" can force models toward faithful reasoning at inference time
- Causal activation patching isolates components at head-level granularity

**Relevance to our work:**
- **Early faithful, late shortcut** is consistent with our finding that early layers carry
  routing infrastructure and late layers optimize for text output (next-token prediction)
- Our answer head H5 operates at the KV cache level — it's a routing circuit. The shortcut
  circuits may use different KV routing patterns than faithful circuits
- GPT-2 Small is much smaller than our models, but the principle of circuit-level
  faithfulness localization is architecturally general

---

### 12. "The Information Bottleneck of Chain-of-Thought and How Latent CoT Overcomes It"
**Date:** October 2025
**Venue:** OpenReview (under review)

**Key findings:**
- **Quantifies the bottleneck:** ~O(log|V|) ≈ 15 bits per token vs ~460 bits per activation
  vector (FP16, 2560 dimensions)
- ~30x information compression from activations to text tokens
- For problems like pointer chasing and parity, standard CoT requires unnecessarily long
  chains; latent CoT with high-dimensional embeddings significantly reduces chain length
- **"Premature commitment" problem:** CoT's discrete symbolic nature forces high-certainty
  decisions at each step, causing exploration failure on hard problems

**Relevance to our work:**
- **Quantifies our "lossy projection" hypothesis:** the text captures ~15/460 = 3.3% of the
  information in a single activation vector. Even with multiple tokens, the cumulative text
  information grows linearly while each V vector independently encodes ~460 bits.
- Our cumV > cumText gap (+0.055-0.114 R) empirically measures this bottleneck: V vectors
  carry more answer-predictive information than the same number of text tokens
- The "premature commitment" problem may explain why our WRRA cases exist: the model commits
  to a computation path in the KV cache but writes a simplified/wrong version in text

---

### 13. "A Survey on Latent Reasoning"
**Authors:** Rui-Jie Zhu et al. (33 co-authors)
**Date:** July 2025
**Venue:** arXiv:2507.06203

**Key findings:**
- Taxonomy: activation-based recurrence, hidden state propagation, fine-tuning strategies,
  infinite-depth via masked diffusion
- Text CoT is "bandwidth-constrained" — latent reasoning overcomes this by operating in
  continuous space
- Comprehensive overview of COCONUT, Quiet-STaR, Latent-SFT, and newer approaches
- Positions latent reasoning as "the frontier of LLM capabilities"

**Relevance to our work:**
- Our Phase 1+2 work provides the MECHANISTIC FOUNDATION for why latent reasoning works:
  the KV cache already carries computation beyond the text (we measured it), and the K>V
  hierarchy reveals the routing-content architecture that latent reasoning exploits
- Updated from cycle 50 literature notes: the survey landscape has grown significantly

---

## Key Synthesis

### 1. K=Routing, V=Content Now Confirmed from 4 Independent Angles

| Angle | Source | K finding | V finding |
|-------|--------|-----------|-----------|
| **Perturbation** (us, Phase 1) | 5 models, 15 conditions | K>V causality (destroying routing catastrophic) | V immune at moderate levels |
| **Quantization** (AsymKV, KV-AdaQuant) | Mathematical proof + engineering | K needs more precision (larger spectral norms) | V tolerates 1-bit quantization |
| **Spectral geometry** (us, exp_062) | 2 models | K lower rank, higher top-1 energy (3.5-4.1x) | V higher effective rank |
| **SAE decomposition** (Ma et al. 2025) | Yi-6B, Mistral-7B, Qwen2.5-32B | K = "sparse routers with Semantic Elbow" | V = "dense content payloads" |

This is now one of the most independently confirmed findings in the KV cache literature.
Our unique contribution: the CAUSAL evidence (destroying K-routing destroys answers) and the
HEAD-LEVEL resolution (H5 primary answer-routing head on 2 model families).

### 2. Reasoning Theater Is Task-Dependent — GSM8K Is Intermediate

The updated Reasoning Theater paper (Boppana et al. v3) shows:
- Easy recall (MMLU): performative reasoning (early internal commitment, text is facade)
- Hard multi-step (GPQA-D): genuine reasoning (computation emerges with text)

GSM8K falls between these extremes:
- Some steps are genuine computation (arithmetic → internal answer emerges during computation)
- Some steps are scaffolding (text reformulation → performative, answer already determined)
- Our forward-looking V probing (partial R≈0.29) measures the "genuine computation" fraction
- The fraction is MODERATE (not 0 and not 1), consistent with GSM8K being intermediate difficulty

### 3. "Correct Internal Representations Despite Task Failure" — Convergent Evidence

Three independent groups now show models encode correct answers internally despite text errors:
1. **Sun et al. (EMNLP 2025):** >90% accuracy recovering correct answers from hidden states
   at error positions (Gemma 2 2B)
2. **Ye et al. (Feb 2026):** "Models encode correct internal representations while completely
   failing the task" (3 model families)
3. **Us (exp_078):** 71.4% correct-alignment at V-probe error positions (p=0.039, Qwen 4B)

Our unique contribution: K/V DECOMPOSITION at error positions (not just residual stream).

### 4. Information Bottleneck QUANTIFIED: ~30x Compression

The information bottleneck paper quantifies what our "lossy projection" hypothesis describes:
- ~460 bits per activation vector (FP16 at 2560 dims)
- ~15 bits per token (log2 of vocabulary size)
- **~30x information compression from hidden states to text**

Our empirical measurement (cumV > cumText by +0.055-0.114 R) captures the observable
consequence of this bottleneck at the KV cache level.

### 5. Counter-Evidence Must Be Taken Seriously

Two papers provide important counter-evidence:
1. **Kudo et al. (2412.01113):** CoT IS faithful — answers emerge iteratively during
   generation, not before. RESOLUTION: faithfulness and lossy projection are compatible.
   The model computes during CoT (faithful) but the text captures only part of the
   computation (lossy). Our partial R≈0.29 is consistent with both.
2. **Zhang et al. (2512.21711):** COCONUT latent tokens are "pseudo-reasoning" that
   exploit shortcuts. RESOLUTION: this is about ARTIFICIAL latent tokens, not natural
   KV cache computation. But the caution is valid — we must rigorously control for
   surface patterns in our probing studies (our GroupKFold + residualization does this).

### 6. Steganographic Computation: Architecture-Level vs Text-Level

The steganography literature (Zolkowski ICLR 2026, Wan March 2026) distinguishes:
- **Text-level steganography:** Encoding information in token choices. Currently rudimentary
  for math (Zolkowski). Can be exploited via finetuning (Wan).
- **KV-level computation:** Information in attention routing patterns. Our finding.
  Architecturally fundamental, not intentional encoding.

Our work bridges these: the KV cache carries computation that the text doesn't reflect.
This isn't intentional steganography — it's an architectural consequence of the attention
mechanism. But the SAFETY IMPLICATIONS are the same: monitoring text alone misses computation.

---

## Impact on Phase 2 Experiment Design

### Experiment B (Paraphrase Disruption) — Strengthened by Literature
- Fragile Thoughts (Aravindan & Kejriwal) validates that CoT IS fragile to text perturbations
- MathError (50-60% loss) >> ExtraSteps (0-6% loss) → token CONTENT matters more than length
- Our paraphrase (preserving numbers, changing surface text) should produce intermediate
  disruption (10-30%), which would be evidence for token-choice information
- Scaling law: effect may be stronger on our 4B model than on frontier models

### GroupKFold Methodology — Validated by Data Leakage Literature
- Multiple papers on data leakage in EEG/MRI/time-series confirm that subject-level
  (problem-level in our case) cross-validation is essential when multiple samples per
  subject share the same label
- Our exp_079 discovery of within-problem leakage is methodologically aligned with this
  literature

### Trajectory-Based Probing — New Direction
- Damirchi et al.'s "Truth as a Trajectory" suggests probing layer-to-layer CHANGES rather
  than static activations. This could avoid lexical confounds (like problem numbers) and
  focus on computation dynamics. Consider for a future experiment.

---

## Papers Already in Previous Scans (Updated Status)

- "Reasoning Theater" (Boppana et al.): v1 in cycle 20 → **v3 with task-dependence update**
- "Probing for Arithmetic Errors" (Sun et al.): covered in cycle 70 → no new update
- "KV Cache Steering" (Belitsky): covered in cycle 70 → no new update
- "NEST" (Karpov): covered in cycle 70 → no new update
- "Steganographic CoT under process supervision" (Skaf et al.): covered in cycle 70 → no update
- "V-information steganographic formalization" (Anwar et al.): covered in cycle 70 → no update
- "Knowing Before Saying" (Afzal et al.): covered in cycle 67 → published at ACL 2025 Findings
- "Early Signs of Steganographic Capabilities" (Zolkowski et al.): covered in cycle 70 →
  **now accepted at ICLR 2026 (poster)**
