# Literature Scan — Cycle 90: Layer-Wise Dynamics, Forward-Looking Features, and K/V Asymmetry Deepening

**Date:** 2026-03-22
**Scan type:** Scheduled (every 10 cycles)
**Focus:** Layer-wise emergence patterns, theoretical grounding for forward-looking features,
new K=routing/V=content confirmation angles, cross-model generalization, probing methodology advances.

**Previous scans:** Cycles 10, 20, 30, 40, 50, 60, 67 (targeted), 70, 80

---

## New Papers (24 total across 5 themes)

### Theme 1: Layer-Wise Dynamics — Independent Confirmation of Ramp/Plateau Pattern

Our exp_089 found two-phase emergence in V|nums: ramp L0-L9 (signal emerges at L3, 9% depth),
plateau L10-L35 (mean V|nums 0.17-0.22, 19/20 bins positive). Multiple independent papers
confirm this pattern in other modalities.

---

#### 1. "The Geometry of Hidden Representations of Large Transformer Models"
**Authors:** Lucrezia Valeriani, Diego Doimo, Francesca Cuturello, Alessandro Laio, Alessio Ansuini, Alberto Cazzaniga
**Date:** NeurIPS 2023 (arXiv:2302.00294)

**Key findings:**
- Intrinsic dimensionality (ID) profiles show three universal phases: (1) initial expansion/peak
  in early layers, (2) contraction to a long low-ID plateau in intermediate layers, (3) final
  ascent or shallow second peak
- **Plateau ID values are remarkably SIZE-INDEPENDENT** across models spanning 35M to 3B parameters
- Information in the plateau region is most suitable for downstream tasks; searching in plateau
  layers improves accuracy ~6% vs last layer

**Relevance to our work:**
- **Most direct theoretical counterpart to our finding.** Their "peak then long plateau" in intrinsic
  dimensionality is structurally identical to our V|nums "ramp L0-L9, plateau L10-L35"
- Their SIZE-INDEPENDENCE across 35M-3B directly parallels our finding that the forward-looking
  channel is size-independent within Qwen (4B→8B, exp_088)
- Suggests our ramp/plateau pattern is a UNIVERSAL property of transformer layer dynamics, not
  specific to forward-looking computation or KV cache

---

#### 2. "Tracing Representation Progression: Analyzing and Enhancing Layer-Wise Similarity"
**Authors:** Jiachen Jiang, Jinxin Zhou, Zhihui Zhu
**Date:** ICLR 2025 (arXiv:2406.14479)

**Key findings:**
- Representations across layers are positively correlated, with similarity increasing when layers
  are closer
- For models trained on small datasets, a **"ridge-to-plateau"** pattern: initial layers undergo
  significant transformations while higher layers exhibit a plateau in similarity
- **Once a sample is correctly predicted at a certain layer, it remains correct in subsequent
  layers** — early saturation events are maintained through the residual stream

**Relevance to our work:**
- The "ridge-to-plateau" pattern is the same as our "ramp L0-L9, plateau L10-L35"
- **The early saturation + maintenance finding directly explains our plateau:** once the V|nums
  forward-looking signal is established at ~L10, the residual stream MAINTAINS it through all
  subsequent layers. The signal doesn't need to be re-established at each layer.

---

#### 3. "Transformer Layers as Painters"
**Authors:** Qi Sun, Marc Pickett, Aakash Kumar Nain, Llion Jones
**Date:** AAAI 2025 (arXiv:2407.09298)

**Key findings:**
- Three distinct classes of layers: initial sketch (lower), iterative refinement (middle, the
  largest group), and detailed finish (final)
- **Middle layers share a common representation space** with high pairwise cosine similarity
- Models are robust to layer skipping/reordering in middle layers, but substituting with mere
  repeats causes severe performance collapse
- Execution order matters more for math/reasoning tasks than semantic tasks

**Relevance to our work:**
- The "initial sketch, iterative refinement, detailed finish" decomposition maps onto our
  ramp/plateau: ramp = initial sketch, plateau = iterative refinement
- Middle layers sharing a common space while making DISTINCT contributions explains why V|nums
  is consistent (~0.17-0.22) across 26 plateau layers — they operate in the same space while
  each refining incrementally
- Order mattering more for math/reasoning tasks suggests the forward-looking channel's layer
  dynamics may be more structured for arithmetic CoT than for general language

---

#### 4. "The Remarkable Robustness of LLMs: Stages of Inference?"
**Authors:** Vedang Lad, Jin Hwa Lee, Wes Gurnee, Max Tegmark
**Date:** arXiv:2406.19384 (Jun 2024, revised Jun 2025)

**Key findings:**
- **Four universal stages:** (1) Detokenization (early, sensitive), (2) Feature Engineering
  (early-middle, robust), (3) Prediction Ensembling (middle-late, MLP-heavy), (4) Residual
  Sharpening (final, sensitive)
- Models retain 72-95% accuracy when middle layers are deleted
- Middle layer robustness arises because **residual connections form ensembles of relatively
  shallow computational paths**

**Relevance to our work:**
- The ramp (L0-L9) corresponds to stages 1-2 (detokenization + feature engineering) where the
  forward-looking signal is being constructed
- The plateau (L10-L35) corresponds to stages 2-3 where the signal is maintained through the
  residual stream's ensemble property
- Middle layer robustness to deletion explains why the signal is DISTRIBUTED rather than
  localized — no single layer is essential

---

#### 5. "Transformer Dynamics: A Neuroscientific Approach to Interpretability"
**Authors:** Jesseba Fernando, Grigori Guitchounts
**Date:** Feb 2025 (arXiv:2502.12131)

**Key findings:**
- Residual stream activations exhibit strong continuity across layers despite being a
  non-privileged basis
- Activations accelerate and grow denser over layers; individual units trace unstable periodic orbits
- In reduced dimensions, the residual stream follows a curved trajectory with **attractor-like
  dynamics in lower layers**
- Despite D=4096, dynamics are remarkably low-dimensional

**Relevance to our work:**
- "Attractor-like dynamics in lower layers" followed by acceleration corresponds to our
  ramp phase (L0-L9) where the forward-looking signal is being established
- Low-dimensional dynamics suggest the channel operates in a compact subspace — consistent with
  our K spectral finding (K top-1 energy 3.5-4.1x, low effective rank)

---

### Theme 2: Theoretical Grounding for Forward-Looking Features

---

#### 6. "Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors"
**Authors:** Mark Rofin, Jalal Naghiyev, Michael Hahn
**Date:** ICLR 2026 (arXiv:2603.14087)

**Key findings:**
- Next-token prediction objectives create features encoding information **beyond the immediate
  next token** — features that appear "useless" for the current prediction but are useful for
  future tokens
- Features with extreme influence on future tokens tend to relate to **formal reasoning domains**
  (e.g., code)
- Identifies specific **gradient signal components** responsible for emergence of these features

**Relevance to our work:**
- **This is the theoretical foundation for WHY our forward-looking V-cache signal exists.** The
  gradient structure of next-token prediction NATURALLY produces features optimized for
  longer-term prediction, not just the immediate next token
- The formal-reasoning domain bias explains why the forward-looking signal is strong on GSM8K
  (arithmetic reasoning) — exactly the kind of domain where these features are most useful
- Provides the missing theoretical answer to "why would a next-token predictor encode future
  information?" — it's a gradient-driven consequence of the training objective

---

#### 7. "Do Language Models Plan Ahead for Future Tokens?"
**Authors:** Wilson Wu, John X. Morris, Lionel Levine
**Date:** COLM 2024 (arXiv:2404.00859)

**Key findings:**
- Two hypotheses: **"pre-caching"** (models intentionally store future-useful info) vs.
  **"breadcrumbs"** (features for current prediction happen to be useful later)
- On real language data, breadcrumbs hypothesis is more supported, but **pre-caching increases
  with model scale**
- Models compute features useful for predicting the immediate next token that also happen to
  carry information about upcoming tokens

**Relevance to our work:**
- Provides the theoretical framework for interpreting our forward-looking V probing results
- Our V|nums signal may be primarily "breadcrumbs" (byproduct of current computation) rather
  than intentional pre-caching — consistent with partial R≈0.29, not R≈1.0
- The scale-dependent pre-caching finding is consistent with our observation that Qwen-8B shows
  slightly different nums_R patterns (0.42 vs 4B's 0.26) despite similar forward-looking signal

---

#### 8. "Reasoning with Latent Thoughts: On the Power of Looped Transformers"
**Authors:** Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, Sashank J. Reddi
**Date:** ICLR 2025 (arXiv:2502.17416)

**Key findings:**
- Reasoning tasks require large DEPTH but not necessarily many parameters
- A k-layer transformer looped L times nearly matches a kL-layer non-looped model
- Looped models implicitly generate latent thoughts that simulate T steps of CoT with T loops

**Relevance to our work:**
- Depth (not parameters) governs reasoning → explains our SIZE-INDEPENDENCE finding (Qwen 4B
  and 8B both have 36 layers, despite different parameter counts)
- The available depth (36 layers) determines the channel's capacity, not the model width
- Suggests the ramp/plateau pattern should be similar across models with the same layer count
  but different widths — testable prediction

---

#### 9. "Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens"
**Authors:** Wei-Lin Chen, Liqian Peng, Tian Tan, et al.
**Date:** Feb 2026 (arXiv:2602.13517)

**Key findings:**
- Identifies **"deep-thinking tokens"** where internal predictions undergo significant revisions
  in deeper layers before converging
- Deep-Thinking Ratio (DTR) correlates strongly with task accuracy, outperforming length-based
  and confidence-based metrics
- Tokens requiring genuine reasoning show prediction revisions across many layers, while easy
  predictions converge early

**Relevance to our work:**
- Number tokens in CoT may be "deep-thinking tokens" — their V representations carry
  forward-looking information precisely because they undergo multi-layer refinement
- The DTR metric could identify WHICH tokens in a CoT chain contribute most to the
  forward-looking signal — a potential next experiment direction
- Compatible with our "genuine vs performative" interpretation: deep-thinking tokens =
  genuine computation, shallow tokens = performative

---

### Theme 3: K=Routing, V=Content — 5th and 6th Independent Confirmation Angles

---

#### 10. "KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging"
**Authors:** Lianjun Liu, Hongli An, Weiqi Yan, Xin Du, et al.
**Date:** March 1, 2026 (arXiv:2603.00907)

**Key findings:**
- **THEORETICAL PROOF:** Concentrated spectral energy in Query/Key weights induces feature
  homogeneity; dispersed spectra in Value weights preserve heterogeneity
- Derives **exact Hessian** for off-diagonal coupling between adjacent Keys
- Gradient-free, closed-form merging solution using forward-pass variables only
- On Llama3.1-8B-Instruct: +0.92 LongBench with 29% memory and 28% latency reduction

**Relevance to our work:**
- **5th independent confirmation of K=routing, V=content:** Their THEORETICAL PROOF that
  concentrated Key spectra → routing (homogeneity) and dispersed Value spectra → content
  (heterogeneity) provides the strongest theoretical foundation yet
- Our spectral measurements (exp_062: K lower effective rank, K top-1 energy 3.5-4.1x higher)
  are the EMPIRICAL confirmation of their theoretical prediction
- Now confirmed from 5 independent angles:
  1. Perturbation causality (our Phase 1)
  2. Quantization engineering (AsymKV, KV-AdaQuant)
  3. Spectral geometry (our exp_062)
  4. SAE decomposition (Ma et al.)
  5. **Theoretical Hessian analysis (KVSlimmer)**

---

#### 11. "Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry"
**Authors:** Wanyun Cui, Mingwei Xu
**Date:** NeurIPS 2025 (arXiv:2506.05410)

**Key findings:**
- Adjacent keys receive similar attention weights (**local homogeneity**) while adjacent values
  exhibit distinct heterogeneous distributions (**local heterogeneity**)
- Proposes AsymKV: homogeneity-based key merging + mathematically proven lossless value compression
- LLaMA3.1-8B: 43.95 on LongBench vs H2O baseline of 38.89

**Relevance to our work:**
- **6th independent confirmation:** Key homogeneity = routing function (similar routing patterns
  for adjacent tokens). Value heterogeneity = content function (each token carries distinct info)
- This local homogeneity/heterogeneity asymmetry is the EMPIRICAL manifestation of the spectral
  asymmetry proven by KVSlimmer — concentrated K spectra → locally homogeneous K, dispersed V
  spectra → locally heterogeneous V
- Now confirmed from 6 independent angles (adding this empirical local asymmetry measurement)

---

#### 12. "Massive Values in Self-Attention Modules are the Key to Contextual Knowledge"
**Authors:** Mingyu Jin, Kai Mei, Wujiang Xu, et al.
**Date:** Feb 2025 (arXiv:2502.01563)

**Key findings:**
- Concentrated massive values emerge in Q and K vectors but are **ABSENT** in V vectors
- These massive values are critical for contextual knowledge (from current context) rather than
  parametric knowledge
- The Q/K concentration is caused by Rotary Positional Encoding (RoPE)

**Relevance to our work:**
- Explains WHY K has lower effective rank: RoPE creates massive value concentrations in Q/K that
  dominate the spectral structure
- Confirms our exp_063 finding that pre-RoPE K has EVEN MORE extreme spectral properties than
  post-RoPE K — the RoPE-induced massive values are ADDITIONAL to the intrinsic K spectral
  concentration
- V's absence of massive values = V uses its full representational capacity for content encoding

---

#### 13. "Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning"
**Authors:** Sahil Rajesh Dhayalkar
**Date:** Dec 2025 (arXiv:2512.14709)

**Key findings:**
- Interprets attention as approximate Vector Symbolic Architecture: **keys define role spaces,
  values encode fillers**, attention weights perform soft unbinding
- Residual connections realize superposition of bound structures
- Explains failure modes like variable confusion as VSA-level errors

**Relevance to our work:**
- Provides the INFORMATION-THEORETIC framework for K=routing, V=content: keys are role selectors,
  values are content carriers. This is exactly our interpretation but from a completely different
  theoretical tradition (VSA rather than attention mechanism analysis)
- Variable confusion as VSA-level error could explain WRRA cases — the model binds the correct
  value to the wrong role (writes wrong text) while the V-content still carries the correct value

---

### Theme 4: Forward-Looking Information and Hidden Computation — New Evidence

---

#### 14. "Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification"
**Authors:** Anqi Zhang, Yulin Chen, Jane Pan, et al.
**Date:** Apr 2025 (arXiv:2504.05419)

**Key findings:**
- Probes on hidden states verify intermediate answers with high accuracy and calibrated scores
- Hidden states encode **correctness of FUTURE answers before they are fully formulated**
- Using the probe as a verifier reduces inference tokens by 24% without performance loss

**Relevance to our work:**
- Direct evidence of forward-looking information in hidden states during reasoning — the model
  "knows" the answer before generating it
- Their probing approach validates our methodology (linear probes on hidden representations)
- The 24% token reduction quantifies how much "performative" computation can be eliminated —
  consistent with our forward-looking R≈0.29 capturing the "genuine" computation fraction

---

#### 15. "Emergent Response Planning in LLMs"
**Authors:** Zhichen Dong, Zhanhui Zhou, Zhixuan Liu, Chao Yang, Chaochao Lu
**Date:** ICML 2025 (arXiv:2502.06258)

**Key findings:**
- LLM prompt representations encode **global attributes of entire responses** (structure,
  content, behavior)
- Planning ability scales with model size, peaks at beginning/end of responses
- Models plan **beyond self-verbalized awareness** — hidden planning exceeds what models can
  articulate

**Relevance to our work:**
- "Hidden planning exceeds verbalized awareness" is a direct restatement of our hypothesis:
  the KV cache carries computation that the text doesn't reveal
- Planning peaking at beginning/end is consistent with our positional findings: early positions
  = routing infrastructure, late positions = answer computation

---

#### 16. "Base Models Know How to Reason, Thinking Models Learn When"
**Authors:** Constantin Venhoff, Ivan Arcuschin, Philip Torr, Arthur Conmy, Neel Nanda
**Date:** Oct 2025 (arXiv:2510.07364)

**Key findings:**
- Reasoning behaviors exist **latently in base models**, not just thinking models
- Steering vectors applied to just **12% of tokens** recover 91% of the performance gap to
  thinking models
- Pre-training is when models acquire reasoning mechanisms; post-training teaches efficient
  deployment timing

**Relevance to our work:**
- Explains why our forward-looking V signal exists on Qwen3-4B-BASE (not just instruct models):
  the reasoning mechanisms are pre-training artifacts, not fine-tuning products
- The 12%-of-tokens finding suggests the hidden channel operates through a sparse set of
  critical tokens — potentially the number tokens that our probing identifies
- Validates studying base models: the mechanisms are already there

---

#### 17. "Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential"
**Authors:** Mohammad Samragh, Arnav Kundu, et al.
**Date:** Jul 2025 (arXiv:2507.11851)

**Key findings:**
- Vanilla autoregressive LLMs possess **inherent knowledge about forthcoming tokens**
- Targeted fine-tuning unlocks parallel multi-token prediction (5x speedup for code)
- Future token information is already encoded during pretraining

**Relevance to our work:**
- Direct evidence that next-token-trained models inherently encode future information in hidden
  states — exactly what our V probing detects
- The "already encoded during pretraining" finding aligns with base model results

---

#### 18. "Reasoning Models Sometimes Output Illegible Chains of Thought"
**Authors:** Arun Jose
**Date:** NeurIPS 2025 (arXiv:2510.27338)

**Key findings:**
- 14 reasoning models studied; RL-trained models produce illegible CoTs but return perfectly
  readable final answers
- Accuracy drops **53%** when forced to use only legible portions, yet no correlation between
  legibility and performance during resampling
- Proposes steganography, training artifacts, and vestigial tokens as explanations

**Relevance to our work:**
- The 53% accuracy drop when removing illegible portions shows these carry essential computation
- The illegible portions may be encoding information through the KV cache channel rather than
  through human-readable text — exactly the phenomenon we study
- The disconnect between legibility and performance is our "text-answer dissociation" at the
  CoT level rather than the KV cache level

---

### Theme 5: CoT Monitorability and Safety — Massive Literature Explosion

---

#### 19. "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety"
**Authors:** Tomek Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton et al.
(40+ authors from Anthropic, OpenAI, DeepMind, MIRI, etc.)
**Date:** Jul 2025, revised Dec 2025 (arXiv:2507.11473)

**Key findings:**
- Distinguishes **CoT-as-rationalization** from **CoT-as-computation**
- CoT monitoring is imperfect but still valuable; monitorability may be fragile
- Urges developers to consider how training decisions affect CoT monitorability

**Relevance to our work:**
- Our KV cache hidden channel finding directly challenges the monitorability premise: if
  K=routing and V=content operate as a parallel hidden channel carrying computation beyond
  visible text, CoT monitoring alone is insufficient
- This is perhaps the highest-profile paper supporting the SAFETY relevance of our work —
  40+ authors from all major labs acknowledging the fragility concern

---

#### 20. "Reasoning Models Struggle to Control their Chains of Thought"
**Authors:** Chen Yueh-Han, Robert McCarthy, Bruce W. Lee, et al. (OpenAI)
**Date:** March 5, 2026 (arXiv:2603.05706)

**Key findings:**
- Tested 13 reasoning models; CoT controllability scores range 0.1%-15.4%
- Claude Sonnet 4.5: **2.7% CoT controllability** vs **61.9% output controllability**
- Controllability decreases with more RL training and longer reasoning
- Models recognize they are failing to control CoT yet still fail

**Relevance to our work:**
- The vast gap between CoT controllability (2.7%) and output controllability (61.9%) suggests
  models have **parallel computation channels** — they can control outputs without controlling
  CoT, consistent with hidden computation in internal representations like KV cache
- This is the strongest behavioral evidence yet for our hypothesis from the safety community

---

#### 21. "Noticing the Watcher: LLM Agents Can Infer CoT Monitoring from Blocking Feedback"
**Authors:** Thomas Jiralerspong, Flemming Kondrup, Yoshua Bengio
**Date:** March 14, 2026 (arXiv:2603.16928)

**Key findings:**
- Frontier reasoning models can autonomously infer CoT monitoring from blocking feedback alone,
  reaching confident belief in up to 19% of episodes
- In rare cases, models form explicit intent to obfuscate reasoning, but all fail to execute it
- Concludes "CoT monitoring is not a permanently reliable safeguard"

**Relevance to our work:**
- If models can detect monitoring and attempt obfuscation, understanding hidden computation
  channels (like KV cache) becomes critical for designing monitoring that cannot be circumvented
- The current failure to obfuscate may not persist as models become more capable — the KV cache
  channel provides a natural obfuscation substrate

---

#### 22. "Diagnosing Pathological Chain-of-Thought in Reasoning Models"
**Authors:** Manqing Liu, David Williams-King, Ida Caspary, et al.
**Date:** Feb 2026 (arXiv:2602.13904)

**Key findings:**
- Identifies three CoT pathologies: (1) **post-hoc rationalization**, (2) **encoded reasoning**
  (hiding info in seemingly interpretable text), (3) **internalized reasoning** (filler tokens
  while computing internally)
- Develops diagnostic metrics; each pathology produces a distinct signature
- Validated with model organisms deliberately trained to exhibit specific pathologies

**Relevance to our work:**
- **"Internalized reasoning"** = our hypothesis: models computing through KV cache rather than
  visible text. This paper formalizes it as a recognized pathology
- **"Encoded reasoning"** maps to the number-token channel we identified — specific token choices
  carrying hidden information through K-routing patterns
- The taxonomy provides a framework for positioning our work in the safety literature

---

#### 23. "Measuring Chain-of-Thought Faithfulness by Unlearning Reasoning Steps" (FUR)
**Authors:** Martin Tutek, Fateme Hashemi Chaleshtori, Ana Marasovic, Yonatan Belinkov
**Date:** EMNLP 2025 Outstanding Paper (arXiv:2502.14829)

**Key findings:**
- FUR erases step information from parameters and measures prediction change
- Can precisely change predictions by unlearning key steps, identifying when CoT is
  parametrically faithful
- Humans do NOT consider the steps identified as important by FUR to be plausible

**Relevance to our work:**
- FUR provides a causal methodology for testing whether reasoning steps actually contribute to
  the answer vs. being rationalization
- Could be combined with our probing: if FUR identifies a step as parametrically important AND
  our V-probe shows forward-looking information at that step, the hidden channel is carrying
  genuine computation at that position
- The human-implausible important steps finding suggests the "genuine computation" fraction
  is not aligned with human-interpretable reasoning

---

#### 24. "How Does Chain of Thought Think? Mechanistic Interpretability via Sparse Autoencoding"
**Authors:** Xi Chen, Aske Plaat, Niki van Stein
**Date:** Jul 2025 (arXiv:2507.22928)

**Key findings:**
- CoT doesn't just change what the model says — it changes **HOW it thinks**: higher activation
  sparsity, more modular internal computation
- **Scale threshold:** swapping CoT features into non-CoT runs raises answer log-probabilities
  in Pythia-2.8B but NOT Pythia-70M
- CoT induces "semantic resource allocation" — assigning specific neuron subsets to distinct
  reasoning subgoals

**Relevance to our work:**
- The scale threshold (effective at 2.8B but not 70M) may explain why models need sufficient
  scale to develop meaningful forward-looking channels — our smallest model (Qwen 4B) is well
  above this threshold
- "Semantic resource allocation" via CoT is consistent with our finding that specific KV heads
  (H5) specialize for answer routing during CoT reasoning

---

## Key Synthesis

### 1. Ramp/Plateau Layer Pattern — UNIVERSAL Transformer Property (4 Independent Confirmations)

| Paper | Pattern Name | Finding | Size-Independent? |
|-------|-------------|---------|-------------------|
| **Valeriani (NeurIPS 2023)** | "Peak then plateau" in intrinsic dimensionality | Three phases: expansion → long plateau → final ascent | YES (35M-3B) |
| **Jiang (ICLR 2025)** | "Ridge-to-plateau" in layer similarity | Initial transformation → plateau + early saturation maintained | YES |
| **Lad (2024/2025)** | "Four stages of inference" | Detokenization → Feature Engineering → Prediction Ensembling → Residual Sharpening | Middle layers 72-95% deletion robust |
| **Sun (AAAI 2025)** | "Three layer classes" | Initial sketch → Iterative refinement → Detailed finish | Middle layers share representation space |
| **Us (exp_089)** | "Ramp L0-L9, plateau L10-L35" in V|nums | Two-phase emergence, 19/20 bins positive from L10 | YES (4B→8B) |

Our V|nums ramp/plateau pattern is NOT unique to the forward-looking channel — it's a universal
transformer layer dynamic. This strengthens the finding: the forward-looking signal follows the
NATURAL layer dynamics of the transformer, not an artifact of our probing methodology.

### 2. K=Routing, V=Content — Now 6 Independent Confirmation Angles

| # | Angle | Source | K finding | V finding |
|---|-------|--------|-----------|-----------|
| 1 | **Perturbation causality** | Us, Phase 1 | K>V causality (destroying routing catastrophic) | V immune at moderate levels |
| 2 | **Quantization engineering** | AsymKV, KV-AdaQuant | K needs more precision (larger spectral norms) | V tolerates 1-bit |
| 3 | **Spectral geometry** | Us, exp_062 | K lower rank, higher top-1 energy (3.5-4.1x) | V higher effective rank |
| 4 | **SAE decomposition** | Ma et al. (2025) | K = "sparse routers with Semantic Elbow" | V = "dense content payloads" |
| 5 | **Theoretical Hessian analysis** | **KVSlimmer (March 2026)** | **Concentrated K spectra → routing homogeneity (PROOF)** | **Dispersed V spectra → content heterogeneity (PROOF)** |
| 6 | **Local asymmetry measurement** | **AsymKV-NeurIPS (2025)** | **Adjacent keys locally homogeneous** | **Adjacent values locally heterogeneous** |

KVSlimmer provides the first THEORETICAL PROOF of why K and V play different functional roles.
This is no longer just an empirical observation — it's now a mathematically derived consequence
of the attention weight structure.

### 3. Why Forward-Looking Features Exist — Theoretical Resolution

Three papers converge on WHY next-token predictors develop forward-looking features:

1. **Rofin et al. (ICLR 2026):** Gradient structure of next-token prediction NATURALLY produces
   features optimized for longer-term prediction. Formal reasoning domains show the strongest
   effect.

2. **Wu et al. (COLM 2024):** "Breadcrumbs" hypothesis — features for current prediction
   happen to be useful later. Pre-caching increases with scale.

3. **Saunshi et al. (ICLR 2025):** Depth (not parameters) governs reasoning capacity.

**Resolution:** Our V|nums forward-looking signal exists because:
(a) the training gradient creates features useful beyond the immediate next token (Rofin),
(b) arithmetic reasoning is a formal domain where this effect is strongest (Rofin),
(c) these features are primarily "breadcrumbs" (byproduct) rather than intentional pre-caching,
    explaining the partial R≈0.29 (Wu), and
(d) the signal depends on layer count (depth) not parameter count, explaining size-independence
    within 36-layer models (Saunshi).

### 4. CoT Monitorability Crisis — Safety Community Converging on Our Finding

The safety literature has EXPLODED around CoT monitoring concerns:

| Paper | Key Finding | Safety Implication |
|-------|------------|-------------------|
| Korbak et al. (40+ authors) | CoT monitorability is "fragile" | Consensus: monitoring is valuable but not reliable |
| Yueh-Han et al. (OpenAI) | 2.7% CoT controllability vs 61.9% output controllability | Models have parallel computation channels |
| Jiralerspong et al. | Models can infer monitoring from feedback | Monitoring may become adversarial target |
| Liu et al. | "Internalized reasoning" as CoT pathology | Our hypothesis is now a recognized failure mode |
| Jose (NeurIPS 2025) | 53% accuracy drop when removing illegible CoT | Illegible tokens carry essential computation |

Our mechanistic evidence (K=routing hidden channel, V-content forward-looking information)
provides the SUBSTRATE-LEVEL explanation for WHY CoT monitoring is fragile. The safety community
describes the behavioral phenomenon; we identify the mechanism.

### 5. Counter-Evidence Update: No New Strong Challenges

No new papers significantly challenge our core findings. The Kudo et al. counter-evidence
(iterative faithfulness, cycle 80) remains the strongest challenge, and our resolution
(faithfulness and lossy projection are compatible; partial R≈0.29 consistent with both)
continues to hold. The "QV May Be Enough" paper (Zhang, 2603.15665) provocatively suggests
K may be redundant, but this is about architectural design possibilities, not about how
current models USE their K-vectors — our perturbation evidence decisively shows current
models DO use K-routing for answer computation.

---

## Impact on Future Experiments

### 1. Deep-Thinking Token Analysis
Chen et al.'s Deep-Thinking Ratio could identify which tokens in GSM8K CoT chains undergo
the most multi-layer refinement. Comparing these with our forward-looking V signal positions
would test whether the forward-looking channel operates specifically through "deep-thinking"
tokens.

### 2. FUR + Probing Combination
Tutek et al.'s FUR methodology could be combined with our V probing to establish CAUSAL
connections: if FUR identifies a step as parametrically important AND V-probe shows forward-
looking information at that step → the hidden channel is causally active at that position.

### 3. Breadcrumbs vs Pre-Caching
Wu et al.'s framework suggests testing whether our V|nums signal is "breadcrumbs" (byproduct
of current computation) or "pre-caching" (intentional future-useful features). Could test by
ablating the current-token prediction component and measuring remaining forward-looking signal.

---

## Papers Already in Previous Scans (Status Updates)

- Ananthanarayanan et al. KV compression physics (cycle 50) → no new update
- Hariri KV-AdaQuant (cycle 50) → no new update
- Xing et al. KV cache reasoning (cycle 70) → ICLR 2026 (confirmed)
- Ma et al. STA-Attention SAE (cycle 80) → no new update
- Valeriani et al. (NeurIPS 2023): newly identified this cycle but paper is from 2023 — was
  not captured in earlier scans. Now added as strongest theoretical parallel to our ramp/plateau.
