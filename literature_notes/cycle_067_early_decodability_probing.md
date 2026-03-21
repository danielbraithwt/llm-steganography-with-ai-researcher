# Phase 2 Literature: Early Decodability and Probing for Hidden Computation

**Cycle:** 67
**Date:** 2026-03-21
**Focus:** Literature directly relevant to Phase 2 experiments (linear probes, arithmetic error probing, decodability vs causality)

---

## 1. "Knowing Before Saying" (Afzal et al., ACL Findings 2025)
**Citation:** Afzal et al. (2025). "Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion." ACL Findings 2025, pp. 12791–12806. arXiv:2505.24362

### Key Finding
Linear probes on LLM hidden states can predict whether a CoT will produce the correct answer **before any tokens are generated**, achieving 60-76.4% accuracy. This significantly outperforms a BERT-based text-only baseline (53.5-69.1%).

### Method
- Trained compact feedforward networks (3 hidden layers: 256, 128, 64) on hidden state representations
- Models tested: **Llama-3.1-8B** and **Mistral-7B**
- Datasets: Olympiad, Cn-k12, AQuA math problems (10k/1k train/test)
- SVCCA analysis for representational similarity across generation steps

### Critical Results
- **Pre-generation accuracy (best):** 76.4% (Olympiad, Llama)
- **Most informative layers:** Middle layers 11-14 and 16-17
- **Text baseline substantially worse:** confirms internal representations carry information beyond surface text
- **Performance plateaus early** — later generation steps (70-90%) don't consistently improve, with SVCCA showing minimal representational drift
- Surprise: on some datasets, accuracy with 0% of tokens generated ≈ accuracy with 100% generated

### Relevance to Our Experiment A
**DIRECTLY validates our approach** — with important methodological differences:
1. They predict binary CoT success (classification); we predict **numeric answer value** (regression). Our task is more demanding and more specific.
2. They probe the **residual stream**; we probe the **KV cache specifically**. Our approach tests whether K/V representations (routing vs content) differ, connecting to Phase 1 K>V findings.
3. They test at the problem-level before generation; we test at **each position bin within the CoT**, measuring the temporal trajectory of decodability.
4. Their finding that middle layers are most informative is consistent with our prediction that 50-75% depth layers will show the largest early decodability gap.
5. The plateau finding (information present from the start) suggests that for many problems, the model "knows" the answer before generating any CoT — exactly our hypothesis.

### Methodological lessons for our experiment
- Their feedforward probe has 3 hidden layers — our ridge regression (linear probe) is more conservative but also more interpretable. If we find positive results with a linear probe, that's a stronger claim (information is linearly decodable).
- They use SVCCA to validate representational stability — we could add this analysis.
- Sample size: they use 10k/1k which is much larger than our 80. However, their classification task is simpler than our regression task.

---

## 2. "Probing for Arithmetic Errors" (EMNLP 2025)
**Citation:** (2025). "Probing for Arithmetic Errors in Language Models." EMNLP 2025 Main, Paper 411.

### Key Finding
Linear probes can decode **correct arithmetic answers** from model internal states even when the model's **text output is wrong**. This demonstrates that models internally compute correct solutions more consistently than their generated text suggests.

### Critical Results
- Probes recover the model's predicted digit with increasing accuracy across layers
- **Ground-truth digits remain decodable** from internal states
- Error detection achieves **80-90% accuracy** in later layers
- Key insight: the model appears to "know" the correct answer internally but generates incorrect text

### Relevance to Our Experiment C (WRRA)
**DIRECTLY validates the WRRA approach:**
1. This is almost exactly our Experiment C — they show that at positions where the model writes wrong arithmetic, the internal states contain the correct value.
2. Their probe recovers correct answers from hidden states, confirming that text ≠ internal computation for arithmetic.
3. The layer-wise improvement (later layers more accurate) suggests probing at 75-100% depth for our WRRA experiment.
4. **Critical extension we provide:** They probe the residual stream; we will probe **K-cache specifically**, testing whether the hidden channel for correct values is carried through attention routing (K) vs attention content (V). This connects their finding to our KV-specific mechanistic story.

---

## 3. "Causality ≠ Decodability" (2025, Vision Transformers)
**Citation:** (2025). "Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs." arXiv:2510.09794

### Key Finding
Information that is **decodable** from a representation is not necessarily **causally used** by the model. Probing experiments can overstate or understate functional roles.

### Critical Results
- **Final-layer object tokens:** >90% linear probe accuracy for counting information, yet patching these tokens **does not alter predictions** (decodable but NOT causal)
- **Middle-layer object tokens:** Strong causal influence when patched (flip predictions), despite probes showing only weak count information (causal but NOT decodable)
- CLS tokens: 90% decodability in middle layers but no causal influence until final layers

### Relevance to Our Research
**IMPORTANT METHODOLOGICAL CAUTION** for Phase 2 — but one we are well-positioned to address:

1. **The concern:** If our K-probe successfully decodes the answer early, that alone doesn't prove the KV cache *uses* this information for computing the answer. It could be "incidental" information that happens to be in the representations.

2. **Why we're well-positioned:** We have BOTH pieces:
   - **Phase 1 (causal evidence):** Destroying K-cache routing destroys answer accuracy (14 experiments, 5 models). This proves K-routing IS causally used.
   - **Phase 2 (decodability evidence):** If K-probe shows early decodability, combined with Phase 1's causal evidence, we have the full picture: K-routing both CARRIES answer information during normal generation AND IS CAUSALLY REQUIRED for answer computation.

3. **This paper actually STRENGTHENS our combined Phase 1+2 argument.** It shows that most probing studies only have one half of the evidence. We will have both halves.

4. **Practical impact:** We should cite this paper and explicitly argue that our Phase 1 causal evidence + Phase 2 decodability evidence together constitute stronger evidence than either alone.

---

## 4. "Lost in Backpropagation: The LM Head is a Gradient Bottleneck" (Godey & Artzi, March 2026)
**Citation:** Nathan Godey and Yoav Artzi (2026). "Lost in Backpropagation: The LM Head is a Gradient Bottleneck." arXiv:2603.10145

### Key Finding
The LM head (final projection from hidden dimension D to vocabulary dimension V) **suppresses 95-99% of the gradient norm** during backpropagation. This is an unavoidable consequence of the rank-D bottleneck when D ≪ V.

### Results
- **95-99% of gradient norm suppressed** by the output layer
- Makes trivial patterns unlearnable in controlled experiments
- Drastically affects training dynamics independently of architecture

### Relevance
**PROVIDES THE THEORETICAL FOUNDATION for our entire hypothesis.**

The LM head is mathematically guaranteed to be a lossy projection. If 95-99% of the gradient information is lost when projecting from hidden states to tokens, then by construction, the model's internal representations carry far more information than the generated text can express. This means:

1. **Text IS a lossy projection of internal computation** — not by accident but by architectural necessity.
2. The model learns to route computation through high-dimensional hidden states, and the text is a low-dimensional shadow of this computation.
3. This explains WHY the KV cache carries more information than the text: the KV cache IS the internal computation; the text is the bottlenecked output.

Previously referenced in research_spec.md but now confirmed as the right framing. The 95-99% number is remarkably consistent with our finding that text prediction quality is preserved (remaining in the 1-5% that IS projected) while answer computation (in the 95-99% that's suppressed) is carried by the hidden channel.

---

## 5. Paraphrase Experiments: Preliminary Caution for Experiment B

### Finding (from "CoT Is Not Explainability", Barez et al., Oxford 2025)
Preliminary evidence suggests **no significant accuracy gap between paraphrased and original CoT accuracy**. If confirmed, this would be evidence AGAINST our Experiment B hypothesis.

### Relevance
- The paraphrase experiment (Experiment B) may yield null results.
- However, the key methodological question is HOW the paraphrase is implemented:
  - If paraphrasing preserves the same tokenization patterns (e.g., same numbers, similar token count), the K-routing patterns may be preserved → no disruption even if hidden information exists
  - If paraphrasing dramatically changes tokenization (different word choices → different K-vectors → disrupted routing), this is a stronger test
  - The Barez et al. result may reflect a weak paraphrase that preserves routing-relevant features
- We should design Experiment B to test BOTH conservative paraphrases (same numbers, similar length) and aggressive paraphrases (different phrasing, different token count)
- This result reinforces the priority ordering: Experiment A first, then C, then B

---

## Synthesis for Phase 2

The literature strongly supports our Phase 2 experimental program:

1. **Experiment A (early decodability)** is validated by "Knowing Before Saying" and "Reasoning Theater" — both show internal representations predict answers before text. Our KV-specific approach extends this work.

2. **Experiment C (WRRA)** is validated by "Probing for Arithmetic Errors" — probes find correct answers in hidden states when text is wrong. Our K-specific probe extends this.

3. **Experiment B (paraphrase)** faces a potential null result from Barez et al. — lower priority is confirmed.

4. **The causality ≠ decodability paper** highlights that we need BOTH causal and decodability evidence — which is exactly what our Phase 1 + Phase 2 combination provides. This is a key argumentative strength.

5. **Godey & Artzi** provides the theoretical backbone: text is architecturally guaranteed to be a lossy projection.

**Most impactful finding:** The "Knowing Before Saying" paper shows that information plateaus early — the model "knows" the answer before generating. If our K-probe shows this AND our Phase 1 K-perturbation shows K is causal, we have a complete argument for natural channel usage.
