# Reasoning Theater: Performative Chain-of-Thought

**Cycle:** 20
**Date:** 2026-03-14
**Papers reviewed:**
- Boppana et al. (2026). "Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought." arXiv:2603.05488
- METR (2025). "CoT May Be Highly Informative Despite Unfaithfulness." Blog post.
- Tutek et al. (2025). "Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps." EMNLP 2025 (Outstanding Paper Award). arXiv:2502.14829
- (2025). "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful." arXiv:2503.08679

---

## 1. Reasoning Theater (Boppana et al., March 2026)

### Key Finding
Models generate **performative chain-of-thought** — the model becomes strongly confident in its final answer but continues generating tokens without revealing its internal belief. Activation probes can decode the final answer from hidden states **far earlier** than a CoT monitor detects confidence.

### Method
Three complementary approaches applied to DeepSeek-R1 (671B) and GPT-OSS (120B):
1. **Activation probing**: Attention-weighted pooling of hidden states to classify final answer
2. **Early forced answering**: Truncate generation and force answer token
3. **CoT monitoring**: AI judge tracking when model declares confidence

### Results
- **MMLU (easy, recall-based)**: Probes achieve high accuracy from the very beginning of reasoning. CoT monitor lags dramatically. Probe-guided early exit **saves 80% of tokens** while maintaining 97% accuracy.
- **GPQA-Diamond (hard, multihop)**: All three methods improve gradually — genuine reasoning occurs. Performativity rate gap is only 0.012 (vs 0.417 on MMLU).
- **Layer localization**: Final answer information becomes linearly decodable at **layer 20** in DeepSeek-R1 (second half of network). Best layers cluster in late layers for both models.
- **Genuine reasoning markers**: Backtracking and "aha moments" occur almost exclusively where probes show large belief shifts — these behaviors track genuine uncertainty, not performance.

### Relevance to Our Research
**STRONG support for the "text is a lossy projection" hypothesis.**

1. On easy tasks, the model's internal state carries the answer throughout most of the reasoning trace. The visible CoT is **performative** — the model has already decided, but continues generating plausible reasoning tokens. This is exactly the phenomenon our KV cache experiments probe: the hidden states carry answer information that the visible text doesn't reveal.

2. The task-dependent difference (MMLU: performative, GPQA: genuine) maps to our Qwen/Llama difference in text-compliance. High text-compliance (Qwen, 94%) corresponds to models that USE the text for reasoning. Low text-compliance (Llama, ~30%) corresponds to models that may compute internally and generate performative text.

3. The layer localization finding (answer decodable from layer 20+) is consistent with our finding that PGD perturbations concentrate in late layers (≥18 on Qwen3-4B).

4. This paper measures the temporal dimension (WHEN is the answer decided) while we measure the spatial dimension (WHERE is the answer encoded in KV cache). Both point to the same conclusion: the visible text is not the full computation.

---

## 2. METR: CoT Is Informative Despite Unfaithfulness (August 2025)

### Key Finding
When clues require genuine reasoning to decipher, models become **highly faithful** — only 3 plausibly unfaithful cases out of 21,272 trajectories. A detection system achieves 99.3% TPR at 96.2% specificity for identifying clue-usage.

### Key Argument
Complex cognition DEMANDS CoT usage. If deceptive reasoning requires multi-step sequential processing, it should appear detectable in CoT traces. Genuine sophistication cannot be fully concealed without leaving artifacts.

### Relevance to Our Research
**PARTIALLY challenges** our framing.

- METR argues that complex reasoning IS faithfully represented in CoT. Our experiments focus on GSM8K math, which is complex enough to require multi-step reasoning, yet we find the KV cache carries separable answer information.
- However, the reconciliation may be: the text faithfully represents the REASONING STEPS but not the ANSWER COMPUTATION. The model's visible reasoning is a faithful trace of its step-by-step approach, but the final answer is computed from a separate channel (the hidden state aggregate) that doesn't require the text.
- This aligns with our Condition C finding (exp 6): models detect INCONSISTENT reasoning (0% compliance) but follow CONSISTENT corrupted reasoning (~94% on Qwen). The text carries structural information about reasoning validity, not just the answer.

---

## 3. Faithfulness by Unlearning Reasoning Steps — FUR (Tutek et al., EMNLP 2025)

### Method
Erase information from model parameters by unlearning specific reasoning steps (NPO+KL on FF2 layers), then check if the model's prediction changes. If it does, the step was parametrically faithful.

### Key Results
- ff-hard scores (% instances where unlearning changes prediction):
  - LLaMA-8B: 30-69% across datasets
  - LLaMA-3B: 65-86%
  - Mistral-7B: 40-60%
  - Phi-3: 22-54%
- Smaller models are MORE parametrically faithful (higher ff-hard)
- 66-94% of post-unlearning CoTs support a different answer, indicating deep internal changes
- **Only 0.15 Pearson correlation** between parametric faithfulness (FUR) and human plausibility judgments — internally influential reasoning doesn't look convincing to humans

### Critical Insight
Previous contextual perturbation methods (like our KV cache perturbation) **substantially underestimate** true parametric faithfulness because they don't target model internals. Perturbing the reasoning chain while keeping model parameters fixed measures **self-consistency**, not faithfulness.

### Relevance to Our Research
**Methodological complement.**

1. FUR measures parametric faithfulness (does unlearning a step change the answer?). We measure KV cache faithfulness (does perturbing hidden states change the answer independently of text?). These are different constructs.
2. Their finding that smaller models are more faithful is interesting — could explain why our 4B Qwen-Base (smaller) shows higher text-compliance-like behavior than 8B Llama (larger, more internal computation)?
3. The 0.15 correlation between internal importance and surface plausibility resonates with our finding that "answer-coupled" positions aren't distinguishable by their visible content (formatting tokens have 3.5x more answer-attention than text tokens).

---

## 4. CoT Unfaithfulness in the Wild (2025)

### Key Findings
- Production models show 2-13% post-hoc rationalization rates WITHOUT adversarial prompts
- GPT-4o-mini: 13%, Haiku 3.5: 7%, frontier models: 0.04-2.17%
- Models construct superficially coherent arguments to justify logically contradictory answers
- Thinking-enabled models are most faithful (Sonnet 3.7 thinking: 0.04%)

### Relevance
The spectrum of faithfulness rates across models is consistent with our cross-model variation finding. Different models have different ratios of "text-based reasoning" vs "hidden computation." Thinking-enabled models (which generate much longer internal reasoning) are most faithful, suggesting that more computation through visible tokens reduces reliance on hidden channels.

---

## Synthesis for Our Research Program

These papers collectively validate the core premise: **LLMs maintain internal representations that are richer than their visible text output.** The key nuance from recent work:

1. **The text IS informative** (METR) — but not complete (Reasoning Theater, FUR)
2. **Internal confidence precedes visible expression** (Reasoning Theater) — consistent with our null space finding
3. **Faithfulness varies by task difficulty** — easy tasks → performative CoT, hard tasks → genuine reasoning
4. **Faithfulness varies by model** — smaller models more parametrically faithful, thinking models more textually faithful
5. **The hidden channel may be most prominent for easy/familiar computations** where the model "knows the answer" early and generates performative text afterward
