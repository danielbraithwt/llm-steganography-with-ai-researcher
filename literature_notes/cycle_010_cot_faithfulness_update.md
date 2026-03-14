# CoT Faithfulness: 2024-2026 Update

**Papers:**
1. "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful" — Arcuschin, Janiak, Krzyzanowski et al. (March 2025, arXiv:2503.08679)
2. "Chain-of-Thought Is Not Explainability" — Barez, Wu (Oxford WhiteBox, 2025)
3. "Dissociation of Faithful and Unfaithful Reasoning in LLMs" (arXiv:2405.15092, 2024)
4. "How Does Chain of Thought Think?" — Chen, Plaat, van Stein (July 2025, arXiv:2507.22928)
5. "Measuring Faithfulness by Unlearning Reasoning Steps" (EMNLP 2025)

**Scanned:** Cycle 10, 2026-03-14

## Key Findings

### CoT Unfaithfulness "In The Wild" (Arcuschin et al. 2025)
Two types of unfaithfulness found on realistic prompts (no artificial bias):
- **Implicit Post-Hoc Rationalization (IPHR):** Models generate explanations rationalizing predetermined answers. Detected via paired comparative questions ("Is X > Y?" and "Is Y > X?") — models sometimes systematically answer "Yes" to both, then generate coherent justifications.
- **Unfaithful Illogical Shortcuts:** Models use subtly illogical reasoning to make speculative answers appear rigorously proven.

**Prevalence rates:**
| Model | IPHR Rate |
|---|---|
| GPT-4o-mini | 13.49% |
| Claude 3.5 Haiku | 7.42% |
| Gemini 1.5 Pro | 6.54% |
| DeepSeek R1 | 0.37% |
| Claude 3.7 Sonnet (thinking) | 0.04% |

Key insight: "CoT is more useful for identifying flawed reasoning and thus discounting unreliable outputs, than it is for certifying the correctness of a model's output."

### Dissociation of Faithful and Unfaithful Reasoning (2024)
- Models arrive at correct answers despite invalid reasoning text
- Two distinct mechanisms: faithful recovery (reasoning genuinely supports conclusion) and unfaithful recovery (bypasses reasoning)
- Models recover more readily from obvious errors; contexts providing stronger evidence for correct answers improve recovery
- These factors have **divergent effects** on faithful vs unfaithful recovery

### Mechanistic CoT Analysis with SAE (Chen et al. 2025)
- Combined sparse autoencoders with activation patching on Pythia (70M, 2.8B) on GSM8K
- CoT prompting leads to "significantly higher activation sparsity and feature interpretability scores" in larger models
- Transferring CoT-reasoning features into non-CoT runs "raises answer log-probabilities significantly in the 2.8B model, but has no reliable effect in 70M"
- **Scale threshold:** Computation organization depends on model scale — smaller models don't benefit
- CoT information is "widely distributed" across features, not concentrated in top-K patches

### Unlearning-Based Faithfulness (EMNLP 2025)
- Faithfulness measured by adverse effect of unlearning reasoning steps on initial prediction
- Uses parameter intervention and unlearning to assess causal relevance
- Unfaithfulness decreases going from non-thinking to thinking models

## Relevance to Our Research

**DIRECTLY RELEVANT — Broad convergence with our hypothesis from multiple angles.**

1. **"CoT is not the computation":** The IPHR finding (models decide first, then rationalize) parallels our mechanistic finding that the KV cache carries answer information independently of the text channel. The model computes the answer in hidden states, then generates plausible text as a "post-hoc rationalization."

2. **Model-specific faithfulness variation:** The wide range of unfaithfulness rates across models (0.04% to 13.49%) parallels our finding that Qwen and Llama differ dramatically in text-dependence (94% vs ~30% compliance in Exp 6). Some models rely more on text; others bypass it.

3. **Scale threshold for CoT computation (Chen et al.):** The finding that CoT features only transfer meaningfully in 2.8B+ models suggests our findings (on 4B-8B models) are above the threshold where hidden computation organizes. Below this threshold, the "hidden channel" may not exist.

4. **Divergent faithful/unfaithful mechanisms:** The finding that faithful and unfaithful recoveries respond differently to context strength is consistent with two separable channels — the text channel (faithful) and the hidden channel (unfaithful bypass).

5. **Thinking models improve faithfulness:** The finding that thinking/reasoning-trained models show less unfaithfulness suggests training can strengthen the text-computation coupling. Our Exp 6 finding that Qwen (base) is more text-compliant than Llama (instruct) may be partly explained by training differences.

## Impact on Evidence

- **Strongly supports:** "CoT unfaithfulness is architecturally guaranteed" framing
- **Extends:** Model-specific faithfulness variation is now well-documented across many models
- **Complicates:** Thinking models reduce unfaithfulness — is the "hidden channel" trainable?
- **New question:** If training can increase faithfulness, does it change the KV cache spatial structure?
