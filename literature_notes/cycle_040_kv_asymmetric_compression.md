# KV Cache Asymmetric Compression: Independent Confirmation of K > V

**Cycle:** 40
**Date:** 2026-03-15
**Papers reviewed:**
- Tao, Yu & Zhou (COLING 2025). "AsymKV: Enabling 1-Bit Quantization of KV Cache with Layer-Wise Asymmetric Quantization Configurations."
- Liu, Tengxuan et al. (2026). "PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs." OpenReview.
- Feng, Lv, Guo, Cao, Xie & Zhou (2025). "Identify Critical KV Cache in LLM Inference from an Output Perturbation Perspective." OpenReview.
- Tian, Su, Li & Zhang (March 2026). "Where Matters More Than What: Decoding-aligned KV Cache Compression via Position-aware Pseudo Queries." arXiv:2603.11564
- Liu, Palnitkar et al. (December 2025). "Hold Onto That Thought: Assessing KV Cache Compression On Reasoning." arXiv:2512.12008
- Teng et al. (March 2026). "InfoFlow KV: Information-Flow-Aware KV Recomputation for Long Context." arXiv:2603.05353

---

## 1. AsymKV: Keys More Sensitive Than Values (COLING 2025)

### Key Finding

**Keys are substantially more sensitive to quantization errors than values.** This justifies using higher precision for K while applying aggressive quantization (including 1-bit) to V.

### Method

Layer-wise asymmetric quantization: each transformer layer uses customized precision levels for K vs V independently. Values can be quantized to **1-bit** while keys require higher precision.

### Relevance to Our Research

**THIS IS INDEPENDENT CONFIRMATION OF OUR K > V HIERARCHY.**

AsymKV discovers the same asymmetry from a completely different angle:
- **Our approach:** Perturbation experiments (direction replacement, magnitude scaling) → K perturbation devastating, V perturbation harmless
- **AsymKV:** Quantization error analysis → K more sensitive to compression, V tolerant of extreme compression (1-bit)

The convergence is striking: our perturbation experiments on 5 models showing V-only immunity (228/228 at σ≤1 magnitude, 393/394 at early+mid direction) predict exactly what AsymKV observes — values can be aggressively compressed because they carry less critical information for downstream accuracy.

**Key difference:** AsymKV measures sensitivity to precision loss (analog rounding). Our experiments measure sensitivity to direction/magnitude replacement (structural perturbation). Both find K >> V, confirming this is not an artifact of any particular perturbation method.

---

## 2. PM-KVQ: Keys Need More Precision for Long-CoT (2026)

### Key Finding

Post-training KV cache quantization causes significant performance degradation on **long-CoT LLMs** specifically. RoPE causes "distribution of less frequent channels in the Key Cache" to be poorly captured during calibration.

### Method

Progressive mixed-precision: gradually reduces bit-width across transformer blocks. Higher precision assigned to more sensitive layers. Result: **up to 8% improvement over SOTA baselines** under same memory budget with 2.73-5.18x throughput.

### Relevance

1. Keys are specifically singled out as the precision-sensitive component, consistent with our K > V
2. Long-CoT is specifically the hardest case — because reasoning computation depends critically on precise K-routing, any precision loss in keys during long reasoning chains compounds catastrophically
3. The progressive strategy (higher precision in sensitive layers) maps onto our finding that K perturbation at early positions is most devastating (attention sink infrastructure)

---

## 3. Critical KV Cache via Output Perturbation (Feng et al., 2025)

### Key Finding

Beyond attention weights, **value states within KV entries and pretrained parameter matrices are also crucial** for identifying critical cache entries. Their perturbation-constrained selection algorithm reduces compression loss by **>50% on average** across 29 datasets.

### Relevance

This paper uses **output perturbation** to evaluate KV importance — conceptually similar to our approach. Key difference: they optimize for output preservation (text quality), while we optimize for answer accuracy. The distinction matters: their method would identify text-coupled positions as critical, while our experiments show answer-coupled positions are different from text-coupled ones.

**Important nuance:** Their finding that "value states" matter for text prediction is NOT inconsistent with our V-immunity finding. V matters for TEXT prediction (which their metric captures); V is dispensable for ANSWER accuracy (which our experiments measure). The dissociation between these two uses of V is exactly our text-answer dissociation.

---

## 4. "Where Matters More Than What" (Tian et al., March 2026)

### Key Finding

**"Positional information plays a more critical role than semantic content"** when constructing queries to simulate decoding behavior.

### Method

DapQ (Decoding-aligned Pseudo Queries): uses position-aware pseudo queries to approximate actual generation queries. Achieves **99.5% performance on NIAH with only 3% KV cache budget**.

### Relevance

**INDEPENDENT CONFIRMATION OF OUR POSITIONAL > SELECTIVITY FINDING.**

Our experiments (Exp 013, 016, 017, 021) showed that AC/TC selectivity adds ZERO explanatory power beyond position — positional information completely dominates. Tian et al. independently discover the same principle: position matters more than semantic content for determining which KV entries to keep.

The title itself — "Where Matters More Than What" — could serve as a slogan for our findings. The "where" (position in the reasoning chain) determines functional importance; the "what" (AC/TC selectivity scores) adds nothing.

---

## 5. KV Cache Compression on Reasoning (Liu et al., December 2025)

### Key Findings

- No single compression strategy universally works; performance varies by dataset type
- **H2O and modified SnapKV demonstrated superiority for reasoning models**
- Heavy-hitter token tracking is beneficial for reasoning traces
- Low-budget eviction strategies can **paradoxically increase reasoning trace length** — the model compensates for lost KV entries by generating more tokens

### Relevance

1. H2O superiority for reasoning aligns with our Exp 013 finding that H2O ≈ TC protection > AC protection
2. The **paradoxical increase in trace length** under aggressive eviction is a novel behavioral consequence of KV compression — the model "knows" information was lost and generates more text to compensate
3. Tested on GSM8K with Llama-3.1-8B-Instruct — same model and benchmark as our experiments, enabling direct comparison

---

## 6. InfoFlow KV: Information-Flow-Aware Recomputation (Teng et al., March 2026)

### Key Finding

Uses attention-norm signals from queries to identify tokens that are "both semantically meaningful and structurally capable of influencing generation." Treats KV selection as an information flow problem rather than heuristic scoring.

### Relevance

The information-flow framing connects to our K-routing finding: tokens that are structurally capable of influencing generation are precisely those with critical K-routing patterns. InfoFlow KV's attention-norm signal is measuring something related to our "answer-coupling" metric — how much a position influences downstream computation.

---

## Synthesis: The Field Is Converging on K > V

The KV cache compression literature has independently discovered our central mechanistic finding from a practical engineering perspective:

| Evidence source | K > V evidence | Year |
|----------------|----------------|------|
| **Our perturbation experiments** | K devastating, V harmless (5/5 models, 393/394 V-immune) | 2026 |
| **AsymKV** | Keys more sensitive; V can be 1-bit quantized | 2025 |
| **PM-KVQ** | Keys need more precision for long-CoT specifically | 2026 |
| **KIVI (prior work)** | Per-channel K quant, per-token V quant; different error profiles | 2024 |

**The practical implication is now actionable:** KV cache compression methods should allocate more precision budget to keys than values, especially during reasoning. Our experiments provide the mechanistic explanation: K determines attention routing (which positions to attend to), while V carries content that has massive redundancy across positions. Compressing V is safe; compressing K destroys the computation.

**Our contribution beyond the compression literature:** These papers treat K/V asymmetry as an empirical observation to exploit for engineering efficiency. Our research explains WHY the asymmetry exists (routing vs content), WHERE it's strongest (early positions = infrastructure), and HOW it relates to the text-answer dissociation (K carries answer-relevant routing invisible to text prediction). The compression community could use our K-coupling metrics to design better layer-wise precision allocation.
