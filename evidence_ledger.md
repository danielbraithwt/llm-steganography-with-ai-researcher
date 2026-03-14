# Evidence Ledger

## Current Summary
Last updated: 2026-03-14 (cycle 20, lit scan)
Cycles completed: 20

### Hypothesis Status: REFRAMED — The null space exists and functional separability is real, but it's GEOMETRIC (distributed across representation dimensions) not SPATIAL (concentrated at specific token positions)

**LITERATURE REFRAMING (Cycle 20):** Independent literature from 2025-2026 provides strong convergent evidence:
- **"Reasoning Theater" (Boppana et al., 2026)**: Models decide answers internally 80% of tokens before visible CoT reveals it. Probes detect answers from layer 20+. Direct evidence for "text is lossy projection."
- **Direction-Magnitude dissociation (2602.11169)**: Same hidden state encodes SEPARABLE functions in different geometric components (direction → language modeling via attention; magnitude → syntax via LayerNorm). Independent validation of functional separability.
- **Phase transitions in KV compression (2603.01426)**: Sharp "safety cliff" at ~90% compression — independent discovery of our SNR cliff phenomenon. Digital-like encoding is replicated.
- **Attention sinks (ICLR 2025)**: Initial tokens serve as structural attention infrastructure. EXPLAINS why early positions are critical on all models — it's architectural infrastructure, not answer information.
- **CoT features are distributed (2507.22928)**: Useful CoT information is widely distributed, not concentrated. Consistent with PGD rho=0.20.

**Revised interpretation after Exp 018 + Lit scan:**
- **Null space EXISTS** (Exp 4) — NOT challenged
- **Spatial structure is weak** (rho=0.20) — but this is EXPECTED if the separation is geometric/distributed rather than spatial/concentrated
- **Position dominance is explained by attention sinks** — early positions are infrastructure, not content. This is a confound in our destruction experiments, not evidence against functional separability
- **The field is converging** on our core hypothesis (text is lossy projection of computation) through independent methods (probing, unlearning, causal mediation, steering)

**Models:**
- **Qwen-Base**: PGD null space exists. Weak spatial structure (bivariate rho=0.20, partial rho=0.16). Destruction dissociation (+23.7pp) is primarily positional.
- **Llama-Instruct**: distributed/analog — position dominates, no PGD null space
- **Qwen-Instruct**: position dominates, destruction dissociation (-38.5pp) is a positional confound

### Evidence Overview
| Claim | Status | Strength | Key Experiments | Notes |
|-------|--------|----------|-----------------|-------|
| Unused output capacity | established | strong | Exp 1 (research_spec) | 4-5 bits/token unused |
| CoT narrows distribution | established | strong | Exp 2 (research_spec) | 3x entropy reduction, median near zero |
| KV cache fragility (SNR cliff) | **Qwen-specific, NOT general** | moderate negative | Exp 3, **Exp 007** | Qwen: cliff at 14 dB; Llama: 100% at 5 dB, no cliff |
| Adversarial null space exists | **Qwen-specific, NOT general** | **moderate negative** | Exp 4, **Exp 008** | Qwen: 377x, 100% success. **Llama: 0.8x, 0% success — produces garbage not valid answers** |
| Null space has spatial structure | **SIGNIFICANTLY WEAKENED** | **weak** | Exp 5, Exp 002, Exp 004, **Exp 018** | **Exp 5's rho=0.78 does NOT replicate under prompt-only-only analysis (rho=0.20). Partial rho\|position = 0.157 pooled / 0.043 per-attack median. Original inflated by mixing attack types. Weak but nonzero residual** |
| Cross-model text-dependence variation | supported | moderate | Exp 6 (research_spec) | Qwen 94% compliant, Llama ~30% |
| Functional separability via zeroing | **disconfirmed** | moderate negative | Exp 002, Exp 004, **Exp 005** | Zeroing even 99% of positions → 86-100% accuracy on both models |
| **Accuracy dissociation via noise** | **established** | **strong** | **Exp 004, Exp 005** | **Replicated: Qwen +23.5pp, Llama +23.8pp at 5% noise** |
| **Noise >> zeroing for ablation** | **established** | **strong** | **Exp 004, Exp 005** | **Confirmed on both Qwen and Llama** |
| **AC positions are general hubs** | **established** | **moderate** | **Exp 004, Exp 005** | **AC ablation hurts text MORE than TC on both models** |
| **Dissociation is architecture-general** | **supported** | **strong** | **Exp 005** | **Effect size nearly identical across Qwen and Llama** |
| **SNR robustness is architecture-SPECIFIC** | **established** | **strong** | **Exp 003, Exp 007** | **Qwen: digital cliff at 14 dB. Llama: robust to 5 dB, no cliff** |
| **Encoding strategy differs: digital (Qwen) vs distributed (Llama)** | **supported** | **moderate** | **Exp 005, Exp 007, Exp 009** | **Position-level and noise-level differences confirmed. Layer-level: both robust, but Qwen slightly more concentrated (std ratio 7.64)** |
| **PGD null space is Qwen-specific** | **established** | **strong** | **Exp 4, Exp 008** | **Qwen: 100% success, 377x norm, valid answers. Llama: 0% success, 0.8x norm, garbage output** |
| **Partial null space exists on Llama** | **supported** | **weak** | **Exp 008** | **PGD CAN change answer-region predictions (0% match) while preserving text (94%), but produces incoherent output not valid answers** |
| **Layer-level redundancy is universal** | **established** | **moderate** | **Exp 009** | **Both models tolerate single-layer KV destruction: Llama 93-100%, Qwen 100% (35/36 layers). Residual stream provides massive compensation** |
| **Layer sensitivity is more concentrated in Qwen** | **supported** | **weak** | **Exp 009** | **std ratio = 7.64 (predicted ≥1.5). But Qwen n=4 — very low confidence. Layer 0 critical (25%), all others 100%** |
| **Encoding distinction is position-level, not layer-level** | **supported** | **moderate** | **Exp 004, 005, 009** | **Position ablation shows clear effects (24pp dissociation). Layer ablation shows near-zero effects on both models. Residual stream provides layer-level redundancy** |
| **Causal bypass validates hidden channel (literature)** | **supported** | **strong (independent)** | **Lit scan cycle 10** | **CMI metric shows bypass regimes where CoT text has zero causal influence on answer. Tested on 11 models including Qwen3-4B. Independent convergent evidence (Sathyanarayanan et al., 2026)** |
| **KV cache carries manipulable semantic information (literature)** | **supported** | **strong (independent)** | **Lit scan cycle 10** | **KV cache steering induces reasoning styles via one-shot KV modification (Belitsky et al., 2025). Proves separable channels from constructive direction (vs our adversarial direction)** |
| **GQA vs MHA plausibly explains Qwen/Llama encoding difference (literature)** | **INVALIDATED** | **disconfirmed** | **Lit scan cycle 10, Exp 011** | **Both Qwen3-4B and Llama-3.1-8B use GQA with 8 KV heads. Architecture difference is NOT MHA vs GQA. Original hypothesis was based on incorrect assumption that Qwen uses MHA** |
| **Text bottleneck framing is mainstream (literature)** | **supported** | **moderate** | **Lit scan cycle 10** | **Coconut, Quiet-STaR, latent reasoning survey all frame text as bottleneck. Coconut shows reasoning IMPROVES when text constraint removed. Our work provides mechanistic KV-level evidence** |
| **H2O heavy-hitters ≠ answer-coupled positions** | **established** | **moderate** | **Exp 011** | **H2O vs AC rho=0.004 (Qwen), 0.11 (Llama). H2O retains 57% TC-selective vs 40% AC-selective. Positions with highest cumulative attention are NOT the answer-relevant ones. Replicated on both models** |
| **KV compression may harm hidden channel** | **supported** | **moderate** | **Exp 011** | **H2O preferentially evicts answer-coupled positions (Q1 has highest AC). Standard compression preserves text coherence but may degrade answer computation. Suggests AC-aware compression could improve accuracy** |
| **AC-protection FAILS on Llama** | **established** | **strong negative** | **Exp 012** | **Protecting high-AC positions is no better than random. H2O_protect ≈ TC_protect >> AC_protect ≈ Random at 1-5% noise. Raw AC score is observational, not causal for answer accuracy on Llama** |
| **AC-aware compression would NOT outperform H2O** | **established** | **strong negative** | **Exp 012** | **Directly contradicts exp_011 suggestion. H2O noises high-AC positions yet achieves BETTER accuracy. Cumulative attention is a better protection metric than answer-token attention** |
| **Raw AC ≠ selectivity (AC-TC)** | **important distinction** | **methodological** | **Exp 004/005 vs Exp 012** | **Selectivity-based destruction works (+24pp dissociation). Raw AC-based protection fails. These are different metrics testing different aspects of position importance** |
| **Selectivity-based protection ALSO fails on Llama** | **established** | **strong negative** | **Exp 013** | **SEL_protect ≈ AC_protect ≈ Random << H2O ≈ TC. Selectivity does NOT rescue the framework. SEL at 1% = 48% vs H2O at 1% = 100%** |
| **Position in causal chain dominates on Llama** | **established** | **strong** | **Exp 013** | **Position-score rho: H2O=-0.56, AC=+0.37, TC=+0.44. Strategies that noise early positions fail; strategies that noise late positions succeed. Position is the dominant factor for answer accuracy on Llama** |
| **TC (text-coupling) is best protection metric** | **established** | **moderate** | **Exp 013** | **TC_protect > H2O_protect at all noise fractions (72% vs 68% at 3%, 36% vs 20% at 5%). Irony: text-coupling best preserves ANSWER accuracy on Llama — consistent with distributed encoding** |
| **AC/SEL spatial structure is Qwen-specific** | **REVISED: much weaker than claimed** | **weak** | **Exp 008, 012, 013, Exp 018** | **PGD rho=0.20 on Qwen (NOT 0.78 — that was inflated by mixed attack types). PGD fails on Llama. AC/SEL protection fails on Llama. "Answer-coupled positions" concept has weak causal validity even on Qwen** |
| ~~Instruction tuning creates ultra-fragile KV encoding~~ | **DISCONFIRMED (pipeline bug)** | **invalidated** | **Exp 014 (buggy), Exp 015 (corrected)** | **Exp_014 had a pipeline bug: generated from AFTER the answer (model produced new questions → 0% accuracy). Exp_015 with corrected pipeline shows 70% baseline maintained at all scales ≤0.3x. NOT ultra-fragile** |
| **Encoding is TRAINING-dependent, not just architecture-dependent** | **established** | **strong** | **Exp 014, Exp 015** | **Qwen-Base=concentrated (positive dissociation +24pp). Qwen-Instruct=REVERSED dissociation (-47pp). Same architecture, different training → different encoding. Instruction tuning reverses which positions are critical** |
| **Three distinct encoding regimes exist (RE-REVISED)** | **partially revised** | **moderate** | **Exp 004, 007, 013, 015, 016** | **Concentrated (Qwen-Base: PGD-validated). Distributed (Llama: position dominates). Qwen-Instruct: early-position-critical (not "reversed" — positional confound). The "three regimes" claim for destruction dissociation reduces to: position matters on all models, PGD works only on Qwen-Base** |
| ~~Dissociation is REVERSED on Qwen-Instruct~~ | **REVISED: predominantly positional confound** | **strong (confound)** | **Exp 015, Exp 016** | **Unconstrained gap replicates (-38.5pp), but POS_EARLY=19.2% vs POS_LATE=65.4% (+46pp). SelTC targets pos 0.242 (early), SelAC targets pos 0.991 (late). Within-early-half gap collapses to +3.8pp. NOT a genuine instruction-tuning channel reversal** |
| **TC sign reversal: Qwen-Instruct vs Llama** | **established** | **moderate** | **Exp 014 vs Exp 013** | **Position vs TC: Qwen-Instruct rho=-0.44 (early=high TC), Llama rho=+0.44 (late=high TC). Fundamentally different attention patterns** |
| **H2O/TC protection is PERFECT on Qwen-Instruct** | **established but positionally explained** | **moderate** | **Exp 015, Exp 016** | **H2O (pos 0.987) and TC (pos 0.946) noise late positions → 65.4% (baseline). AC (pos 0.351) noises early positions → 23.1%. Protection success is driven by WHICH POSITIONS get noised, not by metric-specific channel preservation** |
| **Selectivity-based destruction is confounded with position on ALL models** | **CONFIRMED on ALL models including Qwen-Base** | **strong** | **Exp 013, Exp 016, Exp 017** | **Qwen-Base: SEL-pos rho=-0.201 (NEGATIVE — opposite from instruct). POS_EARLY=34.2% vs POS_LATE=100% (+65.8pp). Within-early-half gap=2.6pp (collapsed). Llama: same pattern. Qwen-Instruct: same pattern. Position dominates across all tested models** |
| **Qwen-Base destruction dissociation is PARTIALLY positional** | **established** | **strong** | **Exp 017** | **+23.7pp unconstrained gap replicates exp_004 (+23.5pp). But within-early-half gap=2.6pp (collapsed). Within-late-half gap=42.1pp (persists, with residual positional confound). SEL-pos rho=-0.201 on Qwen-Base (negative, opposite from instruct +0.57-0.65). Partial confound: position explains ≥70% of variance, selectivity adds genuine residual value in late half** |
| **Late-half selectivity effect may be genuine on Qwen-Base** | **supported** | **weak** | **Exp 017, Exp 018** | **Within late half: SelAC at pos 0.742 → 42.1%, SelTC at pos 0.827 → 84.2% (+42.1pp gap). Cross-strategy at similar positions show 34pp difference. BUT PGD rho (exp_018) = 0.20 (not 0.78), weakening the mechanistic support. Effect may be partially positional even within late half** |
| **PGD perturbation-attention rho=0.78 DOES NOT REPLICATE** | **DISCONFIRMED** | **strong negative** | **Exp 018** | **Prompt-only PGD on Qwen-Base: bivariate rho=0.197 (vs Exp 5's 0.78). Partial rho\|position = 0.157 pooled / 0.043 per-attack median. Within-quartile rhos: 0.10-0.24 (significant but weak). Original Exp 5 likely inflated by mixing prompt-only + reasoning-only + binary-search attacks. Reasoning-only attacks create perturbation-attention correlation by construction (PGD avoids text-coupled positions).** |
| **PGD attack success does not replicate at 100%** | **established** | **moderate** | **Exp 018** | **Exp 018: 0% genuine attack success (26.7% including methodological artifacts) vs Exp 4's 100%. 13/45 attacks show zero perturbation yet different answers due to teacher-forcing vs autoregressive path divergence. The original 100% success was on 6 attacks from 5 problems — may have been a favorable sample** |
| **Position universally dominates answer accuracy** | **established** | **strong** | **Exp 013, 016, 017** | **ALL models: POS_EARLY << POS_LATE at 5% noise. Qwen-Base: +65.8pp gap. Qwen-Instruct: +46.2pp. Llama: comparable pattern. Early positions form the computational foundation; destroying them is universally devastating regardless of selectivity** |
| **SEL-position correlation reverses: Base negative, Instruct positive** | **established** | **moderate** | **Exp 017 vs 016/013** | **Qwen-Base: SEL-pos rho=-0.201 (early=high selectivity). Qwen-Instruct: +0.650. Llama-Instruct: -0.162 (weak). Base models attend to early positions more strongly from the answer token (relative to TC), while instruct models attend more to late positions** |
| **Sharp noise cliff at 0.3x-1.0x on Qwen-Instruct** | **established** | **strong** | **Exp 015** | **Zero degradation at scales ≤0.3x (all strategies=70%). Dramatic effects only at 1.0x. Cliff is between 0.3x and 1.0x additive noise (≈0-10 dB per position)** |
| **Exp_014 pipeline bug invalidates "ultra-fragile" finding** | **methodological** | **critical** | **Exp 015** | **Exp_014 teacher-forced FULL trace including "#### answer", then generated from beyond — model started new questions. Fixed in exp_015: truncate at "####", lookback re-computation. The "ultra-fragile" regime does not exist** |
| **Performative CoT validated (literature)** | **supported** | **strong (independent)** | **Lit scan cycle 20** | **"Reasoning Theater" (2603.05488): Models reach internal answer confidence 80% of tokens before visible CoT reveals it on easy tasks. Probes decode answer from layer 20+. On hard tasks (GPQA), genuine reasoning occurs. Task-dependent performativity directly supports "text is lossy projection" (Boppana et al., March 2026)** |
| **Functional separability is GEOMETRIC not spatial (literature)** | **reframes our findings** | **strong (independent)** | **Lit scan cycle 20** | **Direction-magnitude double dissociation (2602.11169): angular perturbation damages LM 42.9x more, magnitude damages syntax 20.4% more. Different pathways (attention vs LayerNorm). Same hidden state, separable geometric components. Our PGD null space may be a distributed geometric phenomenon, not a localized spatial one** |
| **Phase transition in KV compression (literature)** | **supports SNR cliff** | **strong (independent)** | **Lit scan cycle 20** | **"Physics of KV Compression" (2603.01426): Safety cliff at ~90% compression. Two failure modes: representational erasure and representational rigidity. Relational information fragments before entity information. Independent replication of our digital-like encoding finding** |
| **Attention sinks explain position dominance (literature)** | **reframes position findings** | **strong (independent)** | **Lit scan cycle 20** | **ICLR 2025: Initial tokens receive attention as structural infrastructure (no-op channels for softmax normalization). Removal causes severe degradation. This is POSITIONAL INFRASTRUCTURE, not content. Explains why early positions are critical on ALL models (exps 013, 016, 017) and why layer 0 is critical on Qwen (exp 009)** |
| **Removing redundant KV tokens IMPROVES reasoning (literature)** | **supports TC-noise-preserves-accuracy** | **moderate (independent)** | **Lit scan cycle 20** | **R-KV (NeurIPS 2025): 105% of full-cache accuracy at 16-33% retention. Consistent with our finding that TC-selective noise preserves answer accuracy — "text scaffolding" tokens may actively interfere with answer computation** |
| **CoT faithfulness varies by task difficulty (literature)** | **contextualizes our findings** | **moderate (independent)** | **Lit scan cycle 20** | **METR (2025): Complex tasks show high faithfulness (3/21,272 unfaithful). FUR (EMNLP 2025 Outstanding): 30-86% parametric faithfulness varies by model/dataset. Our hidden channel may be most prominent for easy/familiar computations** |
| **Distributed CoT features (literature)** | **supports weak spatial structure** | **moderate (independent)** | **Lit scan cycle 20** | **"How Does CoT Think?" (2507.22928): Useful CoT information is widely distributed across the network, not concentrated. Scale-dependent — benefits emerge at larger models. Consistent with our PGD rho=0.20 (distributed) rather than 0.78 (concentrated)** |
| **KV cache steering confirms manipulable channels (literature)** | **supported** | **strong (independent)** | **Lit scan cycle 20** | **Updated: One-shot KV intervention outperforms continuous activation steering. Induces controllable reasoning STYLES (stepwise, causal, analogical). KV cache encodes procedural instructions, not just content — possible third channel beyond text/answer (Belitsky et al., 2025, arXiv:2507.08799)** |

### Open Questions
1. ~~Why does 50% position pruning not affect accuracy?~~ **ANSWERED (Exp 004): zeroing is the wrong method.**
2. ~~Does the SNR cliff replicate on Llama-3.1-8B?~~ **ANSWERED (Exp 007): NO. Llama shows 100% accuracy at all SNR ≥5 dB. The cliff is Qwen-specific.**
3. ~~At what pruning fraction does accuracy break, and does it break differently for AC vs TC positions?~~ **ANSWERED: Under noise, AC breaks faster than TC on both models.**
4. ~~Would noise injection at classified positions reveal the dissociation?~~ **ANSWERED: Yes.**
5. **Are AC-selective positions genuinely "answer-specific" or just "generally important hubs"?** **ANSWERED (Exp 004+005): Hubs. AC ablation hurts text MORE on both models. But TC-selective noise preserves answer accuracy, so the separation is real — it's just that AC positions are important for everything, not selectively for answers.**
6. Would a more surgical noise intervention (partial noise, not full replacement) reveal finer-grained separation?
7. ~~Does the accuracy dissociation replicate on Llama-3.1-8B?~~ **ANSWERED (Exp 005): YES, +23.8pp at 5% noise (vs Qwen's +23.5pp)**
8. ~~**NEW:** Why is Llama so much MORE fragile than Qwen under random noise?~~ **PARTIALLY ANSWERED (Exp 007): Llama is fragile to POSITION-SELECTIVE destruction (exp_005) but ROBUST to UNIFORM noise (exp_007). Information is distributed across all positions (analog), so destroying any subset is devastating, but uniform noise averages out. Remaining question: does GQA (8 KV heads shared across 32 query heads) provide inherent noise robustness?**
9. ~~Replicate the adversarial null space experiment (Exp 4) on Llama — does the PGD-discovered null space also exist?~~ **ANSWERED (Exp 008): NO in the strong sense. PGD can change answer-region predictions while preserving text (partial null space), but cannot redirect to a valid alternative answer. 0% success rate vs Qwen's 100%. Perturbation norm 0.8x vs Qwen's 377x.**
10. **NEW:** Does the ~24pp dissociation effect hold on a third model family (e.g., Qwen3-8B)?
11. ~~**NEW:** Does GQA (vs MHA) explain Llama's noise robustness?~~ **INVALIDATED (Exp 011): Both models use GQA (8 KV heads). The difference is NOT GQA vs MHA.**
12. **NEW:** What happens between SNR 0 and 5 dB on Llama? Need finer sampling (1, 2, 3, 4 dB) to characterize the transition.
13. **NEW:** Is the digital vs distributed encoding difference related to model size (4B vs 8B) or architecture (Qwen vs Llama)?
14. **NEW:** Would a TARGETED PGD attack (maximize probability of specific wrong answer) succeed on Llama where untargeted divergence fails?
15. **NEW:** Would reasoning-only or full-sequence PGD attacks succeed on Llama? Prompt-only attacks may be too constrained for distributed encoding.
16. ~~**NEW:** Is the null space failure on Llama due to instruction tuning (format robustness) or architecture? Test PGD on Qwen-Instruct to separate these factors.~~ **PARTIALLY ANSWERED (Exp 014, revised Exp 015): Instruction tuning on Qwen REVERSES the dissociation (TC becomes critical, AC dispensable). Training is a major factor. PGD on Qwen-Instruct would likely fail because AC-selective positions are dispensable. The mechanism is different from Llama (reversed channel vs distributed encoding).**
17. ~~**NEW (Exp 009):** Why does Qwen show only 20% baseline accuracy with eager attention? Is this a model loading issue or a generation configuration problem? Need to verify Qwen layer sensitivity with higher n.~~ **ANSWERED (Exp 014): Qwen3-4B-Instruct (same architecture, eager attention) achieves 90% accuracy. The low Base accuracy is a Base model limitation, not an eager attention issue.**
18. **NEW (Exp 009):** Why is layer 0 specifically critical for Qwen? Is this about initial representation quality or attention pattern bootstrapping?
19. **NEW (Exp 009):** Would multi-layer ablation (2-3 layers simultaneously) break Llama's layer redundancy? The residual stream may have limits.
20. ~~**NEW (Lit scan, cycle 10):** Do H2O "heavy-hitter" positions correlate with our answer-coupled positions?~~ **ANSWERED (Exp 011): NO. H2O vs AC rho=0.004 (Qwen), 0.11 (Llama). H2O preferentially retains TC-selective positions (57%) over AC-selective (40%). Standard compression does NOT preserve the hidden channel.**
21. **NEW (Lit scan, cycle 10):** Can we apply the CMI (CoT Mediation Index) from causal bypass research to quantify text-coupling vs answer-coupling at specific positions?
22. **NEW (Lit scan, cycle 10):** Does KV cache steering (Belitsky et al. 2025) change the AC/TC spatial structure? If steering vectors modify the hidden channel, this would demonstrate bidirectional manipulability.
23. **NEW (Lit scan, cycle 10):** Is token importance temporally dynamic during CoT (Lethe finding)? Our static AC/TC classification may miss important temporal effects. Track per-step attention evolution.
24. ~~**NEW (Lit scan, cycle 10):** Does GQA's KV head sharing ratio quantitatively predict the null space dimensionality reduction?~~ **INVALIDATED (Exp 011): Both models use identical GQA ratio (8 KV heads, 32 query heads). The Qwen/Llama difference is NOT explained by GQA/MHA.**
25. **NEW (Exp 011):** Would AC-aware KV compression (retain positions with highest answer-token attention) outperform H2O (cumulative attention) for answer accuracy? Exp 011 shows H2O evicts answer-relevant positions.
26. ~~**NEW (Exp 011):** What explains the Qwen/Llama encoding difference if NOT GQA vs MHA? Both use 8 KV heads. Candidates: model size (4B vs 8B), training data/procedure, depth (36 vs 32 layers), hidden dimension, or instruction tuning.~~ **PARTIALLY ANSWERED (Exp 014): Instruction tuning is a MAJOR factor. Qwen-Base vs Qwen-Instruct (same architecture) shows completely different encoding. But Qwen-Instruct ≠ Llama-Instruct, so architecture also matters. Remaining candidates: model size, depth, training data.**
27. **NEW (Exp 011):** Is the H2O-AC dissociation driven by position (early vs late in sequence) or by genuine functional differentiation? Need position-controlled analysis.
28. ~~**NEW (Exp 011):** Why does Qwen3-4B-Base with eager attention produce only 16-20% baseline accuracy? This limits all Qwen experiments to n=4.~~ **ANSWERED (Exp 014): Qwen3-4B-Instruct with same eager attention achieves 90%. The low accuracy is a Base model limitation — the base model simply isn't good at 8-shot GSM8K math.**
29. ~~**NEW (Exp 011):** Would AC-aware KV compression outperform H2O for answer accuracy?~~ **ANSWERED (Exp 012): NO. AC-protection is no better than random on Llama. H2O outperforms AC.**
30. ~~**NEW (Exp 012):** Why does raw AC score fail as a protection metric when selectivity-based destruction works?~~ **ANSWERED (Exp 013): Both raw AC and selectivity fail as protection metrics. The success of destruction tests (exp_004/005) reflects hub destruction effects, not channel-specific importance. Position in causal chain is the dominant factor.**
31. ~~**NEW (Exp 012):** Would selectivity-based protection outperform H2O?~~ **ANSWERED (Exp 013): NO. SEL ≈ AC ≈ Random << TC ≈ H2O. Selectivity does not rescue the framework.**
32. ~~**NEW (Exp 012):** Is the AC-protection failure a positional confound?~~ **ANSWERED (Exp 013): YES. Position-AC rho=+0.37 (late=high AC). AC-protect noises early positions (mean 0.38). Position-H2O rho=-0.56. H2O protects early positions. Position in causal chain dominates accuracy.**
33. ~~**NEW (Exp 012):** Does AC-protection work on Qwen (where PGD validates AC)? Exp 012/013 only tested Llama (Qwen n=2).~~ **ANSWERED (Exp 015): AC-protection FAILS on Qwen-Instruct (30% at 3%/1.0x vs 70% baseline). H2O/TC maintain perfect baseline. AC is dispensable on Qwen-Instruct (reversed dissociation). Still untested on Qwen-Base (n=4 too small).**
34. **NEW (Exp 013):** Does position-controlled analysis (within position quartiles) show any value from AC or selectivity beyond positional information? Exp 013 found position dominates, but didn't control for it.
35. **NEW (Exp 013):** Why does TC (text-coupling) outperform H2O for answer accuracy? Both protect positions with high subsequent-token attention, but TC is slightly better. Is TC capturing semantic importance beyond positional priority?
36. **NEW (Exp 013):** Can TC-aware compression outperform H2O on actual KV cache eviction benchmarks (not just noise injection)?
37. ~~**NEW (Exp 014):** Would LOWER noise scales (0.1x or 0.01x norm) reveal strategy-dependent effects on Qwen3-4B-Instruct?~~ **ANSWERED (Exp 015): YES — but at scales ≤0.3x, there is NO effect at all (all strategies = baseline 70%). Strategy-dependent effects only emerge at 1.0x scale, and the dissociation is REVERSED (-47pp). The "ultra-fragility" was a pipeline bug, not a real finding.**
38. **NEW (Exp 014):** Does Llama-3.1-8B-Base show the same encoding as Llama-Instruct? This would complete the 2x2 comparison (Qwen/Llama × Base/Instruct) and determine if instruction tuning universally increases fragility.
39. **NEW (Exp 014):** Why does Position vs TC have OPPOSITE sign on Qwen-Instruct (-0.44) vs Llama-Instruct (+0.44)? This suggests fundamentally different attention patterns — does Qwen frontload computation while Llama distributes it?
40. ~~**NEW (Exp 014):** Does the SNR cliff shift upward on Qwen3-4B-Instruct?~~ **ANSWERED (Exp 015): The cliff is between 0.3x and 1.0x additive noise per position (≈0-10 dB). Zero effect at ≤0.3x, dramatic effects at 1.0x. This is an additive noise cliff, not directly comparable to exp_003's uniform replacement noise cliff at 14 dB. Finer sweep (0.4x-0.8x) would locate it precisely.**
41. ~~**NEW (Exp 014):** Is the ultra-fragile encoding specific to the Qwen3 instruction tuning procedure, or universal?~~ **REVISED (Exp 015): The "ultra-fragile" finding was a pipeline bug. Qwen-Instruct is NOT ultra-fragile — it shows a reversed dissociation with a sharp cliff. The question should now be: does the REVERSED dissociation pattern replicate on other instruct models?**
42. **NEW (Exp 015):** Where exactly is the noise cliff on Qwen-Instruct? Between 0.3x and 1.0x additive noise. A finer sweep (0.4x, 0.5x, 0.6x, 0.8x) would locate it precisely.
43. ~~**NEW (Exp 015):** Why is the dissociation REVERSED on Qwen-Instruct?~~ **ANSWERED (Exp 016): It's a POSITIONAL CONFOUND. SelTC targets early positions (0.242), SelAC targets late positions (0.991). POS_EARLY (19.2%) vs POS_LATE (65.4%) accounts for the gap. Within-early-half gap collapses to +3.8pp. No genuine channel reversal.**
44. **NEW (Exp 015):** Does Llama-Instruct also show reversed dissociation under additive noise at appropriate scales? Exp_004/005 used replacement noise. Additive noise might reveal different patterns.
45. **NEW (Exp 015):** Does the reversed dissociation replicate on other instruct-tuned models (Qwen2.5-Instruct, Mistral-Instruct)? Would determine if reversal is a universal instruction-tuning effect or Qwen-specific.
46. **NEW (Exp 015):** Is the 70% pipeline baseline a ceiling artifact? 9/30 problems consistently fail via KV cache generation even without noise. Would improving the pipeline (longer lookback, better truncation) change the results?
47. ~~**NEW (Exp 016):** Does PGD on Qwen-Base also show the positional confound?~~ **ANSWERED (Exp 018): YES, mostly. Bivariate rho=0.197 (not 0.78). Partial rho|position = 0.157 pooled / 0.043 per-attack median. Within-quartile rhos 0.10-0.24. The correlation is mostly positional, with weak genuine residual. The original rho=0.78 was inflated by including reasoning-only attacks in the pooled analysis.**
48. **NEW (Exp 016):** Why are early positions critical on BOTH Llama-Instruct and Qwen-Instruct? Is this a universal property of autoregressive computation (early tokens build the computational foundation) or specific to instruction-tuned models?
49. ~~**NEW (Exp 016):** Is the destruction dissociation on Qwen-Base ALSO a positional confound?~~ **ANSWERED (Exp 017): PARTIAL CONFOUND. Position dominates (+65.8pp gap). Within-early-half gap collapses to 2.6pp. BUT within-late-half gap persists at 42.1pp, and cross-strategy comparisons at similar positions show 34pp differences. Scenario C (partial confound) best describes Qwen-Base. Position explains ≥70% of variance; selectivity adds genuine residual value in the late half.**
50. **NEW (Exp 016):** Can we design a position-controlled selectivity test? E.g., within narrow position bands (quartiles), compare SelAC vs SelTC destruction. If the gap persists within strict position bins, the selectivity effect is genuine. **Exp 017 provides partial evidence:** within-early-half gap=2.6pp (negligible), within-late-half gap=42.1pp (large but residually confounded). Need QUARTILE-level analysis for definitive answer.
51. **NEW (Exp 017):** Why does SEL-position correlation REVERSE between Base and Instruct? Qwen-Base: rho=-0.201 (early=high selectivity). Qwen-Instruct: rho=+0.650 (late=high selectivity). Does instruction tuning shift where the answer token attends, creating a different selectivity landscape?
52. ~~**NEW (Exp 017):** Does the PGD perturbation-attention correlation (rho=0.78) survive partial correlation controlling for position?~~ **ANSWERED (Exp 018): MOSTLY NOT. Bivariate rho dropped from 0.78 to 0.197. Partial rho = 0.157 pooled / 0.043 per-attack median. Weak but nonzero residual survives position control (within-quartile rhos 0.10-0.24, all p<0.01). The spatial structure claim is dramatically weakened but not completely eliminated.**
53. **NEW (Exp 017):** Why are late-half AC-selective positions more critical than late-half non-selective positions? The 42.1pp within-late-half gap suggests these positions carry genuinely different information. Is this because they are specifically attended to by answer-computing heads, or because they happen to be at a critical position range (0.7-0.8)?
54. **NEW (Exp 017):** Would position-QUARTILE-controlled selectivity (within each of Q1-Q4 by position) reveal genuine selectivity effects? The half-based control is too coarse — positions within a half still vary substantially. Quartile control would provide tighter position matching.
55. **NEW (Exp 018):** Why did PGD attacks fail at a much higher rate in Exp 018 (0% genuine success) vs Exp 4 (100%)? Is this problem selection, sample size (n=6 vs n=45), or code differences? Would replicating on Exp 4's exact problems reproduce the 100% success rate?
56. **NEW (Exp 018):** Does reasoning-only PGD show higher perturbation-attention correlation than prompt-only PGD? This would confirm that the Exp 5 inflation was due to mixing attack types. And if reasoning-only rho is high, it means PGD does discover answer-relevant positions when it can directly target them, but this doesn't prove the positions are pre-existing — it's what the optimizer creates.
57. **NEW (Exp 018):** What IS the bivariate rho for prompt-only-only attacks in the original Exp 5 data? If we can separate by attack type in the original analysis, we can confirm whether the mixed-type inflation hypothesis is correct.
58. **NEW (Exp 018):** Given that spatial structure is much weaker than claimed, what is the ACTUAL mechanism of the PGD null space? If it's not concentrated at answer-coupled positions, how does the optimizer change the answer while preserving text? Possible: the null space is high-dimensional and operates through distributed small changes across many positions, not through concentrated perturbation at specific positions. **REFRAMED (Lit scan cycle 20):** The direction-magnitude dissociation (2602.11169) suggests the null space may operate through GEOMETRIC separation (e.g., direction vs magnitude components of KV vectors) rather than spatial concentration.
59. ~~**NEW (Exp 016):** Why are early positions critical on BOTH Llama-Instruct and Qwen-Instruct?~~ **ANSWERED (Lit scan cycle 20): Attention sinks (ICLR 2025). Early tokens serve as structural attention infrastructure (no-op channels for softmax normalization). Their criticality is an architectural property of transformers, not a learned content property. Removing sinks disrupts routing infrastructure for all subsequent computation.**
60. **NEW (Lit scan cycle 20):** Would direction-only vs magnitude-only KV cache perturbation reveal cleaner text/answer dissociation? The 2602.11169 double dissociation suggests isotropic noise conflates two separable geometric effects.
61. **NEW (Lit scan cycle 20):** Do the tokens R-KV identifies as "redundant" (whose removal IMPROVES reasoning to 105%) overlap with our TC-selective positions? If so, TC positions genuinely interfere with answer computation.
62. **NEW (Lit scan cycle 20):** Does the Reasoning Theater finding (early internal confidence) correspond to our null space? Are the positions where the model "already knows the answer" the same positions that carry answer-relevant KV cache information?
63. **NEW (Lit scan cycle 20):** Can CRV-style attribution graph analysis (ICLR 2026 Oral) characterize the null space mechanistically? What transcoder features are involved in the answer-computation pathway?
64. **NEW (Lit scan cycle 20):** Does controlling for attention sinks (excluding first 2-4 tokens) change the position-controlled selectivity analysis? Current "early=critical" finding may be entirely driven by sinks.
65. **NEW (Lit scan cycle 20):** Is there a third "procedural" channel beyond text/answer? KV cache steering (2507.08799) shows the cache encodes reasoning STYLE (stepwise, causal, analogical). Hub positions may be procedural nodes, not text or answer carriers.

### Confirmed Findings
- LLM output distributions have ~4-5 bits/token unused capacity (Exp 1)
- CoT narrows per-token entropy 3x; median entropy near zero during reasoning (Exp 2)
- **Zeroing KV positions is fundamentally inadequate for position importance studies** — confirmed on BOTH Qwen and Llama (Exp 004, 005)
- **Answer-coupled positions carry disproportionate answer information** — noise at 5% of AC-selective positions drops accuracy ~24pp more than TC-selective positions, replicated across architectures (Exp 004, 005)
- **The ~24pp accuracy dissociation is architecture-general** — nearly identical effect on Qwen3-4B-Base (+23.5pp) and Llama-3.1-8B-Instruct (+23.8pp) (Exp 005)
- **AC-selective positions are hubs, not channel-specific** — they carry more information for both answer and text (Exp 004, 005)
- **Random noise is more destructive than targeted noise** — at 5%, random (5% acc) < SelAC (19%) < SelTC (43%) on Llama, confirming channel separation (Exp 005)
- **SNR cliff is Qwen-specific, NOT architecture-general** — Llama shows 100% accuracy at all SNR ≥5 dB, no cliff in the 5-25 dB range where Qwen collapses (Exp 007)
- **Models use different encoding strategies:** Qwen: digital/concentrated (fragile to uniform noise, SNR cliff at 14 dB). Llama: distributed/analog (robust to uniform noise, fragile to position ablation) (Exp 005+007)
- **Adversarial null space (PGD) is Qwen-specific in the strong sense** — Llama shows 0% attack success vs Qwen's 100%. PGD can change distributions but not redirect to valid alternative answers on Llama (Exp 008)
- **Single-layer KV destruction is tolerated by both architectures** — Llama: 93-100% accuracy for all 32 layers. Qwen: 100% for 35/36 layers (except L0=25%, but n=4). Residual stream provides massive layer-level redundancy (Exp 009)
- **The digital/distributed encoding distinction operates at position-level and noise-level, NOT at layer-level** — both models show high layer redundancy, though Qwen is slightly more layer-concentrated (std ratio=7.64) (Exp 009)
- **H2O heavy-hitter positions do NOT correspond to answer-coupled positions** — H2O vs AC Spearman rho = 0.004 (Qwen), 0.11 (Llama). H2O retains 57% TC-selective vs 40% AC-selective positions. Standard KV compression preferentially evicts answer-relevant positions (Exp 011)
- **Both Qwen3-4B-Base and Llama-3.1-8B use GQA with 8 KV heads** — the GQA-vs-MHA hypothesis is invalidated. Architecture differences must stem from other factors (Exp 011)
- **Selectivity (AC-TC rank) fails as a protection metric on Llama** — SEL ≈ AC ≈ Random. TC > H2O >> Random ≈ AC ≈ SEL. The selectivity framework does not identify causally important positions on Llama (Exp 013)
- **Position in the causal chain dominates answer accuracy on Llama** — Position vs H2O rho=-0.56, Position vs AC rho=+0.37. Strategies noising early positions fail; strategies noising late positions succeed. Position is the primary factor (Exp 013)
- **TC (text-coupling) is the best protection metric on Llama** — TC_protect > H2O_protect at 3% (72% vs 68%) and 5% (36% vs 20%). Text-coupling best preserves answer accuracy because answer computation flows through the full causal chain (distributed encoding) (Exp 013)
- ~~Instruction tuning creates ultra-fragile KV encoding on Qwen~~ — **DISCONFIRMED by Exp 015: pipeline bug in Exp 014.** Qwen-Instruct is NOT ultra-fragile; it shows a reversed dissociation with a sharp cliff between 0.3x-1.0x
- **Encoding is training-dependent, not just architecture-dependent** — Qwen-Base (concentrated, +24pp dissociation), Qwen-Instruct (reversed, -47pp dissociation), Llama-Instruct (distributed, +24pp, position-dominated). Same architecture + different training = different encoding (Exp 014, 015)
- **Qwen3-4B-Base low accuracy is a Base model limitation, not eager attention** — Instruct version achieves 90% with same eager attention setup. The Base model simply underperforms at 8-shot GSM8K (Exp 014)
- ~~Qwen-Instruct shows REVERSED dissociation (-47pp)~~ — **REVISED (Exp 016): The -38.5pp gap is a POSITIONAL CONFOUND. SelTC targets early positions (0.242), SelAC targets late positions (0.991). POS_EARLY=19.2% vs POS_LATE=65.4% (+46pp). Within-early-half gap = +3.8pp (negligible). Not a genuine channel reversal.** (Exp 015, 016)
- **H2O/TC protection works on Qwen-Instruct BECAUSE of position** — H2O (pos 0.987) and TC (pos 0.946) noise late/dispensable positions → maintain baseline. AC (pos 0.351) noises early/critical positions → fails. Protection success is positionally determined, not channel-specific (Exp 015, 016)
- **Sharp additive noise cliff between 0.3x and 1.0x on Qwen-Instruct** — Zero degradation at all scales ≤0.3x. Dramatic strategy-dependent effects only at 1.0x. The cliff is sharper than predicted (Exp 015)
- **Selectivity-based destruction is confounded with position on ALL tested models** — Llama-Instruct (exp_013), Qwen-Instruct (exp_016), AND Qwen-Base (exp_017) all show: within-early-half gaps collapse. Position is the dominant factor across all architectures and training regimes. Only PGD on Qwen-Base is position-independent (Exp 013, 016, 017)
- **Early positions are critical on ALL tested models** — Qwen-Base: POS_EARLY=34.2% vs POS_LATE=100% (+65.8pp) at 5%. Qwen-Instruct: POS_EARLY=19.2% vs POS_LATE=65.4% (+46.2pp) at 3%. Llama: similar pattern. Position dominance is UNIVERSAL, not training-specific (Exp 013, 016, 017)
- **Qwen-Base +23.7pp dissociation is a PARTIAL positional confound** — Replicates exp_004 perfectly (+23.7pp vs +23.5pp). But within-early-half gap=2.6pp (collapsed). Within-late-half gap=42.1pp (persists). SEL-pos rho=-0.201 (negative, opposite from instruct). Position explains ≥70% of variance; genuine residual selectivity exists in late half (Exp 017)
- **SEL-position correlation REVERSES between Base (-0.201) and Instruct (+0.650)** — On Qwen-Base, high-selectivity positions are EARLY; on Qwen-Instruct, they are LATE. Despite opposite directions, both produce the same within-half-collapse pattern, confirming position as the universal dominant factor (Exp 017)

### Disconfirmed or Revised
- **Position-level functional separation via zeroing** (Exp 002): Zeroing is too weak. Methodological limitation, not evidence against spatial structure.
- **Full double dissociation** (Exp 004, 005): Text loss dissociation is reversed on BOTH models. AC positions are hubs important for everything. The "hidden channel" is not cleanly separable at the position level — but noise-based ablation still reveals differential answer importance.
- **Llama's text-resistance = stronger hidden channel** (Exp 005): Llama shows the SAME dissociation effect size as Qwen (~24pp), not larger. Its text-resistance (Exp 6) comes from different computation, not from different spatial structure of the hidden channel.
- **SNR cliff is a general property of transformer KV caches** (Exp 007): The sharp cliff at ~14 dB is Qwen-specific. Llama shows no cliff — 100% accuracy at 5 dB (noise at 56% of signal). The "digital-like fragility" interpretation applies only to Qwen's architecture.
- **GQA vs MHA explains Qwen/Llama encoding differences** (Exp 011): INVALIDATED. Both Qwen3-4B-Base and Llama-3.1-8B use GQA with 8 KV heads. The architecture difference must stem from other factors (model size, training, depth, instruction tuning).
- **H2O heavy-hitters = AC positions ("AC are hubs → H2O keeps them")** (Exp 011): DISCONFIRMED. H2O vs AC rho ≈ 0. H2O measures "popularity" (cumulative attention), which is different from "answer-relevance" (answer-token attention). These are orthogonal importance dimensions.
- **AC-aware compression would outperform H2O** (Exp 012): DISCONFIRMED. AC-protection performs no better than random on Llama. H2O-protection outperforms AC-protection by 24-48pp in the informative noise range (1-5%). Raw answer-token attention does not identify causally important positions for answer accuracy.
- **Selectivity (AC-TC rank) as a protection metric** (Exp 013): DISCONFIRMED. Selectivity performs no better than raw AC and both ≈ random. The selectivity framework does NOT rescue the AC/TC protection approach on Llama. TC > H2O >> Random ≈ AC ≈ SEL.
- **"Answer-coupled positions" are causally distinct on Llama** (Exp 012, 013): DISCONFIRMED. The ~24pp destruction dissociation (exp_004/005) reflects hub destruction effects and positional confounds, not genuine channel separation. Position in the causal chain is the dominant factor. ~~The spatial structure claim is valid only on Qwen (PGD rho=0.78).~~ **FURTHER REVISED (Exp 018): Even on Qwen, PGD rho is only 0.20 (not 0.78). Spatial structure is weak everywhere.**
- **Concentrated encoding survives instruction tuning** (Exp 014, 015): DISCONFIRMED. Qwen3-4B-Instruct shows REVERSED dissociation (-47pp), not the concentrated +24pp pattern of Qwen3-4B-Base. Instruction tuning flips which positions are critical (TC becomes critical, AC becomes dispensable). Note: Exp 014's "ultra-fragile" finding was itself a pipeline bug — the real pattern is reversed, not fragile.
- **Encoding difference is purely architectural (Qwen vs Llama)** (Exp 014, 015): REVISED. Same Qwen architecture shows different encoding under Base vs Instruct training. The Base/Instruct axis is at least as important as the Qwen/Llama axis. Need Llama-Base to complete the 2x2 comparison.
- **Exp 014's "ultra-fragile encoding" finding** (Exp 015): DISCONFIRMED — PIPELINE BUG. Exp 014 teacher-forced the full trace including "#### answer", then generated from beyond the answer. The model started new questions instead of regenerating answers → 0% accuracy everywhere. Fixed pipeline in exp 015 shows Qwen-Instruct is not ultra-fragile at all.
- **"Reversed dissociation" on Qwen-Instruct is a genuine instruction-tuning effect** (Exp 016): DISCONFIRMED — POSITIONAL CONFOUND. SelTC targets early positions (0.242), SelAC targets late positions (0.991). POS_EARLY=19.2% vs POS_LATE=65.4% accounts for the gap. Within-early-half gap collapses to +3.8pp (vs unconstrained -38.5pp). The "three encoding regimes" framework needs revision: only PGD on Qwen-Base provides position-independent evidence.
- **Selectivity-based destruction provides genuine spatial structure evidence** (Exp 013, 016, 017): DISCONFIRMED as a standalone metric. Selectivity correlates with position on ALL models (rho=-0.201 on Qwen-Base, +0.650 on Qwen-Instruct, -0.162 on Llama). Within-early-half controls show gap collapse on ALL models. **Exp 017 CONFIRMS: the original +23.5pp dissociation on Qwen-Base is ALSO primarily positional.** However, within-late-half gap persists on Qwen-Base (42.1pp), suggesting partial genuine selectivity effect. The destruction dissociation framework is confounded but not completely uninformative — it overestimates effect sizes due to positional confound.
- **"~24pp dissociation replicated across architectures" is primarily positional** (Exp 017): REVISED. The near-identical +23.5pp (Qwen-Base) and +23.8pp (Llama, exp_005) dissociation effects are driven by the SAME positional factor (SelAC targets earlier positions, SelTC targets later positions) despite the selectivity-position correlation having OPPOSITE signs (-0.201 vs ~0). This means the "architecture-general" replication was actually an architecture-general positional sensitivity, not architecture-general functional separability.
- **PGD perturbation-attention correlation rho=0.78** (Exp 018): DOES NOT REPLICATE. Prompt-only PGD on 45 problems shows bivariate rho=0.197. Partial rho controlling for position = 0.157 pooled / 0.043 per-attack median. The original Exp 5 analyzed 141 attacks mixing prompt-only, reasoning-only, and binary search. Reasoning-only attacks create perturbation-attention correlation by construction (PGD avoids text-coupled positions). Prompt-only cascade effects are mostly positional. The "last remaining position-independent evidence" for spatial structure is dramatically weakened.
- **PGD attack success rate 100%** (Exp 018): DOES NOT REPLICATE on larger sample. Exp 4 reported 100% on 6 prompt-only attacks from 5 problems. Exp 18 shows 0% genuine success on 45 problems (13 apparent successes are teacher-forcing vs autoregressive divergence artifacts). This may be sample size effect — Exp 4's small, filtered sample was likely favorable.

---

## Experiment Log Summary

### Exp 001 (Cycle 1) — FAILED
**Double Dissociation via KV Position Pruning**
- Used wrong model (instruct instead of base) — no valid results
- See `experiment_log/exp_001.md`

### Exp 002 (Cycle 2) — NEGATIVE RESULT
**Double Dissociation via KV Position Pruning (retry)**
- Model: Qwen3-4B-Base, n=82 valid problems, 50% pruning fraction
- **All conditions achieved 100% accuracy** — no dissociation
- Interpretation: zeroing is too weak (confirmed by Exp 004, 005)
- See `experiment_log/exp_002.md`, `results/exp_002/`

### Exp 003 (Cycle 3) — INCOMPLETE (agent crash)
**Pruning Fraction Sweep**
- Script written but never executed
- Superseded by Exp 004
- See `experiment_log/exp_003.md`

### Exp 004 (Cycle 4) — PARTIAL CONFIRMATION
**Noise Ablation Sweep with Selectivity Classification (Qwen)**
- Model: Qwen3-4B-Base, n=34 valid problems
- **Accuracy dissociation:** +23.5 pp at 5% noise (SelAC=44.1%, SelTC=67.6%)
- **Text loss dissociation reversed** (AC ablation hurts text MORE)
- Evidential strength: moderate
- See `experiment_log/exp_004.md`, `results/exp_004/`

### Exp 005 (Cycle 5) — CROSS-MODEL REPLICATION CONFIRMED
**Noise Ablation Sweep on Llama-3.1-8B (replication of exp_004)**
- Model: meta-llama/Llama-3.1-8B-Instruct, n=21 valid problems
- **Accuracy dissociation replicates:** +23.8 pp at 5% noise (SelAC=19.0%, SelTC=42.9%)
- **Text loss dissociation reversed** — same pattern as Qwen
- **Zeroing robustness confirmed** — 86% accuracy at 99% zeroing
- **Llama more fragile under random noise** — 5% random → 5% acc (vs Qwen's 56%)
- Evidential strength: strong (cross-model replication with quantitatively identical effect)
- See `experiment_log/exp_005.md`, `results/exp_005/`

### Exp 006 (Cycle 6) — FAILED
**SNR Cliff Replication on Llama-3.1-8B (first attempt)**
- `DynamicCache` object is not subscriptable — all problems errored at all SNR levels
- Root cause: transformers 5.3.0 changed DynamicCache API (no `key_cache`/`value_cache` attributes)
- See `experiment_log/exp_006.md`

### Exp 007 (Cycles 7-8) — NEGATIVE REPLICATION (important finding)
**SNR Cliff Replication on Llama-3.1-8B**
- Model: meta-llama/Llama-3.1-8B-Instruct, n=21 valid problems, 15 SNR levels
- **SNR cliff does NOT replicate:** Llama shows 100% accuracy at all SNR ≥5 dB
- **Collapse only at extreme noise:** 0% accuracy at SNR 0 dB (noise = signal)
- **Qwen cliff at 14 dB is architecture-specific**, not a general transformer property
- **Digital (Qwen) vs distributed (Llama) encoding:** reconciles with exp_005 position sensitivity
- Evidential strength: strong (clear negative replication; qualitative difference, not marginal)
- See `experiment_log/exp_007.md`, `results/exp_007/`

### Exp 008 (Cycle 8) — NEGATIVE REPLICATION
**PGD Adversarial Null Space on Llama-3.1-8B**
- Model: meta-llama/Llama-3.1-8B-Instruct, n=17 valid problems, 50 PGD steps
- **PGD null space does NOT replicate in strong sense:** 0% attack success (vs Qwen's 100%)
- **Partial null space exists:** PGD changes answer-region predictions (0% match) while preserving text (94%), but produces garbage, not valid answers
- **Perturbation norm 0.8x** (vs Qwen's 377x)
- Evidential strength: strong (negative replication)
- See `experiment_log/exp_008.md`, `results/exp_008/`

### Exp 009 (Cycle 9) — PARTIAL CONFIRMATION
**Per-Layer Noise Sensitivity Profiling**
- Models: Qwen3-4B-Base (n=4), Llama-3.1-8B-Instruct (n=15)
- **Both models highly layer-redundant:** Llama 93-100% for all layers, Qwen 100% for 35/36
- **Cross-model contrast matches predictions:** std ratio = 7.64 (predicted ≥1.5)
- **Layer 0 critical for Qwen** (25%) — unexpected; predicted critical layers in late layers
- Evidential strength: weak-to-moderate (Qwen n=4 limits conclusions)
- See `experiment_log/exp_009.md`, `results/exp_009/`

### Literature Scan (Cycle 10) — CONVERGENT EVIDENCE FROM INDEPENDENT WORK
**Topics:** CoT faithfulness, causal bypass, KV cache adversarial perturbation, latent reasoning, KV cache compression, GQA vs MHA, residual stream redundancy

**Key papers and connections:**

1. **"Causal Bypass in LLMs" (Sathyanarayanan et al., Feb 2026):** Introduces CoT Mediation Index (CMI) — finds "bypass regimes" where models generate fluent CoT but answer computation flows through latent pathways. CMI ≈ 0 on TruthfulQA, 5/20 GSM8K instances show pure bypass. Tests 11 models including Qwen3-4B. Uses activation patching (complementary to our KV perturbation). **Independent validation of our hidden channel hypothesis.**

2. **"Can Transformer Memory Be Corrupted?" (MTI V.1, Oct 2025):** KV cache perturbation framework (additive noise, zeroing, orthogonal rotations) tested on GPT-2 and LLaMA-2. 15-30% performance reduction. Provides Lipschitz-based theoretical analysis of perturbation propagation. **Independent validation of our perturbation approach; their theory could formalize our SNR cliff.**

3. **"KV Cache Steering" (Belitsky et al., Jul 2025):** Modifies KV cache once after prefilling to induce reasoning styles. K'=K+c^k*S^k. Consistently improves reasoning (GSM8K +0.5-4pp, MATH +7.4pp). **Proves KV cache carries separable, manipulable semantic information — our "null space" from the opposite direction (constructive steering vs adversarial perturbation).**

4. **"Coconut/CCOT" (Meta, Dec 2024) + "Survey on Latent Reasoning" (Jul 2025):** Latent reasoning removes text bottleneck; reasoning improves when freed from token vocabulary. Breadth-first search enabled by continuous thought. **Validates our "text as lossy projection" framing — the text bottleneck actively constrains computation.**

5. **"Bottlenecked Transformers" (May 2025):** Uses Information Bottleneck theory + periodic KV cache rewriting for reasoning (+6.6pp). **Treats KV cache as active computation workspace, not passive memory. Provides IB theory framework for our text/answer channel separation.**

6. **"CoT In The Wild Is Not Always Faithful" (Arcuschin et al., Mar 2025):** IPHR rates: 0.04% (Claude 3.7 thinking) to 13.49% (GPT-4o-mini). Thinking models reduce unfaithfulness. **Supports model-specific faithfulness variation consistent with our Qwen/Llama text-compliance differences.**

7. **"Hold Onto That Thought" (Dec 2025) + "Lethe" (Nov 2024):** H2O/SnapKV retain "heavy-hitter" positions for reasoning. Pyramidal sparsity assumption FAILS for reasoning models. Token importance is temporally dynamic. **Heavy-hitter positions may overlap with our answer-coupled positions. Temporal dynamics suggest our static classification misses effects.**

8. **GQA vs MHA (various):** GQA shares KV heads across multiple query heads (Llama: 8 KV → 32 query). **Plausible mechanistic explanation for Qwen/Llama encoding difference:** GQA's sharing creates inherent redundancy (noise robustness) but reduces null space dimensionality (PGD failure).

9. **"Hyper-Connections" (ICLR 2025):** Single residual stream is a bottleneck. Multiple parallel streams improve performance. **Explains our Exp 009 finding:** layer redundancy comes from residual stream compensation, which is why the digital/distributed distinction operates at position-level, not layer-level.

**Literature impact on hypothesis status:**
- Core claim (hidden channel exists) is STRENGTHENED — independent convergent evidence from 5+ research groups
- Architecture-specific findings (SNR cliff, PGD null space) are now EXPLAINED by GQA vs MHA, not simply "Qwen-specific anomalies"
- The "text bottleneck" framing is mainstream in latent reasoning literature
- Our unique contribution remains the MECHANISTIC evidence (spatial structure, adversarial perturbation) at the KV cache level

See `literature_notes/cycle_010_*.md` for detailed paper summaries

### Exp 011 (Cycle 11) — SURPRISING NEGATIVE RESULT
**H2O Heavy-Hitter vs Answer-Coupled Position Overlap**
- Models: Qwen3-4B-Base (n=4), Llama-3.1-8B-Instruct (n=17)
- **H2O heavy-hitters do NOT overlap with AC positions:** rho=0.004 (Qwen), 0.11 (Llama)
- **H2O preferentially retains TC-selective positions** (57% TC vs 40% AC at 50% retention)
- **Quartile analysis:** positions with lowest H2O (would be evicted) have HIGHEST AC scores
- **GQA discovery:** Qwen3-4B-Base also uses GQA (8 KV heads) — invalidates GQA-vs-MHA hypothesis
- **All pre-registered predictions disconfirmed** — H2O importance is orthogonal to AC/TC
- Evidential strength: moderate (surprising negative result with practical implications)
- See `experiment_log/exp_011.md`, `results/exp_011/`

### Exp 012 (Cycle 12) — MAJOR NEGATIVE RESULT (analyzed in cycle 13)
**AC-Aware vs H2O KV Cache Position Protection**
- Model: Llama-3.1-8B-Instruct (n=21; Qwen excluded, n=2)
- **AC-protection FAILS:** protecting high-AC positions is no better than random
- **H2O ≈ TC >> AC ≈ Random** at all noise fractions 1-5%
- At 5% noise: H2O_protect=29%, TC_protect=24%, AC_protect=0%, Random=0%
- **H2O noises positions with 367x higher AC scores yet achieves better accuracy**
- **Raw AC score is observational, not causal, for answer accuracy on Llama**
- Does NOT invalidate selectivity-based destruction findings (exp_004/005)
- Key distinction: raw AC score vs selectivity (AC-TC rank difference) are different metrics
- Evidential strength: moderate (important negative result; Llama-only, selectivity untested)
- See `experiment_log/exp_012.md`, `results/exp_012/`

### Exp 013 (Cycle 13) — DECISIVE NEGATIVE RESULT
**Selectivity-Based Protection + Positional Analysis**
- Model: Llama-3.1-8B-Instruct (n=25)
- **Selectivity does NOT rescue the AC/TC framework:** SEL ≈ AC ≈ Random << TC ≈ H2O
- **Ranking: TC > H2O >> Random > AC ≈ SEL** at all noise fractions 1-5%
- At 3%: TC=72%, H2O=68%, Random=36%, AC=24%, SEL=28%
- **Positional analysis explains everything:** Position vs H2O rho=-0.56 (early=high). Position vs AC rho=+0.37 (late=high). H2O/TC noise late positions. AC/SEL noise early positions. Early positions are most critical on Llama.
- **TC (text-coupling) best preserves answer accuracy** — irony: the text-focused metric outperforms the answer-focused metric for answers. Consistent with distributed encoding.
- **The "answer-coupled positions" concept is NOT causally valid on Llama.** It works on Qwen (PGD rho=0.78) but fails both as protection and as a compression metric on Llama.
- Evidential strength: strong (decisive negative result with clean mechanistic explanation)
- See `experiment_log/exp_013.md`, `results/exp_013/`

### Exp 014 (Cycle 15) — **INVALIDATED BY PIPELINE BUG**
**Qwen3-4B (Instruct) Encoding Strategy — Base vs Instruct Control**
- Model: Qwen/Qwen3-4B (instruct), n=27 valid problems (90% baseline)
- **⚠️ PIPELINE BUG:** Teacher-forced FULL trace including "#### answer", then generated from beyond the answer. Model produced new questions ("Q: ...") instead of regenerating the answer → 0% accuracy under all conditions
- Original "ultra-fragile encoding" finding was entirely an artifact of this bug
- **Still valid findings:** Position vs TC sign reversal (rho=-0.44 vs Llama +0.44), baseline accuracy (90%), eager attention works on Instruct
- Evidential strength: **invalidated** for main claims (ultra-fragility, no dissociation, no protection)
- See `experiment_log/exp_014.md`, `results/exp_014/`

### Exp 015 (Cycle 16) — REVERSED DISSOCIATION (corrects Exp 014)
**Qwen3-4B-Instruct Noise Scale Sweep (pipeline-corrected)**
- Model: Qwen/Qwen3-4B (instruct), n=30 problems (70% pipeline baseline, 86% direct generation)
- **REVERSED dissociation:** SelTC destruction crashes accuracy (-47pp at 3%/1.0x), SelAC has NO effect. Opposite sign from Qwen-Base (+24pp) and Llama (+24pp)
- **Perfect H2O/TC protection:** Both maintain 70% baseline at ALL noise fractions (1-5%) at 1.0x
- **AC protection FAILS:** 30% at 3%/1.0x (vs 70% baseline). Ranking: H2O = TC >> Random > AC
- **Sharp cliff at 0.3x-1.0x:** Zero degradation at scales ≤0.3x. Dramatic effects only at 1.0x
- **Pipeline fix:** Truncate trace at "####" (teacher-force reasoning only), lookback re-computation (20 tokens), generate answer from end of reasoning
- **Answered Q37, Q40, Q41:** Ultra-fragility was a pipeline bug. Cliff is at 0.3x-1.0x additive noise. Reversed dissociation is a novel fourth pattern
- Evidential strength: strong (n=30, corrects pipeline bug, large effect size, clean results)
- See `experiment_log/exp_015.md`, `results/exp_015/`

### Exp 016 (Cycle 17) — POSITIONAL CONFOUND CONFIRMED
**Positional Confound Analysis on Qwen3-4B-Instruct**
- Model: Qwen/Qwen3-4B (instruct), n=26 problems (65% pipeline baseline, 85% direct generation)
- **Position is the dominant driver:** POS_EARLY=19.2% vs POS_LATE=65.4% (+46pp gap at 3% noise)
- **SelTC targets early positions (0.242), SelAC targets late positions (0.991)** — the -38.5pp "reversed dissociation" maps onto positional effects
- **Within-early-half gap collapses to +3.8pp** (vs unconstrained -38.5pp) — selectivity adds negligible information beyond position
- **Late-half gap persists (-38.5pp) but is NOT position-controlled** — Late+SelTC (pos 0.628) vs Late+SelAC (pos 0.991) still has massive positional confound
- **Protection replicates:** H2O (pos 0.987) = TC (pos 0.946) = 65.4% >> AC (pos 0.351) = 23.1%. Positionally explained.
- **Position-score correlations:** SEL-pos rho=+0.65, AC-pos rho=+0.38, TC-pos rho=-0.48, H2O-pos rho=-0.51
- **Scenario A (positional confound) confirmed for 6/7 pre-registered criteria**
- **MAJOR IMPLICATION:** Selectivity-based destruction dissociations on ALL instruct models are positional confounds. Only PGD on Qwen-Base provides genuine position-independent spatial structure evidence
- Evidential strength: strong (clean confound resolution, replicates exp_013 pattern, n=26)
- See `experiment_log/exp_016.md`, `results/exp_016/`

### Exp 017 (Cycle 18) — PARTIAL POSITIONAL CONFOUND ON QWEN-BASE
**Positional Confound Test on Qwen3-4B-Base**
- Model: Qwen/Qwen3-4B-Base, n=38 valid problems (95% baseline, same seed as exp_004)
- **Exp_004 replication:** +23.7pp unconstrained gap (vs exp_004's +23.5pp) — near-perfect replication
- **Position dominates:** POS_EARLY=34.2% vs POS_LATE=100% (+65.8pp gap at 5% noise)
- **Within-early-half gap COLLAPSES:** +2.6pp at 5%, 0.0pp at 10% — position controls eliminate selectivity effect
- **Within-late-half gap PERSISTS:** +42.1pp at 5%, +36.8pp at 10% — genuine residual selectivity likely
- **SEL-position correlation is NEGATIVE** (rho=-0.201) — OPPOSITE from instruct models (+0.57-0.65). On Base, high selectivity = early positions
- **Cross-strategy at similar positions:** Late+SelAC (pos 0.742) → 42.1% vs SelTC (pos 0.728) → 76.3% — 34pp difference at similar mean position, suggesting genuine selectivity beyond position
- **Scenario C (partial confound, 15% confidence) was BEST FIT** — position explains ≥70% of variance, selectivity adds genuine residual value in the late half
- **MAJOR IMPLICATION:** ALL destruction-based dissociation evidence is primarily positional across ALL models. Only PGD on Qwen-Base (rho=0.78) provides position-independent spatial structure evidence. The ~24pp "replicated across architectures" finding was primarily positional sensitivity, not functional separability.
- Evidential strength: strong (n=38, replicates exp_004, clean positional controls, nuanced result)
- See `experiment_log/exp_017.md`, `results/exp_017/`

### Exp 018 (Cycle 18/19) — DRAMATIC NON-REPLICATION OF PGD SPATIAL STRUCTURE
**Position-Controlled PGD Correlation Analysis**
- Model: Qwen/Qwen3-4B-Base, n=45 valid problems, 50 PGD steps (prompt-only attacks)
- **Bivariate rho(pert, attn) = 0.197** — does NOT replicate Exp 5's 0.78 (4x lower)
- **Partial rho(pert, attn | position) = 0.157 pooled / 0.043 per-attack median** — near zero per-attack
- **Within-quartile rhos: 0.10-0.24** — significant (p<0.01) but weak across all position quartiles
- **rho(pert, position) = 0.14, rho(attn, position) = 0.38** — position moderately confounds attention but not perturbation
- **rho(pert, TC) = 0.26** — opposite sign from Exp 5's -0.06. PGD also affects text-coupled positions.
- **Attack success: 0% genuine** (12/45 apparent successes all have zero perturbation — teacher-forcing vs autoregressive path divergence artifact)
- **Most likely explanation for Exp 5 discrepancy:** Exp 5 pooled 141 attacks mixing prompt-only, reasoning-only, and binary search. Reasoning-only attacks create high perturbation-attention correlation by construction (PGD avoids text positions). Prompt-only cascade effects are mostly positional.
- **Pre-registered predictions: bivariate rho BELOW ALL scenarios (all predicted ≥0.5)**. Partial metrics match Scenario B/C boundary.
- **MAJOR IMPLICATION:** The "last remaining position-independent evidence" for spatial structure is dramatically weakened. The null space EXISTS (Exp 4) but its spatial structure is much weaker than claimed. The hidden channel narrative reduces to: null space exists, but it operates through distributed small changes, not concentrated perturbation at answer-coupled positions.
- Evidential strength: strong negative (n=45, clean methodology, dramatic non-replication)
- See `experiment_log/exp_018.md`, `results/exp_018/`

### Exp 019 (Cycle 19) — DESIGNED BUT NOT EXECUTED
**Position-Controlled Double Dissociation**
- Model: Qwen/Qwen3-4B-Base
- Design: noise AC-selective vs TC-selective positions, measure BOTH answer accuracy AND text prediction accuracy
- Position-controlled via Q3/Q4 quartile strategies
- This is the gold-standard double dissociation test — researcher's #1 priority
- Agent crashed before execution. Script at `scripts/exp_019_double_dissociation.py`
- See `experiment_log/exp_019.md`

### Exp 020 (Cycle 20) — LITERATURE SCAN
**Second Literature Scan (14 papers across 4 topic clusters)**
- **Performative CoT (4 papers):** "Reasoning Theater" shows models decide internally 80% of tokens before visible CoT reveals it. FUR (EMNLP Outstanding) measures parametric faithfulness via unlearning. Wild unfaithfulness at 2-13%.
- **Functional separability (3 papers):** Direction-magnitude double dissociation in hidden states (42.9x vs 20.4% differential damage). CRV verifies reasoning via computational graph fingerprints (ICLR 2026 Oral). CoT features are widely distributed.
- **KV cache compression (6 papers):** Phase transition at ~90% compression (independent SNR cliff discovery). R-KV shows removing redundant tokens IMPROVES accuracy to 105%. RLKV finds only small fraction of heads critical for reasoning.
- **Attention sinks and position (2 papers):** Attention sinks explain why early positions are critical on ALL models — structural infrastructure, not content.
- **Major implication:** Functional separability is real but GEOMETRIC (distributed across representation dimensions) not SPATIAL (concentrated at specific token positions). Our PGD rho=0.20 is consistent with distributed geometric separation. The field is converging on "text is lossy projection" through independent methods.
- Literature notes in `literature_notes/cycle_020_*.md`
- See `experiment_log/exp_020.md`
