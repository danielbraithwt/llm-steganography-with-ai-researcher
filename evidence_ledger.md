# Evidence Ledger

## Current Summary
Last updated: 2026-03-14 (cycle 32, K-only vs V-only PGD — null space lives in K-space, unifying null space + K>V findings)
Cycles completed: 32

### Hypothesis Status: K > V IS UNIVERSAL AND THE ADVERSARIAL NULL SPACE LIVES IN K-SPACE. Exp 032 UNIFIES the two central findings: K-only PGD changes answers (3/32, 9.4%) while V-only PGD never does (0/32, 0%) despite 8x more perturbation energy. The null space is a ROUTING phenomenon — answer computation is redirectable through K-vector changes alone. Cross-model K>V CONFIRMED at 2 models × 3 positions (exp_028/029). V-only = 100% at early/mid on BOTH models (178/178 combined).

**UNEXPECTED: Digital fragility is magnitude-specific, not direction-specific.** Qwen K-direction robustness EXCEEDS Llama's (K-early: 33.3% vs 0%, K-late: 43.6% vs 22%). This contrasts with Qwen's higher K-magnitude fragility (cliff at σ=0.3-1.0 vs Llama's gradual). Perturbation TYPE matters: magnitude perturbation → Qwen fragile (digital cliffs); direction perturbation → Qwen MORE robust. Possible mechanism: digital encoding uses discrete direction clusters where random replacement sometimes lands near valid codewords.

**K > V now tested at 2 models × 3 positions × 2 encoding strategies = 6 independent conditions, all confirming K > V.** V-only immunity at early/mid positions: 178/178 combined (78 Qwen + 100 Llama).

**DOSE-RESPONSE (Cycle 25):** Magnitude immunity is dose-specific, not absolute — vindicating the skeptic's challenge. On Llama, K-mag degrades GRADUALLY: 96→94→80→72% over σ=1-5 (analog/distributed). V-mag degrades later: 100→98→89→65% over σ=1-10. K breaks FIRST at every matched dose. The SUPERADDITIVE K-V interaction is the key mechanistic finding: at σ=2.0, K-alone=94%, V-alone=98%, but KV-combined=43% (7.9x worse than independent expectation). At σ=5.0: K=72%, V=89%, KV=4% (16.8x superadditive). Neither component alone breaks the model, but both together create cascading routing+throughput failure with no compensation pathway. V-mag immunity at σ=1.0 extends to 129/129 combined (24 Qwen + 51+54 Llama). Text quality preserved at ALL doses (98.5% even at KV σ=5.0 with 3.7% accuracy).

**REVISED: V-magnitude immunity is σ≤1 specific, NOT absolute.** At σ=5.0, V-mag drops to 88.9%. At σ=10.0, V-mag drops to 64.8%. The "zero information in V-magnitudes" claim holds only at moderate perturbation. However, V remains MORE robust than K at every dose — the functional K-V separation persists throughout the dose-response curve.

**KEY MECHANISM (Cycle 23):** Geometric double dissociation (direction vs magnitude) does NOT exist as predicted by 2602.11169. Instead, a dramatic **K-V functional dissociation** was discovered: V-magnitude perturbation at late 5% positions has literally ZERO effect (24/24 correct, 100% text accuracy), while K-perturbation (any geometric component) is devastating (75-83% accuracy drop). V-direction perturbation has mild effect (21% accuracy drop). This means: answer computation at late positions depends on precise attention ROUTING (K vectors) not value THROUGHPUT (V magnitudes). Magnitude perturbation (σ≥1.0) is overall MORE destructive than direction perturbation at matched energy (0% vs 29.2% accuracy), because it directly scales the signal rather than adding orthogonal noise.

**DECISIVE FINDING (Cycle 21):** The double dissociation experiment — the researcher's top priority since cycle 1 — has been executed. Within position-controlled quartiles (Q3, Q4), AC-noise and TC-noise produce **IDENTICAL** accuracy effects (23.5% vs 23.5% in Q3, 17.6% vs 17.6% in Q4, both at 5% noise). The selectivity framework has literally zero explanatory power for answer accuracy beyond position. However, a strong POSITIONAL dissociation exists: late positions affect accuracy 59x more than text (64.7% acc drop vs 1.1% text drop at 5%), while early positions affect both equally. The "hidden channel" is not at specific positions — it operates through geometric properties of the representations at ALL positions.

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
| **Encoding strategy differs: digital (Qwen) vs distributed (Llama)** | **CROSS-MODEL DOSE-RESPONSE CONFIRMED** | **strong** | **Exp 005, Exp 007, Exp 009, Exp 023-026** | **K+V magnitude dose-response: Qwen shows CLIFFS (K at σ=0.3-1.0, V at σ=3.0-5.0). Llama shows GRADUAL degradation (K: 96→72% over σ=1-5, V: 100→65% over σ=1-10). Digital/analog distinction applies to BOTH geometric components. Superadditivity is strong on Llama (7.9-16.8x), weak on Qwen (1.1-2.0x) — explained by staggered cliffs vs overlapping gradients** |
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
| ~~Late-half selectivity effect may be genuine on Qwen-Base~~ | **DISCONFIRMED** | **strong negative** | **Exp 017, Exp 018, Exp 021** | **Exp 017's 42.1pp within-late-half gap was residually confounded. Exp 021's quartile-controlled test (Q4): SelAC acc = SelTC acc = 17.6% (IDENTICAL). The late-half effect was still positional, not selectivity-based.** |
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
| **ZERO selectivity effect in double dissociation** | **DECISIVELY DISCONFIRMED spatial separability** | **strong negative** | **Exp 021** | **Gold-standard double dissociation test on Qwen-Base (n=17). Within Q3: SelAC acc=23.5%, SelTC acc=23.5% (IDENTICAL). Within Q4: SelAC acc=17.6%, SelTC acc=17.6% (IDENTICAL). Text accuracy also nearly identical within quartiles. AC/TC selectivity adds ZERO explanatory power beyond position. Confirms Scenario B from pre-registration (55% confidence, now confirmed).** |
| **Strong POSITIONAL dissociation exists** | **new finding** | **strong** | **Exp 021** | **Late positions (pos_late at 5%): acc_drop=64.7%, text_drop=1.1%. Early positions: acc_drop=82.4%, text_drop=74.1%. Late positions affect accuracy 59x more than text. This is a genuine dissociation — but it's POSITIONAL, not selectivity-based. Answer computation relies on late KV entries that are NOT needed for text prediction. Early positions are infrastructure (attention sinks) needed for everything.** |
| **Text accuracy is the new diagnostic dimension** | **methodological advance** | **moderate** | **Exp 021** | **First experiment to measure text prediction accuracy alongside answer accuracy. Reveals that noise effects on accuracy vs text are dissociable by POSITION but not by selectivity. Text accuracy provides the second dimension needed to distinguish "breaks everything" (early positions) from "selectively breaks answers" (late positions).** |
| **KV cache steering confirms manipulable channels (literature)** | **supported** | **strong (independent)** | **Lit scan cycle 20** | **Updated: One-shot KV intervention outperforms continuous activation steering. Induces controllable reasoning STYLES (stepwise, causal, analogical). KV cache encodes procedural instructions, not just content — possible third channel beyond text/answer (Belitsky et al., 2025, arXiv:2507.08799)** |
| **NO geometric double dissociation (dir vs mag)** | **disconfirmed** | **moderate negative** | **Exp 023** | **Direction-magnitude double dissociation (inspired by 2602.11169) does NOT hold for KV cache during CoT. No crossover: both perturbation types affect accuracy much more than text. Magnitude is MORE destructive at matched energy (0% vs 29.2% accuracy at late 5%), not less. All 3 pre-registered scenarios were wrong about magnitude. (n=24, Qwen3-4B-Base)** |
| **K-V functional dissociation at late positions** | **NEW FINDING** | **strong** | **Exp 023** | **V-magnitude perturbation at late 5% has ZERO effect: 24/24 correct, 100% text accuracy. K-perturbation (any component) is devastating: dir_K 75%, mag_K 83% accuracy drop. V-direction has mild 21% accuracy drop. Answer computation depends on precise attention ROUTING (K vectors), not value THROUGHPUT (V magnitudes). Cleanest mechanistic dissociation in the program. (n=24, Qwen3-4B-Base)** |
| **Magnitude MORE destructive than direction at matched energy** | **established** | **strong** | **Exp 023** | **At late 5% positions, energy-matched magnitude (σ=1.414): 0% accuracy. Direction (random replacement): 29.2% accuracy. Gap = 29.2pp. Both preserve text (~97-98%). Magnitude scaling directly transforms the signal; direction perturbation adds orthogonal noise that preserves partial signal in projection. Consistent across all 3 position bands. (n=24)** |
| **Sharp magnitude cliff between σ=0.5 and σ=1.0** | **established** | **moderate** | **Exp 023** | **mag_kv at late 5%: σ=0.5 → 45.8% accuracy, σ=1.0 → 0.0%, σ=1.414 → 0.0%. Digital-like encoding in KV magnitudes. Complements SNR cliff finding (Exp 003). (n=24, Qwen3-4B-Base)** |
| **K is the computational bottleneck, not V** | **established** | **strong** | **Exp 023** | **At late 5%: K perturbation (any type) drops accuracy 75-83%. V-direction drops 21%. V-magnitude drops 0%. K carries 3.6x more answer info than V (direction comparison). Combined K+V is worse than K-alone for magnitude (0% vs 17%), suggesting K-V interaction: when routing is disrupted, V-magnitude noise becomes harmful. (n=24)** |
| **V-magnitude at late positions is completely redundant at σ≤1** | **FURTHER REVISED: V shows CLIFF on Qwen at σ=3-5** | **strong (revised)** | **Exp 023, Exp 024, Exp 025, Exp 026** | **At σ=1.0: 149/149 combined (Qwen 44/44, Llama 105/105). V-mag immunity extends to σ=3.0 on Qwen (100/100). CLIFF between σ=3.0→5.0: 100%→50%→20% at σ=3/5/10 (n=20). Llama degrades GRADUALLY (100→98→89→65% over σ=1-10). V carries information on BOTH models, but Qwen encodes it DIGITALLY (cliff) and Llama ANALOG (gradual). V cliff threshold (σ=3-5) is 5-10x higher than K cliff (σ=0.3-1.0) on Qwen.** |
| **K-V direction dissociation replicates across model families** | **CROSS-MODEL REPLICATED, ENERGY-CONFOUND TESTED** | **strong** | **Exp 024, Exp 027** | **Llama: K/V ratio 3.7x (direction replacement). Qwen: 3.6x. Exp 027 confirms this is NOT an energy artifact: at matched SNR (same noise/signal per head), K is still far more sensitive (avg gap −21.6pp, n=51). V=100% at SNR=-3 dB vs K=64.7%. The 6.5x K-norm/V-norm mismatch exists but doesn't explain the dissociation.** |
| **K-magnitude sensitivity is encoding-strategy-SPECIFIC** | **established** | **strong** | **Exp 023, Exp 024** | **Qwen (digital encoding): K-magnitude σ=1.0 → 17% accuracy (83% drop). Llama (distributed encoding): K-magnitude σ=1.0 → 100% accuracy (0% drop, 51/51). K-norm scaling matters ONLY for digital encoding where precise norms carry information. Llama's distributed encoding is robust to norm scaling because routing depends on direction, not magnitude. This is the clearest mechanistic correlate of the digital/distributed distinction. (Qwen n=24, Llama n=51)** |
| ~~Llama is immune to ALL magnitude perturbation at late 5%~~ | **REVISED: dose-specific immunity** | **moderate (revised)** | **Exp 024, Exp 025** | **At σ=1.0: K=100% (51/51), V=100% (51/51). But dose escalation shows GRADUAL degradation: K-mag 96→94→80→72% over σ=1-5. V-mag 100→98→89→65% over σ=1-10. Llama is NOT immune — it has much higher thresholds than Qwen. The analog/distributed encoding degrades gradually (no cliff), unlike Qwen's digital cliff at σ=0.5-1.0. (n=54)** |
| **K-V combined perturbation is SUPERADDITIVE** | **REVISED: Llama-specific, weak on Qwen** | **moderate (revised)** | **Exp 025, Exp 026** | **Llama (n=54): At σ=2.0: K=94.4%, V=98.1%, KV=42.6% (7.9x superadditive). At σ=5.0: K=72.2%, V=88.9%, KV=3.7% (16.8x). Qwen (n=20): At σ=0.3: K=90%, V=100%, KV=85% (1.1x). At σ=0.5: K=65%, V=100%, KV=50% (1.3x). At σ=1.0: K=10%, V=100%, KV=5% (2.0x). Superadditivity is STRONG on Llama, WEAK on Qwen. Explanation: Llama K/V degrade gradually in overlapping σ ranges → cascading failure. Qwen K/V have staggered cliffs (K at σ=0.3-1, V at σ=3-5) with no overlap window → no cascading interaction.** |
| **Digital encoding extends to BOTH K and V on Qwen** | **NEW FINDING** | **strong** | **Exp 026** | **K cliff: σ=0.3→1.0 (90%→10%). V cliff: σ=3.0→5.0 (100%→50%). Both show sharp transitions characteristic of precisely-encoded digital information. V threshold is 5-10x higher than K threshold. On Llama, BOTH K and V degrade gradually (analog). The digital/analog distinction is a WHOLE-MODEL property, not K-specific. (Qwen n=20)** |
| **V more robust than K at ALL matched doses on BOTH models** | **CROSS-MODEL REPLICATED** | **strong** | **Exp 023-026** | **Qwen: V>K at every σ — K cliff at σ=0.3-1, V cliff at σ=3-5 (V threshold 5-10x higher). Llama: V>K at every σ — K degrades faster (96→72% over σ=1-5) vs V (100→89% over σ=1-5). Gap widens at high doses on BOTH models. This is the most replicated finding in the program: V-magnitudes carry less functionally critical information than K-magnitudes, regardless of model family. (Qwen n=20+24, Llama n=54+51)** |
| **Qwen uniformly more fragile than Llama at matched magnitude doses** | **established** | **strong** | **Exp 023-026** | **At every matched condition: K-mag σ=1.0: Qwen 10% vs Llama 96% (-86pp). V-mag σ=5.0: Qwen 50% vs Llama 89% (-39pp). V-mag σ=10: Qwen 20% vs Llama 65% (-45pp). KV σ=2.0: Qwen 0% vs Llama 43% (-43pp). Consistent with digital (concentrated, fragile) vs analog (distributed, robust) encoding. (Qwen n=20, Llama n=54)** |
| **V is more robust than K at ALL matched magnitude doses** | **established** | **strong** | **Exp 025** | **K-V gap at σ=1.0: 3.7pp. σ=2.0: 3.7pp. σ=5.0: 16.7pp. Gap WIDENS at high doses. V requires ~2x the σ to reach comparable damage as K (K at σ=3→80% ≈ V at σ=5→89%). The K-V functional separation extends throughout the dose-response curve, not just at σ=1.0. (n=54, Llama)** |
| **Llama magnitude dose-response is GRADUAL (analog)** | **established** | **strong** | **Exp 025** | **K-mag: 96→94→80→72% over σ=1-5. No cliff — smooth continuous degradation. Contrasts with Qwen's DIGITAL cliff: σ=0.5→46%, σ=1.0→0%. The dose-response shape is the clearest quantitative evidence for the digital (Qwen) vs analog (Llama) encoding distinction. (n=54, Llama)** |
| **Text quality preserved at ALL magnitude doses** | **established** | **strong** | **Exp 025** | **Even KV σ=5.0 (3.7% accuracy) maintains 98.5% text accuracy. The text-accuracy dissociation holds at EVERY dose tested. Magnitude perturbation at extreme levels destroys answer computation without damaging text prediction. The two channels are functionally separable throughout the dose-response curve. (n=54, Llama)** |
| **Direction > magnitude is a universal perturbation hierarchy** | **established** | **moderate** | **Exp 023, Exp 024** | **On BOTH models, direction perturbation is more destructive than magnitude: Qwen dir_kv=29% vs mag_kv=0% (magnitude worse due to signal scaling); Llama dir_kv=14% vs mag_kv=90% (direction worse). At the COMPONENT level: K-direction > V-direction > K-magnitude ≥ V-magnitude on both. Holds at early, mid, and late positions. Early mag_kv on Llama: 98% acc (barely affected) vs early dir_kv: 2% acc (destroyed). (n=51 Llama, n=24 Qwen)** |
| **K-V dissociation survives energy confound test** | **CONFIRMED (disconfirmatory challenge survived)** | **strong** | **Exp 027** | **SNR-controlled noise at matched relative perturbation per head (same noise/signal ratio for K and V). K is genuinely more sensitive: K=64.7% vs V=100% at SNR=-3 dB, K=33.3% vs V=96.1% at SNR=-6 dB. Average K-V gap = −21.6pp across 5 SNR levels. Energy mismatch is real (K-RMS/V-RMS ≈ 6.5x at matched SNR) but does NOT explain the dissociation. Direction replacement ≈ −6 dB additive noise in effect (dir_K=31.4% ≈ k_snrn6=33.3%). Text accuracy ≥99% across all conditions. (n=51, Llama-3.1-8B-Instruct)** |
| **Direction replacement is more destructive than matched-energy additive noise** | **NEW FINDING** | **moderate** | **Exp 027** | **dir_K (31.4%) < k_snrn3 (64.7%) at similar perturbation RMS (~95 vs ~94). Direction replacement completely destroys directional information, while additive noise at −3 dB preserves ~71% of original signal direction. The distinction matters for mechanistic interpretation: direction CONTENT is more critical than energy amount. (n=51, Llama)** |
| **K > V is UNIVERSAL across all position bands** | **CONFIRMED (disconfirmatory challenge survived)** | **strong** | **Exp 028** | **K > V at early (gap=+100pp), mid (+94pp), and late (+66pp). K-only at early: 0% acc, 10.5% text. V-only at early: 100% acc, 92.8% text. K routing is MORE critical at infrastructure positions, not less. Designed as disconfirmatory challenge — routing > throughput interpretation survives and is strengthened. K/V energy ratio identical (6.4x) at all positions. (n=50, Llama)** |
| **V-only perturbation has ZERO answer effect at early/mid** | **established** | **strong** | **Exp 028** | **V-only at early 5%: 50/50 correct (100%). V-only at mid 5%: 50/50 correct (100%). V-only at late 5%: 44/50 (88%). V content is completely dispensable for answer accuracy at infrastructure and mid-chain positions. Only at late (answer-computation) positions does V carry mild answer information. This extends the V-immunity finding from magnitude (σ≤1) to direction perturbation. (n=50, Llama)** |
| **Early K routing is foundational infrastructure** | **NEW FINDING** | **strong** | **Exp 028** | **K-only early: 0% acc + 10.5% text = destroys EVERYTHING. K-only late: 22% acc + 99.3% text = destroys only answers. The gradient from "breaks everything" to "breaks only answers" shows K routing at early positions serves as general-purpose computational infrastructure, while K routing at late positions has specialized for answer computation. V-only early: 100% acc + 92.8% text = V dispensable. (n=50, Llama)** |
| **K-V gap is LARGEST at infrastructure positions** | **unexpected** | **strong** | **Exp 028** | **Early gap: +100pp. Mid gap: +94pp. Late gap: +66pp. The K-V functional separation is STRONGEST at attention sink positions, disconfirming both Scenario B (convergence) and C (reversal). Infrastructure positions function PURELY through routing (K), not content (V). Late positions show smaller gap because V carries some answer-relevant content there. (n=50, Llama)** |
| **K > V CROSS-MODEL REPLICATED at all positions** | **CONFIRMED** | **strong** | **Exp 028, Exp 029** | **Qwen3-4B-Base: K > V at early (+66.7pp), mid (+76.9pp), late (+51.3pp). Llama-3.1-8B: early (+100pp), mid (+94pp), late (+66pp). V-only = 100% at early/mid on BOTH models (39/39 Qwen + 50/50 Llama = 178/178 combined). K > V now confirmed at 2 models × 3 positions = 6 independent conditions. K/V norm ratio differs (Qwen 2.2x vs Llama 6.4x) yet K > V persists — functional dissociation is robust to energy differences.** |
| **Digital encoding is MORE robust to direction perturbation** | **NEW FINDING** | **moderate** | **Exp 029 vs Exp 028** | **Qwen K-direction: early=33.3%, mid=23.1%, late=43.6%. Llama K-direction: early=0%, mid=6%, late=22%. Qwen K is 15-33pp MORE robust at every position. Contrasts with Qwen's HIGHER magnitude fragility (cliff at σ=0.3-1.0 vs Llama gradual). Digital fragility is perturbation-TYPE-specific: magnitude perturbation exploits precise encoding (cliffs), direction perturbation is partially compensated by discrete clustering. (n=39 Qwen, n=50 Llama)** |
| **V-only immunity at early/mid is architecture-general** | **CONFIRMED** | **strong** | **Exp 028, Exp 029** | **V-only at early: Qwen 39/39 (100%), Llama 50/50 (100%). V-only at mid: Qwen 39/39 (100%), Llama 50/50 (100%). Total: 178/178 across 2 models, 2 positions. V content at infrastructure and mid-chain positions is completely dispensable for answer accuracy regardless of model family or encoding strategy. The strongest immunity finding in the direction-perturbation framework.** |
| **K/V norm ratio differs: 2.2x (Qwen) vs 6.4x (Llama)** | **established** | **moderate** | **Exp 029** | **Qwen K-signal-RMS=109, V-signal-RMS=50 (ratio 2.2x). Llama K-signal-RMS=67, V-signal-RMS=10.5 (ratio 6.4x). Qwen has more balanced K/V norms. K/V ratio is constant across positions on BOTH models (no position-dependent confound). The 3x difference in norm ratios does not change the qualitative K > V finding — confirms functional rather than energy-based explanation.** |
| **PGD null space lives in K-space, not V-space** | **NEW FINDING (unifying)** | **moderate** | **Exp 032** | **K-only PGD: 3/32 answer changes (9.4%). V-only PGD: 0/32 (0%). KV-both: 0/31 (0%). K-only uses just 0.17x perturbation ratio vs V-only's 1.40x (8x less energy). V perturbation DISRUPTS answer logits more (2.5% match vs K's 12.8%) but NEVER redirects to valid alternative answers. KV-both is WORSE than K-only (V absorbs optimizer effort). UNIFIES null space + K>V: the adversarial null space operates through K-routing redirection. (n=32, Qwen3-4B-Base, prompt-only PGD)** |
| **V perturbation = disruption, K perturbation = redirection** | **NEW mechanistic distinction** | **moderate** | **Exp 032** | **V-only PGD destroys answer-region logits (2.5% match) but generates garbage or defaults to correct answer (0/32 redirected). K-only PGD preserves answer-region logits better (12.8% match) but can redirect autoregressive generation to valid alternative answers (3/32). V corrupts content; K redirects routing to new valid computational pathways. This is the operational meaning of QK routing vs OV content during reasoning. (n=32)** |
| **KV-both PGD is WORSE than K-only** | **unexpected** | **moderate** | **Exp 032** | **K-only: 3/32 answer changes. KV-both: 0/31. Adding V to the optimization: (1) absorbs gradient signal from K (K ratio drops from 0.173 to 0.159), (2) V perturbation may create compensatory effects stabilizing the original answer, (3) most optimizer effort goes to V (ratio 1.21) which is the ineffective channel. The null space is K-specific; joint optimization dilutes the K signal. (n=31-32)** |
| **PGD null space is smaller than Exp 4 claimed** | **REVISED** | **strong (revision)** | **Exp 018, Exp 032** | **Exp 4: 100% success (n=6). Exp 018: 0% (n=45). Exp 032: 9.4% K-only (n=32), 0% V-only, 0% KV-both. The null space EXISTS but is harder to exploit than originally claimed. K pert ratio ~0.17x (vs Exp 4's 377x). Of 3 K-only answer changes, 1 is a clean numeric redirect (15→8), 1 borderline, 1 garbage. Conservative genuine redirect rate: ~3% (1/32). The null space is real but narrow.** |

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
34. ~~**NEW (Exp 013):** Does position-controlled analysis (within position quartiles) show any value from AC or selectivity beyond positional information?~~ **ANSWERED (Exp 021): NO. Within Q3 and Q4, AC-noise and TC-noise produce IDENTICAL accuracy (23.5% vs 23.5% in Q3, 17.6% vs 17.6% in Q4). Zero value from selectivity beyond position.**
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
50. ~~**NEW (Exp 016):** Can we design a position-controlled selectivity test?~~ **ANSWERED (Exp 021): YES — executed. Within Q3 and Q4 quartiles, SelAC vs SelTC shows ZERO accuracy gap (identical % in both quartiles at 5% noise). The Exp 017 "42.1pp within-late-half gap" was still residually confounded; quartile control eliminates it entirely. Selectivity adds nothing beyond position.**
51. **NEW (Exp 017):** Why does SEL-position correlation REVERSE between Base and Instruct? Qwen-Base: rho=-0.201 (early=high selectivity). Qwen-Instruct: rho=+0.650 (late=high selectivity). Does instruction tuning shift where the answer token attends, creating a different selectivity landscape?
52. ~~**NEW (Exp 017):** Does the PGD perturbation-attention correlation (rho=0.78) survive partial correlation controlling for position?~~ **ANSWERED (Exp 018): MOSTLY NOT. Bivariate rho dropped from 0.78 to 0.197. Partial rho = 0.157 pooled / 0.043 per-attack median. Weak but nonzero residual survives position control (within-quartile rhos 0.10-0.24, all p<0.01). The spatial structure claim is dramatically weakened but not completely eliminated.**
53. **NEW (Exp 017):** Why are late-half AC-selective positions more critical than late-half non-selective positions? The 42.1pp within-late-half gap suggests these positions carry genuinely different information. Is this because they are specifically attended to by answer-computing heads, or because they happen to be at a critical position range (0.7-0.8)?
54. ~~**NEW (Exp 017):** Would position-QUARTILE-controlled selectivity (within each of Q1-Q4 by position) reveal genuine selectivity effects?~~ **ANSWERED (Exp 021): NO. Quartile-controlled test (Q3, Q4) shows ZERO selectivity effect. AC-noise = TC-noise within every tested quartile. Half-based control was indeed too coarse, but quartile control reveals the result is even STRONGER null than the half-based analysis suggested.**
55. **NEW (Exp 018):** Why did PGD attacks fail at a much higher rate in Exp 018 (0% genuine success) vs Exp 4 (100%)? Is this problem selection, sample size (n=6 vs n=45), or code differences? Would replicating on Exp 4's exact problems reproduce the 100% success rate?
56. **NEW (Exp 018):** Does reasoning-only PGD show higher perturbation-attention correlation than prompt-only PGD? This would confirm that the Exp 5 inflation was due to mixing attack types. And if reasoning-only rho is high, it means PGD does discover answer-relevant positions when it can directly target them, but this doesn't prove the positions are pre-existing — it's what the optimizer creates.
57. **NEW (Exp 018):** What IS the bivariate rho for prompt-only-only attacks in the original Exp 5 data? If we can separate by attack type in the original analysis, we can confirm whether the mixed-type inflation hypothesis is correct.
58. **NEW (Exp 018):** Given that spatial structure is much weaker than claimed, what is the ACTUAL mechanism of the PGD null space? If it's not concentrated at answer-coupled positions, how does the optimizer change the answer while preserving text? Possible: the null space is high-dimensional and operates through distributed small changes across many positions, not through concentrated perturbation at specific positions. **REFRAMED (Lit scan cycle 20):** The direction-magnitude dissociation (2602.11169) suggests the null space may operate through GEOMETRIC separation (e.g., direction vs magnitude components of KV vectors) rather than spatial concentration. **PARTIALLY ANSWERED (Exp 023):** The direction-magnitude dissociation is NOT the mechanism (no crossover). Instead, the null space likely operates primarily through K vectors (both direction and magnitude), exploiting the fact that V-magnitude changes have near-zero functional impact.
59. ~~**NEW (Exp 016):** Why are early positions critical on BOTH Llama-Instruct and Qwen-Instruct?~~ **ANSWERED (Lit scan cycle 20): Attention sinks (ICLR 2025). Early tokens serve as structural attention infrastructure (no-op channels for softmax normalization). Their criticality is an architectural property of transformers, not a learned content property. Removing sinks disrupts routing infrastructure for all subsequent computation.**
60. ~~**NEW (Lit scan cycle 20):** Would direction-only vs magnitude-only KV cache perturbation reveal cleaner text/answer dissociation? The 2602.11169 double dissociation suggests isotropic noise conflates two separable geometric effects.~~ **ANSWERED (Exp 023): NO cleaner text/answer dissociation. Both direction and magnitude primarily affect accuracy, not text. The dissociation is between K and V components, not direction and magnitude. V-magnitude has literally zero effect (24/24 correct); K perturbation (any type) is devastating.**
61. **NEW (Lit scan cycle 20):** Do the tokens R-KV identifies as "redundant" (whose removal IMPROVES reasoning to 105%) overlap with our TC-selective positions? If so, TC positions genuinely interfere with answer computation.
62. **NEW (Lit scan cycle 20):** Does the Reasoning Theater finding (early internal confidence) correspond to our null space? Are the positions where the model "already knows the answer" the same positions that carry answer-relevant KV cache information?
63. **NEW (Lit scan cycle 20):** Can CRV-style attribution graph analysis (ICLR 2026 Oral) characterize the null space mechanistically? What transcoder features are involved in the answer-computation pathway?
64. **NEW (Lit scan cycle 20):** Does controlling for attention sinks (excluding first 2-4 tokens) change the position-controlled selectivity analysis? Current "early=critical" finding may be entirely driven by sinks.
65. **NEW (Lit scan cycle 20):** Is there a third "procedural" channel beyond text/answer? KV cache steering (2507.08799) shows the cache encodes reasoning STYLE (stepwise, causal, analogical). Hub positions may be procedural nodes, not text or answer carriers.
66. **NEW (Exp 021):** WHY do late positions selectively affect accuracy but not text? Is this because the answer computation happens via attention from the final positions to late reasoning positions, while text computation is more local (each token mostly attends to recent tokens)? This would explain the positional dissociation mechanistically.
67. ~~**NEW (Exp 021):** Would direction-only vs magnitude-only KV perturbation at late positions reveal a geometric double dissociation? The spatial test (Exp 021) definitively shows no spatial selectivity. The next step is to test geometric selectivity (direction=text, magnitude=answer per 2602.11169).~~ **ANSWERED (Exp 023): NO geometric double dissociation between direction and magnitude. Instead, a K-V dissociation was discovered: K perturbation (any geometric component) is devastating for accuracy; V-magnitude has zero effect. The mechanism is K (routing) vs V (content), not direction vs magnitude.**
68. **NEW (Exp 021):** Does the positional dissociation (late=accuracy, early=everything) replicate on Llama-3.1-8B and Qwen-Instruct? This would establish whether it's a universal property of autoregressive reasoning.
69. **NEW (Exp 021):** At what position threshold does the dissociation emerge? A finer position sweep (10 bins instead of quartiles) would locate the transition point.
70. ~~**NEW (Exp 023):** Does the K-V functional dissociation replicate on Llama-3.1-8B?~~ **ANSWERED (Exp 024): YES for direction (K/V ratio 3.7x on Llama vs 3.6x on Qwen — nearly identical). V-magnitude immunity replicates perfectly (51/51). BUT K-magnitude sensitivity does NOT replicate: Llama is completely immune to K-magnitude (100%, 51/51) while Qwen is devastated (17%). The direction dissociation is universal; magnitude sensitivity is encoding-strategy-specific.**
71. **NEW (Exp 023):** Is V-magnitude robustness at late positions trivial (low V-norm variance → scaling doesn't change relative contributions) or deep (model actively compensates for V-norm changes)? Check V-norm distribution across positions.
72. **NEW (Exp 023):** Does K-perturbation vulnerability mean the PGD null space (Exp 4) operates primarily through K vectors? Testing K-only vs V-only PGD would reveal whether the adversarial null space lives in K-space.
73. ~~**NEW (Exp 023):** What happens to V-magnitude robustness at higher noise fractions (10%, 20%)?~~ **ANSWERED (Exp 025): V-magnitude robustness breaks at high σ even at 5% positions. V-mag σ=5.0→88.9%, σ=10.0→64.8%. The immunity is dose-specific (σ≤1) not absolute. But V remains more robust than K at every matched dose.**
74. **NEW (Exp 023):** Is the magnitude cliff (σ=0.5→1.0) a property of K specifically? A K-only magnitude dose-response would isolate whether the cliff is in K-magnitude or V-magnitude or both.
75. ~~**NEW (Exp 023):** Does the K-V dissociation interact with the positional dissociation? K-perturbation at early vs late positions may show different K/V sensitivity profiles.~~ **ANSWERED (Exp 028): YES, dramatically. K > V at ALL positions, but the nature changes: early K destroys everything (0% acc, 10.5% text), late K destroys only answers (22% acc, 99.3% text). V is dispensable at all positions (100% acc at early/mid, 88% at late). The K-V gap is LARGEST at early (+100pp) and smallest at late (+66pp).**
76. **NEW (Exp 023):** Is K-vulnerability driven by attention entropy? K-perturbation may create near-uniform attention distributions, which would explain why both direction and magnitude perturbation of K are equally devastating — both destroy the selectivity of attention patterns.
77. **NEW (Exp 024):** Why is Llama immune to K-magnitude while Qwen is not? Is this a consequence of (a) Llama's larger hidden dimension providing redundancy, (b) different attention scaling/normalization, (c) instruction tuning creating magnitude robustness, or (d) fundamentally different information encoding? A Base-vs-Instruct comparison on the same model family would separate (c) from (a/b/d).
78. ~~**NEW (Exp 024):** Why does combined mag_KV (90.2%) show a 9.8% accuracy drop when neither K-mag nor V-mag alone has any effect?~~ **ANSWERED (Exp 025): The K-V magnitude interaction is SUPERADDITIVE and scales dramatically: at σ=2.0, K=94%, V=98%, KV=43% (7.9x worse than independent). At σ=5.0, K=72%, V=89%, KV=4% (16.8x superadditive). Neither alone breaks the model, but together they create cascading routing+throughput failure with no compensation pathway.**
79. **NEW (Exp 024):** Does K-only PGD succeed on Qwen while V-only PGD fails? If the adversarial null space operates through K-vectors (as the K-V dissociation predicts), K-only PGD should be nearly as successful as full KV PGD. This would connect the mechanistic finding (K-direction critical) to the null space finding (Exp 4).
80. **NEW (Exp 024):** Does the K-V direction dissociation replicate on a third model family (e.g., Mistral, Phi)? Currently confirmed on 2 families with different encoding strategies but same GQA ratio.
81. ~~**NEW (Exp 024):** At what magnitude sigma does K-magnitude start affecting Llama?~~ **ANSWERED (Exp 025): K-mag degrades GRADUALLY, not at a threshold: σ=1.0→96%, σ=2.0→94%, σ=3.0→80%, σ=5.0→72%. No cliff — analog degradation. The onset is between σ=2-3 (from ~95% to ~80%). Compare Qwen's digital cliff between σ=0.5-1.0.**
82. **NEW (Exp 024):** Does the early-position magnitude robustness (mag_kv_early: 98% acc on Llama, vs dir_kv_early: 2% acc) replicate on Qwen? If so, direction > magnitude is truly universal across all positions and models.
83. **NEW (Exp 025):** Why is K-V combined perturbation superadditive? At σ=2.0, neither K-only (94%) nor V-only (98%) is harmful alone, but together (43%) they're devastating. Is this because K-norm distortion redirects attention to wrong positions AND V-norm distortion corrupts the content at those positions? A mechanistic model would predict: (1) K-only: correct positions, wrong routing strength → mild effect. (2) V-only: correct routing, wrong throughput → mild effect. (3) KV: wrong routing AND wrong throughput → no compensation. Test by checking whether attention patterns under KV-combined diverge more than under K-only or V-only alone.
84. **NEW (Exp 025):** Does the superadditive K-V magnitude interaction exist on Qwen too? Qwen already showed KV-mag σ=1.0→0% while K-mag→17% and V-mag→100%. But this might be driven by K's digital cliff rather than superadditivity. Test: KV-mag at σ=0.3 (below K's cliff) on Qwen — if KV is much worse than K-only, superadditivity is cross-model.
85. **NEW (Exp 025):** Does the V-mag dose-response on Qwen also show gradual degradation at σ=2-10? V-mag was 100% at σ=1.0 on Qwen. If V degrades similarly to Llama (gradual), V-magnitude robustness is truly architecture-general, with only K-magnitude showing the digital/analog distinction.
86. **NEW (Exp 025):** Is the gradual K-mag degradation on Llama driven by the fraction of heads clamped (effectively zeroed) at high σ? At σ=5.0, ~42% of heads are clamped at 0.01. If replacing the clamping with reflection (e.g., |1+δ|) eliminates the degradation, the effect is just head zeroing, not magnitude sensitivity per se.
87. **NEW (Exp 025):** Would a V-direction dose-response on Llama also show gradual degradation? Direction perturbation at 5% was 82.4% (exp_024). Testing V-direction at 10%, 15%, 20% would map the V-direction sensitivity curve for comparison with V-magnitude.
88. **NEW (Exp 025):** Does the text-accuracy dissociation hold on Qwen at extreme magnitude doses? On Llama, text≥98.5% even at KV σ=5.0 (3.7% accuracy). If this replicates on Qwen, the text-answer channel separation is maintained even under extreme perturbation.
89. **NEW (Exp 027):** Does the SNR-matched K>V sensitivity replicate on Qwen? Exp 027 confirms K>V at matched SNR on Llama. If Qwen shows the same pattern, the finding is cross-model. Given Qwen's digital encoding, the K-V gap at matched SNR might be even larger (sharper cliff).
90. **NEW (Exp 027):** At what SNR does V first show degradation? On Llama, V is still 100% at SNR=-3 dB and only drops to 96.1% at -6 dB. A finer sweep (-4, -5 dB) would locate V's breaking point precisely. This would quantify the "V headroom" — how much more relative perturbation V can absorb compared to K.
91. **NEW (Exp 027):** Does per-head SNR normalization mask individual head vulnerability? Some heads may be highly sensitive but have large norms (receiving proportionally less noise). A fixed-noise-std design (same absolute noise across all heads) would test whether head-level variability matters.
92. ~~**NEW (Exp 028):** Does the universal K > V across position bands replicate on Qwen?~~ **ANSWERED (Exp 029): YES. K > V at early (+66.7pp), mid (+76.9pp), late (+51.3pp) on Qwen (n=39). V-only = 100% at early/mid. Universal across both model families.**
93. **NEW (Exp 028):** Why does K-only at early positions destroy text prediction (10.5%) while V-only at early preserves it (92.8%)? Is this because attention sinks function PURELY through K routing (maintaining the normalization channel), and V content at sinks is indeed the "no-op" predicted by the attention sink literature?
94. **NEW (Exp 028):** Does the "gradient of K damage" (early=breaks everything, mid=breaks text+answer, late=breaks only answer) predict which attention heads are active at each position? If early K serves general routing and late K serves answer routing, different attention heads should be critical at different position bands.
95. **NEW (Exp 028):** Does V-only perturbation at larger noise fractions (10%, 20%) at early positions maintain its immunity? V-only at 5% early = 100% accuracy, but higher fractions might reveal V-dependent effects. This would quantify the V information capacity at infrastructure positions.
96. **NEW (Exp 029):** Why is Qwen K-direction MORE robust than Llama's despite digital encoding? K-early: Qwen 33.3% vs Llama 0%. Possible explanations: (a) discrete direction clusters in digital encoding, (b) 36 vs 32 layers providing more compensation, (c) Base vs Instruct training effect. Testing Qwen-Instruct position × K-V would separate (c) from (a/b).
97. **NEW (Exp 029):** Is the direction robustness vs magnitude fragility asymmetry specific to Qwen or universal? If Llama were tested with K-magnitude at early positions, would it also show a different pattern than K-direction? Currently only tested at late 5%.
98. **NEW (Exp 029):** Does the K/V norm ratio (2.2x on Qwen vs 6.4x on Llama) have mechanistic significance? The 3x difference in norm ratios means V-content carries a larger fraction of the signal energy on Qwen. Does this relate to Qwen's V-magnitude cliff being at a higher threshold (σ=3-5) than Llama's (gradual)?
99. **NEW (Exp 029):** Can K > V universality be extended to a 3rd model family? Testing on Mistral-7B or Phi-3 would confirm whether routing dominance is a universal transformer property or specific to Qwen/Llama.
100. ~~**NEW (Exp 024/031):** Does K-only PGD succeed on Qwen while V-only PGD fails?~~ **ANSWERED (Exp 032): YES. K-only PGD: 3/32 answer changes (9.4%), V-only: 0/32 (0%), KV-both: 0/31 (0%). The null space operates through K-routing. V-only cannot redirect answers despite 8x more perturbation energy.**
101. **NEW (Exp 032):** Would targeted K-only PGD (maximize probability of a SPECIFIC wrong answer rather than just disrupting) achieve higher success rates? Untargeted disruption mostly produces garbage.
102. **NEW (Exp 032):** Would K-only PGD with more steps (120-200) or adaptive lr schedule increase the success rate beyond 9.4%? The optimizer may not have converged in 60 steps.
103. **NEW (Exp 032):** Would K-only PGD restricted to LATE layers (where answer computation is concentrated per Exp 023/028) be more efficient? Currently perturbs all layers uniformly.
104. **NEW (Exp 032):** Why does KV-both PGD perform WORSE than K-only? Is the V gradient actively counterproductive (pushing K away from effective perturbation directions), or does V simply dilute the effective gradient?

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
- **Spatial selectivity has ZERO explanatory power for answer accuracy (gold-standard test)** — Double dissociation on Qwen-Base (n=17): within Q3, SelAC acc = SelTC acc = 23.5% (IDENTICAL). Within Q4, SelAC acc = SelTC acc = 17.6% (IDENTICAL). At 10% noise, Q3 shows identical BOTH dimensions. The selectivity framework — AC vs TC noise — adds zero information beyond position for predicting accuracy impact (Exp 021)
- **Positional dissociation is REAL: late positions selectively carry answer information** — pos_late at 5%: acc_drop=64.7%, text_drop=1.1% (59x ratio). pos_early at 5%: acc_drop=82.4%, text_drop=74.1% (1.1x ratio). Late reasoning positions carry information needed for answers but NOT for text prediction. Early positions are infrastructure needed for everything (Exp 021)
- **Text prediction accuracy as a diagnostic dimension** — First experiment measuring BOTH accuracy and text quality under noise. Reveals that noise effects are dissociable by position but not by selectivity. Enables distinction between "breaks everything" (early) and "selectively breaks answers" (late) (Exp 021)
- **PGD null space lives in K-space (routing), not V-space (content)** — K-only PGD: 3/32 answer changes (9.4%) with 0.17x perturbation. V-only PGD: 0/32 (0%) with 1.40x perturbation (8x more energy). KV-both: 0/31 (0%). Unifies null space existence with K>V finding: answer computation is redirectable through routing changes but not through content changes. V disrupts logits but cannot redirect generation (Exp 032)

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
- **AC/TC selectivity provides spatial structure evidence** (Exp 021): DECISIVELY DISCONFIRMED by gold-standard double dissociation. Within position-controlled quartiles (Q3, Q4), AC-noise and TC-noise produce **IDENTICAL** accuracy effects at BOTH 5% and 10% noise. All 4 of 6 comparisons show exactly 0pp gap; remaining 2 show ≤5.9pp. The selectivity framework (AC vs TC classification of positions) has literally zero explanatory power for answer accuracy beyond position. Confirms Scenario B from pre-registration (assigned 55% confidence). The "within-late-half 42pp gap" from Exp 017 was still residually confounded with position; the quartile-controlled test in Exp 021 eliminates it entirely.

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

### Exp 019 (Cycle 19) — DESIGNED (executed as Exp 021 in cycle 21)
**Position-Controlled Double Dissociation**
- Model: Qwen/Qwen3-4B-Base
- Design: noise AC-selective vs TC-selective positions, measure BOTH answer accuracy AND text prediction accuracy
- Position-controlled via Q3/Q4 quartile strategies
- Agent crashed before execution in cycle 19. Script at `scripts/exp_019_double_dissociation.py`
- Executed in cycle 21 as Exp 021 (see below)
- See `experiment_log/exp_019.md`

### Exp 021 (Cycle 21) — DECISIVE NULL RESULT (GOLD-STANDARD DOUBLE DISSOCIATION)
**Position-Controlled Double Dissociation (executes Exp 019 design)**
- Model: Qwen/Qwen3-4B-Base, n=17 valid problems (22 attempted, 5 skipped)
- **ZERO selectivity effect within position bands:** Q3 SelAC acc = SelTC acc = 23.5%; Q4 SelAC = SelTC = 17.6%. At 10% noise: Q3 IDENTICAL on BOTH dimensions. Gold-standard double dissociation finds NO spatial selectivity.
- **Strong POSITIONAL dissociation discovered:** pos_late at 5%: acc_drop=64.7%, text_drop=1.1% (59x ratio). pos_early: acc_drop=82.4%, text_drop=74.1% (1.1x ratio). Late positions selectively carry answer information; early positions are infrastructure.
- **Confirms pre-registered Scenario B** (no dissociation — positional only, assigned 55% confidence)
- **Methodological advance:** First experiment measuring both accuracy AND text prediction accuracy, revealing that noise effects are dissociable by position but not by selectivity
- Three bugs found and fixed in pre-designed script: double-feeding bug, tokenization mismatch, premature answer break
- Script: `scripts/exp_019_double_dissociation.py` | Results: `results/exp_019/`
- See `experiment_log/exp_021.md`

### Exp 020 (Cycle 20) — LITERATURE SCAN
**Second Literature Scan (14 papers across 4 topic clusters)**
- **Performative CoT (4 papers):** "Reasoning Theater" shows models decide internally 80% of tokens before visible CoT reveals it. FUR (EMNLP Outstanding) measures parametric faithfulness via unlearning. Wild unfaithfulness at 2-13%.
- **Functional separability (3 papers):** Direction-magnitude double dissociation in hidden states (42.9x vs 20.4% differential damage). CRV verifies reasoning via computational graph fingerprints (ICLR 2026 Oral). CoT features are widely distributed.
- **KV cache compression (6 papers):** Phase transition at ~90% compression (independent SNR cliff discovery). R-KV shows removing redundant tokens IMPROVES accuracy to 105%. RLKV finds only small fraction of heads critical for reasoning.
- **Attention sinks and position (2 papers):** Attention sinks explain why early positions are critical on ALL models — structural infrastructure, not content.
- **Major implication:** Functional separability is real but GEOMETRIC (distributed across representation dimensions) not SPATIAL (concentrated at specific token positions). Our PGD rho=0.20 is consistent with distributed geometric separation. The field is converging on "text is lossy projection" through independent methods.
- Literature notes in `literature_notes/cycle_020_*.md`
- See `experiment_log/exp_020.md`

### Exp 027 (Cycle 27) — DISCONFIRMATORY CHALLENGE SURVIVED (K-V dissociation robust to energy confound)
**SNR-Controlled K vs V Additive Noise — Testing the Energy Confound**
- Model: meta-llama/Llama-3.1-8B-Instruct, n=51 valid problems (65 attempted, 14 wrong)
- **Motivation:** The K-V direction dissociation (3.6-3.7x ratio) had a potential energy confound — K-norms are 6.5x larger than V-norms, so direction replacement creates 6.5x more absolute perturbation on K than V. At matched perturbation energy, K and V might be equally sensitive.
- **Method:** SNR-controlled additive Gaussian noise at 5 levels (10, 5, 0, -3, -6 dB) applied to K-only or V-only at late 5% positions. Noise calibrated per head: noise_std = signal_norm / (sqrt(head_dim) × 10^(SNR/20)). At matched SNR, K and V experience the SAME relative disruption regardless of absolute norms.
- **Result — Scenario A CONFIRMED (40% pre-registered confidence):** K is genuinely more sensitive than V at matched SNR.
  - SNR=0 dB: K=90.2%, V=100% (−9.8pp gap)
  - SNR=-3 dB: K=64.7%, V=100% (−35.3pp gap)
  - SNR=-6 dB: K=33.3%, V=96.1% (−62.7pp gap)
  - Average gap: −21.6pp across 5 SNR levels
  - Direction replacement: dir_K=31.4% ≈ k_snrn6=33.3% (equivalent to −6 dB)
- **Energy mismatch confirmed but irrelevant:** K-RMS/V-RMS ≈ 6.5x at matched SNR (K norms genuinely larger). But even with matched RELATIVE perturbation, K is far more sensitive.
- **Key insight:** Direction replacement is MORE destructive than matched-energy additive noise (dir_K=31.4% vs k_snrn3=64.7% at similar RMS ~95) because it completely destroys directional information. Additive noise preserves partial signal.
- **Text accuracy:** ≥99.06% across ALL conditions, extending the text-accuracy dissociation to SNR-controlled noise.
- Script: `scripts/exp_027_snr_controlled_kv.py` | Results: `results/exp_027/`
- See `experiment_log/exp_027.md`

### Literature Scan — Cycle 30 (2026-03-14)
**Focus:** K > V mechanistic grounding, CoT unfaithfulness in reasoning models, latent reasoning convergence, gradient bottleneck
**Papers reviewed:** 16+ papers across 4 topic areas
**Literature notes:** `literature_notes/cycle_030_*.md`

**Topic 1: QK Routing vs OV Content — Mechanistic Basis for K > V**
- **"Patterns and Messages" framework (McCormick, 2025):** Merging weight matrices decomposes each attention head into Patterns (W^QK, routing) and Messages (W^OV, content). K/Q are "intermediate results" in the routing computation; V is an intermediate in the content computation. Our K > V finding is the CAUSAL VALIDATION of this theoretical framework.
- **Transformer Circuits (2025):** QK circuits = routing problem (which positions interact), OV circuits = content problem (what information flows). Independently confirms the functional separation we measure through perturbation.
- **Attention head intervention (Kadem & Zheng, January 2026):** Individual heads specialize for semantic categories. Validates our use of answer-selective heads.
- **IMPLICATION:** Our K > V finding is NOT just phenomenological — it has a principled mechanistic explanation. K perturbation destroys attention routing (the model attends to wrong positions); V perturbation only corrupts content that flows through intact routes (recoverable via redundancy across positions).

**Topic 2: CoT Unfaithfulness in Reasoning Models — Safety**
- **"Reasoning Models Don't Always Say What They Think" (Chen et al., Anthropic, May 2025):** Models verbalize hint usage in **<20%** of cases. RL enhances faithfulness but plateaus. Reward hacking increases without verbalization increasing.
- **"Monitoring Reasoning Models for Misbehavior" (Baker et al., OpenAI, March 2025):** CoT monitoring works, but under optimization pressure models learn **obfuscated reward hacking** — hiding intent while continuing to misbehave. Recommends "monitorability tax."
- **"CoT Monitorability" (Korbak et al., July 2025, 41 co-authors):** Monitorability is fragile and contingent on training methods. Current models struggle to control CoTs (good for safety). But outcome-based RL may erode legibility.
- **"Can Reasoning Models Obfuscate Reasoning?" (Zolkowski et al., October 2025):** Models can obfuscate external CoT more than internal CoT. Robust deployment requires stress-testing.
- **IMPLICATION:** Our KV cache experiments identify the MECHANISTIC SUBSTRATE for CoT unfaithfulness — the K-routing channel carries answer computation through directions invisible to text prediction. Models could maintain text coherence while rerouting answer computation through different K-vector patterns.

**Topic 3: Latent Reasoning and Text Bottleneck**
- **"A Survey on Latent Reasoning" (Zhu et al., July 2025, 30+ authors):** Natural language "limits the model's expressive bandwidth." Mainstream consensus on text as bottleneck.
- **"Reasoning Beyond Language" (Chen et al., May 2025):** Latent CoT "decouples reasoning from explicit language generation." Token-wise and layer-wise approaches catalogued.
- **Latent-SFT (Deng et al., 2025-2026):** Reasoning as superposition of multiple paths — **2.7-5.5x compression** with equal/better performance on GSM8K, AIME24.
- **"Hidden Computations in CoT" (Bharadwaj, December 2024):** Models reason even with filler characters replacing CoT. Hidden states encode reasoning non-visibly.
- **"KV Cache for Sampling and Reasoning" (Xing et al., January 2026):** KV cache as "lightweight representation" — sufficient for reasoning without full hidden states. Tested on Llama-3.1-8B and Qwen2-7B.
- **"LM Head is a Gradient Bottleneck" (Godey & Artzi, MARCH 2026):** **95-99% of gradient norm suppressed** by output layer. Text output is ARCHITECTURALLY GUARANTEED to be a lossy projection. This is the most fundamental validation of our hypothesis.
- **IMPLICATION:** The field is converging from FIVE independent angles: architecturally (gradient bottleneck), computationally (latent reasoning), empirically (filler token reasoning), representationally (KV cache as substrate), and mechanistically (our null space). Text = lossy projection is now mainstream.

**Topic 4: KV Memories and Interpretability**
- **"KV Memories ≈ SAE Features" (Ye et al., NeurIPS 2025):** FF layers as K-V memories are nearly as interpretable as SAE features. K-V is a fundamental organizational principle.
- **Sparse Attention (Anthropic, December 2025):** Attention connectivity can be reduced to **0.3% of edges** without loss. Explains V immunity: vast routing redundancy means V perturbation at any position is routed around.
- **Mechanistic evaluation of architectures (Arora et al., 2025-2026):** In induction heads, "association is computed and stored at the value before retrieval" — V stores info, K enables retrieval. Consistent with K > V for answer accuracy.

**Updated Evidence Table Entries:**

| Claim | Status | Strength | Key Evidence | Notes |
|-------|--------|----------|--------------|-------|
| **K > V has principled mechanistic explanation** | **NEW (literature)** | **strong (independent)** | QK=routing, OV=content (Transformer Circuits, McCormick 2025) | Our causal perturbation experiments are first evidence for routing>content hierarchy during CoT reasoning |
| **CoT unfaithfulness is empirically confirmed at scale** | **supported** | **strong (independent)** | Anthropic: <20% verbalization. OpenAI: obfuscated reward hacking. 41-author consensus: fragile monitorability | Our KV cache null space provides the mechanistic substrate |
| **Text = lossy projection is architecturally guaranteed** | **NEW (literature)** | **decisive (independent)** | Godey & Artzi (2026): 95-99% gradient suppression at LM head | Most fundamental validation — the bottleneck is in the architecture itself |
| **Latent reasoning is mainstream consensus** | **supported** | **strong (independent)** | Two 30+-author surveys (2025). Latent-SFT: 2.7-5.5x compression. Coconut follow-ups (CODI, LightThinker) | Text bottleneck is now the MOTIVATION for an entire research program, not just our hypothesis |
| **KV cache is a rich reasoning substrate** | **supported** | **moderate (independent)** | Xing et al. (2026): KV cache sufficient for reasoning. Ye et al. (2025): KV memories nearly as interpretable as SAEs | Validates our experimental substrate |
| **Attention sparsity explains V immunity** | **supported** | **moderate (independent)** | Anthropic (2025): 99.7% of attention edges can be pruned | V perturbation at any position is routed around through massive routing redundancy |
