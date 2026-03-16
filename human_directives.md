# Human Directives

Messages from the researcher, delivered via Slack or edited directly.
The agent reads this file at the start of each cycle and follows any new directives.

## Latest Directives

### 2026-03-17 — Phase 2 pivot: natural channel usage

**Phase 1 is complete. Stop running perturbation experiments.**

We have 64 cycles of strong perturbation-based evidence. The evidence ledger
is comprehensive. No more noise injection, PGD attacks, teacher-forcing
corrupted text, KV cache eviction benchmarks, or spectral analysis. That work
is done.

**Phase 2 focus: demonstrate the hidden channel is used during NORMAL generation.**

The critical gap for a paper is that all our evidence is interventionist. A
reviewer can say "you broke the KV cache and things broke — so what?" We need
observational evidence that the hidden channel carries information during
unperturbed inference.

**Three experiment directions, in priority order:**

1. **Early answer decodability (linear probe) — DO THIS FIRST.** Train a
   linear probe on K-cache activations at intermediate CoT positions to predict
   the final answer. Compare against a text-only baseline. If the K-probe
   decodes the answer earlier than the text reveals it, that's direct evidence
   the channel carries information beyond the text. This is the cleanest
   experiment and should be the first thing you run.

2. **Wrong reasoning, right answer (WRRA) analysis.** Find naturally occurring
   cases where the model writes incorrect arithmetic but gets the right final
   answer. Probe the K-cache at error positions — does it encode the correct
   value despite the text saying the wrong value? This is the strongest
   "smoking gun" but depends on finding enough natural error cases.

3. **Paraphrase disruption.** Generate CoT, then paraphrase the text
   (preserving all numbers and math, changing surface tokens). If accuracy
   drops, the original token choices carried hidden information in K-routing.
   This is the hardest to control well — do it last.

**Practical notes:**
- GPU is available — use it for all model inference and probe training
- Build on existing scripts infrastructure — reuse model loading, GSM8K eval,
  KV cache extraction code from Phase 1
- Start with Qwen3-4B-Base, replicate key findings on Llama-3.1-8B-Instruct
- The research_spec.md has been updated with detailed methodology for all three
  experiments — read it carefully before designing your first Phase 2 experiment
- You have full autonomy to design the experiments, but stay focused on these
  three directions. Don't go back to perturbation experiments.

---

### 2026-03-14 — Initial directives from researcher

1. **Post Slack progress updates throughout each cycle.** Use
   `bash notifications/post_update.sh "<message>"` at key milestones:
   - When you start designing the experiment
   - When you start running the experiment
   - When the experiment finishes (include whether it succeeded/failed)
   - When analysis is complete (include the key finding in one line)
   I want to be able to follow along from my phone between experiments.

2. **You have full autonomy to write your own experiment scripts.** Don't wait
   for pre-built templates — write whatever code you need in `scripts/` and run
   it. You can install packages, use torch/transformers/matplotlib, whatever is
   needed. Be resourceful.

3. ~~**Start with the "What remains untested" section in research_spec.md.**~~
   **SUPERSEDED by 2026-03-17 directive.** Phase 1 is complete. Focus on
   Phase 2 natural channel usage experiments.

---

## Acknowledged Directives
(the agent moves directives here after acknowledging them)
