# Human Directives

Messages from the researcher, delivered via Slack or edited directly.
The agent reads this file at the start of each cycle and follows any new directives.

## Latest Directives

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

3. **Start with the "What remains untested" section in research_spec.md.**
   The most valuable next experiments are the double dissociation (pruning
   answer-coupled vs text-coupled KV positions) and cross-model replication
   of the KV adversarial experiments on Llama.

---

## Acknowledged Directives
(the agent moves directives here after acknowledging them)
