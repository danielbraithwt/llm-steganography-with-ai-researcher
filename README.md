# Autonomous Research Agent

An autonomous research agent that investigates scientific hypotheses by designing experiments, running them, analyzing results, and iterating — all in a loop powered by Claude Code. Includes bidirectional Slack communication so you can steer the agent from your phone.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     OUTER LOOP (loop.sh)                  │
│  For each cycle:                                          │
│    0. Poll Slack channel for human directives              │
│    1. Invoke Claude Code with AGENT.md                    │
│    2. Claude Code reads directives + state, plans cycle   │
│    3. Claude Code runs experiment via template             │
│    4. Claude Code analyzes results, updates evidence      │
│    5. Claude Code writes status update                    │
│    6. Loop script posts status to Slack + sends email     │
│    7. Sleep, repeat                                       │
└──────────────────────────────────────────────────────────┘

       ┌─────────────┐          ┌──────────────────┐
       │   You        │◄────────│  Slack #research  │
       │  (phone/     │────────►│    -agent         │
       │   laptop)    │         └──────────┬───────┘
       └─────────────┘                     │
                                           │ bidirectional
                                           │
                                    ┌──────┴───────┐
                                    │  loop.sh      │
                                    │  polls for    │
                                    │  directives,  │
                                    │  posts status │
                                    └──────────────┘
```

## Quick Start

1. **Clone and set up:**
   ```bash
   git clone <your-repo-url>
   cd llm-steganography-with-ai-researcher
   bash scripts/init_workspace.sh
   ```

2. **Configure `config.env`** with your Slack bot token, channel ID, and email (see Slack Setup below).

3. **Edit `research_spec.md`** with your hypothesis, evidence standards, and any existing results.

4. **Replace placeholder templates** in `templates/` with your actual experiment scripts. Each template must follow the interface defined in `templates/base_template.py`.

5. **Run the agent:**
   ```bash
   ./loop.sh
   ```

   Options:
   ```bash
   ./loop.sh --max-cycles 20    # Run at most 20 cycles
   ./loop.sh --dry-run           # Propose experiments without executing
   ```

## Slack Setup (Bidirectional Communication)

Once set up, you can steer the agent from your phone by messaging a Slack channel.

1. **Create a Slack app** at [api.slack.com/apps](https://api.slack.com/apps)
2. **Add Bot Token Scopes** under OAuth & Permissions:
   - `channels:history` (read messages)
   - `chat:write` (post status updates)
   - `channels:read` (list channels)
   - `users:read` (resolve user names in directives)
3. **Install to your workspace** and copy the Bot User OAuth Token (`xoxb-...`)
4. **Create a `#research-agent` channel** and invite the bot (`/invite @your-bot-name`)
5. **Copy the channel ID** (right-click channel name > Copy link, ID is the last segment)
6. **Add to `config.env`:**
   ```bash
   SLACK_BOT_TOKEN="xoxb-your-token-here"
   SLACK_CHANNEL_ID="C07ABC123"
   ```

**Sending directives:** Just message `#research-agent` from your phone or desktop:
- "Focus on layer ablation next, skip cross-model for now"
- "Re-run the epsilon sweep with num_problems=200"
- "Pause experiments, just do literature review this cycle"

The agent picks up your messages at the start of each cycle, acknowledges them, and adjusts its plan.

## Monitoring

- **Slack:** The agent posts a status update after each cycle with progress bars, findings, and next steps.
- **Dashboard:** Run `python scripts/review_dashboard.py` for a terminal summary.
- **Evidence ledger:** Read `evidence_ledger.md` for the full evidence record.
- **Experiment logs:** Browse `experiment_log/exp_NNN.md` for individual experiment details.

## Intervening

- **Via Slack:** Message the `#research-agent` channel (preferred — no need to SSH in).
- **Directly:** Stop the loop (`Ctrl+C`), edit `human_directives.md`, restart `./loop.sh`.
- **Emergency stop:** `Ctrl+C` — the loop finishes the current notification and exits cleanly.

## Adding Experiment Templates

1. Create a new Python file in `templates/` that subclasses `ExperimentTemplate` from `base_template.py`.
2. Implement `run_experiment()` — it should return a dict of results.
3. Add an entry to `templates/template_registry.json` with the template's name, script path, parameters, and outputs.
4. The agent will automatically discover it from the registry.

See `templates/example_template.py` for the interface pattern.

## File Structure

| File | Purpose |
|------|---------|
| `loop.sh` | Outer loop — invokes Claude Code per cycle |
| `AGENT.md` | Agent system prompt (the "brain") |
| `research_spec.md` | Hypothesis, evidence standards, scope |
| `evidence_ledger.md` | Running evidence record (agent-updated) |
| `human_directives.md` | Messages from you via Slack |
| `templates/` | Parameterized experiment scripts |
| `experiment_log/` | One markdown file per experiment |
| `literature_notes/` | Literature summaries |
| `results/` | Raw outputs: CSVs, plots, tensors |
| `notifications/` | Slack + email notification scripts |
| `scripts/` | Setup and review utilities |
| `config.env` | Configuration (not committed) |
| `CLAUDE.md` | Claude Code project memory |

## Key Design Principles

- **All state on disk + git:** No persistent agent memory between cycles. Everything is recoverable from the repo.
- **Pre-registration:** Predictions are written before experiments run. No post-hoc rationalization.
- **Disconfirmation seeking:** After 3 confirming experiments, the agent must design a challenge.
- **Mandatory git push:** Every cycle ends with a commit and push. A safety-net commit catches crashes.
- **Human-in-the-loop:** You steer via Slack; the agent follows your directives.
