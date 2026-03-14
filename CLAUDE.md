# Autonomous Research Agent

This is an autonomous research agent that investigates scientific hypotheses
through iterative experimentation.

## How It Works
- `loop.sh` invokes Claude Code once per research cycle
- Before each cycle, `loop.sh` polls Slack for human directives → `human_directives.md`
- Each cycle: read directives + state → design experiment → run → analyze → update evidence
- The agent prompt is in `AGENT.md` — read it fully before each cycle
- The agent writes its own experiment scripts in `scripts/` (or uses templates in `templates/`)
- Evidence accumulates in `evidence_ledger.md`
- Status updates go to `experiment_log/latest_status.json`
- After each cycle, status is posted to Slack; the researcher can reply with directives

## Key Rules
- NEVER modify `AGENT.md`, `research_spec.md`, `loop.sh`, or `config.env`
- ALWAYS read `human_directives.md` at the start of each cycle and follow any new directives
- ALWAYS pre-register predictions before running experiments
- ALWAYS update `evidence_ledger.md` after each experiment
- ALWAYS write an experiment log to `experiment_log/exp_NNN.md`
- ALWAYS `git add`, `git commit`, and `git push` at the END of every cycle — this is non-negotiable
- ALWAYS post Slack progress updates mid-cycle using `bash notifications/post_update.sh "<msg>"`
- Write experiment scripts in `scripts/` — you have full autonomy to implement what you need
- Use WebSearch and WebFetch for literature searches
