#!/usr/bin/env bash
#
# loop.sh — Outer loop for the autonomous research agent.
#
# Invokes Claude Code once per research cycle. Each cycle:
#   0. Poll Slack for human directives
#   1. Invoke Claude Code with AGENT.md instructions
#   2. Run notifications
#   3. Safety-net git commit if needed
#   4. Sleep and repeat
#
# Usage:
#   ./loop.sh [--max-cycles N] [--dry-run]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configuration ───────────────────────────────────────────────────

if [ -f config.env ]; then
    # shellcheck source=/dev/null
    source config.env
else
    echo "Warning: config.env not found — using defaults"
fi

SLEEP_BETWEEN_CYCLES="${SLEEP_BETWEEN_CYCLES:-30}"
MAX_TOTAL_HOURS="${MAX_TOTAL_HOURS:-48}"
MAX_EXPERIMENT_MINUTES="${MAX_EXPERIMENT_MINUTES:-30}"
DRY_RUN="${DRY_RUN:-false}"

# ── Parse arguments ─────────────────────────────────────────────────

MAX_CYCLES=0  # 0 = unlimited
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-cycles)
            MAX_CYCLES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./loop.sh [--max-cycles N] [--dry-run]"
            exit 1
            ;;
    esac
done

# ── State tracking ──────────────────────────────────────────────────

START_TIME=$(date +%s)
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=3
SHUTTING_DOWN=false

# ── Determine next cycle number ─────────────────────────────────────

get_next_cycle() {
    local count
    count=$(find experiment_log/ -name "exp_*.md" 2>/dev/null | wc -l)
    echo $((count + 1))
}

# ── Graceful shutdown ───────────────────────────────────────────────

cleanup() {
    echo ""
    echo "Caught interrupt — finishing up..."
    SHUTTING_DOWN=true
}
trap cleanup SIGINT SIGTERM

# ── Budget check ────────────────────────────────────────────────────

check_budget() {
    local now elapsed_hours
    now=$(date +%s)
    elapsed_hours=$(( (now - START_TIME) / 3600 ))
    if [ "$elapsed_hours" -ge "$MAX_TOTAL_HOURS" ]; then
        echo "Compute budget exhausted ($elapsed_hours hours >= $MAX_TOTAL_HOURS hours)"
        return 1
    fi
    local remaining=$((MAX_TOTAL_HOURS - elapsed_hours))
    if [ "$remaining" -le 2 ]; then
        echo "Warning: Only $remaining hours remaining in compute budget"
    fi
    return 0
}

# ── Main loop ───────────────────────────────────────────────────────

echo "=== Autonomous Research Agent ==="
echo "Started at: $(date -Iseconds)"
echo "Max cycles: ${MAX_CYCLES:-unlimited}"
echo "Dry run: $DRY_RUN"
echo "Sleep between cycles: ${SLEEP_BETWEEN_CYCLES}s"
echo "Compute budget: ${MAX_TOTAL_HOURS}h"
echo "================================="
echo ""

CYCLES_RUN=0

while true; do
    # Check shutdown flag
    if [ "$SHUTTING_DOWN" = true ]; then
        echo "Shutting down gracefully after $CYCLES_RUN cycles."
        break
    fi

    # Check cycle limit
    if [ "$MAX_CYCLES" -gt 0 ] && [ "$CYCLES_RUN" -ge "$MAX_CYCLES" ]; then
        echo "Reached max cycles ($MAX_CYCLES). Stopping."
        break
    fi

    # Check compute budget
    if ! check_budget; then
        break
    fi

    # Check consecutive failures
    if [ "$CONSECUTIVE_FAILURES" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
        echo "ALERT: $CONSECUTIVE_FAILURES consecutive failures. Pausing for human review."
        echo "Fix the issue and restart loop.sh to continue."
        # Send alert
        bash notifications/notify.sh 0 2>/dev/null || true
        break
    fi

    CYCLE_NUM=$(get_next_cycle)
    CYCLE_NUM_PADDED=$(printf "%03d" "$CYCLE_NUM")

    echo "──────────────────────────────────────"
    echo "Cycle $CYCLE_NUM_PADDED — $(date -Iseconds)"
    echo "──────────────────────────────────────"

    # Step 0: Poll Slack for human directives
    echo "[0/4] Polling Slack for directives..."
    python3 notifications/slack_poll.py --output human_directives.md 2>/dev/null || true

    # Step 1: Invoke Claude Code
    echo "[1/4] Running Claude Code agent..."

    DRY_RUN_MSG=""
    if [ "$DRY_RUN" = true ]; then
        DRY_RUN_MSG=" This is a DRY RUN — propose experiments but do NOT execute them."
    fi

    EXPERIMENT_TIMEOUT=$((MAX_EXPERIMENT_MINUTES * 60))

    mkdir -p logs

    CLAUDE_EXIT=0
    claude --print \
        -p "You are an autonomous research agent. Read AGENT.md for your full instructions. This is cycle $CYCLE_NUM. Execute one complete research cycle. The experiment timeout is ${EXPERIMENT_TIMEOUT} seconds.${DRY_RUN_MSG}" \
        --allowedTools "Bash(command:*)" "Read" "Write" "Edit" "WebSearch" "WebFetch" \
        --max-turns 50 \
        2>&1 | tee "logs/cycle_${CYCLE_NUM_PADDED}.log" || CLAUDE_EXIT=$?

    # Step 2: Check result
    if [ "$CLAUDE_EXIT" -ne 0 ]; then
        echo "Warning: Claude Code exited with code $CLAUDE_EXIT"
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
    else
        CONSECUTIVE_FAILURES=0
    fi

    # Step 3: Safety-net git commit
    if [ -n "$(git status --porcelain)" ]; then
        echo "[2/4] Safety net: found uncommitted changes after cycle $CYCLE_NUM_PADDED"
        git add -A
        git commit -m "cycle ${CYCLE_NUM_PADDED}: safety-net commit (agent may have crashed before pushing)" || true
        for retry in 1 2 3; do
            if git push origin main 2>/dev/null; then
                echo "Safety-net push succeeded"
                break
            fi
            echo "Safety-net push attempt $retry failed — retrying in 10s..."
            sleep 10
        done
    fi

    # Step 4: Send notifications
    echo "[3/4] Sending notifications..."
    bash notifications/notify.sh "$CYCLE_NUM" 2>/dev/null || true

    CYCLES_RUN=$((CYCLES_RUN + 1))

    # Check if we should stop
    if [ "$SHUTTING_DOWN" = true ]; then
        echo "Shutting down gracefully after $CYCLES_RUN cycles."
        break
    fi

    echo "[4/4] Sleeping ${SLEEP_BETWEEN_CYCLES}s before next cycle..."
    sleep "$SLEEP_BETWEEN_CYCLES" &
    wait $! 2>/dev/null || true  # Allows Ctrl+C to interrupt sleep
done

echo ""
echo "=== Agent loop finished ==="
echo "Total cycles run: $CYCLES_RUN"
echo "Ended at: $(date -Iseconds)"
