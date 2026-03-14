#!/usr/bin/env bash
#
# notify.sh — Send cycle status to Slack and optionally email
#
# Reads experiment_log/latest_status.json and posts formatted status to Slack.
# Every EMAIL_DIGEST_EVERY cycles, generates and sends an email digest.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

STATUS_FILE="${PROJECT_DIR}/experiment_log/latest_status.json"

# Source config if available
if [ -f "${PROJECT_DIR}/config.env" ]; then
    # shellcheck source=/dev/null
    source "${PROJECT_DIR}/config.env"
fi

# ── Slack notification ──────────────────────────────────────────────

send_slack() {
    if [ ! -f "$STATUS_FILE" ]; then
        echo "No status file found at $STATUS_FILE — skipping Slack notification"
        return 0
    fi

    # Format the message using slack_format.py
    local blocks
    blocks=$(python3 "${SCRIPT_DIR}/slack_format.py" "$STATUS_FILE" 2>/dev/null) || {
        echo "Warning: slack_format.py failed"
        return 0
    }

    # Prefer bot token (richer formatting, threading support)
    if [ -n "${SLACK_BOT_TOKEN:-}" ] && [ -n "${SLACK_CHANNEL_ID:-}" ]; then
        local payload
        payload=$(echo "$blocks" | python3 -c "
import json, sys
blocks = json.load(sys.stdin)
payload = {
    'channel': '${SLACK_CHANNEL_ID}',
    'blocks': blocks.get('blocks', []),
    'text': 'Research agent cycle update'
}
print(json.dumps(payload))
")
        local response
        response=$(curl -s -X POST "https://slack.com/api/chat.postMessage" \
            -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "$payload" 2>/dev/null) || true

        if echo "$response" | python3 -c "import sys,json; sys.exit(0 if json.load(sys.stdin).get('ok') else 1)" 2>/dev/null; then
            echo "Slack notification sent (bot token)"
            # Upload any figures from this cycle
            upload_figures_to_slack
        else
            echo "Warning: Slack bot post may have failed"
            # Fall through to webhook
            send_slack_webhook "$blocks"
        fi
        return 0
    fi

    # Fallback: webhook
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        send_slack_webhook "$blocks"
    else
        echo "No Slack credentials configured — skipping notification"
    fi
}

send_slack_webhook() {
    local blocks="$1"
    curl -s -X POST "${SLACK_WEBHOOK_URL}" \
        -H "Content-Type: application/json" \
        -d "$blocks" >/dev/null 2>&1 || echo "Warning: Slack webhook post failed"
    echo "Slack notification sent (webhook)"
}

# ── Figure uploads ──────────────────────────────────────────────────

upload_figures_to_slack() {
    if [ -z "${SLACK_BOT_TOKEN:-}" ] || [ -z "${SLACK_CHANNEL_ID:-}" ]; then
        return 0
    fi
    if [ ! -f "$STATUS_FILE" ]; then
        return 0
    fi

    # Extract figure paths from latest_status.json
    local figures
    figures=$(python3 -c "
import json, sys
with open('$STATUS_FILE') as f:
    status = json.load(f)
for fig in status.get('figures', []):
    print(fig)
" 2>/dev/null) || return 0

    if [ -z "$figures" ]; then
        return 0
    fi

    echo "Uploading figures to Slack..."
    while IFS= read -r fig_path; do
        local full_path="${PROJECT_DIR}/${fig_path}"
        if [ ! -f "$full_path" ]; then
            echo "  Warning: figure not found: $fig_path"
            continue
        fi

        local filename
        filename=$(basename "$fig_path")

        # Use Slack files.upload v2 API
        local upload_response
        upload_response=$(curl -s -X POST "https://slack.com/api/files.uploadV2" \
            -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
            -F "file=@${full_path}" \
            -F "filename=${filename}" \
            -F "channel_id=${SLACK_CHANNEL_ID}" \
            -F "title=${filename}" \
            2>/dev/null) || true

        if echo "$upload_response" | python3 -c "import sys,json; sys.exit(0 if json.load(sys.stdin).get('ok') else 1)" 2>/dev/null; then
            echo "  Uploaded: $filename"
        else
            echo "  Warning: failed to upload $filename"
        fi
    done <<< "$figures"
}

# ── Email digest ────────────────────────────────────────────────────

maybe_send_email() {
    local cycle_num="${1:-0}"
    local digest_every="${EMAIL_DIGEST_EVERY:-5}"

    if [ -z "${NOTIFICATION_EMAIL:-}" ]; then
        return 0
    fi

    # Only send every N cycles
    if [ "$((cycle_num % digest_every))" -ne 0 ]; then
        return 0
    fi

    echo "Generating email digest (cycle $cycle_num)..."
    local digest
    digest=$(python3 "${SCRIPT_DIR}/email_format.py" \
        --last-n-cycles "$digest_every" \
        --experiment-log-dir "${PROJECT_DIR}/experiment_log/" \
        --evidence-ledger "${PROJECT_DIR}/evidence_ledger.md" 2>/dev/null) || {
        echo "Warning: email_format.py failed"
        return 0
    }

    # Save digest for reference
    echo "$digest" > "${SCRIPT_DIR}/digest.md"

    # Try sendmail first, then Python smtplib
    if command -v sendmail >/dev/null 2>&1; then
        {
            echo "To: ${NOTIFICATION_EMAIL}"
            echo "Subject: Research Agent Digest — Cycle $cycle_num"
            echo "Content-Type: text/plain; charset=utf-8"
            echo ""
            echo "$digest"
        } | sendmail -t 2>/dev/null && echo "Email digest sent via sendmail" && return 0
    fi

    # Fallback: Python smtplib (requires SMTP config — log if unavailable)
    echo "Warning: sendmail not available. Digest saved to ${SCRIPT_DIR}/digest.md"
    echo "To send manually, email the digest to ${NOTIFICATION_EMAIL}"
}

# ── Main ────────────────────────────────────────────────────────────

main() {
    local cycle_num="${1:-0}"

    send_slack
    maybe_send_email "$cycle_num"
}

main "$@"
