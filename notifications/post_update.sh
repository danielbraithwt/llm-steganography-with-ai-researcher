#!/usr/bin/env bash
#
# post_update.sh — Post a short progress message to Slack mid-cycle.
#
# Usage:
#   bash notifications/post_update.sh "Starting experiment 003: KV noise sweep"
#
set -euo pipefail

MESSAGE="${1:-}"
if [ -z "$MESSAGE" ]; then
    echo "Usage: post_update.sh <message>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Source config
if [ -f "${PROJECT_DIR}/config.env" ]; then
    # shellcheck source=/dev/null
    source "${PROJECT_DIR}/config.env"
fi

# Need bot token and channel
if [ -z "${SLACK_BOT_TOKEN:-}" ] || [ -z "${SLACK_CHANNEL_ID:-}" ]; then
    echo "[post_update] No Slack credentials — printing locally: $MESSAGE"
    exit 0
fi

TIMESTAMP=$(date -Iseconds)

payload=$(cat <<PYEOF | python3 -c "
import json, sys
msg = sys.stdin.read().strip()
payload = {
    'channel': '${SLACK_CHANNEL_ID}',
    'text': msg,
    'blocks': [
        {
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': msg
            }
        }
    ]
}
print(json.dumps(payload))
"
:robot_face: *Agent Update* (${TIMESTAMP})
${MESSAGE}
PYEOF
)

response=$(curl -s -X POST "https://slack.com/api/chat.postMessage" \
    -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>/dev/null) || true

if echo "$response" | python3 -c "import sys,json; sys.exit(0 if json.load(sys.stdin).get('ok') else 1)" 2>/dev/null; then
    echo "[post_update] Sent: $MESSAGE"
else
    echo "[post_update] Warning: Slack post may have failed"
fi
