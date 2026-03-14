#!/usr/bin/env python3
"""
Polls Slack channel for human directives.
Uses the Slack conversations.history API to read recent messages,
filters to human-only messages, and writes them to human_directives.md.

Usage: python slack_poll.py --output human_directives.md
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("Warning: requests not installed, Slack polling disabled", file=sys.stderr)
    sys.exit(0)


LAST_POLL_FILE = os.path.join(os.path.dirname(__file__), ".last_poll_ts")


def parse_args():
    parser = argparse.ArgumentParser(description="Poll Slack for human directives")
    parser.add_argument("--output", default="human_directives.md",
                        help="Output file for directives")
    return parser.parse_args()


def get_last_poll_ts():
    """Read the last poll timestamp."""
    if os.path.exists(LAST_POLL_FILE):
        with open(LAST_POLL_FILE) as f:
            return f.read().strip()
    return "0"


def save_last_poll_ts(ts):
    """Save the current timestamp for next poll."""
    with open(LAST_POLL_FILE, "w") as f:
        f.write(ts)


def fetch_messages(token, channel_id, oldest_ts):
    """Fetch messages from Slack channel newer than oldest_ts."""
    url = "https://slack.com/api/conversations.history"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "channel": channel_id,
        "oldest": oldest_ts,
        "limit": 100,
    }

    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("ok"):
        print(f"Slack API error: {data.get('error', 'unknown')}", file=sys.stderr)
        return []

    return data.get("messages", [])


def filter_human_messages(messages):
    """Filter out bot messages, keeping only human-posted messages."""
    human = []
    for msg in messages:
        # Skip bot messages
        if msg.get("bot_id") or msg.get("subtype"):
            continue
        human.append(msg)
    return human


def format_timestamp(ts):
    """Convert Slack timestamp to readable format."""
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def get_user_name(token, user_id):
    """Look up a user's display name from their ID."""
    url = "https://slack.com/api/users.info"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"user": user_id}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            user = data["user"]
            return (user.get("profile", {}).get("display_name")
                    or user.get("real_name")
                    or user.get("name", user_id))
    except Exception:
        pass
    return user_id


def write_directives(output_path, new_messages, token):
    """Write new directives to human_directives.md, preserving acknowledged ones."""
    # Read existing file to preserve acknowledged directives
    acknowledged = ""
    if os.path.exists(output_path):
        with open(output_path) as f:
            content = f.read()
        # Extract acknowledged section
        if "## Acknowledged Directives" in content:
            acknowledged = content.split("## Acknowledged Directives", 1)[1].strip()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    lines = [
        "# Human Directives",
        "",
        "Messages from the researcher, delivered via Slack or edited directly.",
        "The agent reads this file at the start of each cycle and follows any new directives.",
        "",
        f"## Latest Directives (updated: {now})",
        "",
    ]

    if not new_messages:
        lines.append("(no new directives)")
    else:
        for msg in sorted(new_messages, key=lambda m: float(m["ts"])):
            ts_str = format_timestamp(msg["ts"])
            user_name = get_user_name(token, msg.get("user", "unknown"))
            text = msg.get("text", "(empty message)")
            lines.append(f"### [{ts_str}] \u2014 {user_name}")
            lines.append(text)
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Acknowledged Directives")
    if acknowledged:
        lines.append(acknowledged)
    else:
        lines.append("(the agent moves directives here after acknowledging them)")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN", "")
    channel_id = os.environ.get("SLACK_CHANNEL_ID", "")

    if not token or not channel_id:
        # Slack not configured — exit silently
        sys.exit(0)

    oldest_ts = get_last_poll_ts()

    try:
        messages = fetch_messages(token, channel_id, oldest_ts)
    except Exception as e:
        print(f"Error polling Slack: {e}", file=sys.stderr)
        sys.exit(0)

    human_messages = filter_human_messages(messages)

    if human_messages:
        write_directives(args.output, human_messages, token)
        # Update last poll timestamp to the newest message
        newest_ts = max(float(m["ts"]) for m in human_messages)
        save_last_poll_ts(str(newest_ts))
        print(f"Found {len(human_messages)} new directive(s) from Slack", file=sys.stderr)
    else:
        print("No new directives from Slack", file=sys.stderr)


if __name__ == "__main__":
    main()
