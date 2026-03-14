#!/usr/bin/env python3
"""
Formats cycle status as a Slack Block Kit message.
Reads latest_status.json from stdin or file argument, outputs Slack blocks JSON.
"""
import json
import sys


OUTCOME_EMOJI = {
    "confirmed": "white_check_mark",
    "disconfirmed": "x",
    "inconclusive": "arrows_counterclockwise",
    "literature_scan": "books",
    "consolidation": "file_folder",
    "error": "warning",
}

STRENGTH_BAR = {0: 5 * "\u25a1", 1: 1 * "\u25a0" + 4 * "\u25a1", 2: 2 * "\u25a0" + 3 * "\u25a1",
                3: 3 * "\u25a0" + 2 * "\u25a1", 4: 4 * "\u25a0" + 1 * "\u25a1", 5: 5 * "\u25a0"}


def format_status(status):
    """Convert a status dict to Slack Block Kit blocks."""
    outcome = status.get("prediction_outcome", "inconclusive")
    emoji = OUTCOME_EMOJI.get(outcome, "grey_question")
    cycle = status.get("cycle", "?")
    timestamp = status.get("timestamp", "unknown")
    template = status.get("template", "unknown")
    params = status.get("parameters", {})
    finding = status.get("key_finding", "No finding recorded")
    impact = status.get("evidence_impact", "")
    next_planned = status.get("next_planned", "TBD")
    cycle_type = status.get("type", "experiment")
    progress = status.get("progress", {})

    # Format parameters concisely
    params_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "none"

    # Build progress section
    progress_lines = []
    for claim, info in progress.items():
        bar_val = info.get("bar", 0)
        bar = STRENGTH_BAR.get(bar_val, STRENGTH_BAR[0])
        claim_display = claim.replace("_", " ").title()
        progress_lines.append(f"{bar} {claim_display} ({info.get('status', '?')})")
    progress_text = "\n".join(progress_lines) if progress_lines else "No progress data"

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f":{emoji}: Cycle {cycle} — {cycle_type.title()}",
                "emoji": True
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Template:*\n`{template}`"},
                {"type": "mrkdwn", "text": f"*Params:*\n{params_str}"},
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Key Finding:*\n{finding}"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Evidence Impact:*\n{impact}"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Progress:*\n```\n{progress_text}\n```"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Next:* {next_planned}"
            }
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":speech_balloon: Reply in this channel to send directives to the agent | {timestamp}"
                }
            ]
        },
        {"type": "divider"}
    ]

    return blocks


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            status = json.load(f)
    else:
        status = json.load(sys.stdin)

    blocks = format_status(status)
    print(json.dumps({"blocks": blocks}, indent=2))


if __name__ == "__main__":
    main()
