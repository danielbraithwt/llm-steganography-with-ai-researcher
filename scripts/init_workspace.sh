#!/usr/bin/env bash
#
# init_workspace.sh — One-time setup for the autonomous research agent.
#
# Run this once after cloning the repo and setting up config.env.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== Initializing Research Agent Workspace ==="
echo ""

# Check for config.env
if [ ! -f config.env ]; then
    echo "config.env not found — creating from template..."
    cp config.env.example config.env
    echo "Created config.env — please edit it with your Slack token, email, etc."
    echo ""
fi

# Create required directories
echo "Creating directories..."
mkdir -p literature_notes experiment_log results logs

# Install Python dependencies
echo "Installing Python dependencies..."
if command -v pip3 >/dev/null 2>&1; then
    pip3 install -r requirements.txt
elif command -v pip >/dev/null 2>&1; then
    pip install -r requirements.txt
else
    echo "Warning: pip not found — install requirements.txt manually"
fi

# Make scripts executable
echo "Setting permissions..."
chmod +x loop.sh
chmod +x notifications/notify.sh
chmod +x scripts/init_workspace.sh

# Check git setup
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git add -A
    git commit -m "Initial setup: autonomous research agent scaffolding"
    echo ""
    echo "Git repo initialized. Set up a remote with:"
    echo "  git remote add origin <your-repo-url>"
    echo "  git push -u origin main"
else
    echo "Git repository already initialized."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit config.env with your Slack bot token and channel ID"
echo "  2. Edit research_spec.md with your hypothesis and evidence standards"
echo "  3. Replace placeholder templates in templates/ with your experiment code"
echo "  4. Run: ./loop.sh"
echo ""
echo "For Slack setup instructions, see README.md"
