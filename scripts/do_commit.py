#!/usr/bin/env python3
"""Helper to commit and push cycle results."""
import subprocess
import os

os.chdir("/workspace/llm-steganography-with-ai-researcher")

# Commit
r = subprocess.run(
    ["git", "commit", "-F", ".commit_msg.txt"],
    capture_output=True, text=True
)
print("COMMIT stdout:", r.stdout)
print("COMMIT stderr:", r.stderr)
print("COMMIT rc:", r.returncode)

if r.returncode == 0:
    # Push
    r2 = subprocess.run(
        ["git", "push", "origin", "main"],
        capture_output=True, text=True
    )
    print("PUSH stdout:", r2.stdout)
    print("PUSH stderr:", r2.stderr)
    print("PUSH rc:", r2.returncode)

# Clean up
if os.path.exists(".commit_msg.txt"):
    os.remove(".commit_msg.txt")
