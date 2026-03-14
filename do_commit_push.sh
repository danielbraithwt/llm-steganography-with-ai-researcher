#!/bin/bash
set -e

# Commit
git commit -m "cycle 025: magnitude dose-response on Llama — immunities dose-specific, KV superadditive

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

echo "=== COMMIT DONE ==="

# Push with retry
MAX_RETRIES=3
for i in $(seq 1 $MAX_RETRIES); do
    echo "Push attempt $i..."
    if git push origin main; then
        echo "=== PUSH DONE ==="
        exit 0
    fi
    if [ $i -lt $MAX_RETRIES ]; then
        echo "Push failed, retrying in 10 seconds..."
        sleep 10
    fi
done

echo "=== PUSH FAILED after $MAX_RETRIES attempts ==="
exit 1
