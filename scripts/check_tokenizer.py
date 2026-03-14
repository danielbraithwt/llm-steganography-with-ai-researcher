#!/usr/bin/env python3
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Base', trust_remote_code=True)
for s in ['####', '###', '##', '#', '#### 52', '\n#### 52\n', '\n#### ']:
    ids = tok.encode(s, add_special_tokens=False)
    print(f'{s!r:25s} -> {str(ids):30s}  decoded: {[tok.decode([i]) for i in ids]}')

# Now test round-trip
text = "Let's think step by step.\n2 + 3 = 5\n#### 5\n"
ids = tok.encode(text, add_special_tokens=False)
decoded = tok.decode(ids, skip_special_tokens=True)
re_ids = tok.encode(decoded, add_special_tokens=False)
print(f"\nRound-trip test:")
print(f"  Original:   {len(ids)} tokens")
print(f"  Re-encoded: {len(re_ids)} tokens")
print(f"  Match: {ids == re_ids}")
if ids != re_ids:
    for i, (a, b) in enumerate(zip(ids, re_ids)):
        if a != b:
            print(f"  First diff at pos {i}: {a} ({tok.decode([a])!r}) vs {b} ({tok.decode([b])!r})")
            break
