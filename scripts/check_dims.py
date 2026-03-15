#!/usr/bin/env python3
"""Check head dimensions for Qwen3 and Llama."""
from transformers import AutoConfig
for model_name in ['Qwen/Qwen3-4B-Base', 'meta-llama/Llama-3.1-8B-Instruct']:
    c = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"\n{model_name}:")
    d = c.to_dict()
    for k, v in sorted(d.items()):
        if 'dim' in k.lower() or 'head' in k.lower() or 'hidden' in k.lower() or 'rope' in k.lower():
            print(f"  {k}: {v}")
