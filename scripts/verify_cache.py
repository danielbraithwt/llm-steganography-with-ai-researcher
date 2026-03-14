#!/usr/bin/env python3
"""Verify cache access patterns work with the installed transformers version."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Base", torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager")
model.eval()

inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
out = model(**inputs, use_cache=True)
cache = out.past_key_values
print("Cache type:", type(cache))
print("Has key_cache:", hasattr(cache, "key_cache"))
print("Has value_cache:", hasattr(cache, "value_cache"))

if hasattr(cache, "key_cache"):
    print("key_cache[0] shape:", cache.key_cache[0].shape)
    print("value_cache[0] shape:", cache.value_cache[0].shape)

    orig = cache.key_cache[0][:, :, 0:1, :].clone()
    cache.key_cache[0][:, :, 0:1, :] = torch.randn_like(orig)
    modified = cache.key_cache[0][:, :, 0:1, :]
    print("In-place modification works:", not torch.allclose(orig, modified))

    dc = DynamicCache()
    dc.update(cache.key_cache[0].clone(), cache.value_cache[0].clone(), 0)
    print("DynamicCache update works, key shape:", dc.key_cache[0].shape)
    print("All checks passed!")
else:
    print("ERROR: No key_cache attribute. Checking for legacy access...")
    print("dir(cache):", [a for a in dir(cache) if not a.startswith("_")])
