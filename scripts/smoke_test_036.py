#!/usr/bin/env python3
"""Quick smoke test for Qwen3-4B (instruct) model loading and KV cache access."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Qwen/Qwen3-4B...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()

print(f"Layers: {model.config.num_hidden_layers}")
print(f"KV heads: {getattr(model.config, 'num_key_value_heads', '?')}")
print(f"Attn heads: {model.config.num_attention_heads}")
print(f"Hidden: {model.config.hidden_size}")
head_dim = getattr(model.config, 'head_dim',
                   model.config.hidden_size // model.config.num_attention_heads)
print(f"Head dim: {head_dim}")

# Quick generation test
inputs = tokenizer("Q: What is 2+3?\nA: Let's think step by step.", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)[:300]}")

# Cache test
test_input = tokenizer("test", return_tensors="pt").to(model.device)
test_out = model(**test_input, use_cache=True)
test_cache = test_out.past_key_values
print(f"Cache type: {type(test_cache).__name__}")
print(f"Has layers: {hasattr(test_cache, 'layers')}")
k_shape = test_cache.layers[0].keys.shape
print(f"K shape: {k_shape}")

# Check architecture matches Base
print(f"\nArchitecture check (should match Qwen3-4B-Base):")
print(f"  36 layers? {model.config.num_hidden_layers == 36}")
print(f"  8 KV heads? {getattr(model.config, 'num_key_value_heads', None) == 8}")
print(f"  head_dim=128? {head_dim == 128}")

print("\nSMOKE TEST PASSED")
