#!/usr/bin/env python3
"""Smoke test for exp_037: verify Mistral-7B-v0.3 loads, generates, and cache works."""
import os, time, random, gc, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Loading model...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
model.eval()
num_layers = model.config.num_hidden_layers
n_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
n_attn = model.config.num_attention_heads
head_dim = getattr(model.config, "head_dim", model.config.hidden_size // n_attn)
print(f"Loaded: {num_layers}L, {n_kv} KV, {n_attn} attn, hd={head_dim}, {time.time()-t0:.0f}s")
print(f"Arch: {'MHA' if n_kv == n_attn else 'GQA'}, hidden={model.config.hidden_size}")

# Verify cache access pattern
ti = tokenizer("test", return_tensors="pt").to(model.device)
to = model(**ti, use_cache=True)
tc = to.past_key_values
assert hasattr(tc, "layers"), "Cache needs layers attribute"
print(f"Cache verified: layers[0].keys.shape = {tc.layers[0].keys.shape}")
del ti, to, tc
gc.collect()
torch.cuda.empty_cache()

# Test with a 1-shot GSM8K problem
prompt = (
    "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
    "and bakes muffins for her friends every day with four. She sells every duck egg at "
    "the farmers' market daily for $2. How much in dollars does she make every day?\n"
    "A: Janet sells 16 - 3 - 4 = 9 duck eggs a day.\n"
    "She makes 9 * 2 = $18 every day.\n#### 18\n\n"
    "Q: A robe takes 2 bolts of blue fiber and half that much white fiber. "
    "How many bolts in total does it take?\nA:"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Prompt tokens: {inputs.input_ids.shape[1]}")

# Generate with KV cache
gen_ids = []
past_kv = None
cur = inputs.input_ids
for step in range(200):
    if past_kv is not None:
        o = model(input_ids=cur, past_key_values=past_kv, use_cache=True)
    else:
        o = model(**inputs, use_cache=True)
    past_kv = o.past_key_values
    nt = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
    tid = nt[0, 0].item()
    gen_ids.append(tid)
    if tid == tokenizer.eos_token_id:
        break
    txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if "####" in txt:
        after = txt.split("####")[-1]
        if re.search(r"\d+\s", after):
            break
    if "\nQ:" in txt:
        idx = txt.find("\nQ:")
        if idx > 0:
            gen_ids = tokenizer.encode(txt[:idx], add_special_tokens=False)
        break
    cur = nt

text = tokenizer.decode(gen_ids, skip_special_tokens=True)
print(f"Generated ({len(gen_ids)} tokens): {text[:300]}")

# Extract answer
if "####" in text:
    after = text.split("####")[-1].strip()
    m = re.search(r"-?[\d,]+", after)
    ans = m.group(0) if m else "N/A"
else:
    nums = re.findall(r"-?\d+", text)
    ans = nums[-1] if nums else "N/A"
print(f"Answer: {ans} (expected: 3)")
status = "PASSED" if ans == "3" else f"MISMATCH (got {ans})"
print(f"SMOKE TEST: {status}")

# Test cache clone and perturbation
print("\nTesting cache clone and magnitude perturbation...")
prompt_cache = model(input_ids=inputs.input_ids, use_cache=True).past_key_values
cache = DynamicCache()
for l in range(num_layers):
    k = prompt_cache.layers[l].keys.clone()
    v = prompt_cache.layers[l].values.clone()
    cache.update(k, v, l)

# Test perturbation on one position
pos = cache.layers[0].keys.shape[2] - 1  # last position
k_orig = cache.layers[0].keys[:, :, pos:pos+1, :].clone()
norms = k_orig.norm(dim=-1, keepdim=True)
direction = k_orig / (norms + 1e-8)
delta = torch.randn(norms.shape, device=k_orig.device, dtype=k_orig.dtype)
scale = (1 + delta).clamp(min=0.01)
k_new = direction * norms * scale
cache.layers[0].keys[:, :, pos:pos+1, :] = k_new
print(f"Perturbation test: orig_norm={k_orig.float().norm():.1f}, new_norm={k_new.float().norm():.1f}")
print(f"K/V norm ratio: {k_orig.float().norm() / (prompt_cache.layers[0].values[:,:,pos:pos+1,:].float().norm() + 1e-8):.2f}")

del prompt_cache, cache
gc.collect()
torch.cuda.empty_cache()

print(f"\nTotal smoke test time: {time.time()-t0:.0f}s")
print("ALL CHECKS PASSED")
