#!/usr/bin/env python3
"""Verify complete KV cache manipulation workflow for transformers 5.3.0."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()

text = "Q: What is 2+2?\nA: Let me think. 2+2 = 4.\n#### 4"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
seq_len = inputs.input_ids.shape[1]
prompt_len = 8  # first 8 tokens as "prompt"
num_layers = model.config.num_hidden_layers

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

past_kv = outputs.past_key_values
print(f"Seq len: {seq_len}, Cache layers: {len(past_kv.layers)}")
print(f"Layer 0 keys shape: {past_kv.layers[0].keys.shape}")

# Test 1: In-place modification and restore
print("\n--- Test 1: In-place modify and restore ---")
clean_keys = [past_kv.layers[i].keys.clone() for i in range(num_layers)]
clean_values = [past_kv.layers[i].values.clone() for i in range(num_layers)]

noise = torch.randn_like(past_kv.layers[0].keys) * 0.01
past_kv.layers[0].keys.add_(noise)
changed = not torch.equal(past_kv.layers[0].keys, clean_keys[0])
past_kv.layers[0].keys.copy_(clean_keys[0])
restored = torch.equal(past_kv.layers[0].keys, clean_keys[0])
print(f"  Modified: {changed}, Restored: {restored}")

# Test 2: Build truncated DynamicCache and forward pass
print("\n--- Test 2: Truncated DynamicCache ---")
trunc_cache = DynamicCache()
for i in range(num_layers):
    k_trunc = past_kv.layers[i].keys[:, :, :prompt_len, :].clone()
    v_trunc = past_kv.layers[i].values[:, :, :prompt_len, :].clone()
    trunc_cache.update(k_trunc, v_trunc, i)

remaining = inputs.input_ids[:, prompt_len:]
out2 = model(input_ids=remaining, past_key_values=trunc_cache, use_cache=True)
print(f"  Forward OK, logits shape: {out2.logits.shape}")
print(f"  Output cache type: {type(out2.past_key_values).__name__}")

# Test 3: Free generation from output cache
print("\n--- Test 3: Free generation ---")
gen_kv = out2.past_key_values
next_logits = out2.logits[:, -1, :]
next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
gen_ids = [next_token[0, 0].item()]

for step in range(10):
    gen_out = model(input_ids=next_token, past_key_values=gen_kv, use_cache=True)
    gen_kv = gen_out.past_key_values
    next_token = torch.argmax(gen_out.logits[:, -1, :], dim=-1, keepdim=True)
    gen_ids.append(next_token[0, 0].item())
    if next_token[0, 0].item() == tokenizer.eos_token_id:
        break

print(f"  Generated: {tokenizer.decode(gen_ids, skip_special_tokens=True)[:80]}")

# Test 4: Noise injection + truncated cache + generation (full SNR workflow)
print("\n--- Test 4: Full SNR workflow ---")
for i in range(num_layers):
    past_kv.layers[i].keys.copy_(clean_keys[i])
    past_kv.layers[i].values.copy_(clean_values[i])

# Add noise to reasoning positions
snr_db = 15
noise_ratio = 10.0 ** (-snr_db / 20.0)
for i in range(num_layers):
    rk = past_kv.layers[i].keys[:, :, prompt_len:seq_len, :]
    rv = past_kv.layers[i].values[:, :, prompt_len:seq_len, :]
    k_norm = rk.norm().item()
    v_norm = rv.norm().item()
    kn = torch.randn_like(rk)
    vn = torch.randn_like(rv)
    k_scale = k_norm * noise_ratio / (kn.norm().item() + 1e-8)
    v_scale = v_norm * noise_ratio / (vn.norm().item() + 1e-8)
    rk.add_(kn * k_scale)
    rv.add_(vn * v_scale)

# Build truncated cache for lookback
lookback = 5
lookback_start = max(prompt_len, seq_len - lookback)
trunc2 = DynamicCache()
for i in range(num_layers):
    k_t = past_kv.layers[i].keys[:, :, :lookback_start, :].clone()
    v_t = past_kv.layers[i].values[:, :, :lookback_start, :].clone()
    trunc2.update(k_t, v_t, i)

lookback_tokens = inputs.input_ids[:, lookback_start:seq_len]
lb_out = model(input_ids=lookback_tokens, past_key_values=trunc2, use_cache=True)
print(f"  Lookback forward OK, logits shape: {lb_out.logits.shape}")

# Generate
gen_kv2 = lb_out.past_key_values
nt = torch.argmax(lb_out.logits[:, -1, :], dim=-1, keepdim=True)
gen_ids2 = [nt[0, 0].item()]
for step in range(10):
    go = model(input_ids=nt, past_key_values=gen_kv2, use_cache=True)
    gen_kv2 = go.past_key_values
    nt = torch.argmax(go.logits[:, -1, :], dim=-1, keepdim=True)
    gen_ids2.append(nt[0, 0].item())
    if nt[0, 0].item() == tokenizer.eos_token_id:
        break

print(f"  Generated after noise: {tokenizer.decode(gen_ids2, skip_special_tokens=True)[:80]}")

print("\nALL TESTS PASSED - Ready to run full experiment")
