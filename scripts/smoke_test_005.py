#!/usr/bin/env python3
"""Smoke test for exp_005: verify Llama loads, generates, and KV cache is accessible."""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Layers: {model.config.num_hidden_layers}")
print(f"Heads: {model.config.num_attention_heads}")
print(f"KV Heads: {getattr(model.config, 'num_key_value_heads', 'N/A')}")

# Test 1: Simple generation
prompt = "Q: What is 2 + 3?\nA: Let's think step by step. 2 + 3 ="
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"\nTest 1: Generation")
print(f"  Input tokens: {inputs.input_ids.shape[1]}")

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    print(f"  KV cache type: {type(past_kv)}")
    print(f"  Num layers in KV: {len(past_kv)}")

    # Check KV access patterns
    if hasattr(past_kv, 'key_cache'):
        print(f"  Access via key_cache: shape={past_kv.key_cache[0].shape}")
        print(f"  Access via value_cache: shape={past_kv.value_cache[0].shape}")
        kv_access = "key_cache"
    elif hasattr(past_kv, 'layers'):
        layer = past_kv.layers[0]
        print(f"  Access via layers: keys shape={layer.keys.shape}")
        kv_access = "layers"
    else:
        print(f"  Access via tuple: shape={past_kv[0][0].shape}")
        kv_access = "tuple"

    print(f"  KV access method: {kv_access}")

del outputs, past_kv
gc.collect()
torch.cuda.empty_cache()

# Test 2: Attention extraction
print(f"\nTest 2: Attention extraction")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True, use_cache=False)
    print(f"  Num attention layers: {len(outputs.attentions)}")
    print(f"  Attention shape (layer 0): {outputs.attentions[0].shape}")
    print(f"  Attention shape (last layer): {outputs.attentions[-1].shape}")

del outputs
gc.collect()
torch.cuda.empty_cache()

# Test 3: KV cache ablation
print(f"\nTest 3: KV cache ablation")
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    num_layers = len(past_kv)

    # Try ablating position 5
    pos = 5
    for layer_idx in range(num_layers):
        if hasattr(past_kv, 'key_cache'):
            keys = past_kv.key_cache[layer_idx]
            values = past_kv.value_cache[layer_idx]
        elif hasattr(past_kv, 'layers'):
            layer = past_kv.layers[layer_idx]
            keys = layer.keys
            values = layer.values
        else:
            keys = past_kv[layer_idx][0]
            values = past_kv[layer_idx][1]

        k_norm = keys[:, :, pos, :].norm().item()
        k_shape = keys[:, :, pos, :].shape
        k_noise = torch.randn(k_shape, device=keys.device, dtype=keys.dtype)
        k_noise = k_noise * (k_norm / (k_noise.norm().item() + 1e-8))
        keys[:, :, pos, :] = k_noise

        if layer_idx == 0:
            print(f"  Layer 0 key shape at pos {pos}: {k_shape}")
            print(f"  Noise norm matches original: {abs(k_noise.norm().item() - k_norm) < 0.01}")

    # Try building truncated cache
    trunc_kv = DynamicCache()
    for layer_idx in range(num_layers):
        if hasattr(past_kv, 'key_cache'):
            key = past_kv.key_cache[layer_idx][:, :, :pos, :].clone()
            value = past_kv.value_cache[layer_idx][:, :, :pos, :].clone()
        elif hasattr(past_kv, 'layers'):
            layer = past_kv.layers[layer_idx]
            key = layer.keys[:, :, :pos, :].clone()
            value = layer.values[:, :, :pos, :].clone()
        else:
            key = past_kv[layer_idx][0][:, :, :pos, :].clone()
            value = past_kv[layer_idx][1][:, :, :pos, :].clone()
        trunc_kv.update(key, value, layer_idx)

    # Forward through truncated cache
    test_tokens = inputs.input_ids[:, pos:pos+3]
    test_out = model(input_ids=test_tokens, past_key_values=trunc_kv, use_cache=True)
    print(f"  Truncated cache forward pass: logits shape={test_out.logits.shape}")

del outputs, past_kv, trunc_kv, test_out
gc.collect()
torch.cuda.empty_cache()

# Test 4: GSM8K generation
print(f"\nTest 4: GSM8K-style generation")
gsm_prompt = """Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?
A: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
#### 18

Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A:"""

gsm_inputs = tokenizer(gsm_prompt, return_tensors="pt").to(model.device)
print(f"  GSM prompt tokens: {gsm_inputs.input_ids.shape[1]}")

with torch.no_grad():
    gen_ids = model.generate(
        gsm_inputs.input_ids,
        max_new_tokens=100,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    gen_text = tokenizer.decode(gen_ids[0][gsm_inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Generated: {gen_text[:200]}")
    if "####" in gen_text:
        ans = gen_text.split("####")[-1].strip().split()[0] if gen_text.split("####")[-1].strip() else ""
        print(f"  Answer: {ans}")

print(f"\nGPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak")
print("\nAll smoke tests passed!")
