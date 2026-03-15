#!/usr/bin/env python3
"""Smoke test: verify Phi-3.5-mini loads, generates CoT, KV cache perturbation works."""
import torch
import gc
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

print(f"Loading {MODEL_NAME} (no trust_remote_code)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()

num_layers = model.config.num_hidden_layers
n_kv_heads = getattr(model.config, 'num_key_value_heads',
                     getattr(model.config, 'num_attention_heads', '?'))
head_dim = model.config.hidden_size // model.config.num_attention_heads
print(f"Loaded: {num_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

prompt = """Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?
A: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
#### 18

Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: It takes 2/2=<<2/2=1>>1 bolt of white fiber
So the total bolts needed is 2+1=<<2+1=3>>3
#### 3

Q: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?
A:"""

print(f"\nGenerating trace...")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
prompt_len = inputs.input_ids.shape[1]
print(f"Prompt length: {prompt_len} tokens")

generated_ids = []
past_kv = None
current_input = inputs.input_ids

for step in range(200):
    with torch.no_grad():
        if past_kv is not None:
            outputs = model(input_ids=current_input, past_key_values=past_kv, use_cache=True)
        else:
            outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    token_id = next_token[0, 0].item()
    generated_ids.append(token_id)
    if token_id == tokenizer.eos_token_id:
        break
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if "####" in text:
        after = text.split("####")[-1]
        if re.search(r'\d+\s', after):
            break
    if "\nQ:" in text:
        break
    current_input = next_token

text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"\nGenerated ({len(generated_ids)} tokens):")
print(text[:500])

if "####" in text:
    after = text.split("####")[-1].strip()
    m = re.search(r'-?[\d,]+\.?\d*', after)
    ans = m.group(0).replace(',', '') if m else "?"
    print(f"\nExtracted answer: {ans} (expected: 260)")

del past_kv, outputs
gc.collect(); torch.cuda.empty_cache()

print("\nTesting KV cache clone + DynamicCache + perturb...")
with torch.no_grad():
    inputs2 = tokenizer(prompt, return_tensors="pt").to(model.device)
    out2 = model(**inputs2, use_cache=True)
    pc = out2.past_key_values

    new_cache = DynamicCache()
    for l in range(num_layers):
        k = pc.layers[l].keys.clone()
        v = pc.layers[l].values.clone()
        new_cache.update(k, v, l)

    test_tok = torch.tensor([[tokenizer.encode(" If", add_special_tokens=False)[0]]], device=model.device)
    out3 = model(input_ids=test_tok, past_key_values=new_cache, use_cache=True)
    new_cache = out3.past_key_values

    pos = prompt_len
    k_orig = new_cache.layers[0].keys[:, :, pos:pos+1, :].clone()
    noise = torch.randn_like(k_orig)
    norms = k_orig.norm(dim=-1, keepdim=True)
    noise_norms = noise.norm(dim=-1, keepdim=True) + 1e-8
    k_new = noise * (norms / noise_norms)
    new_cache.layers[0].keys[:, :, pos:pos+1, :] = k_new
    print(f"K perturb OK. Orig norm: {k_orig.float().norm():.2f}, New norm: {k_new.float().norm():.2f}")

print(f"\nGPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB used")
print("Smoke test PASSED!")
