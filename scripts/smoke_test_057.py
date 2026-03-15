#!/usr/bin/env python3
"""Smoke test for Exp 057 — verify true H2O computation works."""

import os, sys, json, time, random, gc, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

EXEMPLARS = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
     "a": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "a": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total bolts needed is 2+1=<<2+1=3>>3\n#### 3"},
]

def build_prompt(q):
    p = ""
    for ex in EXEMPLARS:
        p += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
    p += f"Q: {q}\nA:"
    return p

model_name = "Qwen/Qwen3-4B-Base"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

question = "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"
prompt = build_prompt(question)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
prompt_len = inputs.input_ids.shape[1]
print(f"Prompt length: {prompt_len} tokens")

with torch.no_grad():
    gen_out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             use_cache=True, return_dict_in_generate=True)
full_ids = gen_out.sequences[0]
gen_ids = full_ids[prompt_len:]
gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
print(f"Generated {len(gen_ids)} tokens")
del gen_out; gc.collect(); torch.cuda.empty_cache()

# Find truncation
text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
if "####" in text:
    prefix = text[:text.index("####")]
    trunc_point = len(tokenizer.encode(prefix, add_special_tokens=False))
    print(f"Truncation point: {trunc_point}")
else:
    trunc_point = len(gen_ids) - 5
    print(f"No #### found, using {trunc_point}")

full_seq = torch.cat([inputs.input_ids[0], gen_ids[:trunc_point]])
total_len = full_seq.shape[0]
reasoning_len = trunc_point
print(f"Total seq len: {total_len}, reasoning_len: {reasoning_len}")

# Build KV cache
with torch.no_grad():
    outputs = model(input_ids=full_seq.unsqueeze(0), use_cache=True)
full_cache = outputs.past_key_values
del outputs

# K-norm importance
num_layers = model.config.num_hidden_layers
k_norm_imp = np.zeros(total_len)
for l in range(num_layers):
    k = full_cache.layers[l].keys[0]
    k_norms = k.float().norm(dim=-1).cpu().numpy()
    k_norm_imp += k_norms.mean(axis=0)

print(f"K-norm importance shape: {k_norm_imp.shape}")
print(f"K-norm range: [{k_norm_imp[prompt_len:total_len].min():.2f}, "
      f"{k_norm_imp[prompt_len:total_len].max():.2f}]")

# True H2O
print("Computing true H2O importance...")
t0 = time.time()
with torch.no_grad():
    outputs2 = model(input_ids=full_seq.unsqueeze(0),
                     output_attentions=True, use_cache=False)

cumulative = np.zeros(reasoning_len)
for layer_idx in range(num_layers):
    attn = outputs2.attentions[layer_idx]
    reasoning_attn = attn[0, :, prompt_len:total_len, prompt_len:total_len]
    pos_importance = reasoning_attn.float().sum(dim=(0, 1)).cpu().numpy()
    cumulative += pos_importance

del outputs2; gc.collect(); torch.cuda.empty_cache()
t1 = time.time()

print(f"True H2O computed in {t1-t0:.1f}s")
print(f"True H2O shape: {cumulative.shape}")
print(f"True H2O range: [{cumulative.min():.2f}, {cumulative.max():.2f}]")

# Correlation
from scipy import stats
k_reason = k_norm_imp[prompt_len:total_len]
rho, p = stats.spearmanr(k_reason, cumulative)
print(f"Spearman rho(K-norm, True H2O): {rho:.4f} (p={p:.4e})")

# Jaccard overlap at 33%
n_keep = max(1, int(0.33 * reasoning_len))
k_sel = set(np.argsort(k_reason)[-n_keep:].tolist())
t_sel = set(np.argsort(cumulative)[-n_keep:].tolist())
overlap = len(k_sel & t_sel)
jaccard = overlap / len(k_sel | t_sel)
print(f"Jaccard overlap at 33%: {jaccard:.3f} ({overlap}/{len(k_sel | t_sel)})")

# Check GPU memory after
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

del full_cache, model
gc.collect(); torch.cuda.empty_cache()

print("\nSMOKE TEST PASSED")
