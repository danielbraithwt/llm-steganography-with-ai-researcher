#!/usr/bin/env python3
"""Smoke test for exp_063 RoPE confound test."""
import sys
sys.path.insert(0, 'scripts')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from exp_063_rope_confound import (
    get_rope_params, get_rope_cos_sin_from_model,
    inverse_rope, verify_inverse_rope, build_prompt, extract_answer,
    compute_spectral_metrics, load_gsm8k
)
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print('Loading model...')
model_name = 'Qwen/Qwen3-4B-Base'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
model.eval()

rope_theta, head_dim, rope_type = get_rope_params(model)
print(f'RoPE theta: {rope_theta}, head_dim: {head_dim}, rope_type: {rope_type}')

ds = load_gsm8k()
question = ds[0]['question']
prompt = build_prompt(question)
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
prompt_len = inputs.input_ids.shape[1]
print(f'Prompt len: {prompt_len}')

with torch.no_grad():
    output = model.generate(
        **inputs, max_new_tokens=100, do_sample=False,
        return_dict_in_generate=True, use_cache=True)

total_len = output.sequences.shape[1]
reasoning_len = total_len - prompt_len
print(f'Generated {reasoning_len} tokens')

with torch.no_grad():
    out = model(output.sequences[0:1], use_cache=True)

past_kv = out.past_key_values
from transformers import DynamicCache
if isinstance(past_kv, DynamicCache):
    if hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
        k = past_kv.layers[0].keys
        v = past_kv.layers[0].values
    else:
        k = past_kv.key_cache[0]
        v = past_kv.value_cache[0]
else:
    k = past_kv[0][0]
    v = past_kv[0][1]

print(f'K shape: {k.shape}')
k_reasoning = k[0, :, prompt_len:, :].cpu()
v_reasoning = v[0, :, prompt_len:, :].cpu()
print(f'K reasoning shape: {k_reasoning.shape}')

positions = torch.arange(prompt_len, total_len, device='cpu')
cos, sin = get_rope_cos_sin_from_model(model, positions, device='cuda')
print(f'cos shape: {cos.shape}, sin shape: {sin.shape}')

k_pre = inverse_rope(k_reasoning, cos, sin)
print(f'K pre-RoPE shape: {k_pre.shape}')

err = verify_inverse_rope(k_reasoning, cos, sin, k_pre)
print(f'Inverse verification max error: {err:.2e}')
if err < 1e-2:
    print('PASS: Inverse RoPE is correct')
else:
    print(f'FAIL: Inverse error too large: {err}')

k_post_layer = k_reasoning.permute(1, 0, 2).reshape(reasoning_len, -1)
k_pre_layer = k_pre.permute(1, 0, 2).reshape(reasoning_len, -1)
v_layer = v_reasoning.permute(1, 0, 2).reshape(reasoning_len, -1)

k_post_m = compute_spectral_metrics(k_post_layer)
k_pre_m = compute_spectral_metrics(k_pre_layer)
v_m = compute_spectral_metrics(v_layer)

print(f'\nLayer 0 spectral metrics:')
print(f'  K post-RoPE: eff_rank={k_post_m["effective_rank"]:.1f}, top1={k_post_m["top1_energy"]:.4f}, gap={k_post_m["spectral_gap"]:.2f}')
print(f'  K pre-RoPE:  eff_rank={k_pre_m["effective_rank"]:.1f}, top1={k_pre_m["top1_energy"]:.4f}, gap={k_pre_m["spectral_gap"]:.2f}')
print(f'  V:           eff_rank={v_m["effective_rank"]:.1f}, top1={v_m["top1_energy"]:.4f}, gap={v_m["spectral_gap"]:.2f}')
print(f'\nK_post/V top1 ratio: {k_post_m["top1_energy"]/v_m["top1_energy"]:.2f}x')
print(f'K_pre/V top1 ratio: {k_pre_m["top1_energy"]/v_m["top1_energy"]:.2f}x')
print('\nSmoke test PASSED!')
