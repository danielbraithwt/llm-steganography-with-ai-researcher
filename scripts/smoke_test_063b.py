#!/usr/bin/env python3
"""Detailed smoke test: check precision and actual spectral results."""
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

ds = load_gsm8k()
question = ds[0]['question']
prompt = build_prompt(question)
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
prompt_len = inputs.input_ids.shape[1]

with torch.no_grad():
    output = model.generate(
        **inputs, max_new_tokens=100, do_sample=False,
        return_dict_in_generate=True, use_cache=True)

total_len = output.sequences.shape[1]
reasoning_len = total_len - prompt_len

with torch.no_grad():
    out = model(output.sequences[0:1], use_cache=True)

past_kv = out.past_key_values
from transformers import DynamicCache
if isinstance(past_kv, DynamicCache):
    if hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
        k = past_kv.layers[0].keys
    else:
        k = past_kv.key_cache[0]

k_reasoning = k[0, :, prompt_len:, :].cpu()
print(f'K dtype: {k.dtype}, K shape: {k.shape}')
print(f'K reasoning shape: {k_reasoning.shape}')
print(f'K value range: [{k_reasoning.min():.4f}, {k_reasoning.max():.4f}]')

positions = torch.arange(prompt_len, total_len, device='cpu')
cos, sin = get_rope_cos_sin_from_model(model, positions, device='cuda')
print(f'cos dtype: {cos.dtype}, cos range: [{cos.min():.6f}, {cos.max():.6f}]')

# Test roundtrip in pure float32
k_f32 = k_reasoning.float()
cos_u = cos.unsqueeze(0)
sin_u = sin.unsqueeze(0)

# Inverse
def rotate_half_inv(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

k_pre_f32 = (k_f32 * cos_u) + (rotate_half_inv(k_f32) * sin_u)

# Forward again
k_roundtrip = (k_pre_f32 * cos_u) + (rotate_half(k_pre_f32) * sin_u)

roundtrip_error = (k_roundtrip - k_f32).abs()
print(f'\nFloat32 roundtrip error:')
print(f'  max: {roundtrip_error.max():.2e}')
print(f'  mean: {roundtrip_error.mean():.2e}')
print(f'  median: {roundtrip_error.median():.2e}')

# The error is from float16 K values being approximate
# Let's check: if we compare k_roundtrip to the float16 K
error_vs_f16 = (k_roundtrip - k_reasoning.float()).abs()
print(f'\nRoundtrip vs float16 K:')
print(f'  max: {error_vs_f16.max():.2e}')
print(f'  mean: {error_vs_f16.mean():.2e}')

# But the real question is: the K cache values ARE the post-RoPE values
# The model stored them in float16 after applying RoPE
# So the pre-RoPE K we compute should give us meaningful spectral metrics
# even if the roundtrip has float16 error

# Compare spectral metrics across layers
print('\n=== Per-layer spectral comparison (first 5 layers) ===')
for layer_idx in range(5):
    if isinstance(past_kv, DynamicCache):
        if hasattr(past_kv, 'layers'):
            kl = past_kv.layers[layer_idx].keys
            vl = past_kv.layers[layer_idx].values
        else:
            kl = past_kv.key_cache[layer_idx]
            vl = past_kv.value_cache[layer_idx]

    kr = kl[0, :, prompt_len:, :].cpu()
    vr = vl[0, :, prompt_len:, :].cpu()

    k_pre = inverse_rope(kr, cos, sin)

    k_post_layer = kr.permute(1, 0, 2).reshape(reasoning_len, -1)
    k_pre_layer = k_pre.permute(1, 0, 2).reshape(reasoning_len, -1)
    v_layer = vr.permute(1, 0, 2).reshape(reasoning_len, -1)

    k_post_m = compute_spectral_metrics(k_post_layer)
    k_pre_m = compute_spectral_metrics(k_pre_layer)
    v_m = compute_spectral_metrics(v_layer)

    print(f'Layer {layer_idx}: K_post top1={k_post_m["top1_energy"]:.4f}, '
          f'K_pre top1={k_pre_m["top1_energy"]:.4f}, V top1={v_m["top1_energy"]:.4f} | '
          f'K_post rank={k_post_m["effective_rank"]:.1f}, '
          f'K_pre rank={k_pre_m["effective_rank"]:.1f}, V rank={v_m["effective_rank"]:.1f}')

# The key question is: does K_pre look more like V or like K_post?
print('\nVERDICT: If K_pre top1 >> V top1, then K spectral dominance is intrinsic (NOT RoPE artifact)')
print('If K_pre top1 ~ V top1, then K spectral dominance IS a RoPE artifact')
