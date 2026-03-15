#!/usr/bin/env python3
"""Check RoPE structure in both models."""
import torch
from transformers import AutoModelForCausalLM

for model_name in ['Qwen/Qwen3-4B-Base', 'meta-llama/Llama-3.1-8B-Instruct']:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)

    attn = model.model.layers[0].self_attn
    print(f'Attention class: {type(attn).__name__}')
    print('Rotary-related attributes:')
    for attr in dir(attn):
        if 'rot' in attr.lower() or 'rope' in attr.lower():
            obj = getattr(attn, attr)
            print(f'  {attr}: {type(obj).__name__}')

    # Check model-level rotary_emb
    if hasattr(model.model, 'rotary_emb'):
        re = model.model.rotary_emb
        print(f'model.model.rotary_emb: {type(re).__name__}')
        # Try to call it
        positions = torch.arange(10, device='cuda')
        dummy = torch.zeros(1, 1, 128, device='cuda', dtype=torch.float16)
        try:
            result = re(dummy, positions.unsqueeze(0))
            print(f'  rotary_emb(dummy, pos) returns: {type(result)}')
            if isinstance(result, tuple):
                print(f'  cos shape: {result[0].shape}, sin shape: {result[1].shape}')
        except Exception as e:
            print(f'  Error calling rotary_emb: {e}')

    # Check attention-level rotary
    if hasattr(attn, 'rotary_fn'):
        print(f'attn.rotary_fn: {type(attn.rotary_fn).__name__}')
    if hasattr(attn, 'rotary_emb'):
        re = attn.rotary_emb
        print(f'attn.rotary_emb: {type(re).__name__}')
        positions = torch.arange(10, device='cuda')
        dummy = torch.zeros(1, 1, 128, device='cuda', dtype=torch.float16)
        try:
            result = re(dummy, positions.unsqueeze(0))
            print(f'  rotary_emb(dummy, pos) returns: {type(result)}')
            if isinstance(result, tuple):
                print(f'  cos shape: {result[0].shape}, sin shape: {result[1].shape}')
        except Exception as e:
            print(f'  Error calling rotary_emb: {e}')

    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
