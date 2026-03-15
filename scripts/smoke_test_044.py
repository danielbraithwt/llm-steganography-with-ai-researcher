#!/usr/bin/env python3
"""Quick smoke test for exp 044 - verify model, cache, and one problem."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import gc

MODEL_NAME = 'Qwen/Qwen3-4B-Base'
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map='auto',
    attn_implementation='eager')
model.eval()

# Verify cache
test_input = tokenizer('test', return_tensors='pt').to(model.device)
test_out = model(**test_input, use_cache=True)
test_cache = test_out.past_key_values
assert hasattr(test_cache, 'layers'), 'No layers attribute'
k_shape = test_cache.layers[0].keys.shape
print(f'Cache verified: layers[0].keys.shape = {k_shape}')
print(f'num_layers = {model.config.num_hidden_layers}')
print(f'num_kv_heads = {model.config.num_key_value_heads}')
print(f'head_dim = {model.config.hidden_size // model.config.num_attention_heads}')
del test_input, test_out, test_cache
gc.collect(); torch.cuda.empty_cache()

# Quick generation test
prompt = "Q: What is 2 + 3?\nA: 2 + 3 = 5\n#### 5\n\nQ: What is 4 + 7?\nA:"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_new_tokens=50, do_sample=False)
text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f'Generated: {text[:100]}')

print('\nSMOKE TEST PASSED')
