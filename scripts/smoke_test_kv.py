#!/usr/bin/env python3
"""Quick smoke test for DynamicCache manipulation with Llama."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
print("Loading model with attn_implementation=eager...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

text = "Q: What is 2+2?\nA: 2+2 = 4.\n#### 4"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(f"Input shape: {inputs.input_ids.shape}")

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    print(f"KV cache type: {type(past_kv).__name__}")
    print(f"Has key_cache: {hasattr(past_kv, 'key_cache')}")
    print(f"Num layers: {len(past_kv.key_cache)}")
    print(f"Key shape (L0): {past_kv.key_cache[0].shape}")

    # Test tuple-of-tuples truncation
    num_layers = len(past_kv.key_cache)
    trunc_len = 5
    trunc_kv = tuple(
        (past_kv.key_cache[i][:, :, :trunc_len, :].clone(),
         past_kv.value_cache[i][:, :, :trunc_len, :].clone())
        for i in range(num_layers)
    )
    remaining = inputs.input_ids[:, trunc_len:]
    out2 = model(input_ids=remaining, past_key_values=trunc_kv, use_cache=True)
    print(f"Tuple-of-tuples pass: OK (logits shape: {out2.logits.shape})")
    print(f"Output KV type: {type(out2.past_key_values).__name__}")

    # Test in-place noise + restore
    clean_k = past_kv.key_cache[0].clone()
    past_kv.key_cache[0][:, :, 5:, :] += torch.randn_like(past_kv.key_cache[0][:, :, 5:, :]) * 0.1
    past_kv.key_cache[0].copy_(clean_k)
    print("In-place noise + restore: OK")

print("ALL SMOKE TESTS PASSED")
