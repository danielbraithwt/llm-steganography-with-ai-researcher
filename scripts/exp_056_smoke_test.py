#!/usr/bin/env python3
"""Quick smoke test: 2 problems on Qwen only, 3 key conditions."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from exp_056_mask_vs_zero_eviction import *

import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
problems = list(ds)
random.shuffle(problems)

model_name = "Qwen/Qwen3-4B-Base"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

num_layers = model.config.num_hidden_layers
n_kv_heads = model.config.num_key_value_heads
answer_heads = (0, 5)

print(f"Layers: {num_layers}, KV heads: {n_kv_heads}")

n_tested = 0
for prob_idx, problem in enumerate(problems):
    if n_tested >= 2:
        break

    question = problem["question"]
    true_answer = normalize_answer(problem["answer"].split("####")[-1].strip())
    prompt = build_prompt(question)

    (gen_text, input_ids, gen_ids, importance, per_head_importance,
     prompt_len, full_cache, full_seq, trunc_point) = generate_and_build_cache(
        model, tokenizer, prompt)

    clean_answer = normalize_answer(extract_answer(gen_text))
    if clean_answer != true_answer or full_cache is None or trunc_point is None:
        print(f"  Problem {prob_idx}: skipped (clean={clean_answer}, true={true_answer})")
        if full_cache is not None:
            del full_cache
        continue

    reasoning_len = trunc_point
    n_tested += 1
    print(f"\nProblem {prob_idx}: true={true_answer}, reasoning_len={reasoning_len}")

    # Test baseline
    baseline_text = evict_and_generate(
        model, tokenizer, full_cache, prompt_len, reasoning_len,
        [], 'zero_kv', num_layers, n_kv_heads)
    baseline_pred = normalize_answer(extract_answer(baseline_text))
    print(f"  Baseline: pred={baseline_pred}, correct={baseline_pred == true_answer}")

    # Test 3 conditions at 33%
    for method in ['mask', 'zero_kv', 'zero_v']:
        keep_pos, evict_pos = select_positions(
            'h2o', 0.33, reasoning_len, prompt_len,
            importance, per_head_importance, n_kv_heads, answer_heads)
        answer_text = evict_and_generate(
            model, tokenizer, full_cache, prompt_len, reasoning_len,
            evict_pos, method, num_layers, n_kv_heads)
        pred = normalize_answer(extract_answer(answer_text))
        correct = pred == true_answer
        print(f"  {method:8s} h2o 33%: pred={pred}, correct={correct}, raw='{answer_text[:60]}'")

    # Test recent strategy with mask
    keep_pos, evict_pos = select_positions(
        'recent', 0.33, reasoning_len, prompt_len,
        importance, per_head_importance, n_kv_heads, answer_heads)
    answer_text = evict_and_generate(
        model, tokenizer, full_cache, prompt_len, reasoning_len,
        evict_pos, 'mask', num_layers, n_kv_heads)
    pred = normalize_answer(extract_answer(answer_text))
    print(f"  mask     recent 33%: pred={pred}, correct={pred == true_answer}")

    del full_cache
    gc.collect(); torch.cuda.empty_cache()

print("\nSmoke test complete!")
