#!/usr/bin/env python3
"""Quick smoke test for exp_026 — run 2 problems with 3 conditions."""
import os, sys, time, random, gc, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.exp_026_qwen_dose_response import (
    MODEL_NAME, EXEMPLARS, build_prompt, extract_answer, normalize_answer,
    generate_trace, build_prompt_cache, find_truncation_point,
    select_late_positions, evaluate_condition
)

t0 = time.time()
random.seed(42); np.random.seed(42); torch.manual_seed(42)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager")
model.eval()
num_layers = model.config.num_hidden_layers
print(f"Loaded: {num_layers} layers, t={time.time()-t0:.0f}s")

# Verify cache
test_input = tokenizer("test", return_tensors="pt").to(model.device)
test_out = model(**test_input, use_cache=True)
assert hasattr(test_out.past_key_values, "layers")
print(f"Cache OK: {test_out.past_key_values.layers[0].keys.shape}")
del test_input, test_out; gc.collect(); torch.cuda.empty_cache()

ds = load_dataset("openai/gsm8k", "main", split="test")
indices = list(range(len(ds)))
random.shuffle(indices)

n_done = 0
for ds_idx in indices[:20]:
    item = ds[ds_idx]
    true_answer = normalize_answer(extract_answer(item["answer"]))
    prompt = build_prompt(item["question"])
    trace, prompt_ids, reasoning_ids = generate_trace(model, tokenizer, prompt)
    gen_answer = normalize_answer(extract_answer(trace)) if extract_answer(trace) else ""
    if gen_answer != true_answer:
        continue
    trunc_pos = find_truncation_point(reasoning_ids, tokenizer)
    if trunc_pos is None or trunc_pos < 10:
        continue
    reasoning_ids_truncated = reasoning_ids[:, :trunc_pos]
    prompt_len = prompt_ids.shape[1]
    reasoning_len = reasoning_ids_truncated.shape[1]
    if reasoning_len < 20:
        continue

    pc = build_prompt_cache(model, prompt_ids, num_layers)
    clean = evaluate_condition(model, tokenizer, pc, reasoning_ids_truncated,
                               [], prompt_len, num_layers, true_answer)
    if not clean["correct"]:
        continue

    positions = select_late_positions(reasoning_len)
    print(f"\nProblem idx={ds_idx}: R={reasoning_len}, positions={len(positions)}, "
          f"clean_text={clean['text_accuracy']:.3f}")

    test_conds = [("mag_v_10", "v", 1.0), ("mag_k_03", "k", 0.3), ("mag_kv_03", "kv", 0.3)]
    for cond_name, component, sigma in test_conds:
        pc = build_prompt_cache(model, prompt_ids, num_layers)
        ev = evaluate_condition(model, tokenizer, pc, reasoning_ids_truncated,
                                positions, prompt_len, num_layers, true_answer,
                                perturb_component=component, magnitude_sigma=sigma)
        print(f"  {cond_name}: correct={ev['correct']}, text={ev['text_accuracy']:.3f}, "
              f"rms={ev['rms_perturbation']:.1f}")

    n_done += 1
    del prompt_ids, reasoning_ids, reasoning_ids_truncated, pc
    gc.collect(); torch.cuda.empty_cache()
    if n_done >= 2:
        break

print(f"\nSmoke test: {n_done} problems in {time.time()-t0:.0f}s")
if n_done >= 2:
    print("SMOKE TEST PASSED")
else:
    print("WARNING: fewer than 2 valid problems found")
