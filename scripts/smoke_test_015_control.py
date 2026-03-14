#!/usr/bin/env python3
"""Verify that clean KV cache → generation produces the correct answer.
If this fails, the pipeline has a bug; if it passes, the 0.01x fragility is real."""
import os, sys, json, time, random, gc, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

sys.path.insert(0, '/workspace/llm-steganography-with-ai-researcher')
from scripts.exp_015_qwen_instruct_noise_scale import (
    build_prompt, extract_answer, normalize_answer, generate_trace,
    build_kv_cache, clone_kv_cache, run_noised_generation,
    apply_scaled_noise_to_positions, GSM8K_EXEMPLARS
)

MODEL_NAME = "Qwen/Qwen3-4B"
random.seed(42); np.random.seed(42); torch.manual_seed(42)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()
num_layers = model.config.num_hidden_layers

print("Loading GSM8K...")
dataset = load_dataset("openai/gsm8k", "main", split="test")
problems = list(dataset.select(range(30, 36)))

for i, prob in enumerate(problems):
    question = prob["question"]
    true_ans = normalize_answer(prob["answer"].split("####")[-1].strip())
    prompt = build_prompt(question)

    # Phase 1: Direct generation
    trace = generate_trace(model, tokenizer, prompt)
    pred_ans = normalize_answer(extract_answer(trace))
    correct = (pred_ans == true_ans)

    if not correct:
        print(f"Problem {i}: WRONG directly (pred={pred_ans}, true={true_ans}), skip")
        continue

    # Phase 2: Clean KV cache generation (NO noise)
    base_kv, seq_len = build_kv_cache(model, tokenizer, prompt, trace)
    full_text = prompt + trace
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    last_token_id = full_ids[-1].item()
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    clean_gen = run_noised_generation(model, tokenizer, base_kv, seq_len, last_token_id)
    clean_ans = normalize_answer(extract_answer(clean_gen))
    clean_correct = (clean_ans == true_ans)

    print(f"Problem {i}: direct={pred_ans} | clean_kv_gen='{clean_ans}' | true={true_ans} | clean_OK={clean_correct}")
    print(f"  Clean gen text: {clean_gen[:200]}")

    # Phase 3: Tiny noise (0.01x) at 1 position
    if seq_len - prompt_len > 5:
        n_pos = seq_len - prompt_len
        kv_copy = clone_kv_cache(base_kv)
        positions = [n_pos // 2]  # just 1 position in the middle
        apply_scaled_noise_to_positions(kv_copy, positions, prompt_len, num_layers, 0.01)
        tiny_gen = run_noised_generation(model, tokenizer, kv_copy, seq_len, last_token_id)
        tiny_ans = normalize_answer(extract_answer(tiny_gen))
        tiny_correct = (tiny_ans == true_ans)
        print(f"  0.01x noise at 1 pos: '{tiny_ans}' | OK={tiny_correct}")
        print(f"  Tiny gen text: {tiny_gen[:200]}")
        del kv_copy

    # Phase 4: 0.01x noise at 3% of positions
    if seq_len - prompt_len > 5:
        n_noise = max(1, int(n_pos * 0.03))
        kv_copy = clone_kv_cache(base_kv)
        positions = np.random.choice(n_pos, n_noise, replace=False)
        apply_scaled_noise_to_positions(kv_copy, positions, prompt_len, num_layers, 0.01)
        small_gen = run_noised_generation(model, tokenizer, kv_copy, seq_len, last_token_id)
        small_ans = normalize_answer(extract_answer(small_gen))
        small_correct = (small_ans == true_ans)
        print(f"  0.01x noise at 3% ({n_noise} pos): '{small_ans}' | OK={small_correct}")
        print(f"  Small gen text: {small_gen[:200]}")
        del kv_copy

    del base_kv
    gc.collect()
    torch.cuda.empty_cache()
    print()

print("Done.")
