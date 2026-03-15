#!/usr/bin/env python3
"""Smoke test for Exp 044: run 3 problems to verify the pipeline works."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import scripts.exp_044_qwen_5pct_positional_sweep as exp
import torch, time, random, gc, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
os.makedirs(exp.RESULTS_DIR, exist_ok=True)
random.seed(exp.SEED); np.random.seed(exp.SEED); torch.manual_seed(exp.SEED)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(exp.MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    exp.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()
num_layers = model.config.num_hidden_layers
print(f"Model loaded in {time.time()-t0:.1f}s, {num_layers} layers")

# Verify cache
test_input = tokenizer("test", return_tensors="pt").to(model.device)
test_out = model(**test_input, use_cache=True)
test_cache = test_out.past_key_values
assert hasattr(test_cache, "layers"), "No layers attr"
print(f"Cache OK: keys shape = {test_cache.layers[0].keys.shape}")
del test_input, test_out, test_cache; gc.collect(); torch.cuda.empty_cache()

from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
indices = list(range(len(ds)))
random.shuffle(indices)
indices = indices[:5]

tested = 0
for pi, ds_idx in enumerate(indices):
    item = ds[ds_idx]
    true_answer = exp.normalize_answer(exp.extract_answer(item["answer"]))
    prompt = exp.build_prompt(item["question"])
    trace, prompt_ids, reasoning_ids = exp.generate_trace(model, tokenizer, prompt)
    gen_answer = exp.normalize_answer(exp.extract_answer(trace)) if exp.extract_answer(trace) else ""
    print(f"P{pi}: true={true_answer}, gen={gen_answer}, match={gen_answer==true_answer}, R={reasoning_ids.shape[1]}")
    if gen_answer == true_answer:
        trunc_pos = exp.find_truncation_point(reasoning_ids, tokenizer)
        if trunc_pos and trunc_pos >= 10:
            reasoning_trunc = reasoning_ids[:, :trunc_pos]
            prompt_len = prompt_ids.shape[1]
            pc = exp.build_prompt_cache(model, prompt_ids)
            positions = exp.select_decile_positions_subsampled(reasoning_trunc.shape[1], 5, rng_seed=42)
            ev = exp.evaluate_condition(model, tokenizer, pc, reasoning_trunc, positions,
                                        prompt_len, num_layers, true_answer, "k")
            print(f"  K-5: acc={ev['correct']}, text={ev['text_accuracy']:.3f}, n_pos={len(positions)}")
            pc2 = exp.build_prompt_cache(model, prompt_ids)
            v_pos = exp.select_decile_positions_subsampled(reasoning_trunc.shape[1], 9, rng_seed=4209)
            ev2 = exp.evaluate_condition(model, tokenizer, pc2, reasoning_trunc, v_pos,
                                         prompt_len, num_layers, true_answer, "v")
            print(f"  V-9: acc={ev2['correct']}, text={ev2['text_accuracy']:.3f}")
            del pc, pc2, reasoning_trunc
            gc.collect(); torch.cuda.empty_cache()
            tested += 1
            if tested >= 2:
                break
    del prompt_ids, reasoning_ids; gc.collect(); torch.cuda.empty_cache()

print(f"\nSmoke test PASSED: {tested} problems tested in {time.time()-t0:.0f}s")
