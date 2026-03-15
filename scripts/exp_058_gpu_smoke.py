#!/usr/bin/env python3
"""Quick GPU smoke test for exp_058: run 2 problems on Qwen only."""

import os
import sys
import time
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import exp_058_sink_analysis_budget_sweep as exp

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
problems = list(ds)
random.shuffle(problems)

from transformers import AutoModelForCausalLM, AutoTokenizer

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
late_start = int(num_layers * exp.LATE_LAYER_FRACTION)
print(f"Layers: {num_layers}, Late start: {late_start}")

n_tested = 0
for prob_idx, problem in enumerate(problems[:20]):
    if n_tested >= 2:
        break

    question = problem["question"]
    true_answer = exp.normalize_answer(problem["answer"].split("####")[-1].strip())
    prompt = exp.build_prompt(question)

    cache_result = exp.generate_and_build_cache(model, tokenizer, prompt)
    if cache_result is None:
        continue

    gen_text = cache_result['gen_text']
    clean_answer = exp.normalize_answer(exp.extract_answer(gen_text))
    if clean_answer != true_answer:
        del cache_result['full_cache']
        import gc; gc.collect(); torch.cuda.empty_cache()
        continue

    n_tested += 1
    print(f"\nProblem {prob_idx}: true={true_answer}, clean={clean_answer}")
    print(f"  Reasoning len: {cache_result['reasoning_len']}")

    prompt_len = cache_result['prompt_len']
    reasoning_len = cache_result['reasoning_len']
    k_norm_importance = cache_result['k_norm_importance']
    full_seq = cache_result['full_seq']
    full_cache = cache_result['full_cache']

    t0 = time.time()
    true_h2o_all, true_h2o_late = exp.compute_true_h2o_importance(
        model, full_seq, prompt_len, reasoning_len, num_layers)
    print(f"  True H2O computed in {time.time()-t0:.1f}s")

    if true_h2o_all is not None:
        from scipy import stats
        reason_knorm = k_norm_importance[prompt_len:prompt_len + reasoning_len]
        rho, _ = stats.spearmanr(reason_knorm, true_h2o_all)
        rho_late, _ = stats.spearmanr(reason_knorm, true_h2o_late)
        print(f"  K-norm vs all-layer H2O: rho={rho:.3f}")
        print(f"  K-norm vs late-layer H2O: rho={rho_late:.3f}")

    for sel in ['true_h2o', 'k_norm_h2o', 'sink_excluded_h2o', 'late_layer_h2o', 'random', 'recent']:
        keep_pos, evict_pos, keep_idx = exp.select_positions(
            sel, 0.33, reasoning_len, prompt_len,
            k_norm_importance, true_h2o_all, true_h2o_late)

        profile = exp.analyze_position_profile(keep_idx, reasoning_len)

        answer = exp.evict_and_generate(
            model, tokenizer, full_cache, prompt_len, reasoning_len,
            evict_pos, num_layers)
        pred = exp.normalize_answer(exp.extract_answer(answer))
        correct = pred == true_answer

        print(f"  {sel:22s}: acc={'OK' if correct else 'FAIL'} (pred={pred}), "
              f"sink_frac={profile['sink_frac']*100:.1f}%, mean_pos={profile['mean_pos']:.3f}")

    del full_cache
    import gc; gc.collect(); torch.cuda.empty_cache()

print("\nGPU smoke test PASSED")
