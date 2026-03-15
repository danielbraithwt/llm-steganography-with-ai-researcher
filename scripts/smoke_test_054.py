#!/usr/bin/env python3
"""Quick smoke test for exp_054 — 2 problems, Qwen only, 2 strategies, 2 budgets."""
import os, sys, time, random, json, gc, re
import numpy as np
import torch

# Patch the config before importing
sys.path.insert(0, os.path.dirname(__file__))

# Import functions from the main script
exec(open(os.path.join(os.path.dirname(__file__), "exp_054_kv_eviction_benchmark.py")).read().split("if __name__")[0])

# Override
NUM_PROBLEMS = 10
SMOKE_STRATEGIES = ["random", "h2o", "k_preserve"]
SMOKE_BUDGETS = [1.0, 0.50]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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

n_valid = 0
for prob_idx in range(min(NUM_PROBLEMS, len(problems))):
    problem = problems[prob_idx]
    question = problem["question"]
    true_answer = normalize_answer(problem["answer"].split("####")[-1].strip())
    prompt = build_prompt(question)

    print(f"\nProblem {prob_idx}: true={true_answer}")

    try:
        (gen_text, input_ids, gen_ids, importance, per_head_importance,
         prompt_len, full_cache, full_seq, trunc_point) = generate_and_build_cache(
            model, tokenizer, prompt)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
        continue

    clean_answer = normalize_answer(extract_answer(gen_text))
    print(f"  Clean answer: {clean_answer}, correct: {clean_answer == true_answer}")

    if clean_answer != true_answer or full_cache is None or trunc_point is None:
        if full_cache is not None:
            del full_cache; gc.collect(); torch.cuda.empty_cache()
        continue

    n_valid += 1
    reasoning_len = trunc_point
    print(f"  Reasoning len: {reasoning_len}, Prompt len: {prompt_len}")
    print(f"  Importance shape: {importance.shape}, range: [{importance.min():.2f}, {importance.max():.2f}]")

    for budget in SMOKE_BUDGETS:
        for strategy in SMOKE_STRATEGIES:
            if budget == 1.0 and strategy != "random":
                continue
            key = f"{strategy}_{budget:.2f}"
            try:
                sel = select_positions_to_keep(
                    strategy, budget, reasoning_len, prompt_len,
                    importance, per_head_importance, n_kv_heads, answer_heads)

                answer_text = evict_and_generate(
                    model, tokenizer, full_cache, full_seq,
                    prompt_len, reasoning_len, sel, num_layers, n_kv_heads)

                pred = normalize_answer(extract_answer(answer_text))
                correct = pred == true_answer
                print(f"  {key}: pred={pred}, correct={correct}, raw='{answer_text[:60]}'")
            except Exception as e:
                print(f"  {key}: ERROR: {e}")
                import traceback; traceback.print_exc()

    del full_cache; gc.collect(); torch.cuda.empty_cache()

    if n_valid >= 2:
        break

print(f"\nSmoke test done. Valid problems: {n_valid}")
