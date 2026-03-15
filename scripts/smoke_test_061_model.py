#!/usr/bin/env python3
"""Smoke test: run exp 061 on 2 problems with Qwen only."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import torch
import numpy as np
from exp_061_effective_rank import (
    analyze_kv_cache, build_prompt, extract_answer, compute_spectral_metrics,
    EXEMPLARS
)
from transformers import AutoModelForCausalLM, AutoTokenizer

np.random.seed(42)
torch.manual_seed(42)

MODEL = "Qwen/Qwen3-4B-Base"
print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model.eval()

test_questions = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
]

for i, q in enumerate(test_questions):
    print(f"\nProblem {i+1}: {q[:60]}...")
    result = analyze_kv_cache(model, tokenizer, q, max_gen=256)
    if result is None:
        print("  FAILED - no result")
        continue

    print(f"  Answer: {result['answer']}")
    print(f"  Reasoning tokens: {result['reasoning_tokens']}")
    print(f"  Layers: {result['n_layers']}, KV heads: {result['n_kv_heads']}, Head dim: {result['head_dim']}")
    print(f"  Per-layer entries: {len(result['per_layer'])}")
    print(f"  Per-head entries: {len(result['per_head'])}")

    mid_layer = str(result['n_layers'] // 2)
    if mid_layer in result['per_layer']:
        km = result['per_layer'][mid_layer]['K']
        vm = result['per_layer'][mid_layer]['V']
        print(f"  Layer {mid_layer} K effective rank: {km['effective_rank']:.2f} (norm: {km['normalized_eff_rank']:.4f})")
        print(f"  Layer {mid_layer} V effective rank: {vm['effective_rank']:.2f} (norm: {vm['normalized_eff_rank']:.4f})")
        print(f"  Layer {mid_layer} K top-1 energy: {km['top1_energy']:.4f}")
        print(f"  Layer {mid_layer} V top-1 energy: {vm['top1_energy']:.4f}")
        print(f"  Layer {mid_layer} K decay rate: {km['decay_rate']:.4f}")
        print(f"  Layer {mid_layer} V decay rate: {vm['decay_rate']:.4f}")
        print(f"  K/V eff rank ratio: {km['effective_rank']/vm['effective_rank']:.3f}")

print("\nSmoke test COMPLETE")
