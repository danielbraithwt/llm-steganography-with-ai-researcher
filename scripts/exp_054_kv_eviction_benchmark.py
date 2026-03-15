#!/usr/bin/env python3
"""
Experiment 054: KV Cache Eviction Benchmark — Mechanistically-Informed Compression

Tests whether our mechanistic findings (K>V, answer heads, early-position
infrastructure) translate to better KV cache compression strategies.

DESIGN:
  - Generate full 8-shot CoT trace with clean cache
  - Before answer generation, EVICT positions from the KV cache
  - Compare eviction strategies at multiple cache budgets
  - Measure answer accuracy and text coherence after eviction

STRATEGIES:
  1. random: Randomly select positions to keep
  2. h2o: Keep positions with highest cumulative attention (Heavy Hitter Oracle)
  3. recent: Keep most recent N positions + first 4 (attention sinks)
  4. early_priority: Keep earliest positions first (infrastructure hypothesis)
  5. k_preserve: Evict V but keep K at evicted positions (K>V hypothesis)
  6. head_selective: Different budgets per head (answer heads get 2x)

CACHE BUDGETS: 100% (baseline), 75%, 50%, 33%
MODELS: Qwen/Qwen3-4B-Base, meta-llama/Llama-3.1-8B-Instruct
"""

import os
import json
import time
import random
import gc
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ── Config ──────────────────────────────────────────────────────────────
NUM_PROBLEMS = 200  # attempt many, time-limited
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 2048
SEED = 42
TIME_BUDGET = 1700  # seconds total
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_054")

CACHE_BUDGETS = [1.0, 0.75, 0.50, 0.33]  # fraction of reasoning positions to keep

# Strategies to test
STRATEGIES = ["random", "h2o", "recent", "early_priority", "k_preserve", "head_selective"]

EXEMPLARS = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
     "a": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "a": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total bolts needed is 2+1=<<2+1=3>>3\n#### 3"},
    {"q": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
     "a": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*150%=$<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"},
    {"q": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "a": "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624"},
    {"q": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
     "a": "If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.\nIf she feeds the flock 15 cups in the morning, and 25 cups in the afternoon, then the carry-over to the final meal would be 60-15-25=<<60-15-25=20>>20 cups.\n#### 20"},
    {"q": "Kylar went to the store to get water and some apples. The store sold apples for $1 each and water for $3 per bottle. Kylar wanted to buy one bag of apples and 2 bottles of water. How much would Kylar spend if each bag has 6 apples?",
     "a": "A bag has 6 apples and each apple costs $1, so a bag costs 6*1=$<<6*1=6>>6\nKylar wants 2 bottles of water so that would cost 2*3=$<<2*3=6>>6\nAltogether, Kylar would spend 6+6=$<<6+6=12>>12\n#### 12"},
    {"q": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
     "a": "If Seattle has 20 sheep, Charleston has 4 * 20 = <<4*20=80>>80 sheep\nToulouse has 2 * 80 = <<2*80=160>>160 sheep\nTogether, they have 20 + 80 + 160 = <<20+80+160=260>>260 sheep\n#### 260"},
    {"q": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
     "a": "First find how long it takes to download 40% of the file: 200 GB * 0.4 / 2 GB/minute = <<200*0.4/2=40>>40 minutes\nThen find how long it takes to download the whole file once the restart is complete: 200 GB / 2 GB/minute = <<200/2=100>>100 minutes\nThen add the time to download 40% of the file, the restart time, and the time to download the whole file: 40 + 20 + 100 = <<40+20+100=160>>160 minutes\n#### 160"},
]


def build_prompt(question):
    prompt = ""
    for ex in EXEMPLARS:
        prompt += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def extract_answer(text):
    if "####" in text:
        after = text.split("####")[-1].strip()
        m = re.search(r'-?[\d,]+\.?\d*', after)
        return m.group(0).replace(',', '') if m else ""
    m = re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    return nums[-1].replace(',', '') if nums else ""


def normalize_answer(ans):
    ans = ans.strip().replace(",", "").replace("$", "")
    try:
        val = float(ans)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return ans


def find_truncation_point(reasoning_ids, tokenizer):
    """Find where to truncate reasoning before the answer marker."""
    ids_list = reasoning_ids[0].tolist()
    text = tokenizer.decode(ids_list, skip_special_tokens=True)
    if "####" in text:
        prefix = text[:text.index("####")]
        prefix_toks = tokenizer.encode(prefix, add_special_tokens=False)
        pos = len(prefix_toks)
        if pos >= 10:
            return pos
    m = re.search(r'[Tt]he (?:final )?answer is', text)
    if m:
        prefix = text[:m.start()]
        prefix_toks = tokenizer.encode(prefix, add_special_tokens=False)
        pos = len(prefix_toks)
        if pos >= 10:
            return pos
    return None


@torch.no_grad()
def generate_and_build_cache(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS):
    """Generate CoT trace, build KV cache, compute importance scores.

    Uses K-vector L2 norms as memory-efficient proxy for H2O attention importance.
    (Full output_attentions would OOM: 36 layers × 32 heads × seq² × 2 bytes.)

    Returns (gen_text, input_ids, gen_ids, importance_scores, per_head_importance,
             prompt_len, full_cache, full_ids_truncated).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    # Generate full trace (greedy)
    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
    )
    full_ids = gen_out.sequences[0]
    gen_ids = full_ids[prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del gen_out
    gc.collect(); torch.cuda.empty_cache()

    # Find truncation point (before ####)
    reasoning_ids = gen_ids.unsqueeze(0)
    trunc_point = find_truncation_point(reasoning_ids, tokenizer)

    if trunc_point is None or trunc_point < 10:
        return (gen_text, inputs.input_ids, gen_ids, None, None, prompt_len, None, None, None)

    # Build KV cache for prompt + reasoning (up to truncation)
    full_seq = torch.cat([inputs.input_ids[0], gen_ids[:trunc_point]])
    total_len = full_seq.shape[0]
    if total_len > MAX_SEQ_LEN:
        return (gen_text, inputs.input_ids, gen_ids, None, None, prompt_len, None, None, None)

    outputs = model(input_ids=full_seq.unsqueeze(0), use_cache=True)
    full_cache = outputs.past_key_values
    del outputs

    # Compute importance via K-vector L2 norms (memory-efficient H2O proxy)
    # Higher K-norm → position attracts more attention (QK dot product)
    num_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    importance = np.zeros(total_len)
    per_head_importance = np.zeros((n_kv_heads, total_len))

    for l in range(num_layers):
        k = full_cache.key_cache[l][0]  # [n_kv_heads, seq_len, head_dim]
        k_norms = k.float().norm(dim=-1).cpu().numpy()  # [n_kv_heads, seq_len]
        importance += k_norms.mean(axis=0)
        per_head_importance += k_norms

    return (gen_text, inputs.input_ids, gen_ids, importance, per_head_importance,
            prompt_len, full_cache, full_seq, trunc_point)


@torch.no_grad()

def select_positions_to_keep(strategy, budget_frac, reasoning_len, prompt_len,
                             importance, per_head_importance, n_kv_heads,
                             answer_heads=(0, 5)):
    """Select which reasoning positions to keep based on strategy.

    Returns:
      For most strategies: list of position indices (absolute, in full sequence) to KEEP
      For head_selective: dict mapping kv_head_idx -> list of positions to keep
      For k_preserve: (positions_to_keep_full, positions_to_zero_v_only)
    """
    n_keep = max(1, int(budget_frac * reasoning_len))
    # Reasoning positions in absolute terms
    reason_positions = list(range(prompt_len, prompt_len + reasoning_len))
    # Importance scores for reasoning positions only
    reason_attn = importance[prompt_len:prompt_len + reasoning_len]

    if strategy == "random":
        rng = np.random.RandomState(42)
        keep_indices = sorted(rng.choice(reasoning_len, size=n_keep, replace=False))
        return [reason_positions[i] for i in keep_indices]

    elif strategy == "h2o":
        # Keep positions with highest cumulative attention
        top_indices = np.argsort(reason_attn)[-n_keep:]
        top_indices = sorted(top_indices)
        return [reason_positions[i] for i in top_indices]

    elif strategy == "recent":
        # Keep first 4 (attention sinks) + most recent positions
        n_sinks = min(4, reasoning_len)
        n_recent = n_keep - n_sinks
        if n_recent <= 0:
            keep_indices = list(range(n_sinks))
        else:
            keep_indices = list(range(n_sinks)) + list(range(reasoning_len - n_recent, reasoning_len))
        keep_indices = sorted(set(keep_indices))[:n_keep]
        return [reason_positions[i] for i in keep_indices]

    elif strategy == "early_priority":
        # Keep earliest positions first (infrastructure hypothesis)
        keep_indices = list(range(n_keep))
        return [reason_positions[i] for i in keep_indices]

    elif strategy == "k_preserve":
        # Keep top-H2O positions fully, but at EVICTED positions: keep K, zero V
        top_indices = np.argsort(reason_attn)[-n_keep:]
        top_indices_set = set(top_indices)
        full_keep = sorted([reason_positions[i] for i in top_indices])
        v_zero = sorted([reason_positions[i] for i in range(reasoning_len)
                        if i not in top_indices_set])
        return ("k_preserve", full_keep, v_zero)

    elif strategy == "head_selective":
        # Answer heads (H0, H5) get 2x budget; others get reduced
        # Total budget must match: answer_budget * n_answer + other_budget * n_other = total_budget
        n_answer = len(answer_heads)
        n_other = n_kv_heads - n_answer
        # answer_budget = 2 * other_budget
        # 2*other*n_answer + other*n_other = n_keep * n_kv_heads
        # other * (2*n_answer + n_other) = n_keep * n_kv_heads
        other_budget = (n_keep * n_kv_heads) / (2 * n_answer + n_other)
        answer_budget = 2 * other_budget

        head_positions = {}
        for kv_h in range(n_kv_heads):
            if kv_h in answer_heads:
                h_keep = min(reasoning_len, max(1, int(round(answer_budget))))
            else:
                h_keep = min(reasoning_len, max(1, int(round(other_budget))))

            # Use per-head importance to select positions
            h_attn = per_head_importance[kv_h, prompt_len:prompt_len + reasoning_len]
            top_idx = np.argsort(h_attn)[-h_keep:]
            head_positions[kv_h] = sorted([reason_positions[i] for i in top_idx])

        return ("head_selective", head_positions)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


@torch.no_grad()
def evict_and_generate(model, tokenizer, full_cache, full_ids, prompt_len,
                       reasoning_len, strategy_result, num_layers, n_kv_heads,
                       max_answer_tokens=64):
    """Apply eviction to cache, then generate answer from the truncated cache.

    Returns (generated_answer_text, answer_text).
    """
    total_reasoning_end = prompt_len + reasoning_len

    # Build evicted cache
    evicted_cache = DynamicCache()

    if isinstance(strategy_result, tuple) and strategy_result[0] == "k_preserve":
        _, full_keep_positions, v_zero_positions = strategy_result
        # Keep all positions but zero V at evicted positions
        keep_set = set(full_keep_positions)
        vzero_set = set(v_zero_positions)
        # All prompt + reasoning positions kept
        all_keep = list(range(total_reasoning_end))

        for layer_idx in range(num_layers):
            k = full_cache.key_cache[layer_idx][:, :, :total_reasoning_end, :].clone()
            v = full_cache.value_cache[layer_idx][:, :, :total_reasoning_end, :].clone()
            # Zero V at evicted reasoning positions
            for pos in v_zero_positions:
                v[:, :, pos, :] = 0.0
            evicted_cache.update(k, v, layer_idx)

    elif isinstance(strategy_result, tuple) and strategy_result[0] == "head_selective":
        _, head_positions = strategy_result
        # For each head, keep different positions. Zero out evicted positions per head.
        for layer_idx in range(num_layers):
            k = full_cache.key_cache[layer_idx][:, :, :total_reasoning_end, :].clone()
            v = full_cache.value_cache[layer_idx][:, :, :total_reasoning_end, :].clone()
            for kv_h in range(n_kv_heads):
                keep_set = set(head_positions[kv_h])
                for pos in range(prompt_len, total_reasoning_end):
                    if pos not in keep_set:
                        k[:, kv_h, pos, :] = 0.0
                        v[:, kv_h, pos, :] = 0.0
            evicted_cache.update(k, v, layer_idx)

    else:
        # Standard position eviction: keep only selected positions
        keep_positions = strategy_result
        keep_set = set(keep_positions)
        # Prompt positions are ALWAYS kept
        all_prompt = list(range(prompt_len))

        for layer_idx in range(num_layers):
            k = full_cache.key_cache[layer_idx][:, :, :total_reasoning_end, :].clone()
            v = full_cache.value_cache[layer_idx][:, :, :total_reasoning_end, :].clone()
            # Zero out evicted reasoning positions
            for pos in range(prompt_len, total_reasoning_end):
                if pos not in keep_set:
                    k[:, :, pos, :] = 0.0
                    v[:, :, pos, :] = 0.0
            evicted_cache.update(k, v, layer_idx)

    # Generate answer from evicted cache
    # Feed the "####" prefix token to trigger answer generation
    answer_prefix = tokenizer.encode(" ####", add_special_tokens=False)
    answer_prefix_tensor = torch.tensor([answer_prefix], device=model.device)

    # Step through answer prefix
    for i in range(len(answer_prefix)):
        tok = answer_prefix_tensor[:, i:i+1]
        out = model(input_ids=tok, past_key_values=evicted_cache, use_cache=True)
        evicted_cache = out.past_key_values

    # Generate answer tokens
    gen_tokens = []
    next_logits = out.logits[:, -1, :]

    for step in range(max_answer_tokens):
        next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
        token_id = next_tok[0, 0].item()
        gen_tokens.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        out = model(input_ids=next_tok, past_key_values=evicted_cache, use_cache=True)
        evicted_cache = out.past_key_values
        next_logits = out.logits[:, -1, :]

    answer_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    del evicted_cache
    return answer_text


def wilson_ci(n_success, n_total, z=1.96):
    if n_total == 0:
        return (0, 0)
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return (max(0, center - spread), min(1, center + spread))


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_figures(all_results, results_dir):
    """Generate publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for model_key, model_results in all_results.items():
        if not model_results:
            continue

        model_short = model_key.split("/")[-1]

        # Figure 1: Accuracy vs Cache Budget for each strategy
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        colors = {'random': '#999999', 'h2o': '#2196F3', 'recent': '#FF9800',
                  'early_priority': '#4CAF50', 'k_preserve': '#9C27B0',
                  'head_selective': '#F44336'}
        markers = {'random': 'o', 'h2o': 's', 'recent': '^',
                   'early_priority': 'D', 'k_preserve': 'v', 'head_selective': 'P'}

        for strategy in STRATEGIES:
            budgets = []
            accs = []
            ci_lo = []
            ci_hi = []
            for budget in CACHE_BUDGETS:
                key = f"{strategy}_{budget:.2f}"
                if key in model_results:
                    r = model_results[key]
                    budgets.append(budget * 100)
                    accs.append(r['accuracy'] * 100)
                    lo, hi = wilson_ci(r['n_correct'], r['n_valid'])
                    ci_lo.append(lo * 100)
                    ci_hi.append(hi * 100)

            if budgets:
                ax.errorbar(budgets, accs,
                           yerr=[np.array(accs) - np.array(ci_lo),
                                 np.array(ci_hi) - np.array(accs)],
                           label=strategy, color=colors.get(strategy, '#000'),
                           marker=markers.get(strategy, 'o'),
                           linewidth=2, markersize=8, capsize=4)

        ax.set_xlabel('Cache Budget (% of reasoning positions)', fontsize=13)
        ax.set_ylabel('Answer Accuracy (%)', fontsize=13)
        ax.set_title(f'KV Cache Eviction: {model_short}', fontsize=14)
        ax.legend(fontsize=11, loc='lower right')
        ax.set_xlim(25, 105)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'accuracy_vs_budget_{model_short}.png'), dpi=150)
        plt.close()

        # Figure 2: Strategy comparison at 50% budget (bar chart)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        strategy_names = []
        strategy_accs = []
        strategy_errs = []
        strategy_colors = []

        for strategy in STRATEGIES:
            key = f"{strategy}_0.50"
            if key in model_results:
                r = model_results[key]
                strategy_names.append(strategy.replace('_', '\n'))
                acc = r['accuracy'] * 100
                strategy_accs.append(acc)
                lo, hi = wilson_ci(r['n_correct'], r['n_valid'])
                strategy_errs.append([acc - lo*100, hi*100 - acc])
                strategy_colors.append(colors.get(strategy, '#000'))

        if strategy_names:
            x = np.arange(len(strategy_names))
            errs = np.array(strategy_errs).T
            bars = ax.bar(x, strategy_accs, color=strategy_colors, alpha=0.8, width=0.6)
            ax.errorbar(x, strategy_accs, yerr=errs, fmt='none', color='black', capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(strategy_names, fontsize=11)
            ax.set_ylabel('Answer Accuracy (%)', fontsize=13)
            ax.set_title(f'Strategy Comparison at 50% Budget — {model_short}', fontsize=14)
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3, axis='y')

            # Add baseline line
            baseline_key = f"random_1.00"
            if baseline_key in model_results:
                bl = model_results[baseline_key]['accuracy'] * 100
                ax.axhline(y=bl, color='gray', linestyle='--', linewidth=1.5,
                          label=f'Baseline ({bl:.0f}%)')
                ax.legend(fontsize=11)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'strategy_comparison_50pct_{model_short}.png'),
                        dpi=150)
            plt.close()

        # Figure 3: K-preserve vs standard eviction comparison
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for strat, label, color in [("h2o", "H2O (evict K+V)", "#2196F3"),
                                     ("k_preserve", "H2O + K-preserve", "#9C27B0")]:
            budgets = []
            accs = []
            for budget in CACHE_BUDGETS:
                key = f"{strat}_{budget:.2f}"
                if key in model_results:
                    budgets.append(budget * 100)
                    accs.append(model_results[key]['accuracy'] * 100)
            if budgets:
                ax.plot(budgets, accs, label=label, color=color,
                       marker='o', linewidth=2.5, markersize=9)

        ax.set_xlabel('Cache Budget (%)', fontsize=13)
        ax.set_ylabel('Answer Accuracy (%)', fontsize=13)
        ax.set_title(f'K-Preserve Effect — {model_short}', fontsize=14)
        ax.legend(fontsize=12)
        ax.set_xlim(25, 105)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'k_preserve_comparison_{model_short}.png'), dpi=150)
        plt.close()

    print(f"Figures saved to {results_dir}")


def run_model(model_name, problems, time_limit):
    """Run all strategies and budgets on one model."""
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager"  # need attention weights
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    num_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_attn_heads = model.config.num_attention_heads

    print(f"  Layers: {num_layers}, KV heads: {n_kv_heads}, Attn heads: {n_attn_heads}")

    # Determine answer heads based on model
    if "qwen" in model_name.lower():
        answer_heads = (0, 5)
    elif "llama" in model_name.lower():
        answer_heads = (3, 5)  # H5 primary, H3 second-most-critical on Llama
    else:
        answer_heads = (0, 5)

    # Aggregate results
    results = {}  # key: f"{strategy}_{budget}" -> {n_correct, n_valid, accuracy, ...}
    per_problem = []
    start_time = time.time()
    n_valid = 0
    n_attempted = 0

    for prob_idx, problem in enumerate(problems):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            print(f"\n  TIME LIMIT reached at problem {prob_idx} ({elapsed:.0f}s)")
            break

        question = problem["question"]
        true_answer = normalize_answer(problem["answer"].split("####")[-1].strip())
        prompt = build_prompt(question)

        # Generate trace and build cache with importance scores
        try:
            (gen_text, input_ids, gen_ids, importance, per_head_importance,
             prompt_len, full_cache, full_seq, trunc_point) = generate_and_build_cache(
                model, tokenizer, prompt)
        except Exception as e:
            print(f"  Problem {prob_idx}: generation error: {e}")
            continue

        n_attempted += 1

        # Check clean answer
        clean_answer = normalize_answer(extract_answer(gen_text))
        if clean_answer != true_answer:
            if full_cache is not None:
                del full_cache
                gc.collect(); torch.cuda.empty_cache()
            continue  # skip problems where clean model is wrong

        # Check cache was built successfully
        if full_cache is None or trunc_point is None:
            continue

        reasoning_len = trunc_point
        n_valid += 1

        prob_results = {"idx": prob_idx, "true_answer": true_answer,
                        "clean_answer": clean_answer, "reasoning_len": reasoning_len}

        # Test each strategy × budget combination
        for budget in CACHE_BUDGETS:
            for strategy in STRATEGIES:
                if budget == 1.0 and strategy != "random":
                    continue  # at 100% budget, all strategies are equivalent

                key = f"{strategy}_{budget:.2f}"

                try:
                    sel = select_positions_to_keep(
                        strategy, budget, reasoning_len, prompt_len,
                        importance, per_head_importance, n_kv_heads, answer_heads)

                    answer_text = evict_and_generate(
                        model, tokenizer, full_cache, full_seq,
                        prompt_len, reasoning_len, sel, num_layers, n_kv_heads)

                    pred = normalize_answer(extract_answer(answer_text))
                    correct = (pred == true_answer)

                    if key not in results:
                        results[key] = {"n_correct": 0, "n_valid": 0, "answers": []}
                    results[key]["n_correct"] += int(correct)
                    results[key]["n_valid"] += 1
                    results[key]["answers"].append({
                        "prob_idx": prob_idx, "correct": correct,
                        "pred": pred, "true": true_answer,
                        "raw_answer": answer_text[:100]
                    })

                    prob_results[key] = {"correct": correct, "pred": pred,
                                         "raw": answer_text[:80]}

                except Exception as e:
                    print(f"  Problem {prob_idx}, {key}: error: {e}")
                    continue

        per_problem.append(prob_results)
        del full_cache
        gc.collect()
        torch.cuda.empty_cache()

        if n_valid % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Valid: {n_valid}, Attempted: {n_attempted}, "
                  f"Elapsed: {elapsed:.0f}s, "
                  f"Rate: {elapsed/max(1,n_valid):.1f}s/prob")

    # Compute accuracy for each key
    for key in results:
        r = results[key]
        r["accuracy"] = r["n_correct"] / max(1, r["n_valid"])

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results, per_problem, n_valid, n_attempted


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load GSM8K
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = list(ds)
    random.shuffle(problems)

    print(f"Loaded {len(problems)} GSM8K problems")
    print(f"Strategies: {STRATEGIES}")
    print(f"Budgets: {CACHE_BUDGETS}")

    all_results = {}
    all_per_problem = {}
    all_stats = {}

    models = ["Qwen/Qwen3-4B-Base", "meta-llama/Llama-3.1-8B-Instruct"]

    global_start = time.time()

    for model_name in models:
        elapsed = time.time() - global_start
        remaining = TIME_BUDGET - elapsed
        if remaining < 120:
            print(f"\nInsufficient time for {model_name} ({remaining:.0f}s left)")
            break

        model_time = remaining / (len(models) - models.index(model_name))
        model_time = min(model_time, remaining - 60)  # leave 60s buffer

        results, per_problem, n_valid, n_attempted = run_model(
            model_name, problems, model_time)

        all_results[model_name] = results
        all_per_problem[model_name] = per_problem
        all_stats[model_name] = {
            "n_valid": n_valid, "n_attempted": n_attempted,
            "time_used": time.time() - global_start
        }

        # Print summary table for this model
        print(f"\n{'─'*80}")
        print(f"RESULTS: {model_name} (n_valid={n_valid})")
        print(f"{'─'*80}")
        print(f"{'Strategy':<18} {'Budget':<8} {'Acc%':<8} {'n_ok':<6} {'n':<6} {'Wilson 95% CI':<16}")
        print(f"{'─'*18} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*16}")

        for budget in CACHE_BUDGETS:
            for strategy in STRATEGIES:
                if budget == 1.0 and strategy != "random":
                    continue
                key = f"{strategy}_{budget:.2f}"
                if key in results:
                    r = results[key]
                    acc = r["accuracy"] * 100
                    lo, hi = wilson_ci(r["n_correct"], r["n_valid"])
                    ci_str = f"[{lo*100:.1f}, {hi*100:.1f}]"
                    print(f"{strategy:<18} {budget*100:>5.0f}%   {acc:>6.1f}%  "
                          f"{r['n_correct']:<6} {r['n_valid']:<6} {ci_str}")
            if budget < 1.0:
                print()

    # Generate figures
    generate_figures(all_results, RESULTS_DIR)

    # Save results
    summary = {
        "experiment": "054_kv_eviction_benchmark",
        "strategies": STRATEGIES,
        "budgets": CACHE_BUDGETS,
        "stats": all_stats,
    }

    # Prepare serializable results (strip answer lists for summary)
    summary_results = {}
    for model_name, results in all_results.items():
        summary_results[model_name] = {}
        for key, r in results.items():
            summary_results[model_name][key] = {
                "n_correct": r["n_correct"],
                "n_valid": r["n_valid"],
                "accuracy": r["accuracy"],
            }
    summary["results"] = summary_results

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_convert)

    with open(os.path.join(RESULTS_DIR, "per_problem.json"), "w") as f:
        json.dump(all_per_problem, f, indent=2, default=_convert)

    total_time = time.time() - global_start
    print(f"\nTotal time: {total_time:.0f}s")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
