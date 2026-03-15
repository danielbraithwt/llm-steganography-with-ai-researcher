#!/usr/bin/env python3
"""
Experiment 056: Attention-Masking vs Zeroing in KV Cache Eviction
— Testing the Phantom Routing Hypothesis

Exp 055 found that K-preserve (keep K, zero V at evicted positions) was
WORSE than H2O on Llama at 33% budget (47.8% vs 78.3%). The hypothesis
is that this is due to "phantom routing" — the model routes attention
to positions with no V-content, wasting attention budget.

Also, Exp 055 used zeroing (K=V=0) which still lets evicted positions
participate in softmax normalization, diluting attention. True masking
removes positions from attention entirely.

This experiment tests 3 eviction methods:
  1. mask:    Attention mask = 0 at evicted positions (true removal)
  2. zero_kv: Zero both K and V (Exp 055 default — softmax dilution)
  3. zero_v:  Keep K intact, zero only V (phantom routing test)

KEY PREDICTIONS:
  - mask >= zero_kv for ALL strategies (removing dilution always helps)
  - mask >> zero_v on Llama at 33% (phantom routing confirmed)
  - Qwen: mask ≈ zero_kv ≈ zero_v ≈ 100% (digital encoding is robust)
  - Strategy rankings (recent > h2o > random) preserved under masking

IMPLEMENTATION NOTE:
  When using attention_mask with past_key_values, we must also pass explicit
  position_ids to prevent the model from recomputing positions based on the
  mask (which would shift RoPE rotations for post-eviction positions).
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
                           "results", "exp_056")

# Conditions: (method, strategy, budget)
# Focus on 33% budget where Exp 055 showed max differentiation,
# plus 50% for key strategies to check dose-response
CONDITIONS = [
    # 33% budget — primary comparison
    ('mask',    'random', 0.33),
    ('mask',    'h2o',    0.33),
    ('mask',    'recent', 0.33),
    ('zero_kv', 'random', 0.33),
    ('zero_kv', 'h2o',    0.33),
    ('zero_kv', 'recent', 0.33),
    ('zero_kv', 'head_selective', 0.33),
    # K-preserve / phantom routing test at 33%
    ('zero_v',  'h2o',    0.33),
    ('zero_v',  'recent', 0.33),
    # 50% budget — secondary comparison
    ('mask',    'h2o',    0.50),
    ('mask',    'recent', 0.50),
    ('zero_kv', 'h2o',    0.50),
    ('zero_kv', 'recent', 0.50),
]

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
    if "\nQ:" in text:
        text = text[:text.index("\nQ:")]
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
    """Generate CoT trace, build KV cache, compute K-norm importance scores."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

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

    # Compute importance via K-vector L2 norms (H2O proxy)
    num_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    importance = np.zeros(total_len)
    per_head_importance = np.zeros((n_kv_heads, total_len))

    for l in range(num_layers):
        k = full_cache.layers[l].keys[0]  # [n_kv_heads, seq_len, head_dim]
        k_norms = k.float().norm(dim=-1).cpu().numpy()  # [n_kv_heads, seq_len]
        importance += k_norms.mean(axis=0)
        per_head_importance += k_norms

    return (gen_text, inputs.input_ids, gen_ids, importance, per_head_importance,
            prompt_len, full_cache, full_seq, trunc_point)


def select_positions(strategy, budget_frac, reasoning_len, prompt_len,
                     importance, per_head_importance, n_kv_heads, answer_heads):
    """Select which reasoning positions to keep and which to evict.

    Returns:
      For most strategies: (keep_positions, evict_positions)
      For head_selective: ('head_selective', head_positions_dict)
    """
    n_keep = max(1, int(budget_frac * reasoning_len))
    reason_positions = list(range(prompt_len, prompt_len + reasoning_len))
    reason_attn = importance[prompt_len:prompt_len + reasoning_len]

    if strategy == "random":
        rng = np.random.RandomState(42)
        keep_idx = set(rng.choice(reasoning_len, size=n_keep, replace=False).tolist())

    elif strategy == "h2o":
        keep_idx = set(np.argsort(reason_attn)[-n_keep:].tolist())

    elif strategy == "recent":
        n_sinks = min(4, reasoning_len)
        n_recent = max(0, n_keep - n_sinks)
        if n_recent <= 0:
            keep_list = list(range(n_sinks))
        else:
            keep_list = list(range(n_sinks)) + list(range(reasoning_len - n_recent, reasoning_len))
        keep_idx = set(sorted(set(keep_list))[:n_keep])

    elif strategy == "head_selective":
        # Per-head: answer heads get 2x budget, others reduced
        n_answer = len(answer_heads)
        n_other = n_kv_heads - n_answer
        other_budget = (n_keep * n_kv_heads) / (2 * n_answer + n_other)
        answer_budget = 2 * other_budget

        head_positions = {}
        for kv_h in range(n_kv_heads):
            if kv_h in answer_heads:
                h_keep = min(reasoning_len, max(1, int(round(answer_budget))))
            else:
                h_keep = min(reasoning_len, max(1, int(round(other_budget))))
            h_attn = per_head_importance[kv_h, prompt_len:prompt_len + reasoning_len]
            top_idx = np.argsort(h_attn)[-h_keep:]
            head_positions[kv_h] = set(reason_positions[i] for i in top_idx)

        return ('head_selective', head_positions)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    keep_pos = sorted(reason_positions[i] for i in keep_idx)
    evict_pos = sorted(reason_positions[i] for i in range(reasoning_len) if i not in keep_idx)
    return (keep_pos, evict_pos)


@torch.no_grad()
def evict_and_generate(model, tokenizer, full_cache, prompt_len, reasoning_len,
                       evict_positions, method, num_layers, n_kv_heads,
                       head_selective_info=None, max_answer_tokens=64):
    """Apply eviction to cache and generate answer.

    method: 'mask', 'zero_kv', 'zero_v', 'head_selective'
    """
    total_len = prompt_len + reasoning_len
    device = model.device
    evicted_cache = DynamicCache()
    use_mask = (method == 'mask')

    if method == 'head_selective' and head_selective_info is not None:
        # Per-head zeroing (same as Exp 055)
        for l in range(num_layers):
            k = full_cache.layers[l].keys[:, :, :total_len, :].clone()
            v = full_cache.layers[l].values[:, :, :total_len, :].clone()
            for kv_h, keep_set in head_selective_info.items():
                for pos in range(prompt_len, total_len):
                    if pos not in keep_set:
                        k[:, kv_h, pos, :] = 0.0
                        v[:, kv_h, pos, :] = 0.0
            evicted_cache.update(k, v, l)
    else:
        # Pre-compute eviction mask tensor for vectorized operations
        evict_tensor = torch.zeros(total_len, dtype=torch.bool, device=device)
        if evict_positions:
            evict_idx = torch.tensor(evict_positions, dtype=torch.long, device=device)
            evict_tensor[evict_idx] = True

        for l in range(num_layers):
            k = full_cache.layers[l].keys[:, :, :total_len, :].clone()
            v = full_cache.layers[l].values[:, :, :total_len, :].clone()

            if method == 'zero_kv':
                k[:, :, evict_tensor, :] = 0.0
                v[:, :, evict_tensor, :] = 0.0
            elif method == 'zero_v':
                # Keep K intact, zero only V (phantom routing)
                v[:, :, evict_tensor, :] = 0.0
            # method == 'mask': don't modify cache

            evicted_cache.update(k, v, l)

    # Prepare attention mask for 'mask' method
    if use_mask:
        attn_mask = torch.ones(1, total_len, dtype=torch.long, device=device)
        evict_tensor_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
        if evict_positions:
            evict_idx = torch.tensor(evict_positions, dtype=torch.long, device=device)
            evict_tensor_mask[evict_idx] = True
        attn_mask[0, evict_tensor_mask] = 0
    else:
        attn_mask = None

    # Generate answer — feed " ####" prefix then decode
    answer_prefix = tokenizer.encode(" ####", add_special_tokens=False)
    cache_len = total_len  # Track position for explicit position_ids

    out = None
    for i in range(len(answer_prefix)):
        tok = torch.tensor([[answer_prefix[i]]], device=device)
        kwargs = {'input_ids': tok, 'past_key_values': evicted_cache, 'use_cache': True}
        if attn_mask is not None:
            # Extend mask for new token (always 1 = attend)
            attn_mask = torch.cat([attn_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
            kwargs['attention_mask'] = attn_mask
            # Explicit position_ids to prevent mask-based position recomputation
            kwargs['position_ids'] = torch.tensor([[cache_len]], device=device)
        out = model(**kwargs)
        evicted_cache = out.past_key_values
        cache_len += 1

    # Greedy decode answer tokens
    gen_tokens = []
    next_logits = out.logits[:, -1, :]
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))

    for step in range(max_answer_tokens):
        next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
        token_id = next_tok[0, 0].item()
        if token_id == tokenizer.eos_token_id or token_id in newline_ids:
            break
        gen_tokens.append(token_id)
        kwargs = {'input_ids': next_tok, 'past_key_values': evicted_cache, 'use_cache': True}
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
            kwargs['attention_mask'] = attn_mask
            kwargs['position_ids'] = torch.tensor([[cache_len]], device=device)
        out = model(**kwargs)
        evicted_cache = out.past_key_values
        next_logits = out.logits[:, -1, :]
        cache_len += 1

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
    """Generate figures comparing eviction methods."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    method_colors = {'mask': '#E91E63', 'zero_kv': '#2196F3', 'zero_v': '#9C27B0'}
    method_labels = {'mask': 'Mask (true removal)', 'zero_kv': 'Zero K+V', 'zero_v': 'Zero V only (K-preserve)'}
    strategy_markers = {'random': 'o', 'h2o': 's', 'recent': '^', 'head_selective': 'P'}

    for model_key, model_results in all_results.items():
        if not model_results:
            continue
        model_short = model_key.split("/")[-1]

        # Figure 1: Method comparison at 33% budget — grouped bar chart
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        strategies_33 = ['random', 'h2o', 'recent']
        methods_33 = ['mask', 'zero_kv', 'zero_v']
        x = np.arange(len(strategies_33))
        bar_width = 0.25

        for m_idx, method in enumerate(methods_33):
            accs = []
            errs_lo = []
            errs_hi = []
            for strategy in strategies_33:
                key = f"{method}_{strategy}_0.33"
                if key in model_results:
                    r = model_results[key]
                    acc = r['accuracy'] * 100
                    lo, hi = wilson_ci(r['n_correct'], r['n_valid'])
                    accs.append(acc)
                    errs_lo.append(acc - lo * 100)
                    errs_hi.append(hi * 100 - acc)
                else:
                    accs.append(0)
                    errs_lo.append(0)
                    errs_hi.append(0)

            bars = ax.bar(x + m_idx * bar_width, accs, bar_width,
                         label=method_labels[method],
                         color=method_colors[method], alpha=0.85)
            ax.errorbar(x + m_idx * bar_width, accs,
                       yerr=[errs_lo, errs_hi],
                       fmt='none', color='black', capsize=3)

        # Add head_selective (zero_kv only)
        hs_key = "zero_kv_head_selective_0.33"
        if hs_key in model_results:
            r = model_results[hs_key]
            acc = r['accuracy'] * 100
            lo, hi = wilson_ci(r['n_correct'], r['n_valid'])
            ax.bar(len(strategies_33) + 0.25, acc, bar_width,
                   color=method_colors['zero_kv'], alpha=0.85)
            ax.errorbar(len(strategies_33) + 0.25, acc,
                       yerr=[[acc - lo*100], [hi*100 - acc]],
                       fmt='none', color='black', capsize=3)
            strategies_33_labels = ['random', 'h2o', 'recent', 'head_sel\n(zero_kv)']
            ax.set_xticks(list(x + bar_width) + [len(strategies_33) + 0.25])
            ax.set_xticklabels(strategies_33_labels, fontsize=12)
        else:
            ax.set_xticks(x + bar_width)
            ax.set_xticklabels(strategies_33, fontsize=12)

        ax.set_ylabel('Answer Accuracy (%)', fontsize=13)
        ax.set_title(f'Eviction Method Comparison at 33% Budget — {model_short}', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'method_comparison_33pct_{model_short}.png'), dpi=150)
        plt.close()

        # Figure 2: Mask vs Zero_KV comparison (paired) at both budgets
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax_idx, budget in enumerate([0.33, 0.50]):
            ax = axes[ax_idx]
            strategies_to_plot = ['h2o', 'recent']
            x = np.arange(len(strategies_to_plot))
            bar_width = 0.35

            for m_idx, method in enumerate(['mask', 'zero_kv']):
                accs = []
                errs_lo = []
                errs_hi = []
                for strategy in strategies_to_plot:
                    key = f"{method}_{strategy}_{budget:.2f}"
                    if key in model_results:
                        r = model_results[key]
                        acc = r['accuracy'] * 100
                        lo, hi = wilson_ci(r['n_correct'], r['n_valid'])
                        accs.append(acc)
                        errs_lo.append(acc - lo * 100)
                        errs_hi.append(hi * 100 - acc)
                    else:
                        accs.append(0)
                        errs_lo.append(0)
                        errs_hi.append(0)

                ax.bar(x + m_idx * bar_width, accs, bar_width,
                       label=method_labels[method],
                       color=method_colors[method], alpha=0.85)
                ax.errorbar(x + m_idx * bar_width, accs,
                           yerr=[errs_lo, errs_hi],
                           fmt='none', color='black', capsize=4)

            ax.set_xticks(x + bar_width / 2)
            ax.set_xticklabels(strategies_to_plot, fontsize=12)
            ax.set_ylabel('Answer Accuracy (%)', fontsize=12)
            ax.set_title(f'{int(budget*100)}% Budget', fontsize=13)
            ax.set_ylim(0, 110)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=10)

        fig.suptitle(f'Mask vs Zero: {model_short}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'mask_vs_zero_{model_short}.png'), dpi=150)
        plt.close()

        # Figure 3: Phantom routing test — 3-way comparison for h2o at 33%
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        methods_pr = ['mask', 'zero_kv', 'zero_v']
        pr_accs = []
        pr_errs = [[], []]
        pr_colors = []

        for method in methods_pr:
            key = f"{method}_h2o_0.33"
            if key in model_results:
                r = model_results[key]
                acc = r['accuracy'] * 100
                lo, hi = wilson_ci(r['n_correct'], r['n_valid'])
                pr_accs.append(acc)
                pr_errs[0].append(acc - lo * 100)
                pr_errs[1].append(hi * 100 - acc)
                pr_colors.append(method_colors[method])
            else:
                pr_accs.append(0)
                pr_errs[0].append(0)
                pr_errs[1].append(0)
                pr_colors.append('#999')

        x = np.arange(len(methods_pr))
        ax.bar(x, pr_accs, color=pr_colors, alpha=0.85, width=0.5)
        ax.errorbar(x, pr_accs, yerr=pr_errs, fmt='none', color='black', capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([method_labels[m] for m in methods_pr], fontsize=10)
        ax.set_ylabel('Answer Accuracy (%)', fontsize=13)
        ax.set_title(f'Phantom Routing Test (H2O, 33%) — {model_short}', fontsize=14)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'phantom_routing_test_{model_short}.png'), dpi=150)
        plt.close()

    print(f"Figures saved to {results_dir}")


def run_model(model_name, problems, time_limit):
    """Run all conditions on one model."""
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

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
    n_attn_heads = model.config.num_attention_heads

    print(f"  Layers: {num_layers}, KV heads: {n_kv_heads}, Attn heads: {n_attn_heads}")

    # Answer heads based on model
    if "qwen" in model_name.lower():
        answer_heads = (0, 5)
    elif "llama" in model_name.lower():
        answer_heads = (3, 5)
    else:
        answer_heads = (0, 5)

    results = {}
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

        try:
            (gen_text, input_ids, gen_ids, importance, per_head_importance,
             prompt_len, full_cache, full_seq, trunc_point) = generate_and_build_cache(
                model, tokenizer, prompt)
        except Exception as e:
            print(f"  Problem {prob_idx}: generation error: {e}")
            continue

        n_attempted += 1

        clean_answer = normalize_answer(extract_answer(gen_text))
        if clean_answer != true_answer:
            if full_cache is not None:
                del full_cache
                gc.collect(); torch.cuda.empty_cache()
            continue

        if full_cache is None or trunc_point is None:
            continue

        reasoning_len = trunc_point
        n_valid += 1

        prob_results = {"idx": prob_idx, "true_answer": true_answer,
                        "clean_answer": clean_answer, "reasoning_len": reasoning_len}

        # Test each condition
        for method, strategy, budget in CONDITIONS:
            key = f"{method}_{strategy}_{budget:.2f}"

            try:
                sel = select_positions(
                    strategy, budget, reasoning_len, prompt_len,
                    importance, per_head_importance, n_kv_heads, answer_heads)

                if isinstance(sel, tuple) and sel[0] == 'head_selective':
                    _, head_positions = sel
                    answer_text = evict_and_generate(
                        model, tokenizer, full_cache, prompt_len, reasoning_len,
                        [], 'head_selective', num_layers, n_kv_heads,
                        head_selective_info=head_positions)
                else:
                    keep_pos, evict_pos = sel
                    answer_text = evict_and_generate(
                        model, tokenizer, full_cache, prompt_len, reasoning_len,
                        evict_pos, method, num_layers, n_kv_heads)

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
                prob_results[key] = {"correct": correct, "pred": pred, "raw": answer_text[:80]}

            except Exception as e:
                print(f"  Problem {prob_idx}, {key}: error: {e}")
                continue

        # Also test baseline (no eviction)
        try:
            baseline_text = evict_and_generate(
                model, tokenizer, full_cache, prompt_len, reasoning_len,
                [], 'zero_kv', num_layers, n_kv_heads)
            baseline_pred = normalize_answer(extract_answer(baseline_text))
            baseline_correct = (baseline_pred == true_answer)
            bl_key = "baseline_1.00"
            if bl_key not in results:
                results[bl_key] = {"n_correct": 0, "n_valid": 0, "answers": []}
            results[bl_key]["n_correct"] += int(baseline_correct)
            results[bl_key]["n_valid"] += 1
            prob_results["baseline"] = {"correct": baseline_correct, "pred": baseline_pred}
        except Exception as e:
            print(f"  Problem {prob_idx}, baseline: {e}")

        per_problem.append(prob_results)
        del full_cache
        gc.collect()
        torch.cuda.empty_cache()

        if n_valid % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Valid: {n_valid}, Attempted: {n_attempted}, "
                  f"Elapsed: {elapsed:.0f}s, "
                  f"Rate: {elapsed/max(1,n_valid):.1f}s/prob")

    # Compute accuracy
    for key in results:
        r = results[key]
        r["accuracy"] = r["n_correct"] / max(1, r["n_valid"])

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results, per_problem, n_valid, n_attempted


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = list(ds)
    random.shuffle(problems)

    print(f"Loaded {len(problems)} GSM8K problems")
    print(f"Conditions: {len(CONDITIONS)}")
    for method, strat, budget in CONDITIONS:
        print(f"  {method:8s} × {strat:16s} × {budget:.0%}")

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
        model_time = min(model_time, remaining - 60)

        model_results, per_problem, n_valid, n_attempted = run_model(
            model_name, problems, model_time)

        all_results[model_name] = model_results
        all_per_problem[model_name] = per_problem
        all_stats[model_name] = {
            "n_valid": n_valid, "n_attempted": n_attempted,
            "time_used": time.time() - global_start
        }

        # Print summary
        print(f"\n{'─'*90}")
        print(f"RESULTS: {model_name} (n_valid={n_valid})")
        print(f"{'─'*90}")
        print(f"{'Method':<10} {'Strategy':<16} {'Budget':<8} {'Acc%':<8} {'n_ok':<6} {'n':<6} {'Wilson 95% CI':<16}")
        print(f"{'─'*10} {'─'*16} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*16}")

        # Baseline
        bl_key = "baseline_1.00"
        if bl_key in model_results:
            r = model_results[bl_key]
            acc = r["accuracy"] * 100
            lo, hi = wilson_ci(r["n_correct"], r["n_valid"])
            print(f"{'baseline':<10} {'—':<16} {'100%':<8} {acc:>6.1f}%  {r['n_correct']:<6} {r['n_valid']:<6} [{lo*100:.1f}, {hi*100:.1f}]")

        # Group by budget
        for budget in [0.33, 0.50]:
            print(f"\n  --- {int(budget*100)}% budget ---")
            for method, strat, b in CONDITIONS:
                if b != budget:
                    continue
                key = f"{method}_{strat}_{b:.2f}"
                if key in model_results:
                    r = model_results[key]
                    acc = r["accuracy"] * 100
                    lo, hi = wilson_ci(r["n_correct"], r["n_valid"])
                    print(f"{method:<10} {strat:<16} {b*100:>5.0f}%   {acc:>6.1f}%  "
                          f"{r['n_correct']:<6} {r['n_valid']:<6} [{lo*100:.1f}, {hi*100:.1f}]")

    # Generate figures
    generate_figures(all_results, RESULTS_DIR)

    # Save results
    summary = {
        "experiment": "056_mask_vs_zero_eviction",
        "conditions": [(m, s, b) for m, s, b in CONDITIONS],
        "stats": all_stats,
    }

    summary_results = {}
    for model_name, model_results in all_results.items():
        summary_results[model_name] = {}
        for key, r in model_results.items():
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
