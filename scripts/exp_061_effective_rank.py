#!/usr/bin/env python3
"""
Experiment 061: Effective Rank Analysis of K vs V Cache

Computes the spectral properties (singular value decay, effective rank, condition
number) of K and V cache matrices across Qwen (digital) and Llama (analog) during
GSM8K CoT reasoning. Tests whether K-manifold dimensionality predicts encoding type.

Key metrics:
- Effective rank (Roy & Vetterli 2007): exp(Shannon entropy of normalized singular values)
- Participation ratio: (sum(s))^2 / (sum(s^2)) — how many singular values contribute
- Spectral gap: ratio of 1st to 2nd singular value
- Top-k energy: fraction of Frobenius norm captured by top k singular values
- Per-layer and per-head analysis
"""

import os
import json
import time
import gc
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
TIME_BUDGET = 1750  # seconds
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 2048
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_062")

MODELS = [
    "Qwen/Qwen3-4B-Base",
    "meta-llama/Llama-3.1-8B-Instruct",
]

N_PROBLEMS = 20  # number of GSM8K problems to analyze per model

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


def load_gsm8k():
    """Load GSM8K test set."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return ds


def extract_answer(text):
    """Extract numeric answer from #### pattern."""
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '')
    # Try to find first number after 'answer is'
    m = re.search(r'answer\s+is\s+\$?(-?[\d,]+(?:\.\d+)?)', text, re.I)
    if m:
        return m.group(1).replace(',', '')
    return None


def compute_spectral_metrics(matrix):
    """
    Compute spectral metrics for a 2D matrix (positions × head_dim).

    Args:
        matrix: torch.Tensor of shape (n_positions, head_dim)

    Returns:
        dict with spectral metrics
    """
    # Move to float32 for SVD stability
    M = matrix.float()
    n, d = M.shape

    if n < 2 or d < 2:
        return None

    # Compute SVD
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    except Exception:
        return None

    S = S.cpu().numpy()

    # Filter out near-zero singular values
    S = S[S > 1e-10]
    if len(S) < 2:
        return None

    # 1. Effective rank (Roy & Vetterli 2007)
    # = exp(Shannon entropy of normalized singular values)
    p = S / S.sum()
    entropy = -np.sum(p * np.log(p + 1e-30))
    effective_rank = np.exp(entropy)

    # 2. Participation ratio
    # = (sum(s))^2 / (sum(s^2))
    participation_ratio = (S.sum() ** 2) / (np.sum(S ** 2))

    # 3. Spectral gap: s1 / s2
    spectral_gap = S[0] / S[1] if S[1] > 1e-10 else float('inf')

    # 4. Top-k energy: fraction of total energy in top k singular values
    total_energy = np.sum(S ** 2)
    top1_energy = S[0] ** 2 / total_energy
    top5_energy = np.sum(S[:min(5, len(S))] ** 2) / total_energy
    top10_energy = np.sum(S[:min(10, len(S))] ** 2) / total_energy
    top20_energy = np.sum(S[:min(20, len(S))] ** 2) / total_energy

    # 5. Condition number
    condition_number = S[0] / S[-1] if S[-1] > 1e-10 else float('inf')

    # 6. Normalized effective rank (0-1, where 1 = fully distributed)
    max_rank = min(n, d)
    normalized_eff_rank = effective_rank / max_rank

    # 7. Singular value decay rate (slope of log(s) vs index)
    if len(S) >= 3:
        log_s = np.log(S + 1e-30)
        indices = np.arange(len(S))
        # Linear fit
        slope, intercept = np.polyfit(indices, log_s, 1)
        decay_rate = -slope  # positive = faster decay
    else:
        decay_rate = float('nan')

    return {
        'effective_rank': float(effective_rank),
        'normalized_eff_rank': float(normalized_eff_rank),
        'participation_ratio': float(participation_ratio),
        'spectral_gap': float(spectral_gap),
        'top1_energy': float(top1_energy),
        'top5_energy': float(top5_energy),
        'top10_energy': float(top10_energy),
        'top20_energy': float(top20_energy),
        'condition_number': float(min(condition_number, 1e10)),
        'decay_rate': float(decay_rate),
        'n_singular_values': int(len(S)),
        'max_rank': int(max_rank),
        'singular_values_top20': S[:20].tolist(),
    }


def analyze_kv_cache(model, tokenizer, question, max_gen=MAX_GEN_TOKENS, device='cuda'):
    """
    Generate CoT reasoning and extract per-layer, per-head K and V spectral metrics.

    Returns:
        dict with per-layer, per-head spectral metrics for K and V
    """
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    if prompt_len > MAX_SEQ_LEN - max_gen:
        return None

    # Generate with KV cache
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen,
            do_sample=False,
            temperature=1.0,
            return_dict_in_generate=True,
            use_cache=True,
        )

    generated_ids = output.sequences[0][prompt_len:]
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract answer
    answer = extract_answer(gen_text)

    # Now do a forward pass to get the KV cache for the full sequence
    full_ids = output.sequences[0:1]  # (1, seq_len)
    total_len = full_ids.shape[1]
    reasoning_len = total_len - prompt_len

    if reasoning_len < 10:
        return None

    with torch.no_grad():
        out = model(full_ids, use_cache=True)

    past_kv = out.past_key_values

    # Handle DynamicCache (transformers 5.x uses .layers[i].keys/.values)
    from transformers import DynamicCache
    if isinstance(past_kv, DynamicCache):
        # transformers 5.x: DynamicCache uses .layers list
        if hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
            n_layers = len(past_kv.layers)
            sample_k = past_kv.layers[0].keys
        elif hasattr(past_kv, 'key_cache'):
            n_layers = len(past_kv.key_cache)
            sample_k = past_kv.key_cache[0]
        else:
            return None
    else:
        n_layers = len(past_kv)
        sample_k = past_kv[0][0]

    n_kv_heads = sample_k.shape[1]
    head_dim = sample_k.shape[3]

    results = {
        'answer': answer,
        'gen_text_len': len(gen_text),
        'reasoning_tokens': int(reasoning_len),
        'prompt_tokens': int(prompt_len),
        'n_layers': n_layers,
        'n_kv_heads': n_kv_heads,
        'head_dim': head_dim,
        'per_layer': {},
        'per_head': {},
    }

    # Analyze per-layer (all heads concatenated) and per-head spectral properties
    for layer_idx in range(n_layers):
        if isinstance(past_kv, DynamicCache):
            if hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
                k = past_kv.layers[layer_idx].keys    # (1, n_kv_heads, seq_len, head_dim)
                v = past_kv.layers[layer_idx].values
            else:
                k = past_kv.key_cache[layer_idx]
                v = past_kv.value_cache[layer_idx]
        else:
            k = past_kv[layer_idx][0]
            v = past_kv[layer_idx][1]

        # Extract reasoning-only portion (skip prompt)
        k_reasoning = k[0, :, prompt_len:, :]  # (n_kv_heads, reasoning_len, head_dim)
        v_reasoning = v[0, :, prompt_len:, :]

        # Per-layer analysis: concatenate all heads → (reasoning_len, n_kv_heads * head_dim)
        k_layer = k_reasoning.permute(1, 0, 2).reshape(reasoning_len, -1)  # (reasoning_len, n_kv_heads * head_dim)
        v_layer = v_reasoning.permute(1, 0, 2).reshape(reasoning_len, -1)

        k_metrics = compute_spectral_metrics(k_layer)
        v_metrics = compute_spectral_metrics(v_layer)

        if k_metrics and v_metrics:
            results['per_layer'][str(layer_idx)] = {
                'K': k_metrics,
                'V': v_metrics,
            }

        # Per-head analysis (sample every 6th layer to save time)
        if layer_idx % 6 == 0:
            for head_idx in range(n_kv_heads):
                k_head = k_reasoning[head_idx]  # (reasoning_len, head_dim)
                v_head = v_reasoning[head_idx]

                k_head_metrics = compute_spectral_metrics(k_head)
                v_head_metrics = compute_spectral_metrics(v_head)

                if k_head_metrics and v_head_metrics:
                    key = f"L{layer_idx}_H{head_idx}"
                    results['per_head'][key] = {
                        'K': k_head_metrics,
                        'V': v_head_metrics,
                    }

    # Clean up
    del past_kv, out, output
    gc.collect()
    torch.cuda.empty_cache()

    return results


def aggregate_results(all_results):
    """Aggregate spectral metrics across problems."""

    # Per-layer aggregation
    layer_agg = {}
    for prob_result in all_results:
        for layer_str, layer_data in prob_result['per_layer'].items():
            if layer_str not in layer_agg:
                layer_agg[layer_str] = {'K': [], 'V': []}
            for kv_type in ['K', 'V']:
                layer_agg[layer_str][kv_type].append(layer_data[kv_type])

    layer_summary = {}
    for layer_str in sorted(layer_agg.keys(), key=int):
        layer_summary[layer_str] = {}
        for kv_type in ['K', 'V']:
            metrics_list = layer_agg[layer_str][kv_type]
            if not metrics_list:
                continue
            summary = {}
            for metric_name in ['effective_rank', 'normalized_eff_rank', 'participation_ratio',
                               'spectral_gap', 'top1_energy', 'top5_energy', 'top10_energy',
                               'top20_energy', 'condition_number', 'decay_rate']:
                vals = [m[metric_name] for m in metrics_list
                       if not (isinstance(m[metric_name], float) and (np.isnan(m[metric_name]) or np.isinf(m[metric_name])))]
                if vals:
                    summary[metric_name] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'median': float(np.median(vals)),
                        'min': float(np.min(vals)),
                        'max': float(np.max(vals)),
                    }
            layer_summary[layer_str][kv_type] = summary

    # Per-head aggregation
    head_agg = {}
    for prob_result in all_results:
        for head_key, head_data in prob_result['per_head'].items():
            if head_key not in head_agg:
                head_agg[head_key] = {'K': [], 'V': []}
            for kv_type in ['K', 'V']:
                head_agg[head_key][kv_type].append(head_data[kv_type])

    head_summary = {}
    for head_key in sorted(head_agg.keys()):
        head_summary[head_key] = {}
        for kv_type in ['K', 'V']:
            metrics_list = head_agg[head_key][kv_type]
            if not metrics_list:
                continue
            summary = {}
            for metric_name in ['effective_rank', 'normalized_eff_rank', 'participation_ratio',
                               'spectral_gap', 'top1_energy', 'top5_energy', 'top10_energy',
                               'decay_rate']:
                vals = [m[metric_name] for m in metrics_list
                       if not (isinstance(m[metric_name], float) and (np.isnan(m[metric_name]) or np.isinf(m[metric_name])))]
                if vals:
                    summary[metric_name] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                    }
            head_summary[head_key][kv_type] = summary

    # Global K vs V comparison
    all_k_eff_rank = []
    all_v_eff_rank = []
    all_k_norm_eff_rank = []
    all_v_norm_eff_rank = []
    all_k_decay = []
    all_v_decay = []
    all_k_top1 = []
    all_v_top1 = []
    all_k_top10 = []
    all_v_top10 = []
    all_k_spectral_gap = []
    all_v_spectral_gap = []

    for prob_result in all_results:
        for layer_data in prob_result['per_layer'].values():
            all_k_eff_rank.append(layer_data['K']['effective_rank'])
            all_v_eff_rank.append(layer_data['V']['effective_rank'])
            all_k_norm_eff_rank.append(layer_data['K']['normalized_eff_rank'])
            all_v_norm_eff_rank.append(layer_data['V']['normalized_eff_rank'])
            if not np.isnan(layer_data['K']['decay_rate']):
                all_k_decay.append(layer_data['K']['decay_rate'])
            if not np.isnan(layer_data['V']['decay_rate']):
                all_v_decay.append(layer_data['V']['decay_rate'])
            all_k_top1.append(layer_data['K']['top1_energy'])
            all_v_top1.append(layer_data['V']['top1_energy'])
            all_k_top10.append(layer_data['K']['top10_energy'])
            all_v_top10.append(layer_data['V']['top10_energy'])
            if not np.isinf(layer_data['K']['spectral_gap']):
                all_k_spectral_gap.append(layer_data['K']['spectral_gap'])
            if not np.isinf(layer_data['V']['spectral_gap']):
                all_v_spectral_gap.append(layer_data['V']['spectral_gap'])

    global_summary = {
        'K_effective_rank': {'mean': float(np.mean(all_k_eff_rank)), 'std': float(np.std(all_k_eff_rank)), 'median': float(np.median(all_k_eff_rank))},
        'V_effective_rank': {'mean': float(np.mean(all_v_eff_rank)), 'std': float(np.std(all_v_eff_rank)), 'median': float(np.median(all_v_eff_rank))},
        'K_normalized_eff_rank': {'mean': float(np.mean(all_k_norm_eff_rank)), 'std': float(np.std(all_k_norm_eff_rank))},
        'V_normalized_eff_rank': {'mean': float(np.mean(all_v_norm_eff_rank)), 'std': float(np.std(all_v_norm_eff_rank))},
        'K_decay_rate': {'mean': float(np.mean(all_k_decay)), 'std': float(np.std(all_k_decay))} if all_k_decay else {},
        'V_decay_rate': {'mean': float(np.mean(all_v_decay)), 'std': float(np.std(all_v_decay))} if all_v_decay else {},
        'K_top1_energy': {'mean': float(np.mean(all_k_top1)), 'std': float(np.std(all_k_top1))},
        'V_top1_energy': {'mean': float(np.mean(all_v_top1)), 'std': float(np.std(all_v_top1))},
        'K_top10_energy': {'mean': float(np.mean(all_k_top10)), 'std': float(np.std(all_k_top10))},
        'V_top10_energy': {'mean': float(np.mean(all_v_top10)), 'std': float(np.std(all_v_top10))},
        'K_spectral_gap': {'mean': float(np.mean(all_k_spectral_gap)), 'std': float(np.std(all_k_spectral_gap))} if all_k_spectral_gap else {},
        'V_spectral_gap': {'mean': float(np.mean(all_v_spectral_gap)), 'std': float(np.std(all_v_spectral_gap))} if all_v_spectral_gap else {},
        'K_vs_V_eff_rank_ratio': float(np.mean(all_k_eff_rank) / np.mean(all_v_eff_rank)) if np.mean(all_v_eff_rank) > 0 else float('inf'),
        'n_layer_observations': len(all_k_eff_rank),
    }

    return {
        'per_layer': layer_summary,
        'per_head': head_summary,
        'global': global_summary,
    }


def generate_figures(model_results, model_name, results_dir):
    """Generate visualization figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    short_name = model_name.split('/')[-1]

    # Figure 1: Per-layer effective rank K vs V
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = sorted(model_results['per_layer'].keys(), key=int)
    k_eff_ranks = [model_results['per_layer'][l]['K']['effective_rank']['mean'] for l in layers]
    v_eff_ranks = [model_results['per_layer'][l]['V']['effective_rank']['mean'] for l in layers]
    k_eff_rank_std = [model_results['per_layer'][l]['K']['effective_rank']['std'] for l in layers]
    v_eff_rank_std = [model_results['per_layer'][l]['V']['effective_rank']['std'] for l in layers]
    layer_nums = [int(l) for l in layers]

    axes[0].errorbar(layer_nums, k_eff_ranks, yerr=k_eff_rank_std, label='K cache', marker='o', capsize=3, markersize=4)
    axes[0].errorbar(layer_nums, v_eff_ranks, yerr=v_eff_rank_std, label='V cache', marker='s', capsize=3, markersize=4)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Effective Rank')
    axes[0].set_title(f'{short_name}: Effective Rank per Layer')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # K/V ratio per layer
    ratios = [k/v if v > 0 else 0 for k, v in zip(k_eff_ranks, v_eff_ranks)]
    axes[1].bar(layer_nums, ratios, color=['#e74c3c' if r < 1 else '#2ecc71' for r in ratios], alpha=0.7)
    axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='K=V')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('K/V Effective Rank Ratio')
    axes[1].set_title(f'{short_name}: K/V Rank Ratio per Layer')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'effective_rank_per_layer_{short_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Top-k energy decay (how fast singular values decay)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_top1 = [model_results['per_layer'][l]['K']['top1_energy']['mean'] for l in layers]
    v_top1 = [model_results['per_layer'][l]['V']['top1_energy']['mean'] for l in layers]
    k_top10 = [model_results['per_layer'][l]['K']['top10_energy']['mean'] for l in layers]
    v_top10 = [model_results['per_layer'][l]['V']['top10_energy']['mean'] for l in layers]

    axes[0].plot(layer_nums, k_top1, 'o-', label='K top-1', markersize=4)
    axes[0].plot(layer_nums, v_top1, 's-', label='V top-1', markersize=4)
    axes[0].plot(layer_nums, k_top10, 'o--', label='K top-10', markersize=4, alpha=0.7)
    axes[0].plot(layer_nums, v_top10, 's--', label='V top-10', markersize=4, alpha=0.7)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Fraction of Total Energy')
    axes[0].set_title(f'{short_name}: Energy Concentration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Decay rate per layer
    k_decay = [model_results['per_layer'][l]['K']['decay_rate']['mean'] for l in layers
               if 'decay_rate' in model_results['per_layer'][l]['K']]
    v_decay = [model_results['per_layer'][l]['V']['decay_rate']['mean'] for l in layers
               if 'decay_rate' in model_results['per_layer'][l]['V']]
    decay_layers_k = [int(l) for l in layers if 'decay_rate' in model_results['per_layer'][l]['K']]
    decay_layers_v = [int(l) for l in layers if 'decay_rate' in model_results['per_layer'][l]['V']]

    axes[1].plot(decay_layers_k, k_decay, 'o-', label='K decay rate', markersize=4)
    axes[1].plot(decay_layers_v, v_decay, 's-', label='V decay rate', markersize=4)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Decay Rate (steeper = more concentrated)')
    axes[1].set_title(f'{short_name}: Singular Value Decay Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'energy_decay_{short_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Spectral gap per layer
    fig, ax = plt.subplots(figsize=(10, 5))

    k_gap = [model_results['per_layer'][l]['K']['spectral_gap']['mean'] for l in layers
             if 'spectral_gap' in model_results['per_layer'][l]['K']]
    v_gap = [model_results['per_layer'][l]['V']['spectral_gap']['mean'] for l in layers
             if 'spectral_gap' in model_results['per_layer'][l]['V']]
    gap_layers_k = [int(l) for l in layers if 'spectral_gap' in model_results['per_layer'][l]['K']]
    gap_layers_v = [int(l) for l in layers if 'spectral_gap' in model_results['per_layer'][l]['V']]

    ax.plot(gap_layers_k, k_gap, 'o-', label='K spectral gap (s1/s2)', markersize=4)
    ax.plot(gap_layers_v, v_gap, 's-', label='V spectral gap (s1/s2)', markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Spectral Gap (s1/s2)')
    ax.set_title(f'{short_name}: Spectral Gap per Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'spectral_gap_{short_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return 3  # number of figures


def generate_comparison_figure(all_model_results, results_dir):
    """Generate cross-model comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model_names = list(all_model_results.keys())
    if len(model_names) < 2:
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']

    for idx, model_name in enumerate(model_names):
        short_name = model_name.split('/')[-1]
        data = all_model_results[model_name]

        layers = sorted(data['per_layer'].keys(), key=int)
        layer_nums = [int(l) for l in layers]

        # K effective rank
        k_eff = [data['per_layer'][l]['K']['effective_rank']['mean'] for l in layers]
        axes[0, 0].plot(layer_nums, k_eff, 'o-', label=short_name, color=colors[idx], markersize=3)

        # V effective rank
        v_eff = [data['per_layer'][l]['V']['effective_rank']['mean'] for l in layers]
        axes[0, 1].plot(layer_nums, v_eff, 's-', label=short_name, color=colors[idx], markersize=3)

        # K/V ratio
        ratios = [k/v if v > 0 else 0 for k, v in zip(k_eff, v_eff)]
        axes[1, 0].plot(layer_nums, ratios, 'o-', label=short_name, color=colors[idx], markersize=3)

        # K top-1 energy
        k_top1 = [data['per_layer'][l]['K']['top1_energy']['mean'] for l in layers]
        v_top1 = [data['per_layer'][l]['V']['top1_energy']['mean'] for l in layers]
        axes[1, 1].plot(layer_nums, k_top1, 'o-', label=f'{short_name} K', color=colors[idx], markersize=3)
        axes[1, 1].plot(layer_nums, v_top1, 's--', label=f'{short_name} V', color=colors[idx], markersize=3, alpha=0.5)

    axes[0, 0].set_title('K Cache Effective Rank')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Effective Rank')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('V Cache Effective Rank')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Effective Rank')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('K/V Effective Rank Ratio')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Ratio (K/V)')
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Top-1 Singular Value Energy')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Fraction of Total Energy')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Cross-Model K/V Cache Spectral Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cross_model_spectral_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return 1


def main():
    start_time = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load GSM8K
    print("Loading GSM8K...")
    ds = load_gsm8k()

    # Shuffle and select problems
    indices = list(range(len(ds)))
    np.random.shuffle(indices)

    all_model_results = {}
    all_model_raw = {}

    for model_name in MODELS:
        model_start = time.time()
        elapsed_total = model_start - start_time
        remaining = TIME_BUDGET - elapsed_total

        if remaining < 120:
            print(f"\n⚠️  Time budget nearly exhausted ({remaining:.0f}s remaining). Skipping {model_name}.")
            break

        short_name = model_name.split('/')[-1]
        print(f"\n{'='*60}")
        print(f"Model: {short_name}")
        print(f"Elapsed: {elapsed_total:.0f}s, Remaining: {remaining:.0f}s")
        print(f"{'='*60}")

        # Load model
        print(f"Loading {short_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        prob_results = []
        prob_idx = 0
        attempt = 0

        per_model_budget = remaining / max(1, len(MODELS) - list(all_model_results.keys()).__len__())

        while len(prob_results) < N_PROBLEMS and attempt < N_PROBLEMS * 3:
            if time.time() - model_start > per_model_budget:
                print(f"  ⏰ Model time budget reached ({per_model_budget:.0f}s)")
                break

            idx = indices[attempt % len(indices)]
            attempt += 1

            question = ds[idx]['question']
            true_answer = extract_answer(ds[idx]['answer'])

            try:
                result = analyze_kv_cache(model, tokenizer, question)
                if result is None:
                    continue

                # Check if model got the right answer
                if result['answer'] and true_answer:
                    try:
                        correct = abs(float(result['answer']) - float(true_answer)) < 0.01
                    except (ValueError, TypeError):
                        correct = result['answer'].strip() == true_answer.strip()
                else:
                    correct = False

                if not correct:
                    continue  # Only analyze problems where model is correct

                result['question'] = question
                result['true_answer'] = true_answer
                prob_results.append(result)

                n_valid = len(prob_results)
                k_eff = np.mean([r['per_layer'][list(r['per_layer'].keys())[len(r['per_layer'])//2]]['K']['effective_rank']
                                for r in prob_results])
                v_eff = np.mean([r['per_layer'][list(r['per_layer'].keys())[len(r['per_layer'])//2]]['V']['effective_rank']
                                for r in prob_results])

                print(f"  [{n_valid}/{N_PROBLEMS}] idx={idx}, tokens={result['reasoning_tokens']}, "
                      f"K_eff_rank_mid={k_eff:.1f}, V_eff_rank_mid={v_eff:.1f}, "
                      f"time={time.time()-model_start:.0f}s")

            except Exception as e:
                print(f"  ❌ Error on problem {idx}: {e}")
                continue

        print(f"\n{short_name}: {len(prob_results)} valid problems in {time.time()-model_start:.0f}s")

        # Aggregate
        if prob_results:
            agg = aggregate_results(prob_results)
            all_model_results[model_name] = agg
            all_model_raw[model_name] = prob_results

            # Print summary
            g = agg['global']
            print(f"\n--- {short_name} Global Summary ---")
            print(f"  K effective rank: {g['K_effective_rank']['mean']:.2f} ± {g['K_effective_rank']['std']:.2f} (median {g['K_effective_rank']['median']:.2f})")
            print(f"  V effective rank: {g['V_effective_rank']['mean']:.2f} ± {g['V_effective_rank']['std']:.2f} (median {g['V_effective_rank']['median']:.2f})")
            print(f"  K/V eff rank ratio: {g['K_vs_V_eff_rank_ratio']:.3f}")
            print(f"  K normalized eff rank: {g['K_normalized_eff_rank']['mean']:.4f} ± {g['K_normalized_eff_rank']['std']:.4f}")
            print(f"  V normalized eff rank: {g['V_normalized_eff_rank']['mean']:.4f} ± {g['V_normalized_eff_rank']['std']:.4f}")
            if g['K_decay_rate']:
                print(f"  K decay rate: {g['K_decay_rate']['mean']:.4f} ± {g['K_decay_rate']['std']:.4f}")
            if g['V_decay_rate']:
                print(f"  V decay rate: {g['V_decay_rate']['mean']:.4f} ± {g['V_decay_rate']['std']:.4f}")
            print(f"  K top-1 energy: {g['K_top1_energy']['mean']:.4f} ± {g['K_top1_energy']['std']:.4f}")
            print(f"  V top-1 energy: {g['V_top1_energy']['mean']:.4f} ± {g['V_top1_energy']['std']:.4f}")
            print(f"  K top-10 energy: {g['K_top10_energy']['mean']:.4f} ± {g['K_top10_energy']['std']:.4f}")
            print(f"  V top-10 energy: {g['V_top10_energy']['mean']:.4f} ± {g['V_top10_energy']['std']:.4f}")
            if g['K_spectral_gap']:
                print(f"  K spectral gap: {g['K_spectral_gap']['mean']:.4f} ± {g['K_spectral_gap']['std']:.4f}")
            if g['V_spectral_gap']:
                print(f"  V spectral gap: {g['V_spectral_gap']['mean']:.4f} ± {g['V_spectral_gap']['std']:.4f}")
            print(f"  N layer observations: {g['n_layer_observations']}")

            # Generate per-model figures
            n_figs = generate_figures(agg, model_name, RESULTS_DIR)
            print(f"  Generated {n_figs} figures for {short_name}")

            # Save per-model results
            save_data = {
                'model': model_name,
                'n_problems': len(prob_results),
                'aggregate': agg,
            }
            with open(os.path.join(RESULTS_DIR, f'spectral_results_{short_name}.json'), 'w') as f:
                json.dump(save_data, f, indent=2)

        # Unload model
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-model comparison figure
    if len(all_model_results) >= 2:
        n_comp = generate_comparison_figure(all_model_results, RESULTS_DIR)
        print(f"\nGenerated {n_comp} comparison figure(s)")

    # Save combined results
    combined = {
        'models': {},
        'cross_model_comparison': {},
    }
    for model_name, agg in all_model_results.items():
        short = model_name.split('/')[-1]
        combined['models'][short] = agg['global']

    # Cross-model comparison
    if len(all_model_results) >= 2:
        model_list = list(all_model_results.keys())
        m1, m2 = model_list[0], model_list[1]
        s1, s2 = m1.split('/')[-1], m2.split('/')[-1]
        g1, g2 = all_model_results[m1]['global'], all_model_results[m2]['global']

        combined['cross_model_comparison'] = {
            f'{s1}_K_eff_rank': g1['K_effective_rank']['mean'],
            f'{s2}_K_eff_rank': g2['K_effective_rank']['mean'],
            f'{s1}_V_eff_rank': g1['V_effective_rank']['mean'],
            f'{s2}_V_eff_rank': g2['V_effective_rank']['mean'],
            f'{s1}_K_vs_V_ratio': g1['K_vs_V_eff_rank_ratio'],
            f'{s2}_K_vs_V_ratio': g2['K_vs_V_eff_rank_ratio'],
            f'{s1}_K_decay': g1['K_decay_rate'].get('mean', None),
            f'{s2}_K_decay': g2['K_decay_rate'].get('mean', None),
            f'{s1}_K_top1': g1['K_top1_energy']['mean'],
            f'{s2}_K_top1': g2['K_top1_energy']['mean'],
            'K_rank_lower_on_digital': g1['K_effective_rank']['mean'] < g2['K_effective_rank']['mean'],
        }

    with open(os.path.join(RESULTS_DIR, 'combined_results.json'), 'w') as f:
        json.dump(combined, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Models analyzed: {len(all_model_results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
