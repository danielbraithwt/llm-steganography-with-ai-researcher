#!/usr/bin/env python3
"""
Experiment 064: Per-Head Spectral Analysis — Are Answer Heads Geometrically Distinct?

Computes spectral properties of K and V cache at the individual HEAD level for ALL
layers, then compares answer heads (H0, H5) vs dispensable heads (H1-4, H6-7).

Tests whether the spectral geometry (Exp 062-063) connects to head specialization
(Exp 045-048): do answer heads have distinct K-cache spectral properties?
"""

import os
import json
import time
import gc
import re
import sys

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
TIME_BUDGET = 1700  # seconds (conservative — leave buffer for analysis/figures)
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 2048
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_064")

MODELS = [
    "Qwen/Qwen3-4B-Base",
    "meta-llama/Llama-3.1-8B-Instruct",
]

N_PROBLEMS = 20  # per model
ANSWER_HEADS = [0, 5]  # H0 and H5 (established in Exp 045-046)
DISPENSABLE_HEADS = [1, 2, 3, 4, 6, 7]  # all others

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
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return ds


def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '')
    m = re.search(r'answer\s+is\s+\$?(-?[\d,]+(?:\.\d+)?)', text, re.I)
    if m:
        return m.group(1).replace(',', '')
    return None


def compute_spectral_metrics(matrix):
    """Compute spectral metrics for a 2D matrix (positions x head_dim)."""
    M = matrix.float()
    n, d = M.shape
    if n < 2 or d < 2:
        return None
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    except Exception:
        return None

    S = S.cpu().numpy()
    S = S[S > 1e-10]
    if len(S) < 2:
        return None

    # Effective rank
    p = S / S.sum()
    entropy = -np.sum(p * np.log(p + 1e-30))
    effective_rank = np.exp(entropy)

    # Top-1 energy
    total_energy = np.sum(S ** 2)
    top1_energy = S[0] ** 2 / total_energy
    top10_energy = np.sum(S[:min(10, len(S))] ** 2) / total_energy

    # Spectral gap
    spectral_gap = S[0] / S[1] if S[1] > 1e-10 else float('inf')

    # Normalized effective rank
    max_rank = min(n, d)
    normalized_eff_rank = effective_rank / max_rank

    # Decay rate
    if len(S) >= 3:
        log_s = np.log(S + 1e-30)
        indices = np.arange(len(S))
        slope, _ = np.polyfit(indices, log_s, 1)
        decay_rate = -slope
    else:
        decay_rate = float('nan')

    return {
        'effective_rank': float(effective_rank),
        'normalized_eff_rank': float(normalized_eff_rank),
        'spectral_gap': float(spectral_gap),
        'top1_energy': float(top1_energy),
        'top10_energy': float(top10_energy),
        'decay_rate': float(decay_rate),
        'n_singular_values': int(len(S)),
        'max_rank': int(max_rank),
    }


def analyze_per_head(model, tokenizer, question, max_gen=MAX_GEN_TOKENS, device='cuda'):
    """Generate CoT and extract per-head spectral metrics at ALL layers."""
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    if prompt_len > MAX_SEQ_LEN - max_gen:
        return None

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
    answer = extract_answer(gen_text)

    # Full forward pass for KV cache
    full_ids = output.sequences[0:1]
    total_len = full_ids.shape[1]
    reasoning_len = total_len - prompt_len

    if reasoning_len < 10:
        return None

    with torch.no_grad():
        out = model(full_ids, use_cache=True)

    past_kv = out.past_key_values

    # Handle DynamicCache
    from transformers import DynamicCache
    if isinstance(past_kv, DynamicCache):
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
        'reasoning_tokens': int(reasoning_len),
        'n_layers': n_layers,
        'n_kv_heads': n_kv_heads,
        'head_dim': head_dim,
        'per_head': {},  # key: "L{layer}_H{head}" → {K: metrics, V: metrics}
    }

    for layer_idx in range(n_layers):
        if isinstance(past_kv, DynamicCache):
            if hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
                k = past_kv.layers[layer_idx].keys
                v = past_kv.layers[layer_idx].values
            else:
                k = past_kv.key_cache[layer_idx]
                v = past_kv.value_cache[layer_idx]
        else:
            k = past_kv[layer_idx][0]
            v = past_kv[layer_idx][1]

        # Reasoning-only portion
        k_reasoning = k[0, :, prompt_len:, :]  # (n_kv_heads, reasoning_len, head_dim)
        v_reasoning = v[0, :, prompt_len:, :]

        for head_idx in range(n_kv_heads):
            k_head = k_reasoning[head_idx]  # (reasoning_len, head_dim)
            v_head = v_reasoning[head_idx]

            k_metrics = compute_spectral_metrics(k_head)
            v_metrics = compute_spectral_metrics(v_head)

            if k_metrics and v_metrics:
                key = f"L{layer_idx}_H{head_idx}"
                results['per_head'][key] = {
                    'K': k_metrics,
                    'V': v_metrics,
                }

    del past_kv, out, output
    gc.collect()
    torch.cuda.empty_cache()

    return results


def analyze_head_groups(all_results, n_kv_heads, n_layers):
    """Analyze spectral differences between answer and dispensable heads."""
    # Collect per-head metrics across all layers and problems
    head_metrics = {h: {'K_eff_rank': [], 'K_top1': [], 'K_gap': [],
                        'V_eff_rank': [], 'V_top1': [], 'V_gap': []}
                    for h in range(n_kv_heads)}

    # Also collect paired observations for Wilcoxon (answer_mean vs dispensable_mean per layer×problem)
    paired_answer_k_eff = []
    paired_disp_k_eff = []
    paired_answer_k_top1 = []
    paired_disp_k_top1 = []
    paired_answer_v_eff = []
    paired_disp_v_eff = []
    paired_answer_v_top1 = []
    paired_disp_v_top1 = []

    for prob_result in all_results:
        for layer_idx in range(n_layers):
            ans_k_effs = []
            disp_k_effs = []
            ans_k_top1s = []
            disp_k_top1s = []
            ans_v_effs = []
            disp_v_effs = []
            ans_v_top1s = []
            disp_v_top1s = []

            for head_idx in range(n_kv_heads):
                key = f"L{layer_idx}_H{head_idx}"
                if key not in prob_result['per_head']:
                    continue
                data = prob_result['per_head'][key]

                # Collect per-head
                head_metrics[head_idx]['K_eff_rank'].append(data['K']['effective_rank'])
                head_metrics[head_idx]['K_top1'].append(data['K']['top1_energy'])
                head_metrics[head_idx]['K_gap'].append(data['K']['spectral_gap'])
                head_metrics[head_idx]['V_eff_rank'].append(data['V']['effective_rank'])
                head_metrics[head_idx]['V_top1'].append(data['V']['top1_energy'])
                head_metrics[head_idx]['V_gap'].append(data['V']['spectral_gap'])

                # Group for paired test
                if head_idx in ANSWER_HEADS:
                    ans_k_effs.append(data['K']['effective_rank'])
                    ans_k_top1s.append(data['K']['top1_energy'])
                    ans_v_effs.append(data['V']['effective_rank'])
                    ans_v_top1s.append(data['V']['top1_energy'])
                else:
                    disp_k_effs.append(data['K']['effective_rank'])
                    disp_k_top1s.append(data['K']['top1_energy'])
                    disp_v_effs.append(data['V']['effective_rank'])
                    disp_v_top1s.append(data['V']['top1_energy'])

            if ans_k_effs and disp_k_effs:
                paired_answer_k_eff.append(np.mean(ans_k_effs))
                paired_disp_k_eff.append(np.mean(disp_k_effs))
                paired_answer_k_top1.append(np.mean(ans_k_top1s))
                paired_disp_k_top1.append(np.mean(disp_k_top1s))
                paired_answer_v_eff.append(np.mean(ans_v_effs))
                paired_disp_v_eff.append(np.mean(disp_v_effs))
                paired_answer_v_top1.append(np.mean(ans_v_top1s))
                paired_disp_v_top1.append(np.mean(disp_v_top1s))

    # Per-head summary
    head_summary = {}
    for h in range(n_kv_heads):
        m = head_metrics[h]
        head_summary[f'H{h}'] = {
            'group': 'answer' if h in ANSWER_HEADS else 'dispensable',
            'K_eff_rank_mean': float(np.mean(m['K_eff_rank'])) if m['K_eff_rank'] else None,
            'K_eff_rank_std': float(np.std(m['K_eff_rank'])) if m['K_eff_rank'] else None,
            'K_top1_mean': float(np.mean(m['K_top1'])) if m['K_top1'] else None,
            'K_top1_std': float(np.std(m['K_top1'])) if m['K_top1'] else None,
            'K_gap_mean': float(np.mean(m['K_gap'])) if m['K_gap'] else None,
            'K_gap_std': float(np.std(m['K_gap'])) if m['K_gap'] else None,
            'V_eff_rank_mean': float(np.mean(m['V_eff_rank'])) if m['V_eff_rank'] else None,
            'V_eff_rank_std': float(np.std(m['V_eff_rank'])) if m['V_eff_rank'] else None,
            'V_top1_mean': float(np.mean(m['V_top1'])) if m['V_top1'] else None,
            'V_top1_std': float(np.std(m['V_top1'])) if m['V_top1'] else None,
            'V_gap_mean': float(np.mean(m['V_gap'])) if m['V_gap'] else None,
            'V_gap_std': float(np.std(m['V_gap'])) if m['V_gap'] else None,
            'n_observations': len(m['K_eff_rank']),
        }

    # Statistical tests (paired Wilcoxon)
    stat_tests = {}
    test_pairs = [
        ('K_eff_rank', paired_answer_k_eff, paired_disp_k_eff, 'less'),  # answer < dispensable
        ('K_top1', paired_answer_k_top1, paired_disp_k_top1, 'greater'),  # answer > dispensable
        ('V_eff_rank', paired_answer_v_eff, paired_disp_v_eff, 'two-sided'),
        ('V_top1', paired_answer_v_top1, paired_disp_v_top1, 'two-sided'),
    ]

    for name, ans_vals, disp_vals, alt in test_pairs:
        ans_arr = np.array(ans_vals)
        disp_arr = np.array(disp_vals)
        if len(ans_arr) > 5:
            try:
                stat, p = stats.wilcoxon(ans_arr, disp_arr, alternative=alt)
                stat_tests[name] = {
                    'statistic': float(stat),
                    'p_value': float(p),
                    'alternative': alt,
                    'n_pairs': int(len(ans_arr)),
                    'answer_mean': float(np.mean(ans_arr)),
                    'dispensable_mean': float(np.mean(disp_arr)),
                    'diff_mean': float(np.mean(ans_arr - disp_arr)),
                    'diff_std': float(np.std(ans_arr - disp_arr)),
                }
            except Exception as e:
                stat_tests[name] = {'error': str(e)}

    # Group-level summary
    ans_heads_data = [head_summary[f'H{h}'] for h in ANSWER_HEADS]
    disp_heads_data = [head_summary[f'H{h}'] for h in DISPENSABLE_HEADS]

    group_summary = {
        'answer_K_eff_rank': float(np.mean([d['K_eff_rank_mean'] for d in ans_heads_data if d['K_eff_rank_mean']])),
        'dispensable_K_eff_rank': float(np.mean([d['K_eff_rank_mean'] for d in disp_heads_data if d['K_eff_rank_mean']])),
        'answer_K_top1': float(np.mean([d['K_top1_mean'] for d in ans_heads_data if d['K_top1_mean']])),
        'dispensable_K_top1': float(np.mean([d['K_top1_mean'] for d in disp_heads_data if d['K_top1_mean']])),
        'answer_K_gap': float(np.mean([d['K_gap_mean'] for d in ans_heads_data if d['K_gap_mean']])),
        'dispensable_K_gap': float(np.mean([d['K_gap_mean'] for d in disp_heads_data if d['K_gap_mean']])),
        'answer_V_eff_rank': float(np.mean([d['V_eff_rank_mean'] for d in ans_heads_data if d['V_eff_rank_mean']])),
        'dispensable_V_eff_rank': float(np.mean([d['V_eff_rank_mean'] for d in disp_heads_data if d['V_eff_rank_mean']])),
        'answer_V_top1': float(np.mean([d['V_top1_mean'] for d in ans_heads_data if d['V_top1_mean']])),
        'dispensable_V_top1': float(np.mean([d['V_top1_mean'] for d in disp_heads_data if d['V_top1_mean']])),
    }

    # Compute ratios
    if group_summary['dispensable_K_eff_rank'] > 0:
        group_summary['K_eff_rank_ratio_ans_disp'] = group_summary['answer_K_eff_rank'] / group_summary['dispensable_K_eff_rank']
    if group_summary['dispensable_K_top1'] > 0:
        group_summary['K_top1_ratio_ans_disp'] = group_summary['answer_K_top1'] / group_summary['dispensable_K_top1']
    if group_summary['dispensable_V_eff_rank'] > 0:
        group_summary['V_eff_rank_ratio_ans_disp'] = group_summary['answer_V_eff_rank'] / group_summary['dispensable_V_eff_rank']

    # Per-layer answer vs dispensable (for layer-resolved figure)
    layer_comparison = {}
    for prob_result in all_results:
        for layer_idx in range(n_layers):
            if layer_idx not in layer_comparison:
                layer_comparison[layer_idx] = {
                    'ans_K_eff': [], 'disp_K_eff': [],
                    'ans_K_top1': [], 'disp_K_top1': [],
                    'ans_V_eff': [], 'disp_V_eff': [],
                }
            for head_idx in range(n_kv_heads):
                key = f"L{layer_idx}_H{head_idx}"
                if key not in prob_result['per_head']:
                    continue
                data = prob_result['per_head'][key]
                if head_idx in ANSWER_HEADS:
                    layer_comparison[layer_idx]['ans_K_eff'].append(data['K']['effective_rank'])
                    layer_comparison[layer_idx]['ans_K_top1'].append(data['K']['top1_energy'])
                    layer_comparison[layer_idx]['ans_V_eff'].append(data['V']['effective_rank'])
                else:
                    layer_comparison[layer_idx]['disp_K_eff'].append(data['K']['effective_rank'])
                    layer_comparison[layer_idx]['disp_K_top1'].append(data['K']['top1_energy'])
                    layer_comparison[layer_idx]['disp_V_eff'].append(data['V']['effective_rank'])

    layer_summary = {}
    for layer_idx in sorted(layer_comparison.keys()):
        lc = layer_comparison[layer_idx]
        layer_summary[layer_idx] = {
            'ans_K_eff_mean': float(np.mean(lc['ans_K_eff'])) if lc['ans_K_eff'] else None,
            'disp_K_eff_mean': float(np.mean(lc['disp_K_eff'])) if lc['disp_K_eff'] else None,
            'ans_K_top1_mean': float(np.mean(lc['ans_K_top1'])) if lc['ans_K_top1'] else None,
            'disp_K_top1_mean': float(np.mean(lc['disp_K_top1'])) if lc['disp_K_top1'] else None,
            'ans_V_eff_mean': float(np.mean(lc['ans_V_eff'])) if lc['ans_V_eff'] else None,
            'disp_V_eff_mean': float(np.mean(lc['disp_V_eff'])) if lc['disp_V_eff'] else None,
        }

    # Rank heads by K top-1 energy (higher = more concentrated routing)
    head_ranking = sorted(
        [(h, head_summary[f'H{h}']['K_top1_mean']) for h in range(n_kv_heads)
         if head_summary[f'H{h}']['K_top1_mean'] is not None],
        key=lambda x: x[1], reverse=True
    )

    # Correlation: head criticality (from Exp 045/046) vs spectral concentration
    # Functional criticality: accuracy when that head is destroyed (lower = more critical)
    # H0=67%, H5=50% on Qwen; H0=63.6%, H5=18.2% on Llama (from Exp 045/046)
    # We use 1-accuracy as criticality (higher = more critical)

    return {
        'head_summary': head_summary,
        'group_summary': group_summary,
        'stat_tests': stat_tests,
        'layer_comparison': {str(k): v for k, v in layer_summary.items()},
        'head_ranking_by_K_top1': [(h, float(v)) for h, v in head_ranking],
    }


def generate_figures(analysis, model_name, n_layers, results_dir):
    """Generate per-head spectral comparison figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    short_name = model_name.split('/')[-1]
    n_figs = 0

    # Figure 1: Per-head K effective rank (bar chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    heads = list(range(8))
    k_eff_ranks = [analysis['head_summary'][f'H{h}']['K_eff_rank_mean'] for h in heads]
    k_eff_stds = [analysis['head_summary'][f'H{h}']['K_eff_rank_std'] for h in heads]
    v_eff_ranks = [analysis['head_summary'][f'H{h}']['V_eff_rank_mean'] for h in heads]
    v_eff_stds = [analysis['head_summary'][f'H{h}']['V_eff_rank_std'] for h in heads]

    colors_k = ['#e74c3c' if h in ANSWER_HEADS else '#3498db' for h in heads]
    colors_v = ['#c0392b' if h in ANSWER_HEADS else '#2980b9' for h in heads]

    bars = axes[0].bar(heads, k_eff_ranks, yerr=k_eff_stds, color=colors_k,
                       alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('KV Head Index')
    axes[0].set_ylabel('K Effective Rank (mean across layers)')
    axes[0].set_title(f'{short_name}: K Effective Rank per Head\n(Red=Answer [H0,H5], Blue=Dispensable)')
    axes[0].set_xticks(heads)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add group means
    ans_mean = analysis['group_summary']['answer_K_eff_rank']
    disp_mean = analysis['group_summary']['dispensable_K_eff_rank']
    axes[0].axhline(y=ans_mean, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Answer mean={ans_mean:.1f}')
    axes[0].axhline(y=disp_mean, color='#3498db', linestyle='--', alpha=0.7, label=f'Disp mean={disp_mean:.1f}')
    axes[0].legend(fontsize=8)

    bars = axes[1].bar(heads, v_eff_ranks, yerr=v_eff_stds, color=colors_v,
                       alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('KV Head Index')
    axes[1].set_ylabel('V Effective Rank (mean across layers)')
    axes[1].set_title(f'{short_name}: V Effective Rank per Head')
    axes[1].set_xticks(heads)
    axes[1].grid(True, alpha=0.3, axis='y')

    ans_v_mean = analysis['group_summary']['answer_V_eff_rank']
    disp_v_mean = analysis['group_summary']['dispensable_V_eff_rank']
    axes[1].axhline(y=ans_v_mean, color='#c0392b', linestyle='--', alpha=0.7, label=f'Answer mean={ans_v_mean:.1f}')
    axes[1].axhline(y=disp_v_mean, color='#2980b9', linestyle='--', alpha=0.7, label=f'Disp mean={disp_v_mean:.1f}')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'per_head_eff_rank_{short_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    n_figs += 1

    # Figure 2: Per-head K top-1 energy (bar chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_top1s = [analysis['head_summary'][f'H{h}']['K_top1_mean'] for h in heads]
    k_top1_stds = [analysis['head_summary'][f'H{h}']['K_top1_std'] for h in heads]
    v_top1s = [analysis['head_summary'][f'H{h}']['V_top1_mean'] for h in heads]
    v_top1_stds = [analysis['head_summary'][f'H{h}']['V_top1_std'] for h in heads]

    axes[0].bar(heads, k_top1s, yerr=k_top1_stds, color=colors_k,
                alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('KV Head Index')
    axes[0].set_ylabel('K Top-1 Energy (fraction)')
    axes[0].set_title(f'{short_name}: K Top-1 Energy per Head\n(Higher = more concentrated)')
    axes[0].set_xticks(heads)
    axes[0].grid(True, alpha=0.3, axis='y')

    ans_k_top1 = analysis['group_summary']['answer_K_top1']
    disp_k_top1 = analysis['group_summary']['dispensable_K_top1']
    axes[0].axhline(y=ans_k_top1, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Answer={ans_k_top1:.3f}')
    axes[0].axhline(y=disp_k_top1, color='#3498db', linestyle='--', alpha=0.7, label=f'Disp={disp_k_top1:.3f}')
    axes[0].legend(fontsize=8)

    axes[1].bar(heads, v_top1s, yerr=v_top1_stds, color=colors_v,
                alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('KV Head Index')
    axes[1].set_ylabel('V Top-1 Energy (fraction)')
    axes[1].set_title(f'{short_name}: V Top-1 Energy per Head')
    axes[1].set_xticks(heads)
    axes[1].grid(True, alpha=0.3, axis='y')

    ans_v_top1 = analysis['group_summary']['answer_V_top1']
    disp_v_top1 = analysis['group_summary']['dispensable_V_top1']
    axes[1].axhline(y=ans_v_top1, color='#c0392b', linestyle='--', alpha=0.7, label=f'Answer={ans_v_top1:.3f}')
    axes[1].axhline(y=disp_v_top1, color='#2980b9', linestyle='--', alpha=0.7, label=f'Disp={disp_v_top1:.3f}')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'per_head_top1_energy_{short_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    n_figs += 1

    # Figure 3: Per-layer answer vs dispensable K effective rank
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lc = analysis['layer_comparison']
    layer_nums = sorted([int(k) for k in lc.keys()])
    ans_k_effs = [lc[str(l)]['ans_K_eff_mean'] for l in layer_nums if lc[str(l)]['ans_K_eff_mean'] is not None]
    disp_k_effs = [lc[str(l)]['disp_K_eff_mean'] for l in layer_nums if lc[str(l)]['disp_K_eff_mean'] is not None]
    valid_layers_k = [l for l in layer_nums if lc[str(l)]['ans_K_eff_mean'] is not None]

    axes[0].plot(valid_layers_k, ans_k_effs, 'o-', color='#e74c3c', label='Answer heads (H0+H5)', markersize=3)
    axes[0].plot(valid_layers_k, disp_k_effs, 's-', color='#3498db', label='Dispensable heads', markersize=3)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('K Effective Rank')
    axes[0].set_title(f'{short_name}: K Effective Rank\nAnswer vs Dispensable per Layer')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    ans_k_top1s = [lc[str(l)]['ans_K_top1_mean'] for l in layer_nums if lc[str(l)]['ans_K_top1_mean'] is not None]
    disp_k_top1s = [lc[str(l)]['disp_K_top1_mean'] for l in layer_nums if lc[str(l)]['disp_K_top1_mean'] is not None]
    valid_layers_t1 = [l for l in layer_nums if lc[str(l)]['ans_K_top1_mean'] is not None]

    axes[1].plot(valid_layers_t1, ans_k_top1s, 'o-', color='#e74c3c', label='Answer heads (H0+H5)', markersize=3)
    axes[1].plot(valid_layers_t1, disp_k_top1s, 's-', color='#3498db', label='Dispensable heads', markersize=3)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('K Top-1 Energy')
    axes[1].set_title(f'{short_name}: K Top-1 Energy\nAnswer vs Dispensable per Layer')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'layer_resolved_ans_vs_disp_{short_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    n_figs += 1

    # Figure 4: Head ranking by K top-1 (horizontal bar)
    fig, ax = plt.subplots(figsize=(8, 5))

    ranking = analysis['head_ranking_by_K_top1']
    head_labels = [f'H{h}' for h, v in ranking]
    values = [v for h, v in ranking]
    bar_colors = ['#e74c3c' if h in ANSWER_HEADS else '#3498db' for h, v in ranking]

    bars = ax.barh(range(len(ranking)), values, color=bar_colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(ranking)))
    ax.set_yticklabels(head_labels)
    ax.set_xlabel('K Top-1 Energy (mean across all layers)')
    ax.set_title(f'{short_name}: Head Ranking by K Spectral Concentration\n(Red=Answer, Blue=Dispensable)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # Most concentrated at top

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'head_ranking_{short_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    n_figs += 1

    return n_figs


def generate_cross_model_figure(all_analyses, results_dir):
    """Generate cross-model comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = list(all_analyses.keys())
    if len(models) < 2:
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, model_name in enumerate(models):
        short_name = model_name.split('/')[-1]
        analysis = all_analyses[model_name]
        color = ['#e74c3c', '#2ecc71'][idx]

        heads = list(range(8))

        # K effective rank per head
        k_effs = [analysis['head_summary'][f'H{h}']['K_eff_rank_mean'] for h in heads]
        offset = idx * 0.35 - 0.175
        bars = axes[0, 0].bar([h + offset for h in heads], k_effs, width=0.3,
                              color=color, alpha=0.7, label=short_name)
        axes[0, 0].set_title('K Effective Rank per Head')
        axes[0, 0].set_xlabel('Head')
        axes[0, 0].set_ylabel('Effective Rank')
        axes[0, 0].set_xticks(heads)
        axes[0, 0].legend(fontsize=8)

        # K top-1 energy per head
        k_top1s = [analysis['head_summary'][f'H{h}']['K_top1_mean'] for h in heads]
        bars = axes[0, 1].bar([h + offset for h in heads], k_top1s, width=0.3,
                              color=color, alpha=0.7, label=short_name)
        axes[0, 1].set_title('K Top-1 Energy per Head')
        axes[0, 1].set_xlabel('Head')
        axes[0, 1].set_ylabel('Top-1 Energy')
        axes[0, 1].set_xticks(heads)
        axes[0, 1].legend(fontsize=8)

        # Answer vs dispensable group comparison
        gs = analysis['group_summary']
        axes[1, 0].bar(idx * 3, gs['answer_K_eff_rank'],
                       color=color, alpha=0.9,
                       edgecolor='black', linewidth=0.5, width=0.8)
        axes[1, 0].bar(idx * 3 + 1, gs['dispensable_K_eff_rank'],
                       color=color, alpha=0.5,
                       edgecolor='black', linewidth=0.5, width=0.8)

        # V effective rank per head
        v_effs = [analysis['head_summary'][f'H{h}']['V_eff_rank_mean'] for h in heads]
        bars = axes[1, 1].bar([h + offset for h in heads], v_effs, width=0.3,
                              color=color, alpha=0.7, label=short_name)
        axes[1, 1].set_title('V Effective Rank per Head')
        axes[1, 1].set_xlabel('Head')
        axes[1, 1].set_ylabel('Effective Rank')
        axes[1, 1].set_xticks(heads)
        axes[1, 1].legend(fontsize=8)

    # Clean up axes[1,0]
    axes[1, 0].set_title('K Eff Rank: Answer vs Dispensable Groups')
    model_shorts = [m.split('/')[-1][:8] for m in models]
    axes[1, 0].set_xticks([0, 1, 3, 4])
    axes[1, 0].set_xticklabels([f'{model_shorts[0]}\nAns', f'{model_shorts[0]}\nDisp',
                                 f'{model_shorts[1]}\nAns', f'{model_shorts[1]}\nDisp'],
                                fontsize=8)
    axes[1, 0].set_ylabel('K Effective Rank')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cross-Model Per-Head Spectral Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cross_model_per_head_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    return 1


def main():
    start_time = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading GSM8K...")
    ds = load_gsm8k()

    indices = list(range(len(ds)))
    np.random.shuffle(indices)

    all_analyses = {}
    all_raw = {}

    for model_name in MODELS:
        model_start = time.time()
        elapsed_total = model_start - start_time
        remaining = TIME_BUDGET - elapsed_total

        if remaining < 120:
            print(f"\n!! Time budget nearly exhausted ({remaining:.0f}s). Skipping {model_name}.")
            break

        short_name = model_name.split('/')[-1]
        print(f"\n{'='*60}")
        print(f"Model: {short_name}")
        print(f"Elapsed: {elapsed_total:.0f}s, Remaining: {remaining:.0f}s")
        print(f"{'='*60}")

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
        attempt = 0
        n_kv_heads = None
        n_layers_model = None

        per_model_budget = remaining / max(1, len(MODELS) - len(all_analyses))

        while len(prob_results) < N_PROBLEMS and attempt < N_PROBLEMS * 3:
            if time.time() - model_start > per_model_budget:
                print(f"  Time budget reached ({per_model_budget:.0f}s)")
                break

            idx = indices[attempt % len(indices)]
            attempt += 1

            question = ds[idx]['question']
            true_answer = extract_answer(ds[idx]['answer'])

            try:
                result = analyze_per_head(model, tokenizer, question)
                if result is None:
                    continue

                # Check correctness
                if result['answer'] and true_answer:
                    try:
                        correct = abs(float(result['answer']) - float(true_answer)) < 0.01
                    except (ValueError, TypeError):
                        correct = result['answer'].strip() == true_answer.strip()
                else:
                    correct = False

                if not correct:
                    continue

                n_kv_heads = result['n_kv_heads']
                n_layers_model = result['n_layers']
                result['question'] = question
                result['true_answer'] = true_answer
                prob_results.append(result)

                n_valid = len(prob_results)
                elapsed = time.time() - model_start
                rate = elapsed / n_valid
                print(f"  [{n_valid}/{N_PROBLEMS}] idx={idx}, tokens={result['reasoning_tokens']}, "
                      f"heads={n_kv_heads}, layers={n_layers_model}, "
                      f"rate={rate:.1f}s/prob, elapsed={elapsed:.0f}s")

            except Exception as e:
                print(f"  Error on problem {idx}: {e}")
                continue

        print(f"\n{short_name}: {len(prob_results)} valid problems in {time.time()-model_start:.0f}s")

        if prob_results and n_kv_heads and n_layers_model:
            analysis = analyze_head_groups(prob_results, n_kv_heads, n_layers_model)
            all_analyses[model_name] = analysis
            all_raw[model_name] = prob_results

            # Print summary
            gs = analysis['group_summary']
            print(f"\n--- {short_name} Head Group Summary ---")
            print(f"  ANSWER heads (H0, H5):")
            print(f"    K eff rank: {gs['answer_K_eff_rank']:.2f}")
            print(f"    K top-1:    {gs['answer_K_top1']:.4f}")
            print(f"    K gap:      {gs['answer_K_gap']:.3f}")
            print(f"    V eff rank: {gs['answer_V_eff_rank']:.2f}")
            print(f"    V top-1:    {gs['answer_V_top1']:.4f}")
            print(f"  DISPENSABLE heads (H1-4, H6-7):")
            print(f"    K eff rank: {gs['dispensable_K_eff_rank']:.2f}")
            print(f"    K top-1:    {gs['dispensable_K_top1']:.4f}")
            print(f"    K gap:      {gs['dispensable_K_gap']:.3f}")
            print(f"    V eff rank: {gs['dispensable_V_eff_rank']:.2f}")
            print(f"    V top-1:    {gs['dispensable_V_top1']:.4f}")
            print(f"  RATIOS (answer/dispensable):")
            print(f"    K eff rank: {gs.get('K_eff_rank_ratio_ans_disp', 'N/A')}")
            print(f"    K top-1:    {gs.get('K_top1_ratio_ans_disp', 'N/A')}")
            print(f"    V eff rank: {gs.get('V_eff_rank_ratio_ans_disp', 'N/A')}")

            print(f"\n  Per-head K effective rank:")
            for h in range(n_kv_heads):
                hs = analysis['head_summary'][f'H{h}']
                marker = " <-- ANSWER" if h in ANSWER_HEADS else ""
                print(f"    H{h}: {hs['K_eff_rank_mean']:.2f} +/- {hs['K_eff_rank_std']:.2f} "
                      f"(top1={hs['K_top1_mean']:.4f}, gap={hs['K_gap_mean']:.3f}){marker}")

            print(f"\n  Per-head V effective rank:")
            for h in range(n_kv_heads):
                hs = analysis['head_summary'][f'H{h}']
                marker = " <-- ANSWER" if h in ANSWER_HEADS else ""
                print(f"    H{h}: {hs['V_eff_rank_mean']:.2f} +/- {hs['V_eff_rank_std']:.2f} "
                      f"(top1={hs['V_top1_mean']:.4f}){marker}")

            print(f"\n  Head ranking by K top-1 energy (most concentrated first):")
            for rank, (h, val) in enumerate(analysis['head_ranking_by_K_top1']):
                marker = " <-- ANSWER" if h in ANSWER_HEADS else ""
                print(f"    #{rank+1}: H{h} = {val:.4f}{marker}")

            print(f"\n  Statistical tests (paired Wilcoxon):")
            for name, test_result in analysis['stat_tests'].items():
                if 'error' in test_result:
                    print(f"    {name}: ERROR - {test_result['error']}")
                else:
                    sig = "**" if test_result['p_value'] < 0.01 else "*" if test_result['p_value'] < 0.05 else "ns"
                    print(f"    {name}: p={test_result['p_value']:.4f} {sig} "
                          f"(ans={test_result['answer_mean']:.3f}, disp={test_result['dispensable_mean']:.3f}, "
                          f"diff={test_result['diff_mean']:.3f}, n={test_result['n_pairs']})")

            # Generate figures
            n_figs = generate_figures(analysis, model_name, n_layers_model, RESULTS_DIR)
            print(f"  Generated {n_figs} figures for {short_name}")

            # Save results
            save_data = {
                'model': model_name,
                'n_problems': len(prob_results),
                'n_layers': n_layers_model,
                'n_kv_heads': n_kv_heads,
                'analysis': analysis,
            }
            with open(os.path.join(RESULTS_DIR, f'spectral_per_head_{short_name}.json'), 'w') as f:
                json.dump(save_data, f, indent=2)

        # Unload model
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-model comparison
    if len(all_analyses) >= 2:
        n_comp = generate_cross_model_figure(all_analyses, RESULTS_DIR)
        print(f"\nGenerated {n_comp} cross-model comparison figure(s)")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Models analyzed: {len(all_analyses)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
