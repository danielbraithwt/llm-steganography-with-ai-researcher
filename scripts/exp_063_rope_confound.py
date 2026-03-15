#!/usr/bin/env python3
"""
Experiment 063: RoPE Confound Test for K Cache Spectral Dominance

Tests whether the K cache spectral dominance (lower effective rank, higher top-1
energy) found in Exp 062 is an artifact of Rotary Position Embedding (RoPE).

RoPE is applied to K but not V. Position-dependent rotations could create
artificial low-rank structure in the K cache. This experiment reverses the
RoPE rotation to recover pre-RoPE K representations and compares their spectral
properties against post-RoPE K and V.

Key question: Does pre-RoPE K still show spectral dominance over V?
- If YES: The K>V spectral asymmetry is an intrinsic property of K representations
- If NO: The asymmetry is (partially) a RoPE artifact

Method:
1. Generate 8-shot CoT reasoning on GSM8K problems
2. Extract post-RoPE K and V from KV cache
3. Reverse RoPE rotation mathematically to get pre-RoPE K
4. Compute spectral metrics on all three: pre-RoPE K, post-RoPE K, V
5. Compare and test statistical significance
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
TIME_BUDGET = 1700  # seconds
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 2048
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_063")

MODELS = [
    "Qwen/Qwen3-4B-Base",
    "meta-llama/Llama-3.1-8B-Instruct",
]

N_PROBLEMS = 20  # number of GSM8K problems per model

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


def get_rope_params(model, layer_idx=0):
    """Extract RoPE parameters from the model."""
    try:
        # Get rope_theta from config (may be in rope_scaling dict or top-level)
        rope_theta = getattr(model.config, 'rope_theta', None)
        if rope_theta is None:
            rope_scaling = getattr(model.config, 'rope_scaling', None)
            if rope_scaling and isinstance(rope_scaling, dict):
                rope_theta = rope_scaling.get('rope_theta', 10000.0)
            else:
                rope_theta = 10000.0

        # Get rope_type
        rope_scaling = getattr(model.config, 'rope_scaling', None)
        rope_type = 'default'
        if rope_scaling and isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get('rope_type', 'default')

        # Get head dim
        head_dim = model.config.hidden_size // model.config.num_attention_heads

        return float(rope_theta), int(head_dim), rope_type
    except Exception as e:
        print(f"Warning: Could not extract RoPE params: {e}")
        return 10000.0, 128, 'default'


def get_rope_cos_sin_from_model(model, positions, device='cpu'):
    """
    Extract RoPE cos/sin from the model's rotary embedding module.

    This handles all RoPE variants (vanilla, NTK-aware, llama3, YaRN)
    because it uses the model's own implementation.

    Args:
        model: the loaded model
        positions: (seq_len,) tensor of position indices
        device: device for computation

    Returns:
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    """
    # Both Qwen3 and Llama have model.model.rotary_emb
    rotary_emb = model.model.rotary_emb

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    dummy_x = torch.zeros(1, 1, head_dim, device=device, dtype=torch.float16)
    position_ids = positions.unsqueeze(0).to(device)  # (1, seq_len)

    with torch.no_grad():
        cos, sin = rotary_emb(dummy_x, position_ids)
        # cos, sin shape: (1, seq_len, head_dim)
        if cos.dim() == 3:
            cos = cos.squeeze(0)  # (seq_len, head_dim)
            sin = sin.squeeze(0)

    return cos.float().cpu(), sin.float().cpu()


def inverse_rope(k_post, cos, sin):
    """
    Reverse RoPE rotation to recover pre-RoPE K.

    Forward RoPE: k_post = k_pre * cos + rotate_half(k_pre) * sin
    where rotate_half(x) = cat(-x[d/2:], x[:d/2])

    Inverse: k_pre = k_post * cos + rotate_half_inv(k_post) * sin
    where rotate_half_inv(x) = cat(x[d/2:], -x[:d/2])

    Args:
        k_post: (n_heads, seq_len, head_dim) - post-RoPE K values
        cos: (seq_len, head_dim) - RoPE cos values
        sin: (seq_len, head_dim) - RoPE sin values

    Returns:
        k_pre: (n_heads, seq_len, head_dim) - pre-RoPE K values
    """
    def rotate_half_inv(x):
        """Inverse of rotate_half: cat(x2, -x1)"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((x2, -x1), dim=-1)

    # Add head dimension to cos/sin: (seq_len, head_dim) -> (1, seq_len, head_dim)
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)

    k_pre = (k_post.float() * cos) + (rotate_half_inv(k_post.float()) * sin)
    return k_pre.to(k_post.dtype)


def verify_inverse_rope(k_post, cos, sin, k_pre):
    """
    Verify inverse RoPE by re-applying forward RoPE and checking match.
    Returns max absolute error.
    """
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos_u = cos.unsqueeze(0)
    sin_u = sin.unsqueeze(0)

    k_reapplied = (k_pre.float() * cos_u) + (rotate_half(k_pre.float()) * sin_u)
    max_err = (k_reapplied - k_post.float()).abs().max().item()
    return max_err


def compute_spectral_metrics(matrix):
    """Compute spectral metrics for a 2D matrix (positions x dim)."""
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

    # Participation ratio
    participation_ratio = (S.sum() ** 2) / (np.sum(S ** 2))

    # Spectral gap
    spectral_gap = S[0] / S[1] if S[1] > 1e-10 else float('inf')

    # Top-k energy
    total_energy = np.sum(S ** 2)
    top1_energy = S[0] ** 2 / total_energy
    top5_energy = np.sum(S[:min(5, len(S))] ** 2) / total_energy
    top10_energy = np.sum(S[:min(10, len(S))] ** 2) / total_energy

    # Condition number
    condition_number = S[0] / S[-1] if S[-1] > 1e-10 else float('inf')

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
        'participation_ratio': float(participation_ratio),
        'spectral_gap': float(spectral_gap),
        'top1_energy': float(top1_energy),
        'top5_energy': float(top5_energy),
        'top10_energy': float(top10_energy),
        'condition_number': float(min(condition_number, 1e10)),
        'decay_rate': float(decay_rate),
        'n_singular_values': int(len(S)),
        'max_rank': int(max_rank),
        'singular_values_top20': S[:20].tolist(),
    }


def analyze_kv_cache_with_rope(model, tokenizer, question, rope_theta, head_dim,
                                rope_cos_sin_cache=None,
                                max_gen=MAX_GEN_TOKENS, device='cuda'):
    """
    Generate CoT reasoning and extract pre-RoPE K, post-RoPE K, and V spectral metrics.
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
    answer = extract_answer(gen_text)

    # Forward pass to get full KV cache
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
    actual_head_dim = sample_k.shape[3]

    # Position indices for reasoning portion
    reasoning_positions = torch.arange(prompt_len, total_len, device='cpu')

    # Get RoPE cos/sin from the model's own rotary embedding (handles all RoPE variants)
    cos, sin = get_rope_cos_sin_from_model(model, reasoning_positions, device=device)

    results = {
        'answer': answer,
        'reasoning_tokens': int(reasoning_len),
        'prompt_tokens': int(prompt_len),
        'n_layers': n_layers,
        'n_kv_heads': n_kv_heads,
        'head_dim': actual_head_dim,
        'rope_theta': rope_theta,
        'per_layer': {},
        'rope_verification': [],
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

        # Extract reasoning-only: (n_kv_heads, reasoning_len, head_dim)
        k_reasoning = k[0, :, prompt_len:, :].cpu()
        v_reasoning = v[0, :, prompt_len:, :].cpu()

        # Compute pre-RoPE K via inverse rotation
        k_pre_rope = inverse_rope(k_reasoning, cos, sin)

        # Verify inverse (on first problem, first 3 layers)
        if layer_idx < 3 and len(results['rope_verification']) < 3:
            err = verify_inverse_rope(k_reasoning, cos, sin, k_pre_rope)
            results['rope_verification'].append({
                'layer': layer_idx,
                'max_abs_error': err,
            })

        # Per-layer analysis: concatenate heads -> (reasoning_len, n_kv_heads * head_dim)
        k_post_layer = k_reasoning.permute(1, 0, 2).reshape(reasoning_len, -1)
        k_pre_layer = k_pre_rope.permute(1, 0, 2).reshape(reasoning_len, -1)
        v_layer = v_reasoning.permute(1, 0, 2).reshape(reasoning_len, -1)

        k_post_metrics = compute_spectral_metrics(k_post_layer)
        k_pre_metrics = compute_spectral_metrics(k_pre_layer)
        v_metrics = compute_spectral_metrics(v_layer)

        if k_post_metrics and k_pre_metrics and v_metrics:
            results['per_layer'][str(layer_idx)] = {
                'K_post_rope': k_post_metrics,
                'K_pre_rope': k_pre_metrics,
                'V': v_metrics,
            }

    del past_kv, out, output
    gc.collect()
    torch.cuda.empty_cache()

    return results


def aggregate_results(all_results):
    """Aggregate spectral metrics across problems for all three conditions."""

    conditions = ['K_post_rope', 'K_pre_rope', 'V']
    metrics_to_agg = ['effective_rank', 'normalized_eff_rank', 'spectral_gap',
                      'top1_energy', 'top5_energy', 'top10_energy', 'decay_rate']

    # Per-layer aggregation
    layer_agg = {}
    for prob_result in all_results:
        for layer_str, layer_data in prob_result['per_layer'].items():
            if layer_str not in layer_agg:
                layer_agg[layer_str] = {c: [] for c in conditions}
            for cond in conditions:
                layer_agg[layer_str][cond].append(layer_data[cond])

    layer_summary = {}
    for layer_str in sorted(layer_agg.keys(), key=int):
        layer_summary[layer_str] = {}
        for cond in conditions:
            metrics_list = layer_agg[layer_str][cond]
            if not metrics_list:
                continue
            summary = {}
            for metric_name in metrics_to_agg:
                vals = [m[metric_name] for m in metrics_list
                       if not (isinstance(m[metric_name], float) and
                               (np.isnan(m[metric_name]) or np.isinf(m[metric_name])))]
                if vals:
                    summary[metric_name] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'median': float(np.median(vals)),
                    }
            layer_summary[layer_str][cond] = summary

    # Global aggregation
    global_data = {c: {m: [] for m in metrics_to_agg} for c in conditions}

    for prob_result in all_results:
        for layer_data in prob_result['per_layer'].values():
            for cond in conditions:
                for metric_name in metrics_to_agg:
                    val = layer_data[cond][metric_name]
                    if not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                        global_data[cond][metric_name].append(val)

    global_summary = {}
    for cond in conditions:
        global_summary[cond] = {}
        for metric_name in metrics_to_agg:
            vals = global_data[cond][metric_name]
            if vals:
                global_summary[cond][metric_name] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'median': float(np.median(vals)),
                    'n': len(vals),
                }

    # Compute ratios
    ratios = {}
    for metric_name in ['effective_rank', 'top1_energy', 'spectral_gap']:
        kpost = global_summary['K_post_rope'][metric_name]['mean']
        kpre = global_summary['K_pre_rope'][metric_name]['mean']
        v = global_summary['V'][metric_name]['mean']

        ratios[metric_name] = {
            'K_post_over_V': kpost / v if v > 0 else float('inf'),
            'K_pre_over_V': kpre / v if v > 0 else float('inf'),
            'K_post_over_K_pre': kpost / kpre if kpre > 0 else float('inf'),
            'rope_contribution_pct': abs(kpost - kpre) / abs(kpost - v) * 100 if abs(kpost - v) > 1e-10 else 0.0,
        }

    # Statistical test: paired Wilcoxon for K_pre vs V within each problem
    from scipy import stats
    stat_tests = {}
    for metric_name in ['effective_rank', 'top1_energy', 'spectral_gap']:
        # Collect per-problem averages (across layers)
        pre_per_prob = []
        v_per_prob = []
        post_per_prob = []
        for prob_result in all_results:
            pre_vals = [prob_result['per_layer'][l]['K_pre_rope'][metric_name]
                       for l in prob_result['per_layer']
                       if not np.isnan(prob_result['per_layer'][l]['K_pre_rope'][metric_name])
                       and not np.isinf(prob_result['per_layer'][l]['K_pre_rope'][metric_name])]
            v_vals = [prob_result['per_layer'][l]['V'][metric_name]
                     for l in prob_result['per_layer']
                     if not np.isnan(prob_result['per_layer'][l]['V'][metric_name])
                     and not np.isinf(prob_result['per_layer'][l]['V'][metric_name])]
            post_vals = [prob_result['per_layer'][l]['K_post_rope'][metric_name]
                        for l in prob_result['per_layer']
                        if not np.isnan(prob_result['per_layer'][l]['K_post_rope'][metric_name])
                        and not np.isinf(prob_result['per_layer'][l]['K_post_rope'][metric_name])]
            if pre_vals and v_vals and post_vals:
                pre_per_prob.append(np.mean(pre_vals))
                v_per_prob.append(np.mean(v_vals))
                post_per_prob.append(np.mean(post_vals))

        if len(pre_per_prob) >= 5:
            # K_pre vs V
            try:
                stat_pre_v, p_pre_v = stats.wilcoxon(pre_per_prob, v_per_prob)
            except Exception:
                stat_pre_v, p_pre_v = float('nan'), float('nan')
            # K_post vs V
            try:
                stat_post_v, p_post_v = stats.wilcoxon(post_per_prob, v_per_prob)
            except Exception:
                stat_post_v, p_post_v = float('nan'), float('nan')
            # K_pre vs K_post
            try:
                stat_pre_post, p_pre_post = stats.wilcoxon(pre_per_prob, post_per_prob)
            except Exception:
                stat_pre_post, p_pre_post = float('nan'), float('nan')

            stat_tests[metric_name] = {
                'K_pre_vs_V': {
                    'statistic': float(stat_pre_v),
                    'p_value': float(p_pre_v),
                    'n': len(pre_per_prob),
                    'pre_mean': float(np.mean(pre_per_prob)),
                    'v_mean': float(np.mean(v_per_prob)),
                },
                'K_post_vs_V': {
                    'statistic': float(stat_post_v),
                    'p_value': float(p_post_v),
                    'n': len(post_per_prob),
                },
                'K_pre_vs_K_post': {
                    'statistic': float(stat_pre_post),
                    'p_value': float(p_pre_post),
                    'n': len(pre_per_prob),
                },
            }

    return {
        'per_layer': layer_summary,
        'global': global_summary,
        'ratios': ratios,
        'statistical_tests': stat_tests,
    }


def generate_figures(model_results, model_name, results_dir):
    """Generate comparison figures for pre-RoPE K, post-RoPE K, and V."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    short_name = model_name.split('/')[-1]
    n_figs = 0

    layers = sorted(model_results['per_layer'].keys(), key=int)
    layer_nums = [int(l) for l in layers]

    # Figure 1: Effective rank comparison — three conditions per layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cond, label, marker, color in [
        ('K_post_rope', 'K (post-RoPE)', 'o', '#e74c3c'),
        ('K_pre_rope', 'K (pre-RoPE)', '^', '#9b59b6'),
        ('V', 'V (no RoPE)', 's', '#2ecc71'),
    ]:
        vals = [model_results['per_layer'][l][cond]['effective_rank']['mean']
                for l in layers if 'effective_rank' in model_results['per_layer'][l].get(cond, {})]
        stds = [model_results['per_layer'][l][cond]['effective_rank']['std']
                for l in layers if 'effective_rank' in model_results['per_layer'][l].get(cond, {})]
        axes[0].errorbar(layer_nums[:len(vals)], vals, yerr=stds,
                        label=label, marker=marker, color=color, capsize=3, markersize=4)

    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Effective Rank')
    axes[0].set_title(f'{short_name}: Effective Rank\n(pre-RoPE K vs post-RoPE K vs V)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # K/V ratios for pre and post
    post_vals = [model_results['per_layer'][l]['K_post_rope']['effective_rank']['mean'] for l in layers]
    pre_vals = [model_results['per_layer'][l]['K_pre_rope']['effective_rank']['mean'] for l in layers]
    v_vals = [model_results['per_layer'][l]['V']['effective_rank']['mean'] for l in layers]

    post_ratios = [k/v if v > 0 else 0 for k, v in zip(post_vals, v_vals)]
    pre_ratios = [k/v if v > 0 else 0 for k, v in zip(pre_vals, v_vals)]

    x = np.arange(len(layer_nums))
    width = 0.35
    axes[1].bar(x - width/2, post_ratios, width, label='K_post/V', color='#e74c3c', alpha=0.7)
    axes[1].bar(x + width/2, pre_ratios, width, label='K_pre/V', color='#9b59b6', alpha=0.7)
    axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='K=V')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('K/V Effective Rank Ratio')
    axes[1].set_title(f'{short_name}: K/V Ratio Before & After RoPE')
    axes[1].set_xticks(x[::max(1, len(x)//10)])
    axes[1].set_xticklabels([str(l) for l in layer_nums[::max(1, len(layer_nums)//10)]])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'effective_rank_rope_{short_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    n_figs += 1

    # Figure 2: Top-1 energy comparison (KEY FIGURE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cond, label, marker, color in [
        ('K_post_rope', 'K (post-RoPE)', 'o', '#e74c3c'),
        ('K_pre_rope', 'K (pre-RoPE)', '^', '#9b59b6'),
        ('V', 'V (no RoPE)', 's', '#2ecc71'),
    ]:
        vals = [model_results['per_layer'][l][cond]['top1_energy']['mean']
                for l in layers if 'top1_energy' in model_results['per_layer'][l].get(cond, {})]
        axes[0].plot(layer_nums[:len(vals)], vals,
                    f'{marker}-', label=label, color=color, markersize=4)

    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Top-1 Energy Fraction')
    axes[0].set_title(f'{short_name}: Top-1 SV Energy\n(KEY: does pre-RoPE K still dominate V?)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Spectral gap
    for cond, label, marker, color in [
        ('K_post_rope', 'K (post-RoPE)', 'o', '#e74c3c'),
        ('K_pre_rope', 'K (pre-RoPE)', '^', '#9b59b6'),
        ('V', 'V (no RoPE)', 's', '#2ecc71'),
    ]:
        vals = [model_results['per_layer'][l][cond]['spectral_gap']['mean']
                for l in layers if 'spectral_gap' in model_results['per_layer'][l].get(cond, {})]
        axes[1].plot(layer_nums[:len(vals)], vals,
                    f'{marker}-', label=label, color=color, markersize=4)

    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Spectral Gap (s1/s2)')
    axes[1].set_title(f'{short_name}: Spectral Gap')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'top1_energy_rope_{short_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    n_figs += 1

    # Figure 3: Summary bar chart — global metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    g = model_results['global']
    conditions = ['K_post_rope', 'K_pre_rope', 'V']
    labels = ['K post-RoPE', 'K pre-RoPE', 'V']
    colors = ['#e74c3c', '#9b59b6', '#2ecc71']

    for ax, metric, title in [
        (axes[0], 'effective_rank', 'Effective Rank'),
        (axes[1], 'top1_energy', 'Top-1 Energy'),
        (axes[2], 'spectral_gap', 'Spectral Gap (s1/s2)'),
    ]:
        means = [g[c][metric]['mean'] for c in conditions]
        stds = [g[c][metric]['std'] for c in conditions]
        bars = ax.bar(range(3), means, yerr=stds, color=colors, alpha=0.8, capsize=5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_title(f'{short_name}: {title}')
        ax.grid(True, alpha=0.3, axis='y')
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{mean:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'global_summary_{short_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    n_figs += 1

    return n_figs


def generate_cross_model_figure(all_model_results, results_dir):
    """Generate cross-model comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model_names = list(all_model_results.keys())
    if len(model_names) < 2:
        return 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row_idx, model_name in enumerate(model_names[:2]):
        short = model_name.split('/')[-1]
        data = all_model_results[model_name]
        layers = sorted(data['per_layer'].keys(), key=int)
        layer_nums = [int(l) for l in layers]

        for col_idx, (metric, title) in enumerate([
            ('effective_rank', 'Effective Rank'),
            ('top1_energy', 'Top-1 Energy'),
            ('spectral_gap', 'Spectral Gap'),
        ]):
            ax = axes[row_idx, col_idx]
            for cond, label, marker, color in [
                ('K_post_rope', 'K post', 'o', '#e74c3c'),
                ('K_pre_rope', 'K pre', '^', '#9b59b6'),
                ('V', 'V', 's', '#2ecc71'),
            ]:
                vals = [data['per_layer'][l][cond][metric]['mean']
                       for l in layers if metric in data['per_layer'][l].get(cond, {})]
                ax.plot(layer_nums[:len(vals)], vals,
                       f'{marker}-', label=label, color=color, markersize=3)

            ax.set_title(f'{short}: {title}')
            ax.set_xlabel('Layer')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.suptitle('RoPE Confound Test: Pre-RoPE K vs Post-RoPE K vs V', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cross_model_rope_comparison.png'),
                dpi=150, bbox_inches='tight')
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

    all_model_results = {}
    all_model_raw = {}

    for model_name in MODELS:
        model_start = time.time()
        elapsed_total = model_start - start_time
        remaining = TIME_BUDGET - elapsed_total

        if remaining < 120:
            print(f"\nTime budget nearly exhausted ({remaining:.0f}s). Skipping {model_name}.")
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

        # Get RoPE parameters
        rope_theta, head_dim, rope_type = get_rope_params(model)
        print(f"  RoPE theta: {rope_theta}, head_dim: {head_dim}, rope_type: {rope_type}")

        prob_results = []
        attempt = 0
        per_model_budget = remaining / max(1, len(MODELS) - len(all_model_results))

        while len(prob_results) < N_PROBLEMS and attempt < N_PROBLEMS * 3:
            if time.time() - model_start > per_model_budget:
                print(f"  Model time budget reached ({per_model_budget:.0f}s)")
                break

            idx = indices[attempt % len(indices)]
            attempt += 1

            question = ds[idx]['question']
            true_answer = extract_answer(ds[idx]['answer'])

            try:
                result = analyze_kv_cache_with_rope(
                    model, tokenizer, question, rope_theta, head_dim)
                if result is None:
                    continue

                # Only analyze correctly-solved problems
                if result['answer'] and true_answer:
                    try:
                        correct = abs(float(result['answer']) - float(true_answer)) < 0.01
                    except (ValueError, TypeError):
                        correct = result['answer'].strip() == true_answer.strip()
                else:
                    correct = False

                if not correct:
                    continue

                # Print RoPE verification on first problem
                if len(prob_results) == 0 and result['rope_verification']:
                    print(f"  RoPE inverse verification:")
                    for v in result['rope_verification']:
                        print(f"    Layer {v['layer']}: max error = {v['max_abs_error']:.2e}")

                result['question'] = question
                result['true_answer'] = true_answer
                prob_results.append(result)

                n_valid = len(prob_results)
                # Print running summary
                mid_layer = str(result['n_layers'] // 2)
                if mid_layer in result['per_layer']:
                    ld = result['per_layer'][mid_layer]
                    print(f"  [{n_valid}/{N_PROBLEMS}] idx={idx}, "
                          f"K_post_top1={ld['K_post_rope']['top1_energy']:.3f}, "
                          f"K_pre_top1={ld['K_pre_rope']['top1_energy']:.3f}, "
                          f"V_top1={ld['V']['top1_energy']:.3f}, "
                          f"time={time.time()-model_start:.0f}s")

            except Exception as e:
                print(f"  Error on problem {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{short_name}: {len(prob_results)} valid problems in {time.time()-model_start:.0f}s")

        if prob_results:
            agg = aggregate_results(prob_results)
            all_model_results[model_name] = agg
            all_model_raw[model_name] = prob_results

            # Print summary
            g = agg['global']
            print(f"\n--- {short_name} Global Summary ---")
            for cond, label in [('K_post_rope', 'K post-RoPE'), ('K_pre_rope', 'K pre-RoPE'), ('V', 'V')]:
                print(f"  {label}:")
                print(f"    Eff rank: {g[cond]['effective_rank']['mean']:.2f} +/- {g[cond]['effective_rank']['std']:.2f}")
                print(f"    Top-1 energy: {g[cond]['top1_energy']['mean']:.4f} +/- {g[cond]['top1_energy']['std']:.4f}")
                print(f"    Spectral gap: {g[cond]['spectral_gap']['mean']:.4f} +/- {g[cond]['spectral_gap']['std']:.4f}")

            print(f"\n  Ratios:")
            for metric_name, ratio_data in agg['ratios'].items():
                print(f"    {metric_name}:")
                print(f"      K_post/V = {ratio_data['K_post_over_V']:.4f}")
                print(f"      K_pre/V  = {ratio_data['K_pre_over_V']:.4f}")
                print(f"      K_post/K_pre = {ratio_data['K_post_over_K_pre']:.4f}")
                print(f"      RoPE contribution = {ratio_data['rope_contribution_pct']:.1f}%")

            print(f"\n  Statistical tests (Wilcoxon signed-rank):")
            for metric_name, test_data in agg.get('statistical_tests', {}).items():
                for comp, comp_data in test_data.items():
                    p = comp_data.get('p_value', float('nan'))
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    print(f"    {metric_name} {comp}: p={p:.4f} {sig} (n={comp_data.get('n', '?')})")

            # Generate figures
            n_figs = generate_figures(agg, model_name, RESULTS_DIR)
            print(f"  Generated {n_figs} figures for {short_name}")

            # Save per-model results
            save_data = {
                'model': model_name,
                'n_problems': len(prob_results),
                'rope_theta': rope_theta,
                'head_dim': head_dim,
                'aggregate': agg,
            }
            with open(os.path.join(RESULTS_DIR, f'rope_confound_{short_name}.json'), 'w') as f:
                json.dump(save_data, f, indent=2)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-model comparison
    if len(all_model_results) >= 2:
        n_comp = generate_cross_model_figure(all_model_results, RESULTS_DIR)
        print(f"\nGenerated {n_comp} cross-model comparison figure(s)")

    # Print final verdict
    print(f"\n{'='*60}")
    print("VERDICT SUMMARY")
    print(f"{'='*60}")
    for model_name, agg in all_model_results.items():
        short = model_name.split('/')[-1]
        g = agg['global']
        r = agg['ratios']

        pre_top1 = g['K_pre_rope']['top1_energy']['mean']
        v_top1 = g['V']['top1_energy']['mean']
        post_top1 = g['K_post_rope']['top1_energy']['mean']

        pre_rank = g['K_pre_rope']['effective_rank']['mean']
        v_rank = g['V']['effective_rank']['mean']
        post_rank = g['K_post_rope']['effective_rank']['mean']

        rope_pct_rank = r['effective_rank']['rope_contribution_pct']
        rope_pct_top1 = r['top1_energy']['rope_contribution_pct']

        print(f"\n{short}:")
        print(f"  Effective rank: K_post={post_rank:.1f}, K_pre={pre_rank:.1f}, V={v_rank:.1f}")
        print(f"    K_pre/V ratio = {pre_rank/v_rank:.3f} {'(K_pre < V => intrinsic)' if pre_rank < v_rank else '(K_pre >= V => RoPE artifact)'}")
        print(f"    RoPE contribution to K-V gap: {rope_pct_rank:.1f}%")
        print(f"  Top-1 energy: K_post={post_top1:.3f}, K_pre={pre_top1:.3f}, V={v_top1:.3f}")
        print(f"    K_pre/V ratio = {pre_top1/v_top1:.2f}x {'(K_pre > V => intrinsic)' if pre_top1 > v_top1 else '(K_pre <= V => RoPE artifact)'}")
        print(f"    RoPE contribution to K-V gap: {rope_pct_top1:.1f}%")

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
