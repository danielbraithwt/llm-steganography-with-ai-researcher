#!/usr/bin/env python3
"""
Experiment 058: Sink-Dominance Analysis and Budget Crossover
— Why are K-norm and true H2O anti-correlated?

Exp 057 found K-norm and cumulative attention are anti-correlated (rho=-0.431
on Llama). Leading hypothesis: true H2O selects attention sinks (infrastructure),
while K-norm selects content-dense positions.

This experiment:
1. Profiles the positional distribution of true H2O vs K-norm selections
2. Tests sink-excluded and late-layer-only true H2O as fixes
3. Sweeps budgets 25-75% under masking to find the precise crossover point
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
                           "results", "exp_058")

# Budget sweep values
BUDGETS = [0.25, 0.33, 0.40, 0.50, 0.60, 0.75]

# Base selections (tested at all budgets under masking)
BASE_SELECTIONS = ['true_h2o', 'k_norm_h2o', 'random', 'recent']

# Modified H2O variants (tested at 33% and 50% under masking only)
MODIFIED_SELECTIONS = ['sink_excluded_h2o', 'late_layer_h2o']
MODIFIED_BUDGETS = [0.33, 0.50]

# Late layers for late_layer_h2o (layers 18+ out of 36 for Qwen, adjusted per model)
LATE_LAYER_FRACTION = 0.5  # use top 50% of layers

# Number of sink positions to exclude
N_SINKS = 4

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


def wilson_ci(n_success, n_total, z=1.96):
    if n_total == 0:
        return (0, 0)
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return (max(0, center - spread), min(1, center + spread))


@torch.no_grad()
def compute_true_h2o_importance(model, full_seq, prompt_len, reasoning_len, num_layers):
    """Compute true cumulative attention importance for each reasoning position."""
    total_len = prompt_len + reasoning_len
    try:
        outputs = model(
            input_ids=full_seq[:total_len].unsqueeze(0),
            output_attentions=True,
            use_cache=False,
        )
        # All-layer cumulative importance
        cumulative_all = np.zeros(reasoning_len)
        # Late-layer cumulative importance (top half of layers)
        late_start = int(num_layers * LATE_LAYER_FRACTION)
        cumulative_late = np.zeros(reasoning_len)

        for layer_idx in range(num_layers):
            attn = outputs.attentions[layer_idx]  # [1, num_heads, seq_len, seq_len]
            reasoning_attn = attn[0, :, prompt_len:total_len, prompt_len:total_len]
            pos_importance = reasoning_attn.float().sum(dim=(0, 1)).cpu().numpy()
            cumulative_all += pos_importance
            if layer_idx >= late_start:
                cumulative_late += pos_importance

        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        return cumulative_all, cumulative_late

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        print("    OOM with output_attentions — skipping true H2O")
        return None, None


@torch.no_grad()
def generate_and_build_cache(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS):
    """Generate CoT trace and build KV cache."""
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

    reasoning_ids = gen_ids.unsqueeze(0)
    trunc_point = find_truncation_point(reasoning_ids, tokenizer)
    if trunc_point is None or trunc_point < 10:
        return None

    full_seq = torch.cat([inputs.input_ids[0], gen_ids[:trunc_point]])
    total_len = full_seq.shape[0]
    if total_len > MAX_SEQ_LEN:
        return None

    outputs = model(input_ids=full_seq.unsqueeze(0), use_cache=True)
    full_cache = outputs.past_key_values
    del outputs

    num_layers = model.config.num_hidden_layers
    reasoning_len = trunc_point

    k_norm_importance = np.zeros(total_len)
    for l in range(num_layers):
        k = full_cache.layers[l].keys[0]
        k_norms = k.float().norm(dim=-1).cpu().numpy()
        k_norm_importance += k_norms.mean(axis=0)

    return {
        'gen_text': gen_text,
        'full_seq': full_seq,
        'full_cache': full_cache,
        'prompt_len': prompt_len,
        'reasoning_len': reasoning_len,
        'k_norm_importance': k_norm_importance,
        'total_len': total_len,
    }


def select_positions(selection, budget_frac, reasoning_len, prompt_len,
                     k_norm_importance, true_h2o_all, true_h2o_late):
    """Select which reasoning positions to keep based on selection method."""
    n_keep = max(1, int(budget_frac * reasoning_len))
    reason_positions = list(range(prompt_len, prompt_len + reasoning_len))

    if selection == "random":
        rng = np.random.RandomState(42)
        keep_idx = set(rng.choice(reasoning_len, size=n_keep, replace=False).tolist())

    elif selection == "k_norm_h2o":
        reason_scores = k_norm_importance[prompt_len:prompt_len + reasoning_len]
        keep_idx = set(np.argsort(reason_scores)[-n_keep:].tolist())

    elif selection == "true_h2o":
        if true_h2o_all is None:
            reason_scores = k_norm_importance[prompt_len:prompt_len + reasoning_len]
        else:
            reason_scores = true_h2o_all
        keep_idx = set(np.argsort(reason_scores)[-n_keep:].tolist())

    elif selection == "sink_excluded_h2o":
        if true_h2o_all is None:
            reason_scores = k_norm_importance[prompt_len:prompt_len + reasoning_len]
        else:
            reason_scores = true_h2o_all.copy()
        # Zero out sink positions so they're not selected (unless budget is so large
        # they'd be included anyway)
        n_sinks_to_exclude = min(N_SINKS, reasoning_len)
        reason_scores_excluded = reason_scores.copy()
        reason_scores_excluded[:n_sinks_to_exclude] = -np.inf
        keep_idx = set(np.argsort(reason_scores_excluded)[-n_keep:].tolist())

    elif selection == "late_layer_h2o":
        if true_h2o_late is None:
            reason_scores = k_norm_importance[prompt_len:prompt_len + reasoning_len]
        else:
            reason_scores = true_h2o_late
        keep_idx = set(np.argsort(reason_scores)[-n_keep:].tolist())

    elif selection == "recent":
        n_sinks = min(4, reasoning_len)
        n_recent = max(0, n_keep - n_sinks)
        if n_recent <= 0:
            keep_list = list(range(n_sinks))
        else:
            keep_list = list(range(n_sinks)) + list(range(reasoning_len - n_recent, reasoning_len))
        keep_idx = set(sorted(set(keep_list))[:n_keep])

    else:
        raise ValueError(f"Unknown selection: {selection}")

    keep_pos = sorted(reason_positions[i] for i in keep_idx)
    evict_pos = sorted(reason_positions[i] for i in range(reasoning_len) if i not in keep_idx)
    return keep_pos, evict_pos, keep_idx


@torch.no_grad()
def evict_and_generate(model, tokenizer, full_cache, prompt_len, reasoning_len,
                       evict_positions, num_layers, max_answer_tokens=64):
    """Apply masking eviction and generate answer."""
    total_len = prompt_len + reasoning_len
    device = model.device
    evicted_cache = DynamicCache()

    # For masking: don't modify cache, use attention mask
    evict_tensor = torch.zeros(total_len, dtype=torch.bool, device=device)
    if evict_positions:
        evict_idx = torch.tensor(evict_positions, dtype=torch.long, device=device)
        evict_tensor[evict_idx] = True

    for l in range(num_layers):
        k = full_cache.layers[l].keys[:, :, :total_len, :].clone()
        v = full_cache.layers[l].values[:, :, :total_len, :].clone()
        evicted_cache.update(k, v, l)

    # Attention mask with 0 at evicted positions
    attn_mask = torch.ones(1, total_len, dtype=torch.long, device=device)
    attn_mask[0, evict_tensor] = 0

    # Generate answer — feed " ####" prefix then decode
    answer_prefix = tokenizer.encode(" ####", add_special_tokens=False)
    cache_len = total_len

    out = None
    for i in range(len(answer_prefix)):
        tok = torch.tensor([[answer_prefix[i]]], device=device)
        attn_mask = torch.cat([attn_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
        out = model(input_ids=tok, past_key_values=evicted_cache, use_cache=True,
                    attention_mask=attn_mask,
                    position_ids=torch.tensor([[cache_len]], device=device))
        evicted_cache = out.past_key_values
        cache_len += 1

    gen_tokens = []
    next_logits = out.logits[:, -1, :]
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))

    for step in range(max_answer_tokens):
        next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
        token_id = next_tok[0, 0].item()
        if token_id == tokenizer.eos_token_id or token_id in newline_ids:
            break
        gen_tokens.append(token_id)
        attn_mask = torch.cat([attn_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
        out = model(input_ids=next_tok, past_key_values=evicted_cache, use_cache=True,
                    attention_mask=attn_mask,
                    position_ids=torch.tensor([[cache_len]], device=device))
        evicted_cache = out.past_key_values
        next_logits = out.logits[:, -1, :]
        cache_len += 1

    answer_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    del evicted_cache
    return answer_text


def analyze_position_profile(keep_idx_set, reasoning_len, n_quintiles=5):
    """Analyze positional distribution of selected positions."""
    keep_list = sorted(keep_idx_set)
    n_keep = len(keep_list)
    if n_keep == 0 or reasoning_len == 0:
        return {}

    bin_size = reasoning_len / n_quintiles
    quintile_counts = [0] * n_quintiles
    for pos in keep_list:
        q = min(int(pos / bin_size), n_quintiles - 1)
        quintile_counts[q] += 1

    quintile_fracs = [c / n_keep for c in quintile_counts]

    # Sink analysis: how many of first N_SINKS positions are selected?
    n_sinks = min(N_SINKS, reasoning_len)
    sinks_selected = sum(1 for p in keep_list if p < n_sinks)
    sink_frac = sinks_selected / n_keep

    # Recent analysis: how many of last 10% are selected?
    n_recent = max(1, int(0.1 * reasoning_len))
    recent_threshold = reasoning_len - n_recent
    recent_selected = sum(1 for p in keep_list if p >= recent_threshold)
    recent_frac = recent_selected / n_keep

    # Mean position (normalized to [0, 1])
    mean_pos = np.mean(keep_list) / reasoning_len if keep_list else 0.5

    return {
        'quintile_fracs': quintile_fracs,
        'quintile_counts': quintile_counts,
        'sink_frac': sink_frac,
        'sinks_selected': sinks_selected,
        'recent_frac': recent_frac,
        'recent_selected': recent_selected,
        'mean_pos': mean_pos,
        'n_keep': n_keep,
    }


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_figures(all_results, all_profiles, all_overlap, results_dir):
    """Generate figures for the experiment."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sel_colors = {
        'true_h2o': '#4CAF50',
        'k_norm_h2o': '#2196F3',
        'random': '#9E9E9E',
        'recent': '#FF9800',
        'sink_excluded_h2o': '#E91E63',
        'late_layer_h2o': '#9C27B0',
    }
    sel_labels = {
        'true_h2o': 'True H2O',
        'k_norm_h2o': 'K-Norm',
        'random': 'Random',
        'recent': 'Recent',
        'sink_excluded_h2o': 'Sink-Excl H2O',
        'late_layer_h2o': 'Late-Layer H2O',
    }

    for model_key, model_results in all_results.items():
        if not model_results:
            continue
        model_short = model_key.split("/")[-1]

        # ── Figure 1: Budget crossover curve ──
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        for sel in BASE_SELECTIONS + MODIFIED_SELECTIONS:
            budgets_plot = []
            accs_plot = []
            for budget in BUDGETS:
                key = f"mask_{sel}_{budget:.2f}"
                if key in model_results:
                    r = model_results[key]
                    budgets_plot.append(budget * 100)
                    accs_plot.append(r['accuracy'] * 100)
            if budgets_plot:
                marker = 'o' if sel in BASE_SELECTIONS else 's'
                linestyle = '-' if sel in BASE_SELECTIONS else '--'
                ax.plot(budgets_plot, accs_plot, marker=marker, linestyle=linestyle,
                       color=sel_colors.get(sel, '#000'),
                       label=sel_labels.get(sel, sel), linewidth=2, markersize=8)

        ax.set_xlabel('Budget (% of reasoning positions kept)', fontsize=13)
        ax.set_ylabel('Answer Accuracy (%)', fontsize=13)
        ax.set_title(f'Budget Crossover: True H2O vs K-Norm — {model_short}', fontsize=14)
        ax.set_ylim(0, 110)
        ax.set_xlim(20, 80)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'budget_crossover_{model_short}.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

        # ── Figure 2: Position profile comparison at 33% ──
        if model_key in all_profiles and all_profiles[model_key]:
            profiles = all_profiles[model_key]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            for ax_idx, sel in enumerate(['true_h2o', 'k_norm_h2o']):
                ax = axes[ax_idx]
                # Collect quintile fracs across all problems
                all_quintiles = []
                for p in profiles:
                    key = f'{sel}_0.33'
                    if key in p and 'quintile_fracs' in p[key]:
                        all_quintiles.append(p[key]['quintile_fracs'])

                if all_quintiles:
                    mean_q = np.mean(all_quintiles, axis=0)
                    std_q = np.std(all_quintiles, axis=0)
                    x = np.arange(5)
                    ax.bar(x, mean_q * 100, yerr=std_q * 100,
                          color=sel_colors[sel], alpha=0.8, capsize=5,
                          edgecolor='black', linewidth=0.5)
                    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5,
                              label='Uniform (20%)')
                    ax.set_xticks(x)
                    ax.set_xticklabels(['Q1\n(earliest)', 'Q2', 'Q3', 'Q4', 'Q5\n(latest)'],
                                      fontsize=10)
                    ax.set_ylabel('% of Selected Positions', fontsize=12)
                    ax.set_title(f'{sel_labels[sel]} Position Profile at 33%', fontsize=13)
                    ax.set_ylim(0, max(60, max(mean_q * 100) + 15))
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3, axis='y')

            fig.suptitle(f'Where Do Selections Land? — {model_short}', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'position_profile_{model_short}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()

            # ── Figure 3: Sink fraction comparison ──
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            sel_list = ['true_h2o', 'k_norm_h2o', 'random', 'sink_excluded_h2o', 'late_layer_h2o']
            sink_data = {}
            for sel in sel_list:
                fracs = []
                for p in profiles:
                    key = f'{sel}_0.33'
                    if key in p and 'sink_frac' in p[key]:
                        fracs.append(p[key]['sink_frac'])
                if fracs:
                    sink_data[sel] = fracs

            if sink_data:
                x = np.arange(len(sink_data))
                sels_present = list(sink_data.keys())
                means = [np.mean(sink_data[s]) * 100 for s in sels_present]
                stds = [np.std(sink_data[s]) * 100 for s in sels_present]
                colors = [sel_colors.get(s, '#000') for s in sels_present]

                ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                      capsize=5, edgecolor='black', linewidth=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels([sel_labels.get(s, s) for s in sels_present],
                                  fontsize=10, rotation=15)
                ax.set_ylabel('Sink Positions as % of Selection', fontsize=12)
                ax.set_title(f'Sink Dominance at 33% Budget — {model_short}', fontsize=13)
                ax.grid(True, alpha=0.3, axis='y')

                for i, (m, s) in enumerate(zip(means, stds)):
                    ax.text(i, m + s + 1, f'{m:.1f}%', ha='center', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'sink_dominance_{model_short}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()

            # ── Figure 4: Mean position of selections ──
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            for sel in ['true_h2o', 'k_norm_h2o', 'sink_excluded_h2o', 'late_layer_h2o']:
                mean_positions = []
                for p in profiles:
                    key = f'{sel}_0.33'
                    if key in p and 'mean_pos' in p[key]:
                        mean_positions.append(p[key]['mean_pos'])
                if mean_positions:
                    ax.hist(mean_positions, bins=20, alpha=0.5,
                           color=sel_colors.get(sel, '#000'),
                           label=f'{sel_labels.get(sel, sel)} (mean={np.mean(mean_positions):.3f})',
                           edgecolor='black', linewidth=0.5)

            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Center (0.5)')
            ax.set_xlabel('Mean Normalized Position of Selection', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'Mean Position of Selected Positions at 33% — {model_short}', fontsize=13)
            ax.legend(fontsize=10)
            ax.set_xlim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'mean_position_{model_short}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()

        # ── Figure 5: Modified H2O comparison at 33% and 50% ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        for ax_idx, budget in enumerate([0.33, 0.50]):
            ax = axes[ax_idx]
            all_sels = ['true_h2o', 'sink_excluded_h2o', 'late_layer_h2o',
                       'k_norm_h2o', 'random', 'recent']
            accs = []
            errs_lo = []
            errs_hi = []
            colors = []
            labels = []

            for sel in all_sels:
                key = f"mask_{sel}_{budget:.2f}"
                if key in model_results:
                    r = model_results[key]
                    acc = r['accuracy'] * 100
                    lo, hi = wilson_ci(r['n_correct'], r['n_valid'])
                    accs.append(acc)
                    errs_lo.append(acc - lo * 100)
                    errs_hi.append(hi * 100 - acc)
                    colors.append(sel_colors.get(sel, '#000'))
                    labels.append(sel_labels.get(sel, sel))

            if accs:
                x = np.arange(len(accs))
                ax.bar(x, accs, color=colors, alpha=0.85, width=0.6,
                      edgecolor='black', linewidth=0.5)
                ax.errorbar(x, accs, yerr=[errs_lo, errs_hi],
                           fmt='none', color='black', capsize=5)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=9, rotation=20, ha='right')
                ax.set_ylabel('Answer Accuracy (%)', fontsize=12)
                ax.set_title(f'All Selections at {int(budget*100)}% Budget (Masking)', fontsize=13)
                ax.set_ylim(0, 110)
                ax.grid(True, alpha=0.3, axis='y')

                for i, acc in enumerate(accs):
                    ax.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')

        fig.suptitle(f'Modified H2O Strategies — {model_short}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'modified_h2o_{model_short}.png'),
                   dpi=150, bbox_inches='tight')
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
    late_start = int(num_layers * LATE_LAYER_FRACTION)
    print(f"  Layers: {num_layers}, Late-layer start: {late_start}")

    results = {}
    per_problem_profiles = []
    overlap_stats = []
    start_time = time.time()
    n_valid = 0
    n_attempted = 0
    n_h2o_computed = 0

    # Build all conditions to test
    conditions = []
    for budget in BUDGETS:
        for sel in BASE_SELECTIONS:
            conditions.append(('mask', sel, budget))
    for budget in MODIFIED_BUDGETS:
        for sel in MODIFIED_SELECTIONS:
            conditions.append(('mask', sel, budget))

    print(f"  Total conditions: {len(conditions)}")

    for prob_idx, problem in enumerate(problems):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            print(f"\n  TIME LIMIT at problem {prob_idx} ({elapsed:.0f}s)")
            break

        question = problem["question"]
        true_answer = normalize_answer(problem["answer"].split("####")[-1].strip())
        prompt = build_prompt(question)

        try:
            cache_result = generate_and_build_cache(model, tokenizer, prompt)
        except Exception as e:
            print(f"  Problem {prob_idx}: generation error: {e}")
            continue

        if cache_result is None:
            continue

        n_attempted += 1
        gen_text = cache_result['gen_text']
        clean_answer = normalize_answer(extract_answer(gen_text))
        if clean_answer != true_answer:
            del cache_result['full_cache']
            gc.collect(); torch.cuda.empty_cache()
            continue

        full_cache = cache_result['full_cache']
        prompt_len = cache_result['prompt_len']
        reasoning_len = cache_result['reasoning_len']
        k_norm_importance = cache_result['k_norm_importance']
        full_seq = cache_result['full_seq']
        n_valid += 1

        # Compute true H2O importance (all-layer and late-layer)
        true_h2o_all, true_h2o_late = compute_true_h2o_importance(
            model, full_seq, prompt_len, reasoning_len, num_layers)

        if true_h2o_all is not None:
            n_h2o_computed += 1

        # Position profile analysis for each selection at 33%
        prob_profile = {'prob_idx': prob_idx, 'reasoning_len': reasoning_len}

        # Compute Spearman correlation between K-norm and true H2O at this problem
        if true_h2o_all is not None:
            from scipy import stats
            reason_knorm = k_norm_importance[prompt_len:prompt_len + reasoning_len]
            rho, p_val = stats.spearmanr(reason_knorm, true_h2o_all)
            prob_profile['spearman_rho'] = float(rho)
            prob_profile['spearman_p'] = float(p_val)

            # Also check late-layer vs K-norm
            if true_h2o_late is not None:
                rho_late, _ = stats.spearmanr(reason_knorm, true_h2o_late)
                prob_profile['spearman_rho_late_vs_knorm'] = float(rho_late)

        # Profile selections at 33% for each method
        for sel in BASE_SELECTIONS + MODIFIED_SELECTIONS:
            try:
                _, _, keep_idx = select_positions(
                    sel, 0.33, reasoning_len, prompt_len,
                    k_norm_importance, true_h2o_all, true_h2o_late)
                profile = analyze_position_profile(keep_idx, reasoning_len)
                prob_profile[f'{sel}_0.33'] = profile
            except Exception:
                pass

        per_problem_profiles.append(prob_profile)

        # Test each condition
        for method, selection, budget in conditions:
            key = f"{method}_{selection}_{budget:.2f}"
            try:
                keep_pos, evict_pos, _ = select_positions(
                    selection, budget, reasoning_len, prompt_len,
                    k_norm_importance, true_h2o_all, true_h2o_late)

                answer_text = evict_and_generate(
                    model, tokenizer, full_cache, prompt_len, reasoning_len,
                    evict_pos, num_layers)

                pred = normalize_answer(extract_answer(answer_text))
                correct = (pred == true_answer)

                if key not in results:
                    results[key] = {"n_correct": 0, "n_valid": 0, "answers": []}
                results[key]["n_correct"] += int(correct)
                results[key]["n_valid"] += 1
                results[key]["answers"].append({
                    "prob_idx": prob_idx, "correct": correct,
                    "pred": pred, "true": true_answer
                })

            except Exception as e:
                print(f"  Problem {prob_idx}, {key}: error: {e}")
                continue

        del full_cache
        gc.collect(); torch.cuda.empty_cache()

        if n_valid % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Valid: {n_valid}, Attempted: {n_attempted}, "
                  f"H2O: {n_h2o_computed}/{n_valid}, "
                  f"Elapsed: {elapsed:.0f}s, Rate: {elapsed/max(1,n_valid):.1f}s/prob")

    # Compute accuracy
    for key in results:
        r = results[key]
        r["accuracy"] = r["n_correct"] / max(1, r["n_valid"])

    print(f"\n  True H2O computed: {n_h2o_computed}/{n_valid}")

    # Print results summary
    print(f"\n{'─'*90}")
    print(f"RESULTS: {model_name} (n_valid={n_valid})")
    print(f"{'─'*90}")

    # Budget sweep table
    print(f"\n  {'Budget':<8}", end="")
    for sel in BASE_SELECTIONS + MODIFIED_SELECTIONS:
        print(f"  {sel:<18}", end="")
    print()
    print(f"  {'─'*8}", end="")
    for _ in range(len(BASE_SELECTIONS) + len(MODIFIED_SELECTIONS)):
        print(f"  {'─'*18}", end="")
    print()

    for budget in BUDGETS:
        line = f"  {int(budget*100):>3}%    "
        for sel in BASE_SELECTIONS + MODIFIED_SELECTIONS:
            key = f"mask_{sel}_{budget:.2f}"
            if key in results:
                r = results[key]
                line += f"  {r['accuracy']*100:>6.1f}%           "
            else:
                line += f"  {'—':>6}             "
        print(line)

    # Position profile summary
    if per_problem_profiles:
        print(f"\n  POSITION PROFILE SUMMARY (at 33% budget):")
        for sel in ['true_h2o', 'k_norm_h2o', 'sink_excluded_h2o', 'late_layer_h2o']:
            sink_fracs = []
            mean_positions = []
            for p in per_problem_profiles:
                pkey = f'{sel}_0.33'
                if pkey in p and 'sink_frac' in p[pkey]:
                    sink_fracs.append(p[pkey]['sink_frac'])
                    mean_positions.append(p[pkey]['mean_pos'])
            if sink_fracs:
                print(f"    {sel:20s}: sink_frac={np.mean(sink_fracs)*100:.1f}% "
                      f"± {np.std(sink_fracs)*100:.1f}%, "
                      f"mean_pos={np.mean(mean_positions):.3f} "
                      f"± {np.std(mean_positions):.3f}")

        # Spearman rho summary
        rhos = [p['spearman_rho'] for p in per_problem_profiles if 'spearman_rho' in p]
        if rhos:
            print(f"\n    K-norm vs True H2O Spearman rho: "
                  f"mean={np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
        rhos_late = [p['spearman_rho_late_vs_knorm'] for p in per_problem_profiles
                     if 'spearman_rho_late_vs_knorm' in p]
        if rhos_late:
            print(f"    K-norm vs Late-Layer H2O Spearman rho: "
                  f"mean={np.mean(rhos_late):.3f} ± {np.std(rhos_late):.3f}")

    del model
    gc.collect(); torch.cuda.empty_cache()

    return results, per_problem_profiles, n_valid, n_attempted


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
    print(f"Budget sweep: {[f'{b:.0%}' for b in BUDGETS]}")
    print(f"Base selections: {BASE_SELECTIONS}")
    print(f"Modified selections: {MODIFIED_SELECTIONS}")

    all_results = {}
    all_profiles = {}
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

        model_results, profiles, n_valid, n_attempted = run_model(
            model_name, problems, model_time)

        all_results[model_name] = model_results
        all_profiles[model_name] = profiles
        all_stats[model_name] = {
            "n_valid": n_valid, "n_attempted": n_attempted,
            "time_used": time.time() - global_start
        }

    # Generate figures
    generate_figures(all_results, all_profiles, {}, RESULTS_DIR)

    # Save results
    summary = {
        "experiment": "058_sink_analysis_budget_sweep",
        "budgets": BUDGETS,
        "base_selections": BASE_SELECTIONS,
        "modified_selections": MODIFIED_SELECTIONS,
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

    # Save profile summaries
    summary["profile_summaries"] = {}
    for model_name, profiles in all_profiles.items():
        model_summary = {}
        for sel in BASE_SELECTIONS + MODIFIED_SELECTIONS:
            sink_fracs = []
            mean_positions = []
            quintile_fracs_all = []
            for p in profiles:
                pkey = f'{sel}_0.33'
                if pkey in p and 'sink_frac' in p[pkey]:
                    sink_fracs.append(p[pkey]['sink_frac'])
                    mean_positions.append(p[pkey]['mean_pos'])
                    quintile_fracs_all.append(p[pkey]['quintile_fracs'])
            if sink_fracs:
                model_summary[sel] = {
                    'sink_frac_mean': float(np.mean(sink_fracs)),
                    'sink_frac_std': float(np.std(sink_fracs)),
                    'mean_pos_mean': float(np.mean(mean_positions)),
                    'mean_pos_std': float(np.std(mean_positions)),
                    'quintile_fracs_mean': np.mean(quintile_fracs_all, axis=0).tolist(),
                    'quintile_fracs_std': np.std(quintile_fracs_all, axis=0).tolist(),
                }
        rhos = [p['spearman_rho'] for p in profiles if 'spearman_rho' in p]
        if rhos:
            model_summary['spearman_rho_knorm_vs_h2o'] = {
                'mean': float(np.mean(rhos)),
                'std': float(np.std(rhos)),
            }
        rhos_late = [p.get('spearman_rho_late_vs_knorm') for p in profiles
                     if 'spearman_rho_late_vs_knorm' in p]
        if rhos_late:
            model_summary['spearman_rho_knorm_vs_late_h2o'] = {
                'mean': float(np.mean(rhos_late)),
                'std': float(np.std(rhos_late)),
            }
        summary["profile_summaries"][model_name] = model_summary

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_convert)

    # Save per-problem profiles (compact version without full quintile arrays)
    compact_profiles = {}
    for model_name, profiles in all_profiles.items():
        compact = []
        for p in profiles:
            entry = {
                'prob_idx': p.get('prob_idx'),
                'reasoning_len': p.get('reasoning_len'),
                'spearman_rho': p.get('spearman_rho'),
                'spearman_rho_late_vs_knorm': p.get('spearman_rho_late_vs_knorm'),
            }
            for sel in ['true_h2o', 'k_norm_h2o', 'sink_excluded_h2o', 'late_layer_h2o']:
                pkey = f'{sel}_0.33'
                if pkey in p:
                    entry[f'{sel}_sink_frac'] = p[pkey].get('sink_frac')
                    entry[f'{sel}_mean_pos'] = p[pkey].get('mean_pos')
            compact.append(entry)
        compact_profiles[model_name] = compact

    with open(os.path.join(RESULTS_DIR, "profiles.json"), "w") as f:
        json.dump(compact_profiles, f, indent=2, default=_convert)

    total_time = time.time() - global_start
    print(f"\nTotal time: {total_time:.0f}s")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
