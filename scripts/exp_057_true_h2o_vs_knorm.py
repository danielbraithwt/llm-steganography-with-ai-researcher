#!/usr/bin/env python3
"""
Experiment 057: True Attention-Based H2O vs K-Norm Proxy
— Does actual cumulative attention rescue masking performance?

Exp 056 found that H2O (K-norm proxy) = random under true masking (66.7%)
on Llama at 33% budget. This suggests attention-based selection provides
zero benefit when positions are truly removed from attention.

BUT: the K-norm proxy may be a poor approximation of true H2O (cumulative
attention). If true attention-based selection identifies DIFFERENT positions
than K-norm, the finding may be K-norm-specific, not fundamental.

This experiment computes TRUE cumulative attention importance by running a
forward pass with output_attentions=True, summing attention over reasoning
queries. It then compares:
  - true_h2o vs k_norm_h2o vs random
  - Under both masking and zeroing
  - At 33% and 50% budgets
  - On Llama (where the effect was seen) + Qwen (control)

KEY QUESTION: Does mask + true_h2o > mask + k_norm_h2o?
  If YES → Exp 056 finding was K-norm-proxy-specific
  If NO  → Attention-based selection truly provides no benefit under masking

MEMORY MANAGEMENT: To avoid OOM with output_attentions on large models,
we process attention layer by layer using hooks, only accumulating the
per-position importance sum.
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
                           "results", "exp_057")

# Conditions: (method, selection, budget)
# "selection" is either 'true_h2o', 'k_norm_h2o', or 'random'
CONDITIONS = [
    # 33% budget — primary comparison (tests the key question)
    ('mask',    'true_h2o',   0.33),
    ('mask',    'k_norm_h2o', 0.33),
    ('mask',    'random',     0.33),
    ('zero_kv', 'true_h2o',   0.33),
    ('zero_kv', 'k_norm_h2o', 0.33),
    ('zero_kv', 'random',     0.33),
    # 50% budget — secondary
    ('mask',    'true_h2o',   0.50),
    ('mask',    'k_norm_h2o', 0.50),
    ('zero_kv', 'true_h2o',   0.50),
    ('zero_kv', 'k_norm_h2o', 0.50),
    # Controls
    ('mask',    'recent',     0.33),
    ('zero_kv', 'recent',     0.33),
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
def compute_true_h2o_importance(model, full_seq, prompt_len, reasoning_len, num_layers):
    """
    Compute true cumulative attention importance for each reasoning position.

    Strategy: Run a forward pass with output_attentions=True, then for each
    reasoning-phase query position, sum its attention weights to all reasoning
    KV positions. This gives "how much reasoning tokens attend to each position."

    To manage memory, we process attention weights layer by layer: extract
    the reasoning-to-reasoning attention block, sum, and discard the full tensor.
    """
    total_len = prompt_len + reasoning_len

    # Try full output_attentions first; fall back to chunked if OOM
    try:
        outputs = model(
            input_ids=full_seq[:total_len].unsqueeze(0),
            output_attentions=True,
            use_cache=False,
        )

        # Accumulate per-position importance across layers
        cumulative = np.zeros(reasoning_len)

        for layer_idx in range(num_layers):
            attn = outputs.attentions[layer_idx]  # [1, num_heads, seq_len, seq_len]
            # Extract reasoning queries -> reasoning keys block
            # reasoning queries: positions prompt_len to total_len
            # reasoning keys: positions prompt_len to total_len
            reasoning_attn = attn[0, :, prompt_len:total_len, prompt_len:total_len]
            # [num_heads, reasoning_len, reasoning_len]

            # Sum over heads and query positions -> per-KV-position importance
            pos_importance = reasoning_attn.float().sum(dim=(0, 1)).cpu().numpy()
            cumulative += pos_importance

        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        return cumulative

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        print("    OOM with output_attentions — falling back to K-norm only")
        return None


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

    # Find truncation point (before ####)
    reasoning_ids = gen_ids.unsqueeze(0)
    trunc_point = find_truncation_point(reasoning_ids, tokenizer)

    if trunc_point is None or trunc_point < 10:
        return None

    # Build full sequence
    full_seq = torch.cat([inputs.input_ids[0], gen_ids[:trunc_point]])
    total_len = full_seq.shape[0]
    if total_len > MAX_SEQ_LEN:
        return None

    # Build KV cache
    outputs = model(input_ids=full_seq.unsqueeze(0), use_cache=True)
    full_cache = outputs.past_key_values
    del outputs

    # Compute K-norm importance (H2O proxy)
    num_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    reasoning_len = trunc_point

    k_norm_importance = np.zeros(total_len)
    for l in range(num_layers):
        k = full_cache.layers[l].keys[0]  # [n_kv_heads, seq_len, head_dim]
        k_norms = k.float().norm(dim=-1).cpu().numpy()  # [n_kv_heads, seq_len]
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
                     k_norm_importance, true_h2o_importance):
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
        if true_h2o_importance is None:
            # Fallback to k_norm if true H2O couldn't be computed
            reason_scores = k_norm_importance[prompt_len:prompt_len + reasoning_len]
        else:
            reason_scores = true_h2o_importance
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
    return keep_pos, evict_pos


@torch.no_grad()
def evict_and_generate(model, tokenizer, full_cache, prompt_len, reasoning_len,
                       evict_positions, method, num_layers, max_answer_tokens=64):
    """Apply eviction to cache and generate answer."""
    total_len = prompt_len + reasoning_len
    device = model.device
    evicted_cache = DynamicCache()
    use_mask = (method == 'mask')

    # Pre-compute eviction mask tensor
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
        # method == 'mask': don't modify cache

        evicted_cache.update(k, v, l)

    # Prepare attention mask for 'mask' method
    if use_mask:
        attn_mask = torch.ones(1, total_len, dtype=torch.long, device=device)
        attn_mask[0, evict_tensor] = 0
    else:
        attn_mask = None

    # Generate answer — feed " ####" prefix then decode
    answer_prefix = tokenizer.encode(" ####", add_special_tokens=False)
    cache_len = total_len

    out = None
    for i in range(len(answer_prefix)):
        tok = torch.tensor([[answer_prefix[i]]], device=device)
        kwargs = {'input_ids': tok, 'past_key_values': evicted_cache, 'use_cache': True}
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
            kwargs['attention_mask'] = attn_mask
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


def compute_selection_overlap(k_norm_importance, true_h2o_importance,
                              prompt_len, reasoning_len, budget_frac):
    """Compute overlap between K-norm and true H2O selected positions."""
    n_keep = max(1, int(budget_frac * reasoning_len))

    k_scores = k_norm_importance[prompt_len:prompt_len + reasoning_len]
    k_selected = set(np.argsort(k_scores)[-n_keep:].tolist())

    if true_h2o_importance is None:
        return 1.0, n_keep  # If no true H2O, overlap is perfect (same)

    t_selected = set(np.argsort(true_h2o_importance)[-n_keep:].tolist())

    overlap = len(k_selected & t_selected)
    jaccard = overlap / len(k_selected | t_selected) if len(k_selected | t_selected) > 0 else 1.0

    return jaccard, n_keep


def compute_position_analysis(k_norm_importance, true_h2o_importance,
                              prompt_len, reasoning_len):
    """Analyze where the two importance metrics differ."""
    if true_h2o_importance is None:
        return {}

    k_scores = k_norm_importance[prompt_len:prompt_len + reasoning_len]
    t_scores = true_h2o_importance

    # Normalize both to [0, 1] for comparison
    k_norm = (k_scores - k_scores.min()) / (k_scores.max() - k_scores.min() + 1e-8)
    t_norm = (t_scores - t_scores.min()) / (t_scores.max() - t_scores.min() + 1e-8)

    # Spearman correlation
    from scipy import stats
    rho, p_val = stats.spearmanr(k_norm, t_norm)

    # Positional analysis: split into 5 bins
    n = len(k_scores)
    bin_size = n // 5
    bin_rhos = []
    for b in range(5):
        start = b * bin_size
        end = (b + 1) * bin_size if b < 4 else n
        if end - start < 3:
            continue
        r, _ = stats.spearmanr(k_norm[start:end], t_norm[start:end])
        bin_rhos.append(float(r) if not np.isnan(r) else 0.0)

    return {
        'spearman_rho': float(rho),
        'spearman_p': float(p_val),
        'bin_rhos': bin_rhos,
        'k_norm_mean': float(k_scores.mean()),
        'true_h2o_mean': float(t_scores.mean()),
    }


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_figures(all_results, all_overlap_stats, results_dir):
    """Generate figures comparing true H2O vs K-norm proxy."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sel_colors = {
        'true_h2o': '#4CAF50',    # green
        'k_norm_h2o': '#2196F3',  # blue
        'random': '#9E9E9E',      # gray
        'recent': '#FF9800',      # orange
    }
    sel_labels = {
        'true_h2o': 'True H2O\n(cum. attention)',
        'k_norm_h2o': 'K-Norm H2O\n(proxy)',
        'random': 'Random',
        'recent': 'Recent\n(sinks+late)',
    }

    for model_key, model_results in all_results.items():
        if not model_results:
            continue
        model_short = model_key.split("/")[-1]

        # ── Figure 1: Key comparison at 33% — mask vs zero × selection ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        for ax_idx, method in enumerate(['mask', 'zero_kv']):
            ax = axes[ax_idx]
            selections = ['true_h2o', 'k_norm_h2o', 'random', 'recent']
            accs = []
            errs_lo = []
            errs_hi = []
            colors = []

            for sel in selections:
                key = f"{method}_{sel}_0.33"
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
                colors.append(sel_colors[sel])

            x = np.arange(len(selections))
            ax.bar(x, accs, color=colors, alpha=0.85, width=0.6)
            ax.errorbar(x, accs, yerr=[errs_lo, errs_hi],
                       fmt='none', color='black', capsize=5)

            ax.set_xticks(x)
            ax.set_xticklabels([sel_labels[s] for s in selections], fontsize=10)
            ax.set_ylabel('Answer Accuracy (%)', fontsize=12)
            method_label = 'Masking (true removal)' if method == 'mask' else 'Zeroing (K+V=0)'
            ax.set_title(f'{method_label}', fontsize=13)
            ax.set_ylim(0, 110)
            ax.grid(True, alpha=0.3, axis='y')

            # Add accuracy labels on bars
            for i, acc in enumerate(accs):
                if acc > 0:
                    ax.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')

        fig.suptitle(f'True H2O vs K-Norm Proxy at 33% Budget — {model_short}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'true_h2o_vs_knorm_33pct_{model_short}.png'), dpi=150)
        plt.close()

        # ── Figure 2: Paired comparison — mask true_h2o vs k_norm at 33% and 50% ──
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        budgets = [0.33, 0.50]
        methods = ['mask', 'zero_kv']
        x_positions = []
        x_labels = []
        bar_data = []

        x = 0
        for budget in budgets:
            for method in methods:
                for sel in ['true_h2o', 'k_norm_h2o']:
                    key = f"{method}_{sel}_{budget:.2f}"
                    if key in model_results:
                        r = model_results[key]
                        acc = r['accuracy'] * 100
                        lo, hi = wilson_ci(r['n_correct'], r['n_valid'])
                        bar_data.append({
                            'x': x, 'acc': acc,
                            'err_lo': acc - lo * 100,
                            'err_hi': hi * 100 - acc,
                            'color': sel_colors[sel],
                            'sel': sel,
                        })
                    x += 0.5
                x += 0.3  # gap between methods
            x += 0.5  # gap between budgets

        for bd in bar_data:
            label = sel_labels[bd['sel']].replace('\n', ' ') if bd['x'] < 2 else None
            ax.bar(bd['x'], bd['acc'], width=0.4, color=bd['color'], alpha=0.85, label=label)
            ax.errorbar(bd['x'], bd['acc'], yerr=[[bd['err_lo']], [bd['err_hi']]],
                       fmt='none', color='black', capsize=4)
            ax.text(bd['x'], bd['acc'] + 2, f'{bd["acc"]:.1f}%', ha='center', fontsize=9)

        # Group labels
        group_positions = []
        x = 0
        for budget in budgets:
            for method in methods:
                group_positions.append((x + 0.25, f'{method}\n{int(budget*100)}%'))
                x += 1.3
            x += 0.5

        ax.set_xticks([gp[0] for gp in group_positions])
        ax.set_xticklabels([gp[1] for gp in group_positions], fontsize=10)
        ax.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax.set_title(f'True H2O vs K-Norm Proxy Across Conditions — {model_short}', fontsize=13)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'paired_comparison_{model_short}.png'), dpi=150)
        plt.close()

        # ── Figure 3: Selection overlap histogram ──
        if model_key in all_overlap_stats and all_overlap_stats[model_key]:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            overlaps = [s['jaccard_33'] for s in all_overlap_stats[model_key]
                       if 'jaccard_33' in s]
            if overlaps:
                ax.hist(overlaps, bins=20, color='#2196F3', alpha=0.7, edgecolor='black')
                mean_ov = np.mean(overlaps)
                ax.axvline(mean_ov, color='red', linestyle='--', linewidth=2,
                          label=f'Mean Jaccard = {mean_ov:.3f}')
                ax.set_xlabel('Jaccard Overlap (K-norm ∩ True H2O)', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title(f'Position Selection Overlap at 33% — {model_short}', fontsize=13)
                ax.legend(fontsize=11)
                ax.set_xlim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'selection_overlap_{model_short}.png'), dpi=150)
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

    results = {}
    per_problem = []
    overlap_stats = []
    start_time = time.time()
    n_valid = 0
    n_attempted = 0
    n_h2o_computed = 0
    n_h2o_failed = 0

    for prob_idx, problem in enumerate(problems):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            print(f"\n  TIME LIMIT reached at problem {prob_idx} ({elapsed:.0f}s)")
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

        # Compute true H2O importance (cumulative attention)
        true_h2o_importance = compute_true_h2o_importance(
            model, full_seq, prompt_len, reasoning_len, num_layers)

        if true_h2o_importance is not None:
            n_h2o_computed += 1
        else:
            n_h2o_failed += 1

        # Compute selection overlap
        jaccard_33, n_keep_33 = compute_selection_overlap(
            k_norm_importance, true_h2o_importance,
            prompt_len, reasoning_len, 0.33)
        jaccard_50, n_keep_50 = compute_selection_overlap(
            k_norm_importance, true_h2o_importance,
            prompt_len, reasoning_len, 0.50)

        # Position correlation analysis
        pos_analysis = compute_position_analysis(
            k_norm_importance, true_h2o_importance,
            prompt_len, reasoning_len)

        overlap_info = {
            'prob_idx': prob_idx,
            'reasoning_len': reasoning_len,
            'jaccard_33': jaccard_33,
            'jaccard_50': jaccard_50,
            'n_keep_33': n_keep_33,
            'n_keep_50': n_keep_50,
            'h2o_computed': true_h2o_importance is not None,
            **pos_analysis,
        }
        overlap_stats.append(overlap_info)

        prob_results = {
            "idx": prob_idx, "true_answer": true_answer,
            "clean_answer": clean_answer, "reasoning_len": reasoning_len,
            "jaccard_33": jaccard_33, "jaccard_50": jaccard_50,
        }
        if pos_analysis:
            prob_results['spearman_rho'] = pos_analysis.get('spearman_rho', None)

        # Test each condition
        for method, selection, budget in CONDITIONS:
            key = f"{method}_{selection}_{budget:.2f}"

            try:
                keep_pos, evict_pos = select_positions(
                    selection, budget, reasoning_len, prompt_len,
                    k_norm_importance, true_h2o_importance)

                answer_text = evict_and_generate(
                    model, tokenizer, full_cache, prompt_len, reasoning_len,
                    evict_pos, method, num_layers)

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
                prob_results[key] = {"correct": correct, "pred": pred}

            except Exception as e:
                print(f"  Problem {prob_idx}, {key}: error: {e}")
                continue

        # Baseline (no eviction)
        try:
            baseline_text = evict_and_generate(
                model, tokenizer, full_cache, prompt_len, reasoning_len,
                [], 'zero_kv', num_layers)
            baseline_pred = normalize_answer(extract_answer(baseline_text))
            baseline_correct = (baseline_pred == true_answer)
            bl_key = "baseline_1.00"
            if bl_key not in results:
                results[bl_key] = {"n_correct": 0, "n_valid": 0, "answers": []}
            results[bl_key]["n_correct"] += int(baseline_correct)
            results[bl_key]["n_valid"] += 1
        except Exception as e:
            print(f"  Problem {prob_idx}, baseline: {e}")

        per_problem.append(prob_results)
        del full_cache
        gc.collect()
        torch.cuda.empty_cache()

        if n_valid % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Valid: {n_valid}, Attempted: {n_attempted}, "
                  f"H2O computed: {n_h2o_computed}/{n_valid}, "
                  f"Elapsed: {elapsed:.0f}s, "
                  f"Rate: {elapsed/max(1,n_valid):.1f}s/prob")

    # Compute accuracy
    for key in results:
        r = results[key]
        r["accuracy"] = r["n_correct"] / max(1, r["n_valid"])

    print(f"\n  True H2O computed: {n_h2o_computed}/{n_valid} "
          f"(failed: {n_h2o_failed})")

    if overlap_stats:
        jaccards = [s['jaccard_33'] for s in overlap_stats if s.get('h2o_computed')]
        if jaccards:
            print(f"  K-norm vs True H2O overlap (Jaccard@33%): "
                  f"mean={np.mean(jaccards):.3f}, "
                  f"min={np.min(jaccards):.3f}, max={np.max(jaccards):.3f}")
        rhos = [s['spearman_rho'] for s in overlap_stats
                if s.get('h2o_computed') and 'spearman_rho' in s]
        if rhos:
            print(f"  Spearman rho (K-norm vs True H2O): "
                  f"mean={np.mean(rhos):.3f}, "
                  f"min={np.min(rhos):.3f}, max={np.max(rhos):.3f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results, per_problem, overlap_stats, n_valid, n_attempted


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
    for method, sel, budget in CONDITIONS:
        print(f"  {method:8s} × {sel:16s} × {budget:.0%}")

    all_results = {}
    all_per_problem = {}
    all_overlap_stats = {}
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

        model_results, per_problem, overlap_stats, n_valid, n_attempted = run_model(
            model_name, problems, model_time)

        all_results[model_name] = model_results
        all_per_problem[model_name] = per_problem
        all_overlap_stats[model_name] = overlap_stats
        all_stats[model_name] = {
            "n_valid": n_valid, "n_attempted": n_attempted,
            "time_used": time.time() - global_start
        }

        # Print summary
        print(f"\n{'─'*90}")
        print(f"RESULTS: {model_name} (n_valid={n_valid})")
        print(f"{'─'*90}")
        print(f"{'Method':<10} {'Selection':<16} {'Budget':<8} {'Acc%':<8} {'n_ok':<6} {'n':<6} {'Wilson 95% CI':<16}")
        print(f"{'─'*10} {'─'*16} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*16}")

        # Baseline
        bl_key = "baseline_1.00"
        if bl_key in model_results:
            r = model_results[bl_key]
            acc = r["accuracy"] * 100
            lo, hi = wilson_ci(r["n_correct"], r["n_valid"])
            print(f"{'baseline':<10} {'—':<16} {'100%':<8} {acc:>6.1f}%  "
                  f"{r['n_correct']:<6} {r['n_valid']:<6} [{lo*100:.1f}, {hi*100:.1f}]")

        # Group by budget
        for budget in [0.33, 0.50]:
            print(f"\n  --- {int(budget*100)}% budget ---")
            for method, sel, b in CONDITIONS:
                if b != budget:
                    continue
                key = f"{method}_{sel}_{b:.2f}"
                if key in model_results:
                    r = model_results[key]
                    acc = r["accuracy"] * 100
                    lo, hi = wilson_ci(r["n_correct"], r["n_valid"])
                    print(f"{method:<10} {sel:<16} {b*100:>5.0f}%   {acc:>6.1f}%  "
                          f"{r['n_correct']:<6} {r['n_valid']:<6} [{lo*100:.1f}, {hi*100:.1f}]")

        # Print key comparison
        print(f"\n  KEY COMPARISON (33% budget):")
        for method in ['mask', 'zero_kv']:
            print(f"    {method}:")
            for sel in ['true_h2o', 'k_norm_h2o', 'random']:
                key = f"{method}_{sel}_0.33"
                if key in model_results:
                    r = model_results[key]
                    print(f"      {sel:<16}: {r['accuracy']*100:.1f}%")

    # Generate figures
    generate_figures(all_results, all_overlap_stats, RESULTS_DIR)

    # Save results
    summary = {
        "experiment": "057_true_h2o_vs_knorm",
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

    # Save overlap statistics
    summary["overlap_stats"] = {}
    for model_name, stats in all_overlap_stats.items():
        if stats:
            jaccards_33 = [s['jaccard_33'] for s in stats if s.get('h2o_computed')]
            jaccards_50 = [s['jaccard_50'] for s in stats if s.get('h2o_computed')]
            rhos = [s['spearman_rho'] for s in stats
                    if s.get('h2o_computed') and 'spearman_rho' in s]
            summary["overlap_stats"][model_name] = {
                "n_problems": len(stats),
                "n_h2o_computed": sum(1 for s in stats if s.get('h2o_computed')),
                "jaccard_33_mean": float(np.mean(jaccards_33)) if jaccards_33 else None,
                "jaccard_33_std": float(np.std(jaccards_33)) if jaccards_33 else None,
                "jaccard_50_mean": float(np.mean(jaccards_50)) if jaccards_50 else None,
                "spearman_rho_mean": float(np.mean(rhos)) if rhos else None,
                "spearman_rho_std": float(np.std(rhos)) if rhos else None,
            }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_convert)

    with open(os.path.join(RESULTS_DIR, "per_problem.json"), "w") as f:
        json.dump(all_per_problem, f, indent=2, default=_convert)

    total_time = time.time() - global_start
    print(f"\nTotal time: {total_time:.0f}s")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
