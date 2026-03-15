#!/usr/bin/env python3
"""
Experiment 045: Per-Head K vs V Direction Perturbation on Qwen3-4B-Base

Tests whether the K > V hierarchy is uniform across attention heads or driven
by specific "answer-routing" heads. For each of 8 KV heads, perturb K-only
(or V-only) directions at that single head across ALL layers and ALL positions.

This is a fundamentally different perturbation geometry from positional sweeps:
- Positional: ALL heads at SOME positions (12.5% capacity via 1 head everywhere)
- Per-head: ONE head at ALL positions (12.5% capacity via all positions at 1 head)

DISCONFIRMATORY: If per-head K perturbation is well-tolerated (>70% acc),
K-routing has head-level redundancy, weakening the "fragile K-routing" narrative.
If some V-heads are as destructive as K-heads, the K > V hierarchy isn't absolute.

CONDITIONS:
  - 8 K-only per-head conditions: head_k_0 through head_k_7
  - 8 V-only per-head conditions: head_v_0 through head_v_7
"""

import os
import json
import time
import random
import gc
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B-Base"
NUM_PROBLEMS = 200  # attempt many, time-limited
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 2048
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_045")

# 8-shot exemplars (same as throughout program)
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


@torch.no_grad()
def generate_trace(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS):
    """Generate CoT trace. Returns (text, prompt_ids, reasoning_ids)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = inputs.input_ids
    generated_ids = []
    past_kv = None
    current_input = inputs.input_ids

    for step in range(max_tokens):
        if past_kv is not None:
            outputs = model(input_ids=current_input, past_key_values=past_kv, use_cache=True)
        else:
            outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        token_id = next_token[0, 0].item()
        generated_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break
        current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "####" in current_text:
            after = current_text.split("####")[-1]
            if re.search(r'\d+\s*\n', after) or re.search(r'\d+\s+\S', after):
                break
        if re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?\s*-?[\d,]+', current_text):
            break
        if "\nQ:" in current_text or "\n\nQ:" in current_text:
            idx = current_text.find("\nQ:")
            if idx > 0:
                truncated_text = current_text[:idx]
                generated_ids = tokenizer.encode(truncated_text, add_special_tokens=False)
            break
        current_input = next_token

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    reasoning_ids = torch.tensor([generated_ids], device=model.device)

    del past_kv, outputs
    gc.collect(); torch.cuda.empty_cache()
    return text, prompt_ids, reasoning_ids


@torch.no_grad()
def build_prompt_cache(model, prompt_ids):
    """Build clean KV cache for prompt tokens only."""
    outputs = model(input_ids=prompt_ids, use_cache=True)
    prompt_cache = outputs.past_key_values
    del outputs
    gc.collect(); torch.cuda.empty_cache()
    return prompt_cache


def perturb_direction_single_head(tensor, head_idx):
    """Replace direction of a single head with random, preserve norm.
    tensor shape: [1, n_heads, seq_len_so_far, head_dim]
    We perturb only head_idx at the last position.
    Returns the modified tensor (in-place) and perturbation L2.
    """
    orig = tensor[:, head_idx:head_idx+1, -1:, :].clone()
    noise = torch.randn_like(orig)
    norm = orig.norm(dim=-1, keepdim=True)
    noise_norm = noise.norm(dim=-1, keepdim=True) + 1e-8
    new_val = noise * (norm / noise_norm)
    tensor[:, head_idx:head_idx+1, -1:, :] = new_val
    perturb_l2 = (new_val - orig).float().norm().item() ** 2
    signal_l2 = orig.float().norm().item() ** 2
    return perturb_l2, signal_l2


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
def evaluate_per_head(model, tokenizer, prompt_cache, reasoning_tokens,
                      prompt_len, num_layers, true_answer,
                      target_head, component='k'):
    """
    Step through reasoning tokens, perturb direction at ONE head across all
    layers and all positions. Measure answer accuracy and text prediction accuracy.
    """
    reasoning_len = reasoning_tokens.shape[1]

    # Clone prompt cache
    cache = DynamicCache()
    for l in range(num_layers):
        k = prompt_cache.layers[l].keys.clone()
        v = prompt_cache.layers[l].values.clone()
        cache.update(k, v, l)

    text_correct = 0
    text_total = 0
    last_logits = None

    perturb_l2_total = 0.0
    signal_l2_total = 0.0
    perturb_count = 0

    for i in range(reasoning_len):
        token = reasoning_tokens[:, i:i+1]
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

        # Perturb the target head at this position (just added) across all layers
        for l in range(num_layers):
            if component in ('k', 'kv'):
                p_l2, s_l2 = perturb_direction_single_head(
                    cache.layers[l].keys, target_head)
                perturb_l2_total += p_l2
                signal_l2_total += s_l2
            if component in ('v', 'kv'):
                p_l2, s_l2 = perturb_direction_single_head(
                    cache.layers[l].values, target_head)
                perturb_l2_total += p_l2
                signal_l2_total += s_l2
            perturb_count += 1

        # Check prediction for next token
        if i < reasoning_len - 1:
            predicted = out.logits[:, -1, :].argmax(dim=-1).item()
            actual = reasoning_tokens[:, i + 1].item()
            if predicted == actual:
                text_correct += 1
            text_total += 1

        last_logits = out.logits
        del out

    text_accuracy = text_correct / text_total if text_total > 0 else 0.0

    # Generate answer
    next_tok = last_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_tok[0, 0].item()]
    gen_cache = cache
    del last_logits

    for _ in range(80):
        g = model(input_ids=next_tok, past_key_values=gen_cache, use_cache=True)
        gen_cache = g.past_key_values
        next_tok = g.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tid = next_tok[0, 0].item()
        generated.append(tid)
        if tid == tokenizer.eos_token_id:
            break
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if "####" in decoded:
            after = decoded.split("####")[-1]
            if re.search(r'\d[\d,]*\.?\d*\s*\n', after):
                break
            if re.search(r'\d[\d,]*\.?\d*\s+\S', after):
                break
        if re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?\s*-?[\d,]+', decoded):
            break
        if "\nQ:" in decoded:
            break
        del g

    answer_text = tokenizer.decode(generated, skip_special_tokens=True)
    raw_ans = extract_answer(answer_text) if answer_text else ""
    answer = normalize_answer(raw_ans) if raw_ans else ""
    correct = (answer == true_answer)

    perturb_rms = (perturb_l2_total / max(perturb_count, 1)) ** 0.5
    signal_rms = (signal_l2_total / max(perturb_count, 1)) ** 0.5

    del cache, gen_cache
    gc.collect(); torch.cuda.empty_cache()

    return {
        'correct': correct,
        'answer': answer,
        'text_accuracy': text_accuracy,
        'answer_text': answer_text[:200],
        'perturb_rms': perturb_rms,
        'signal_rms': signal_rms,
    }


@torch.no_grad()
def evaluate_clean(model, tokenizer, prompt_cache, reasoning_tokens,
                   prompt_len, num_layers, true_answer):
    """Clean evaluation (no perturbation)."""
    reasoning_len = reasoning_tokens.shape[1]
    cache = DynamicCache()
    for l in range(num_layers):
        k = prompt_cache.layers[l].keys.clone()
        v = prompt_cache.layers[l].values.clone()
        cache.update(k, v, l)

    text_correct = 0
    text_total = 0
    last_logits = None

    for i in range(reasoning_len):
        token = reasoning_tokens[:, i:i+1]
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

        if i < reasoning_len - 1:
            predicted = out.logits[:, -1, :].argmax(dim=-1).item()
            actual = reasoning_tokens[:, i + 1].item()
            if predicted == actual:
                text_correct += 1
            text_total += 1

        last_logits = out.logits
        del out

    text_accuracy = text_correct / text_total if text_total > 0 else 0.0

    next_tok = last_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_tok[0, 0].item()]
    gen_cache = cache
    del last_logits

    for _ in range(80):
        g = model(input_ids=next_tok, past_key_values=gen_cache, use_cache=True)
        gen_cache = g.past_key_values
        next_tok = g.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tid = next_tok[0, 0].item()
        generated.append(tid)
        if tid == tokenizer.eos_token_id:
            break
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if "####" in decoded:
            after = decoded.split("####")[-1]
            if re.search(r'\d[\d,]*\.?\d*\s*\n', after):
                break
            if re.search(r'\d[\d,]*\.?\d*\s+\S', after):
                break
        if re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?\s*-?[\d,]+', decoded):
            break
        if "\nQ:" in decoded:
            break
        del g

    answer_text = tokenizer.decode(generated, skip_special_tokens=True)
    raw_ans = extract_answer(answer_text) if answer_text else ""
    answer = normalize_answer(raw_ans) if raw_ans else ""
    correct = (answer == true_answer)

    del cache, gen_cache
    gc.collect(); torch.cuda.empty_cache()

    return {
        'correct': correct,
        'answer': answer,
        'text_accuracy': text_accuracy,
        'answer_text': answer_text[:200],
    }


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def wilson_ci(n_success, n_total, z=1.96):
    """Wilson score interval."""
    if n_total == 0:
        return 0.0, 0.0
    p = n_success / n_total
    denom = 1 + z*z/n_total
    center = p + z*z/(2*n_total)
    halfwidth = z * ((p*(1-p)/n_total + z*z/(4*n_total*n_total)) ** 0.5)
    lo = max(0, (center - halfwidth) / denom)
    hi = min(1, (center + halfwidth) / denom)
    return lo, hi


def generate_figures(agg, n_kv_heads, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heads = list(range(n_kv_heads))
    head_labels = [f"H{h}" for h in heads]

    # ── Figure 1: K-only per-head accuracy ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    k_accs = []
    k_acc_los = []
    k_acc_his = []
    k_texts = []
    v_accs = []
    v_acc_los = []
    v_acc_his = []
    v_texts = []

    for h in heads:
        k_key = f"head_k_{h}"
        v_key = f"head_v_{h}"
        if k_key in agg:
            k_accs.append(agg[k_key]['accuracy'] * 100)
            k_acc_los.append(agg[k_key]['wilson_ci_lo'] * 100)
            k_acc_his.append(agg[k_key]['wilson_ci_hi'] * 100)
            k_texts.append(agg[k_key]['text_accuracy'] * 100)
        else:
            k_accs.append(float('nan'))
            k_acc_los.append(float('nan'))
            k_acc_his.append(float('nan'))
            k_texts.append(float('nan'))

        if v_key in agg:
            v_accs.append(agg[v_key]['accuracy'] * 100)
            v_acc_los.append(agg[v_key]['wilson_ci_lo'] * 100)
            v_acc_his.append(agg[v_key]['wilson_ci_hi'] * 100)
            v_texts.append(agg[v_key]['text_accuracy'] * 100)
        else:
            v_accs.append(float('nan'))
            v_acc_los.append(float('nan'))
            v_acc_his.append(float('nan'))
            v_texts.append(float('nan'))

    x = np.arange(n_kv_heads)
    width = 0.35

    # Panel 1: Answer accuracy — K vs V per head
    ax1.bar(x - width/2, k_accs, width, color='#e74c3c', alpha=0.8,
            label='K-only', zorder=5)
    ax1.bar(x + width/2, v_accs, width, color='#3498db', alpha=0.8,
            label='V-only', zorder=5)

    # Error bars
    k_err_lo = [a - lo for a, lo in zip(k_accs, k_acc_los)]
    k_err_hi = [hi - a for a, hi in zip(k_accs, k_acc_his)]
    v_err_lo = [a - lo for a, lo in zip(v_accs, v_acc_los)]
    v_err_hi = [hi - a for a, hi in zip(v_accs, v_acc_his)]
    ax1.errorbar(x - width/2, k_accs, yerr=[k_err_lo, k_err_hi],
                 fmt='none', color='black', capsize=3, zorder=6)
    ax1.errorbar(x + width/2, v_accs, yerr=[v_err_lo, v_err_hi],
                 fmt='none', color='black', capsize=3, zorder=6)

    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax1.set_title('Exp 045: Per-Head K vs V Direction Perturbation\n'
                  'Qwen3-4B-Base — Each bar = 1 head perturbed across ALL layers & positions',
                  fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_ylim(-5, 115)

    # Panel 2: Text accuracy
    ax2.bar(x - width/2, k_texts, width, color='#e74c3c', alpha=0.5,
            label='K-only text', zorder=5)
    ax2.bar(x + width/2, v_texts, width, color='#3498db', alpha=0.5,
            label='V-only text', zorder=5)

    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel('KV Head Index', fontsize=12)
    ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(head_labels, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'per_head_kv_accuracy.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: K > V gap per head ──
    fig, ax = plt.subplots(figsize=(12, 7))

    gaps = []
    for h in heads:
        k_key = f"head_k_{h}"
        v_key = f"head_v_{h}"
        if k_key in agg and v_key in agg:
            gap = (agg[v_key]['accuracy'] - agg[k_key]['accuracy']) * 100
        else:
            gap = float('nan')
        gaps.append(gap)

    colors = ['#e74c3c' if g > 0 else '#3498db' for g in gaps]
    ax.bar(x, gaps, 0.6, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=1)

    for i, gap in enumerate(gaps):
        if not np.isnan(gap):
            ax.text(i, gap + (2 if gap > 0 else -4),
                    f'{gap:+.0f}pp', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('KV Head Index', fontsize=12)
    ax.set_ylabel('V-acc minus K-acc (pp)', fontsize=12)
    ax.set_title('K > V Gap Per Head (positive = V survives better than K)\n'
                 'Red bars: K > V confirmed at this head',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(head_labels, fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'kv_gap_per_head.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3: Head importance ranking ──
    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort heads by K-only accuracy (lowest = most important for answer)
    k_acc_pairs = [(h, k_accs[h]) for h in heads if not np.isnan(k_accs[h])]
    k_acc_sorted = sorted(k_acc_pairs, key=lambda x: x[1])

    sorted_heads = [p[0] for p in k_acc_sorted]
    sorted_k_accs = [p[1] for p in k_acc_sorted]
    sorted_v_accs = [v_accs[h] for h in sorted_heads]

    x_sorted = np.arange(len(sorted_heads))
    ax.bar(x_sorted - width/2, sorted_k_accs, width, color='#e74c3c', alpha=0.8,
           label='K-only acc')
    ax.bar(x_sorted + width/2, sorted_v_accs, width, color='#3498db', alpha=0.8,
           label='V-only acc')

    ax.set_xlabel('KV Head (sorted by K-importance, most important left)', fontsize=12)
    ax.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax.set_title('Head Importance Ranking\n'
                 'Sorted by K-only accuracy: lower = more critical for answer',
                 fontsize=13)
    ax.set_xticks(x_sorted)
    ax.set_xticklabels([f"H{h}" for h in sorted_heads], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(-5, 115)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'head_importance_ranking.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 4: Dissociation per head ──
    fig, ax = plt.subplots(figsize=(12, 7))

    k_dissocs = []
    v_dissocs = []
    for h in heads:
        k_key = f"head_k_{h}"
        v_key = f"head_v_{h}"
        k_dissocs.append(agg.get(k_key, {}).get('dissociation', float('nan')) * 100)
        v_dissocs.append(agg.get(v_key, {}).get('dissociation', float('nan')) * 100)

    ax.bar(x - width/2, k_dissocs, width, color='#e74c3c', alpha=0.8,
           label='K-only dissociation')
    ax.bar(x + width/2, v_dissocs, width, color='#3498db', alpha=0.8,
           label='V-only dissociation')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('KV Head Index', fontsize=12)
    ax.set_ylabel('Dissociation (AccDrop - TextDrop, %)', fontsize=12)
    ax.set_title('Text-Answer Dissociation Per Head\n'
                 'Higher = more selective answer disruption without text disruption',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(head_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dissociation_per_head.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 5: Signal norms per head ──
    fig, ax = plt.subplots(figsize=(12, 6))

    k_sigs = []
    v_sigs = []
    for h in heads:
        k_key = f"head_k_{h}"
        v_key = f"head_v_{h}"
        k_sigs.append(agg.get(k_key, {}).get('signal_rms', 0))
        v_sigs.append(agg.get(v_key, {}).get('signal_rms', 0))

    ax.bar(x - width/2, k_sigs, width, color='#e74c3c', alpha=0.7, label='K signal RMS')
    ax.bar(x + width/2, v_sigs, width, color='#3498db', alpha=0.7, label='V signal RMS')

    ax.set_xlabel('KV Head Index', fontsize=12)
    ax.set_ylabel('Signal RMS (norm of original vectors)', fontsize=12)
    ax.set_title('KV Cache Signal Norms Per Head\n'
                 '(Energy check: are norms uniform across heads?)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(head_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'signal_norms_per_head.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    model.eval()
    num_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, 'num_key_value_heads',
                         getattr(model.config, 'num_attention_heads', 8))
    n_attn_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // model.config.num_attention_heads)
    print(f"Model loaded: {num_layers} layers, {n_kv_heads} KV heads, "
          f"{n_attn_heads} attn heads, head_dim={head_dim}, t={time.time()-t0:.0f}s")

    # Verify cache access
    test_input = tokenizer("test", return_tensors="pt").to(model.device)
    test_out = model(**test_input, use_cache=True)
    test_cache = test_out.past_key_values
    assert hasattr(test_cache, 'layers'), "Cache must have layers attribute"
    k_shape = test_cache.layers[0].keys.shape
    print(f"Cache verified: layers[0].keys.shape = {k_shape}")
    del test_input, test_out, test_cache
    gc.collect(); torch.cuda.empty_cache()

    # Build conditions: K-only and V-only for each head
    conditions = []
    for h in range(n_kv_heads):
        conditions.append((f"head_k_{h}", h, "k"))
    for h in range(n_kv_heads):
        conditions.append((f"head_v_{h}", h, "v"))

    print(f"\nConditions: {len(conditions)} ({n_kv_heads} K-heads + {n_kv_heads} V-heads)")

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:NUM_PROBLEMS]

    results = []
    n_valid = 0
    n_skip_wrong = 0
    n_skip_trunc = 0

    for pi, ds_idx in enumerate(indices):
        elapsed = time.time() - t0
        if elapsed > 1500:  # Leave 5 min buffer for figures
            print(f"\n  Time limit at problem {pi}/{len(indices)}, elapsed={elapsed:.0f}s")
            break

        item = ds[ds_idx]
        true_answer = normalize_answer(extract_answer(item['answer']))
        prompt = build_prompt(item['question'])

        print(f"\nProblem {pi+1}/{len(indices)} (idx={ds_idx}), "
              f"true={true_answer}, t={elapsed:.0f}s")

        trace, prompt_ids, reasoning_ids = generate_trace(model, tokenizer, prompt)
        gen_answer = normalize_answer(extract_answer(trace)) if extract_answer(trace) else ""
        if gen_answer != true_answer:
            print(f"  WRONG: got '{gen_answer}', expected '{true_answer}'")
            n_skip_wrong += 1
            del prompt_ids, reasoning_ids
            gc.collect(); torch.cuda.empty_cache()
            continue

        trunc_pos = find_truncation_point(reasoning_ids, tokenizer)
        if trunc_pos is None or trunc_pos < 10:
            print(f"  Skip: no truncation point found")
            n_skip_trunc += 1
            del prompt_ids, reasoning_ids
            gc.collect(); torch.cuda.empty_cache()
            continue

        reasoning_ids_truncated = reasoning_ids[:, :trunc_pos]
        prompt_len = prompt_ids.shape[1]
        reasoning_len = reasoning_ids_truncated.shape[1]

        if reasoning_len < 20 or (prompt_len + reasoning_len) > MAX_SEQ_LEN:
            print(f"  Skip: R={reasoning_len}, total={prompt_len + reasoning_len}")
            n_skip_trunc += 1
            del prompt_ids, reasoning_ids
            gc.collect(); torch.cuda.empty_cache()
            continue

        # Clean evaluation
        prompt_cache = build_prompt_cache(model, prompt_ids)
        clean_eval = evaluate_clean(
            model, tokenizer, prompt_cache, reasoning_ids_truncated,
            prompt_len, num_layers, true_answer)
        clean_text_acc = clean_eval['text_accuracy']
        print(f"  Clean: acc={'Y' if clean_eval['correct'] else 'N'}, "
              f"text={clean_text_acc:.1%}, R={reasoning_len}, ans='{clean_eval['answer']}'")

        if not clean_eval['correct']:
            print(f"  Skip: clean evaluation wrong answer")
            n_skip_wrong += 1
            del prompt_ids, reasoning_ids, reasoning_ids_truncated, prompt_cache
            gc.collect(); torch.cuda.empty_cache()
            continue

        n_valid += 1
        problem_result = {
            'ds_idx': ds_idx,
            'true_answer': true_answer,
            'reasoning_len': reasoning_len,
            'clean_text_acc': clean_text_acc,
            'clean_correct': clean_eval['correct'],
            'evaluations': {},
        }

        # Run all conditions
        for cond_name, head_idx, component in conditions:
            pc = build_prompt_cache(model, prompt_ids)
            ev = evaluate_per_head(
                model, tokenizer, pc, reasoning_ids_truncated,
                prompt_len, num_layers, true_answer,
                target_head=head_idx, component=component)

            print(f"    {cond_name}: acc={'Y' if ev['correct'] else 'N'}, "
                  f"text={ev['text_accuracy']:.1%}")

            problem_result['evaluations'][cond_name] = {
                'correct': ev['correct'],
                'answer': ev['answer'],
                'text_accuracy': ev['text_accuracy'],
                'perturb_rms': ev['perturb_rms'],
                'signal_rms': ev['signal_rms'],
            }

        results.append(problem_result)
        del prompt_ids, reasoning_ids, reasoning_ids_truncated, prompt_cache
        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_valid} valid, {len(results)} processed, "
          f"{n_skip_wrong} wrong, {n_skip_trunc} truncation skips")

    summary = {
        'experiment': 'exp_045_per_head_kv_perturbation',
        'model': MODEL_NAME,
        'num_layers': num_layers,
        'n_kv_heads': n_kv_heads,
        'n_attn_heads': n_attn_heads,
        'head_dim': head_dim,
        'architecture': 'MHA' if n_kv_heads == n_attn_heads else 'GQA',
        'n_valid': n_valid,
        'n_processed': len(results),
        'n_skip_wrong': n_skip_wrong,
        'n_skip_trunc': n_skip_trunc,
        'conditions': [c[0] for c in conditions],
    }

    agg = {}
    for cond_tuple in conditions:
        cond_name = cond_tuple[0]
        accs = [1 if p['evaluations'][cond_name]['correct'] else 0
                for p in results if cond_name in p.get('evaluations', {})]
        txts = [p['evaluations'][cond_name]['text_accuracy']
                for p in results if cond_name in p.get('evaluations', {})]
        ctxts = [p['clean_text_acc'] for p in results
                 if cond_name in p.get('evaluations', {})]
        p_rms = [p['evaluations'][cond_name].get('perturb_rms', 0)
                 for p in results if cond_name in p.get('evaluations', {})]
        s_rms = [p['evaluations'][cond_name].get('signal_rms', 0)
                 for p in results if cond_name in p.get('evaluations', {})]

        if accs:
            n_count = len(accs)
            n_correct = sum(accs)
            a_mean = np.mean(accs)
            t_mean = np.mean(txts)
            ct_mean = np.mean(ctxts)
            a_drop = 1.0 - a_mean
            t_drop = ct_mean - t_mean
            dissoc = a_drop - t_drop
            w_lo, w_hi = wilson_ci(n_correct, n_count)

            agg[cond_name] = {
                'n': n_count,
                'n_correct': n_correct,
                'accuracy': float(a_mean),
                'text_accuracy': float(t_mean),
                'clean_text_accuracy': float(ct_mean),
                'accuracy_drop': float(a_drop),
                'text_drop': float(t_drop),
                'dissociation': float(dissoc),
                'wilson_ci_lo': float(w_lo),
                'wilson_ci_hi': float(w_hi),
                'perturb_rms': float(np.mean(p_rms)),
                'signal_rms': float(np.mean(s_rms)),
            }

    # Print summary table
    print(f"\n{'Condition':<12} {'Acc%':>6} {'Text%':>6} {'AccDrop':>8} {'TxtDrop':>8} "
          f"{'Dissoc':>8} {'n':>3} {'Wilson 95% CI':>18}")
    print("-" * 90)
    for cond_tuple in conditions:
        cond_name = cond_tuple[0]
        if cond_name in agg:
            d = agg[cond_name]
            print(f"  {cond_name:<10} {d['accuracy']*100:>5.1f}% {d['text_accuracy']*100:>5.1f}% "
                  f"{d['accuracy_drop']*100:>7.1f}% {d['text_drop']*100:>7.1f}% "
                  f"{d['dissociation']*100:>7.1f}% {d['n']:>3}"
                  f" [{d['wilson_ci_lo']*100:>5.1f}, {d['wilson_ci_hi']*100:>5.1f}]")

    # ── KEY ANALYSIS ──
    print(f"\n{'='*60}")
    print(f"KEY ANALYSIS: PER-HEAD K vs V DIRECTION PERTURBATION")
    print(f"{'='*60}")

    # K vs V comparison
    k_accs_all = []
    v_accs_all = []
    for h in range(n_kv_heads):
        k_key = f"head_k_{h}"
        v_key = f"head_v_{h}"
        if k_key in agg:
            k_accs_all.append(agg[k_key]['accuracy'] * 100)
        if v_key in agg:
            v_accs_all.append(agg[v_key]['accuracy'] * 100)

    if k_accs_all and v_accs_all:
        print(f"\n  K-only per-head: mean acc = {np.mean(k_accs_all):.1f}%, "
              f"range = [{min(k_accs_all):.1f}%, {max(k_accs_all):.1f}%]")
        print(f"  V-only per-head: mean acc = {np.mean(v_accs_all):.1f}%, "
              f"range = [{min(v_accs_all):.1f}%, {max(v_accs_all):.1f}%]")
        print(f"  Mean K > V gap: {np.mean(v_accs_all) - np.mean(k_accs_all):+.1f}pp")

        # Head heterogeneity
        k_std = np.std(k_accs_all)
        v_std = np.std(v_accs_all)
        k_range = max(k_accs_all) - min(k_accs_all)
        v_range = max(v_accs_all) - min(v_accs_all)
        print(f"\n  K-head heterogeneity: std={k_std:.1f}pp, range={k_range:.1f}pp")
        print(f"  V-head heterogeneity: std={v_std:.1f}pp, range={v_range:.1f}pp")

        if k_range > 20:
            print(f"  --> K-heads are HETEROGENEOUS: some heads matter much more than others")
        elif k_range > 10:
            print(f"  --> K-heads show MODERATE heterogeneity")
        else:
            print(f"  --> K-heads are HOMOGENEOUS: all heads roughly equally important")

        # Count how many heads show K > V
        n_k_gt_v = sum(1 for h in range(n_kv_heads)
                       if f"head_k_{h}" in agg and f"head_v_{h}" in agg
                       and agg[f"head_k_{h}"]['accuracy'] < agg[f"head_v_{h}"]['accuracy'])
        print(f"\n  K > V confirmed at {n_k_gt_v}/{n_kv_heads} heads")

        # Compare with positional perturbation
        print(f"\n  --- Comparison with Positional Perturbation ---")
        print(f"  Per-head K (12.5% of capacity): mean acc = {np.mean(k_accs_all):.1f}%")
        print(f"  Per-position K 5% (5% of capacity): mean acc = ~14% (Exp 044)")
        print(f"  Per-position K 10% (10% of capacity): mean acc = ~1.5% (Exp 042)")

    summary['aggregated'] = agg
    summary['elapsed_seconds'] = time.time() - t0

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=_convert)
    with open(os.path.join(RESULTS_DIR, 'per_problem.json'), 'w') as f:
        json.dump(results, f, indent=2, default=_convert)

    print("\nGenerating figures...")
    generate_figures(agg, n_kv_heads, RESULTS_DIR)
    print(f"\nDone. Total: {summary['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
