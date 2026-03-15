#!/usr/bin/env python3
"""
Experiment 048: Multi-Head K-Direction Perturbation Threshold on Llama-3.1-8B-Instruct

DISCONFIRMATORY: Tests whether Llama shows the same two-regime redundancy curve
as Qwen (Exp 047), or a GRADUAL capacity-dependent curve predicted by analog encoding.

From Exp 046 single-head K-accuracy on Llama:
  H5: 18.2% (most critical)
  H3: 27.3%
  H2: 36.4%
  H4: 51.5%
  H6: 51.5%
  H1: 60.6%
  H0: 63.6%
  H7: 90.9% (most dispensable)

CONDITIONS (all K-only direction perturbation at ALL positions, ALL layers):
  Dose-matched 25% (2 heads each):
    1. critical_h35:  H3 + H5  (most critical pair, mean single-head=22.7%)
    2. mid_h46:       H4 + H6  (intermediate pair, mean=51.5%)
    3. mid_h01:       H0 + H1  (upper-intermediate, mean=62.1%)
    4. disp_h07:      H0 + H7  (most dispensable pair, mean=77.3%)
  50% (4 heads each):
    5. crit4_h2345:   H2+H3+H4+H5 (4 most critical, mean=33.4%)
    6. disp4_h0167:   H0+H1+H6+H7 (4 least critical, mean=66.7%)
  75% (6 heads):
    7. leave_disp:    leave only H0+H7 (most dispensable pair)
  100% (8 heads):
    8. all8:          all heads

KEY TEST: If accuracy tracks average single-head sensitivity smoothly (gradual),
Llama has no two-regime pattern. If accuracy shows sharp discontinuities
(critical pair near 0%, dispensable pair near 80%+), two-regime is universal.
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
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NUM_PROBLEMS = 200  # attempt many, time-limited
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 2048
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_048")

# Multi-head conditions adapted for Llama's sensitivity profile
# From Exp 046: H5=18.2%, H3=27.3%, H2=36.4%, H4=51.5%, H6=51.5%, H1=60.6%, H0=63.6%, H7=90.9%
CONDITIONS = [
    # 25% capacity (2 heads) — sorted by expected destructiveness
    ("critical_h35",  [3, 5]),    # Most critical pair (mean single-head: 22.7%)
    ("mid_h46",       [4, 6]),    # Intermediate pair (mean: 51.5%)
    ("mid_h01",       [0, 1]),    # Upper-intermediate pair (mean: 62.1%)
    ("disp_h07",      [0, 7]),    # Most dispensable pair (mean: 77.3%)
    # 50% capacity (4 heads)
    ("crit4_h2345",   [2, 3, 4, 5]),  # 4 most critical heads
    ("disp4_h0167",   [0, 1, 6, 7]),  # 4 least critical heads
    # 75% capacity (6 heads) — leave only most dispensable pair
    ("leave_disp",    [1, 2, 3, 4, 5, 6]),  # leave only H0+H7
    # 100% capacity
    ("all8",          [0, 1, 2, 3, 4, 5, 6, 7]),
]

# Reference single-head accuracies from Exp 046 for analysis
SINGLE_HEAD_ACC = {
    0: 63.6, 1: 60.6, 2: 36.4, 3: 27.3, 4: 51.5, 5: 18.2, 6: 51.5, 7: 90.9
}

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
    Returns (perturbation_l2_sq, signal_l2_sq).
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
def evaluate_multi_head(model, tokenizer, prompt_cache, reasoning_tokens,
                        prompt_len, num_layers, true_answer,
                        target_heads):
    """
    Step through reasoning tokens, perturb K-direction at MULTIPLE heads
    across all layers and all positions. Measure answer accuracy and text
    prediction accuracy.
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

        # Perturb ALL target heads at this position across all layers
        for l in range(num_layers):
            for h in target_heads:
                p_l2, s_l2 = perturb_direction_single_head(
                    cache.layers[l].keys, h)
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
        'n_heads_perturbed': len(target_heads),
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


def generate_figures(agg, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond_order = [c[0] for c in CONDITIONS]
    caps = {c[0]: len(c[1]) / 8 * 100 for c in CONDITIONS}

    # Compute mean single-head accuracy for each condition
    mean_single = {}
    for cname, heads in CONDITIONS:
        mean_single[cname] = np.mean([SINGLE_HEAD_ACC[h] for h in heads])

    # ── Figure 1: Multi-head threshold curve ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    names = []
    accs = []
    acc_los = []
    acc_his = []
    texts = []
    cap_fracs = []

    for cond_name in cond_order:
        if cond_name in agg:
            d = agg[cond_name]
            names.append(cond_name)
            accs.append(d['accuracy'] * 100)
            acc_los.append(d['wilson_ci_lo'] * 100)
            acc_his.append(d['wilson_ci_hi'] * 100)
            texts.append(d['text_accuracy'] * 100)
            cap_fracs.append(caps[cond_name])

    x = np.arange(len(names))

    colors_25 = '#2ecc71'
    colors_50 = '#f39c12'
    colors_75 = '#e74c3c'
    colors_100 = '#8e44ad'

    bar_colors = []
    for c in cap_fracs:
        if c <= 25:
            bar_colors.append(colors_25)
        elif c <= 50:
            bar_colors.append(colors_50)
        elif c <= 75:
            bar_colors.append(colors_75)
        else:
            bar_colors.append(colors_100)

    bars = ax1.bar(x, accs, 0.6, color=bar_colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5, zorder=5)

    err_lo = [max(0, a - lo) for a, lo in zip(accs, acc_los)]
    err_hi = [max(0, hi - a) for a, hi in zip(accs, acc_his)]
    ax1.errorbar(x, accs, yerr=[err_lo, err_hi],
                 fmt='none', color='black', capsize=4, zorder=6)

    for i, (acc, cap) in enumerate(zip(accs, cap_fracs)):
        ax1.text(i, acc + 3, f'{acc:.0f}%', ha='center', fontsize=9, fontweight='bold')
        ax1.text(i, -8, f'{cap:.0f}%', ha='center', fontsize=8, color='gray')

    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax1.set_title('Exp 048: Multi-Head K-Direction Perturbation Threshold\n'
                  'Llama-3.1-8B-Instruct — Dose-matched comparisons at 25%, 50%, 75%, 100%',
                  fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax1.set_ylim(-15, 115)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.text(0.02, 0.02, 'Bottom labels: % of K-routing capacity destroyed',
             transform=ax1.transAxes, fontsize=8, color='gray')

    ax2.bar(x, texts, 0.6, color=bar_colors, alpha=0.5,
            edgecolor='black', linewidth=0.5, zorder=5)
    for i, txt in enumerate(texts):
        ax2.text(i, txt + 2, f'{txt:.0f}%', ha='center', fontsize=9)

    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
    ax2.set_xlabel('Condition', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'multi_head_threshold.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: Dose-matched comparison at 25% ──
    fig, ax = plt.subplots(figsize=(10, 7))

    dose25_conds = ["critical_h35", "mid_h46", "mid_h01", "disp_h07"]
    dose25_labels = ["H3+H5\n(critical)", "H4+H6\n(mid)", "H0+H1\n(upper-mid)", "H0+H7\n(dispensable)"]
    dose25_accs = []
    dose25_txts = []
    dose25_ci_lo = []
    dose25_ci_hi = []

    for cn in dose25_conds:
        if cn in agg:
            dose25_accs.append(agg[cn]['accuracy'] * 100)
            dose25_txts.append(agg[cn]['text_accuracy'] * 100)
            dose25_ci_lo.append(agg[cn]['wilson_ci_lo'] * 100)
            dose25_ci_hi.append(agg[cn]['wilson_ci_hi'] * 100)
        else:
            dose25_accs.append(0)
            dose25_txts.append(0)
            dose25_ci_lo.append(0)
            dose25_ci_hi.append(0)

    x25 = np.arange(len(dose25_conds))
    width = 0.35

    bars1 = ax.bar(x25 - width/2, dose25_accs, width, color='#e74c3c', alpha=0.8,
                   label='Answer Accuracy', zorder=5)
    bars2 = ax.bar(x25 + width/2, dose25_txts, width, color='#3498db', alpha=0.8,
                   label='Text Accuracy', zorder=5)

    err_lo = [max(0, a - lo) for a, lo in zip(dose25_accs, dose25_ci_lo)]
    err_hi = [max(0, hi - a) for a, hi in zip(dose25_accs, dose25_ci_hi)]
    ax.errorbar(x25 - width/2, dose25_accs, yerr=[err_lo, err_hi],
                fmt='none', color='black', capsize=4, zorder=6)

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Dose-Matched Comparison at 25% Capacity (Llama)\n'
                 'Critical pair (H3+H5) vs Intermediate vs Dispensable pair',
                 fontsize=13)
    ax.set_xticks(x25)
    ax.set_xticklabels(dose25_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(-5, 115)

    if dose25_accs and len(dose25_accs) >= 2:
        gap = max(dose25_accs) - min(dose25_accs)
        ax.annotate(f'Range: {gap:.0f}pp', xy=(0.5, 0.95),
                   xycoords='axes fraction', fontsize=11,
                   ha='center', fontweight='bold',
                   color='red' if gap > 30 else 'orange' if gap > 15 else 'gray')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dose_matched_25pct.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3: Dose-matched at 50% ──
    fig, ax = plt.subplots(figsize=(8, 6))

    dose50_conds = ["crit4_h2345", "disp4_h0167"]
    dose50_labels = ["H2+H3+H4+H5\n(4 critical)", "H0+H1+H6+H7\n(4 dispensable)"]
    dose50_accs = []
    dose50_txts = []

    for cn in dose50_conds:
        if cn in agg:
            dose50_accs.append(agg[cn]['accuracy'] * 100)
            dose50_txts.append(agg[cn]['text_accuracy'] * 100)
        else:
            dose50_accs.append(0)
            dose50_txts.append(0)

    x50 = np.arange(len(dose50_conds))
    ax.bar(x50 - width/2, dose50_accs, width, color='#e74c3c', alpha=0.8,
           label='Answer Accuracy')
    ax.bar(x50 + width/2, dose50_txts, width, color='#3498db', alpha=0.8,
           label='Text Accuracy')

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Dose-Matched Comparison at 50% Capacity (Llama)\n'
                 '4 Critical vs 4 Dispensable Heads',
                 fontsize=13)
    ax.set_xticks(x50)
    ax.set_xticklabels(dose50_labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(-5, 115)

    for i, (a, t) in enumerate(zip(dose50_accs, dose50_txts)):
        ax.text(i - width/2, a + 2, f'{a:.0f}%', ha='center', fontsize=10, fontweight='bold')
        ax.text(i + width/2, t + 2, f'{t:.0f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dose_matched_50pct.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 4: Accuracy vs Mean Single-Head Sensitivity (KEY TEST) ──
    fig, ax = plt.subplots(figsize=(10, 7))

    # For each 2-head condition, plot multi-head accuracy vs mean single-head accuracy
    for cn in dose25_conds:
        if cn in agg:
            ms = mean_single[cn]
            ma = agg[cn]['accuracy'] * 100
            color = '#e74c3c' if 'critical' in cn else '#f39c12' if 'mid' in cn else '#2ecc71'
            ax.scatter(ms, ma, s=200, c=color, edgecolors='black', linewidths=1.5,
                      zorder=6, label=cn)
            ax.annotate(cn, (ms, ma), textcoords="offset points",
                       xytext=(10, 5), fontsize=9)

    # Add 50% conditions
    for cn in dose50_conds:
        if cn in agg:
            ms = mean_single[cn]
            ma = agg[cn]['accuracy'] * 100
            color = '#e74c3c' if 'crit' in cn else '#2ecc71'
            ax.scatter(ms, ma, s=200, c=color, edgecolors='black', linewidths=1.5,
                      marker='s', zorder=6, label=cn)
            ax.annotate(cn, (ms, ma), textcoords="offset points",
                       xytext=(10, 5), fontsize=9)

    # Plot identity line (multi-head acc = mean single-head acc)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='y = x (perfect tracking)')

    # Add Qwen reference: answer_h05 = 3.7%, mean single ≈ 58.3%
    ax.scatter(58.35, 3.7, s=200, c='gray', edgecolors='black', linewidths=1.5,
              marker='D', zorder=5, alpha=0.5, label='Qwen H0+H5 (Exp 047)')

    ax.set_xlabel('Mean Single-Head K-Accuracy (%)\n(from Exp 046)', fontsize=12)
    ax.set_ylabel('Multi-Head K-Accuracy (%)\n(this experiment)', fontsize=12)
    ax.set_title('Does Multi-Head Accuracy Track Mean Single-Head Sensitivity?\n'
                 'Linear = GRADUAL (analog); Discontinuity = TWO-REGIME (digital)',
                 fontsize=13)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    # Compute correlation
    x_vals = [mean_single[cn] for cn in dose25_conds + dose50_conds if cn in agg]
    y_vals = [agg[cn]['accuracy'] * 100 for cn in dose25_conds + dose50_conds if cn in agg]
    if len(x_vals) >= 3:
        from scipy.stats import pearsonr, spearmanr
        r_pearson, _ = pearsonr(x_vals, y_vals)
        r_spearman, _ = spearmanr(x_vals, y_vals)
        ax.text(0.02, 0.98, f'Pearson r = {r_pearson:.3f}\nSpearman ρ = {r_spearman:.3f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_vs_sensitivity.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 5: Cross-model comparison (Qwen vs Llama redundancy curves) ──
    fig, ax = plt.subplots(figsize=(10, 7))

    # Qwen data from Exp 047
    qwen_n = [0, 1, 2, 2, 4, 4, 6, 8]
    qwen_acc = [100, 89.1, 3.7, 98.8, 22.2, 0.0, 0.0, 7.4]
    qwen_labels = ['clean', '1h (mean)', 'H0+H5', 'disp pair (mean)', 'disp4', 'ans+disp', 'leave_ans', 'all8']

    # Llama data from this experiment
    llama_n = [0, 1]
    llama_acc = [100, 50.0]  # Exp 046 mean K-acc

    for cn, heads in CONDITIONS:
        if cn in agg:
            llama_n.append(len(heads))
            llama_acc.append(agg[cn]['accuracy'] * 100)

    # Plot both curves
    # Qwen: answer-head points in red, dispensable in blue
    ax.scatter([2], [3.7], c='#e74c3c', s=120, marker='s', zorder=6,
              edgecolors='black', linewidths=1)
    ax.scatter([2], [98.8], c='#3498db', s=120, marker='s', zorder=6,
              edgecolors='black', linewidths=1)
    ax.scatter([4], [0.0], c='#e74c3c', s=120, marker='s', zorder=6,
              edgecolors='black', linewidths=1, label='Qwen critical heads')
    ax.scatter([4], [22.2], c='#3498db', s=120, marker='s', zorder=6,
              edgecolors='black', linewidths=1, label='Qwen dispensable heads')
    ax.scatter([0, 1, 6, 8], [100, 89.1, 0.0, 7.4], c='gray', s=80, marker='s',
              zorder=5, edgecolors='black', linewidths=0.5, alpha=0.5,
              label='Qwen other')

    # Llama: critical in red, dispensable in green
    for cn, heads in CONDITIONS:
        if cn in agg:
            nh = len(heads)
            ma = agg[cn]['accuracy'] * 100
            if 'critical' in cn or 'crit' in cn:
                color = '#e74c3c'
            elif 'disp' in cn:
                color = '#2ecc71'
            else:
                color = '#f39c12'
            ax.scatter(nh, ma, c=color, s=120, marker='o', zorder=6,
                      edgecolors='black', linewidths=1)
    ax.scatter([], [], c='#e74c3c', s=120, marker='o', edgecolors='black',
              linewidths=1, label='Llama critical heads')
    ax.scatter([], [], c='#2ecc71', s=120, marker='o', edgecolors='black',
              linewidths=1, label='Llama dispensable heads')
    ax.scatter([0, 1], [100, 50.0], c='gray', s=80, marker='o',
              zorder=5, edgecolors='black', linewidths=0.5, alpha=0.5,
              label='Llama other')

    ax.set_xlabel('Number of K-Heads Destroyed', fontsize=12)
    ax.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax.set_title('Cross-Model Redundancy Curves\n'
                 'Squares = Qwen (Exp 047), Circles = Llama (Exp 048)',
                 fontsize=13)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xticks([0, 1, 2, 4, 6, 8])
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cross_model_redundancy.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 6: Dissociation by condition ──
    fig, ax = plt.subplots(figsize=(14, 7))

    dissocs = []
    acc_drops = []
    txt_drops = []
    cond_names_plot = []

    for cond_name in cond_order:
        if cond_name in agg:
            d = agg[cond_name]
            dissocs.append(d['dissociation'] * 100)
            acc_drops.append(d['accuracy_drop'] * 100)
            txt_drops.append(d['text_drop'] * 100)
            cond_names_plot.append(cond_name)

    x_d = np.arange(len(cond_names_plot))
    ax.bar(x_d - width/2, acc_drops, width, color='#e74c3c', alpha=0.8,
           label='Accuracy Drop')
    ax.bar(x_d + width/2, txt_drops, width, color='#3498db', alpha=0.8,
           label='Text Drop')

    for i, dis in enumerate(dissocs):
        y_pos = max(acc_drops[i], txt_drops[i]) + 3
        ax.text(i, y_pos, f'Δ={dis:+.0f}pp', ha='center', fontsize=8,
                fontweight='bold', color='green' if dis > 10 else 'gray')

    ax.set_ylabel('Drop from Clean (%)', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_title('Text-Answer Dissociation by Condition (Llama)\n'
                 'Green Δ = AccDrop - TextDrop (positive = answer-selective disruption)',
                 fontsize=13)
    ax.set_xticks(x_d)
    ax.set_xticklabels(cond_names_plot, rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dissociation_by_condition.png'),
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

    print(f"\nConditions ({len(CONDITIONS)}):")
    for cname, heads in CONDITIONS:
        ms = np.mean([SINGLE_HEAD_ACC[h] for h in heads])
        print(f"  {cname}: heads={heads} ({len(heads)}/8 = {len(heads)/8*100:.0f}%) "
              f"mean_single_head_acc={ms:.1f}%")

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
        for cond_name, head_list in CONDITIONS:
            pc = build_prompt_cache(model, prompt_ids)
            ev = evaluate_multi_head(
                model, tokenizer, pc, reasoning_ids_truncated,
                prompt_len, num_layers, true_answer,
                target_heads=head_list)

            print(f"    {cond_name} ({len(head_list)}h): "
                  f"acc={'Y' if ev['correct'] else 'N'}, "
                  f"text={ev['text_accuracy']:.1%}")

            problem_result['evaluations'][cond_name] = {
                'correct': ev['correct'],
                'answer': ev['answer'],
                'text_accuracy': ev['text_accuracy'],
                'perturb_rms': ev['perturb_rms'],
                'signal_rms': ev['signal_rms'],
                'n_heads': len(head_list),
            }

        results.append(problem_result)
        del prompt_ids, reasoning_ids, reasoning_ids_truncated, prompt_cache
        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_valid} valid, {len(results)} processed, "
          f"{n_skip_wrong} wrong, {n_skip_trunc} truncation skips")

    summary = {
        'experiment': 'exp_048_llama_multi_head_threshold',
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
        'conditions': {c[0]: c[1] for c in CONDITIONS},
        'single_head_reference': SINGLE_HEAD_ACC,
    }

    agg = {}
    for cond_name, head_list in CONDITIONS:
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

            ms_acc = np.mean([SINGLE_HEAD_ACC[h] for h in head_list])

            agg[cond_name] = {
                'n': n_count,
                'n_correct': n_correct,
                'n_heads': len(head_list),
                'capacity_pct': len(head_list) / 8 * 100,
                'mean_single_head_acc': float(ms_acc),
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
    print(f"\n{'Condition':<18} {'Heads':>5} {'Cap%':>5} {'MeanSH':>7} {'Acc%':>6} {'Text%':>6} "
          f"{'AccDrop':>8} {'TxtDrop':>8} {'Dissoc':>8} {'n':>3} {'Wilson 95% CI':>18}")
    print("-" * 120)
    for cond_name, head_list in CONDITIONS:
        if cond_name in agg:
            d = agg[cond_name]
            print(f"  {cond_name:<16} {len(head_list):>5} {d['capacity_pct']:>4.0f}% "
                  f"{d['mean_single_head_acc']:>5.1f}% "
                  f"{d['accuracy']*100:>5.1f}% {d['text_accuracy']*100:>5.1f}% "
                  f"{d['accuracy_drop']*100:>7.1f}% {d['text_drop']*100:>7.1f}% "
                  f"{d['dissociation']*100:>7.1f}% {d['n']:>3}"
                  f" [{d['wilson_ci_lo']*100:>5.1f}, {d['wilson_ci_hi']*100:>5.1f}]")

    # ── KEY ANALYSIS ──
    print(f"\n{'='*60}")
    print(f"KEY ANALYSIS: LLAMA MULTI-HEAD K-DIRECTION PERTURBATION THRESHOLD")
    print(f"{'='*60}")

    # Dose-matched comparison at 25%
    print(f"\n  --- Dose-matched at 25% (2 heads) ---")
    dose25 = {}
    for cn in ["critical_h35", "mid_h46", "mid_h01", "disp_h07"]:
        if cn in agg:
            dose25[cn] = agg[cn]['accuracy'] * 100
            ms = agg[cn]['mean_single_head_acc']
            print(f"  {cn} (mean_SH={ms:.1f}%): {dose25[cn]:.1f}% "
                  f"[{agg[cn]['wilson_ci_lo']*100:.1f}, {agg[cn]['wilson_ci_hi']*100:.1f}]")

    if len(dose25) >= 2:
        crit_acc = dose25.get("critical_h35", None)
        disp_acc = dose25.get("disp_h07", None)
        if crit_acc is not None and disp_acc is not None:
            gap = disp_acc - crit_acc
            print(f"\n  Critical pair (H3+H5): {crit_acc:.1f}%")
            print(f"  Dispensable pair (H0+H7): {disp_acc:.1f}%")
            print(f"  Gap (disp - critical): {gap:+.1f}pp")

            # Compare to Qwen gap (95.1pp)
            print(f"  Qwen gap (Exp 047): +95.1pp")
            if gap > 60:
                print(f"  --> LARGE gap: Two-regime pattern CONFIRMED on Llama")
            elif gap > 30:
                print(f"  --> MODERATE gap: Partial specialization (weaker than Qwen)")
            elif gap > 15:
                print(f"  --> MODEST gap: Continuous gradient dominates (GRADUAL)")
            else:
                print(f"  --> SMALL/NO gap: DISCONFIRMS two-regime pattern on Llama")

    # Test linearity: does accuracy track mean single-head sensitivity?
    print(f"\n  --- Linearity Test ---")
    x_sh = []
    y_mh = []
    labels = []
    for cn in ["critical_h35", "mid_h46", "mid_h01", "disp_h07",
                "crit4_h2345", "disp4_h0167"]:
        if cn in agg:
            x_sh.append(agg[cn]['mean_single_head_acc'])
            y_mh.append(agg[cn]['accuracy'] * 100)
            labels.append(cn)
            print(f"  {cn}: mean_SH={agg[cn]['mean_single_head_acc']:.1f}%, "
                  f"multi_acc={agg[cn]['accuracy']*100:.1f}%")

    if len(x_sh) >= 3:
        from scipy.stats import pearsonr, spearmanr
        r_p, p_p = pearsonr(x_sh, y_mh)
        r_s, p_s = spearmanr(x_sh, y_mh)
        print(f"\n  Pearson r = {r_p:.3f} (p = {p_p:.4f})")
        print(f"  Spearman ρ = {r_s:.3f} (p = {p_s:.4f})")
        if r_p > 0.9:
            print(f"  --> STRONG linearity: GRADUAL pattern confirmed (analog)")
        elif r_p > 0.7:
            print(f"  --> MODERATE linearity: mixed gradual/regime pattern")
        else:
            print(f"  --> WEAK linearity: two-regime or nonlinear pattern")

    # Dose-matched at 50%
    print(f"\n  --- Dose-matched at 50% (4 heads) ---")
    for cn in ["crit4_h2345", "disp4_h0167"]:
        if cn in agg:
            print(f"  {cn}: {agg[cn]['accuracy']*100:.1f}% "
                  f"[{agg[cn]['wilson_ci_lo']*100:.1f}, {agg[cn]['wilson_ci_hi']*100:.1f}]")

    if "crit4_h2345" in agg and "disp4_h0167" in agg:
        c4_acc = agg["crit4_h2345"]['accuracy'] * 100
        d4_acc = agg["disp4_h0167"]['accuracy'] * 100
        gap50 = d4_acc - c4_acc
        print(f"  Gap (disp4 - crit4): {gap50:+.1f}pp")
        print(f"  Qwen gap at 50% (disp4 - ans+disp): +22.2pp")

    # Leave-only and total
    print(f"\n  --- Threshold extremes ---")
    if "leave_disp" in agg:
        print(f"  Leave only H0+H7 (75% destroyed): {agg['leave_disp']['accuracy']*100:.1f}% "
              f"[{agg['leave_disp']['wilson_ci_lo']*100:.1f}, {agg['leave_disp']['wilson_ci_hi']*100:.1f}]")
    if "all8" in agg:
        print(f"  All 8 heads (100% destroyed): {agg['all8']['accuracy']*100:.1f}% "
              f"[{agg['all8']['wilson_ci_lo']*100:.1f}, {agg['all8']['wilson_ci_hi']*100:.1f}]")

    # Redundancy curve comparison
    print(f"\n  --- Redundancy Curve (Llama vs Qwen) ---")
    print(f"  {'Heads':>6} {'Llama (this)':>14} {'Qwen (047)':>14}")
    print(f"  {'0':>6} {'100.0%':>14} {'100.0%':>14}")
    print(f"  {'1 (mean)':>6} {'50.0%':>14} {'89.1%':>14}")
    for cn in ["critical_h35", "disp_h07"]:
        if cn in agg:
            print(f"  {'2 '+cn:>6} {agg[cn]['accuracy']*100:>13.1f}%", end="")
            if cn == "critical_h35":
                print(f" {'3.7% (ans)':>14}")
            else:
                print(f" {'98.8% (disp)':>14}")
    for cn in ["crit4_h2345", "disp4_h0167"]:
        if cn in agg:
            print(f"  {'4 '+cn:>6} {agg[cn]['accuracy']*100:>13.1f}%", end="")
            if 'crit' in cn:
                print(f" {'0.0% (ans+d)':>14}")
            else:
                print(f" {'22.2% (disp4)':>14}")
    if "leave_disp" in agg:
        print(f"  {'6':>6} {agg['leave_disp']['accuracy']*100:>13.1f}% {'0.0%':>14}")
    if "all8" in agg:
        print(f"  {'8':>6} {agg['all8']['accuracy']*100:>13.1f}% {'7.4%':>14}")

    # Per-problem overlap analysis
    if n_valid >= 5:
        print(f"\n  --- Per-Problem Analysis (critical_h35 vs disp_h07) ---")
        crit_fail_disp_survive = 0
        disp_fail_crit_survive = 0
        both_fail = 0
        both_survive = 0

        for p in results:
            if "critical_h35" in p['evaluations'] and "disp_h07" in p['evaluations']:
                crit_ok = p['evaluations']['critical_h35']['correct']
                disp_ok = p['evaluations']['disp_h07']['correct']
                if not crit_ok and disp_ok:
                    crit_fail_disp_survive += 1
                elif crit_ok and not disp_ok:
                    disp_fail_crit_survive += 1
                elif not crit_ok and not disp_ok:
                    both_fail += 1
                else:
                    both_survive += 1

        total_paired = crit_fail_disp_survive + disp_fail_crit_survive + both_fail + both_survive
        if total_paired > 0:
            print(f"  H3+H5 fails, H0+H7 survives: {crit_fail_disp_survive}/{total_paired}")
            print(f"  H0+H7 fails, H3+H5 survives: {disp_fail_crit_survive}/{total_paired}")
            print(f"  Both fail: {both_fail}/{total_paired}")
            print(f"  Both survive: {both_survive}/{total_paired}")
            concordance = crit_fail_disp_survive / total_paired * 100
            print(f"  Concordance (critical fails, disp survives): {concordance:.1f}%")
            print(f"  Qwen concordance (Exp 047): 92.6%")

    summary['aggregated'] = agg
    summary['elapsed_seconds'] = time.time() - t0

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=_convert)
    with open(os.path.join(RESULTS_DIR, 'per_problem.json'), 'w') as f:
        json.dump(results, f, indent=2, default=_convert)

    print("\nGenerating figures...")
    generate_figures(agg, RESULTS_DIR)
    print(f"\nDone. Total: {summary['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
