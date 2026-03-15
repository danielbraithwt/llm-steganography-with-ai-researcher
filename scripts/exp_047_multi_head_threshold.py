#!/usr/bin/env python3
"""
Experiment 047: Multi-Head K-Direction Perturbation Threshold on Qwen3-4B-Base

Tests whether "answer heads" (H5, H0) are genuinely specialized or whether
head-level redundancy collapses under multi-head perturbation.

DISCONFIRMATORY: If all 2-head combos at 25% capacity produce similar accuracy,
"answer heads" are just noise in single-head sensitivity.

CONDITIONS (all K-only direction perturbation at ALL positions, ALL layers):
  Dose-matched 25% (2 heads each):
    1. answer_h05:    H0 + H5  (answer heads)
    2. disp_h12:      H1 + H2  (dispensable)
    3. disp_h34:      H3 + H4  (dispensable)
    4. disp_h67:      H6 + H7  (dispensable + marginal)
  50% (4 heads each):
    5. disp4_h1234:   H1+H2+H3+H4 (4 dispensable)
    6. ans_disp_h0125: H0+H1+H2+H5 (answer + dispensable)
  75% (6 heads):
    7. leave_ans_only: H1+H2+H3+H4+H6+H7 (leave only answer heads H0+H5)
  100% (8 heads):
    8. all8:          all heads
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
                           "results", "exp_047")

# Multi-head conditions: (name, set_of_heads_to_perturb)
CONDITIONS = [
    # 25% capacity (2 heads)
    ("answer_h05",     [0, 5]),
    ("disp_h12",       [1, 2]),
    ("disp_h34",       [3, 4]),
    ("disp_h67",       [6, 7]),
    # 50% capacity (4 heads)
    ("disp4_h1234",    [1, 2, 3, 4]),
    ("ans_disp_h0125", [0, 1, 2, 5]),
    # 75% capacity (6 heads) — leave only answer heads
    ("leave_ans_only", [1, 2, 3, 4, 6, 7]),
    # 100% capacity
    ("all8",           [0, 1, 2, 3, 4, 5, 6, 7]),
]

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

    Args:
        target_heads: list of head indices to perturb simultaneously
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

    # ── Figure 1: Multi-head threshold curve ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Sort by capacity fraction for threshold plot
    cond_order = [c[0] for c in CONDITIONS]
    caps = {c[0]: len(c[1]) / 8 * 100 for c in CONDITIONS}

    # Panel 1: Answer accuracy by condition
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

    # Color by capacity fraction
    colors_25 = '#2ecc71'  # green for 25%
    colors_50 = '#f39c12'  # orange for 50%
    colors_75 = '#e74c3c'  # red for 75%
    colors_100 = '#8e44ad'  # purple for 100%

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

    # Error bars (clamp to non-negative for matplotlib)
    err_lo = [max(0, a - lo) for a, lo in zip(accs, acc_los)]
    err_hi = [max(0, hi - a) for a, hi in zip(accs, acc_his)]
    ax1.errorbar(x, accs, yerr=[err_lo, err_hi],
                 fmt='none', color='black', capsize=4, zorder=6)

    # Add percentage labels on bars
    for i, (acc, cap) in enumerate(zip(accs, cap_fracs)):
        ax1.text(i, acc + 3, f'{acc:.0f}%', ha='center', fontsize=9, fontweight='bold')
        ax1.text(i, -8, f'{cap:.0f}%', ha='center', fontsize=8, color='gray')

    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax1.set_title('Exp 047: Multi-Head K-Direction Perturbation Threshold\n'
                  'Qwen3-4B-Base — Dose-matched comparisons at 25%, 50%, 75%, 100%',
                  fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax1.set_ylim(-15, 115)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.text(0.02, 0.02, 'Bottom labels: % of K-routing capacity destroyed',
             transform=ax1.transAxes, fontsize=8, color='gray')

    # Panel 2: Text accuracy
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

    dose25_conds = ["answer_h05", "disp_h12", "disp_h34", "disp_h67"]
    dose25_labels = ["H0+H5\n(answer)", "H1+H2\n(disp)", "H3+H4\n(disp)", "H6+H7\n(disp+marg)"]
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

    # Error bars on accuracy (clamp to non-negative)
    err_lo = [max(0, a - lo) for a, lo in zip(dose25_accs, dose25_ci_lo)]
    err_hi = [max(0, hi - a) for a, hi in zip(dose25_accs, dose25_ci_hi)]
    ax.errorbar(x25 - width/2, dose25_accs, yerr=[err_lo, err_hi],
                fmt='none', color='black', capsize=4, zorder=6)

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Dose-Matched Comparison at 25% Capacity\n'
                 'Answer Heads (H0+H5) vs Dispensable Head Pairs',
                 fontsize=13)
    ax.set_xticks(x25)
    ax.set_xticklabels(dose25_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(-5, 115)

    # Add significance annotation if answer heads significantly differ
    if dose25_accs and len(dose25_accs) >= 2:
        ans_acc = dose25_accs[0]
        disp_max = max(dose25_accs[1:])
        gap = disp_max - ans_acc
        if gap > 0:
            ax.annotate(f'Gap: {gap:.0f}pp', xy=(0.5, 0.95),
                       xycoords='axes fraction', fontsize=11,
                       ha='center', fontweight='bold',
                       color='red' if gap > 20 else 'orange')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dose_matched_25pct.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3: Dose-matched at 50% ──
    fig, ax = plt.subplots(figsize=(8, 6))

    dose50_conds = ["disp4_h1234", "ans_disp_h0125"]
    dose50_labels = ["H1+H2+H3+H4\n(4 dispensable)", "H0+H1+H2+H5\n(answer + disp)"]
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
    ax.set_title('Dose-Matched Comparison at 50% Capacity\n'
                 '4 Dispensable vs Answer+Dispensable',
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

    # ── Figure 4: Redundancy curve (accuracy vs heads destroyed) ──
    fig, ax = plt.subplots(figsize=(10, 7))

    # Get single-head reference from Exp 045
    single_head_mean = 89.1  # from Exp 045

    # Build curve: 0, 1 (from 045), 2, 4, 6, 8 heads
    n_heads_list = [0, 1]
    acc_list = [100.0, single_head_mean]

    # For 2 heads: average of all 25% conditions
    two_head_accs = [agg[cn]['accuracy'] * 100 for cn in ["answer_h05", "disp_h12", "disp_h34", "disp_h67"] if cn in agg]
    if two_head_accs:
        n_heads_list.append(2)
        acc_list.append(np.mean(two_head_accs))

    # 4 heads
    four_head_accs = [agg[cn]['accuracy'] * 100 for cn in ["disp4_h1234", "ans_disp_h0125"] if cn in agg]
    if four_head_accs:
        n_heads_list.append(4)
        acc_list.append(np.mean(four_head_accs))

    # 6 heads
    if "leave_ans_only" in agg:
        n_heads_list.append(6)
        acc_list.append(agg["leave_ans_only"]['accuracy'] * 100)

    # 8 heads
    if "all8" in agg:
        n_heads_list.append(8)
        acc_list.append(agg["all8"]['accuracy'] * 100)

    ax.plot(n_heads_list, acc_list, 'ko-', linewidth=2, markersize=8, zorder=5)

    # Add individual points for dose-matched conditions
    for cn in ["answer_h05", "disp_h12", "disp_h34", "disp_h67"]:
        if cn in agg:
            color = '#e74c3c' if cn == "answer_h05" else '#3498db'
            marker = 's' if cn == "answer_h05" else 'o'
            ax.scatter(2, agg[cn]['accuracy'] * 100, c=color, s=100, zorder=6,
                      marker=marker, edgecolors='black', linewidths=1,
                      label=cn if cn in ["answer_h05", "disp_h12"] else "")

    for cn in ["disp4_h1234", "ans_disp_h0125"]:
        if cn in agg:
            color = '#e74c3c' if 'ans' in cn else '#3498db'
            marker = 's' if 'ans' in cn else 'o'
            ax.scatter(4, agg[cn]['accuracy'] * 100, c=color, s=100, zorder=6,
                      marker=marker, edgecolors='black', linewidths=1)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Number of K-Heads Destroyed', fontsize=12)
    ax.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax.set_title('K-Routing Redundancy Curve\n'
                 'Red squares = answer heads included, Blue circles = dispensable only',
                 fontsize=13)
    ax.set_xticks([0, 1, 2, 4, 6, 8])
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'redundancy_curve.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 5: Dissociation by condition ──
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

    # Add dissociation values
    for i, dis in enumerate(dissocs):
        y_pos = max(acc_drops[i], txt_drops[i]) + 3
        ax.text(i, y_pos, f'Δ={dis:+.0f}pp', ha='center', fontsize=8,
                fontweight='bold', color='green' if dis > 10 else 'gray')

    ax.set_ylabel('Drop from Clean (%)', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_title('Text-Answer Dissociation by Condition\n'
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
        print(f"  {cname}: heads={heads} ({len(heads)}/8 = {len(heads)/8*100:.0f}%)")

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
        'experiment': 'exp_047_multi_head_threshold',
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

            agg[cond_name] = {
                'n': n_count,
                'n_correct': n_correct,
                'n_heads': len(head_list),
                'capacity_pct': len(head_list) / 8 * 100,
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
    print(f"\n{'Condition':<18} {'Heads':>5} {'Cap%':>5} {'Acc%':>6} {'Text%':>6} "
          f"{'AccDrop':>8} {'TxtDrop':>8} {'Dissoc':>8} {'n':>3} {'Wilson 95% CI':>18}")
    print("-" * 110)
    for cond_name, head_list in CONDITIONS:
        if cond_name in agg:
            d = agg[cond_name]
            print(f"  {cond_name:<16} {len(head_list):>5} {d['capacity_pct']:>4.0f}% "
                  f"{d['accuracy']*100:>5.1f}% {d['text_accuracy']*100:>5.1f}% "
                  f"{d['accuracy_drop']*100:>7.1f}% {d['text_drop']*100:>7.1f}% "
                  f"{d['dissociation']*100:>7.1f}% {d['n']:>3}"
                  f" [{d['wilson_ci_lo']*100:>5.1f}, {d['wilson_ci_hi']*100:>5.1f}]")

    # ── KEY ANALYSIS ──
    print(f"\n{'='*60}")
    print(f"KEY ANALYSIS: MULTI-HEAD K-DIRECTION PERTURBATION THRESHOLD")
    print(f"{'='*60}")

    # Dose-matched comparison at 25%
    print(f"\n  --- Dose-matched at 25% (2 heads) ---")
    dose25 = {}
    for cn in ["answer_h05", "disp_h12", "disp_h34", "disp_h67"]:
        if cn in agg:
            dose25[cn] = agg[cn]['accuracy'] * 100
            print(f"  {cn}: {dose25[cn]:.1f}% [{agg[cn]['wilson_ci_lo']*100:.1f}, {agg[cn]['wilson_ci_hi']*100:.1f}]")

    if "answer_h05" in dose25:
        ans_acc = dose25["answer_h05"]
        disp_accs = [dose25[cn] for cn in ["disp_h12", "disp_h34", "disp_h67"] if cn in dose25]
        if disp_accs:
            disp_mean = np.mean(disp_accs)
            disp_max = max(disp_accs)
            gap = disp_mean - ans_acc
            print(f"\n  Answer heads (H0+H5): {ans_acc:.1f}%")
            print(f"  Dispensable mean: {disp_mean:.1f}%")
            print(f"  Gap (disp_mean - answer): {gap:+.1f}pp")
            if gap > 25:
                print(f"  --> CONFIRMS answer-head specialization (gap > 25pp)")
            elif gap > 10:
                print(f"  --> MODEST specialization (10 < gap < 25pp)")
            elif gap > 0:
                print(f"  --> WEAK/NO specialization (gap < 10pp)")
            else:
                print(f"  --> Answer heads NOT more destructive! DISCONFIRMS specialization")

    # Dose-matched at 50%
    print(f"\n  --- Dose-matched at 50% (4 heads) ---")
    for cn in ["disp4_h1234", "ans_disp_h0125"]:
        if cn in agg:
            print(f"  {cn}: {agg[cn]['accuracy']*100:.1f}% "
                  f"[{agg[cn]['wilson_ci_lo']*100:.1f}, {agg[cn]['wilson_ci_hi']*100:.1f}]")

    if "disp4_h1234" in agg and "ans_disp_h0125" in agg:
        d4_acc = agg["disp4_h1234"]['accuracy'] * 100
        a4_acc = agg["ans_disp_h0125"]['accuracy'] * 100
        gap50 = d4_acc - a4_acc
        print(f"  Gap (disp4 - ans+disp): {gap50:+.1f}pp")

    # Leave-only and total
    print(f"\n  --- Threshold extremes ---")
    if "leave_ans_only" in agg:
        print(f"  Leave only H0+H5 (75% destroyed): {agg['leave_ans_only']['accuracy']*100:.1f}% "
              f"[{agg['leave_ans_only']['wilson_ci_lo']*100:.1f}, {agg['leave_ans_only']['wilson_ci_hi']*100:.1f}]")
    if "all8" in agg:
        print(f"  All 8 heads (100% destroyed): {agg['all8']['accuracy']*100:.1f}% "
              f"[{agg['all8']['wilson_ci_lo']*100:.1f}, {agg['all8']['wilson_ci_hi']*100:.1f}]")

    # Redundancy curve
    print(f"\n  --- Redundancy Curve ---")
    print(f"  0 heads: 100% (clean)")
    print(f"  1 head (Exp 045): 89.1% mean")
    if dose25:
        print(f"  2 heads: {np.mean(list(dose25.values())):.1f}% (all 2-head combos avg)")
    if "disp4_h1234" in agg and "ans_disp_h0125" in agg:
        avg_4 = (agg["disp4_h1234"]['accuracy'] + agg["ans_disp_h0125"]['accuracy']) / 2 * 100
        print(f"  4 heads: {avg_4:.1f}% (both 4-head combos avg)")
    if "leave_ans_only" in agg:
        print(f"  6 heads: {agg['leave_ans_only']['accuracy']*100:.1f}%")
    if "all8" in agg:
        print(f"  8 heads: {agg['all8']['accuracy']*100:.1f}%")

    # Per-problem overlap analysis for dose-matched conditions
    if n_valid >= 5:
        print(f"\n  --- Per-Problem Analysis ---")
        # Which problems fail on answer heads but survive on dispensable?
        ans_fail_disp_survive = 0
        disp_fail_ans_survive = 0
        both_fail = 0
        both_survive = 0

        for p in results:
            if "answer_h05" in p['evaluations'] and "disp_h12" in p['evaluations']:
                ans_ok = p['evaluations']['answer_h05']['correct']
                disp_ok = p['evaluations']['disp_h12']['correct']
                if not ans_ok and disp_ok:
                    ans_fail_disp_survive += 1
                elif ans_ok and not disp_ok:
                    disp_fail_ans_survive += 1
                elif not ans_ok and not disp_ok:
                    both_fail += 1
                else:
                    both_survive += 1

        total_paired = ans_fail_disp_survive + disp_fail_ans_survive + both_fail + both_survive
        if total_paired > 0:
            print(f"  H0+H5 fails, H1+H2 survives: {ans_fail_disp_survive}/{total_paired}")
            print(f"  H1+H2 fails, H0+H5 survives: {disp_fail_ans_survive}/{total_paired}")
            print(f"  Both fail: {both_fail}/{total_paired}")
            print(f"  Both survive: {both_survive}/{total_paired}")

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
