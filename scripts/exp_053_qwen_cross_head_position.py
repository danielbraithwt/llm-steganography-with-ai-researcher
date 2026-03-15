#!/usr/bin/env python3
"""
Experiment 053: Cross-Head Position-Independence on Qwen3-4B-Base

Tests whether position-independence extends beyond H5/H7 (Exp 049) to
H0 (second answer head, 66.7% acc) and H3 (moderately dispensable, 87.5% acc).

Combined with Exp 049 data (H5=50%, H7=95.8%), gives a 4-point
criticality spectrum for position-dependence on Qwen — directly
comparable to Llama's 4-head spectrum (Exp 051+052).

DESIGN: 2x3 factorial + 2 all-position references
  - Factor 1: Head (H0 = second answer head, H3 = moderately dispensable)
  - Factor 2: Position band (early 33%, mid 33%, late 33%)
  - References: H0-all, H3-all

8 conditions per problem:
  h0_early, h0_mid, h0_late, h0_all,
  h3_early, h3_mid, h3_late, h3_all
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
                           "results", "exp_053")

# From Exp 045 single-head results on Qwen:
# H5=50.0%, H0=66.7%, H3=87.5%, H7=95.8%
# H5 and H7 already tested in Exp 049 — now test H0 and H3
CRITICAL_HEAD = 0   # Second answer head (66.7% single-head acc)
DISP_HEAD = 3       # Moderately dispensable (87.5% single-head acc)

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
    """Replace direction of a single head at last position, preserve norm.
    tensor shape: [1, n_heads, seq_len, head_dim]
    Returns (perturbation_l2_squared, signal_l2_squared).
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
def evaluate_head_position(model, tokenizer, prompt_cache, reasoning_tokens,
                           prompt_len, num_layers, true_answer,
                           target_head, position_band, reasoning_len):
    """
    Step through reasoning tokens, perturb K-direction at ONE head
    at positions within a specific band. Measure answer accuracy and
    text prediction accuracy.

    position_band: 'early', 'mid', 'late', or 'all'
    """
    r_len = reasoning_tokens.shape[1]

    # Define position bounds for each band
    third = r_len / 3.0
    if position_band == 'early':
        pos_start, pos_end = 0, int(round(third))
    elif position_band == 'mid':
        pos_start, pos_end = int(round(third)), int(round(2 * third))
    elif position_band == 'late':
        pos_start, pos_end = int(round(2 * third)), r_len
    else:  # 'all'
        pos_start, pos_end = 0, r_len

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
    positions_perturbed = 0

    for i in range(r_len):
        token = reasoning_tokens[:, i:i+1]
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

        # Perturb only if this position is within the target band
        if pos_start <= i < pos_end:
            for l in range(num_layers):
                p_l2, s_l2 = perturb_direction_single_head(
                    cache.layers[l].keys, target_head)
                perturb_l2_total += p_l2
                signal_l2_total += s_l2
                perturb_count += 1
            positions_perturbed += 1

        # Check prediction for next token
        if i < r_len - 1:
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
        'positions_perturbed': positions_perturbed,
        'total_positions': r_len,
        'band_start': pos_start,
        'band_end': pos_end,
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


def generate_figures(agg, n_valid, results_dir, exp_049_data=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions_order = [
        'h0_early', 'h0_mid', 'h0_late', 'h0_all',
        'h3_early', 'h3_mid', 'h3_late', 'h3_all',
    ]
    labels = [
        'H0\nearly', 'H0\nmid', 'H0\nlate', 'H0\nall',
        'H3\nearly', 'H3\nmid', 'H3\nlate', 'H3\nall',
    ]

    # ── Figure 1: Accuracy by condition (KEY FIGURE) ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    accs = [agg.get(c, {}).get('accuracy', float('nan')) * 100 for c in conditions_order]
    texts = [agg.get(c, {}).get('text_accuracy', float('nan')) * 100 for c in conditions_order]
    ci_los = [agg.get(c, {}).get('wilson_ci_lo', float('nan')) * 100 for c in conditions_order]
    ci_his = [agg.get(c, {}).get('wilson_ci_hi', float('nan')) * 100 for c in conditions_order]

    x = np.arange(len(conditions_order))
    colors = ['#e74c3c'] * 4 + ['#3498db'] * 4  # Red=H0, Blue=H3

    # Panel 1: Answer accuracy
    bars = ax1.bar(x, accs, 0.6, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    err_lo = [a - lo for a, lo in zip(accs, ci_los)]
    err_hi = [hi - a for a, hi in zip(accs, ci_his)]
    ax1.errorbar(x, accs, yerr=[err_lo, err_hi], fmt='none', color='black', capsize=4)

    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=3.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax1.set_title(f'Exp 053: Cross-Head Position-Independence — Qwen3-4B-Base (n={n_valid})\n'
                  'K-direction perturbation at specific head × position band',
                  fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_ylim(-5, 115)

    for i, acc in enumerate(accs):
        if not np.isnan(acc):
            ax1.text(i, acc + 3, f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Panel 2: Text accuracy
    ax2.bar(x, texts, 0.6, color=colors, alpha=0.5, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(x=3.5, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Condition (Head × Position Band)', fontsize=12)
    ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.grid(True, alpha=0.2, axis='y')

    for i, txt in enumerate(texts):
        if not np.isnan(txt):
            ax2.text(i, txt + 1.5, f'{txt:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'head_position_accuracy.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 1: head_position_accuracy.png")

    # ── Figure 2: Interaction plot (line plot) ──
    fig, ax = plt.subplots(figsize=(10, 7))

    positions = ['early', 'mid', 'late']
    h0_accs = [agg.get(f'h0_{p}', {}).get('accuracy', float('nan')) * 100 for p in positions]
    h3_accs = [agg.get(f'h3_{p}', {}).get('accuracy', float('nan')) * 100 for p in positions]

    px = [0, 1, 2]
    ax.plot(px, h0_accs, 'ro-', markersize=10, linewidth=2.5, label='H0 (2nd answer head)')
    ax.plot(px, h3_accs, 'bs-', markersize=10, linewidth=2.5, label='H3 (moderately dispensable)')

    # Add CI shading
    for head_name, head_accs, color in [('h0', h0_accs, '#e74c3c'), ('h3', h3_accs, '#3498db')]:
        ci_lo = [agg.get(f'{head_name}_{p}', {}).get('wilson_ci_lo', float('nan')) * 100
                 for p in positions]
        ci_hi = [agg.get(f'{head_name}_{p}', {}).get('wilson_ci_hi', float('nan')) * 100
                 for p in positions]
        ax.fill_between(px, ci_lo, ci_hi, alpha=0.15, color=color)

    # Reference lines for H0-all and H3-all
    h0_all = agg.get('h0_all', {}).get('accuracy', float('nan')) * 100
    h3_all = agg.get('h3_all', {}).get('accuracy', float('nan')) * 100
    ax.axhline(y=h0_all, color='#e74c3c', linestyle='--', alpha=0.5,
               label=f'H0-all ({h0_all:.1f}%)')
    ax.axhline(y=h3_all, color='#3498db', linestyle='--', alpha=0.5,
               label=f'H3-all ({h3_all:.1f}%)')

    ax.set_xlabel('Position Band (thirds of reasoning chain)', fontsize=12)
    ax.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax.set_title('H0 & H3 Position Interaction — Qwen3-4B-Base\n'
                 'Does digital encoding provide position-independent routing for ALL heads?',
                 fontsize=13)
    ax.set_xticks(px)
    ax.set_xticklabels(['Early (0-33%)', 'Mid (33-66%)', 'Late (66-100%)'], fontsize=11)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-5, 115)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'interaction_plot.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 2: interaction_plot.png")

    # ── Figure 3: Dissociation by condition ──
    fig, ax = plt.subplots(figsize=(14, 7))

    dissocs = [agg.get(c, {}).get('dissociation', float('nan')) * 100 for c in conditions_order]
    acc_drops = [agg.get(c, {}).get('accuracy_drop', float('nan')) * 100 for c in conditions_order]
    txt_drops = [agg.get(c, {}).get('text_drop', float('nan')) * 100 for c in conditions_order]

    width = 0.3
    ax.bar(x - width, acc_drops, width, color='#e74c3c', alpha=0.7, label='AccDrop')
    ax.bar(x, txt_drops, width, color='#3498db', alpha=0.7, label='TextDrop')
    ax.bar(x + width, dissocs, width, color='#2ecc71', alpha=0.7, label='Dissociation')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=3.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Percentage Points', fontsize=12)
    ax.set_title('Text-Answer Dissociation by Head × Position\n'
                 'Positive dissociation = answer disrupted more than text', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dissociation_by_condition.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 3: dissociation_by_condition.png")

    # ── Figure 4: Position band dose check ──
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in conditions_order:
        if cond in agg:
            pos_perturbed = agg[cond].get('mean_positions_perturbed', 0)
            total_pos = agg[cond].get('mean_total_positions', 1)
            frac = pos_perturbed / total_pos if total_pos > 0 else 0
            ax.bar(cond, frac * 100, color='#95a5a6', alpha=0.7, edgecolor='black')

    ax.set_ylabel('Fraction of Positions Perturbed (%)', fontsize=12)
    ax.set_title('Position Band Dose Check\n'
                 'Each band = ~33% of positions, all = 100%', fontsize=12)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions_order],
                       fontsize=9, rotation=0)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dose_check.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 4: dose_check.png")

    # ── Figure 5: 4-head criticality spectrum (combine with Exp 049) ──
    # Reference data from Exp 049: H5 and H7
    exp049_heads = {
        'H5': {'sh_acc': 50.0, 'early': 55.8, 'mid': 65.1, 'late': 65.1, 'all': 60.5},
        'H7': {'sh_acc': 95.8, 'early': 100.0, 'mid': 100.0, 'late': 100.0, 'all': 100.0},
    }

    # Current data
    this_heads = {}
    h0_vals = {}
    h3_vals = {}
    for band in ['early', 'mid', 'late']:
        h0_key = f'h0_{band}'
        h3_key = f'h3_{band}'
        if h0_key in agg:
            h0_vals[band] = agg[h0_key]['accuracy'] * 100
        if h3_key in agg:
            h3_vals[band] = agg[h3_key]['accuracy'] * 100

    if h0_vals and h3_vals:
        this_heads['H0'] = {'sh_acc': 66.7, **h0_vals}
        this_heads['H3'] = {'sh_acc': 87.5, **h3_vals}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Panel 1: 4-head interaction plot
        all_heads = {**exp049_heads, **this_heads}
        head_order = ['H5', 'H0', 'H3', 'H7']
        colors_heads = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']

        for hi, head_name in enumerate(head_order):
            if head_name in all_heads:
                h = all_heads[head_name]
                accs_h = [h.get('early', np.nan), h.get('mid', np.nan), h.get('late', np.nan)]
                marker = 'o' if head_name in this_heads else 's'
                ls = '-' if head_name in this_heads else '--'
                ax1.plot(px, accs_h, f'{marker}{ls}', markersize=10, linewidth=2.5,
                         color=colors_heads[hi],
                         label=f'{head_name} (SH={h["sh_acc"]:.0f}%)')

        ax1.set_xlabel('Position Band', fontsize=12)
        ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax1.set_title('Qwen 4-Head Criticality Spectrum\n'
                      '(H5/H7 from Exp 049, H0/H3 from Exp 053)', fontsize=12)
        ax1.set_xticks(px)
        ax1.set_xticklabels(['Early', 'Mid', 'Late'], fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.2)
        ax1.set_ylim(-5, 115)

        # Panel 2: Criticality vs range scatter
        crit_data = []
        for head_name in head_order:
            if head_name in all_heads:
                h = all_heads[head_name]
                bands_vals = [h.get('early', np.nan), h.get('mid', np.nan), h.get('late', np.nan)]
                valid_vals = [v for v in bands_vals if not np.isnan(v)]
                if valid_vals:
                    rng = max(valid_vals) - min(valid_vals)
                    crit_data.append((head_name, h['sh_acc'], rng))

        if crit_data:
            sh_accs_scatter = [d[1] for d in crit_data]
            ranges_scatter = [d[2] for d in crit_data]
            names_scatter = [d[0] for d in crit_data]

            for i, (name, sh, rng) in enumerate(crit_data):
                color_idx = head_order.index(name) if name in head_order else 0
                ax2.scatter(sh, rng, s=150, color=colors_heads[color_idx],
                           edgecolors='black', linewidth=1, zorder=5)
                ax2.annotate(name, (sh, rng), textcoords="offset points",
                            xytext=(10, 5), fontsize=12, fontweight='bold')

            # Fit line and compute correlation
            if len(sh_accs_scatter) >= 3:
                from scipy import stats
                slope, intercept, r_val, p_val, std_err = stats.linregress(
                    sh_accs_scatter, ranges_scatter)
                x_fit = np.linspace(min(sh_accs_scatter) - 5, max(sh_accs_scatter) + 5, 50)
                y_fit = slope * x_fit + intercept
                ax2.plot(x_fit, y_fit, 'k--', alpha=0.5, linewidth=1.5,
                        label=f'r={r_val:.3f}, p={p_val:.3f}')
                ax2.legend(fontsize=11)

            ax2.set_xlabel('Single-Head Accuracy (%)', fontsize=12)
            ax2.set_ylabel('Position Range (pp)', fontsize=12)
            ax2.set_title('Criticality vs Position-Dependence\n'
                          'Qwen: All heads flat (position-independent)?', fontsize=12)
            ax2.grid(True, alpha=0.2)
            ax2.axhline(y=15, color='red', linestyle=':', alpha=0.5,
                       label='15pp threshold')

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'criticality_spectrum.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure 5: criticality_spectrum.png")


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
    assert k_shape[1] >= max(CRITICAL_HEAD, DISP_HEAD) + 1, \
        f"Not enough KV heads: {k_shape[1]} < {max(CRITICAL_HEAD, DISP_HEAD) + 1}"
    del test_input, test_out, test_cache
    gc.collect(); torch.cuda.empty_cache()

    # Build conditions: head × position band
    conditions = []
    for head_idx, head_name in [(CRITICAL_HEAD, 'h0'), (DISP_HEAD, 'h3')]:
        for band in ['early', 'mid', 'late', 'all']:
            cond_name = f"{head_name}_{band}"
            conditions.append((cond_name, head_idx, band))

    print(f"\nConditions: {len(conditions)}")
    for c in conditions:
        print(f"  {c[0]}: head={c[1]}, band={c[2]}")

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
        for cond_name, head_idx, band in conditions:
            pc = build_prompt_cache(model, prompt_ids)
            ev = evaluate_head_position(
                model, tokenizer, pc, reasoning_ids_truncated,
                prompt_len, num_layers, true_answer,
                target_head=head_idx, position_band=band,
                reasoning_len=reasoning_len)

            print(f"    {cond_name}: acc={'Y' if ev['correct'] else 'N'}, "
                  f"text={ev['text_accuracy']:.1%}, "
                  f"pos={ev['positions_perturbed']}/{ev['total_positions']}")

            problem_result['evaluations'][cond_name] = {
                'correct': ev['correct'],
                'answer': ev['answer'],
                'text_accuracy': ev['text_accuracy'],
                'perturb_rms': ev['perturb_rms'],
                'signal_rms': ev['signal_rms'],
                'positions_perturbed': ev['positions_perturbed'],
                'total_positions': ev['total_positions'],
                'band_start': ev['band_start'],
                'band_end': ev['band_end'],
            }

        results.append(problem_result)
        del prompt_ids, reasoning_ids, reasoning_ids_truncated, prompt_cache
        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_valid} valid, {len(results)} processed, "
          f"{n_skip_wrong} wrong, {n_skip_trunc} truncation skips")

    summary = {
        'experiment': 'exp_053_qwen_cross_head_position',
        'model': MODEL_NAME,
        'num_layers': num_layers,
        'n_kv_heads': n_kv_heads,
        'n_attn_heads': n_attn_heads,
        'head_dim': head_dim,
        'architecture': 'MHA' if n_kv_heads == n_attn_heads else 'GQA',
        'critical_head': CRITICAL_HEAD,
        'disp_head': DISP_HEAD,
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
        pos_p = [p['evaluations'][cond_name].get('positions_perturbed', 0)
                 for p in results if cond_name in p.get('evaluations', {})]
        tot_p = [p['evaluations'][cond_name].get('total_positions', 0)
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
                'mean_positions_perturbed': float(np.mean(pos_p)),
                'mean_total_positions': float(np.mean(tot_p)),
            }

    # Print summary table
    print(f"\n{'Condition':<12} {'Acc%':>6} {'Text%':>6} {'AccDrop':>8} {'TxtDrop':>8} "
          f"{'Dissoc':>8} {'n':>3} {'Wilson 95% CI':>18} {'Pos%':>6}")
    print("-" * 100)
    for cond_tuple in conditions:
        cond_name = cond_tuple[0]
        if cond_name in agg:
            d = agg[cond_name]
            pos_frac = d['mean_positions_perturbed'] / d['mean_total_positions'] * 100 \
                if d['mean_total_positions'] > 0 else 0
            print(f"  {cond_name:<10} {d['accuracy']*100:>5.1f}% {d['text_accuracy']*100:>5.1f}% "
                  f"{d['accuracy_drop']*100:>7.1f}% {d['text_drop']*100:>7.1f}% "
                  f"{d['dissociation']*100:>7.1f}% {d['n']:>3}"
                  f" [{d['wilson_ci_lo']*100:>5.1f}, {d['wilson_ci_hi']*100:>5.1f}]"
                  f" {pos_frac:>5.1f}%")

    # ── KEY ANALYSIS ──
    print(f"\n{'='*60}")
    print(f"KEY ANALYSIS: CROSS-HEAD POSITION-INDEPENDENCE")
    print(f"{'='*60}")

    h0_accs = {}
    h3_accs = {}
    for band in ['early', 'mid', 'late', 'all']:
        h0_key = f'h0_{band}'
        h3_key = f'h3_{band}'
        if h0_key in agg:
            h0_accs[band] = agg[h0_key]['accuracy'] * 100
        if h3_key in agg:
            h3_accs[band] = agg[h3_key]['accuracy'] * 100

    print(f"\n  H0 (2nd answer head, SH=66.7%) accuracy by position band:")
    for b in ['early', 'mid', 'late', 'all']:
        if b in h0_accs:
            print(f"    {b:>6}: {h0_accs[b]:.1f}%")

    print(f"\n  H3 (moderately dispensable, SH=87.5%) accuracy by position band:")
    for b in ['early', 'mid', 'late', 'all']:
        if b in h3_accs:
            print(f"    {b:>6}: {h3_accs[b]:.1f}%")

    # Position range analysis
    for head_name, head_accs, sh_acc in [('H0', h0_accs, 66.7), ('H3', h3_accs, 87.5)]:
        if all(b in head_accs for b in ['early', 'mid', 'late']):
            rng = max(head_accs[b] for b in ['early', 'mid', 'late']) - \
                  min(head_accs[b] for b in ['early', 'mid', 'late'])
            print(f"\n  {head_name} position range: {rng:.1f}pp (SH={sh_acc}%)")
            if rng > 25:
                print(f"  --> {head_name} is POSITION-DEPENDENT (large range)")
            elif rng > 15:
                print(f"  --> {head_name} shows MODERATE position dependence")
            else:
                print(f"  --> {head_name} is POSITION-INDEPENDENT (range < 15pp)")

    # Interaction analysis
    if all(b in h0_accs for b in ['early', 'late']) and \
       all(b in h3_accs for b in ['early', 'late']):
        h0_gradient = h0_accs['late'] - h0_accs['early']
        h3_gradient = h3_accs['late'] - h3_accs['early']
        interaction = h0_gradient - h3_gradient
        print(f"\n  H0 late-vs-early gradient: {h0_gradient:+.1f}pp")
        print(f"  H3 late-vs-early gradient: {h3_gradient:+.1f}pp")
        print(f"  Interaction (H0_gradient - H3_gradient): {interaction:+.1f}pp")

    # 4-head spectrum comparison (with Exp 049 data)
    print(f"\n  {'='*50}")
    print(f"  4-HEAD QWEN SPECTRUM (combined with Exp 049)")
    print(f"  {'='*50}")

    spectrum = []
    # Exp 049 data
    spectrum.append(('H5', 50.0, 9.3, 55.8, 65.1, 65.1, 60.5, 'Exp 049'))
    spectrum.append(('H7', 95.8, 0.0, 100.0, 100.0, 100.0, 100.0, 'Exp 049'))

    if all(b in h0_accs for b in ['early', 'mid', 'late', 'all']):
        h0_range = max(h0_accs[b] for b in ['early', 'mid', 'late']) - \
                   min(h0_accs[b] for b in ['early', 'mid', 'late'])
        spectrum.append(('H0', 66.7, h0_range, h0_accs['early'], h0_accs['mid'],
                        h0_accs['late'], h0_accs['all'], 'Exp 053'))
    if all(b in h3_accs for b in ['early', 'mid', 'late', 'all']):
        h3_range = max(h3_accs[b] for b in ['early', 'mid', 'late']) - \
                   min(h3_accs[b] for b in ['early', 'mid', 'late'])
        spectrum.append(('H3', 87.5, h3_range, h3_accs['early'], h3_accs['mid'],
                        h3_accs['late'], h3_accs['all'], 'Exp 053'))

    spectrum.sort(key=lambda x: x[1])  # Sort by SH acc

    print(f"\n  {'Head':<5} {'SH_Acc%':>8} {'Range':>7} {'Early':>7} {'Mid':>7} "
          f"{'Late':>7} {'All':>7} {'Source':>8}")
    print(f"  {'-'*65}")
    for s in spectrum:
        print(f"  {s[0]:<5} {s[1]:>7.1f}% {s[2]:>6.1f}pp {s[3]:>6.1f}% "
              f"{s[4]:>6.1f}% {s[5]:>6.1f}% {s[6]:>6.1f}% {s[7]:>8}")

    # Correlation analysis
    if len(spectrum) >= 3:
        from scipy import stats
        sh_accs_sp = [s[1] for s in spectrum]
        ranges_sp = [s[2] for s in spectrum]
        if len(set(ranges_sp)) > 1:  # Need variance to compute correlation
            r_pearson, p_pearson = stats.pearsonr(sh_accs_sp, ranges_sp)
            r_spearman, p_spearman = stats.spearmanr(sh_accs_sp, ranges_sp)
            print(f"\n  Pearson r = {r_pearson:.3f} (p = {p_pearson:.3f})")
            print(f"  Spearman rho = {r_spearman:.3f} (p = {p_spearman:.3f})")
        else:
            print(f"\n  All ranges near zero — no correlation to compute (all position-independent)")

    # Per-problem concordance
    if results:
        for head_name, head_key in [('H0', 'h0'), ('H3', 'h3')]:
            late_fails_early_survives = 0
            early_fails_late_survives = 0
            both_fail = 0
            both_survive = 0
            for p in results:
                ev = p['evaluations']
                early_key = f'{head_key}_early'
                late_key = f'{head_key}_late'
                if early_key in ev and late_key in ev:
                    early_ok = ev[early_key]['correct']
                    late_ok = ev[late_key]['correct']
                    if not late_ok and early_ok:
                        late_fails_early_survives += 1
                    elif late_ok and not early_ok:
                        early_fails_late_survives += 1
                    elif not late_ok and not early_ok:
                        both_fail += 1
                    else:
                        both_survive += 1

            total_conc = late_fails_early_survives + early_fails_late_survives + \
                         both_fail + both_survive
            if total_conc > 0:
                print(f"\n  {head_name} per-problem concordance (late vs early), n={total_conc}:")
                print(f"    Late-fail/Early-ok: {late_fails_early_survives} "
                      f"({late_fails_early_survives/total_conc*100:.1f}%)")
                print(f"    Early-fail/Late-ok: {early_fails_late_survives} "
                      f"({early_fails_late_survives/total_conc*100:.1f}%)")
                print(f"    Both-fail: {both_fail} ({both_fail/total_conc*100:.1f}%)")
                print(f"    Both-survive: {both_survive} ({both_survive/total_conc*100:.1f}%)")

    summary['aggregated'] = agg
    summary['elapsed_seconds'] = time.time() - t0

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=_convert)
    with open(os.path.join(RESULTS_DIR, 'per_problem.json'), 'w') as f:
        json.dump(results, f, indent=2, default=_convert)

    print("\nGenerating figures...")
    generate_figures(agg, n_valid, RESULTS_DIR)
    print(f"\nDone. Total: {summary['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
