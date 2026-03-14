#!/usr/bin/env python3
"""
Experiment 022: Geometric Double Dissociation — Direction vs Magnitude KV Perturbation

Tests whether functional separability in the KV cache operates through geometric
properties (direction vs magnitude) of KV vectors.

Key insight: exp_019/021's norm-matched Gaussian replacement IS direction-only
perturbation. This experiment adds the missing magnitude-only condition.

Conditions at late/mid/early positions:
  - Direction perturbation: random direction, preserved per-head norm
  - Magnitude perturbation: scaled norm, preserved direction
  - K-only / V-only decomposition
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
NUM_PROBLEMS = 200
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 1536
NOISE_FRAC = 0.05  # 5% of positions
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_022")

EXEMPLARS = [
    {"q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "a": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = <<21-15=6>>6 trees planted.\n#### 6"},
    {"q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "a": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = <<3+2=5>>5 cars are now in the parking lot.\n#### 5"},
    {"q": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "a": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = <<32+42=74>>74. After eating 35, they had 74 - 35 = <<74-35=39>>39 pieces left.\n#### 39"},
    {"q": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?", "a": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = <<20-12=8>>8 lollipops.\n#### 8"},
    {"q": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "a": "Shawn started with 5 toys. He got 2 toys each from mom and dad. So he got 2 + 2 = <<2+2=4>>4 more toys. Now he has 5 + 4 = <<5+4=9>>9 toys.\n#### 9"},
    {"q": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?", "a": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 4 * 5 = <<4*5=20>>20 computers were added. 9 + 20 = <<9+20=29>>29 computers now.\n#### 29"},
    {"q": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?", "a": "Michael started with 58 golf balls. After losing 23, he had 58 - 23 = <<58-23=35>>35. After losing 2 more, he had 35 - 2 = <<35-2=33>>33 golf balls.\n#### 33"},
    {"q": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "a": "Olivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = <<5*3=15>>15 dollars. She has 23 - 15 = <<23-15=8>>8 dollars left.\n#### 8"},
]

# ── Condition definitions ───────────────────────────────────────────────
# Each condition: (name, position_type, perturb_component, perturb_type, sigma)
# position_type: 'late', 'mid', 'early'
# perturb_component: 'kv', 'k', 'v'
# perturb_type: 'direction', 'magnitude'
# sigma: only used for magnitude (ignored for direction)
CONDITIONS = [
    # Late 5%
    ("dir_kv_late",       "late",  "kv", "direction", 0.0),
    ("mag_kv_05_late",    "late",  "kv", "magnitude", 0.5),
    ("mag_kv_10_late",    "late",  "kv", "magnitude", 1.0),
    ("mag_kv_14_late",    "late",  "kv", "magnitude", 1.414),
    ("dir_k_late",        "late",  "k",  "direction", 0.0),
    ("dir_v_late",        "late",  "v",  "direction", 0.0),
    ("mag_k_10_late",     "late",  "k",  "magnitude", 1.0),
    ("mag_v_10_late",     "late",  "v",  "magnitude", 1.0),
    # Mid 5% (positions 47.5%-52.5%)
    ("dir_kv_mid",        "mid",   "kv", "direction", 0.0),
    ("mag_kv_10_mid",     "mid",   "kv", "magnitude", 1.0),
    # Early 5%
    ("dir_kv_early",      "early", "kv", "direction", 0.0),
    ("mag_kv_10_early",   "early", "kv", "magnitude", 1.0),
]


def build_prompt(question):
    prompt = ""
    for ex in EXEMPLARS:
        prompt += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
    prompt += f"Q: {question}\nA: Let's think step by step.\n"
    return prompt


def extract_answer(text):
    if "####" in text:
        after = text.split("####")[-1].strip()
        m = re.search(r'[\d,]+\.?\d*', after)
        return m.group(0).replace(',', '') if m else ""
    nums = re.findall(r'[\d,]+\.?\d*', text)
    return nums[-1].replace(',', '') if nums else ""


def normalize_answer(ans):
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
    truncated = False

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
            if re.search(r'\d+\s*\n', after):
                break
        if "\nQ:" in current_text or "\n\nQ:" in current_text:
            idx = current_text.find("\nQ:")
            if idx > 0:
                truncated_text = current_text[:idx]
                generated_ids = tokenizer.encode(truncated_text, add_special_tokens=False)
                truncated = True
            break
        current_input = next_token

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    reasoning_ids = torch.tensor([generated_ids], device=model.device)

    del past_kv, outputs
    gc.collect(); torch.cuda.empty_cache()
    return text, prompt_ids, reasoning_ids


@torch.no_grad()
def build_prompt_cache(model, prompt_ids, num_layers):
    """Build clean KV cache for prompt tokens only."""
    outputs = model(input_ids=prompt_ids, use_cache=True)
    prompt_cache = outputs.past_key_values
    del outputs
    gc.collect(); torch.cuda.empty_cache()
    return prompt_cache


def select_positions(reasoning_len, position_type, noise_frac=NOISE_FRAC):
    """Select positions by position band."""
    n = max(1, int(reasoning_len * noise_frac))
    if position_type == 'late':
        return list(range(max(0, reasoning_len - n), reasoning_len))
    elif position_type == 'early':
        return list(range(min(n, reasoning_len)))
    elif position_type == 'mid':
        mid = reasoning_len // 2
        half_n = n // 2
        start = max(0, mid - half_n)
        end = min(reasoning_len, start + n)
        return list(range(start, end))
    return []


def perturb_direction(tensor):
    """Replace direction with random, preserve per-head norm.
    tensor shape: [1, n_heads, 1, head_dim]
    """
    noise = torch.randn_like(tensor)
    # Per-head norm matching
    norms = tensor.norm(dim=-1, keepdim=True)  # [1, n_heads, 1, 1]
    noise_norms = noise.norm(dim=-1, keepdim=True) + 1e-8
    return noise * (norms / noise_norms)


def perturb_magnitude(tensor, sigma=1.0):
    """Scale per-head norm randomly, preserve direction exactly.
    tensor shape: [1, n_heads, 1, head_dim]
    """
    # Direction unit vectors per head
    norms = tensor.norm(dim=-1, keepdim=True)  # [1, n_heads, 1, 1]
    direction = tensor / (norms + 1e-8)
    # Random scale factor per head: (1 + delta), delta ~ N(0, sigma)
    delta = torch.randn(norms.shape, device=tensor.device, dtype=tensor.dtype) * sigma
    scale = (1 + delta).clamp(min=0.01)
    return direction * norms * scale


@torch.no_grad()
def evaluate_condition(model, tokenizer, prompt_cache, reasoning_tokens,
                       positions_to_noise, prompt_len, num_layers, true_answer,
                       perturb_component='kv', perturb_type='direction',
                       magnitude_sigma=1.0):
    """
    Step through reasoning tokens, apply geometric perturbation at selected positions.
    Returns answer accuracy and text prediction accuracy.
    """
    reasoning_len = reasoning_tokens.shape[1]
    noise_set = set(positions_to_noise)

    # Clone prompt cache
    cache = DynamicCache()
    for l in range(num_layers):
        k = prompt_cache.layers[l].keys.clone()
        v = prompt_cache.layers[l].values.clone()
        cache.update(k, v, l)

    text_correct = 0
    text_total = 0
    last_logits = None

    # Track perturbation stats
    perturb_l2_total = 0.0
    perturb_count = 0

    for i in range(reasoning_len):
        token = reasoning_tokens[:, i:i+1]
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

        # Apply perturbation at this position if selected
        if i in noise_set:
            for l in range(num_layers):
                pos = prompt_len + i
                k_orig = cache.layers[l].keys[:, :, pos:pos+1, :].clone()
                v_orig = cache.layers[l].values[:, :, pos:pos+1, :].clone()

                # Perturb K
                if perturb_component in ('kv', 'k'):
                    k_slice = cache.layers[l].keys[:, :, pos:pos+1, :]
                    if perturb_type == 'direction':
                        k_new = perturb_direction(k_slice)
                    else:
                        k_new = perturb_magnitude(k_slice, sigma=magnitude_sigma)
                    cache.layers[l].keys[:, :, pos:pos+1, :] = k_new
                    perturb_l2_total += (k_new - k_orig).float().norm().item() ** 2

                # Perturb V
                if perturb_component in ('kv', 'v'):
                    v_slice = cache.layers[l].values[:, :, pos:pos+1, :]
                    if perturb_type == 'direction':
                        v_new = perturb_direction(v_slice)
                    else:
                        v_new = perturb_magnitude(v_slice, sigma=magnitude_sigma)
                    cache.layers[l].values[:, :, pos:pos+1, :] = v_new
                    perturb_l2_total += (v_new - v_orig).float().norm().item() ** 2

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

    # Generate answer from last reasoning token's logits
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
        if "\nQ:" in decoded:
            break
        del g

    answer_text = tokenizer.decode(generated, skip_special_tokens=True)
    answer = normalize_answer(extract_answer(answer_text)) if extract_answer(answer_text) else ""
    correct = (answer == true_answer)

    # Compute RMS perturbation
    rms_perturb = (perturb_l2_total / max(perturb_count, 1)) ** 0.5

    del cache, gen_cache
    gc.collect(); torch.cuda.empty_cache()

    return {
        'correct': correct,
        'answer': answer,
        'text_accuracy': text_accuracy,
        'answer_text': answer_text[:150],
        'rms_perturbation': rms_perturb,
    }


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_figures(results, agg, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Figure 1: Direction vs Magnitude at Late positions ──
    late_conds = [c for c in CONDITIONS if c[1] == 'late']
    names = [c[0].replace('_late', '') for c in late_conds]
    acc_vals = []
    txt_vals = []
    for cond in late_conds:
        key = cond[0]
        if key in agg:
            acc_vals.append(agg[key]['accuracy'] * 100)
            txt_vals.append(agg[key]['text_accuracy'] * 100)
        else:
            acc_vals.append(0)
            txt_vals.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(names))
    w = 0.35

    # Color by perturbation type
    colors = []
    for c in late_conds:
        if c[3] == 'direction':
            colors.append('#e74c3c')  # red for direction
        else:
            colors.append('#3498db')  # blue for magnitude

    bars1 = ax1.bar(x, acc_vals, w, color=colors, alpha=0.85)
    ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax1.set_title('Answer Accuracy — Late 5% Positions', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.2, axis='y')
    for b, v in zip(bars1, acc_vals):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(x, txt_vals, w, color=colors, alpha=0.85)
    ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
    ax2.set_title('Text Quality — Late 5% Positions', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.2, axis='y')
    for b, v in zip(bars2, txt_vals):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', alpha=0.85, label='Direction'),
                       Patch(facecolor='#3498db', alpha=0.85, label='Magnitude')]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.suptitle('Exp 022: Direction vs Magnitude KV Perturbation (Late 5%)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dir_vs_mag_late.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: Magnitude dose-response curve ──
    mag_conds = [("mag_kv_05_late", 0.5), ("mag_kv_10_late", 1.0), ("mag_kv_14_late", 1.414)]
    dir_key = "dir_kv_late"

    sigmas = [s for _, s in mag_conds]
    mag_accs = [agg[k]['accuracy'] * 100 if k in agg else 0 for k, _ in mag_conds]
    mag_txts = [agg[k]['text_accuracy'] * 100 if k in agg else 0 for k, _ in mag_conds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(sigmas, mag_accs, 'o-', color='#3498db', linewidth=2, markersize=8, label='Magnitude')
    if dir_key in agg:
        ax1.axhline(y=agg[dir_key]['accuracy'] * 100, color='#e74c3c',
                    linestyle='--', linewidth=2, label=f"Direction ({agg[dir_key]['accuracy']*100:.0f}%)")
    ax1.set_xlabel('Magnitude sigma', fontsize=12)
    ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax1.set_title('Magnitude Dose-Response: Accuracy', fontsize=13)
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(sigmas, mag_txts, 'o-', color='#3498db', linewidth=2, markersize=8, label='Magnitude')
    if dir_key in agg:
        ax2.axhline(y=agg[dir_key]['text_accuracy'] * 100, color='#e74c3c',
                    linestyle='--', linewidth=2, label=f"Direction ({agg[dir_key]['text_accuracy']*100:.0f}%)")
    ax2.set_xlabel('Magnitude sigma', fontsize=12)
    ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
    ax2.set_title('Magnitude Dose-Response: Text Quality', fontsize=13)
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Exp 022: Magnitude Perturbation Dose-Response', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'magnitude_dose_response.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3: K vs V decomposition ──
    kv_conds = [
        ("dir_kv_late", "Dir K+V"),
        ("dir_k_late", "Dir K only"),
        ("dir_v_late", "Dir V only"),
        ("mag_kv_10_late", "Mag K+V"),
        ("mag_k_10_late", "Mag K only"),
        ("mag_v_10_late", "Mag V only"),
    ]
    kv_names = [n for _, n in kv_conds]
    kv_accs = [agg[k]['accuracy'] * 100 if k in agg else 0 for k, _ in kv_conds]
    kv_txts = [agg[k]['text_accuracy'] * 100 if k in agg else 0 for k, _ in kv_conds]
    kv_colors = ['#e74c3c', '#ff9999', '#ff9999', '#3498db', '#99ccff', '#99ccff']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(kv_names))

    bars1 = ax1.bar(x, kv_accs, 0.6, color=kv_colors, alpha=0.85)
    ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax1.set_title('K vs V Decomposition: Accuracy', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(kv_names, rotation=30, ha='right', fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.2, axis='y')
    for b, v in zip(bars1, kv_accs):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(x, kv_txts, 0.6, color=kv_colors, alpha=0.85)
    ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
    ax2.set_title('K vs V Decomposition: Text Quality', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(kv_names, rotation=30, ha='right', fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.2, axis='y')
    for b, v in zip(bars2, kv_txts):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Exp 022: K vs V Component Decomposition (Late 5%)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'k_vs_v_decomposition.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 4: Position comparison (early/mid/late × dir/mag) ──
    pos_conds = [
        ("dir_kv_early", "Dir Early"),
        ("mag_kv_10_early", "Mag Early"),
        ("dir_kv_mid", "Dir Mid"),
        ("mag_kv_10_mid", "Mag Mid"),
        ("dir_kv_late", "Dir Late"),
        ("mag_kv_10_late", "Mag Late"),
    ]
    pos_names = [n for _, n in pos_conds]
    pos_accs = [agg[k]['accuracy'] * 100 if k in agg else 0 for k, _ in pos_conds]
    pos_txts = [agg[k]['text_accuracy'] * 100 if k in agg else 0 for k, _ in pos_conds]
    pos_colors = ['#e74c3c', '#3498db'] * 3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(pos_names))

    bars1 = ax1.bar(x, pos_accs, 0.6, color=pos_colors, alpha=0.85)
    ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax1.set_title('Position × Geometry: Accuracy', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pos_names, rotation=30, ha='right', fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.2, axis='y')
    for b, v in zip(bars1, pos_accs):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(x, pos_txts, 0.6, color=pos_colors, alpha=0.85)
    ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
    ax2.set_title('Position × Geometry: Text Quality', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pos_names, rotation=30, ha='right', fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.2, axis='y')
    for b, v in zip(bars2, pos_txts):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Exp 022: Position × Geometric Component', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'position_x_geometry.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 5: Dissociation summary (scatter: acc_drop vs text_drop) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    for cond_name, data in agg.items():
        color = '#e74c3c' if 'dir' in cond_name else '#3498db'
        marker = 'o' if 'late' in cond_name else ('s' if 'mid' in cond_name else '^')
        ax.scatter(data['text_drop'] * 100, data['accuracy_drop'] * 100,
                  color=color, marker=marker, s=120, alpha=0.8, zorder=5)
        ax.annotate(cond_name.replace('_late', '').replace('_mid', '*').replace('_early', '**'),
                   (data['text_drop'] * 100, data['accuracy_drop'] * 100),
                   textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Equal effect')
    ax.set_xlabel('Text Drop (%)', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Dissociation Map: Accuracy Drop vs Text Drop\n'
                 'Red=Direction, Blue=Magnitude | o=Late, s=Mid, ^=Early', fontsize=12)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dissociation_map.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers, t={time.time()-t0:.0f}s")

    # Verify cache access pattern
    test_input = tokenizer("test", return_tensors="pt").to(model.device)
    test_out = model(**test_input, use_cache=True)
    test_cache = test_out.past_key_values
    assert hasattr(test_cache, 'layers'), "Cache must have layers attribute"
    print(f"Cache access verified: layers[0].keys.shape = {test_cache.layers[0].keys.shape}")
    del test_input, test_out, test_cache
    gc.collect(); torch.cuda.empty_cache()

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:NUM_PROBLEMS]

    results = []
    n_valid = 0

    for pi, ds_idx in enumerate(indices):
        elapsed = time.time() - t0
        if elapsed > 1550:  # Leave buffer for figures
            print(f"\n  Time limit at problem {pi}/{len(indices)}")
            break

        item = ds[ds_idx]
        true_answer = normalize_answer(extract_answer(item['answer']))
        prompt = build_prompt(item['question'])

        print(f"\nProblem {pi+1}/{len(indices)} (idx={ds_idx}), "
              f"true={true_answer}, t={elapsed:.0f}s")

        trace, prompt_ids, reasoning_ids = generate_trace(model, tokenizer, prompt)
        gen_answer = normalize_answer(extract_answer(trace)) if extract_answer(trace) else ""
        if gen_answer != true_answer:
            print(f"  WRONG: {gen_answer}")
            continue

        # Truncate before "####" (token 820 in Qwen tokenizer)
        HASH4_TOKEN = 820
        ids_list = reasoning_ids[0].tolist()
        hash_pos = None
        for hi, tid in enumerate(ids_list):
            if tid == HASH4_TOKEN:
                hash_pos = hi
                break
        if hash_pos is None or hash_pos < 10:
            print(f"  Skip: no #### token in reasoning IDs")
            continue
        reasoning_ids_truncated = reasoning_ids[:, :hash_pos]
        prompt_len = prompt_ids.shape[1]
        reasoning_len = reasoning_ids_truncated.shape[1]

        if reasoning_len < 20 or (prompt_len + reasoning_len) > MAX_SEQ_LEN:
            print(f"  Skip: R={reasoning_len}, total={prompt_len + reasoning_len}")
            continue

        # Clean evaluation
        prompt_cache = build_prompt_cache(model, prompt_ids, num_layers)
        clean_eval = evaluate_condition(
            model, tokenizer, prompt_cache, reasoning_ids_truncated,
            [], prompt_len, num_layers, true_answer)
        clean_text_acc = clean_eval['text_accuracy']
        print(f"  Clean: acc={'Y' if clean_eval['correct'] else 'N'}, "
              f"text={clean_text_acc:.1%}, ans='{clean_eval['answer']}'")

        if not clean_eval['correct']:
            print(f"  Skip: clean evaluation wrong answer")
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
        for cond_name, pos_type, component, ptype, sigma in CONDITIONS:
            positions = select_positions(reasoning_len, pos_type)
            if not positions:
                continue

            mean_pos = np.mean([p / max(reasoning_len - 1, 1) for p in positions])

            pc = build_prompt_cache(model, prompt_ids, num_layers)
            ev = evaluate_condition(
                model, tokenizer, pc, reasoning_ids_truncated,
                positions, prompt_len, num_layers, true_answer,
                perturb_component=component, perturb_type=ptype,
                magnitude_sigma=sigma)

            print(f"    {cond_name}: acc={'Y' if ev['correct'] else 'N'}, "
                  f"text={ev['text_accuracy']:.1%}, pos={mean_pos:.3f}, "
                  f"rms={ev['rms_perturbation']:.1f}")

            problem_result['evaluations'][cond_name] = {
                'correct': ev['correct'],
                'answer': ev['answer'],
                'text_accuracy': ev['text_accuracy'],
                'mean_pos': float(mean_pos),
                'n_noised': len(positions),
                'rms_perturbation': ev['rms_perturbation'],
            }

        results.append(problem_result)
        del prompt_ids, reasoning_ids, reasoning_ids_truncated, prompt_cache
        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_valid} valid, {len(results)} processed")

    summary = {
        'experiment': 'exp_022_geometric_dissociation',
        'model': MODEL_NAME,
        'n_valid': n_valid,
        'n_processed': len(results),
        'noise_fraction': NOISE_FRAC,
        'conditions': [c[0] for c in CONDITIONS],
    }

    agg = {}
    for cond_name, _, _, _, _ in CONDITIONS:
        accs = [1 if p['evaluations'][cond_name]['correct'] else 0
                for p in results if cond_name in p.get('evaluations', {})]
        txts = [p['evaluations'][cond_name]['text_accuracy']
                for p in results if cond_name in p.get('evaluations', {})]
        ctxts = [p['clean_text_acc'] for p in results if cond_name in p.get('evaluations', {})]
        rms_list = [p['evaluations'][cond_name]['rms_perturbation']
                    for p in results if cond_name in p.get('evaluations', {})]

        if accs:
            a_mean = np.mean(accs)
            t_mean = np.mean(txts)
            ct_mean = np.mean(ctxts)
            a_drop = 1.0 - a_mean
            t_drop = ct_mean - t_mean
            dissoc = a_drop - t_drop
            agg[cond_name] = {
                'n': len(accs),
                'accuracy': float(a_mean),
                'text_accuracy': float(t_mean),
                'clean_text_accuracy': float(ct_mean),
                'accuracy_drop': float(a_drop),
                'text_drop': float(t_drop),
                'dissociation': float(dissoc),
                'rms_perturbation': float(np.mean(rms_list)),
            }

    # Print summary table
    print(f"\n{'Condition':<22} {'Acc%':>6} {'Text%':>6} {'AccDrop':>8} {'TxtDrop':>8} "
          f"{'Dissoc':>8} {'RMS':>8} {'n':>3}")
    print("-" * 80)
    for cond_name, _, _, _, _ in CONDITIONS:
        if cond_name in agg:
            d = agg[cond_name]
            print(f"  {cond_name:<20} {d['accuracy']*100:>5.1f}% {d['text_accuracy']*100:>5.1f}% "
                  f"{d['accuracy_drop']*100:>7.1f}% {d['text_drop']*100:>7.1f}% "
                  f"{d['dissociation']*100:>7.1f}% {d['rms_perturbation']:>7.1f} {d['n']:>3}")

    # Key comparisons
    print(f"\n── KEY COMPARISONS ──")
    for pos_label, dir_key, mag_key in [
        ("LATE", "dir_kv_late", "mag_kv_10_late"),
        ("LATE (energy-matched)", "dir_kv_late", "mag_kv_14_late"),
        ("MID", "dir_kv_mid", "mag_kv_10_mid"),
        ("EARLY", "dir_kv_early", "mag_kv_10_early"),
    ]:
        if dir_key in agg and mag_key in agg:
            d_acc = agg[dir_key]['accuracy_drop']
            m_acc = agg[mag_key]['accuracy_drop']
            d_txt = agg[dir_key]['text_drop']
            m_txt = agg[mag_key]['text_drop']
            acc_gap = (d_acc - m_acc) * 100
            txt_gap = (d_txt - m_txt) * 100
            print(f"\n  {pos_label}:")
            print(f"    Direction: acc_drop={d_acc*100:.1f}%, text_drop={d_txt*100:.1f}%")
            print(f"    Magnitude: acc_drop={m_acc*100:.1f}%, text_drop={m_txt*100:.1f}%")
            print(f"    Gap: acc_drop dir-mag = {acc_gap:+.1f}pp, text_drop dir-mag = {txt_gap:+.1f}pp")
            # Double dissociation check
            dd = (d_acc > m_acc and d_txt < m_txt) or (d_acc < m_acc and d_txt > m_txt)
            print(f"    Double dissociation: {'YES' if dd else 'NO'}")

    # K vs V comparison
    print(f"\n── K vs V DECOMPOSITION ──")
    for label, k_key, v_key in [
        ("Direction", "dir_k_late", "dir_v_late"),
        ("Magnitude", "mag_k_10_late", "mag_v_10_late"),
    ]:
        if k_key in agg and v_key in agg:
            print(f"\n  {label}:")
            print(f"    K-only: acc={agg[k_key]['accuracy']*100:.1f}%, text={agg[k_key]['text_accuracy']*100:.1f}%")
            print(f"    V-only: acc={agg[v_key]['accuracy']*100:.1f}%, text={agg[v_key]['text_accuracy']*100:.1f}%")

    summary['aggregated'] = agg
    summary['elapsed_seconds'] = time.time() - t0

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=_convert)
    with open(os.path.join(RESULTS_DIR, 'per_problem.json'), 'w') as f:
        json.dump(results, f, indent=2, default=_convert)

    print("\nGenerating figures...")
    generate_figures(results, agg, RESULTS_DIR)
    print(f"\nDone. Total: {summary['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
