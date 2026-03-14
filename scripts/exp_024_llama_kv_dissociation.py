#!/usr/bin/env python3
"""
Experiment 024: Cross-Model K-V Dissociation Replication on Llama-3.1-8B

Replicates exp_022/023's K-V geometric dissociation on Llama-3.1-8B-Instruct.
Tests whether the dramatic K-V functional dissociation found on Qwen3-4B-Base
(V-magnitude at late 5% = ZERO effect, K devastating) generalizes across
model families.

Conditions at late 5% and 10%:
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
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NUM_PROBLEMS = 200  # attempt many, time-limited
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 1536
SEED = 42
HASH4_TOKEN = 827  # "####" in Llama tokenizer
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_024")

# Same 8-shot exemplars used throughout program (from exp_005)
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

# ── Condition definitions ───────────────────────────────────────────────
# (name, position_type, noise_frac, perturb_component, perturb_type, sigma)
CONDITIONS = [
    # Late 5% — core K/V decomposition (matches exp_022/023 on Qwen)
    ("dir_kv_late",       "late", 0.05, "kv", "direction", 0.0),
    ("dir_k_late",        "late", 0.05, "k",  "direction", 0.0),
    ("dir_v_late",        "late", 0.05, "v",  "direction", 0.0),
    ("mag_k_10_late",     "late", 0.05, "k",  "magnitude", 1.0),
    ("mag_v_10_late",     "late", 0.05, "v",  "magnitude", 1.0),
    ("mag_kv_10_late",    "late", 0.05, "kv", "magnitude", 1.0),
    # Late 10% — higher dose for Llama's robustness
    ("dir_kv_late10",     "late", 0.10, "kv", "direction", 0.0),
    ("mag_kv_10_late10",  "late", 0.10, "kv", "magnitude", 1.0),
    # Early 5% — position control
    ("dir_kv_early",      "early", 0.05, "kv", "direction", 0.0),
    ("mag_kv_10_early",   "early", 0.05, "kv", "magnitude", 1.0),
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
    # Instruct model pattern
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
        # Instruct model answer pattern
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
def build_prompt_cache(model, prompt_ids, num_layers):
    """Build clean KV cache for prompt tokens only."""
    outputs = model(input_ids=prompt_ids, use_cache=True)
    prompt_cache = outputs.past_key_values
    del outputs
    gc.collect(); torch.cuda.empty_cache()
    return prompt_cache


def select_positions(reasoning_len, position_type, noise_frac):
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
    norms = tensor.norm(dim=-1, keepdim=True)  # [1, n_heads, 1, 1]
    noise_norms = noise.norm(dim=-1, keepdim=True) + 1e-8
    return noise * (norms / noise_norms)


def perturb_magnitude(tensor, sigma=1.0):
    """Scale per-head norm randomly, preserve direction exactly.
    tensor shape: [1, n_heads, 1, head_dim]
    """
    norms = tensor.norm(dim=-1, keepdim=True)
    direction = tensor / (norms + 1e-8)
    delta = torch.randn(norms.shape, device=tensor.device, dtype=tensor.dtype) * sigma
    scale = (1 + delta).clamp(min=0.01)
    return direction * norms * scale


def find_truncation_point(reasoning_ids, tokenizer):
    """Find where to truncate reasoning before the answer marker.
    Returns index to truncate at, or None if no marker found.
    Handles both '####' token and 'The answer is' pattern.
    """
    ids_list = reasoning_ids[0].tolist()

    # Method 1: Look for #### token (827 in Llama)
    for i, tid in enumerate(ids_list):
        if tid == HASH4_TOKEN:
            if i >= 10:
                return i
            break

    # Method 2: Decode and search for "#### " or "The answer is"
    text = tokenizer.decode(ids_list, skip_special_tokens=True)
    if "####" in text:
        # Find the token position where #### starts
        prefix = text[:text.index("####")]
        prefix_toks = tokenizer.encode(prefix, add_special_tokens=False)
        pos = len(prefix_toks)
        if pos >= 10:
            return pos

    # Method 3: "The answer is" pattern (common in instruct models)
    m = re.search(r'[Tt]he (?:final )?answer is', text)
    if m:
        prefix = text[:m.start()]
        prefix_toks = tokenizer.encode(prefix, add_special_tokens=False)
        pos = len(prefix_toks)
        if pos >= 10:
            return pos

    return None


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
        if re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?\s*-?[\d,]+', decoded):
            break
        if "\nQ:" in decoded:
            break
        del g

    answer_text = tokenizer.decode(generated, skip_special_tokens=True)
    raw_ans = extract_answer(answer_text)
    answer = normalize_answer(raw_ans) if raw_ans else ""
    correct = (answer == true_answer)

    rms_perturb = (perturb_l2_total / max(perturb_count, 1)) ** 0.5

    del cache, gen_cache
    gc.collect(); torch.cuda.empty_cache()

    return {
        'correct': correct,
        'answer': answer,
        'text_accuracy': text_accuracy,
        'answer_text': answer_text[:200],
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

    # ── Figure 1: K vs V decomposition at late 5% ──
    kv_conds = [
        ("dir_kv_late", "Dir K+V"),
        ("dir_k_late", "Dir K"),
        ("dir_v_late", "Dir V"),
        ("mag_kv_10_late", "Mag K+V"),
        ("mag_k_10_late", "Mag K"),
        ("mag_v_10_late", "Mag V"),
    ]
    kv_names = [n for k, n in kv_conds if k in agg]
    kv_keys = [k for k, _ in kv_conds if k in agg]
    kv_accs = [agg[k]['accuracy'] * 100 for k in kv_keys]
    kv_txts = [agg[k]['text_accuracy'] * 100 for k in kv_keys]
    kv_colors = []
    for k in kv_keys:
        if 'dir' in k:
            kv_colors.append('#e74c3c' if 'kv' in k else '#ff9999')
        else:
            kv_colors.append('#3498db' if 'kv' in k else '#99ccff')

    if kv_names:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(kv_names))

        bars1 = ax1.bar(x, kv_accs, 0.6, color=kv_colors, alpha=0.85)
        ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax1.set_title('Llama K vs V: Accuracy (Late 5%)', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(kv_names, rotation=30, ha='right', fontsize=10)
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.2, axis='y')
        for b, v in zip(bars1, kv_accs):
            ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f'{v:.0f}%', ha='center', va='bottom', fontsize=9)

        bars2 = ax2.bar(x, kv_txts, 0.6, color=kv_colors, alpha=0.85)
        ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
        ax2.set_title('Llama K vs V: Text Quality (Late 5%)', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(kv_names, rotation=30, ha='right', fontsize=10)
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.2, axis='y')
        for b, v in zip(bars2, kv_txts):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f'{v:.0f}%', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Exp 024: K vs V Decomposition on Llama-3.1-8B (Late 5%)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'llama_kv_decomposition.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 2: Cross-model comparison (Qwen vs Llama K-V pattern) ──
    # Qwen results from exp_023 (hardcoded for comparison)
    qwen_results = {
        'dir_kv': 29.2, 'dir_k': 25.0, 'dir_v': 79.2,
        'mag_kv_10': 0.0, 'mag_k_10': 16.7, 'mag_v_10': 100.0,
    }
    comparison_keys = ['dir_kv', 'dir_k', 'dir_v', 'mag_kv_10', 'mag_k_10', 'mag_v_10']
    comparison_labels = ['Dir K+V', 'Dir K', 'Dir V', 'Mag K+V', 'Mag K', 'Mag V']

    qwen_vals = [qwen_results[k] for k in comparison_keys]
    llama_vals = []
    for k in comparison_keys:
        lk = k + '_late'
        llama_vals.append(agg[lk]['accuracy'] * 100 if lk in agg else float('nan'))

    if any(not np.isnan(v) for v in llama_vals):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(comparison_labels))
        w = 0.35
        bars1 = ax.bar(x - w/2, qwen_vals, w, label='Qwen3-4B-Base', color='#e67e22', alpha=0.85)
        bars2 = ax.bar(x + w/2, llama_vals, w, label='Llama-3.1-8B-Inst', color='#2ecc71', alpha=0.85)
        ax.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax.set_title('Cross-Model K-V Dissociation: Accuracy at Late 5%', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_labels, rotation=30, ha='right', fontsize=11)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')
        for b, v in zip(bars1, qwen_vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f'{v:.0f}%', ha='center', va='bottom', fontsize=8)
        for b, v in zip(bars2, llama_vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                        f'{v:.0f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'cross_model_kv_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 3: Dissociation map (accuracy drop vs text drop) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    for cond_name, data in agg.items():
        color = '#e74c3c' if 'dir' in cond_name else '#3498db'
        marker = 'o' if 'late' in cond_name and '10' not in cond_name.split('late')[-1][:2] else \
                 's' if 'late10' in cond_name else '^'
        ax.scatter(data['text_drop'] * 100, data['accuracy_drop'] * 100,
                  color=color, marker=marker, s=120, alpha=0.8, zorder=5)
        label = cond_name.replace('_late', '').replace('_early', '*')
        ax.annotate(label, (data['text_drop'] * 100, data['accuracy_drop'] * 100),
                   textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Equal effect')
    ax.set_xlabel('Text Drop (%)', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Llama Dissociation Map\nRed=Direction, Blue=Magnitude | o=5%, s=10%, ^=Early',
                 fontsize=12)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'llama_dissociation_map.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 4: 5% vs 10% noise comparison ──
    dose_conds = [
        ("dir_kv_late", "Dir 5%"), ("dir_kv_late10", "Dir 10%"),
        ("mag_kv_10_late", "Mag 5%"), ("mag_kv_10_late10", "Mag 10%"),
    ]
    dose_keys = [k for k, _ in dose_conds if k in agg]
    dose_names = [n for k, n in dose_conds if k in agg]
    if dose_keys:
        dose_accs = [agg[k]['accuracy'] * 100 for k in dose_keys]
        dose_txts = [agg[k]['text_accuracy'] * 100 for k in dose_keys]
        dose_colors = ['#e74c3c', '#c0392b', '#3498db', '#2980b9'][:len(dose_keys)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(dose_names))
        ax1.bar(x, dose_accs, 0.6, color=dose_colors, alpha=0.85)
        ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax1.set_title('Noise Dose-Response: Accuracy', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(dose_names, rotation=30, ha='right')
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.2, axis='y')
        for i, v in enumerate(dose_accs):
            ax1.text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=9)

        ax2.bar(x, dose_txts, 0.6, color=dose_colors, alpha=0.85)
        ax2.set_ylabel('Text Accuracy (%)', fontsize=12)
        ax2.set_title('Noise Dose-Response: Text', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(dose_names, rotation=30, ha='right')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.2, axis='y')
        for i, v in enumerate(dose_txts):
            ax2.text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=9)

        plt.suptitle('Exp 024: Llama 5% vs 10% Noise', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'llama_dose_response.png'),
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
    k_shape = test_cache.layers[0].keys.shape
    print(f"Cache verified: layers[0].keys.shape = {k_shape}")
    print(f"  n_kv_heads={k_shape[1]}, head_dim={k_shape[3]}")
    del test_input, test_out, test_cache
    gc.collect(); torch.cuda.empty_cache()

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
        if elapsed > 1500:  # Leave buffer for figures
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

        # Find truncation point (before answer marker)
        trunc_pos = find_truncation_point(reasoning_ids, tokenizer)
        if trunc_pos is None or trunc_pos < 10:
            print(f"  Skip: no truncation point found in trace")
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
        prompt_cache = build_prompt_cache(model, prompt_ids, num_layers)
        clean_eval = evaluate_condition(
            model, tokenizer, prompt_cache, reasoning_ids_truncated,
            [], prompt_len, num_layers, true_answer)
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
        for cond_name, pos_type, noise_frac, component, ptype, sigma in CONDITIONS:
            positions = select_positions(reasoning_len, pos_type, noise_frac)
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
                  f"text={ev['text_accuracy']:.1%}, n_pos={len(positions)}, "
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
    print(f"RESULTS: {n_valid} valid, {len(results)} processed, "
          f"{n_skip_wrong} wrong, {n_skip_trunc} truncation skips")

    summary = {
        'experiment': 'exp_024_llama_kv_dissociation',
        'model': MODEL_NAME,
        'n_valid': n_valid,
        'n_processed': len(results),
        'n_skip_wrong': n_skip_wrong,
        'n_skip_trunc': n_skip_trunc,
        'conditions': [c[0] for c in CONDITIONS],
    }

    agg = {}
    for cond_tuple in CONDITIONS:
        cond_name = cond_tuple[0]
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
    for cond_tuple in CONDITIONS:
        cond_name = cond_tuple[0]
        if cond_name in agg:
            d = agg[cond_name]
            print(f"  {cond_name:<20} {d['accuracy']*100:>5.1f}% {d['text_accuracy']*100:>5.1f}% "
                  f"{d['accuracy_drop']*100:>7.1f}% {d['text_drop']*100:>7.1f}% "
                  f"{d['dissociation']*100:>7.1f}% {d['rms_perturbation']:>7.1f} {d['n']:>3}")

    # Key K vs V comparisons
    print(f"\n── K vs V DECOMPOSITION (Late 5%) ──")
    for label, k_key, v_key in [
        ("Direction", "dir_k_late", "dir_v_late"),
        ("Magnitude", "mag_k_10_late", "mag_v_10_late"),
    ]:
        if k_key in agg and v_key in agg:
            print(f"\n  {label}:")
            print(f"    K-only: acc={agg[k_key]['accuracy']*100:.1f}%, "
                  f"text={agg[k_key]['text_accuracy']*100:.1f}%")
            print(f"    V-only: acc={agg[v_key]['accuracy']*100:.1f}%, "
                  f"text={agg[v_key]['text_accuracy']*100:.1f}%")
            k_acc_drop = agg[k_key]['accuracy_drop']
            v_acc_drop = agg[v_key]['accuracy_drop']
            ratio = k_acc_drop / max(v_acc_drop, 0.001)
            print(f"    K/V acc_drop ratio: {ratio:.1f}x")

    # Cross-model comparison
    print(f"\n── CROSS-MODEL COMPARISON (Qwen exp_023 vs Llama exp_024) ──")
    qwen = {'dir_kv': 29.2, 'dir_k': 25.0, 'dir_v': 79.2,
            'mag_kv_10': 0.0, 'mag_k_10': 16.7, 'mag_v_10': 100.0}
    for k, qv in qwen.items():
        lk = k + '_late'
        if lk in agg:
            lv = agg[lk]['accuracy'] * 100
            print(f"  {k:<12}: Qwen={qv:.0f}%, Llama={lv:.0f}%, gap={lv-qv:+.0f}pp")

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
