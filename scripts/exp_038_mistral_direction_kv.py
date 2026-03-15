#!/usr/bin/env python3
"""
Experiment 038: Mistral-7B-v0.3 Direction Perturbation — K vs V at 3 Position Bands

Exp_037 showed K ≈ V on Mistral-Base under MAGNITUDE perturbation (gap +0-2pp).
This was the only model out of 5 variants to show K ≈ V.

Direction perturbation (random replacement, norm-preserved) is the more
mechanistically meaningful test: it destroys the attention ROUTING pattern
(for K) or the VALUE content (for V) while preserving energy.

On Llama-3.1-8B (exp_028), direction perturbation showed:
  - Late 5%:  K=22%, V=88%  → V-K gap = +66pp (K > V CONFIRMED)
  - Mid 5%:   K=6%,  V=100% → V-K gap = +94pp
  - Early 5%: K=0%,  V=100% → V-K gap = +100pp

If Mistral shows:
  A) K > V under direction perturbation → K ≈ V was magnitude-specific.
     The routing > throughput hierarchy is universal under the RIGHT perturbation type.
  B) K ≈ V under direction perturbation too → K > V is genuinely non-universal.
     Mistral's architecture (sliding window attention?) creates balanced K-V usage.
  C) V > K (reversal) → fundamental K-V dynamics differ on Mistral.

All at 5% noise fraction, matching exp_028 exactly for cross-model comparison.
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
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
NUM_PROBLEMS = 200  # attempt many, time-limited
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 2048
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_038")

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

# ── Condition definitions ───────────────────────────────────────────────
# (name, position_type, noise_frac, perturb_component)
CONDITIONS = [
    # K-only vs V-only at THREE position bands
    ("dir_k_early",  "early", 0.05, "k"),
    ("dir_v_early",  "early", 0.05, "v"),
    ("dir_k_mid",    "mid",   0.05, "k"),
    ("dir_v_mid",    "mid",   0.05, "v"),
    ("dir_k_late",   "late",  0.05, "k"),
    ("dir_v_late",   "late",  0.05, "v"),
    # Combined K+V for calibration
    ("dir_kv_early", "early", 0.05, "kv"),
    ("dir_kv_mid",   "mid",   0.05, "kv"),
    ("dir_kv_late",  "late",  0.05, "kv"),
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
    norms = tensor.norm(dim=-1, keepdim=True)
    noise_norms = noise.norm(dim=-1, keepdim=True) + 1e-8
    return noise * (norms / noise_norms)


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
def evaluate_condition(model, tokenizer, prompt_cache, reasoning_tokens,
                       positions_to_noise, prompt_len, num_layers, true_answer,
                       perturb_component='kv'):
    """
    Step through reasoning tokens, apply direction perturbation at selected positions.
    Returns answer accuracy, text prediction accuracy, and perturbation stats.
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

    k_perturb_l2 = 0.0
    v_perturb_l2 = 0.0
    k_signal_l2 = 0.0
    v_signal_l2 = 0.0
    perturb_count = 0

    for i in range(reasoning_len):
        token = reasoning_tokens[:, i:i+1]
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

        if i in noise_set:
            for l in range(num_layers):
                pos = prompt_len + i
                k_orig = cache.layers[l].keys[:, :, pos:pos+1, :].clone()
                v_orig = cache.layers[l].values[:, :, pos:pos+1, :].clone()

                # Record signal norms
                k_signal_l2 += k_orig.float().norm().item() ** 2
                v_signal_l2 += v_orig.float().norm().item() ** 2

                # Perturb K
                if perturb_component in ('kv', 'k'):
                    k_new = perturb_direction(cache.layers[l].keys[:, :, pos:pos+1, :])
                    cache.layers[l].keys[:, :, pos:pos+1, :] = k_new
                    k_perturb_l2 += (k_new - k_orig).float().norm().item() ** 2

                # Perturb V
                if perturb_component in ('kv', 'v'):
                    v_new = perturb_direction(cache.layers[l].values[:, :, pos:pos+1, :])
                    cache.layers[l].values[:, :, pos:pos+1, :] = v_new
                    v_perturb_l2 += (v_new - v_orig).float().norm().item() ** 2

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
    raw_ans = extract_answer(answer_text)
    answer = normalize_answer(raw_ans) if raw_ans else ""
    correct = (answer == true_answer)

    k_rms = (k_perturb_l2 / max(perturb_count, 1)) ** 0.5
    v_rms = (v_perturb_l2 / max(perturb_count, 1)) ** 0.5
    k_signal_rms = (k_signal_l2 / max(perturb_count, 1)) ** 0.5
    v_signal_rms = (v_signal_l2 / max(perturb_count, 1)) ** 0.5

    del cache, gen_cache
    gc.collect(); torch.cuda.empty_cache()

    return {
        'correct': correct,
        'answer': answer,
        'text_accuracy': text_accuracy,
        'answer_text': answer_text[:200],
        'k_rms': k_rms,
        'v_rms': v_rms,
        'k_signal_rms': k_signal_rms,
        'v_signal_rms': v_signal_rms,
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


def generate_figures(results, agg, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference data from exp_028 (Llama-3.1-8B direction perturbation)
    llama_ref = {
        'dir_k_early': 0.0, 'dir_v_early': 100.0, 'dir_kv_early': 0.0,
        'dir_k_mid': 6.0, 'dir_v_mid': 100.0, 'dir_kv_mid': 2.0,
        'dir_k_late': 22.0, 'dir_v_late': 88.0, 'dir_kv_late': 14.0,
    }

    # ── Figure 1: Position × Component interaction (THE KEY FIGURE) ──
    positions = ['Early 5%', 'Mid 5%', 'Late 5%']
    k_keys = ['dir_k_early', 'dir_k_mid', 'dir_k_late']
    v_keys = ['dir_v_early', 'dir_v_mid', 'dir_v_late']
    kv_keys = ['dir_kv_early', 'dir_kv_mid', 'dir_kv_late']

    k_accs = [agg[k]['accuracy'] * 100 if k in agg else float('nan') for k in k_keys]
    v_accs = [agg[k]['accuracy'] * 100 if k in agg else float('nan') for k in v_keys]
    kv_accs = [agg[k]['accuracy'] * 100 if k in agg else float('nan') for k in kv_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(positions))
    w = 0.2

    bars_k = ax1.bar(x - w*1.5, k_accs, w, label='Mistral K-only', color='#e74c3c', alpha=0.85)
    bars_v = ax1.bar(x - w*0.5, v_accs, w, label='Mistral V-only', color='#3498db', alpha=0.85)
    bars_kv = ax1.bar(x + w*0.5, kv_accs, w, label='Mistral K+V', color='#9b59b6', alpha=0.85)

    # Llama reference (thin bars)
    llama_k = [llama_ref.get(k, float('nan')) for k in k_keys]
    llama_v = [llama_ref.get(k, float('nan')) for k in v_keys]
    ax1.bar(x + w*1.5, llama_k, w*0.5, label='Llama K (ref)', color='#e74c3c',
            alpha=0.3, edgecolor='#e74c3c', linewidth=1.5, hatch='//')
    ax1.bar(x + w*2.0, llama_v, w*0.5, label='Llama V (ref)', color='#3498db',
            alpha=0.3, edgecolor='#3498db', linewidth=1.5, hatch='//')

    ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax1.set_title('Position × K-V Component: Accuracy\n(Mistral-BASE vs Llama-Instruct reference)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions, fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.2, axis='y')
    for bars, vals in [(bars_k, k_accs), (bars_v, v_accs), (bars_kv, kv_accs)]:
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                        f'{v:.0f}%', ha='center', va='bottom', fontsize=8)

    # Text accuracy panel
    k_txts = [agg[k]['text_accuracy'] * 100 if k in agg else float('nan') for k in k_keys]
    v_txts = [agg[k]['text_accuracy'] * 100 if k in agg else float('nan') for k in v_keys]
    kv_txts = [agg[k]['text_accuracy'] * 100 if k in agg else float('nan') for k in kv_keys]

    bars_k2 = ax2.bar(x - w, k_txts, w, label='K-only', color='#e74c3c', alpha=0.85)
    bars_v2 = ax2.bar(x, v_txts, w, label='V-only', color='#3498db', alpha=0.85)
    bars_kv2 = ax2.bar(x + w, kv_txts, w, label='K+V combined', color='#9b59b6', alpha=0.85)

    ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
    ax2.set_title('Position × K-V Component: Text Quality', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions, fontsize=11)
    ax2.set_ylim(80, 101)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, axis='y')
    for bars, vals in [(bars_k2, k_txts), (bars_v2, v_txts), (bars_kv2, kv_txts)]:
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Exp 038: Mistral-7B-v0.3 Direction Perturbation — K vs V at 3 Position Bands',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'position_kv_interaction.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: K-V gap by position (Mistral vs Llama) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    mistral_gaps = []
    llama_gaps = []
    gap_labels = []
    for i, (pos_label, k_key, v_key) in enumerate([
        ('Early 5%', 'dir_k_early', 'dir_v_early'),
        ('Mid 5%', 'dir_k_mid', 'dir_v_mid'),
        ('Late 5%', 'dir_k_late', 'dir_v_late'),
    ]):
        if k_key in agg and v_key in agg:
            gap_labels.append(pos_label)
            m_gap = (agg[v_key]['accuracy'] - agg[k_key]['accuracy']) * 100
            mistral_gaps.append(m_gap)
            l_gap = llama_ref.get(v_key, 0) - llama_ref.get(k_key, 0)
            llama_gaps.append(l_gap)

    if gap_labels:
        x = np.arange(len(gap_labels))
        w = 0.3
        bars1 = ax.bar(x - w/2, mistral_gaps, w, label='Mistral-BASE (THIS)',
                        color='#8e44ad', alpha=0.85, edgecolor='black')
        bars2 = ax.bar(x + w/2, llama_gaps, w, label='Llama-Instruct (exp_028)',
                        color='#27ae60', alpha=0.6, edgecolor='black')
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_ylabel('V-acc minus K-acc (pp)', fontsize=12)
        ax.set_title('K-V Accuracy Gap: Mistral-BASE vs Llama-Instruct\n'
                     '(Positive = K more critical for answer)', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(gap_labels, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')
        for bars, vals in [(bars1, mistral_gaps), (bars2, llama_gaps)]:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2,
                        b.get_height() + (2 if v >= 0 else -4),
                        f'{v:+.0f}pp', ha='center',
                        va='bottom' if v >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'kv_gap_mistral_vs_llama.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 3: Dissociation map ──
    fig, ax = plt.subplots(figsize=(10, 8))
    markers = {'early': '^', 'mid': 's', 'late': 'o'}
    colors = {'k': '#e74c3c', 'v': '#3498db', 'kv': '#9b59b6'}
    for cond_name, data in agg.items():
        parts = cond_name.split('_')
        if 'early' in cond_name:
            pos = 'early'
        elif 'mid' in cond_name:
            pos = 'mid'
        else:
            pos = 'late'
        comp = parts[1]

        ax.scatter(data['text_drop'] * 100, data['accuracy_drop'] * 100,
                  color=colors.get(comp, 'gray'), marker=markers.get(pos, 'o'),
                  s=150, alpha=0.8, zorder=5)
        label = cond_name.replace('dir_', '')
        ax.annotate(label, (data['text_drop'] * 100, data['accuracy_drop'] * 100),
                   textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Equal effect')
    ax.set_xlabel('Text Drop (%)', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Dissociation Map: Position × Component\n'
                 'Red=K, Blue=V, Purple=KV | ^=Early, □=Mid, ○=Late', fontsize=12)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dissociation_map.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 4: Cross-model K-V gap comparison ──
    # Collect all models' late-5% direction K-V gaps
    fig, ax = plt.subplots(figsize=(10, 6))
    models_data = []

    # Qwen-Base exp_023: dir_k_late=29.2%, dir_v_late=79.2% (approximate from text)
    # Llama exp_028: dir_k_late=22%, dir_v_late=88%
    # Phi exp_034: dir_k_late values
    # Mistral: this experiment

    # Only show what we have confidently
    model_names = ['Qwen-Base\n(exp_023)', 'Llama-Inst\n(exp_028)', 'Mistral-BASE\n(THIS)']
    # Qwen late 5% direction: from exp_023, dir_kv at late 5% = 29.2% with direction
    # Actually from exp_023 notes: magnitude perturbation. Let me use the direction data from earlier.
    # exp_024 Llama direction: K=35.3%, V=86.3% (late 5%)
    # exp_028 Llama: K=22%, V=88% (late 5% replication)
    # For cross-model, use exp_028 Llama and this experiment's Mistral

    if 'dir_k_late' in agg and 'dir_v_late' in agg:
        late_k = agg['dir_k_late']['accuracy'] * 100
        late_v = agg['dir_v_late']['accuracy'] * 100

        bar_data = [
            ('Llama-Inst\n(exp_028)', 22.0, 88.0),
            ('Mistral-BASE\n(THIS exp)', late_k, late_v),
        ]

        x = np.arange(len(bar_data))
        w = 0.3
        k_vals = [d[1] for d in bar_data]
        v_vals = [d[2] for d in bar_data]
        names = [d[0] for d in bar_data]

        ax.bar(x - w/2, k_vals, w, label='K-only direction', color='#e74c3c', alpha=0.85)
        ax.bar(x + w/2, v_vals, w, label='V-only direction', color='#3498db', alpha=0.85)

        for i in range(len(bar_data)):
            gap = v_vals[i] - k_vals[i]
            ax.text(i, max(k_vals[i], v_vals[i]) + 3,
                    f'gap={gap:+.0f}pp', ha='center', fontsize=11, fontweight='bold')

        ax.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax.set_title('Cross-Model K-V Direction Gap at Late 5%\n'
                     'Does K > V (routing > throughput) hold on Mistral?', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylim(0, 115)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'cross_model_kv_gap.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 5: Energy check ──
    fig, ax = plt.subplots(figsize=(10, 6))
    k_sig_vals = [agg[k].get('k_signal_rms', 0) for k in k_keys if k in agg]
    v_sig_vals = [agg[k].get('v_signal_rms', 0) for k in v_keys if k in agg]
    k_pert_vals = [agg[k].get('k_rms', 0) for k in k_keys if k in agg]
    v_pert_vals = [agg[k].get('v_rms', 0) for k in v_keys if k in agg]
    pos_labels = [p for i, p in enumerate(positions) if k_keys[i] in agg]

    if pos_labels:
        x = np.arange(len(pos_labels))
        w = 0.2
        ax.bar(x - w*1.5, k_pert_vals, w, label='K perturb RMS', color='#e74c3c', alpha=0.7)
        ax.bar(x - w*0.5, k_sig_vals, w, label='K signal RMS', color='#ff9999', alpha=0.7)
        ax.bar(x + w*0.5, v_pert_vals, w, label='V perturb RMS', color='#3498db', alpha=0.7)
        ax.bar(x + w*1.5, v_sig_vals, w, label='V signal RMS', color='#99ccff', alpha=0.7)

        ax.set_ylabel('RMS', fontsize=12)
        ax.set_title('Signal and Perturbation Norms by Position\n'
                     '(Direction replacement: perturb RMS ≈ signal RMS × sqrt(2))', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(pos_labels, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
        # Add K/V ratio annotations
        for i in range(len(pos_labels)):
            if v_sig_vals[i] > 0:
                ratio = k_sig_vals[i] / v_sig_vals[i]
                ax.text(i, max(k_sig_vals[i], v_sig_vals[i]) + 2,
                        f'K/V={ratio:.1f}x', ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'energy_by_position.png'),
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
                         getattr(model.config, 'num_attention_heads', '?'))
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
        for cond_name, pos_type, noise_frac, component in CONDITIONS:
            positions = select_positions(reasoning_len, pos_type, noise_frac)
            if not positions:
                continue

            mean_pos = np.mean([p / max(reasoning_len - 1, 1) for p in positions])

            pc = build_prompt_cache(model, prompt_ids, num_layers)
            ev = evaluate_condition(
                model, tokenizer, pc, reasoning_ids_truncated,
                positions, prompt_len, num_layers, true_answer,
                perturb_component=component)

            print(f"    {cond_name}: acc={'Y' if ev['correct'] else 'N'}, "
                  f"text={ev['text_accuracy']:.1%}, n_pos={len(positions)}, "
                  f"k_rms={ev['k_rms']:.1f}, v_rms={ev['v_rms']:.1f}")

            problem_result['evaluations'][cond_name] = {
                'correct': ev['correct'],
                'answer': ev['answer'],
                'text_accuracy': ev['text_accuracy'],
                'mean_pos': float(mean_pos),
                'n_noised': len(positions),
                'k_rms': ev['k_rms'],
                'v_rms': ev['v_rms'],
                'k_signal_rms': ev['k_signal_rms'],
                'v_signal_rms': ev['v_signal_rms'],
            }

        results.append(problem_result)
        del prompt_ids, reasoning_ids, reasoning_ids_truncated, prompt_cache
        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_valid} valid, {len(results)} processed, "
          f"{n_skip_wrong} wrong, {n_skip_trunc} truncation skips")

    summary = {
        'experiment': 'exp_038_mistral_direction_kv',
        'model': MODEL_NAME,
        'num_layers': num_layers,
        'n_kv_heads': n_kv_heads if isinstance(n_kv_heads, int) else str(n_kv_heads),
        'n_attn_heads': n_attn_heads,
        'head_dim': head_dim,
        'architecture': 'MHA' if n_kv_heads == n_attn_heads else 'GQA',
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
        ctxts = [p['clean_text_acc'] for p in results
                 if cond_name in p.get('evaluations', {})]
        k_rms_list = [p['evaluations'][cond_name].get('k_rms', 0)
                      for p in results if cond_name in p.get('evaluations', {})]
        v_rms_list = [p['evaluations'][cond_name].get('v_rms', 0)
                      for p in results if cond_name in p.get('evaluations', {})]
        k_sig_list = [p['evaluations'][cond_name].get('k_signal_rms', 0)
                      for p in results if cond_name in p.get('evaluations', {})]
        v_sig_list = [p['evaluations'][cond_name].get('v_signal_rms', 0)
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
                'k_rms': float(np.mean(k_rms_list)),
                'v_rms': float(np.mean(v_rms_list)),
                'k_signal_rms': float(np.mean(k_sig_list)),
                'v_signal_rms': float(np.mean(v_sig_list)),
            }

    # Print summary table
    print(f"\n{'Condition':<18} {'Acc%':>6} {'Text%':>6} {'AccDrop':>8} {'TxtDrop':>8} "
          f"{'Dissoc':>8} {'K_RMS':>7} {'V_RMS':>7} {'n':>3} {'Wilson 95% CI':>18}")
    print("-" * 105)
    for cond_tuple in CONDITIONS:
        cond_name = cond_tuple[0]
        if cond_name in agg:
            d = agg[cond_name]
            print(f"  {cond_name:<16} {d['accuracy']*100:>5.1f}% {d['text_accuracy']*100:>5.1f}% "
                  f"{d['accuracy_drop']*100:>7.1f}% {d['text_drop']*100:>7.1f}% "
                  f"{d['dissociation']*100:>7.1f}% {d['k_rms']:>6.1f} {d['v_rms']:>6.1f} {d['n']:>3}"
                  f" [{d['wilson_ci_lo']*100:>5.1f}, {d['wilson_ci_hi']*100:>5.1f}]")

    # ── KEY ANALYSIS ──
    print(f"\n{'='*60}")
    print(f"KEY ANALYSIS: K > V UNDER DIRECTION PERTURBATION ON MISTRAL")
    print(f"{'='*60}")

    # Reference: Llama exp_028
    print(f"\n  Reference (Llama-Instruct, exp_028):")
    print(f"    Early 5%: K=0%, V=100%   → gap=+100pp")
    print(f"    Mid 5%:   K=6%, V=100%   → gap=+94pp")
    print(f"    Late 5%:  K=22%, V=88%   → gap=+66pp")

    print(f"\n  Mistral-BASE (THIS EXPERIMENT):")
    for pos_label, k_key, v_key, kv_key in [
        ("Early 5%", "dir_k_early", "dir_v_early", "dir_kv_early"),
        ("Mid 5%",   "dir_k_mid",   "dir_v_mid",   "dir_kv_mid"),
        ("Late 5%",  "dir_k_late",  "dir_v_late",  "dir_kv_late"),
    ]:
        if k_key in agg and v_key in agg:
            k_acc = agg[k_key]['accuracy'] * 100
            v_acc = agg[v_key]['accuracy'] * 100
            kv_acc = agg[kv_key]['accuracy'] * 100 if kv_key in agg else float('nan')
            k_txt = agg[k_key]['text_accuracy'] * 100
            v_txt = agg[v_key]['text_accuracy'] * 100
            gap = v_acc - k_acc
            k_drop = agg[k_key]['accuracy_drop'] * 100
            v_drop = agg[v_key]['accuracy_drop'] * 100
            ratio = k_drop / max(v_drop, 0.1)
            print(f"\n    {pos_label}:")
            print(f"      K-only:  acc={k_acc:.1f}% [{agg[k_key]['wilson_ci_lo']*100:.1f}, {agg[k_key]['wilson_ci_hi']*100:.1f}], text={k_txt:.1f}%")
            print(f"      V-only:  acc={v_acc:.1f}% [{agg[v_key]['wilson_ci_lo']*100:.1f}, {agg[v_key]['wilson_ci_hi']*100:.1f}], text={v_txt:.1f}%")
            if not np.isnan(kv_acc):
                print(f"      KV-comb: acc={kv_acc:.1f}%")
            print(f"      V-K gap: {gap:+.1f}pp  {'(K > V CONFIRMED)' if gap > 10 else '(K ~ V)' if abs(gap) <= 10 else '(V > K)'}")

            # Energy check
            k_sig = agg[k_key].get('k_signal_rms', 0)
            v_sig = agg[v_key].get('v_signal_rms', 0)
            if k_sig > 0 and v_sig > 0:
                print(f"      K-signal-RMS: {k_sig:.1f}, V-signal-RMS: {v_sig:.1f}, K/V ratio: {k_sig/v_sig:.1f}x")

    # VERDICT
    print(f"\n{'='*60}")
    print(f"== K > V UNIVERSALITY VERDICT ==")
    if 'dir_k_late' in agg and 'dir_v_late' in agg:
        k_late = agg['dir_k_late']['accuracy'] * 100
        v_late = agg['dir_v_late']['accuracy'] * 100
        gap_late = v_late - k_late
        if gap_late > 20:
            print(f"  K > V CONFIRMED on Mistral under direction perturbation (gap={gap_late:+.0f}pp)")
            print(f"  -> K ≈ V from exp_037 was MAGNITUDE-SPECIFIC")
            print(f"  -> Routing > throughput hierarchy IS universal across all 5 model variants")
        elif gap_late > 5:
            print(f"  K > V WEAKLY confirmed on Mistral (gap={gap_late:+.0f}pp)")
            print(f"  -> Weaker than Llama (+66pp) but still present")
        elif abs(gap_late) <= 5:
            print(f"  K ≈ V on Mistral EVEN UNDER DIRECTION perturbation (gap={gap_late:+.0f}pp)")
            print(f"  -> K > V is genuinely NON-UNIVERSAL")
            print(f"  -> Mistral architecture/training creates balanced K-V usage")
        else:
            print(f"  V > K on Mistral (gap={gap_late:+.0f}pp) — REVERSAL")
            print(f"  -> Fundamentally different K-V dynamics on Mistral")

    # V-only immunity check at early+mid
    print(f"\n-- V-ONLY DIRECTION IMMUNITY CHECK --")
    for key in ['dir_v_early', 'dir_v_mid']:
        if key in agg:
            d = agg[key]
            n_corr = d['n_correct']
            n_tot = d['n']
            print(f"  {key}: {n_corr}/{n_tot} correct ({d['accuracy']*100:.1f}%)")

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
