#!/usr/bin/env python3
"""
Experiment 027: SNR-Controlled K vs V Additive Noise — Testing the Energy Confound

The K-V direction dissociation (K/V accuracy ratio = 3.6-3.7x across both models)
is the central mechanistic finding. However, direction replacement creates
perturbation proportional to the original norm: K-dir RMS=95 vs V-dir RMS=15
(6.4x gap in exp_024). The 3.6x accuracy ratio might be an artifact of this
unequal perturbation energy.

This experiment tests: at MATCHED relative perturbation (same SNR per head),
is K still more destructive than V?

Method: Additive Gaussian noise at controlled SNR, applied to K-only or V-only
at late 5% positions. Also includes direction replacement for reference.

SNR formula: SNR_dB = 10 * log10(||signal||^2 / E[||noise||^2])
  noise_std_per_element = ||signal_h|| / (sqrt(head_dim) * 10^(SNR_dB/20))

At matched SNR, K and V experience the SAME relative disruption (noise/signal ratio),
controlling for the fact that K-norms >> V-norms.
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
NOISE_FRAC = 0.05  # 5% of positions
SEED = 42
HASH4_TOKEN = 827  # "####" in Llama tokenizer
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_027")

# Same 8-shot exemplars as exp_024/025 (Llama format)
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
# SNR levels in dB
SNR_LEVELS = [10, 5, 0, -3, -6]

# (name, component, perturbation_type, snr_db_or_none)
CONDITIONS = []
for snr in SNR_LEVELS:
    snr_label = f"snr{snr}" if snr >= 0 else f"snrn{abs(snr)}"
    CONDITIONS.append((f"k_{snr_label}", "k", "snr", snr))
    CONDITIONS.append((f"v_{snr_label}", "v", "snr", snr))
# Direction replacement (replication from exp_024)
CONDITIONS.append(("dir_k", "k", "direction", None))
CONDITIONS.append(("dir_v", "v", "direction", None))


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


def find_truncation_point(reasoning_ids, tokenizer):
    """Find where to truncate reasoning before the answer marker."""
    ids_list = reasoning_ids[0].tolist()

    for i, tid in enumerate(ids_list):
        if tid == HASH4_TOKEN:
            if i >= 10:
                return i
            break

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


def select_late_positions(reasoning_len, noise_frac=NOISE_FRAC):
    """Select late positions (last noise_frac of reasoning tokens)."""
    n = max(1, int(reasoning_len * noise_frac))
    return list(range(max(0, reasoning_len - n), reasoning_len))


def perturb_snr(tensor, snr_db):
    """Add noise at target SNR (per head).
    tensor shape: [1, n_heads, 1, head_dim]

    SNR_dB = 10 * log10(||signal||^2 / E[||noise||^2])
    noise_std = ||signal_h|| / (sqrt(head_dim) * 10^(SNR_dB/20))
    """
    head_dim = tensor.shape[-1]
    signal_norm = tensor.norm(dim=-1, keepdim=True)  # [1, n_heads, 1, 1]
    noise_std = signal_norm / (head_dim**0.5 * 10**(snr_db / 20.0))
    noise = torch.randn_like(tensor) * noise_std
    return tensor + noise


def perturb_direction(tensor):
    """Replace direction with random unit vector, preserving norm.
    tensor shape: [1, n_heads, 1, head_dim]
    """
    norms = tensor.norm(dim=-1, keepdim=True)
    random_dir = torch.randn_like(tensor)
    random_dir = random_dir / (random_dir.norm(dim=-1, keepdim=True) + 1e-8)
    return random_dir * norms


@torch.no_grad()
def evaluate_condition(model, tokenizer, prompt_cache, reasoning_tokens,
                       positions_to_noise, prompt_len, num_layers, true_answer,
                       component='k', perturb_type='snr', snr_db=0):
    """
    Step through reasoning tokens, apply perturbation at selected positions.
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

    # Track perturbation statistics
    k_perturb_l2_total = 0.0
    v_perturb_l2_total = 0.0
    k_signal_l2_total = 0.0
    v_signal_l2_total = 0.0
    perturb_count = 0

    for i in range(reasoning_len):
        token = reasoning_tokens[:, i:i+1]
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

        if i in noise_set:
            for l in range(num_layers):
                pos = prompt_len + i

                if component in ('k', 'kv'):
                    k_slice = cache.layers[l].keys[:, :, pos:pos+1, :]
                    k_orig = k_slice.clone()
                    if perturb_type == 'snr':
                        k_new = perturb_snr(k_slice, snr_db)
                    else:  # direction
                        k_new = perturb_direction(k_slice)
                    cache.layers[l].keys[:, :, pos:pos+1, :] = k_new
                    k_perturb_l2_total += (k_new - k_orig).float().norm().item() ** 2
                    k_signal_l2_total += k_orig.float().norm().item() ** 2

                if component in ('v', 'kv'):
                    v_slice = cache.layers[l].values[:, :, pos:pos+1, :]
                    v_orig = v_slice.clone()
                    if perturb_type == 'snr':
                        v_new = perturb_snr(v_slice, snr_db)
                    else:  # direction
                        v_new = perturb_direction(v_slice)
                    cache.layers[l].values[:, :, pos:pos+1, :] = v_new
                    v_perturb_l2_total += (v_new - v_orig).float().norm().item() ** 2
                    v_signal_l2_total += v_orig.float().norm().item() ** 2

                perturb_count += 1

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

    # Compute RMS and actual SNR
    k_rms = (k_perturb_l2_total / max(perturb_count, 1)) ** 0.5
    v_rms = (v_perturb_l2_total / max(perturb_count, 1)) ** 0.5
    k_signal_rms = (k_signal_l2_total / max(perturb_count, 1)) ** 0.5
    v_signal_rms = (v_signal_l2_total / max(perturb_count, 1)) ** 0.5

    # Actual SNR achieved
    k_actual_snr = 10 * np.log10(k_signal_l2_total / max(k_perturb_l2_total, 1e-10)) if k_perturb_l2_total > 0 else float('inf')
    v_actual_snr = 10 * np.log10(v_signal_l2_total / max(v_perturb_l2_total, 1e-10)) if v_perturb_l2_total > 0 else float('inf')

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
        'k_actual_snr_db': float(k_actual_snr),
        'v_actual_snr_db': float(v_actual_snr),
    }


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_figures(agg, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Figure 1: K vs V at matched SNR (THE KEY COMPARISON) ──
    k_snr_conds = [(f"k_snr{s}" if s >= 0 else f"k_snrn{abs(s)}", s) for s in SNR_LEVELS]
    v_snr_conds = [(f"v_snr{s}" if s >= 0 else f"v_snrn{abs(s)}", s) for s in SNR_LEVELS]

    k_snrs = [s for c, s in k_snr_conds if c in agg]
    k_accs = [agg[c]['accuracy'] * 100 for c, _ in k_snr_conds if c in agg]
    v_snrs = [s for c, s in v_snr_conds if c in agg]
    v_accs = [agg[c]['accuracy'] * 100 for c, _ in v_snr_conds if c in agg]

    if k_snrs and v_snrs:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(k_snrs, k_accs, 'o-', color='#e74c3c', linewidth=2.5,
                markersize=12, label='K-only noise', zorder=5)
        ax.plot(v_snrs, v_accs, 's-', color='#2ecc71', linewidth=2.5,
                markersize=12, label='V-only noise', zorder=5)
        for s, a in zip(k_snrs, k_accs):
            ax.annotate(f'{a:.0f}%', (s, a), textcoords="offset points",
                       xytext=(0, 14), ha='center', fontsize=11, color='#e74c3c',
                       fontweight='bold')
        for s, a in zip(v_snrs, v_accs):
            ax.annotate(f'{a:.0f}%', (s, a), textcoords="offset points",
                       xytext=(0, -18), ha='center', fontsize=11, color='#2ecc71',
                       fontweight='bold')
        # Add direction replacement reference
        if 'dir_k' in agg and 'dir_v' in agg:
            ax.axhline(y=agg['dir_k']['accuracy'] * 100, color='#e74c3c',
                       linestyle=':', alpha=0.5, label=f"K dir-replace ({agg['dir_k']['accuracy']*100:.0f}%)")
            ax.axhline(y=agg['dir_v']['accuracy'] * 100, color='#2ecc71',
                       linestyle=':', alpha=0.5, label=f"V dir-replace ({agg['dir_v']['accuracy']*100:.0f}%)")

        ax.set_xlabel('SNR (dB) — lower = more noise', fontsize=13)
        ax.set_ylabel('Answer Accuracy (%)', fontsize=13)
        ax.set_title('K vs V at Matched SNR (Late 5%, Llama-3.1-8B)\n'
                     'Does K remain more sensitive when controlling for perturbation energy?',
                     fontsize=14)
        ax.set_ylim(-5, 110)
        ax.invert_xaxis()  # Lower SNR = more noise → right side
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'kv_snr_matched_accuracy.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 2: Actual RMS perturbation at each SNR level ──
    k_rms_vals = [agg[c]['mean_k_rms'] for c, _ in k_snr_conds if c in agg]
    v_rms_vals = [agg[c]['mean_v_rms'] for c, _ in v_snr_conds if c in agg]

    if k_rms_vals and v_rms_vals:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # RMS perturbation
        ax1.plot(k_snrs, k_rms_vals, 'o-', color='#e74c3c', linewidth=2.5,
                 markersize=10, label='K noise RMS')
        ax1.plot(v_snrs, v_rms_vals, 's-', color='#2ecc71', linewidth=2.5,
                 markersize=10, label='V noise RMS')
        if 'dir_k' in agg and 'dir_v' in agg:
            ax1.axhline(y=agg['dir_k']['mean_k_rms'], color='#e74c3c',
                        linestyle=':', alpha=0.5, label=f"K dir-replace RMS={agg['dir_k']['mean_k_rms']:.1f}")
            ax1.axhline(y=agg['dir_v']['mean_v_rms'], color='#2ecc71',
                        linestyle=':', alpha=0.5, label=f"V dir-replace RMS={agg['dir_v']['mean_v_rms']:.1f}")
        ax1.set_xlabel('SNR (dB)', fontsize=12)
        ax1.set_ylabel('RMS Perturbation', fontsize=12)
        ax1.set_title('Absolute Perturbation Energy', fontsize=13)
        ax1.invert_xaxis()
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # K/V RMS ratio
        if len(k_rms_vals) == len(v_rms_vals):
            ratios = [k / max(v, 1e-6) for k, v in zip(k_rms_vals, v_rms_vals)]
            ax2.bar(range(len(SNR_LEVELS)), ratios, color='#9b59b6', alpha=0.7)
            ax2.set_xticks(range(len(SNR_LEVELS)))
            ax2.set_xticklabels([str(s) for s in SNR_LEVELS])
            ax2.set_xlabel('SNR (dB)', fontsize=12)
            ax2.set_ylabel('K-RMS / V-RMS Ratio', fontsize=12)
            ax2.set_title('K/V Perturbation Energy Ratio at Matched SNR', fontsize=13)
            ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal energy')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Perturbation Energy: Same SNR → Different Absolute RMS', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'perturbation_energy_analysis.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 3: K/V accuracy gap vs SNR ──
    if k_snrs and v_snrs and len(k_accs) == len(v_accs):
        gaps = [k - v for k, v in zip(k_accs, v_accs)]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c' if g < 0 else '#2ecc71' for g in gaps]
        bars = ax.bar(range(len(SNR_LEVELS)), gaps, color=colors, alpha=0.8)
        ax.set_xticks(range(len(SNR_LEVELS)))
        ax.set_xticklabels([f'{s} dB' for s in SNR_LEVELS])
        ax.set_xlabel('SNR (dB)', fontsize=13)
        ax.set_ylabel('K_accuracy - V_accuracy (pp)', fontsize=13)
        ax.set_title('K-V Accuracy Gap at Matched SNR\n'
                     'Red bars: K is more sensitive (routing > throughput)',
                     fontsize=14)
        ax.axhline(y=0, color='gray', linewidth=1)
        for b, g in zip(bars, gaps):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + (2 if g >= 0 else -4),
                    f'{g:+.1f}pp', ha='center', fontsize=11, fontweight='bold')
        # Add direction replacement gap as reference
        if 'dir_k' in agg and 'dir_v' in agg:
            dir_gap = (agg['dir_k']['accuracy'] - agg['dir_v']['accuracy']) * 100
            ax.axhline(y=dir_gap, color='purple', linestyle='--', alpha=0.5,
                       label=f'Direction replacement gap: {dir_gap:+.1f}pp')
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'kv_accuracy_gap.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 4: Text accuracy (both should be high) ──
    k_txts = [agg[c]['text_accuracy'] * 100 for c, _ in k_snr_conds if c in agg]
    v_txts = [agg[c]['text_accuracy'] * 100 for c, _ in v_snr_conds if c in agg]
    if k_txts and v_txts:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_snrs, k_txts, 'o-', color='#e74c3c', linewidth=2,
                markersize=8, label='K-noise text acc')
        ax.plot(v_snrs, v_txts, 's-', color='#2ecc71', linewidth=2,
                markersize=8, label='V-noise text acc')
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
        ax.set_title('Text Quality at Each SNR Level', fontsize=13)
        ax.set_ylim(80, 101)
        ax.invert_xaxis()
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'text_accuracy_vs_snr.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 5: All conditions bar chart ──
    all_keys = [c[0] for c in CONDITIONS if c[0] in agg]
    if all_keys:
        all_accs = [agg[c]['accuracy'] * 100 for c in all_keys]
        all_txts = [agg[c]['text_accuracy'] * 100 for c in all_keys]
        colors = ['#e74c3c' if c.startswith('k_') or c == 'dir_k' else '#2ecc71'
                  for c in all_keys]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        x = np.arange(len(all_keys))

        bars1 = ax1.bar(x, all_accs, 0.6, color=colors, alpha=0.85)
        ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax1.set_title('All Conditions: Accuracy', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_keys, rotation=45, ha='right', fontsize=8)
        ax1.set_ylim(0, 110)
        ax1.grid(True, alpha=0.2, axis='y')
        for b, v in zip(bars1, all_accs):
            ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f'{v:.0f}%', ha='center', va='bottom', fontsize=8)

        bars2 = ax2.bar(x, all_txts, 0.6, color=colors, alpha=0.85)
        ax2.set_ylabel('Text Accuracy (%)', fontsize=12)
        ax2.set_title('All Conditions: Text Quality', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_keys, rotation=45, ha='right', fontsize=8)
        ax2.set_ylim(80, 101)
        ax2.grid(True, alpha=0.2, axis='y')
        for b, v in zip(bars2, all_txts):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.85, label='K perturbation'),
            Patch(facecolor='#2ecc71', alpha=0.85, label='V perturbation'),
        ]
        ax1.legend(handles=legend_elements, fontsize=9)
        plt.suptitle('Exp 027: SNR-Controlled K vs V (Late 5%, Llama)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'all_conditions_bar.png'),
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
            [], prompt_len, num_layers, true_answer,
            component='k', perturb_type='snr', snr_db=100)  # effectively clean
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

        # Select late 5% positions
        positions = select_late_positions(reasoning_len)

        # Run all conditions
        for cond_name, component, perturb_type, snr_db in CONDITIONS:
            pc = build_prompt_cache(model, prompt_ids, num_layers)
            ev = evaluate_condition(
                model, tokenizer, pc, reasoning_ids_truncated,
                positions, prompt_len, num_layers, true_answer,
                component=component, perturb_type=perturb_type,
                snr_db=snr_db if snr_db is not None else 0)

            k_rms_str = f"k_rms={ev['k_rms']:.1f}" if ev['k_rms'] > 0 else ""
            v_rms_str = f"v_rms={ev['v_rms']:.1f}" if ev['v_rms'] > 0 else ""
            rms_str = f"{k_rms_str} {v_rms_str}".strip()

            print(f"    {cond_name}: acc={'Y' if ev['correct'] else 'N'}, "
                  f"text={ev['text_accuracy']:.1%}, {rms_str}")

            problem_result['evaluations'][cond_name] = {
                'correct': ev['correct'],
                'answer': ev['answer'],
                'text_accuracy': ev['text_accuracy'],
                'n_noised': len(positions),
                'k_rms': ev['k_rms'],
                'v_rms': ev['v_rms'],
                'k_signal_rms': ev['k_signal_rms'],
                'v_signal_rms': ev['v_signal_rms'],
                'k_actual_snr_db': ev['k_actual_snr_db'],
                'v_actual_snr_db': ev['v_actual_snr_db'],
            }

        results.append(problem_result)
        del prompt_ids, reasoning_ids, reasoning_ids_truncated, prompt_cache
        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_valid} valid, {len(results)} processed, "
          f"{n_skip_wrong} wrong, {n_skip_trunc} truncation skips")

    summary = {
        'experiment': 'exp_027_snr_controlled_kv',
        'model': MODEL_NAME,
        'n_valid': n_valid,
        'n_processed': len(results),
        'n_skip_wrong': n_skip_wrong,
        'n_skip_trunc': n_skip_trunc,
        'conditions': [c[0] for c in CONDITIONS],
        'snr_levels_db': SNR_LEVELS,
    }

    agg = {}
    for cond_tuple in CONDITIONS:
        cond_name = cond_tuple[0]
        accs = [1 if p['evaluations'][cond_name]['correct'] else 0
                for p in results if cond_name in p.get('evaluations', {})]
        txts = [p['evaluations'][cond_name]['text_accuracy']
                for p in results if cond_name in p.get('evaluations', {})]
        ctxts = [p['clean_text_acc'] for p in results if cond_name in p.get('evaluations', {})]
        k_rms_list = [p['evaluations'][cond_name]['k_rms']
                      for p in results if cond_name in p.get('evaluations', {})]
        v_rms_list = [p['evaluations'][cond_name]['v_rms']
                      for p in results if cond_name in p.get('evaluations', {})]
        k_signal_rms_list = [p['evaluations'][cond_name]['k_signal_rms']
                             for p in results if cond_name in p.get('evaluations', {})]
        v_signal_rms_list = [p['evaluations'][cond_name]['v_signal_rms']
                             for p in results if cond_name in p.get('evaluations', {})]

        if accs:
            a_mean = np.mean(accs)
            t_mean = np.mean(txts)
            ct_mean = np.mean(ctxts)
            a_drop = 1.0 - a_mean
            t_drop = ct_mean - t_mean
            dissoc = a_drop - t_drop
            n_correct = sum(accs)
            agg[cond_name] = {
                'n': len(accs),
                'n_correct': int(n_correct),
                'accuracy': float(a_mean),
                'text_accuracy': float(t_mean),
                'clean_text_accuracy': float(ct_mean),
                'accuracy_drop': float(a_drop),
                'text_drop': float(t_drop),
                'dissociation': float(dissoc),
                'mean_k_rms': float(np.mean(k_rms_list)),
                'mean_v_rms': float(np.mean(v_rms_list)),
                'mean_k_signal_rms': float(np.mean(k_signal_rms_list)),
                'mean_v_signal_rms': float(np.mean(v_signal_rms_list)),
            }

    # Print summary table
    print(f"\n{'Cond':<12} {'Type':>5} {'SNR':>5} {'Acc%':>6} {'Text%':>6} "
          f"{'AccDrp':>7} {'TxtDrp':>7} {'Dissoc':>7} {'K_RMS':>7} {'V_RMS':>7} {'n':>3}")
    print("-" * 90)
    for cond_name, component, ptype, snr in CONDITIONS:
        if cond_name in agg:
            d = agg[cond_name]
            snr_str = f"{snr}" if snr is not None else "N/A"
            print(f"  {cond_name:<10} {component:>5} {snr_str:>5} "
                  f"{d['accuracy']*100:>5.1f}% {d['text_accuracy']*100:>5.1f}% "
                  f"{d['accuracy_drop']*100:>6.1f}% {d['text_drop']*100:>6.1f}% "
                  f"{d['dissociation']*100:>6.1f}% {d['mean_k_rms']:>6.1f} "
                  f"{d['mean_v_rms']:>6.1f} {d['n']:>3}")

    # ── Critical comparison: K vs V at matched SNR ──
    print(f"\n{'='*60}")
    print("CRITICAL COMPARISON: K vs V at matched SNR")
    print("-" * 60)
    for snr in SNR_LEVELS:
        snr_label = f"snr{snr}" if snr >= 0 else f"snrn{abs(snr)}"
        k_key = f"k_{snr_label}"
        v_key = f"v_{snr_label}"
        if k_key in agg and v_key in agg:
            k_acc = agg[k_key]['accuracy'] * 100
            v_acc = agg[v_key]['accuracy'] * 100
            gap = k_acc - v_acc
            k_rms = agg[k_key]['mean_k_rms']
            v_rms = agg[v_key]['mean_v_rms']
            rms_ratio = k_rms / max(v_rms, 1e-6)
            print(f"  SNR={snr:>3}dB: K_acc={k_acc:>5.1f}%  V_acc={v_acc:>5.1f}%  "
                  f"Gap={gap:>+6.1f}pp  K_RMS={k_rms:.1f}  V_RMS={v_rms:.1f}  "
                  f"RMS_ratio={rms_ratio:.1f}x")

    # Direction replacement comparison
    if 'dir_k' in agg and 'dir_v' in agg:
        k_acc = agg['dir_k']['accuracy'] * 100
        v_acc = agg['dir_v']['accuracy'] * 100
        gap = k_acc - v_acc
        k_rms = agg['dir_k']['mean_k_rms']
        v_rms = agg['dir_v']['mean_v_rms']
        rms_ratio = k_rms / max(v_rms, 1e-6)
        print(f"  Dir-replace: K_acc={k_acc:>5.1f}%  V_acc={v_acc:>5.1f}%  "
              f"Gap={gap:>+6.1f}pp  K_RMS={k_rms:.1f}  V_RMS={v_rms:.1f}  "
              f"RMS_ratio={rms_ratio:.1f}x")

    print(f"\nVERDICT:")
    # Compute average K-V gap across SNR levels
    gaps = []
    for snr in SNR_LEVELS:
        snr_label = f"snr{snr}" if snr >= 0 else f"snrn{abs(snr)}"
        k_key = f"k_{snr_label}"
        v_key = f"v_{snr_label}"
        if k_key in agg and v_key in agg:
            gaps.append(agg[k_key]['accuracy'] * 100 - agg[v_key]['accuracy'] * 100)
    if gaps:
        mean_gap = np.mean(gaps)
        if mean_gap < -10:
            print(f"  K is MORE sensitive than V at matched SNR (avg gap={mean_gap:+.1f}pp)")
            print(f"  → CONFIRMS: K genuinely carries more answer info per unit representation")
        elif mean_gap > 10:
            print(f"  V is MORE sensitive than K at matched SNR (avg gap={mean_gap:+.1f}pp)")
            print(f"  → CHALLENGES: K-V result was an energy artifact!")
        else:
            print(f"  K and V are SIMILAR at matched SNR (avg gap={mean_gap:+.1f}pp)")
            print(f"  → PARTIALLY CHALLENGES: Some of the 3.6x ratio was energy artifact")

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
