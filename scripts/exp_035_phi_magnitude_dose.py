#!/usr/bin/env python3
"""
Experiment 035: Magnitude Dose-Response on Phi-3.5-mini-instruct (MHA)

Tests K-magnitude and V-magnitude perturbation at escalating doses on Phi
to determine:
  1. Does Phi show digital (Qwen-like cliffs) or analog (Llama-like gradual) encoding?
  2. Does K > V hold under magnitude perturbation on MHA architecture?
  3. Is there superadditive K-V interaction on a 3rd model family?

All conditions at late 5% positions (where cleanest K-V dissociation exists).
Model: microsoft/Phi-3.5-mini-instruct (3.82B, MHA with 32 KV heads)
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

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
NUM_PROBLEMS = 200
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 2048
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_035")

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

# Conditions: (name, perturb_component, sigma)
# All at late 5% positions
CONDITIONS = [
    # K-magnitude dose escalation (spans Qwen cliff range σ=0.3-1.0 and beyond)
    ("mag_k_05",  "k",  0.5),
    ("mag_k_10",  "k",  1.0),
    ("mag_k_20",  "k",  2.0),
    ("mag_k_50",  "k",  5.0),
    # V-magnitude dose escalation (spans Qwen V cliff range σ=3-5 and beyond)
    ("mag_v_10",  "v",  1.0),
    ("mag_v_20",  "v",  2.0),
    ("mag_v_30",  "v",  3.0),
    ("mag_v_50",  "v",  5.0),
    ("mag_v_100", "v",  10.0),
    # K+V magnitude combined (superadditivity test)
    ("mag_kv_10", "kv", 1.0),
    ("mag_kv_20", "kv", 2.0),
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
    gc.collect()
    torch.cuda.empty_cache()
    return text, prompt_ids, reasoning_ids


@torch.no_grad()
def build_prompt_cache(model, prompt_ids, num_layers):
    """Build clean KV cache for prompt tokens only."""
    outputs = model(input_ids=prompt_ids, use_cache=True)
    prompt_cache = outputs.past_key_values
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    return prompt_cache


def select_late_positions(reasoning_len, noise_frac=0.05):
    """Select late positions (last noise_frac of reasoning tokens)."""
    n = max(1, int(reasoning_len * noise_frac))
    return list(range(max(0, reasoning_len - n), reasoning_len))


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
                       perturb_component='kv', magnitude_sigma=1.0):
    """
    Step through reasoning tokens, apply magnitude perturbation at selected positions.
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
    scale_factors_k = []
    scale_factors_v = []

    for i in range(reasoning_len):
        token = reasoning_tokens[:, i:i+1]
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

        if i in noise_set:
            for l in range(num_layers):
                pos = prompt_len + i
                k_orig = cache.layers[l].keys[:, :, pos:pos+1, :].clone()
                v_orig = cache.layers[l].values[:, :, pos:pos+1, :].clone()

                k_signal_l2 += k_orig.float().norm().item() ** 2
                v_signal_l2 += v_orig.float().norm().item() ** 2

                if perturb_component in ('kv', 'k'):
                    k_new = perturb_magnitude(
                        cache.layers[l].keys[:, :, pos:pos+1, :],
                        sigma=magnitude_sigma)
                    k_orig_norms = k_orig.norm(dim=-1)
                    k_new_norms = k_new.norm(dim=-1)
                    effective_scale = (k_new_norms / (k_orig_norms + 1e-8)).float()
                    scale_factors_k.extend(effective_scale.flatten().tolist())
                    cache.layers[l].keys[:, :, pos:pos+1, :] = k_new
                    k_perturb_l2 += (k_new - k_orig).float().norm().item() ** 2

                if perturb_component in ('kv', 'v'):
                    v_new = perturb_magnitude(
                        cache.layers[l].values[:, :, pos:pos+1, :],
                        sigma=magnitude_sigma)
                    v_orig_norms = v_orig.norm(dim=-1)
                    v_new_norms = v_new.norm(dim=-1)
                    effective_scale = (v_new_norms / (v_orig_norms + 1e-8)).float()
                    scale_factors_v.extend(effective_scale.flatten().tolist())
                    cache.layers[l].values[:, :, pos:pos+1, :] = v_new
                    v_perturb_l2 += (v_new - v_orig).float().norm().item() ** 2

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

    k_rms = (k_perturb_l2 / max(perturb_count, 1)) ** 0.5
    v_rms = (v_perturb_l2 / max(perturb_count, 1)) ** 0.5
    k_signal_rms = (k_signal_l2 / max(perturb_count, 1)) ** 0.5
    v_signal_rms = (v_signal_l2 / max(perturb_count, 1)) ** 0.5

    # Scale factor stats
    sf_k_stats = {}
    if scale_factors_k:
        sf_arr = np.array(scale_factors_k)
        sf_k_stats = {
            'mean': float(np.mean(sf_arr)),
            'std': float(np.std(sf_arr)),
            'min': float(np.min(sf_arr)),
            'max': float(np.max(sf_arr)),
        }
    sf_v_stats = {}
    if scale_factors_v:
        sf_arr = np.array(scale_factors_v)
        sf_v_stats = {
            'mean': float(np.mean(sf_arr)),
            'std': float(np.std(sf_arr)),
            'min': float(np.min(sf_arr)),
            'max': float(np.max(sf_arr)),
        }

    del cache, gen_cache
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'correct': correct,
        'answer': answer,
        'text_accuracy': text_accuracy,
        'answer_text': answer_text[:200],
        'k_rms': k_rms,
        'v_rms': v_rms,
        'k_signal_rms': k_signal_rms,
        'v_signal_rms': v_signal_rms,
        'scale_factors_k': sf_k_stats,
        'scale_factors_v': sf_v_stats,
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

    # Figure 1: K-magnitude dose-response curve
    k_conds = [("mag_k_05", 0.5), ("mag_k_10", 1.0), ("mag_k_20", 2.0), ("mag_k_50", 5.0)]
    k_sigmas = [s for c, s in k_conds if c in agg]
    k_accs = [agg[c]['accuracy'] * 100 for c, _ in k_conds if c in agg]
    k_txts = [agg[c]['text_accuracy'] * 100 for c, _ in k_conds if c in agg]

    if k_sigmas:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_sigmas, k_accs, 'o-', color='#e74c3c', linewidth=2.5,
                markersize=10, label='Phi K-mag Accuracy', zorder=5)
        ax.plot(k_sigmas, k_txts, 's--', color='#3498db', linewidth=2,
                markersize=8, label='Phi K-mag Text', zorder=4)
        for s, a, t in zip(k_sigmas, k_accs, k_txts):
            ax.annotate(f'{a:.0f}%', (s, a), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=10, color='#e74c3c',
                       fontweight='bold')
        # Reference lines from Qwen and Llama
        qwen_k = {0.5: 45.8, 1.0: 10.0}  # approx from exp_023/026
        llama_k = {1.0: 96.0, 2.0: 94.0, 3.0: 80.0, 5.0: 72.0}
        qwen_pts = [(s, v) for s, v in qwen_k.items()]
        llama_pts = [(s, v) for s, v in llama_k.items()]
        if qwen_pts:
            ax.plot([p[0] for p in qwen_pts], [p[1] for p in qwen_pts],
                    'v:', color='#e67e22', alpha=0.6, markersize=8,
                    label='Qwen K-mag (ref)')
        if llama_pts:
            ax.plot([p[0] for p in llama_pts], [p[1] for p in llama_pts],
                    '^:', color='#27ae60', alpha=0.6, markersize=8,
                    label='Llama K-mag (ref)')
        ax.set_xlabel('Magnitude σ', fontsize=13)
        ax.set_ylabel('Accuracy (%)', fontsize=13)
        ax.set_title('Phi K-Magnitude Dose-Response (Late 5%)\n'
                     'Digital (cliff) or Analog (gradual)?', fontsize=14)
        ax.set_ylim(-5, 110)
        ax.set_xlim(0, 5.5)
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'k_magnitude_dose_response.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 2: V-magnitude dose-response curve
    v_conds = [("mag_v_10", 1.0), ("mag_v_20", 2.0), ("mag_v_30", 3.0), ("mag_v_50", 5.0), ("mag_v_100", 10.0)]
    v_sigmas = [s for c, s in v_conds if c in agg]
    v_accs = [agg[c]['accuracy'] * 100 for c, _ in v_conds if c in agg]
    v_txts = [agg[c]['text_accuracy'] * 100 for c, _ in v_conds if c in agg]

    if v_sigmas:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(v_sigmas, v_accs, 'o-', color='#e74c3c', linewidth=2.5,
                markersize=10, label='Phi V-mag Accuracy', zorder=5)
        ax.plot(v_sigmas, v_txts, 's--', color='#3498db', linewidth=2,
                markersize=8, label='Phi V-mag Text', zorder=4)
        for s, a, t in zip(v_sigmas, v_accs, v_txts):
            ax.annotate(f'{a:.0f}%', (s, a), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=10, color='#e74c3c',
                       fontweight='bold')
        # Reference: Qwen V cliff at σ=3-5, Llama V gradual
        qwen_v = {1.0: 100.0, 3.0: 100.0, 5.0: 50.0, 10.0: 20.0}
        llama_v = {1.0: 100.0, 2.0: 98.0, 5.0: 89.0, 10.0: 65.0}
        ax.plot(list(qwen_v.keys()), list(qwen_v.values()),
                'v:', color='#e67e22', alpha=0.6, markersize=8,
                label='Qwen V-mag (ref)')
        ax.plot(list(llama_v.keys()), list(llama_v.values()),
                '^:', color='#27ae60', alpha=0.6, markersize=8,
                label='Llama V-mag (ref)')
        ax.set_xlabel('Magnitude σ', fontsize=13)
        ax.set_ylabel('Accuracy (%)', fontsize=13)
        ax.set_title('Phi V-Magnitude Dose-Response (Late 5%)\n'
                     'Does V carry more info on MHA?', fontsize=14)
        ax.set_ylim(-5, 110)
        ax.set_xlim(0, 11)
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'v_magnitude_dose_response.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 3: Combined K vs V dose-response
    if k_sigmas and v_sigmas:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(k_sigmas, k_accs, 'o-', color='#e74c3c', linewidth=2.5,
                markersize=10, label='K-mag Accuracy', zorder=5)
        ax.plot(v_sigmas, v_accs, 's-', color='#2ecc71', linewidth=2.5,
                markersize=10, label='V-mag Accuracy', zorder=5)
        # KV combined
        kv_conds = [("mag_kv_10", 1.0), ("mag_kv_20", 2.0)]
        kv_sigmas = [s for c, s in kv_conds if c in agg]
        kv_accs = [agg[c]['accuracy'] * 100 for c, _ in kv_conds if c in agg]
        if kv_sigmas:
            ax.plot(kv_sigmas, kv_accs, '^-', color='#9b59b6', linewidth=2,
                    markersize=10, label='KV-mag Accuracy', zorder=4)

        for s, a in zip(k_sigmas, k_accs):
            ax.annotate(f'{a:.0f}%', (s, a), textcoords="offset points",
                       xytext=(-15, 10), ha='center', fontsize=9, color='#e74c3c')
        for s, a in zip(v_sigmas, v_accs):
            ax.annotate(f'{a:.0f}%', (s, a), textcoords="offset points",
                       xytext=(15, 10), ha='center', fontsize=9, color='#2ecc71')
        if kv_sigmas:
            for s, a in zip(kv_sigmas, kv_accs):
                ax.annotate(f'{a:.0f}%', (s, a), textcoords="offset points",
                           xytext=(0, -15), ha='center', fontsize=9, color='#9b59b6')

        ax.set_xlabel('Magnitude σ', fontsize=13)
        ax.set_ylabel('Answer Accuracy (%)', fontsize=13)
        ax.set_title('Phi K vs V Magnitude Dose-Response (Late 5%)\n'
                     'K > V under magnitude perturbation on MHA?', fontsize=14)
        ax.set_ylim(-5, 110)
        ax.set_xlim(0, 11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'kv_combined_dose_response.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 4: Cross-model encoding comparison (3 families)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # K-mag comparison
    qwen_k_data = [(0.5, 45.8), (1.0, 10.0)]
    llama_k_data = [(1.0, 96.0), (2.0, 94.0), (3.0, 80.0), (5.0, 72.0)]
    if k_sigmas:
        ax1.plot(k_sigmas, k_accs, 'o-', color='#e74c3c', linewidth=2.5,
                markersize=10, label='Phi (MHA)', zorder=5)
    ax1.plot([p[0] for p in qwen_k_data], [p[1] for p in qwen_k_data],
            'v-', color='#e67e22', linewidth=2, markersize=8, label='Qwen (GQA)')
    ax1.plot([p[0] for p in llama_k_data], [p[1] for p in llama_k_data],
            '^-', color='#27ae60', linewidth=2, markersize=8, label='Llama (GQA)')
    ax1.set_xlabel('Magnitude σ', fontsize=12)
    ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax1.set_title('K-Magnitude: 3-Family Comparison', fontsize=13)
    ax1.set_ylim(-5, 110)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # V-mag comparison
    qwen_v_data = [(1.0, 100.0), (3.0, 100.0), (5.0, 50.0), (10.0, 20.0)]
    llama_v_data = [(1.0, 100.0), (2.0, 98.0), (5.0, 89.0), (10.0, 65.0)]
    if v_sigmas:
        ax2.plot(v_sigmas, v_accs, 'o-', color='#e74c3c', linewidth=2.5,
                markersize=10, label='Phi (MHA)', zorder=5)
    ax2.plot([p[0] for p in qwen_v_data], [p[1] for p in qwen_v_data],
            'v-', color='#e67e22', linewidth=2, markersize=8, label='Qwen (GQA)')
    ax2.plot([p[0] for p in llama_v_data], [p[1] for p in llama_v_data],
            '^-', color='#27ae60', linewidth=2, markersize=8, label='Llama (GQA)')
    ax2.set_xlabel('Magnitude σ', fontsize=12)
    ax2.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax2.set_title('V-Magnitude: 3-Family Comparison', fontsize=13)
    ax2.set_ylim(-5, 110)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Exp 035: Cross-Model Encoding Strategy Comparison\n'
                 'Digital (Qwen cliffs) vs Analog (Llama gradual) vs Phi (?)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cross_model_encoding.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5: Superadditivity analysis
    kv_conds_all = [("mag_kv_10", 1.0), ("mag_kv_20", 2.0)]
    kv_present = [(c, s) for c, s in kv_conds_all if c in agg]
    if kv_present:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = []
        actual_vals = []
        independent_vals = []
        for cond_name, sigma in kv_present:
            k_key = f"mag_k_{int(sigma*10):02d}"
            v_key = f"mag_v_{int(sigma*10):02d}"
            if k_key in agg and v_key in agg:
                k_acc = agg[k_key]['accuracy']
                v_acc = agg[v_key]['accuracy']
                kv_acc = agg[cond_name]['accuracy']
                independent_pred = k_acc * v_acc  # if independent
                labels.append(f'σ={sigma}')
                actual_vals.append(kv_acc * 100)
                independent_vals.append(independent_pred * 100)

        if labels:
            x = np.arange(len(labels))
            w = 0.3
            ax.bar(x - w/2, actual_vals, w, label='KV-combined (actual)',
                   color='#9b59b6', alpha=0.85)
            ax.bar(x + w/2, independent_vals, w, label='K×V (independent pred)',
                   color='#95a5a6', alpha=0.85)
            ax.set_ylabel('Answer Accuracy (%)', fontsize=12)
            ax.set_title('Superadditivity Test on Phi (Late 5%)\n'
                         'Is KV-combined worse than K×V independent prediction?',
                         fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_ylim(0, 110)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.2, axis='y')
            for i, (a, p) in enumerate(zip(actual_vals, independent_vals)):
                if p > 0:
                    ratio = a / p
                    ax.text(i, max(a, p) + 3, f'ratio={ratio:.2f}x',
                            ha='center', fontsize=10, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'superadditivity.png'),
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
    print(f"Architecture: {'MHA' if n_kv_heads == n_attn_heads else 'GQA'}")

    # Verify cache access pattern
    test_input = tokenizer("test", return_tensors="pt").to(model.device)
    test_out = model(**test_input, use_cache=True)
    test_cache = test_out.past_key_values
    assert hasattr(test_cache, 'layers'), "Cache must have layers attribute"
    k_shape = test_cache.layers[0].keys.shape
    print(f"Cache verified: layers[0].keys.shape = {k_shape}")
    del test_input, test_out, test_cache
    gc.collect()
    torch.cuda.empty_cache()

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
            gc.collect()
            torch.cuda.empty_cache()
            continue

        trunc_pos = find_truncation_point(reasoning_ids, tokenizer)
        if trunc_pos is None or trunc_pos < 10:
            print(f"  Skip: no truncation point found in trace")
            n_skip_trunc += 1
            del prompt_ids, reasoning_ids
            gc.collect()
            torch.cuda.empty_cache()
            continue

        reasoning_ids_truncated = reasoning_ids[:, :trunc_pos]
        prompt_len = prompt_ids.shape[1]
        reasoning_len = reasoning_ids_truncated.shape[1]

        if reasoning_len < 20 or (prompt_len + reasoning_len) > MAX_SEQ_LEN:
            print(f"  Skip: R={reasoning_len}, total={prompt_len + reasoning_len}")
            n_skip_trunc += 1
            del prompt_ids, reasoning_ids
            gc.collect()
            torch.cuda.empty_cache()
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
            gc.collect()
            torch.cuda.empty_cache()
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

        # Select late 5% positions (same for all conditions)
        positions = select_late_positions(reasoning_len, noise_frac=0.05)

        # Run all conditions
        for cond_name, component, sigma in CONDITIONS:
            pc = build_prompt_cache(model, prompt_ids, num_layers)
            ev = evaluate_condition(
                model, tokenizer, pc, reasoning_ids_truncated,
                positions, prompt_len, num_layers, true_answer,
                perturb_component=component, magnitude_sigma=sigma)

            print(f"    {cond_name}: acc={'Y' if ev['correct'] else 'N'}, "
                  f"text={ev['text_accuracy']:.1%}, n_pos={len(positions)}, "
                  f"k_rms={ev['k_rms']:.1f}, v_rms={ev['v_rms']:.1f}")

            problem_result['evaluations'][cond_name] = {
                'correct': ev['correct'],
                'answer': ev['answer'],
                'text_accuracy': ev['text_accuracy'],
                'n_noised': len(positions),
                'k_rms': ev['k_rms'],
                'v_rms': ev['v_rms'],
                'k_signal_rms': ev['k_signal_rms'],
                'v_signal_rms': ev['v_signal_rms'],
                'scale_factors_k': ev['scale_factors_k'],
                'scale_factors_v': ev['scale_factors_v'],
            }

        results.append(problem_result)
        del prompt_ids, reasoning_ids, reasoning_ids_truncated, prompt_cache
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_valid} valid, {len(results)} processed, "
          f"{n_skip_wrong} wrong, {n_skip_trunc} truncation skips")

    summary = {
        'experiment': 'exp_035_phi_magnitude_dose',
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
            a_mean = np.mean(accs)
            t_mean = np.mean(txts)
            ct_mean = np.mean(ctxts)
            a_drop = 1.0 - a_mean
            t_drop = ct_mean - t_mean
            dissoc = a_drop - t_drop
            n_count = len(accs)
            p_hat = a_mean
            z = 1.96
            if n_count > 0:
                wilson_lo = (p_hat + z*z/(2*n_count) - z*((p_hat*(1-p_hat)/n_count + z*z/(4*n_count*n_count))**0.5)) / (1 + z*z/n_count)
                wilson_hi = (p_hat + z*z/(2*n_count) + z*((p_hat*(1-p_hat)/n_count + z*z/(4*n_count*n_count))**0.5)) / (1 + z*z/n_count)
            else:
                wilson_lo = wilson_hi = 0
            agg[cond_name] = {
                'n': n_count,
                'accuracy': float(a_mean),
                'text_accuracy': float(t_mean),
                'clean_text_accuracy': float(ct_mean),
                'accuracy_drop': float(a_drop),
                'text_drop': float(t_drop),
                'dissociation': float(dissoc),
                'wilson_ci_lo': float(max(0, wilson_lo)),
                'wilson_ci_hi': float(min(1, wilson_hi)),
                'k_rms': float(np.mean(k_rms_list)),
                'v_rms': float(np.mean(v_rms_list)),
                'k_signal_rms': float(np.mean(k_sig_list)),
                'v_signal_rms': float(np.mean(v_sig_list)),
            }

    # Print summary table
    print(f"\n{'Condition':<14} {'σ':>4} {'Acc%':>6} {'Text%':>6} {'AccDrop':>8} "
          f"{'TxtDrop':>8} {'Dissoc':>8} {'K_RMS':>8} {'V_RMS':>8} {'n':>3} {'Wilson 95% CI':>18}")
    print("-" * 110)
    for cond_tuple in CONDITIONS:
        cond_name = cond_tuple[0]
        sigma = cond_tuple[2]
        if cond_name in agg:
            d = agg[cond_name]
            print(f"  {cond_name:<12} {sigma:>4.1f} {d['accuracy']*100:>5.1f}% "
                  f"{d['text_accuracy']*100:>5.1f}% "
                  f"{d['accuracy_drop']*100:>7.1f}% {d['text_drop']*100:>7.1f}% "
                  f"{d['dissociation']*100:>7.1f}% {d['k_rms']:>7.1f} {d['v_rms']:>7.1f} {d['n']:>3}"
                  f" [{d['wilson_ci_lo']*100:>5.1f}, {d['wilson_ci_hi']*100:>5.1f}]")

    # K-mag dose-response summary
    print(f"\n── K-MAGNITUDE DOSE-RESPONSE (Phi, late 5%) ──")
    for cond_name, _, sigma in CONDITIONS:
        if cond_name.startswith('mag_k_') and cond_name in agg:
            d = agg[cond_name]
            n_correct = int(round(d['accuracy'] * d['n']))
            print(f"  σ={sigma:.1f}: {d['accuracy']*100:.1f}% ({n_correct}/{d['n']})")

    print(f"\n── V-MAGNITUDE DOSE-RESPONSE (Phi, late 5%) ──")
    for cond_name, _, sigma in CONDITIONS:
        if cond_name.startswith('mag_v_') and cond_name in agg:
            d = agg[cond_name]
            n_correct = int(round(d['accuracy'] * d['n']))
            print(f"  σ={sigma:.1f}: {d['accuracy']*100:.1f}% ({n_correct}/{d['n']})")

    print(f"\n── KV-MAGNITUDE COMBINED ──")
    for cond_name, _, sigma in CONDITIONS:
        if cond_name.startswith('mag_kv_') and cond_name in agg:
            d = agg[cond_name]
            n_correct = int(round(d['accuracy'] * d['n']))
            print(f"  σ={sigma:.1f}: {d['accuracy']*100:.1f}% ({n_correct}/{d['n']})")

    # Superadditivity analysis
    print(f"\n── SUPERADDITIVITY ANALYSIS ──")
    for sigma in [1.0, 2.0]:
        k_key = f"mag_k_{int(sigma*10):02d}"
        v_key = f"mag_v_{int(sigma*10):02d}"
        kv_key = f"mag_kv_{int(sigma*10):02d}"
        if k_key in agg and v_key in agg and kv_key in agg:
            k_acc = agg[k_key]['accuracy']
            v_acc = agg[v_key]['accuracy']
            kv_acc = agg[kv_key]['accuracy']
            indep = k_acc * v_acc
            ratio = kv_acc / max(indep, 0.001)
            print(f"  σ={sigma:.1f}: K={k_acc*100:.1f}%, V={v_acc*100:.1f}%, "
                  f"KV={kv_acc*100:.1f}%, K×V(indep)={indep*100:.1f}%, "
                  f"ratio={ratio:.2f}x {'SUPERADDITIVE' if ratio < 0.8 else 'INDEPENDENT'}")

    # Cross-model comparison
    print(f"\n── CROSS-MODEL COMPARISON: Encoding Strategy ──")
    print(f"  Qwen (GQA, digital): K cliff at σ=0.3-1.0, V cliff at σ=3-5")
    print(f"  Llama (GQA, analog): K gradual σ=1-5 (96→72%), V gradual σ=1-10 (100→65%)")
    if 'mag_k_05' in agg and 'mag_k_10' in agg:
        k05 = agg['mag_k_05']['accuracy'] * 100
        k10 = agg['mag_k_10']['accuracy'] * 100
        print(f"  Phi (MHA): K σ=0.5→{k05:.0f}%, σ=1.0→{k10:.0f}%")
        if k05 > 80 and k10 > 80:
            print(f"  → Phi K-mag is ROBUST like Llama (analog)")
        elif k05 < 50 or k10 < 20:
            print(f"  → Phi K-mag shows CLIFF like Qwen (digital)")
        else:
            print(f"  → Phi K-mag is INTERMEDIATE")
    if 'mag_v_30' in agg and 'mag_v_50' in agg:
        v30 = agg['mag_v_30']['accuracy'] * 100
        v50 = agg['mag_v_50']['accuracy'] * 100
        print(f"  Phi V: σ=3.0→{v30:.0f}%, σ=5.0→{v50:.0f}%")

    # K > V at matched doses
    print(f"\n── K > V AT MATCHED DOSES ──")
    for sigma in [1.0, 5.0]:
        k_key = f"mag_k_{int(sigma*10):02d}"
        v_key = f"mag_v_{int(sigma*10):02d}"
        if k_key in agg and v_key in agg:
            k_acc = agg[k_key]['accuracy'] * 100
            v_acc = agg[v_key]['accuracy'] * 100
            gap = v_acc - k_acc
            print(f"  σ={sigma:.1f}: K={k_acc:.1f}%, V={v_acc:.1f}%, "
                  f"V-K gap={gap:+.1f}pp {'K > V CONFIRMED' if gap > 10 else 'K ≈ V' if abs(gap) < 10 else 'V > K !!!'}")

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
