#!/usr/bin/env python3
"""
Experiment 026: Qwen V-Magnitude Dose-Response + KV Superadditivity

Tests:
  1. V-magnitude dose-response on Qwen (σ=1-10): cliff or gradual?
  2. K-magnitude fine-grained dose-response around the cliff (σ=0.3-2.0)
  3. KV-combined superadditivity: does it replicate from Llama (exp_025)?

Key prior results on Qwen (exp_023):
  - K-mag σ=0.5: 45.8%, σ=1.0: 0% (CLIFF)
  - V-mag σ=1.0: 100% (24/24 immune)
  - KV-mag σ=1.0: 0%
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
MAX_SEQ_LEN = 1536
NOISE_FRAC = 0.05  # 5% of positions
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_026")

# Same 8-shot exemplars as exp_022/023 (Qwen format)
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
# (name, perturb_component, sigma)
# All conditions at late 5% positions
CONDITIONS = [
    # V-magnitude dose escalation (PRIMARY — filling the gap)
    ("mag_v_03",  "v",  0.3),   # needed for superadditivity at σ=0.3
    ("mag_v_05",  "v",  0.5),   # needed for superadditivity at σ=0.5
    ("mag_v_10",  "v",  1.0),   # replication of exp_023: 100%
    ("mag_v_20",  "v",  2.0),
    ("mag_v_30",  "v",  3.0),
    ("mag_v_50",  "v",  5.0),
    ("mag_v_100", "v",  10.0),
    # K-magnitude fine-grained around cliff
    ("mag_k_03",  "k",  0.3),   # below cliff
    ("mag_k_05",  "k",  0.5),   # at cliff edge (exp_023: 45.8%)
    ("mag_k_10",  "k",  1.0),   # past cliff (exp_023: 16.7%)
    ("mag_k_20",  "k",  2.0),
    # KV-combined superadditivity test
    ("mag_kv_03", "kv", 0.3),   # below K cliff — key superadditivity test
    ("mag_kv_05", "kv", 0.5),   # at K cliff edge
    ("mag_kv_10", "kv", 1.0),   # past cliff (exp_023: 0%)
    ("mag_kv_20", "kv", 2.0),
    ("mag_kv_50", "kv", 5.0),
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


def perturb_magnitude(tensor, sigma=1.0):
    """Scale per-head norm randomly, preserve direction exactly.
    tensor shape: [1, n_heads, 1, head_dim]
    """
    norms = tensor.norm(dim=-1, keepdim=True)
    direction = tensor / (norms + 1e-8)
    delta = torch.randn(norms.shape, device=tensor.device, dtype=tensor.dtype) * sigma
    scale = (1 + delta).clamp(min=0.01)
    return direction * norms * scale


@torch.no_grad()
def evaluate_condition(model, tokenizer, prompt_cache, reasoning_tokens,
                       positions_to_noise, prompt_len, num_layers, true_answer,
                       perturb_component='kv', magnitude_sigma=1.0):
    """
    Step through reasoning tokens, apply magnitude perturbation at selected positions.
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

        if i in noise_set:
            for l in range(num_layers):
                pos = prompt_len + i
                k_orig = cache.layers[l].keys[:, :, pos:pos+1, :].clone()
                v_orig = cache.layers[l].values[:, :, pos:pos+1, :].clone()

                if perturb_component in ('kv', 'k'):
                    k_slice = cache.layers[l].keys[:, :, pos:pos+1, :]
                    k_new = perturb_magnitude(k_slice, sigma=magnitude_sigma)
                    cache.layers[l].keys[:, :, pos:pos+1, :] = k_new
                    perturb_l2_total += (k_new - k_orig).float().norm().item() ** 2

                if perturb_component in ('kv', 'v'):
                    v_slice = cache.layers[l].values[:, :, pos:pos+1, :]
                    v_new = perturb_magnitude(v_slice, sigma=magnitude_sigma)
                    cache.layers[l].values[:, :, pos:pos+1, :] = v_new
                    perturb_l2_total += (v_new - v_orig).float().norm().item() ** 2

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


def generate_figures(agg, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Figure 1: V-magnitude dose-response curve ──
    v_conds = [("mag_v_03", 0.3), ("mag_v_05", 0.5), ("mag_v_10", 1.0),
               ("mag_v_20", 2.0), ("mag_v_30", 3.0),
               ("mag_v_50", 5.0), ("mag_v_100", 10.0)]
    v_sigmas = [s for c, s in v_conds if c in agg]
    v_accs = [agg[c]['accuracy'] * 100 for c, _ in v_conds if c in agg]
    v_txts = [agg[c]['text_accuracy'] * 100 for c, _ in v_conds if c in agg]

    if v_sigmas:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(v_sigmas, v_accs, 'o-', color='#2ecc71', linewidth=2.5,
                markersize=10, label='Answer Accuracy', zorder=5)
        ax.plot(v_sigmas, v_txts, 's--', color='#3498db', linewidth=2,
                markersize=8, label='Text Accuracy', zorder=4)
        for s, a, t in zip(v_sigmas, v_accs, v_txts):
            ax.annotate(f'{a:.0f}%', (s, a), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=10, color='#2ecc71',
                       fontweight='bold')
        # Add Llama V-mag reference points
        llama_v = [(1.0, 100.0), (2.0, 98.1), (5.0, 88.9), (10.0, 64.8)]
        ax.plot([x[0] for x in llama_v], [x[1] for x in llama_v],
                '^:', color='#e67e22', linewidth=1.5, markersize=7, alpha=0.7,
                label='Llama V-mag (exp_025)')
        ax.set_xlabel('Magnitude σ', fontsize=13)
        ax.set_ylabel('Accuracy (%)', fontsize=13)
        ax.set_title('Qwen V-Magnitude Dose-Response (Late 5%)\nCliff (digital) or gradual (analog)?',
                     fontsize=14)
        ax.set_ylim(-5, 110)
        ax.set_xlim(0, 11)
        ax.legend(fontsize=11, loc='lower left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'v_magnitude_dose_response.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 2: K-magnitude fine-grained dose-response ──
    k_conds = [("mag_k_03", 0.3), ("mag_k_05", 0.5), ("mag_k_10", 1.0), ("mag_k_20", 2.0)]
    k_sigmas = [s for c, s in k_conds if c in agg]
    k_accs = [agg[c]['accuracy'] * 100 for c, _ in k_conds if c in agg]
    k_txts = [agg[c]['text_accuracy'] * 100 for c, _ in k_conds if c in agg]

    if k_sigmas:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_sigmas, k_accs, 'o-', color='#e74c3c', linewidth=2.5,
                markersize=10, label='Answer Accuracy', zorder=5)
        ax.plot(k_sigmas, k_txts, 's--', color='#3498db', linewidth=2,
                markersize=8, label='Text Accuracy', zorder=4)
        for s, a, t in zip(k_sigmas, k_accs, k_txts):
            ax.annotate(f'{a:.0f}%', (s, a), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=10, color='#e74c3c',
                       fontweight='bold')
        # Add exp_023 reference points
        ax.plot([0.5, 1.0], [45.8, 16.7], 'x', color='gray', markersize=12,
                markeredgewidth=2, label='Exp_023 (n=24)', zorder=6)
        # Add Llama K-mag reference
        llama_k = [(1.0, 96.3), (2.0, 94.4), (3.0, 79.6), (5.0, 72.2)]
        ax.plot([x[0] for x in llama_k], [x[1] for x in llama_k],
                '^:', color='#e67e22', linewidth=1.5, markersize=7, alpha=0.7,
                label='Llama K-mag (exp_025)')
        ax.set_xlabel('Magnitude σ', fontsize=13)
        ax.set_ylabel('Accuracy (%)', fontsize=13)
        ax.set_title('Qwen K-Magnitude Dose-Response (Late 5%)\nFine-grained cliff characterization',
                     fontsize=14)
        ax.set_ylim(-5, 110)
        ax.set_xlim(0, 2.5)
        ax.legend(fontsize=11, loc='center right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'k_magnitude_dose_response.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 3: KV superadditivity comparison ──
    kv_conds = [("mag_kv_03", 0.3), ("mag_kv_05", 0.5), ("mag_kv_10", 1.0),
                ("mag_kv_20", 2.0), ("mag_kv_50", 5.0)]
    kv_sigmas = [s for c, s in kv_conds if c in agg]
    kv_accs = [agg[c]['accuracy'] * 100 for c, _ in kv_conds if c in agg]

    if kv_sigmas and k_sigmas:
        fig, ax = plt.subplots(figsize=(12, 6))
        # K-only
        ax.plot(k_sigmas, k_accs, 'o-', color='#e74c3c', linewidth=2.5,
                markersize=10, label='K-mag only', zorder=5)
        # V-only (use available sigmas)
        if v_sigmas:
            ax.plot(v_sigmas, v_accs, 's-', color='#2ecc71', linewidth=2.5,
                    markersize=10, label='V-mag only', zorder=5)
        # KV-combined
        ax.plot(kv_sigmas, kv_accs, '^-', color='#9b59b6', linewidth=2.5,
                markersize=10, label='KV-mag combined', zorder=5)

        # Expected independent line (K_acc * V_acc / 100)
        expected_sigmas = []
        expected_accs = []
        k_dict = {s: a for s, a in zip(k_sigmas, k_accs)}
        v_dict = {s: a for s, a in zip(v_sigmas, v_accs)}
        for s in kv_sigmas:
            k_a = k_dict.get(s)
            v_a = v_dict.get(s)
            if k_a is not None and v_a is not None:
                expected_sigmas.append(s)
                expected_accs.append(k_a * v_a / 100)
        if expected_sigmas:
            ax.plot(expected_sigmas, expected_accs, 'D--', color='gray',
                    linewidth=1.5, markersize=8, alpha=0.7,
                    label='Expected if independent')

        for s, a in zip(kv_sigmas, kv_accs):
            ax.annotate(f'{a:.0f}%', (s, a), textcoords="offset points",
                       xytext=(12, -5), ha='left', fontsize=9, color='#9b59b6',
                       fontweight='bold')

        ax.set_xlabel('Magnitude σ', fontsize=13)
        ax.set_ylabel('Answer Accuracy (%)', fontsize=13)
        ax.set_title('KV Superadditivity Test on Qwen (Late 5%)\nIs KV-combined worse than independent prediction?',
                     fontsize=14)
        ax.set_ylim(-5, 110)
        ax.set_xlim(0, 5.5)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'kv_superadditivity.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 4: Cross-model K vs V comparison ──
    if k_sigmas and v_sigmas:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Qwen
        ax1.plot(k_sigmas, k_accs, 'o-', color='#e74c3c', linewidth=2.5,
                 markersize=10, label='K-mag')
        if v_sigmas:
            ax1.plot(v_sigmas, v_accs, 's-', color='#2ecc71', linewidth=2.5,
                     markersize=10, label='V-mag')
        ax1.set_title('Qwen3-4B-Base', fontsize=13)
        ax1.set_xlabel('Magnitude σ', fontsize=12)
        ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax1.set_ylim(-5, 110)
        ax1.set_xlim(0, 11)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Llama reference
        llama_k = [(1.0, 96.3), (2.0, 94.4), (3.0, 79.6), (5.0, 72.2)]
        llama_v = [(1.0, 100.0), (2.0, 98.1), (5.0, 88.9), (10.0, 64.8)]
        ax2.plot([x[0] for x in llama_k], [x[1] for x in llama_k],
                 'o-', color='#e74c3c', linewidth=2.5, markersize=10, label='K-mag')
        ax2.plot([x[0] for x in llama_v], [x[1] for x in llama_v],
                 's-', color='#2ecc71', linewidth=2.5, markersize=10, label='V-mag')
        ax2.set_title('Llama-3.1-8B (exp_025)', fontsize=13)
        ax2.set_xlabel('Magnitude σ', fontsize=12)
        ax2.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax2.set_ylim(-5, 110)
        ax2.set_xlim(0, 11)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Cross-Model K vs V Magnitude Dose-Response\nDigital (Qwen) vs Analog (Llama)',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'cross_model_kv_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 5: All conditions bar chart ──
    all_conds = [c[0] for c in CONDITIONS]
    all_keys = [c for c in all_conds if c in agg]
    if all_keys:
        all_accs = [agg[c]['accuracy'] * 100 for c in all_keys]
        all_txts = [agg[c]['text_accuracy'] * 100 for c in all_keys]
        colors = []
        for c in all_keys:
            if c.startswith('mag_k_'):
                colors.append('#e74c3c')
            elif c.startswith('mag_v_'):
                colors.append('#2ecc71')
            else:
                colors.append('#9b59b6')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        x = np.arange(len(all_keys))

        bars1 = ax1.bar(x, all_accs, 0.6, color=colors, alpha=0.85)
        ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax1.set_title('All Conditions: Accuracy', fontsize=13)
        ax1.set_xticks(x)
        labels = [c.replace('mag_', '').replace('_', ' σ=') for c in all_keys]
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax1.set_ylim(0, 110)
        ax1.grid(True, alpha=0.2, axis='y')
        for b, v in zip(bars1, all_accs):
            ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f'{v:.0f}%', ha='center', va='bottom', fontsize=7)

        bars2 = ax2.bar(x, all_txts, 0.6, color=colors, alpha=0.85)
        ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
        ax2.set_title('All Conditions: Text Quality', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylim(90, 101)
        ax2.grid(True, alpha=0.2, axis='y')
        for b, v in zip(bars2, all_txts):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=7)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.85, label='K-magnitude'),
            Patch(facecolor='#2ecc71', alpha=0.85, label='V-magnitude'),
            Patch(facecolor='#9b59b6', alpha=0.85, label='KV-magnitude'),
        ]
        ax1.legend(handles=legend_elements, fontsize=9)
        plt.suptitle('Exp 026: Qwen Magnitude Dose-Response (Late 5%)', fontsize=14)
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

        # Select late 5% positions
        positions = select_late_positions(reasoning_len)

        # Run all conditions
        for cond_name, component, sigma in CONDITIONS:
            pc = build_prompt_cache(model, prompt_ids, num_layers)
            ev = evaluate_condition(
                model, tokenizer, pc, reasoning_ids_truncated,
                positions, prompt_len, num_layers, true_answer,
                perturb_component=component, magnitude_sigma=sigma)

            print(f"    {cond_name}: acc={'Y' if ev['correct'] else 'N'}, "
                  f"text={ev['text_accuracy']:.1%}, n_pos={len(positions)}, "
                  f"rms={ev['rms_perturbation']:.1f}")

            problem_result['evaluations'][cond_name] = {
                'correct': ev['correct'],
                'answer': ev['answer'],
                'text_accuracy': ev['text_accuracy'],
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
        'experiment': 'exp_026_qwen_dose_response',
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
                'rms_perturbation': float(np.mean(rms_list)),
            }

    # Print summary table
    print(f"\n{'Condition':<14} {'σ':>4} {'Acc%':>6} {'Text%':>6} {'AccDrop':>8} "
          f"{'TxtDrop':>8} {'Dissoc':>8} {'RMS':>8} {'n':>3}")
    print("-" * 80)
    for cond_tuple in CONDITIONS:
        cond_name = cond_tuple[0]
        sigma = cond_tuple[2]
        if cond_name in agg:
            d = agg[cond_name]
            print(f"  {cond_name:<12} {sigma:>4.1f} {d['accuracy']*100:>5.1f}% "
                  f"{d['text_accuracy']*100:>5.1f}% "
                  f"{d['accuracy_drop']*100:>7.1f}% {d['text_drop']*100:>7.1f}% "
                  f"{d['dissociation']*100:>7.1f}% {d['rms_perturbation']:>7.1f} {d['n']:>3}")

    # V-mag dose-response summary
    print(f"\n── V-MAGNITUDE DOSE-RESPONSE (PRIMARY) ──")
    for cond_name, _, sigma in CONDITIONS:
        if cond_name.startswith('mag_v_') and cond_name in agg:
            d = agg[cond_name]
            print(f"  σ={sigma:.1f}: {d['accuracy']*100:.1f}% ({d['n_correct']}/{d['n']})")

    print(f"\n── K-MAGNITUDE DOSE-RESPONSE ──")
    for cond_name, _, sigma in CONDITIONS:
        if cond_name.startswith('mag_k_') and cond_name in agg:
            d = agg[cond_name]
            print(f"  σ={sigma:.1f}: {d['accuracy']*100:.1f}% ({d['n_correct']}/{d['n']})")

    print(f"\n── KV-COMBINED (SUPERADDITIVITY TEST) ──")
    for cond_name, _, sigma in CONDITIONS:
        if cond_name.startswith('mag_kv_') and cond_name in agg:
            d = agg[cond_name]
            k_key = f"mag_k_{cond_name.split('_')[-1]}"
            v_key = f"mag_v_{cond_name.split('_')[-1]}"
            k_acc = agg[k_key]['accuracy'] * 100 if k_key in agg else None
            v_acc = agg[v_key]['accuracy'] * 100 if v_key in agg else None
            kv_acc = d['accuracy'] * 100
            if k_acc is not None and v_acc is not None:
                expected = k_acc * v_acc / 100
                ratio = expected / max(kv_acc, 0.1)
                print(f"  σ={sigma:.1f}: KV={kv_acc:.1f}% | K={k_acc:.1f}% V={v_acc:.1f}% "
                      f"Expected={expected:.1f}% | Superadditive={ratio:.1f}x")
            else:
                print(f"  σ={sigma:.1f}: KV={kv_acc:.1f}% ({d['n_correct']}/{d['n']})")

    # Cross-model comparison with exp_023
    print(f"\n── COMPARISON WITH EXP_023 (Qwen, n=24) ──")
    exp023 = {'mag_k_05': 45.8, 'mag_k_10': 16.7, 'mag_v_10': 100.0, 'mag_kv_10': 0.0}
    for key, exp023_val in exp023.items():
        if key in agg:
            new_val = agg[key]['accuracy'] * 100
            print(f"  {key}: exp_023={exp023_val:.1f}%, exp_026={new_val:.1f}%, "
                  f"diff={new_val-exp023_val:+.1f}pp")

    # Cross-model comparison with exp_025 (Llama)
    print(f"\n── CROSS-MODEL: Qwen (exp_026) vs Llama (exp_025) ──")
    llama_results = {
        'mag_k_10': 96.3, 'mag_k_20': 94.4,
        'mag_v_10': 100.0, 'mag_v_20': 98.1, 'mag_v_50': 88.9, 'mag_v_100': 64.8,
        'mag_kv_20': 42.6, 'mag_kv_50': 3.7,
    }
    for key, llama_val in llama_results.items():
        if key in agg:
            qwen_val = agg[key]['accuracy'] * 100
            print(f"  {key}: Qwen={qwen_val:.1f}%, Llama={llama_val:.1f}%, "
                  f"gap={qwen_val-llama_val:+.1f}pp")

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
