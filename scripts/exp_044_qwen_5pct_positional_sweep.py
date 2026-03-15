#!/usr/bin/env python3
"""
Experiment 044: 5% Dose 10-Decile K-Only Positional Sweep on Qwen3-4B-Base

Cross-model comparison with Exp 043 (Llama 5%). Tests whether Qwen's digital
K-direction encoding creates a different accuracy profile at 5% dose compared
to Llama's analog encoding, which showed accuracy saturated at 0-2.6% for
bins 0-6.

DISCONFIRMATORY: If Qwen shows the same flat pattern as Llama, the encoding
taxonomy (Finding #4) has no functional consequence for positional robustness.

CONDITIONS:
  - 10 K-only direction perturbation bins: decile_k_0 through decile_k_9
    Each bin perturbs 50% of positions within the decile (~5% of total chain)
  - 1 V-only control at bin 9 (50% of positions in 90-100% decile)
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
SUBSAMPLE_FRACTION = 0.5  # 50% of decile → 5% of total chain
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_044")

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
CONDITIONS = []
for d in range(10):
    CONDITIONS.append((f"decile_k_{d}", d, "k"))
CONDITIONS.append(("decile_v_9", 9, "v"))  # V-only control at latest decile


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


def select_decile_positions_subsampled(reasoning_len, decile, fraction=SUBSAMPLE_FRACTION, rng_seed=None):
    """Select a fraction of positions within a decile bin.
    At fraction=0.5, this gives ~5% of total chain (50% of a 10% decile).
    """
    start_frac = decile / 10.0
    end_frac = (decile + 1) / 10.0
    start_pos = int(reasoning_len * start_frac)
    end_pos = int(reasoning_len * end_frac)
    if end_pos <= start_pos:
        end_pos = start_pos + 1
    end_pos = min(end_pos, reasoning_len)
    all_positions = list(range(start_pos, end_pos))

    n_select = max(1, int(len(all_positions) * fraction))
    rng = random.Random(rng_seed if rng_seed is not None else (decile * 1000 + reasoning_len))
    selected = sorted(rng.sample(all_positions, min(n_select, len(all_positions))))
    return selected


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
                       perturb_component='k'):
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
    raw_ans = extract_answer(answer_text) if answer_text else ""
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


def generate_figures(results, agg, results_dir, exp043_agg=None, exp042_agg=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decile_labels = [f"{d*10}-{(d+1)*10}%" for d in range(10)]

    # ── Figure 1: THE KEY FIGURE — Accuracy vs Position Decile ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    k_accs = []
    k_acc_los = []
    k_acc_his = []
    k_texts = []
    for d in range(10):
        key = f"decile_k_{d}"
        if key in agg:
            k_accs.append(agg[key]['accuracy'] * 100)
            k_acc_los.append(agg[key]['wilson_ci_lo'] * 100)
            k_acc_his.append(agg[key]['wilson_ci_hi'] * 100)
            k_texts.append(agg[key]['text_accuracy'] * 100)
        else:
            k_accs.append(float('nan'))
            k_acc_los.append(float('nan'))
            k_acc_his.append(float('nan'))
            k_texts.append(float('nan'))

    x = np.arange(10)

    # Panel 1: Answer accuracy
    ax1.errorbar(x, k_accs,
                 yerr=[np.array(k_accs) - np.array(k_acc_los),
                       np.array(k_acc_his) - np.array(k_accs)],
                 fmt='o-', color='#e74c3c', linewidth=2, markersize=8,
                 capsize=5, label='K-only 5% dose', zorder=5)

    # V-only control point at bin 9
    v9_key = "decile_v_9"
    if v9_key in agg:
        v9_acc = agg[v9_key]['accuracy'] * 100
        v9_lo = agg[v9_key]['wilson_ci_lo'] * 100
        v9_hi = agg[v9_key]['wilson_ci_hi'] * 100
        ax1.errorbar([9], [v9_acc], yerr=[[v9_acc - v9_lo], [v9_hi - v9_acc]],
                     fmt='s', color='#3498db', markersize=10, capsize=5,
                     label='V-only 5% dose (control)', zorder=6)

    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
    ax1.set_title('Exp 044: K-Only Direction Perturbation at 5% Dose by Position Decile\n'
                  'Qwen3-4B-Base (Digital Encoding) — Does digital encoding protect?',
                  fontsize=13)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(-5, 115)

    # Panel 2: Text accuracy
    ax2.plot(x, k_texts, 'o-', color='#2ecc71', linewidth=2, markersize=8,
             label='K-only text accuracy (5% dose)', zorder=5)

    if v9_key in agg:
        v9_text = agg[v9_key]['text_accuracy'] * 100
        ax2.plot([9], [v9_text], 's', color='#3498db', markersize=10,
                 label='V-only text accuracy', zorder=6)

    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Position Decile (% of reasoning chain)', fontsize=12)
    ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(decile_labels, fontsize=10, rotation=45)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'positional_sweep_accuracy.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: Dissociation by position ──
    fig, ax = plt.subplots(figsize=(12, 7))

    acc_drops = []
    text_drops = []
    dissociation = []
    for d in range(10):
        key = f"decile_k_{d}"
        if key in agg:
            acc_drops.append(agg[key]['accuracy_drop'] * 100)
            text_drops.append(agg[key]['text_drop'] * 100)
            dissociation.append(agg[key]['dissociation'] * 100)
        else:
            acc_drops.append(float('nan'))
            text_drops.append(float('nan'))
            dissociation.append(float('nan'))

    ax.bar(x - 0.2, acc_drops, 0.35, label='Accuracy Drop (%)',
           color='#e74c3c', alpha=0.8)
    ax.bar(x + 0.2, text_drops, 0.35, label='Text Drop (%)',
           color='#2ecc71', alpha=0.8)
    ax.plot(x, dissociation, 'k^-', linewidth=2, markersize=8,
            label='Dissociation (AccDrop - TextDrop)', zorder=5)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Position Decile (% of reasoning chain)', fontsize=12)
    ax.set_ylabel('Effect Size (%)', fontsize=12)
    ax.set_title('Text-Answer Dissociation by Position Decile (5% dose, Qwen-Base)\n'
                 'High dissociation = answer channel selectively disrupted',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(decile_labels, fontsize=10, rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dissociation_by_position.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3: Cross-model comparison (Qwen 5% vs Llama 5%) ──
    if exp043_agg:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Qwen (this experiment)
        qwen_accs = [agg.get(f"decile_k_{d}", {}).get('accuracy', float('nan')) * 100 for d in range(10)]
        qwen_texts = [agg.get(f"decile_k_{d}", {}).get('text_accuracy', float('nan')) * 100 for d in range(10)]

        # Llama (Exp 043)
        llama_accs = [exp043_agg.get(f"decile_k_{d}", {}).get('accuracy', float('nan')) * 100 for d in range(10)]
        llama_texts = [exp043_agg.get(f"decile_k_{d}", {}).get('text_accuracy', float('nan')) * 100 for d in range(10)]

        # Panel 1: Answer accuracy comparison
        ax1.plot(x, qwen_accs, 's-', color='#e74c3c', linewidth=2, markersize=8,
                 label='Qwen3-4B-Base (digital)')
        ax1.plot(x, llama_accs, 'o--', color='#c0392b', linewidth=2, markersize=8,
                 alpha=0.6, label='Llama-3.1-8B (analog)')

        # V-only comparisons
        qwen_v9 = agg.get("decile_v_9", {})
        llama_v9 = exp043_agg.get("decile_v_9", {})
        if qwen_v9:
            ax1.plot([8.8], [qwen_v9.get('accuracy', 0) * 100], 'D', color='#3498db',
                     markersize=10, label='V-only Qwen 5%')
        if llama_v9:
            ax1.plot([9.2], [llama_v9.get('accuracy', 0) * 100], 'D', color='#2980b9',
                     markersize=10, alpha=0.6, label='V-only Llama 5%')

        ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Position Decile', fontsize=12)
        ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax1.set_title('Answer Accuracy: Qwen (Digital) vs Llama (Analog)\nBoth at 5% Dose', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(decile_labels, fontsize=9, rotation=45)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)
        ax1.set_ylim(-5, 115)

        # Panel 2: Text accuracy comparison
        ax2.plot(x, qwen_texts, 's-', color='#27ae60', linewidth=2, markersize=8,
                 label='Qwen3-4B-Base (digital)')
        ax2.plot(x, llama_texts, 'o--', color='#1e8449', linewidth=2, markersize=8,
                 alpha=0.6, label='Llama-3.1-8B (analog)')

        ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Position Decile', fontsize=12)
        ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
        ax2.set_title('Text Accuracy: Qwen (Digital) vs Llama (Analog)\nBoth at 5% Dose', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(decile_labels, fontsize=9, rotation=45)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        plt.suptitle('Cross-Model Comparison at 5% Dose\n'
                     'Does digital encoding create accuracy gradients?',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'cross_model_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 4: Cross-dose comparison (Qwen 5% vs Qwen 10%) ──
    if exp042_agg:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Qwen 10% (Exp 042)
        qwen10_accs = [exp042_agg.get(f"decile_k_{d}", {}).get('accuracy', float('nan')) * 100 for d in range(10)]
        qwen10_texts = [exp042_agg.get(f"decile_k_{d}", {}).get('text_accuracy', float('nan')) * 100 for d in range(10)]

        ax1.plot(x, qwen10_accs, 'o--', color='#e74c3c', linewidth=2, markersize=8,
                 alpha=0.6, label='10% dose (Exp 042)')
        ax1.plot(x, k_accs, 's-', color='#c0392b', linewidth=2, markersize=8,
                 label='5% dose (Exp 044)')

        # V-only comparison
        v10 = exp042_agg.get("decile_v_9", {})
        if v10:
            ax1.plot([9.2], [v10.get('accuracy', 0) * 100], 'D', color='#3498db',
                     markersize=10, alpha=0.6, label='V-only 10% (Exp 042)')
        if v9_key in agg:
            ax1.plot([8.8], [agg[v9_key].get('accuracy', 0) * 100], 'D', color='#2980b9',
                     markersize=10, label='V-only 5% (Exp 044)')

        ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Position Decile', fontsize=12)
        ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
        ax1.set_title('Answer Accuracy: 5% vs 10% Dose on Qwen-Base', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(decile_labels, fontsize=9, rotation=45)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)
        ax1.set_ylim(-5, 115)

        ax2.plot(x, qwen10_texts, 'o--', color='#27ae60', linewidth=2, markersize=8,
                 alpha=0.6, label='10% dose (Exp 042)')
        ax2.plot(x, k_texts, 's-', color='#1e8449', linewidth=2, markersize=8,
                 label='5% dose (Exp 044)')

        ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Position Decile', fontsize=12)
        ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
        ax2.set_title('Text Accuracy: 5% vs 10% Dose on Qwen-Base', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(decile_labels, fontsize=9, rotation=45)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        plt.suptitle('Qwen3-4B-Base: Cross-Dose Comparison\n'
                     'Does 5% dose reveal accuracy gradients that 10% saturated?',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'cross_dose_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 5: K-signal norms by position (energy check) ──
    fig, ax = plt.subplots(figsize=(12, 6))

    k_sig_vals = []
    for d in range(10):
        key = f"decile_k_{d}"
        if key in agg:
            k_sig_vals.append(agg[key].get('k_signal_rms', 0))
        else:
            k_sig_vals.append(float('nan'))

    ax.bar(x, k_sig_vals, 0.5, color='#e74c3c', alpha=0.7, label='K-signal RMS')

    if v9_key in agg:
        v_sig = agg[v9_key].get('v_signal_rms', 0)
        ax.bar([9.6], [v_sig], 0.3, color='#3498db', alpha=0.7, label='V-signal RMS (bin 9)')

    ax.set_xlabel('Position Decile', fontsize=12)
    ax.set_ylabel('Signal RMS', fontsize=12)
    ax.set_title('KV Cache Signal Norms by Position (5% dose, Qwen-Base)\n'
                 '(Energy check: are norms uniform across positions?)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(decile_labels, fontsize=10, rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'signal_norms_by_position.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 6: 2x2 Matrix (model x dose) ──
    if exp043_agg and exp042_agg:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)

        # Load Exp 041 (Llama 10%) if available
        exp041_path = os.path.join(os.path.dirname(results_dir), 'exp_041', 'summary.json')
        exp041_agg = None
        if os.path.exists(exp041_path):
            with open(exp041_path, 'r') as f:
                exp041_agg = json.load(f).get('aggregated', None)

        datasets = [
            (axes[0, 0], exp042_agg, 'Qwen-Base 10% dose (Exp 042)', '#8e44ad'),
            (axes[0, 1], agg, 'Qwen-Base 5% dose (Exp 044)', '#e74c3c'),
            (axes[1, 0], exp041_agg, 'Llama-8B 10% dose (Exp 041)', '#2980b9'),
            (axes[1, 1], exp043_agg, 'Llama-8B 5% dose (Exp 043)', '#27ae60'),
        ]

        for ax, data, title, color in datasets:
            if data is None:
                ax.set_title(title + '\n(data not available)', fontsize=11)
                continue

            accs = [data.get(f"decile_k_{d}", {}).get('accuracy', float('nan')) * 100 for d in range(10)]
            texts = [data.get(f"decile_k_{d}", {}).get('text_accuracy', float('nan')) * 100 for d in range(10)]

            ax.plot(x, accs, 'o-', color=color, linewidth=2, markersize=7, label='Answer Acc')
            ax.plot(x, texts, 's--', color=color, linewidth=1.5, markersize=6, alpha=0.6, label='Text Acc')

            v9 = data.get("decile_v_9", {})
            if v9:
                ax.plot([9], [v9.get('accuracy', 0) * 100], 'D', color='gold',
                       markersize=10, zorder=6, label='V-only')

            ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.2)
            ax.set_ylim(-5, 115)

        for ax in axes[1, :]:
            ax.set_xlabel('Position Decile', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(decile_labels, fontsize=8, rotation=45)
        for ax in axes[:, 0]:
            ax.set_ylabel('Accuracy (%)', fontsize=11)

        plt.suptitle('2x2 Matrix: Model (Digital vs Analog) x Dose (10% vs 5%)\n'
                     'Does digital encoding create accuracy gradients at lower dose?',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'model_dose_matrix.png'),
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
        for cond_name, decile, component in CONDITIONS:
            positions = select_decile_positions_subsampled(
                reasoning_len, decile, fraction=SUBSAMPLE_FRACTION,
                rng_seed=SEED + ds_idx * 100 + decile)
            if not positions:
                continue

            mean_pos_frac = np.mean([p / max(reasoning_len - 1, 1) for p in positions])

            pc = build_prompt_cache(model, prompt_ids)
            ev = evaluate_condition(
                model, tokenizer, pc, reasoning_ids_truncated,
                positions, prompt_len, num_layers, true_answer,
                perturb_component=component)

            print(f"    {cond_name}: acc={'Y' if ev['correct'] else 'N'}, "
                  f"text={ev['text_accuracy']:.1%}, n_pos={len(positions)}/{reasoning_len}, "
                  f"frac={len(positions)/reasoning_len:.1%}")

            problem_result['evaluations'][cond_name] = {
                'correct': ev['correct'],
                'answer': ev['answer'],
                'text_accuracy': ev['text_accuracy'],
                'mean_pos_frac': float(mean_pos_frac),
                'n_noised': len(positions),
                'n_total_positions': reasoning_len,
                'effective_dose': len(positions) / reasoning_len,
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
        'experiment': 'exp_044_qwen_5pct_positional_sweep',
        'model': MODEL_NAME,
        'dose': '5% (50% sub-sample of each decile)',
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
        doses = [p['evaluations'][cond_name].get('effective_dose', 0.05)
                 for p in results if cond_name in p.get('evaluations', {})]
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
                'mean_effective_dose': float(np.mean(doses)),
                'k_rms': float(np.mean(k_rms_list)),
                'v_rms': float(np.mean(v_rms_list)),
                'k_signal_rms': float(np.mean(k_sig_list)),
                'v_signal_rms': float(np.mean(v_sig_list)),
            }

    # Print summary table
    print(f"\n{'Condition':<16} {'Acc%':>6} {'Text%':>6} {'AccDrop':>8} {'TxtDrop':>8} "
          f"{'Dissoc':>8} {'Dose':>6} {'n':>3} {'Wilson 95% CI':>18}")
    print("-" * 110)
    for cond_tuple in CONDITIONS:
        cond_name = cond_tuple[0]
        if cond_name in agg:
            d = agg[cond_name]
            print(f"  {cond_name:<14} {d['accuracy']*100:>5.1f}% {d['text_accuracy']*100:>5.1f}% "
                  f"{d['accuracy_drop']*100:>7.1f}% {d['text_drop']*100:>7.1f}% "
                  f"{d['dissociation']*100:>7.1f}% {d['mean_effective_dose']*100:>5.1f}% {d['n']:>3}"
                  f" [{d['wilson_ci_lo']*100:>5.1f}, {d['wilson_ci_hi']*100:>5.1f}]")

    # ── KEY ANALYSIS ──
    print(f"\n{'='*60}")
    print(f"KEY ANALYSIS: 5% DOSE POSITIONAL PROFILE ON QWEN-BASE")
    print(f"{'='*60}")

    # Check for accuracy gradient
    acc_by_bin = []
    text_by_bin = []
    for d in range(10):
        key = f"decile_k_{d}"
        if key in agg:
            acc_by_bin.append(agg[key]['accuracy'] * 100)
            text_by_bin.append(agg[key]['text_accuracy'] * 100)

    if len(acc_by_bin) == 10:
        x_vals = np.arange(10)
        acc_slope = np.polyfit(x_vals, acc_by_bin, 1)[0]
        text_slope = np.polyfit(x_vals, text_by_bin, 1)[0]
        acc_r = np.corrcoef(x_vals, acc_by_bin)[0, 1]
        text_r = np.corrcoef(x_vals, text_by_bin)[0, 1]

        print(f"\n  ACCURACY gradient: slope={acc_slope:.1f}pp/bin, r={acc_r:.3f}")
        print(f"  TEXT gradient: slope={text_slope:.1f}pp/bin, r={text_r:.3f}")

        # Compare with Exp 043 (Llama 5%)
        print(f"\n  Comparison with Llama 5% (Exp 043):")
        print(f"    Llama acc slope: 1.0pp/bin, r=0.639")
        print(f"    Qwen acc slope: {acc_slope:.1f}pp/bin, r={acc_r:.3f}")
        if acc_slope > 3.0 and acc_r > 0.7:
            print(f"  --> DIGITAL ENCODING CREATES ACCURACY GRADIENT (steeper than Llama)")
        elif acc_slope > 1.5:
            print(f"  --> MODEST gradient, slightly stronger than Llama")
        else:
            print(f"  --> NO meaningful gradient (similar to Llama's flat pattern)")

        # Coarse summary for comparison with Exp 042 (10%)
        early_acc = np.mean(acc_by_bin[:3])
        mid_acc = np.mean(acc_by_bin[3:7])
        late_acc = np.mean(acc_by_bin[7:])
        print(f"\n  Coarse summary:")
        print(f"    Early (0-30%): acc={early_acc:.1f}%  [Exp 042 10%: 1.3%]  [Llama 5%: 1.8%]")
        print(f"    Mid   (30-70%): acc={mid_acc:.1f}%  [Exp 042 10%: 1.0%]  [Llama 5%: 1.3%]")
        print(f"    Late  (70-100%): acc={late_acc:.1f}%  [Exp 042 10%: 2.7%]  [Llama 5%: 8.8%]")

        # Check for monotonicity
        is_monotonic = all(acc_by_bin[i+1] >= acc_by_bin[i] for i in range(9))
        print(f"\n  Monotonically increasing: {is_monotonic}")

        late_early_gap = late_acc - early_acc
        bin0_ci_width = (agg['decile_k_0']['wilson_ci_hi'] - agg['decile_k_0']['wilson_ci_lo']) * 100
        bin9_ci_width = (agg['decile_k_9']['wilson_ci_hi'] - agg['decile_k_9']['wilson_ci_lo']) * 100
        print(f"  Late - Early gap: {late_early_gap:+.1f}pp")
        print(f"  Bin 0 CI width: {bin0_ci_width:.1f}pp, Bin 9 CI width: {bin9_ci_width:.1f}pp")

        if late_early_gap > max(bin0_ci_width, bin9_ci_width):
            print(f"  --> GAP EXCEEDS CI WIDTH: accuracy gradient IS significant")
        else:
            print(f"  --> Gap smaller than CI width: accuracy gradient may not be significant")

    # V-only control
    print(f"\n  --- V-Only Control (bin 9) ---")
    v9_key = "decile_v_9"
    k9_key = "decile_k_9"
    if v9_key in agg:
        v9 = agg[v9_key]
        print(f"  V-only: acc={v9['accuracy']*100:.1f}% [{v9['wilson_ci_lo']*100:.1f}, {v9['wilson_ci_hi']*100:.1f}]")
        print(f"  [Exp 042 Qwen 10%: 56.0%]  [Exp 043 Llama 5%: 92.1%]")
        if k9_key in agg:
            k9 = agg[k9_key]
            print(f"  K-only: acc={k9['accuracy']*100:.1f}% [{k9['wilson_ci_lo']*100:.1f}, {k9['wilson_ci_hi']*100:.1f}]")
            gap = v9['accuracy']*100 - k9['accuracy']*100
            print(f"  V-K gap at bin 9: {gap:+.1f}pp")

    # Encoding comparison
    print(f"\n  --- Digital vs Analog Encoding Assessment ---")
    if len(acc_by_bin) == 10:
        max_acc = max(acc_by_bin)
        min_acc = min(acc_by_bin)
        print(f"  Qwen 5% accuracy range: {min_acc:.1f}% - {max_acc:.1f}%")
        print(f"  Llama 5% accuracy range: 0.0% - 15.8%")

        if max_acc > 20:
            print(f"  --> DIGITAL ENCODING IS PROTECTIVE: Qwen shows higher accuracy at some positions")
        elif max_acc > 10:
            print(f"  --> MODEST protection: Qwen slightly better than Llama at some positions")
        else:
            print(f"  --> NO protection: digital encoding makes no difference at 5% dose")

    summary['aggregated'] = agg
    summary['elapsed_seconds'] = time.time() - t0

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=_convert)
    with open(os.path.join(RESULTS_DIR, 'per_problem.json'), 'w') as f:
        json.dump(results, f, indent=2, default=_convert)

    # Load comparison data
    exp043_agg = None
    exp043_path = os.path.join(os.path.dirname(RESULTS_DIR), 'exp_043', 'summary.json')
    if os.path.exists(exp043_path):
        with open(exp043_path, 'r') as f:
            exp043_agg = json.load(f).get('aggregated', None)
        print(f"\nLoaded Exp 043 (Llama 5%) data for cross-model comparison")

    exp042_agg = None
    exp042_path = os.path.join(os.path.dirname(RESULTS_DIR), 'exp_042', 'summary.json')
    if os.path.exists(exp042_path):
        with open(exp042_path, 'r') as f:
            exp042_agg = json.load(f).get('aggregated', None)
        print(f"Loaded Exp 042 (Qwen 10%) data for cross-dose comparison")

    print("\nGenerating figures...")
    generate_figures(results, agg, RESULTS_DIR, exp043_agg, exp042_agg)
    print(f"\nDone. Total: {summary['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
