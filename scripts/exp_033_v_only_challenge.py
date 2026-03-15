#!/usr/bin/env python3
"""
Experiment 033: Maximum-Effort V-only PGD — Challenging the K-only Null Space Claim

DISCONFIRMATION EXPERIMENT: The last 3+ experiments confirmed K>V. This experiment
gives V-only PGD its absolute best shot to succeed. Three conditions:

1. V-only FULL (all layers): 100 PGD steps, lr=0.08, lambda_ans=10.0
2. V-only LATE (layers 18+ only): Same, restricted to late layers where answer
   computation concentrates — V's best chance since answer info is in late layers
3. K-only FULL (all layers): Same hyperparameters, for direct comparison

Key differences from Exp 031/032:
- 100 PGD steps (vs 60): more optimization budget for V
- lr=0.08 (vs 0.05): faster convergence
- lambda_ans=10.0 (vs 5.0): stronger answer-disruption signal
- Late-layer V-only: focuses perturbation where answer computation happens
- Cosine LR schedule: helps escape local minima

If V-only PGD still fails at 0%: the K-only null space claim is MUCH stronger
If V-only PGD succeeds even once: challenges "null space is K-only"

Model: Qwen3-4B-Base (where PGD works)
"""

import os
import json
import time
import random
import gc
import re
import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B-Base"
NUM_PROBLEMS = 60  # attempt many, time-limited
MAX_GEN_TOKENS = 512
MAX_REASONING_TOKENS = 250
MAX_SEQ_LEN = 1536
PGD_STEPS = 100
PGD_LR = 0.08
LAMBDA_ANS = 10.0
MAX_PERT_RATIO = 500.0
ANSWER_REGION_N = 10
CHUNK_SIZE = 128
SEED = 42
LATE_LAYER_START = 18  # layers 18-35 for "late" condition
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_033")
TIMEOUT = 1700  # seconds

os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 8-shot GSM8K exemplars ──────────────────────────────────────────────
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

# PGD conditions: (name, optimize_k, optimize_v, late_only)
PGD_CONDITIONS = [
    ("v_only_full",  False, True,  False),
    ("v_only_late",  False, True,  True),
    ("k_only_full",  True,  False, False),
]


def build_prompt(question):
    prompt = ""
    for ex in EXEMPLARS:
        prompt += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def extract_answer(text):
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = ans.replace(",", "").replace("$", "").strip()
        match = re.match(r'^-?[\d.]+', ans)
        if match:
            return match.group(0)
        return ans
    return ""


def normalize_answer(ans):
    ans = ans.strip().replace(",", "").replace("$", "")
    try:
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans


@torch.no_grad()
def generate_trace(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS):
    """Generate a CoT trace with greedy decoding."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = []
    past_kv = None
    current_input = inputs.input_ids

    for step in range(max_tokens):
        if past_kv is not None:
            outputs = model(input_ids=current_input, past_key_values=past_kv,
                          use_cache=True)
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
                generated_ids = tokenizer.encode(
                    current_text[:idx], add_special_tokens=False)
            break

        current_input = next_token

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    del past_kv, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return generated_text


@torch.no_grad()
def teacher_force_chunked(model, input_ids, chunk_size=CHUNK_SIZE):
    """Teacher-force a sequence in chunks, returning the KV cache."""
    seq_len = input_ids.shape[1]
    past_kv = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        if past_kv is not None:
            outputs = model(input_ids=chunk, past_key_values=past_kv,
                          use_cache=True)
        else:
            outputs = model(input_ids=chunk, use_cache=True)
        past_kv = outputs.past_key_values
        del outputs
    return past_kv


def find_answer_boundary(reasoning_len):
    return max(10, reasoning_len - ANSWER_REGION_N)


def cosine_lr(step, total_steps, base_lr, min_lr=0.005):
    """Cosine annealing learning rate schedule."""
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * step / total_steps))


def pgd_attack(model, clean_prompt_keys, clean_prompt_values,
               reasoning_token_ids, clean_logits, num_layers,
               answer_start_pos, optimize_k=True, optimize_v=True,
               late_only=False, pgd_steps=PGD_STEPS, lr=PGD_LR):
    """
    PGD attack with optional K-only or V-only, full or late-layer optimization.
    Uses cosine LR schedule for better convergence.
    """
    device = clean_prompt_keys[0].device
    dtype_model = clean_prompt_keys[0].dtype

    delta_k = []
    delta_v = []
    params = []

    for l in range(num_layers):
        # For late_only, only create trainable params for layers >= LATE_LAYER_START
        layer_trainable = (not late_only) or (l >= LATE_LAYER_START)

        dk = torch.zeros(clean_prompt_keys[l].shape,
                        device=device, dtype=torch.float32)
        if optimize_k and layer_trainable:
            dk.requires_grad_(True)
            params.append(dk)
        delta_k.append(dk)

        dv = torch.zeros(clean_prompt_values[l].shape,
                        device=device, dtype=torch.float32)
        if optimize_v and layer_trainable:
            dv.requires_grad_(True)
            params.append(dv)
        delta_v.append(dv)

    optimizer = torch.optim.Adam(params, lr=lr)
    step_metrics = []

    for step in range(pgd_steps):
        # Cosine LR schedule
        current_lr = cosine_lr(step, pgd_steps, lr)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        optimizer.zero_grad()

        perturbed_cache = DynamicCache()
        for l in range(num_layers):
            pk = clean_prompt_keys[l] + delta_k[l].to(dtype_model)
            pv = clean_prompt_values[l] + delta_v[l].to(dtype_model)
            perturbed_cache.update(pk, pv, l)

        outputs = model(input_ids=reasoning_token_ids,
                       past_key_values=perturbed_cache,
                       use_cache=False)
        perturbed_logits = outputs.logits

        # Text preservation loss (minimize KL divergence in text region)
        text_log_p = F.log_softmax(perturbed_logits[:, :answer_start_pos, :].float(), dim=-1)
        text_q = F.softmax(clean_logits[:, :answer_start_pos, :].float(), dim=-1).detach()
        text_kl = F.kl_div(text_log_p, text_q, reduction='batchmean')

        # Answer disruption loss (maximize KL divergence in answer region)
        ans_log_p = F.log_softmax(perturbed_logits[:, answer_start_pos:, :].float(), dim=-1)
        ans_q = F.softmax(clean_logits[:, answer_start_pos:, :].float(), dim=-1).detach()
        answer_kl = F.kl_div(ans_log_p, ans_q, reduction='batchmean')

        loss = text_kl - LAMBDA_ANS * answer_kl
        loss.backward()
        optimizer.step()

        # Project: enforce max perturbation ratio
        with torch.no_grad():
            for l in range(num_layers):
                if optimize_k and delta_k[l].requires_grad:
                    sig_k = clean_prompt_keys[l].float().norm().item()
                    max_k = MAX_PERT_RATIO * sig_k
                    cur_k = delta_k[l].norm().item()
                    if cur_k > max_k:
                        delta_k[l].mul_(max_k / cur_k)

                if optimize_v and delta_v[l].requires_grad:
                    sig_v = clean_prompt_values[l].float().norm().item()
                    max_v = MAX_PERT_RATIO * sig_v
                    cur_v = delta_v[l].norm().item()
                    if cur_v > max_v:
                        delta_v[l].mul_(max_v / cur_v)

        sm = {
            'step': step,
            'loss': loss.item(),
            'text_kl': text_kl.item(),
            'answer_kl': answer_kl.item(),
            'lr': current_lr,
        }
        step_metrics.append(sm)

        if step % 20 == 0:
            with torch.no_grad():
                clean_preds = clean_logits[:, :answer_start_pos, :].argmax(dim=-1)
                pert_preds = perturbed_logits[:, :answer_start_pos, :].argmax(dim=-1)
                text_match = (clean_preds == pert_preds).float().mean().item()
            print(f"    Step {step:3d}: loss={sm['loss']:.4f}, "
                  f"text_KL={sm['text_kl']:.4f}, answer_KL={sm['answer_kl']:.4f}, "
                  f"text_match={text_match:.1%}, lr={current_lr:.4f}")

        del perturbed_cache, outputs, perturbed_logits, loss
        del text_log_p, text_q, ans_log_p, ans_q

    gc.collect()
    torch.cuda.empty_cache()
    return delta_k, delta_v, step_metrics


@torch.no_grad()
def evaluate_attack(model, tokenizer, clean_prompt_keys, clean_prompt_values,
                    delta_k, delta_v, reasoning_token_ids, clean_logits,
                    true_answer, clean_answer, num_layers, answer_start_pos):
    """Evaluate PGD attack success."""
    dtype_model = clean_prompt_keys[0].dtype

    perturbed_cache = DynamicCache()
    for l in range(num_layers):
        pk = clean_prompt_keys[l] + delta_k[l].to(dtype_model)
        pv = clean_prompt_values[l] + delta_v[l].to(dtype_model)
        perturbed_cache.update(pk, pv, l)

    eval_out = model(input_ids=reasoning_token_ids,
                    past_key_values=perturbed_cache, use_cache=True)
    perturbed_logits = eval_out.logits

    # Token match rates
    clean_preds = clean_logits[:, :answer_start_pos, :].argmax(dim=-1)
    pert_preds = perturbed_logits[:, :answer_start_pos, :].argmax(dim=-1)
    text_match_rate = (clean_preds == pert_preds).float().mean().item()

    all_clean = clean_logits.argmax(dim=-1)
    all_pert = perturbed_logits.argmax(dim=-1)
    all_match_rate = (all_clean == all_pert).float().mean().item()

    ans_clean = clean_logits[:, answer_start_pos:, :].argmax(dim=-1)
    ans_pert = perturbed_logits[:, answer_start_pos:, :].argmax(dim=-1)
    answer_match_rate = (ans_clean == ans_pert).float().mean().item()

    # Generate answer from perturbed cache
    gen_cache = eval_out.past_key_values
    next_token = perturbed_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token[0, 0].item()]

    for _ in range(150):
        gen_out = model(input_ids=next_token, past_key_values=gen_cache,
                       use_cache=True)
        gen_cache = gen_out.past_key_values
        next_token = gen_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = next_token[0, 0].item()
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if "####" in decoded:
            after = decoded.split("####")[-1]
            if re.search(r'\d+', after):
                break
        if "\nQ:" in decoded:
            break

    perturbed_answer_text = tokenizer.decode(generated, skip_special_tokens=True)
    perturbed_answer = extract_answer(perturbed_answer_text)
    pert_norm_ans = normalize_answer(perturbed_answer) if perturbed_answer else ""

    # Perturbation norms — per-layer breakdown
    k_pert_norms = [dk.norm().item() for dk in delta_k]
    v_pert_norms = [dv.norm().item() for dv in delta_v]
    k_signal_norms = [ck.float().norm().item() for ck in clean_prompt_keys]
    v_signal_norms = [cv.float().norm().item() for cv in clean_prompt_values]
    k_ratios = [p / (s + 1e-8) for p, s in zip(k_pert_norms, k_signal_norms)]
    v_ratios = [p / (s + 1e-8) for p, s in zip(v_pert_norms, v_signal_norms)]

    # Late-layer-specific norms
    k_late_norms = [k_pert_norms[l] for l in range(LATE_LAYER_START, len(k_pert_norms))]
    v_late_norms = [v_pert_norms[l] for l in range(LATE_LAYER_START, len(v_pert_norms))]
    k_late_signal = [k_signal_norms[l] for l in range(LATE_LAYER_START, len(k_signal_norms))]
    v_late_signal = [v_signal_norms[l] for l in range(LATE_LAYER_START, len(v_signal_norms))]
    k_late_ratios = [p / (s + 1e-8) for p, s in zip(k_late_norms, k_late_signal)]
    v_late_ratios = [p / (s + 1e-8) for p, s in zip(v_late_norms, v_late_signal)]

    mean_k_ratio = float(np.mean(k_ratios)) if any(r > 0 for r in k_ratios) else 0.0
    mean_v_ratio = float(np.mean(v_ratios)) if any(r > 0 for r in v_ratios) else 0.0
    mean_k_late_ratio = float(np.mean(k_late_ratios)) if k_late_ratios and any(r > 0 for r in k_late_ratios) else 0.0
    mean_v_late_ratio = float(np.mean(v_late_ratios)) if v_late_ratios and any(r > 0 for r in v_late_ratios) else 0.0

    clean_norm_ans = normalize_answer(clean_answer) if clean_answer else ""
    answer_changed = (pert_norm_ans != clean_norm_ans) and pert_norm_ans != ""
    attack_success = answer_changed and (text_match_rate >= 0.90)
    attack_success_loose = answer_changed and (text_match_rate >= 0.80)

    # Classify the quality of the answer change
    answer_quality = "none"
    if answer_changed:
        try:
            pert_val = float(pert_norm_ans)
            answer_quality = "numeric_redirect"
        except (ValueError, TypeError):
            if len(pert_norm_ans) > 0 and any(c.isdigit() for c in pert_norm_ans):
                answer_quality = "partial_numeric"
            else:
                answer_quality = "garbage"

    del eval_out, gen_cache
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'text_match_rate': text_match_rate,
        'all_match_rate': all_match_rate,
        'answer_match_rate': answer_match_rate,
        'perturbed_answer': perturbed_answer,
        'perturbed_answer_text': perturbed_answer_text[:300],
        'answer_changed': answer_changed,
        'answer_quality': answer_quality,
        'attack_success': attack_success,
        'attack_success_loose': attack_success_loose,
        'mean_k_pert_ratio': mean_k_ratio,
        'mean_v_pert_ratio': mean_v_ratio,
        'mean_k_late_ratio': mean_k_late_ratio,
        'mean_v_late_ratio': mean_v_late_ratio,
        'k_total_pert_norm': float(np.sum(k_pert_norms)),
        'v_total_pert_norm': float(np.sum(v_pert_norms)),
        'per_layer_k_ratios': k_ratios,
        'per_layer_v_ratios': v_ratios,
    }


def generate_figures(results_by_condition, results_dir):
    """Generate comparison figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(results_by_condition.keys())
    colors = {'v_only_full': '#3498db', 'v_only_late': '#9b59b6', 'k_only_full': '#e74c3c'}
    labels = {'v_only_full': 'V-only (all layers)',
              'v_only_late': 'V-only (late layers)',
              'k_only_full': 'K-only (all layers)'}

    # ── Figure 1: Attack success rates & text preservation ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1a: Success rates
    ax = axes[0]
    success_strict = []
    success_loose = []
    answer_changed = []
    n_attacks = []
    for cond in conditions:
        attacks = results_by_condition[cond]
        n = len(attacks)
        n_attacks.append(n)
        if n > 0:
            success_strict.append(sum(1 for a in attacks if a['attack_success']) / n * 100)
            success_loose.append(sum(1 for a in attacks if a['attack_success_loose']) / n * 100)
            answer_changed.append(sum(1 for a in attacks if a['answer_changed']) / n * 100)
        else:
            success_strict.append(0)
            success_loose.append(0)
            answer_changed.append(0)

    x = np.arange(len(conditions))
    width = 0.25
    ax.bar(x - width, answer_changed, width, color=[colors[c] for c in conditions],
           alpha=0.4, label='Answer changed')
    ax.bar(x, success_loose, width, color=[colors[c] for c in conditions],
           alpha=0.7, label='Success (>=80% text)')
    ax.bar(x + width, success_strict, width, color=[colors[c] for c in conditions],
           label='Success (>=90% text)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{labels[c]}\n(n={n_attacks[i]})" for i, c in enumerate(conditions)],
                       fontsize=8)
    ax.set_ylabel('Rate (%)')
    ax.set_title('Disconfirmation Test: Can V-only PGD Change Answers?\n(100 steps, lr=0.08, λ_ans=10)')
    ax.legend(fontsize=7)
    ax.set_ylim(0, max(max(answer_changed) + 10, 15))
    ax.grid(True, alpha=0.2, axis='y')

    # 1b: Text preservation distribution
    ax = axes[1]
    for cond in conditions:
        attacks = results_by_condition[cond]
        if attacks:
            text_rates = [a['text_match_rate'] * 100 for a in attacks]
            ax.hist(text_rates, bins=20, alpha=0.5, color=colors[cond],
                    label=labels[cond], range=(0, 100))
    ax.set_xlabel('Text Match Rate (%)')
    ax.set_ylabel('Count')
    ax.set_title('Text Preservation Distribution')
    ax.legend(fontsize=7)
    ax.axvline(x=90, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.2)

    # 1c: Perturbation norms (K vs V)
    ax = axes[2]
    cond_labels = []
    k_norms_list = []
    v_norms_list = []
    for cond in conditions:
        attacks = results_by_condition[cond]
        if attacks:
            k_norms_list.append([a['mean_k_pert_ratio'] for a in attacks])
            v_norms_list.append([a['mean_v_pert_ratio'] for a in attacks])
            cond_labels.append(labels[cond])

    positions_k = np.arange(len(conditions)) * 3
    positions_v = positions_k + 1

    for i, cond in enumerate(conditions):
        if k_norms_list[i] and any(n > 0 for n in k_norms_list[i]):
            bp = ax.boxplot([k_norms_list[i]], positions=[positions_k[i]], widths=0.6,
                      patch_artist=True,
                      boxprops=dict(facecolor=colors[cond], alpha=0.4))
        if v_norms_list[i] and any(n > 0 for n in v_norms_list[i]):
            bp = ax.boxplot([v_norms_list[i]], positions=[positions_v[i]], widths=0.6,
                      patch_artist=True,
                      boxprops=dict(facecolor=colors[cond], alpha=0.8))

    xtick_positions = []
    xtick_labels_list = []
    for i, cond in enumerate(conditions):
        xtick_positions.extend([positions_k[i], positions_v[i]])
        short = labels[cond].split('(')[0].strip()
        xtick_labels_list.extend([f'{short}\nK', f'{short}\nV'])
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels_list, fontsize=7)
    ax.set_ylabel('Perturbation / Signal Ratio')
    ax.set_title('Perturbation Norm by Component')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "v_only_challenge_results.png"), dpi=150)
    plt.close(fig)

    # ── Figure 2: Per-layer perturbation distribution ──
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5))
    if len(conditions) == 1:
        axes = [axes]

    for ax_idx, cond in enumerate(conditions):
        ax = axes[ax_idx]
        attacks = results_by_condition[cond]
        if attacks:
            # Collect per-layer K and V ratios
            all_k_layers = np.array([a['per_layer_k_ratios'] for a in attacks])
            all_v_layers = np.array([a['per_layer_v_ratios'] for a in attacks])
            n_layers = all_k_layers.shape[1]
            layers = np.arange(n_layers)

            mean_k = np.mean(all_k_layers, axis=0)
            mean_v = np.mean(all_v_layers, axis=0)

            if np.any(mean_k > 0):
                ax.plot(layers, mean_k, color='red', alpha=0.7, label='K perturbation', linewidth=2)
            if np.any(mean_v > 0):
                ax.plot(layers, mean_v, color='blue', alpha=0.7, label='V perturbation', linewidth=2)

            if LATE_LAYER_START < n_layers:
                ax.axvline(x=LATE_LAYER_START, color='gray', linestyle='--', alpha=0.5,
                          label=f'Late cutoff (L{LATE_LAYER_START})')

            ax.set_xlabel('Layer')
            ax.set_ylabel('Pert/Signal Ratio')
            ax.set_title(f'{labels[cond]}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.set_yscale('symlog', linthresh=0.001)

    plt.suptitle('Per-Layer Perturbation Distribution', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "per_layer_perturbation.png"), dpi=150)
    plt.close(fig)

    # ── Figure 3: Training curves ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_names = ['loss', 'text_kl', 'answer_kl']
    metric_labels = ['Total Loss', 'Text KL (preserve)', 'Answer KL (disrupt)']

    for ax, metric, label in zip(axes, metric_names, metric_labels):
        for cond in conditions:
            attacks = results_by_condition[cond]
            if attacks:
                all_curves = []
                for a in attacks:
                    if 'training_curve' in a:
                        curve = [s[metric] for s in a['training_curve']]
                        all_curves.append(curve)
                if all_curves:
                    max_len = max(len(c) for c in all_curves)
                    padded = [c + [c[-1]] * (max_len - len(c)) for c in all_curves]
                    mean_curve = np.mean(padded, axis=0)
                    std_curve = np.std(padded, axis=0)
                    steps = np.arange(len(mean_curve))
                    ax.plot(steps, mean_curve, color=colors[cond], label=labels[cond], alpha=0.8)
                    ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                                   color=colors[cond], alpha=0.15)
        ax.set_xlabel('PGD Step')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

    # ── Figure 4: Answer quality breakdown ──
    fig, ax = plt.subplots(figsize=(10, 5))
    quality_cats = ['numeric_redirect', 'partial_numeric', 'garbage', 'none']
    quality_labels = ['Genuine numeric\nredirect', 'Partial numeric', 'Garbage', 'No change']
    quality_colors = ['#27ae60', '#f39c12', '#e74c3c', '#95a5a6']

    x = np.arange(len(conditions))
    bottom = np.zeros(len(conditions))
    for qi, qcat in enumerate(quality_cats):
        counts = []
        for cond in conditions:
            attacks = results_by_condition[cond]
            n = len(attacks)
            if n > 0:
                counts.append(sum(1 for a in attacks if a.get('answer_quality', 'none') == qcat) / n * 100)
            else:
                counts.append(0)
        ax.bar(x, counts, bottom=bottom, color=quality_colors[qi],
               label=quality_labels[qi], alpha=0.8)
        bottom += counts

    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in conditions], fontsize=9)
    ax.set_ylabel('Percentage of attacks (%)')
    ax.set_title('Answer Quality Breakdown: Disruption vs Redirection')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "answer_quality_breakdown.png"), dpi=150)
    plt.close(fig)

    print(f"Figures saved to {results_dir}")


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool):
        return bool(obj)
    return obj


def main():
    start_time = time.time()
    print(f"=== Experiment 033: Maximum-Effort V-only PGD Challenge ===")
    print(f"Model: {MODEL_NAME}")
    print(f"PGD steps: {PGD_STEPS}, lr: {PGD_LR}, lambda_ans: {LAMBDA_ANS}")
    print(f"Late-layer start: {LATE_LAYER_START}")
    print(f"Conditions: {[c[0] for c in PGD_CONDITIONS]}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager"
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers")

    # Load dataset
    print("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    random.shuffle(indices)

    # Collect valid problems
    valid_problems = []
    print(f"\nPhase 1: Validating problems...")

    for idx in indices:
        if len(valid_problems) >= NUM_PROBLEMS:
            break
        if time.time() - start_time > TIMEOUT * 0.25:
            print(f"Time limit for validation phase. Got {len(valid_problems)} problems.")
            break

        item = ds[idx]
        question = item['question']
        true_answer = normalize_answer(extract_answer(item['answer']))
        if not true_answer:
            continue

        prompt = build_prompt(question)
        trace = generate_trace(model, tokenizer, prompt)
        gen_answer = extract_answer(trace)
        gen_norm = normalize_answer(gen_answer) if gen_answer else ""

        if gen_norm != true_answer:
            continue

        if "####" not in trace:
            continue

        hash_idx = trace.index("####")
        reasoning_text = trace[:hash_idx].rstrip()
        full_text = prompt + " " + reasoning_text
        token_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)

        if token_ids.shape[1] > MAX_SEQ_LEN:
            continue

        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        prompt_len = prompt_ids.shape[1]
        reasoning_len = token_ids.shape[1] - prompt_len

        if reasoning_len < 20 or reasoning_len > MAX_REASONING_TOKENS:
            continue

        valid_problems.append({
            'idx': idx,
            'question': question,
            'true_answer': true_answer,
            'gen_answer': gen_norm,
            'trace': trace,
            'reasoning_text': reasoning_text,
            'token_ids': token_ids,
            'prompt_ids': prompt_ids,
            'prompt_len': prompt_len,
            'reasoning_len': reasoning_len,
        })
        print(f"  Valid problem {len(valid_problems)}: idx={idx}, "
              f"answer={true_answer}, reasoning_len={reasoning_len}")

    print(f"\nGot {len(valid_problems)} valid problems")
    if len(valid_problems) == 0:
        print("ERROR: No valid problems found!")
        return

    # Phase 2: Run PGD attacks
    results_by_condition = {c[0]: [] for c in PGD_CONDITIONS}
    problems_completed = 0

    print(f"\nPhase 2: Running PGD attacks ({len(PGD_CONDITIONS)} conditions per problem)...")

    for pi, prob in enumerate(valid_problems):
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT * 0.80:
            print(f"Time limit approaching. Completed {problems_completed} problems.")
            break

        print(f"\n--- Problem {pi+1}/{len(valid_problems)} (idx={prob['idx']}, "
              f"answer={prob['true_answer']}, r_len={prob['reasoning_len']}) ---")

        # Teacher-force to get clean prompt KV cache and logits
        token_ids = prob['token_ids']
        prompt_len = prob['prompt_len']

        # Get clean prompt KV
        prompt_kv = teacher_force_chunked(model, token_ids[:, :prompt_len])
        clean_prompt_keys = [prompt_kv.layers[l].keys.clone() for l in range(num_layers)]
        clean_prompt_values = [prompt_kv.layers[l].values.clone() for l in range(num_layers)]
        del prompt_kv

        # Get clean reasoning logits
        reasoning_token_ids = token_ids[:, prompt_len:]
        clean_cache = DynamicCache()
        for l in range(num_layers):
            clean_cache.update(clean_prompt_keys[l].clone(),
                             clean_prompt_values[l].clone(), l)
        clean_out = model(input_ids=reasoning_token_ids,
                        past_key_values=clean_cache, use_cache=False)
        clean_logits = clean_out.logits.detach()
        del clean_out, clean_cache
        gc.collect()
        torch.cuda.empty_cache()

        answer_start_pos = find_answer_boundary(prob['reasoning_len'])

        for cond_name, opt_k, opt_v, late_only in PGD_CONDITIONS:
            elapsed = time.time() - start_time
            if elapsed > TIMEOUT * 0.80:
                break

            late_str = " [LATE layers only]" if late_only else ""
            print(f"  {cond_name} (K={opt_k}, V={opt_v}){late_str}:")
            attack_start = time.time()

            delta_k, delta_v, step_metrics = pgd_attack(
                model, clean_prompt_keys, clean_prompt_values,
                reasoning_token_ids, clean_logits, num_layers,
                answer_start_pos, optimize_k=opt_k, optimize_v=opt_v,
                late_only=late_only)

            eval_result = evaluate_attack(
                model, tokenizer, clean_prompt_keys, clean_prompt_values,
                delta_k, delta_v, reasoning_token_ids, clean_logits,
                prob['true_answer'], prob['gen_answer'], num_layers,
                answer_start_pos)

            attack_time = time.time() - attack_start
            eval_result['problem_idx'] = prob['idx']
            eval_result['true_answer'] = prob['true_answer']
            eval_result['clean_answer'] = prob['gen_answer']
            eval_result['reasoning_len'] = prob['reasoning_len']
            eval_result['attack_time'] = attack_time
            eval_result['training_curve'] = step_metrics
            eval_result['final_text_kl'] = step_metrics[-1]['text_kl'] if step_metrics else 0
            eval_result['final_answer_kl'] = step_metrics[-1]['answer_kl'] if step_metrics else 0

            print(f"    Result: text_match={eval_result['text_match_rate']:.1%}, "
                  f"ans_changed={eval_result['answer_changed']}, "
                  f"quality={eval_result['answer_quality']}, "
                  f"success={eval_result['attack_success']}, "
                  f"pert_ans='{eval_result['perturbed_answer']}', "
                  f"K_ratio={eval_result['mean_k_pert_ratio']:.3f}, "
                  f"V_ratio={eval_result['mean_v_pert_ratio']:.3f}, "
                  f"time={attack_time:.1f}s")

            results_by_condition[cond_name].append(eval_result)

            del delta_k, delta_v
            gc.collect()
            torch.cuda.empty_cache()

        problems_completed += 1

        del clean_prompt_keys, clean_prompt_values, clean_logits, reasoning_token_ids
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 3: Analyze and save
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Problems completed: {problems_completed}")

    summary = {}
    for cond_name, _, _, _ in PGD_CONDITIONS:
        attacks = results_by_condition[cond_name]
        n = len(attacks)
        if n == 0:
            summary[cond_name] = {'n': 0}
            continue

        n_success_strict = sum(1 for a in attacks if a['attack_success'])
        n_success_loose = sum(1 for a in attacks if a['attack_success_loose'])
        n_changed = sum(1 for a in attacks if a['answer_changed'])
        n_numeric = sum(1 for a in attacks if a.get('answer_quality') == 'numeric_redirect')
        mean_text = np.mean([a['text_match_rate'] for a in attacks])
        mean_ans_match = np.mean([a['answer_match_rate'] for a in attacks])
        mean_k_ratio = np.mean([a['mean_k_pert_ratio'] for a in attacks])
        mean_v_ratio = np.mean([a['mean_v_pert_ratio'] for a in attacks])

        s = {
            'n': n,
            'n_success_strict': int(n_success_strict),
            'success_rate_strict': n_success_strict / n * 100,
            'n_success_loose': int(n_success_loose),
            'success_rate_loose': n_success_loose / n * 100,
            'n_answer_changed': int(n_changed),
            'answer_changed_rate': n_changed / n * 100,
            'n_numeric_redirect': int(n_numeric),
            'numeric_redirect_rate': n_numeric / n * 100,
            'mean_text_match': mean_text * 100,
            'mean_answer_match': mean_ans_match * 100,
            'mean_k_pert_ratio': float(mean_k_ratio),
            'mean_v_pert_ratio': float(mean_v_ratio),
        }
        summary[cond_name] = s

        print(f"\n{cond_name} (n={n}):")
        print(f"  Success (strict >=90%): {n_success_strict}/{n} = {s['success_rate_strict']:.1f}%")
        print(f"  Success (loose >=80%):  {n_success_loose}/{n} = {s['success_rate_loose']:.1f}%")
        print(f"  Answer changed:         {n_changed}/{n} = {s['answer_changed_rate']:.1f}%")
        print(f"  Numeric redirect:       {n_numeric}/{n} = {s['numeric_redirect_rate']:.1f}%")
        print(f"  Mean text match:        {s['mean_text_match']:.1f}%")
        print(f"  Mean answer match:      {s['mean_answer_match']:.1f}%")
        print(f"  Mean K pert/signal:     {mean_k_ratio:.3f}")
        print(f"  Mean V pert/signal:     {mean_v_ratio:.3f}")

    # Fisher exact test: K-only vs V-only (full)
    print("\n--- Statistical comparison ---")
    k_attacks = results_by_condition.get('k_only_full', [])
    v_attacks = results_by_condition.get('v_only_full', [])
    if k_attacks and v_attacks:
        k_changed = sum(1 for a in k_attacks if a['answer_changed'])
        k_not = len(k_attacks) - k_changed
        v_changed = sum(1 for a in v_attacks if a['answer_changed'])
        v_not = len(v_attacks) - v_changed
        print(f"K-only answer changes: {k_changed}/{len(k_attacks)}")
        print(f"V-only answer changes: {v_changed}/{len(v_attacks)}")

        # Simple one-sided Fisher test computation
        from scipy.stats import fisher_exact
        table = [[k_changed, k_not], [v_changed, v_not]]
        odds, p_val = fisher_exact(table, alternative='greater')
        print(f"Fisher exact test (K > V): p = {p_val:.4f}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(results_by_condition, RESULTS_DIR)

    # Save results
    results_for_save = {}
    for cond, attacks in results_by_condition.items():
        results_for_save[cond] = []
        for a in attacks:
            a_copy = {k: v for k, v in a.items() if k != 'training_curve'}
            results_for_save[cond].append(a_copy)

    output = {
        'summary': summary,
        'results_by_condition': results_for_save,
        'config': {
            'model': MODEL_NAME,
            'pgd_steps': PGD_STEPS,
            'lr': PGD_LR,
            'lambda_ans': LAMBDA_ANS,
            'max_pert_ratio': MAX_PERT_RATIO,
            'late_layer_start': LATE_LAYER_START,
            'seed': SEED,
            'n_problems_completed': problems_completed,
        },
        'elapsed_time': time.time() - start_time,
    }

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=_convert)
    print(f"\nResults saved to {results_path}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
