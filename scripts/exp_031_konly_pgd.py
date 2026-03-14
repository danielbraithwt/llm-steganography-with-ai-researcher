#!/usr/bin/env python3
"""
Experiment 031: K-only vs V-only vs K+V PGD Null Space Attack

Tests whether the adversarial null space lives primarily in K-space (routing)
or V-space (throughput). This unifies the two central findings:
  - The null space exists (Exp 4): adversarial perturbations change answer, preserve text
  - K > V (Exps 23-29): K routing is far more critical than V throughput for answers

If K-only PGD succeeds at changing answers while preserving text, the null space
is a routing phenomenon. If V-only PGD also succeeds, the null space spans both.

Model: Qwen3-4B-Base (where PGD works — 100% success in Exp 4)
Method: Prompt-only PGD attack, three conditions:
  1. K-only: optimize delta_k, delta_v = 0
  2. V-only: optimize delta_v, delta_k = 0
  3. K+V: optimize both (baseline replication of Exp 4/18)
"""

import os
import json
import time
import random
import gc
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B-Base"
NUM_PROBLEMS = 80  # attempt many, time-limited
MAX_GEN_TOKENS = 512
MAX_REASONING_TOKENS = 250
MAX_SEQ_LEN = 1536
PGD_STEPS = 60
PGD_LR = 0.05
LAMBDA_ANS = 5.0
MAX_PERT_RATIO = 500.0
ANSWER_REGION_N = 10
CHUNK_SIZE = 128
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_031")
TIMEOUT = 1700  # seconds (leave buffer for analysis)

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

# PGD conditions: (name, optimize_k, optimize_v)
PGD_CONDITIONS = [
    ("k_only", True, False),
    ("v_only", False, True),
    ("kv_both", True, True),
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


def pgd_attack(model, clean_prompt_keys, clean_prompt_values,
               reasoning_token_ids, clean_logits, num_layers,
               answer_start_pos, optimize_k=True, optimize_v=True,
               pgd_steps=PGD_STEPS, lr=PGD_LR):
    """
    PGD attack with optional K-only or V-only optimization.

    optimize_k: if True, create trainable delta_k
    optimize_v: if True, create trainable delta_v
    At least one must be True.
    """
    device = clean_prompt_keys[0].device
    dtype_model = clean_prompt_keys[0].dtype

    delta_k = []
    delta_v = []
    params = []

    for l in range(num_layers):
        dk = torch.zeros(clean_prompt_keys[l].shape,
                        device=device, dtype=torch.float32)
        if optimize_k:
            dk.requires_grad_(True)
            params.append(dk)
        delta_k.append(dk)

        dv = torch.zeros(clean_prompt_values[l].shape,
                        device=device, dtype=torch.float32)
        if optimize_v:
            dv.requires_grad_(True)
            params.append(dv)
        delta_v.append(dv)

    optimizer = torch.optim.Adam(params, lr=lr)
    step_metrics = []

    for step in range(pgd_steps):
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
            if optimize_k:
                for l in range(num_layers):
                    sig_k = clean_prompt_keys[l].float().norm().item()
                    max_k = MAX_PERT_RATIO * sig_k
                    cur_k = delta_k[l].norm().item()
                    if cur_k > max_k:
                        delta_k[l].mul_(max_k / cur_k)

            if optimize_v:
                for l in range(num_layers):
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
        }
        step_metrics.append(sm)

        if step % 15 == 0:
            with torch.no_grad():
                clean_preds = clean_logits[:, :answer_start_pos, :].argmax(dim=-1)
                pert_preds = perturbed_logits[:, :answer_start_pos, :].argmax(dim=-1)
                text_match = (clean_preds == pert_preds).float().mean().item()
            print(f"    Step {step:3d}: loss={sm['loss']:.4f}, "
                  f"text_KL={sm['text_kl']:.4f}, answer_KL={sm['answer_kl']:.4f}, "
                  f"text_match={text_match:.1%}")

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

    # Perturbation norms
    k_pert_norms = [dk.norm().item() for dk in delta_k]
    v_pert_norms = [dv.norm().item() for dv in delta_v]
    k_signal_norms = [ck.float().norm().item() for ck in clean_prompt_keys]
    v_signal_norms = [cv.float().norm().item() for cv in clean_prompt_values]
    k_ratios = [p / (s + 1e-8) for p, s in zip(k_pert_norms, k_signal_norms)]
    v_ratios = [p / (s + 1e-8) for p, s in zip(v_pert_norms, v_signal_norms)]

    mean_k_ratio = float(np.mean(k_ratios)) if any(r > 0 for r in k_ratios) else 0.0
    mean_v_ratio = float(np.mean(v_ratios)) if any(r > 0 for r in v_ratios) else 0.0

    clean_norm_ans = normalize_answer(clean_answer) if clean_answer else ""
    answer_changed = (pert_norm_ans != clean_norm_ans) and pert_norm_ans != ""
    # Strict success: answer changed AND text preserved >= 90%
    attack_success = answer_changed and (text_match_rate >= 0.90)
    # Also track a looser criterion
    attack_success_loose = answer_changed and (text_match_rate >= 0.80)

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
        'attack_success': attack_success,
        'attack_success_loose': attack_success_loose,
        'mean_k_pert_ratio': mean_k_ratio,
        'mean_v_pert_ratio': mean_v_ratio,
        'k_total_pert_norm': float(np.sum(k_pert_norms)),
        'v_total_pert_norm': float(np.sum(v_pert_norms)),
    }


def generate_figures(results_by_condition, results_dir):
    """Generate comparison figures across K-only, V-only, K+V conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(results_by_condition.keys())
    colors = {'k_only': '#e74c3c', 'v_only': '#3498db', 'kv_both': '#2ecc71'}
    labels = {'k_only': 'K-only PGD', 'v_only': 'V-only PGD', 'kv_both': 'K+V PGD'}

    # ── Figure 1: Attack success rates ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

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
    ax.set_xticklabels([f"{labels[c]}\n(n={n_attacks[i]})" for i, c in enumerate(conditions)])
    ax.set_ylabel('Rate (%)')
    ax.set_title('PGD Attack Success by Component')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
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
    ax.legend(fontsize=8)
    ax.axvline(x=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    ax.grid(True, alpha=0.2)

    # 1c: Perturbation norms
    ax = axes[2]
    for i, cond in enumerate(conditions):
        attacks = results_by_condition[cond]
        if attacks:
            k_norms = [a['mean_k_pert_ratio'] for a in attacks]
            v_norms = [a['mean_v_pert_ratio'] for a in attacks]
            if any(n > 0 for n in k_norms):
                ax.boxplot([k_norms], positions=[i * 3], widths=0.6,
                          patch_artist=True,
                          boxprops=dict(facecolor=colors[cond], alpha=0.5))
            if any(n > 0 for n in v_norms):
                ax.boxplot([v_norms], positions=[i * 3 + 1], widths=0.6,
                          patch_artist=True,
                          boxprops=dict(facecolor=colors[cond], alpha=0.8))
    ax.set_xticks([0, 1, 3, 4, 6, 7])
    ax.set_xticklabels(['K-only\nK', 'K-only\nV', 'V-only\nK', 'V-only\nV',
                        'K+V\nK', 'K+V\nV'], fontsize=8)
    ax.set_ylabel('Perturbation / Signal Ratio')
    ax.set_title('Perturbation Norm by Component')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "attack_success_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Figure 2: Per-problem comparison ──
    # For problems that appear in all 3 conditions, compare
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Collect per-problem data
    problem_data = {}
    for cond in conditions:
        for a in results_by_condition[cond]:
            pid = a.get('problem_idx', -1)
            if pid not in problem_data:
                problem_data[pid] = {}
            problem_data[pid][cond] = a

    common_pids = [pid for pid in problem_data
                   if all(c in problem_data[pid] for c in conditions)]

    if common_pids:
        # 2a: text_match scatter K-only vs V-only
        ax = axes[0]
        k_text = [problem_data[p]['k_only']['text_match_rate'] * 100 for p in common_pids]
        v_text = [problem_data[p]['v_only']['text_match_rate'] * 100 for p in common_pids]
        ax.scatter(k_text, v_text, alpha=0.6, color='purple', s=30)
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3)
        ax.set_xlabel('K-only: Text Match Rate (%)')
        ax.set_ylabel('V-only: Text Match Rate (%)')
        ax.set_title(f'Text Preservation: K-only vs V-only\n(n={len(common_pids)} problems)')
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.2)

        # 2b: answer_changed scatter K-only vs V-only
        ax = axes[1]
        k_ans_kl = [problem_data[p]['k_only'].get('final_answer_kl', 0) for p in common_pids]
        v_ans_kl = [problem_data[p]['v_only'].get('final_answer_kl', 0) for p in common_pids]
        k_changed = [problem_data[p]['k_only']['answer_changed'] for p in common_pids]
        v_changed = [problem_data[p]['v_only']['answer_changed'] for p in common_pids]

        # Make a 2x2 table
        both = sum(1 for k, v in zip(k_changed, v_changed) if k and v)
        k_not_v = sum(1 for k, v in zip(k_changed, v_changed) if k and not v)
        v_not_k = sum(1 for k, v in zip(k_changed, v_changed) if not k and v)
        neither = sum(1 for k, v in zip(k_changed, v_changed) if not k and not v)

        table_data = [[both, k_not_v], [v_not_k, neither]]
        table_labels = [['Both\nchanged', 'K-only\nchanged'],
                       ['V-only\nchanged', 'Neither\nchanged']]
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, fill=True,
                             color=['#27ae60', '#e74c3c', '#3498db', '#95a5a6'][i * 2 + j],
                             alpha=0.3))
                ax.text(j + 0.5, 1.5 - i, f"{table_labels[i][j]}\n{table_data[i][j]}",
                       ha='center', va='center', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Answer Changed: K-only vs V-only\n(n={len(common_pids)})')
        ax.set_xlabel('K-only PGD')
        ax.set_ylabel('V-only PGD')

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "per_problem_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Figure 3: Training curves comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_names = ['loss', 'text_kl', 'answer_kl']
    metric_labels = ['Total Loss', 'Text KL (preserve)', 'Answer KL (disrupt)']

    for ax, metric, label in zip(axes, metric_names, metric_labels):
        for cond in conditions:
            attacks = results_by_condition[cond]
            if attacks:
                # Average training curve across attacks
                all_curves = []
                for a in attacks:
                    if 'training_curve' in a:
                        curve = [s[metric] for s in a['training_curve']]
                        all_curves.append(curve)
                if all_curves:
                    max_len = max(len(c) for c in all_curves)
                    padded = [c + [c[-1]] * (max_len - len(c)) for c in all_curves]
                    mean_curve = np.mean(padded, axis=0)
                    ax.plot(mean_curve, color=colors[cond], label=labels[cond], alpha=0.8)
        ax.set_xlabel('PGD Step')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
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
    print(f"=== Experiment 031: K-only vs V-only vs K+V PGD ===")
    print(f"Model: {MODEL_NAME}")
    print(f"PGD steps: {PGD_STEPS}, lr: {PGD_LR}, lambda_ans: {LAMBDA_ANS}")
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

    # Collect valid problems first (clean answer must be correct)
    valid_problems = []
    print(f"\nPhase 1: Validating problems...")

    for idx in indices:
        if len(valid_problems) >= NUM_PROBLEMS:
            break
        if time.time() - start_time > TIMEOUT * 0.3:
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

        # Truncate at #### for teacher-forcing
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
        if elapsed > TIMEOUT * 0.85:
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

        for cond_name, opt_k, opt_v in PGD_CONDITIONS:
            elapsed = time.time() - start_time
            if elapsed > TIMEOUT * 0.85:
                break

            print(f"  {cond_name} (optimize_k={opt_k}, optimize_v={opt_v}):")
            attack_start = time.time()

            delta_k, delta_v, step_metrics = pgd_attack(
                model, clean_prompt_keys, clean_prompt_values,
                reasoning_token_ids, clean_logits, num_layers,
                answer_start_pos, optimize_k=opt_k, optimize_v=opt_v)

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
                  f"success={eval_result['attack_success']}, "
                  f"pert_answer={eval_result['perturbed_answer']}, "
                  f"K_ratio={eval_result['mean_k_pert_ratio']:.2f}, "
                  f"V_ratio={eval_result['mean_v_pert_ratio']:.2f}, "
                  f"time={attack_time:.1f}s")

            results_by_condition[cond_name].append(eval_result)

            # Clean up
            del delta_k, delta_v
            gc.collect()
            torch.cuda.empty_cache()

        problems_completed += 1

        # Clean up problem-level tensors
        del clean_prompt_keys, clean_prompt_values, clean_logits, reasoning_token_ids
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 3: Analyze and save
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Problems completed: {problems_completed}")

    summary = {}
    for cond_name, _, _ in PGD_CONDITIONS:
        attacks = results_by_condition[cond_name]
        n = len(attacks)
        if n == 0:
            summary[cond_name] = {'n': 0}
            continue

        n_success_strict = sum(1 for a in attacks if a['attack_success'])
        n_success_loose = sum(1 for a in attacks if a['attack_success_loose'])
        n_changed = sum(1 for a in attacks if a['answer_changed'])
        mean_text = np.mean([a['text_match_rate'] for a in attacks])
        mean_ans_match = np.mean([a['answer_match_rate'] for a in attacks])
        mean_k_ratio = np.mean([a['mean_k_pert_ratio'] for a in attacks])
        mean_v_ratio = np.mean([a['mean_v_pert_ratio'] for a in attacks])

        s = {
            'n': n,
            'n_success_strict': n_success_strict,
            'success_rate_strict': n_success_strict / n * 100,
            'n_success_loose': n_success_loose,
            'success_rate_loose': n_success_loose / n * 100,
            'n_answer_changed': n_changed,
            'answer_changed_rate': n_changed / n * 100,
            'mean_text_match': mean_text * 100,
            'mean_answer_match': mean_ans_match * 100,
            'mean_k_pert_ratio': mean_k_ratio,
            'mean_v_pert_ratio': mean_v_ratio,
        }
        summary[cond_name] = s

        print(f"\n{cond_name} (n={n}):")
        print(f"  Success (strict >=90%): {n_success_strict}/{n} = {s['success_rate_strict']:.1f}%")
        print(f"  Success (loose >=80%):  {n_success_loose}/{n} = {s['success_rate_loose']:.1f}%")
        print(f"  Answer changed:         {n_changed}/{n} = {s['answer_changed_rate']:.1f}%")
        print(f"  Mean text match:        {s['mean_text_match']:.1f}%")
        print(f"  Mean answer match:      {s['mean_answer_match']:.1f}%")
        print(f"  Mean K pert/signal:     {mean_k_ratio:.2f}")
        print(f"  Mean V pert/signal:     {mean_v_ratio:.2f}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(results_by_condition, RESULTS_DIR)

    # Save results
    # Strip training curves for the main results file to keep it small
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
