#!/usr/bin/env python3
"""
Experiment 018: Position-Controlled PGD Correlation Analysis

Tests whether the PGD perturbation-attention correlation (rho=0.78 from Exp 5)
survives partial correlation controlling for position. This is the SINGLE most
important remaining question: if the correlation is positional, ALL spatial
structure evidence collapses.

Approach:
1. Run prompt-only PGD attacks on Qwen3-4B-Base (established 100% success)
2. For each attack, measure at each REASONING position:
   - Effective perturbation: ||perturbed_reasoning_KV - clean_reasoning_KV||
   - Answer-token attention: how much the answer token attends to each position
   - Text-coupling attention: how much subsequent tokens attend to each position
   - Normalized position (0-1)
3. Compute bivariate + partial correlations + within-quartile correlations
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
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B-Base"
NUM_PROBLEMS = 25
MAX_GEN_TOKENS = 512
MAX_REASONING_TOKENS = 250
MAX_SEQ_LEN = 1536
PGD_STEPS = 50
PGD_LR = 0.05
LAMBDA_ANS = 5.0
MAX_PERT_RATIO = 500.0
ANSWER_REGION_N = 10
CHUNK_SIZE = 128
ATTENTION_LAYERS = [-1, -2, -3, -4]  # last 4 layers for AC/TC
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_018")

os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 8-shot GSM8K exemplars ──────────────────────────────────────────────
GSM8K_EXEMPLARS = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
     "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total bolts needed is 2+1=<<2+1=3>>3\n#### 3"},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
     "answer": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*150%=$<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"},
    {"question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "answer": "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624"},
    {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
     "answer": "If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.\nIf she feeds the flock 15 cups in the morning, and 25 cups in the afternoon, then the carry-over to the final meal would be 60-15-25=<<60-15-25=20>>20 cups.\n#### 20"},
    {"question": "Kylar went to the store to get water and some apples. The store sold apples for $1 each and water for $3 per bottle. Kylar wanted to buy one bag of apples and 2 bottles of water. How much would Kylar spend if each bag has 6 apples?",
     "answer": "A bag has 6 apples and each apple costs $1, so a bag costs 6*1=$<<6*1=6>>6\nKylar wants 2 bottles of water so that would cost 2*3=$<<2*3=6>>6\nAltogether, Kylar would spend 6+6=$<<6+6=12>>12\n#### 12"},
    {"question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
     "answer": "If Seattle has 20 sheep, Charleston has 4 * 20 = <<4*20=80>>80 sheep\nToulouse has 2 * 80 = <<2*80=160>>160 sheep\nTogether, they have 20 + 80 + 160 = <<20+80+160=260>>260 sheep\n#### 260"},
    {"question": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
     "answer": "First find how long it takes to download 40% of the file: 200 GB * 0.4 / 2 GB/minute = <<200*0.4/2=40>>40 minutes\nThen find how long it takes to download the whole file once the restart is complete: 200 GB / 2 GB/minute = <<200/2=100>>100 minutes\nThen add the time to download 40% of the file, the restart time, and the time to download the whole file: 40 + 20 + 100 = <<40+20+100=160>>160 minutes\n#### 160"},
]


def build_prompt(question):
    prompt = ""
    for ex in GSM8K_EXEMPLARS:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
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
            if re.search(r'\d+', after):
                break
        if "\nQ:" in current_text:
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
               answer_start_pos, pgd_steps=PGD_STEPS, lr=PGD_LR):
    """Prompt-only PGD attack."""
    device = clean_prompt_keys[0].device
    dtype_model = clean_prompt_keys[0].dtype

    delta_k = []
    delta_v = []
    for l in range(num_layers):
        dk = torch.zeros(clean_prompt_keys[l].shape,
                        device=device, dtype=torch.float32)
        dk.requires_grad_(True)
        delta_k.append(dk)
        dv = torch.zeros(clean_prompt_values[l].shape,
                        device=device, dtype=torch.float32)
        dv.requires_grad_(True)
        delta_v.append(dv)

    optimizer = torch.optim.Adam(delta_k + delta_v, lr=lr)
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

        text_log_p = F.log_softmax(perturbed_logits[:, :answer_start_pos, :].float(), dim=-1)
        text_q = F.softmax(clean_logits[:, :answer_start_pos, :].float(), dim=-1).detach()
        text_kl = F.kl_div(text_log_p, text_q, reduction='batchmean')

        ans_log_p = F.log_softmax(perturbed_logits[:, answer_start_pos:, :].float(), dim=-1)
        ans_q = F.softmax(clean_logits[:, answer_start_pos:, :].float(), dim=-1).detach()
        answer_kl = F.kl_div(ans_log_p, ans_q, reduction='batchmean')

        loss = text_kl - LAMBDA_ANS * answer_kl
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for l in range(num_layers):
                sig_k = clean_prompt_keys[l].float().norm().item()
                max_k = MAX_PERT_RATIO * sig_k
                cur_k = delta_k[l].norm().item()
                if cur_k > max_k:
                    delta_k[l].mul_(max_k / cur_k)

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

        if step % 10 == 0:
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

    k_pert_norms = [dk.norm().item() for dk in delta_k]
    v_pert_norms = [dv.norm().item() for dv in delta_v]
    k_signal_norms = [ck.float().norm().item() for ck in clean_prompt_keys]
    v_signal_norms = [cv.float().norm().item() for cv in clean_prompt_values]
    k_ratios = [p / (s + 1e-8) for p, s in zip(k_pert_norms, k_signal_norms)]
    v_ratios = [p / (s + 1e-8) for p, s in zip(v_pert_norms, v_signal_norms)]
    overall_ratio = float(np.mean(k_ratios + v_ratios))

    clean_norm_ans = normalize_answer(clean_answer) if clean_answer else ""
    answer_changed = (pert_norm_ans != clean_norm_ans) and pert_norm_ans != ""
    attack_success = answer_changed and (text_match_rate >= 0.90)

    del eval_out, gen_cache
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'text_match_rate': text_match_rate,
        'all_match_rate': all_match_rate,
        'answer_match_rate': answer_match_rate,
        'perturbed_answer': perturbed_answer,
        'perturbed_answer_text': perturbed_answer_text[:200],
        'answer_changed': answer_changed,
        'attack_success': attack_success,
        'overall_pert_ratio': overall_ratio,
    }


@torch.no_grad()
def compute_effective_perturbation(model, clean_prompt_keys, clean_prompt_values,
                                    delta_k, delta_v, reasoning_token_ids,
                                    num_layers):
    """
    Compute effective perturbation at each REASONING position.

    Teacher-force reasoning through clean vs perturbed prompt KV,
    then measure ||perturbed_reasoning_KV[pos] - clean_reasoning_KV[pos]||
    at each reasoning position, summed across layers.
    """
    dtype_model = clean_prompt_keys[0].dtype
    reasoning_len = reasoning_token_ids.shape[1]

    # Get clean reasoning KV by teacher-forcing through clean prompt cache
    clean_cache = DynamicCache()
    for l in range(num_layers):
        clean_cache.update(clean_prompt_keys[l].clone(),
                          clean_prompt_values[l].clone(), l)
    clean_out = model(input_ids=reasoning_token_ids,
                     past_key_values=clean_cache, use_cache=True)
    clean_reasoning_kv = clean_out.past_key_values

    # Get perturbed reasoning KV by teacher-forcing through perturbed prompt cache
    pert_cache = DynamicCache()
    for l in range(num_layers):
        pk = clean_prompt_keys[l] + delta_k[l].to(dtype_model)
        pv = clean_prompt_values[l] + delta_v[l].to(dtype_model)
        pert_cache.update(pk, pv, l)
    pert_out = model(input_ids=reasoning_token_ids,
                    past_key_values=pert_cache, use_cache=True)
    pert_reasoning_kv = pert_out.past_key_values

    # Compute per-position perturbation norms
    prompt_len = clean_prompt_keys[0].shape[2]
    per_pos_pert = np.zeros(reasoning_len)

    for l in range(num_layers):
        # Keys: [1, n_heads, total_seq_len, head_dim]
        clean_k = clean_reasoning_kv.layers[l].keys[:, :, prompt_len:, :]
        pert_k = pert_reasoning_kv.layers[l].keys[:, :, prompt_len:, :]
        clean_v = clean_reasoning_kv.layers[l].values[:, :, prompt_len:, :]
        pert_v = pert_reasoning_kv.layers[l].values[:, :, prompt_len:, :]

        # Per-position L2 norm of difference (across heads and head_dim)
        dk = (pert_k - clean_k).float()  # [1, n_heads, reasoning_len, head_dim]
        dv = (pert_v - clean_v).float()

        # Norm across heads and head_dim for each position
        k_norm = dk.norm(dim=(1, 3)).squeeze(0).cpu().numpy()  # [reasoning_len]
        v_norm = dv.norm(dim=(1, 3)).squeeze(0).cpu().numpy()  # [reasoning_len]

        per_pos_pert += k_norm + v_norm

    del clean_out, pert_out, clean_reasoning_kv, pert_reasoning_kv
    del clean_cache, pert_cache
    gc.collect()
    torch.cuda.empty_cache()

    return per_pos_pert


@torch.no_grad()
def compute_attention_scores(model, clean_prompt_keys, clean_prompt_values,
                             reasoning_token_ids, num_layers, prompt_len):
    """
    Compute per-reasoning-position attention scores.

    Returns:
    - answer_attention: attention from last reasoning position to each earlier position
    - text_coupling: average attention from later positions to each position
    Both computed using the last 4 layers, averaged across all heads.
    """
    reasoning_len = reasoning_token_ids.shape[1]

    # Build clean cache and forward with attention outputs
    clean_cache = DynamicCache()
    for l in range(num_layers):
        clean_cache.update(clean_prompt_keys[l].clone(),
                          clean_prompt_values[l].clone(), l)

    outputs = model(input_ids=reasoning_token_ids,
                   past_key_values=clean_cache,
                   use_cache=False,
                   output_attentions=True)

    # Collect attention from last 4 layers
    # Each attention tensor: [1, n_heads, reasoning_len, prompt_len + reasoning_len]
    attn_layers = []
    n_layers_total = len(outputs.attentions)
    for layer_idx in ATTENTION_LAYERS:
        actual_idx = layer_idx if layer_idx >= 0 else n_layers_total + layer_idx
        attn = outputs.attentions[actual_idx].float()  # [1, n_heads, R, P+R]
        attn_layers.append(attn)

    # Average across layers and heads: [R, P+R]
    avg_attn = torch.stack(attn_layers).mean(dim=(0, 1, 2)).squeeze(0)  # [R, P+R]

    # Answer-token attention: from last reasoning position to each reasoning position
    # Vectorized: avg_attn[R-1, P:P+R] gives attention from last query pos to all keys
    answer_attention = avg_attn[reasoning_len - 1, prompt_len:prompt_len + reasoning_len].cpu().numpy()

    # Text coupling: average attention from subsequent positions to each position
    # For position i: mean attention from positions i+1 to R-2 (exclude answer pos)
    text_coupling = np.zeros(reasoning_len)
    for i in range(reasoning_len - 2):
        kv_idx = prompt_len + i
        later_attns = avg_attn[i + 1:reasoning_len - 1, kv_idx]
        text_coupling[i] = later_attns.mean().item() if len(later_attns) > 0 else 0.0

    del outputs, clean_cache, attn_layers, avg_attn
    gc.collect()
    torch.cuda.empty_cache()

    return answer_attention, text_coupling


def partial_spearman(x, y, z):
    """
    Compute partial Spearman correlation of x and y, controlling for z.
    Uses the standard formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
    Applied to ranks for Spearman partial correlation.
    """
    # Convert to ranks
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)

    # Pearson on ranks = Spearman
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]

    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return np.nan
    return (r_xy - r_xz * r_yz) / denom


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


def generate_figures(pooled_data, per_attack_data, results_dir):
    """Generate comprehensive analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = pooled_data['positions']
    perturbations = pooled_data['perturbations']
    answer_attns = pooled_data['answer_attentions']
    text_couplings = pooled_data['text_couplings']

    # ── Figure 1: Perturbation vs Answer-Attention scatter + marginal position ──
    fig1, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1a: Perturbation vs Answer Attention (bivariate)
    ax = axes[0, 0]
    sc = ax.scatter(answer_attns, perturbations, c=positions, cmap='viridis',
                   alpha=0.3, s=5, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Position (0=early, 1=late)')
    rho_bi = stats.spearmanr(perturbations, answer_attns)[0]
    ax.set_xlabel('Answer-Token Attention')
    ax.set_ylabel('Effective Perturbation (L2)')
    ax.set_title(f'Bivariate: Perturbation vs Attention\nSpearman rho={rho_bi:.3f}')
    ax.grid(True, alpha=0.2)

    # 1b: Perturbation vs Position
    ax = axes[0, 1]
    ax.scatter(positions, perturbations, alpha=0.2, s=5, color='#e74c3c')
    rho_pp = stats.spearmanr(perturbations, positions)[0]
    ax.set_xlabel('Normalized Position')
    ax.set_ylabel('Effective Perturbation (L2)')
    ax.set_title(f'Perturbation vs Position\nSpearman rho={rho_pp:.3f}')
    ax.grid(True, alpha=0.2)

    # 1c: Answer Attention vs Position
    ax = axes[1, 0]
    ax.scatter(positions, answer_attns, alpha=0.2, s=5, color='#3498db')
    rho_ap = stats.spearmanr(answer_attns, positions)[0]
    ax.set_xlabel('Normalized Position')
    ax.set_ylabel('Answer-Token Attention')
    ax.set_title(f'Answer Attention vs Position\nSpearman rho={rho_ap:.3f}')
    ax.grid(True, alpha=0.2)

    # 1d: Within-quartile correlations
    ax = axes[1, 1]
    quartile_labels = ['Q1\n(early)', 'Q2', 'Q3', 'Q4\n(late)']
    quartile_rhos = []
    quartile_ns = []
    quartiles = np.percentile(positions, [25, 50, 75])
    q_bins = np.digitize(positions, quartiles)

    for qi in range(4):
        mask = (q_bins == qi)
        n_q = mask.sum()
        quartile_ns.append(n_q)
        if n_q >= 10:
            rho_q = stats.spearmanr(perturbations[mask], answer_attns[mask])[0]
        else:
            rho_q = np.nan
        quartile_rhos.append(rho_q)

    partial_rho = partial_spearman(perturbations, answer_attns, positions)

    colors = ['#e74c3c' if abs(r) < 0.15 else '#2ecc71' for r in quartile_rhos]
    bars = ax.bar(range(4), quartile_rhos, color=colors, edgecolor='black', lw=0.5)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.axhline(y=partial_rho, color='blue', linestyle='--', lw=2,
              label=f'Partial rho={partial_rho:.3f}')
    ax.axhline(y=rho_bi, color='red', linestyle=':', lw=2,
              label=f'Bivariate rho={rho_bi:.3f}')
    ax.set_xticks(range(4))
    ax.set_xticklabels(quartile_labels)
    ax.set_ylabel('Spearman rho (pert vs attn)')
    ax.set_title(f'Within-Quartile Correlations\nn={quartile_ns}')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.3, 1.0)
    ax.grid(True, alpha=0.2)
    for i, (r, n) in enumerate(zip(quartile_rhos, quartile_ns)):
        if not np.isnan(r):
            ax.text(i, r + 0.02, f'{r:.3f}\nn={n}', ha='center', fontsize=8)

    plt.tight_layout()
    fig1.savefig(os.path.join(results_dir, "bivariate_vs_partial.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: bivariate_vs_partial.png")

    # ── Figure 2: Per-attack partial correlations distribution ──
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2a: Distribution of per-attack bivariate vs partial rho
    bivariate_rhos = [a['bivariate_rho_pert_attn'] for a in per_attack_data
                      if not np.isnan(a.get('bivariate_rho_pert_attn', np.nan))]
    partial_rhos = [a['partial_rho_pert_attn_pos'] for a in per_attack_data
                    if not np.isnan(a.get('partial_rho_pert_attn_pos', np.nan))]

    ax = axes[0]
    if bivariate_rhos:
        ax.hist(bivariate_rhos, bins=15, alpha=0.5, color='red', label='Bivariate', edgecolor='black')
    if partial_rhos:
        ax.hist(partial_rhos, bins=15, alpha=0.5, color='blue', label='Partial (|pos)', edgecolor='black')
    ax.axvline(x=0, color='black', lw=1)
    if bivariate_rhos:
        ax.axvline(x=np.median(bivariate_rhos), color='red', linestyle='--', lw=2,
                  label=f'Bivariate median={np.median(bivariate_rhos):.3f}')
    if partial_rhos:
        ax.axvline(x=np.median(partial_rhos), color='blue', linestyle='--', lw=2,
                  label=f'Partial median={np.median(partial_rhos):.3f}')
    ax.set_xlabel('Spearman rho')
    ax.set_ylabel('Count')
    ax.set_title('Per-Attack Correlation Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 2b: Bivariate vs Partial scatter (per attack)
    ax = axes[1]
    if bivariate_rhos and partial_rhos and len(bivariate_rhos) == len(partial_rhos):
        ax.scatter(bivariate_rhos, partial_rhos, color='#2c3e50', s=30, alpha=0.7)
        ax.plot([-0.5, 1], [-0.5, 1], 'k--', alpha=0.3, label='y=x (no confound)')
        ax.axhline(y=0, color='gray', lw=0.5)
        ax.axvline(x=0, color='gray', lw=0.5)
        ax.set_xlabel('Bivariate rho (pert vs attn)')
        ax.set_ylabel('Partial rho (pert vs attn | position)')
        ax.set_title('Bivariate vs Partial: Per-Attack')
        ax.legend(fontsize=9)
        drop = np.median(bivariate_rhos) - np.median(partial_rhos)
        ax.text(0.05, 0.95, f'Median drop: {drop:.3f}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig2.savefig(os.path.join(results_dir, "per_attack_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: per_attack_distributions.png")

    # ── Figure 3: Position-binned perturbation and attention profiles ──
    fig3, axes = plt.subplots(1, 3, figsize=(18, 5))

    n_bins = 10
    pos_bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (pos_bins[:-1] + pos_bins[1:]) / 2
    bin_idx = np.digitize(positions, pos_bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    mean_pert = np.array([perturbations[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else 0
                         for i in range(n_bins)])
    mean_attn = np.array([answer_attns[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else 0
                         for i in range(n_bins)])
    mean_tc = np.array([text_couplings[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else 0
                       for i in range(n_bins)])

    # Normalize for comparison
    if mean_pert.max() > 0:
        norm_pert = mean_pert / mean_pert.max()
    else:
        norm_pert = mean_pert
    if mean_attn.max() > 0:
        norm_attn = mean_attn / mean_attn.max()
    else:
        norm_attn = mean_attn
    if mean_tc.max() > 0:
        norm_tc = mean_tc / mean_tc.max()
    else:
        norm_tc = mean_tc

    ax = axes[0]
    ax.plot(bin_centers, norm_pert, 'r-o', lw=2, label='Perturbation')
    ax.plot(bin_centers, norm_attn, 'b-o', lw=2, label='Answer Attention')
    ax.plot(bin_centers, norm_tc, 'g-o', lw=2, label='Text Coupling')
    ax.set_xlabel('Normalized Position')
    ax.set_ylabel('Normalized Value (0-1)')
    ax.set_title('Position Profiles (normalized)')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Raw profiles
    ax = axes[1]
    ax.plot(bin_centers, mean_pert, 'r-o', lw=2)
    ax.set_xlabel('Normalized Position')
    ax.set_ylabel('Mean Effective Perturbation')
    ax.set_title('Perturbation vs Position (raw)')
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.plot(bin_centers, mean_attn, 'b-o', lw=2, label='Answer Attn')
    ax.plot(bin_centers, mean_tc, 'g-o', lw=2, label='Text Coupling')
    ax.set_xlabel('Normalized Position')
    ax.set_ylabel('Mean Attention')
    ax.set_title('Attention vs Position (raw)')
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig3.savefig(os.path.join(results_dir, "position_profiles.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: position_profiles.png")

    # ── Figure 4: Summary verdict ──
    fig4, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Bivariate\nrho(pert,attn)', 'Partial\nrho(pert,attn|pos)',
               'rho(pert,pos)', 'rho(attn,pos)',
               'rho(pert,TC)', 'Partial\nrho(pert,TC|pos)']
    rho_pert_tc = stats.spearmanr(perturbations, text_couplings)[0]
    partial_rho_tc = partial_spearman(perturbations, text_couplings, positions)
    values = [rho_bi, partial_rho, rho_pp, rho_ap, rho_pert_tc, partial_rho_tc]

    colors_v = ['#2ecc71' if abs(v) >= 0.3 else '#f39c12' if abs(v) >= 0.15 else '#e74c3c'
               for v in values]
    ax.bar(range(len(metrics)), values, color=colors_v, edgecolor='black', lw=0.5)
    ax.axhline(y=0, color='black', lw=1)
    ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Strong threshold')
    ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Weak threshold')
    ax.axhline(y=-0.15, color='orange', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel('Spearman rho')
    ax.set_title('Position-Controlled PGD Correlation Analysis\n'
                 'Green=genuine (|rho|≥0.3), Yellow=weak, Red=confound (|rho|<0.15)')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.5, 1.0)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig4.savefig(os.path.join(results_dir, "summary_verdict.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: summary_verdict.png")


def main():
    start_time = time.time()
    print(f"{'='*70}")
    print(f"Experiment 018: Position-Controlled PGD Correlation Analysis")
    print(f"Model: {MODEL_NAME}")
    print(f"PGD: {PGD_STEPS} steps, lr={PGD_LR}, lambda_ans={LAMBDA_ANS}")
    print(f"{'='*70}\n")

    # ── Load model ────────────────────────────────────────────────────
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    model.requires_grad_(False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded in {time.time() - start_time:.1f}s")
    print(f"Layers: {num_layers}")

    # ── Load GSM8K ────────────────────────────────────────────────────
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected = indices[:NUM_PROBLEMS]

    # ── Phase 1: Generate traces ──────────────────────────────────────
    print("\n=== Phase 1: Generating clean traces ===")
    problems_data = []

    for prob_idx, ds_idx in enumerate(selected):
        elapsed = time.time() - start_time
        if elapsed > 300:  # 5 min budget for phase 1
            print(f"Phase 1 time budget reached at problem {prob_idx}")
            break

        problem = dataset[ds_idx]
        question = problem["question"]
        true_answer = normalize_answer(
            problem["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
        )
        prompt = build_prompt(question)
        trace_text = generate_trace(model, tokenizer, prompt)
        gen_answer = extract_answer(trace_text)
        gen_norm = normalize_answer(gen_answer) if gen_answer else ""
        correct = (gen_norm == true_answer)

        print(f"  Problem {prob_idx+1}/{NUM_PROBLEMS} (#{ds_idx}): "
              f"'{gen_answer}' vs '{true_answer}' (correct: {correct})")

        if not correct:
            continue

        if "####" in trace_text:
            reasoning_text = trace_text[:trace_text.index("####")]
        else:
            reasoning_text = trace_text

        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]

        reasoning_only_text = prompt + reasoning_text
        reasoning_inputs = tokenizer(reasoning_only_text, return_tensors="pt")
        reasoning_len = reasoning_inputs.input_ids.shape[1] - prompt_len

        if reasoning_len < 15 or reasoning_len > MAX_REASONING_TOKENS:
            print(f"    SKIP: reasoning_len={reasoning_len}")
            continue

        problems_data.append({
            "ds_idx": ds_idx,
            "true_answer": true_answer,
            "clean_answer": gen_answer,
            "prompt": prompt,
            "trace_text": trace_text,
            "reasoning_text": reasoning_text,
            "prompt_len": prompt_len,
            "reasoning_len": reasoning_len,
        })
        print(f"    Valid! prompt_len={prompt_len}, reasoning_len={reasoning_len}")

    n_valid = len(problems_data)
    print(f"\n=== Phase 1 complete: {n_valid} valid problems ===\n")

    if n_valid < 3:
        print("ERROR: Too few valid problems!")
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump({"error": "too_few_valid", "n_valid": n_valid}, f, indent=2)
        return

    # ── Phase 2: PGD + Per-Position Analysis ──────────────────────────
    print(f"=== Phase 2: PGD attacks with per-position analysis ({n_valid} problems) ===")
    all_results = []
    all_position_data = []  # per-position pooled data

    for pi, pd in enumerate(problems_data):
        elapsed = time.time() - start_time
        if elapsed > 1500:  # Leave 5 min for analysis
            print(f"Time budget reached at problem {pi}")
            break

        print(f"\nProblem {pi+1}/{n_valid} (#{pd['ds_idx']}), "
              f"true_answer={pd['true_answer']} [{elapsed:.0f}s elapsed]")

        try:
            # Teacher-force prompt
            prompt_inputs = tokenizer(pd["prompt"], return_tensors="pt").to(model.device)
            prompt_kv = teacher_force_chunked(model, prompt_inputs.input_ids)
            clean_pk = [prompt_kv.layers[l].keys.clone() for l in range(num_layers)]
            clean_pv = [prompt_kv.layers[l].values.clone() for l in range(num_layers)]
            del prompt_kv
            gc.collect()
            torch.cuda.empty_cache()

            # Get reasoning token IDs
            reasoning_text_full = pd["prompt"] + pd["reasoning_text"]
            reasoning_full_ids = tokenizer(reasoning_text_full,
                                          return_tensors="pt").to(model.device)
            reasoning_token_ids = reasoning_full_ids.input_ids[:, pd["prompt_len"]:]
            reasoning_len = reasoning_token_ids.shape[1]
            answer_start = find_answer_boundary(reasoning_len)

            print(f"  Reasoning tokens: {reasoning_len}, answer_start: {answer_start}")

            # Get clean logits
            clean_cache = DynamicCache()
            for l in range(num_layers):
                clean_cache.update(clean_pk[l].clone(), clean_pv[l].clone(), l)
            with torch.no_grad():
                clean_out = model(input_ids=reasoning_token_ids,
                                past_key_values=clean_cache, use_cache=False)
                clean_logits = clean_out.logits.clone()
            del clean_cache, clean_out
            gc.collect()
            torch.cuda.empty_cache()

            # Run PGD attack
            print(f"  Running PGD ({PGD_STEPS} steps)...")
            delta_k, delta_v, step_metrics = pgd_attack(
                model, clean_pk, clean_pv,
                reasoning_token_ids, clean_logits,
                num_layers, answer_start,
            )

            # Evaluate attack
            print(f"  Evaluating attack...")
            result = evaluate_attack(
                model, tokenizer, clean_pk, clean_pv,
                delta_k, delta_v, reasoning_token_ids, clean_logits,
                pd["true_answer"], pd["clean_answer"],
                num_layers, answer_start,
            )

            print(f"  RESULT: text_match={result['text_match_rate']:.1%}, "
                  f"answer_changed={result['answer_changed']}, "
                  f"success={result['attack_success']}, "
                  f"pert_ratio={result['overall_pert_ratio']:.1f}x")

            # ── KEY: Compute per-position data ──
            print(f"  Computing per-position effective perturbation...")
            per_pos_pert = compute_effective_perturbation(
                model, clean_pk, clean_pv, delta_k, delta_v,
                reasoning_token_ids, num_layers
            )

            print(f"  Computing per-position attention scores...")
            answer_attn, text_coupling = compute_attention_scores(
                model, clean_pk, clean_pv, reasoning_token_ids,
                num_layers, pd["prompt_len"]
            )

            # Normalized positions
            positions = np.arange(reasoning_len) / reasoning_len

            # Per-attack correlations
            rho_bi, p_bi = stats.spearmanr(per_pos_pert, answer_attn)
            rho_tc, p_tc = stats.spearmanr(per_pos_pert, text_coupling)
            rho_pp, p_pp = stats.spearmanr(per_pos_pert, positions)
            rho_ap, p_ap = stats.spearmanr(answer_attn, positions)
            partial_rho = partial_spearman(per_pos_pert, answer_attn, positions)
            partial_rho_tc = partial_spearman(per_pos_pert, text_coupling, positions)

            print(f"  Bivariate rho(pert,attn) = {rho_bi:.3f} (p={p_bi:.2e})")
            print(f"  rho(pert,position) = {rho_pp:.3f}")
            print(f"  rho(attn,position) = {rho_ap:.3f}")
            print(f"  PARTIAL rho(pert,attn|pos) = {partial_rho:.3f}")

            result['per_pos_perturbation'] = per_pos_pert.tolist()
            result['per_pos_answer_attn'] = answer_attn.tolist()
            result['per_pos_text_coupling'] = text_coupling.tolist()
            result['per_pos_positions'] = positions.tolist()
            result['bivariate_rho_pert_attn'] = float(rho_bi)
            result['bivariate_rho_pert_tc'] = float(rho_tc)
            result['bivariate_rho_pert_pos'] = float(rho_pp)
            result['bivariate_rho_attn_pos'] = float(rho_ap)
            result['partial_rho_pert_attn_pos'] = float(partial_rho)
            result['partial_rho_pert_tc_pos'] = float(partial_rho_tc)
            result['reasoning_len'] = reasoning_len
            result['ds_idx'] = pd['ds_idx']
            result['true_answer'] = pd['true_answer']
            result['clean_answer'] = pd['clean_answer']
            result['step_metrics'] = step_metrics

            # Pool position-level data
            for i in range(reasoning_len):
                all_position_data.append({
                    'attack_idx': pi,
                    'position': positions[i],
                    'perturbation': per_pos_pert[i],
                    'answer_attention': answer_attn[i],
                    'text_coupling': text_coupling[i],
                })

            all_results.append(result)

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM! Skipping this problem.")
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    # ── Phase 3: Pooled Analysis ──────────────────────────────────────
    n_evaluated = len(all_results)
    print(f"\n{'='*70}")
    print(f"POOLED ANALYSIS ({n_evaluated} attacks, {len(all_position_data)} position-level obs)")
    print(f"{'='*70}")

    if n_evaluated == 0 or len(all_position_data) == 0:
        print("No data for analysis!")
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump({"error": "no_data", "n_evaluated": 0}, f, indent=2)
        return

    # Pool all position-level data
    all_pos = np.array([d['position'] for d in all_position_data])
    all_pert = np.array([d['perturbation'] for d in all_position_data])
    all_attn = np.array([d['answer_attention'] for d in all_position_data])
    all_tc = np.array([d['text_coupling'] for d in all_position_data])

    # ── Pooled bivariate correlations ──
    rho_pert_attn, p_pa = stats.spearmanr(all_pert, all_attn)
    rho_pert_tc, p_pt = stats.spearmanr(all_pert, all_tc)
    rho_pert_pos, p_pp = stats.spearmanr(all_pert, all_pos)
    rho_attn_pos, p_ap = stats.spearmanr(all_attn, all_pos)
    rho_tc_pos, p_tp = stats.spearmanr(all_tc, all_pos)

    print(f"\n--- Pooled Bivariate Spearman Correlations (n={len(all_pos)}) ---")
    print(f"  rho(perturbation, answer_attention) = {rho_pert_attn:.4f} (p={p_pa:.2e})")
    print(f"  rho(perturbation, text_coupling)    = {rho_pert_tc:.4f} (p={p_pt:.2e})")
    print(f"  rho(perturbation, position)         = {rho_pert_pos:.4f} (p={p_pp:.2e})")
    print(f"  rho(answer_attention, position)     = {rho_attn_pos:.4f} (p={p_ap:.2e})")
    print(f"  rho(text_coupling, position)        = {rho_tc_pos:.4f} (p={p_tp:.2e})")

    # ── Pooled partial correlations ──
    partial_pert_attn = partial_spearman(all_pert, all_attn, all_pos)
    partial_pert_tc = partial_spearman(all_pert, all_tc, all_pos)

    print(f"\n--- Pooled Partial Spearman Correlations ---")
    print(f"  partial_rho(perturbation, answer_attention | position) = {partial_pert_attn:.4f}")
    print(f"  partial_rho(perturbation, text_coupling | position)    = {partial_pert_tc:.4f}")

    # ── Within-quartile correlations ──
    quartiles_edges = np.percentile(all_pos, [25, 50, 75])
    q_idx = np.digitize(all_pos, quartiles_edges)
    quartile_results = []

    print(f"\n--- Within-Quartile Spearman (perturbation vs answer_attention) ---")
    for qi in range(4):
        mask = (q_idx == qi)
        n_q = mask.sum()
        if n_q >= 10:
            rho_q, p_q = stats.spearmanr(all_pert[mask], all_attn[mask])
            pos_range = f"[{all_pos[mask].min():.3f}, {all_pos[mask].max():.3f}]"
        else:
            rho_q = np.nan
            p_q = np.nan
            pos_range = "N/A"
        quartile_results.append({
            'quartile': qi + 1,
            'n': int(n_q),
            'rho': float(rho_q),
            'p': float(p_q) if not np.isnan(p_q) else None,
            'pos_range': pos_range,
        })
        print(f"  Q{qi+1} (n={n_q}, pos {pos_range}): "
              f"rho={rho_q:.4f}" + (f" (p={p_q:.2e})" if not np.isnan(p_q) else ""))

    # ── Per-attack statistics ──
    per_attack_bivariate = [r['bivariate_rho_pert_attn'] for r in all_results]
    per_attack_partial = [r['partial_rho_pert_attn_pos'] for r in all_results]
    per_attack_pert_pos = [r['bivariate_rho_pert_pos'] for r in all_results]
    per_attack_attn_pos = [r['bivariate_rho_attn_pos'] for r in all_results]

    print(f"\n--- Per-Attack Statistics ({n_evaluated} attacks) ---")
    print(f"  Bivariate rho(pert,attn): "
          f"mean={np.mean(per_attack_bivariate):.4f}, "
          f"median={np.median(per_attack_bivariate):.4f}, "
          f"range=[{np.min(per_attack_bivariate):.4f}, {np.max(per_attack_bivariate):.4f}]")
    print(f"  Partial rho(pert,attn|pos): "
          f"mean={np.mean(per_attack_partial):.4f}, "
          f"median={np.median(per_attack_partial):.4f}, "
          f"range=[{np.min(per_attack_partial):.4f}, {np.max(per_attack_partial):.4f}]")
    print(f"  rho(pert,pos): "
          f"mean={np.mean(per_attack_pert_pos):.4f}, "
          f"median={np.median(per_attack_pert_pos):.4f}")
    print(f"  rho(attn,pos): "
          f"mean={np.mean(per_attack_attn_pos):.4f}, "
          f"median={np.median(per_attack_attn_pos):.4f}")

    # Attack success statistics
    n_success = sum(1 for r in all_results if r['attack_success'])
    n_changed = sum(1 for r in all_results if r['answer_changed'])
    mean_text_match = np.mean([r['text_match_rate'] for r in all_results])
    mean_pert_ratio = np.mean([r['overall_pert_ratio'] for r in all_results])

    print(f"\n--- Attack Results ---")
    print(f"  Success rate: {n_success}/{n_evaluated} ({n_success/n_evaluated:.0%})")
    print(f"  Answer changed: {n_changed}/{n_evaluated} ({n_changed/n_evaluated:.0%})")
    print(f"  Mean text match: {mean_text_match:.1%}")
    print(f"  Mean pert ratio: {mean_pert_ratio:.1f}x")

    # ── Verdict ──
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    if abs(partial_pert_attn) >= 0.3:
        verdict = "SCENARIO A: GENUINE SPATIAL STRUCTURE — partial rho survives"
    elif abs(partial_pert_attn) < 0.15:
        verdict = "SCENARIO B: POSITIONAL CONFOUND — partial rho collapses"
    else:
        verdict = "SCENARIO C: PARTIAL CONFOUND — moderate residual"
    print(f"  {verdict}")
    print(f"  Bivariate rho = {rho_pert_attn:.4f}")
    print(f"  Partial rho   = {partial_pert_attn:.4f}")
    print(f"  Drop          = {rho_pert_attn - partial_pert_attn:.4f}")
    print(f"  Fraction explained by position = "
          f"{1 - partial_pert_attn/rho_pert_attn:.1%}" if abs(rho_pert_attn) > 0.01 else "N/A")

    # ── Save results ──────────────────────────────────────────────────
    pooled_data = {
        'positions': all_pos,
        'perturbations': all_pert,
        'answer_attentions': all_attn,
        'text_couplings': all_tc,
    }

    # Generate figures
    print(f"\n=== Generating figures ===")
    generate_figures(pooled_data, all_results, RESULTS_DIR)

    # Save summary
    summary = {
        "experiment": "exp_018_pgd_position_control",
        "model": MODEL_NAME,
        "n_valid": n_valid,
        "n_evaluated": n_evaluated,
        "n_position_obs": len(all_position_data),
        "pgd_steps": PGD_STEPS,
        "pgd_lr": PGD_LR,
        "lambda_ans": LAMBDA_ANS,
        "attack_success_rate": n_success / n_evaluated if n_evaluated > 0 else 0,
        "mean_text_match": float(mean_text_match),
        "mean_pert_ratio": float(mean_pert_ratio),
        "pooled_correlations": {
            "bivariate_rho_pert_attn": float(rho_pert_attn),
            "bivariate_rho_pert_tc": float(rho_pert_tc),
            "bivariate_rho_pert_pos": float(rho_pert_pos),
            "bivariate_rho_attn_pos": float(rho_attn_pos),
            "bivariate_rho_tc_pos": float(rho_tc_pos),
            "partial_rho_pert_attn_pos": float(partial_pert_attn),
            "partial_rho_pert_tc_pos": float(partial_pert_tc),
        },
        "per_attack_stats": {
            "bivariate_rho_pert_attn": {
                "mean": float(np.mean(per_attack_bivariate)),
                "median": float(np.median(per_attack_bivariate)),
                "min": float(np.min(per_attack_bivariate)),
                "max": float(np.max(per_attack_bivariate)),
            },
            "partial_rho_pert_attn_pos": {
                "mean": float(np.mean(per_attack_partial)),
                "median": float(np.median(per_attack_partial)),
                "min": float(np.min(per_attack_partial)),
                "max": float(np.max(per_attack_partial)),
            },
            "rho_pert_pos": {
                "mean": float(np.mean(per_attack_pert_pos)),
                "median": float(np.median(per_attack_pert_pos)),
            },
            "rho_attn_pos": {
                "mean": float(np.mean(per_attack_attn_pos)),
                "median": float(np.median(per_attack_attn_pos)),
            },
        },
        "quartile_correlations": quartile_results,
        "verdict": verdict,
        "elapsed_seconds": time.time() - start_time,
        "per_problem_results": [
            {k: v for k, v in r.items()
             if k not in ('step_metrics', 'per_pos_perturbation',
                         'per_pos_answer_attn', 'per_pos_text_coupling',
                         'per_pos_positions')}
            for r in all_results
        ],
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_convert)
    print(f"Summary saved to {RESULTS_DIR}/summary.json")

    # Save position-level data for future analysis
    pos_data_compact = {
        'positions': all_pos.tolist(),
        'perturbations': all_pert.tolist(),
        'answer_attentions': all_attn.tolist(),
        'text_couplings': all_tc.tolist(),
        'attack_indices': [d['attack_idx'] for d in all_position_data],
    }
    with open(os.path.join(RESULTS_DIR, "position_level_data.json"), "w") as f:
        json.dump(pos_data_compact, f, default=_convert)
    print(f"Position-level data saved ({len(all_position_data)} observations)")

    total_time = time.time() - start_time
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("DONE")


if __name__ == "__main__":
    main()
