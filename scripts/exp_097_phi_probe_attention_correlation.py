#!/usr/bin/env python3
"""
Experiment 097: Cross-Model Probe-Attention Correlation on Phi-3.5-mini

Replicates exp_096 (probe-attention correlation) on Phi-3.5-mini-Instruct (MHA)
to test whether information-directed attention generalizes beyond Qwen (GQA).

Key differences from exp_096:
- Phi-3.5-mini-Instruct (MHA, 32 KV heads, analog encoding)
- V|nums landscape from exp_086 (Phi position-sweep, L16 and L24)
- Probes 4 layers: L8 (early), L16 (mid), L24 (deep), L31 (final)
- Adds QUADRATIC position control to partial correlation (addresses
  exp_096 confound #1: non-linear recency)

Core question: Does Phi's answer-step attention align with positions encoding
hidden information (V|nums), beyond what linear AND quadratic position effects
explain? If yes → information-directed attention is a general property, not
Qwen/GQA-specific.
"""

import os
import json
import time
import gc
import re
import sys
import random
import warnings

import numpy as np
import torch
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

T0 = time.time()
TIME_BUDGET = 6600  # 110 min
MAX_GEN = 512
MODEL_NAME = 'microsoft/Phi-3.5-mini-instruct'
N_PROBLEMS = 200
N_BINS = 20
N_CONTROL_STEPS = 5
# 4 layers spanning 32-layer Phi
LAYERS_PROBE = [8, 16, 24, 31]

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_097"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-computed V|nums from exp_086 (Phi-3.5-mini, ~168 correct GSM8K problems)
EXP086_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_086" / "results.json"

# ── Plain-text 8-shot exemplars (same as prior experiments) ──
EXEMPLARS = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
     "a": "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\nShe makes 9 * 2 = $18 every day at the farmer's market.\n#### 18"},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "a": "It takes 2/2 = 1 bolt of white fiber.\nSo the total bolts needed is 2 + 1 = 3.\n#### 3"},
    {"q": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
     "a": "The cost of the house and repairs came out to 80,000 + 50,000 = $130,000.\nHe increased the value of the house by 80,000 * 1.5 = $120,000.\nSo the new value of the house is 120,000 + 80,000 = $200,000.\nSo he made a profit of 200,000 - 130,000 = $70,000.\n#### 70000"},
    {"q": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "a": "He writes each friend 3 * 2 = 6 pages a week.\nSo he writes 6 * 2 = 12 pages every week.\nThat means he writes 12 * 52 = 624 pages a year.\n#### 624"},
    {"q": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
     "a": "If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3 * 20 = 60 cups of feed per day.\nIf she feeds the flock 15 cups in the morning, and 25 cups in the afternoon, then the carry-over to the final meal would be 60 - 15 - 25 = 20 cups.\n#### 20"},
    {"q": "Kylar went to the store to get water and some apples. The store sold apples for $1 each and water for $3 per bottle. Kylar wanted to buy one bag of apples and 2 bottles of water. How much would Kylar spend if each bag has 6 apples?",
     "a": "A bag has 6 apples and each apple costs $1, so a bag costs 6 * 1 = $6.\nKylar wants 2 bottles of water so that would cost 2 * 3 = $6.\nAltogether, Kylar would spend 6 + 6 = $12.\n#### 12"},
    {"q": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
     "a": "If Seattle has 20 sheep, Charleston has 4 * 20 = 80 sheep.\nToulouse has 2 * 80 = 160 sheep.\nTogether, they have 20 + 80 + 160 = 260 sheep.\n#### 260"},
    {"q": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
     "a": "First find how long it takes to download 40% of the file: 200 * 0.4 / 2 = 40 minutes.\nThen find how long it takes to download the whole file once the restart is complete: 200 / 2 = 100 minutes.\nThen add the time to download 40% of the file, the restart time, and the time to download the whole file: 40 + 20 + 100 = 160 minutes.\n#### 160"},
]


def build_prompt(question):
    prompt = ""
    for ex in EXEMPLARS:
        prompt += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '')
    m = re.search(r'answer\s+is\s+\$?(-?[\d,]+(?:\.\d+)?)', text, re.I)
    if m:
        return m.group(1).replace(',', '')
    return None


def extract_gold(answer_text):
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', answer_text)
    if m:
        return m.group(1).replace(',', '')
    return None


def load_gsm8k():
    from datasets import load_dataset
    return load_dataset("openai/gsm8k", "main", split="test")


def find_answer_position(full_ids, prompt_len, tokenizer):
    """Find position of the first answer digit token after '####'."""
    gen_ids = full_ids[prompt_len:]
    text_so_far = ""
    hash_found = False
    for i, tid in enumerate(gen_ids):
        tok = tokenizer.decode([tid], skip_special_tokens=False)
        text_so_far += tok
        if not hash_found:
            if "####" in text_so_far:
                hash_found = True
            continue
        if hash_found:
            stripped = tok.strip()
            if stripped and (stripped[0].isdigit() or stripped[0] == '-'):
                return prompt_len + i
    return None


def bin_attention(attn_row, prompt_len, chain_end, n_bins=20):
    """Bin attention weights into chain bins + prompt fraction."""
    total = float(attn_row.sum())
    if total < 1e-10:
        return np.zeros(n_bins + 1, dtype=np.float64)

    prompt_frac = float(attn_row[:prompt_len].sum()) / total
    chain_len = chain_end - prompt_len
    if chain_len <= 0:
        result = np.zeros(n_bins + 1, dtype=np.float64)
        result[-1] = prompt_frac
        return result

    chain_attn = attn_row[prompt_len:chain_end].astype(np.float64)
    binned = np.zeros(n_bins, dtype=np.float64)
    bin_size = chain_len / n_bins
    for b in range(n_bins):
        b_start = int(b * bin_size)
        b_end = int((b + 1) * bin_size)
        if b == n_bins - 1:
            b_end = chain_len
        if b_end > b_start:
            binned[b] = float(chain_attn[b_start:b_end].sum()) / total

    result = np.zeros(n_bins + 1, dtype=np.float64)
    result[:n_bins] = binned
    result[-1] = prompt_frac
    return result


def partial_corr(x, y, z_cols):
    """Partial correlation of x and y, controlling for z (can be multi-column).
    z_cols: 2D array (n, k) where k is number of confounds."""
    from sklearn.linear_model import LinearRegression
    if len(x) < 5 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 1.0
    z = np.array(z_cols)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    x_resid = x - LinearRegression().fit(z, x).predict(z)
    y_resid = y - LinearRegression().fit(z, y).predict(z)
    if np.std(x_resid) < 1e-10 or np.std(y_resid) < 1e-10:
        return 0.0, 1.0
    return stats.pearsonr(x_resid, y_resid)


def main():
    t0 = time.time()

    print("=" * 70)
    print("Experiment 097: Cross-Model Probe-Attention Correlation (Phi-3.5-mini)")
    print("Does information-directed attention generalize beyond Qwen?")
    print("=" * 70)

    # ═══════════════════════════════════════════════════
    # Load pre-computed V|nums from exp_086 (Phi position-sweep)
    # ═══════════════════════════════════════════════════
    print("\nLoading V|nums information landscape from exp_086 (Phi)...")
    with open(EXP086_PATH) as f:
        exp086 = json.load(f)

    info_landscapes = {}
    for layer_key in ['L16', 'L24']:
        res = exp086['results'][layer_key]
        info_landscapes[layer_key] = {
            'V_R': np.array(res['V_R']),
            'V_nums_R': np.array(res['V_nums_R']),
            'nums_R': np.array(res['nums_R']),
            'K_nums_R': np.array(res.get('K_nums_R', res['V_nums_R'])),
        }
        print(f"  {layer_key}: V|nums mean={np.mean(res['V_nums_R']):.3f}, "
              f"V_R mean={np.mean(res['V_R']):.3f}, "
              f"nums_R mean={np.mean(res['nums_R']):.3f}")

    # Map experiment layers to exp_086 layers
    layer_to_info = {8: 'L16', 16: 'L16', 24: 'L24', 31: 'L24'}
    bin_centers = np.array(exp086['bin_centers'])

    # ═══════════════════════════════════════════════════
    # Load model
    # ═══════════════════════════════════════════════════
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load WITHOUT trust_remote_code to use transformers' built-in Phi3 implementation
    # (the custom HF code has compatibility issues with transformers 5.3.0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map='auto',
        attn_implementation='eager'
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_q_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, 'num_key_value_heads', n_q_heads)
    group_size = n_q_heads // n_kv_heads
    layers_probe = [l for l in LAYERS_PROBE if l < n_layers]

    print(f"\nModel: {MODEL_NAME}, {n_layers} layers")
    print(f"Q heads: {n_q_heads}, KV heads: {n_kv_heads}, group_size: {group_size}")
    print(f"Probing {len(layers_probe)} layers: {layers_probe}")

    ds = load_gsm8k()
    print(f"GSM8K: {len(ds)} problems")

    # ═══════════════════════════════════════════════════
    # PHASE 1: Generate CoT traces
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 1: Generating CoT traces")
    print(f"{'='*70}")

    generations = []
    for i in range(min(N_PROBLEMS, len(ds))):
        if time.time() - t0 > TIME_BUDGET * 0.20:
            print(f"  Time budget (20%) reached at problem {i}")
            break

        question = ds[i]['question']
        gold = extract_gold(ds[i]['answer'])
        prompt = build_prompt(question)

        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        prompt_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_GEN, do_sample=False,
                temperature=1.0, use_cache=True
            )

        gen_ids = output[0][prompt_len:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_answer(gen_text)
        correct = pred is not None and gold is not None and str(pred).strip() == str(gold).strip()

        generations.append({
            'idx': i, 'gold': gold, 'pred': pred, 'correct': correct,
            'gen_ids': gen_ids, 'prompt_len': prompt_len,
            'full_ids': output[0].tolist(), 'gen_text': gen_text,
        })

        if (i + 1) % 50 == 0:
            nc = sum(g['correct'] for g in generations)
            print(f"  {i+1} problems: {nc}/{len(generations)} correct "
                  f"({100*nc/len(generations):.1f}%), {time.time()-t0:.0f}s")

    n_correct = sum(g['correct'] for g in generations)
    print(f"\nGeneration complete: {n_correct}/{len(generations)} correct "
          f"({100*n_correct/len(generations):.1f}%)")

    correct_gens = [g for g in generations if g['correct']]

    # ═══════════════════════════════════════════════════
    # PHASE 2: Extract attention at answer & control steps
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 2: Extracting attention at answer and control steps")
    print(f"{'='*70}")

    # For Phi MHA with 32 KV heads, we'll aggregate into 8 groups of 4 for
    # presentation (comparable to Qwen's 8 KV groups) but also keep full 32-head data
    N_KV_GROUPS = 8
    kv_group_size = n_kv_heads // N_KV_GROUPS if n_kv_heads >= N_KV_GROUPS else 1

    answer_attn = {l: [] for l in layers_probe}
    control_attn = {l: [] for l in layers_probe}
    raw_answer_attn = {l: [] for l in layers_probe}
    raw_control_attn = {l: [] for l in layers_probe}
    valid_problems = []

    # V|nums lookup
    v_nums_l16 = info_landscapes['L16']['V_nums_R']
    v_nums_l24 = info_landscapes['L24']['V_nums_R']

    def vnums_lookup(rel_pos, layer):
        v_nums = v_nums_l16 if layer <= 16 else v_nums_l24
        bin_idx = min(int(rel_pos * N_BINS), N_BINS - 1)
        return v_nums[bin_idx]

    for pi, gen in enumerate(correct_gens):
        if time.time() - t0 > TIME_BUDGET * 0.65:
            print(f"  Time budget (65%) reached at problem {pi}")
            break

        full_ids = gen['full_ids']
        prompt_len = gen['prompt_len']
        answer_pos = find_answer_position(full_ids, prompt_len, tokenizer)

        if answer_pos is None:
            continue

        chain_len = answer_pos - prompt_len
        if chain_len < 10:
            continue

        # Select control positions
        ctrl_fracs = [0.25, 0.40, 0.50, 0.60, 0.75]
        ctrl_positions = [prompt_len + int(f * chain_len) for f in ctrl_fracs]
        ctrl_positions = [p for p in ctrl_positions if prompt_len + 5 < p < answer_pos]

        if len(ctrl_positions) < 3:
            continue

        # Forward pass with attention
        input_ids = torch.tensor([full_ids[:answer_pos + 1]], device=model.device)
        try:
            with torch.no_grad():
                outputs = model(input_ids, output_attentions=True, use_cache=False)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at problem {pi} (seq_len={len(full_ids)}), skipping")
            torch.cuda.empty_cache()
            continue

        for l in layers_probe:
            attn = outputs.attentions[l]  # (1, n_q_heads, seq_len, seq_len)

            # Answer step attention from answer_pos
            ans_attn_q = attn[0, :, answer_pos, :answer_pos + 1].float().cpu().numpy()

            # For MHA: each Q head IS a KV head (group_size=1)
            # Average Q heads within each KV group
            ans_attn_kv = np.zeros((n_kv_heads, answer_pos + 1))
            for kv_h in range(n_kv_heads):
                q_start = kv_h * group_size
                q_end = (kv_h + 1) * group_size
                ans_attn_kv[kv_h] = ans_attn_q[q_start:q_end].mean(axis=0)

            # Bin answer attention
            ans_binned = np.zeros((n_kv_heads, N_BINS + 1))
            for kv_h in range(n_kv_heads):
                ans_binned[kv_h] = bin_attention(
                    ans_attn_kv[kv_h], prompt_len, answer_pos, N_BINS)

            answer_attn[l].append(ans_binned)

            # Position-resolved answer attention
            chain_positions = np.arange(prompt_len, answer_pos)
            rel_positions = (chain_positions - prompt_len) / chain_len
            v_lookup = np.array([vnums_lookup(rp, l) for rp in rel_positions])
            pos_attn = ans_attn_kv[:, prompt_len:answer_pos]
            raw_answer_attn[l].append({
                'rel_pos': rel_positions,
                'v_nums': v_lookup,
                'attn': pos_attn,
            })

            # Control steps
            ctrl_binned_list = []
            ctrl_raw_list = []
            for cp in ctrl_positions:
                ctrl_attn_q = attn[0, :, cp, :cp + 1].float().cpu().numpy()
                ctrl_attn_kv = np.zeros((n_kv_heads, cp + 1))
                for kv_h in range(n_kv_heads):
                    q_start = kv_h * group_size
                    q_end = (kv_h + 1) * group_size
                    ctrl_attn_kv[kv_h] = ctrl_attn_q[q_start:q_end].mean(axis=0)

                cb = np.zeros((n_kv_heads, N_BINS + 1))
                for kv_h in range(n_kv_heads):
                    cb[kv_h] = bin_attention(
                        np.concatenate([ctrl_attn_kv[kv_h], np.zeros(answer_pos + 1 - (cp + 1))]),
                        prompt_len, answer_pos, N_BINS)
                ctrl_binned_list.append(cb)

                ctrl_chain_len = cp - prompt_len
                ctrl_chain_positions = np.arange(prompt_len, cp)
                ctrl_rel = (ctrl_chain_positions - prompt_len) / chain_len
                ctrl_vl = np.array([vnums_lookup(rp, l) for rp in ctrl_rel])
                ctrl_pa = ctrl_attn_kv[:, prompt_len:cp]
                ctrl_raw_list.append({
                    'rel_pos': ctrl_rel, 'v_nums': ctrl_vl, 'attn': ctrl_pa,
                })

            control_attn[l].append(np.mean(ctrl_binned_list, axis=0))
            raw_control_attn[l].append(ctrl_raw_list)

        valid_problems.append(gen)
        del outputs
        torch.cuda.empty_cache()

        if (pi + 1) % 25 == 0:
            print(f"  {pi+1}/{len(correct_gens)} extracted, "
                  f"{len(valid_problems)} valid, {time.time()-t0:.0f}s")

    print(f"\nAttention extracted: {len(valid_problems)} valid problems")

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    n_valid = len(valid_problems)
    if n_valid < 20:
        print(f"ERROR: Only {n_valid} valid problems — too few for analysis")
        sys.exit(1)

    # ═══════════════════════════════════════════════════
    # PHASE 3: Partial correlation analysis (LINEAR + QUADRATIC position control)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 3: Partial correlation (attention × V|nums | position)")
    print(f"  Linear AND quadratic position control — addresses exp_096 confound #1")
    print(f"{'='*70}")

    all_results = {'config': {
        'model': MODEL_NAME, 'n_problems_generated': len(generations),
        'n_correct': n_correct, 'accuracy_pct': round(100 * n_correct / len(generations), 1),
        'n_valid': n_valid, 'layers': layers_probe, 'n_kv_heads': n_kv_heads,
        'n_kv_groups': N_KV_GROUPS, 'group_size': group_size,
        'vnums_source': 'exp_086',
    }}

    for l in layers_probe:
        ans_partial_r_linear = np.zeros((n_valid, n_kv_heads))
        ctrl_partial_r_linear = np.zeros((n_valid, n_kv_heads))
        ans_partial_r_quad = np.zeros((n_valid, n_kv_heads))
        ctrl_partial_r_quad = np.zeros((n_valid, n_kv_heads))

        for pi in range(n_valid):
            ans_data = raw_answer_attn[l][pi]
            ctrl_data_list = raw_control_attn[l][pi]

            for kv_h in range(n_kv_heads):
                attn_vec = ans_data['attn'][kv_h]
                v_nums_vec = ans_data['v_nums']
                pos_vec = ans_data['rel_pos']

                # Linear position control
                r_lin, _ = partial_corr(attn_vec, v_nums_vec, pos_vec)
                ans_partial_r_linear[pi, kv_h] = r_lin

                # Quadratic position control: [pos, pos^2]
                pos_quad = np.column_stack([pos_vec, pos_vec ** 2])
                r_quad, _ = partial_corr(attn_vec, v_nums_vec, pos_quad)
                ans_partial_r_quad[pi, kv_h] = r_quad

                # Control steps
                ctrl_rs_lin = []
                ctrl_rs_quad = []
                for cd in ctrl_data_list:
                    ca = cd['attn'][kv_h]
                    cv = cd['v_nums']
                    cp = cd['rel_pos']
                    if len(ca) >= 5:
                        cr_lin, _ = partial_corr(ca, cv, cp)
                        ctrl_rs_lin.append(cr_lin)
                        cp_quad = np.column_stack([cp, cp ** 2])
                        cr_quad, _ = partial_corr(ca, cv, cp_quad)
                        ctrl_rs_quad.append(cr_quad)
                if ctrl_rs_lin:
                    ctrl_partial_r_linear[pi, kv_h] = np.mean(ctrl_rs_lin)
                if ctrl_rs_quad:
                    ctrl_partial_r_quad[pi, kv_h] = np.mean(ctrl_rs_quad)

        # Compute statistics for both linear and quadratic
        layer_results = {'layer': l, 'info_key': layer_to_info[l], 'n_problems': n_valid}

        for control_type, ans_pr, ctrl_pr in [
            ('linear', ans_partial_r_linear, ctrl_partial_r_linear),
            ('quadratic', ans_partial_r_quad, ctrl_partial_r_quad),
        ]:
            delta_partial = ans_pr - ctrl_pr
            head_results = {}

            for kv_h in range(n_kv_heads):
                apr = ans_pr[:, kv_h]
                mean_apr = float(np.mean(apr))
                se_apr = float(np.std(apr) / np.sqrt(n_valid))
                t_apr, p_apr = stats.ttest_1samp(apr, 0, alternative='greater')

                dp = delta_partial[:, kv_h]
                mean_dp = float(np.mean(dp))
                se_dp = float(np.std(dp) / np.sqrt(n_valid))
                if np.any(dp != 0):
                    _, p_wilcox_dp = stats.wilcoxon(dp, alternative='greater')
                else:
                    p_wilcox_dp = 1.0

                cpr = ctrl_pr[:, kv_h]

                head_results[f'H{kv_h}'] = {
                    'ans_partial_r': mean_apr,
                    'ans_partial_se': se_apr,
                    'ans_partial_p': float(p_apr),
                    'ctrl_partial_r': float(np.mean(cpr)),
                    'delta_partial': mean_dp,
                    'delta_partial_se': se_dp,
                    'delta_partial_p': float(p_wilcox_dp),
                    'n_ans_positive': int(np.sum(apr > 0)),
                    'n_ans_negative': int(np.sum(apr < 0)),
                }

            layer_results[f'partial_corr_{control_type}'] = head_results

            # Aggregate across heads (for compact reporting)
            all_apr = [head_results[f'H{h}']['ans_partial_r'] for h in range(n_kv_heads)]
            all_ctrl = [head_results[f'H{h}']['ctrl_partial_r'] for h in range(n_kv_heads)]
            n_sig = sum(1 for h in range(n_kv_heads) if head_results[f'H{h}']['ans_partial_p'] < 0.001)
            n_positive_mean = sum(1 for r in all_apr if r > 0)

            layer_results[f'aggregate_{control_type}'] = {
                'mean_ans_partial_r': float(np.mean(all_apr)),
                'min_ans_partial_r': float(np.min(all_apr)),
                'max_ans_partial_r': float(np.max(all_apr)),
                'mean_ctrl_partial_r': float(np.mean(all_ctrl)),
                'n_positive_heads': n_positive_mean,
                'n_significant_p001': n_sig,
                'mean_delta': float(np.mean([head_results[f'H{h}']['delta_partial'] for h in range(n_kv_heads)])),
            }

        # Group-level stats (8 groups of 4 heads for comparison with Qwen)
        group_results = {}
        for control_type in ['linear', 'quadratic']:
            hr = layer_results[f'partial_corr_{control_type}']
            group_stats = {}
            for g in range(N_KV_GROUPS):
                heads_in_group = list(range(g * kv_group_size, (g + 1) * kv_group_size))
                group_apr = [hr[f'H{h}']['ans_partial_r'] for h in heads_in_group]
                group_ctrl = [hr[f'H{h}']['ctrl_partial_r'] for h in heads_in_group]
                group_delta = [hr[f'H{h}']['delta_partial'] for h in heads_in_group]
                group_stats[f'G{g}'] = {
                    'heads': heads_in_group,
                    'mean_ans_r': float(np.mean(group_apr)),
                    'mean_ctrl_r': float(np.mean(group_ctrl)),
                    'mean_delta': float(np.mean(group_delta)),
                }
            group_results[control_type] = group_stats
        layer_results['group_stats'] = group_results

        all_results[f'L{l}'] = layer_results

        # Print summary
        for control_type in ['linear', 'quadratic']:
            agg = layer_results[f'aggregate_{control_type}']
            print(f"\n  Layer {l} ({control_type} position control):")
            print(f"    Mean ans partial_r = {agg['mean_ans_partial_r']:.4f}  "
                  f"[{agg['min_ans_partial_r']:.4f}, {agg['max_ans_partial_r']:.4f}]")
            print(f"    Mean ctrl partial_r = {agg['mean_ctrl_partial_r']:.4f}")
            print(f"    Positive heads: {agg['n_positive_heads']}/{n_kv_heads}  "
                  f"Significant (p<0.001): {agg['n_significant_p001']}/{n_kv_heads}")
            print(f"    Mean delta (answer-control): {agg['mean_delta']:.4f}")

        # Print group-level (comparable to Qwen's KV heads)
        print(f"\n    Group-level (quadratic control):")
        print(f"    {'Group':<8} {'ans_r':>7} {'ctrl_r':>7} {'delta':>7}")
        for g in range(N_KV_GROUPS):
            gs = group_results['quadratic'][f'G{g}']
            print(f"    G{g} (H{gs['heads'][0]:02d}-H{gs['heads'][-1]:02d})"
                  f"  {gs['mean_ans_r']:>7.4f} {gs['mean_ctrl_r']:>7.4f} {gs['mean_delta']:>7.4f}")

    # ═══════════════════════════════════════════════════
    # PHASE 3b: Ecological correlation (binned, for visualization)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 3b: Ecological correlation (binned)")
    print(f"{'='*70}")

    for l in layers_probe:
        ans_arr = np.array(answer_attn[l])  # (n_prob, n_kv, n_bins+1)
        ctrl_arr = np.array(control_attn[l])
        delta_arr = ans_arr[:, :, :N_BINS] - ctrl_arr[:, :, :N_BINS]
        mean_delta = delta_arr.mean(axis=0)  # (n_kv, n_bins)

        info_key = layer_to_info[l]
        info = info_landscapes[info_key]

        # Average delta across all heads
        mean_delta_all = mean_delta.mean(axis=0)  # (n_bins,)
        # Ecological correlation with V|nums
        eco_r_all, eco_p_all = stats.pearsonr(mean_delta_all, info['V_nums_R'])

        # Per-group ecological correlation
        eco_group = {}
        for g in range(N_KV_GROUPS):
            heads_in_group = list(range(g * kv_group_size, (g + 1) * kv_group_size))
            group_delta = mean_delta[heads_in_group].mean(axis=0)
            r, p = stats.pearsonr(group_delta, info['V_nums_R'])
            eco_group[f'G{g}'] = {'r': float(r), 'p': float(p)}

        # Landscape comparison (all heads averaged)
        landscape_comp = {}
        for name, vals in info.items():
            r, p = stats.pearsonr(mean_delta_all, vals)
            landscape_comp[name] = {'r': float(r), 'p': float(p)}

        all_results[f'L{l}']['ecological'] = {
            'mean_r_all_heads': float(eco_r_all),
            'mean_p_all_heads': float(eco_p_all),
            'per_group': eco_group,
            'landscape_comparison': landscape_comp,
            'mean_delta_all': mean_delta_all.tolist(),
        }

        print(f"\n  Layer {l}: Ecological r(Δattn, V|nums) = {eco_r_all:.3f} (p={eco_p_all:.4f})")
        print(f"    V|nums: {landscape_comp['V_nums_R']['r']:.3f}  "
              f"V_R: {landscape_comp['V_R']['r']:.3f}  "
              f"nums_R: {landscape_comp['nums_R']['r']:.3f}")
        print(f"    Per-group: " + "  ".join(
            f"G{g}={eco_group[f'G{g}']['r']:.3f}" for g in range(N_KV_GROUPS)))

    # ═══════════════════════════════════════════════════
    # PHASE 4: Cross-model comparison summary
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 4: Cross-model comparison (Phi vs Qwen from exp_096)")
    print(f"{'='*70}")

    # Load exp_096 results for comparison
    exp096_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_096" / "results.json"
    qwen_comparison = {}
    if exp096_path.exists():
        with open(exp096_path) as f:
            exp096_raw = json.load(f)
        # exp_096 nests layer data under 'results'
        exp096 = exp096_raw.get('results', exp096_raw)
        print("\n  Qwen (exp_096) vs Phi (this experiment):")
        print(f"  {'Layer':>8} {'Qwen ans_r':>12} {'Qwen ctrl_r':>12} {'Phi ans_r':>12} {'Phi ctrl_r':>12}")
        for ql, pl in [(9, 8), (18, 16), (27, 24), (35, 31)]:
            qwen_key = f'L{ql}'
            phi_key = f'L{pl}'
            if qwen_key in exp096 and 'partial_corr' in exp096[qwen_key]:
                # Qwen: mean across 8 KV heads
                qwen_heads = exp096[qwen_key]['partial_corr']
                qwen_ans = np.mean([qwen_heads[f'H{h}']['ans_partial_r'] for h in range(8)])
                qwen_ctrl = np.mean([qwen_heads[f'H{h}']['ctrl_partial_r'] for h in range(8)])
            else:
                qwen_ans, qwen_ctrl = float('nan'), float('nan')

            phi_agg = all_results[phi_key]['aggregate_linear']
            phi_ans = phi_agg['mean_ans_partial_r']
            phi_ctrl = phi_agg['mean_ctrl_partial_r']

            print(f"  L{ql:02d}/L{pl:02d}    {qwen_ans:>10.4f}   {qwen_ctrl:>10.4f}   "
                  f"{phi_ans:>10.4f}   {phi_ctrl:>10.4f}")

            qwen_comparison[f'{ql}_{pl}'] = {
                'qwen_ans': float(qwen_ans), 'qwen_ctrl': float(qwen_ctrl),
                'phi_ans': phi_ans, 'phi_ctrl': phi_ctrl,
            }
    all_results['cross_model_comparison'] = qwen_comparison

    # ═══════════════════════════════════════════════════
    # PHASE 5: Visualization
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 5: Visualization")
    print(f"{'='*70}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Figure 1: Partial correlation summary (linear vs quadratic) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, l in enumerate(layers_probe):
        ax = axes[ax_idx // 2, ax_idx % 2]
        lr = all_results[f'L{l}']

        # Get per-group values for linear and quadratic
        groups = range(N_KV_GROUPS)
        lin_vals = [lr['group_stats']['linear'][f'G{g}']['mean_ans_r'] for g in groups]
        quad_vals = [lr['group_stats']['quadratic'][f'G{g}']['mean_ans_r'] for g in groups]
        lin_ctrl = [lr['group_stats']['linear'][f'G{g}']['mean_ctrl_r'] for g in groups]
        quad_ctrl = [lr['group_stats']['quadratic'][f'G{g}']['mean_ctrl_r'] for g in groups]

        x = np.arange(N_KV_GROUPS)
        w = 0.2
        ax.bar(x - 1.5*w, lin_vals, w, label='Answer (linear)', color='steelblue', alpha=0.8)
        ax.bar(x - 0.5*w, quad_vals, w, label='Answer (quadratic)', color='navy', alpha=0.8)
        ax.bar(x + 0.5*w, lin_ctrl, w, label='Control (linear)', color='salmon', alpha=0.6)
        ax.bar(x + 1.5*w, quad_ctrl, w, label='Control (quadratic)', color='darkred', alpha=0.6)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Head Group')
        ax.set_ylabel('Partial r (attn, V|nums | position)')
        ax.set_title(f'Layer {l}: Answer vs Control (Phi-3.5-mini)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'G{g}' for g in groups])
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'partial_correlation_summary.png', dpi=150)
    plt.close(fig)
    print("  Saved partial_correlation_summary.png")

    # ── Figure 2: Cross-model comparison ──
    if qwen_comparison:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        layer_pairs = [(9, 8), (18, 16), (27, 24), (35, 31)]
        x = np.arange(len(layer_pairs))
        qwen_ans_vals = []
        qwen_ctrl_vals = []
        phi_ans_vals = []
        phi_ctrl_vals = []
        labels = []
        for ql, pl in layer_pairs:
            key = f'{ql}_{pl}'
            if key in qwen_comparison:
                qwen_ans_vals.append(qwen_comparison[key]['qwen_ans'])
                qwen_ctrl_vals.append(qwen_comparison[key]['qwen_ctrl'])
                phi_ans_vals.append(qwen_comparison[key]['phi_ans'])
                phi_ctrl_vals.append(qwen_comparison[key]['phi_ctrl'])
                labels.append(f'L{ql}/L{pl}')

        w = 0.2
        ax.bar(x - 1.5*w, qwen_ans_vals, w, label='Qwen Answer', color='steelblue')
        ax.bar(x - 0.5*w, phi_ans_vals, w, label='Phi Answer', color='seagreen')
        ax.bar(x + 0.5*w, qwen_ctrl_vals, w, label='Qwen Control', color='salmon', alpha=0.6)
        ax.bar(x + 1.5*w, phi_ctrl_vals, w, label='Phi Control', color='lightgreen', alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Layer (Qwen / Phi)')
        ax.set_ylabel('Mean partial r (attn, V|nums | position)')
        ax.set_title('Cross-Model: Information-Directed Attention\nQwen3-4B-Base (GQA) vs Phi-3.5-mini (MHA)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'cross_model_comparison.png', dpi=150)
        plt.close(fig)
        print("  Saved cross_model_comparison.png")

    # ── Figure 3: Delta attention vs V|nums landscape overlay ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, l in enumerate(layers_probe):
        ax = axes[ax_idx // 2, ax_idx % 2]
        info_key = layer_to_info[l]
        v_nums = info_landscapes[info_key]['V_nums_R']
        eco = all_results[f'L{l}']['ecological']
        mean_delta = np.array(eco['mean_delta_all'])

        ax2 = ax.twinx()
        ax.bar(bin_centers, mean_delta, width=4, alpha=0.6, color='steelblue',
               label='Δ attention (answer - control)')
        ax2.plot(bin_centers, v_nums, 'ro-', linewidth=2, markersize=4,
                 label='V|nums (exp_086)')
        ax.set_xlabel('Chain position (%)')
        ax.set_ylabel('Δ attention', color='steelblue')
        ax2.set_ylabel('V|nums (Pearson R)', color='red')
        eco_r = eco['mean_r_all_heads']
        ax.set_title(f'Layer {l}: Δattn vs V|nums  r={eco_r:.3f}')
        if ax_idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'delta_attention_vs_vnums.png', dpi=150)
    plt.close(fig)
    print("  Saved delta_attention_vs_vnums.png")

    # ── Figure 4: Linear vs quadratic robustness ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    lin_means = []
    quad_means = []
    layer_labels = []
    for l in layers_probe:
        lr = all_results[f'L{l}']
        lin_means.append(lr['aggregate_linear']['mean_ans_partial_r'])
        quad_means.append(lr['aggregate_quadratic']['mean_ans_partial_r'])
        layer_labels.append(f'L{l}')
    x = np.arange(len(layers_probe))
    ax.bar(x - 0.15, lin_means, 0.3, label='Linear control', color='steelblue')
    ax.bar(x + 0.15, quad_means, 0.3, label='Quadratic control', color='navy')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean partial r (answer step)')
    ax.set_title('Robustness: Linear vs Quadratic Position Control\nPhi-3.5-mini-Instruct')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'linear_vs_quadratic_control.png', dpi=150)
    plt.close(fig)
    print("  Saved linear_vs_quadratic_control.png")

    # ── Figure 5: Per-head heatmap (answer partial_r, quadratic control) ──
    fig, axes = plt.subplots(1, len(layers_probe), figsize=(20, 5))
    if len(layers_probe) == 1:
        axes = [axes]
    for ax_idx, l in enumerate(layers_probe):
        ax = axes[ax_idx]
        hr = all_results[f'L{l}']['partial_corr_quadratic']
        vals = [hr[f'H{h}']['ans_partial_r'] for h in range(n_kv_heads)]
        # Reshape to 4x8 grid for visualization
        n_rows = 4
        n_cols = n_kv_heads // n_rows
        grid = np.array(vals).reshape(n_rows, n_cols)
        im = ax.imshow(grid, cmap='RdBu_r', vmin=-0.15, vmax=0.35, aspect='auto')
        ax.set_title(f'L{l} quadratic partial_r')
        ax.set_xlabel('Head (within row)')
        ax.set_ylabel('Head row')
        for i in range(n_rows):
            for j in range(n_cols):
                h_idx = i * n_cols + j
                ax.text(j, i, f'{vals[h_idx]:.2f}', ha='center', va='center', fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle('Per-Head Partial r(attn, V|nums | pos+pos²) at Answer Step — Phi-3.5-mini', fontsize=12)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'per_head_heatmap.png', dpi=150)
    plt.close(fig)
    print("  Saved per_head_heatmap.png")

    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

    # ═══════════════════════════════════════════════════
    # Final summary
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {len(generations)} generated, {n_correct} correct, {n_valid} with attention")
    print(f"Time: {time.time()-t0:.0f}s")

    print(f"\nPer-layer summary (quadratic position control):")
    print(f"{'Layer':>6} {'Mean ans_r':>12} {'Mean ctrl_r':>12} {'Delta':>8} "
          f"{'Pos/Total':>10} {'Sig(p<.001)':>12}")
    for l in layers_probe:
        agg = all_results[f'L{l}']['aggregate_quadratic']
        print(f"  L{l:02d}   {agg['mean_ans_partial_r']:>10.4f}   "
              f"{agg['mean_ctrl_partial_r']:>10.4f}   {agg['mean_delta']:>6.4f}   "
              f"{agg['n_positive_heads']:>3}/{n_kv_heads:<3}     "
              f"{agg['n_significant_p001']:>3}/{n_kv_heads}")

    # Cross-model
    if qwen_comparison:
        print(f"\nCross-model answer-step partial_r (linear control):")
        print(f"{'Layers':>10} {'Qwen':>8} {'Phi':>8} {'Ratio':>8}")
        for ql, pl in [(9, 8), (18, 16), (27, 24), (35, 31)]:
            key = f'{ql}_{pl}'
            if key in qwen_comparison:
                qa = qwen_comparison[key]['qwen_ans']
                pa = qwen_comparison[key]['phi_ans']
                ratio = pa / qa if abs(qa) > 0.001 else float('nan')
                print(f"  L{ql:02d}/L{pl:02d}  {qa:>7.4f}  {pa:>7.4f}  {ratio:>7.2f}x")

    print(f"\nLinear vs quadratic robustness:")
    for l in layers_probe:
        lr = all_results[f'L{l}']
        lin = lr['aggregate_linear']['mean_ans_partial_r']
        quad = lr['aggregate_quadratic']['mean_ans_partial_r']
        pct = 100 * quad / lin if abs(lin) > 0.001 else float('nan')
        print(f"  L{l}: linear={lin:.4f}  quadratic={quad:.4f}  ({pct:.0f}% retained)")


if __name__ == '__main__':
    main()
