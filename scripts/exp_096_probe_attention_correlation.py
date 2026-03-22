#!/usr/bin/env python3
"""
Experiment 096: Probe-Attention Correlation — Does the Model Retrieve Hidden Info?

Bridges two Phase 2 findings:
- Probing (exps 083-089): V|nums encodes answer info beyond text at each chain position
- Attention (exp 095): Model shifts attention to late-chain at the answer step

Core question: Does the model's answer-step attention ALIGN with the V|nums
information landscape? If so, the model isn't just reading recent text — it's
retrieving from positions that encode hidden information.

Method:
1. Generate CoT with Qwen3-4B-Base (200 problems)
2. Re-encode correct problems with output_attentions=True at 4 key layers
3. Extract attention at answer step and 5 control steps per problem
4. Bin attention by source position (20 bins)
5. Compute delta_attention = answer_attn - control_attn per bin per KV head
6. Load pre-computed V|nums from exp_084 as "information landscape"
7. Ecological correlation: r(delta_attention, V|nums) across 20 bins per head
8. Per-problem paired test: is attention-V|nums alignment higher at answer step?
9. Compare answer heads (H0, H5) vs dispensable heads
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
MODEL_NAME = 'Qwen/Qwen3-4B-Base'
N_PROBLEMS = 200
N_BINS = 20
N_CONTROL_STEPS = 5
# 4 layers: early ramp (9), mid-plateau (18), deep plateau (27), final (35)
LAYERS_PROBE = [9, 18, 27, 35]

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_096"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-computed V|nums from exp_084 (Qwen3-4B-Base, ~180 correct GSM8K problems)
EXP084_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_084" / "results.json"

# ── Plain-text 8-shot exemplars (same as exp_095) ──
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
    """
    Bin attention weights into chain bins + prompt fraction.

    attn_row: 1D array of attention weights (length = positions attended to)
    Returns: array of length n_bins + 1 ([chain bins..., prompt_frac])
    """
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


def main():
    t0 = time.time()

    print("=" * 70)
    print("Experiment 096: Probe-Attention Correlation")
    print("Does the model retrieve from positions encoding hidden information?")
    print("=" * 70)

    # ═══════════════════════════════════════════════════
    # Load pre-computed V|nums from exp_084
    # ═══════════════════════════════════════════════════
    print("\nLoading V|nums information landscape from exp_084...")
    with open(EXP084_PATH) as f:
        exp084 = json.load(f)

    info_landscapes = {}
    for layer_key in ['L18', 'L27']:
        res = exp084['results'][layer_key]
        info_landscapes[layer_key] = {
            'V_R': np.array(res['V_R']),
            'V_nums_R': np.array(res['V_nums_R']),
            'nums_R': np.array(res['nums_R']),
            'K_nums_R': np.array(res['K_nums_R']),
        }
        print(f"  {layer_key}: V|nums mean={np.mean(res['V_nums_R']):.3f}, "
              f"V_R mean={np.mean(res['V_R']):.3f}, "
              f"nums_R mean={np.mean(res['nums_R']):.3f}")

    # Map experiment layers to exp_084 layers for correlation
    # L9 and L35 don't have exp_084 data, use L18 and L27
    layer_to_info = {9: 'L18', 18: 'L18', 27: 'L27', 35: 'L27'}

    # ═══════════════════════════════════════════════════
    # Load model
    # ═══════════════════════════════════════════════════
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager'
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

    # Data structure: per problem, per layer, per KV head → binned attention at answer/ctrl
    # answer_attn[layer][problem] = (n_kv_heads, n_bins+1)
    # control_attn[layer][problem] = (n_kv_heads, n_bins+1) — averaged over control steps
    answer_attn = {l: [] for l in layers_probe}
    control_attn = {l: [] for l in layers_probe}
    # Also store position-resolved data for partial correlation analysis
    # raw_answer_attn[layer][problem] = list of (rel_pos, v_nums_val, attn_per_head)
    raw_answer_attn = {l: [] for l in layers_probe}
    raw_control_attn = {l: [] for l in layers_probe}
    valid_problems = []

    # Build V|nums lookup function (map relative position to V|nums value)
    v_nums_l18 = info_landscapes['L18']['V_nums_R']
    v_nums_l27 = info_landscapes['L27']['V_nums_R']
    def vnums_lookup(rel_pos, layer):
        """Look up V|nums value for a relative chain position (0-1)."""
        v_nums = v_nums_l18 if layer <= 18 else v_nums_l27
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

        # Select control positions (mid-chain, evenly spaced 20-80%)
        ctrl_fracs = [0.25, 0.40, 0.50, 0.60, 0.75]
        ctrl_positions = [prompt_len + int(f * chain_len) for f in ctrl_fracs]
        # Ensure control positions are within chain and have enough context
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

        # Extract attention at answer and control positions for each layer
        for l in layers_probe:
            attn = outputs.attentions[l]  # (1, n_q_heads, seq_len, seq_len)

            # Answer step: attention from answer_pos to all positions
            ans_attn_q = attn[0, :, answer_pos, :answer_pos + 1].float().cpu().numpy()
            # Average across Q heads within each KV group → (n_kv_heads, seq_len)
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

            # Store position-resolved answer attention for partial correlation
            chain_positions = np.arange(prompt_len, answer_pos)
            rel_positions = (chain_positions - prompt_len) / chain_len
            v_lookup = np.array([vnums_lookup(rp, l) for rp in rel_positions])
            # Attention from answer step to each chain position, per KV head
            pos_attn = ans_attn_kv[:, prompt_len:answer_pos]  # (n_kv, chain_len)
            raw_answer_attn[l].append({
                'rel_pos': rel_positions,
                'v_nums': v_lookup,
                'attn': pos_attn,  # (n_kv_heads, chain_len)
            })

            # Control steps: average attention from control positions
            ctrl_binned_list = []
            ctrl_raw_list = []
            for cp in ctrl_positions:
                ctrl_attn_q = attn[0, :, cp, :cp + 1].float().cpu().numpy()
                ctrl_attn_kv = np.zeros((n_kv_heads, cp + 1))
                for kv_h in range(n_kv_heads):
                    q_start = kv_h * group_size
                    q_end = (kv_h + 1) * group_size
                    ctrl_attn_kv[kv_h] = ctrl_attn_q[q_start:q_end].mean(axis=0)

                # Bin control attention in absolute chain coordinates (zero-padded)
                cb = np.zeros((n_kv_heads, N_BINS + 1))
                for kv_h in range(n_kv_heads):
                    cb[kv_h] = bin_attention(
                        np.concatenate([ctrl_attn_kv[kv_h], np.zeros(answer_pos + 1 - (cp + 1))]),
                        prompt_len, answer_pos, N_BINS)
                ctrl_binned_list.append(cb)

                # Position-resolved control attention (only for positions ≤ cp)
                ctrl_chain_len = cp - prompt_len
                ctrl_chain_positions = np.arange(prompt_len, cp)
                ctrl_rel = (ctrl_chain_positions - prompt_len) / chain_len  # absolute chain fraction
                ctrl_vl = np.array([vnums_lookup(rp, l) for rp in ctrl_rel])
                ctrl_pa = ctrl_attn_kv[:, prompt_len:cp]  # (n_kv, ctrl_chain_len)
                ctrl_raw_list.append({
                    'rel_pos': ctrl_rel, 'v_nums': ctrl_vl, 'attn': ctrl_pa,
                })

            control_attn[l].append(np.mean(ctrl_binned_list, axis=0))
            raw_control_attn[l].append(ctrl_raw_list)

        valid_problems.append(gen)
        del outputs
        torch.cuda.empty_cache()

        if (pi + 1) % 50 == 0:
            print(f"  {pi+1}/{len(correct_gens)} extracted, "
                  f"{len(valid_problems)} valid, {time.time()-t0:.0f}s")

    print(f"\nAttention extracted: {len(valid_problems)} valid problems")

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ═══════════════════════════════════════════════════
    # PHASE 3: Compute delta_attention and correlations
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 3: Correlation analysis")
    print(f"{'='*70}")

    n_valid = len(valid_problems)
    all_results = {}

    for l in layers_probe:
        # Stack into arrays: (n_problems, n_kv_heads, n_bins+1)
        ans_arr = np.array(answer_attn[l])  # (n_prob, n_kv, n_bins+1)
        ctrl_arr = np.array(control_attn[l])  # (n_prob, n_kv, n_bins+1)

        # Delta attention (answer - control), chain bins only (exclude prompt)
        delta_arr = ans_arr[:, :, :N_BINS] - ctrl_arr[:, :, :N_BINS]
        # Mean across problems
        mean_delta = delta_arr.mean(axis=0)  # (n_kv, n_bins)

        # Also compute mean answer and control profiles
        mean_ans = ans_arr[:, :, :N_BINS].mean(axis=0)
        mean_ctrl = ctrl_arr[:, :, :N_BINS].mean(axis=0)

        # Get info landscape for this layer
        info_key = layer_to_info[l]
        info = info_landscapes[info_key]

        layer_results = {
            'layer': l, 'info_key': info_key, 'n_problems': n_valid,
            'mean_delta': mean_delta.tolist(),
            'mean_ans': mean_ans.tolist(),
            'mean_ctrl': mean_ctrl.tolist(),
        }

        # ── Ecological correlation: r(delta_attention, V|nums) across bins ──
        eco_corr = {}
        for landscape_name, landscape_vals in info.items():
            per_head_r = []
            per_head_p = []
            for kv_h in range(n_kv_heads):
                r, p = stats.pearsonr(mean_delta[kv_h], landscape_vals)
                per_head_r.append(float(r))
                per_head_p.append(float(p))
            eco_corr[landscape_name] = {
                'r': per_head_r, 'p': per_head_p,
                'mean_r': float(np.mean(per_head_r)),
                'h0_r': per_head_r[0], 'h5_r': per_head_r[5],
            }

        layer_results['ecological_corr'] = eco_corr

        # ── Per-problem paired comparison ──
        # For each problem, compute r(attention_profile, V|nums) at answer vs control
        # Use V_nums_R as primary landscape
        v_nums = info['V_nums_R']
        per_problem_ans_corr = np.zeros((n_valid, n_kv_heads))
        per_problem_ctrl_corr = np.zeros((n_valid, n_kv_heads))

        for pi in range(n_valid):
            for kv_h in range(n_kv_heads):
                # Answer-step attention vs V|nums
                ans_profile = ans_arr[pi, kv_h, :N_BINS]
                ctrl_profile = ctrl_arr[pi, kv_h, :N_BINS]

                # Only correlate if enough variance
                if np.std(ans_profile) > 1e-10 and np.std(v_nums) > 1e-10:
                    r_ans, _ = stats.pearsonr(ans_profile, v_nums)
                    per_problem_ans_corr[pi, kv_h] = r_ans
                if np.std(ctrl_profile) > 1e-10:
                    r_ctrl, _ = stats.pearsonr(ctrl_profile, v_nums)
                    per_problem_ctrl_corr[pi, kv_h] = r_ctrl

        # Delta correlation: answer_corr - control_corr per problem
        delta_corr = per_problem_ans_corr - per_problem_ctrl_corr  # (n_prob, n_kv)

        paired_results = {}
        for kv_h in range(n_kv_heads):
            dc = delta_corr[:, kv_h]
            mean_dc = float(np.mean(dc))
            se_dc = float(np.std(dc) / np.sqrt(n_valid))
            # Wilcoxon signed-rank test (paired, non-parametric)
            if np.any(dc != 0):
                stat, p_wilcox = stats.wilcoxon(dc, alternative='greater')
            else:
                stat, p_wilcox = 0.0, 1.0
            # Also t-test
            t_stat, p_ttest = stats.ttest_1samp(dc, 0, alternative='greater')

            paired_results[f'H{kv_h}'] = {
                'mean_delta_corr': mean_dc,
                'se': se_dc,
                'ci95': [mean_dc - 1.96 * se_dc, mean_dc + 1.96 * se_dc],
                'wilcoxon_p': float(p_wilcox),
                'ttest_p': float(p_ttest),
                'mean_ans_corr': float(np.mean(per_problem_ans_corr[:, kv_h])),
                'mean_ctrl_corr': float(np.mean(per_problem_ctrl_corr[:, kv_h])),
                'n_positive': int(np.sum(dc > 0)),
                'n_negative': int(np.sum(dc < 0)),
            }

        layer_results['paired_test'] = paired_results

        # ── Bootstrap ecological correlation significance ──
        n_boot = 1000
        boot_corr = {}
        for landscape_name in ['V_nums_R', 'V_R', 'nums_R']:
            landscape_vals = info[landscape_name]
            head_boot = {}
            for kv_h in range(n_kv_heads):
                observed_r = eco_corr[landscape_name]['r'][kv_h]
                boot_r = []
                for _ in range(n_boot):
                    idx = np.random.choice(n_valid, n_valid, replace=True)
                    boot_delta = delta_arr[idx, kv_h, :].mean(axis=0)
                    r, _ = stats.pearsonr(boot_delta, landscape_vals)
                    boot_r.append(r)
                boot_r = np.array(boot_r)
                ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
                p_boot = float(np.mean(boot_r <= 0))  # one-tailed: fraction ≤ 0
                head_boot[f'H{kv_h}'] = {
                    'r': observed_r, 'ci': [float(ci_lo), float(ci_hi)],
                    'p_boot': p_boot,
                }
            boot_corr[landscape_name] = head_boot

        layer_results['bootstrap_eco'] = boot_corr

        all_results[f'L{l}'] = layer_results

        # Print summary for this layer
        print(f"\n  Layer {l} (info from {info_key}):")
        print(f"  {'Head':<5} {'delta_V|nums_r':>14} {'eco_p':>8} {'paired_dc':>10} "
              f"{'wilcox_p':>9} {'ans_r':>7} {'ctrl_r':>7}")
        for kv_h in range(n_kv_heads):
            hname = f'H{kv_h}'
            eco_r = eco_corr['V_nums_R']['r'][kv_h]
            eco_p = boot_corr['V_nums_R'][hname]['p_boot']
            pr = paired_results[hname]
            tag = ""
            if kv_h == 0:
                tag = " ← H0"
            elif kv_h == 5:
                tag = " ← H5"
            print(f"  {hname:<5} {eco_r:>14.3f} {eco_p:>8.4f} {pr['mean_delta_corr']:>10.4f} "
                  f"{pr['wilcoxon_p']:>9.4f} {pr['mean_ans_corr']:>7.3f} "
                  f"{pr['mean_ctrl_corr']:>7.3f}{tag}")

    # ═══════════════════════════════════════════════════
    # PHASE 3b: Position-resolved PARTIAL correlation (controls for position)
    # This is the PRIMARY analysis — immune to the positional confound
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 3b: Partial correlation (attention × V|nums | position)")
    print("  Does the model attend to high-V|nums positions BEYOND recency?")
    print(f"{'='*70}")

    from sklearn.linear_model import LinearRegression

    def partial_corr_1d(x, y, z):
        """Partial correlation of x and y, controlling for z (1D confound)."""
        if len(x) < 5 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0, 1.0
        z = np.array(z).reshape(-1, 1)
        x_resid = x - LinearRegression().fit(z, x).predict(z)
        y_resid = y - LinearRegression().fit(z, y).predict(z)
        if np.std(x_resid) < 1e-10 or np.std(y_resid) < 1e-10:
            return 0.0, 1.0
        return stats.pearsonr(x_resid, y_resid)

    partial_results = {}
    for l in layers_probe:
        # For each problem, compute partial_r(attn, V|nums | position) at answer step
        ans_partial_r = np.zeros((n_valid, n_kv_heads))
        ctrl_partial_r = np.zeros((n_valid, n_kv_heads))

        for pi in range(n_valid):
            ans_data = raw_answer_attn[l][pi]
            ctrl_data_list = raw_control_attn[l][pi]

            for kv_h in range(n_kv_heads):
                # Answer step: partial_r(attn, v_nums | rel_pos) across chain positions
                attn_vec = ans_data['attn'][kv_h]  # (chain_len,)
                v_nums_vec = ans_data['v_nums']  # (chain_len,)
                pos_vec = ans_data['rel_pos']  # (chain_len,)
                r, _ = partial_corr_1d(attn_vec, v_nums_vec, pos_vec)
                ans_partial_r[pi, kv_h] = r

                # Control steps: average partial_r across controls
                ctrl_rs = []
                for cd in ctrl_data_list:
                    ca = cd['attn'][kv_h]
                    cv = cd['v_nums']
                    cp = cd['rel_pos']
                    if len(ca) >= 5:
                        cr, _ = partial_corr_1d(ca, cv, cp)
                        ctrl_rs.append(cr)
                if ctrl_rs:
                    ctrl_partial_r[pi, kv_h] = np.mean(ctrl_rs)

        # Delta partial correlation: answer - control
        delta_partial = ans_partial_r - ctrl_partial_r

        layer_partial = {}
        for kv_h in range(n_kv_heads):
            # Test: is answer partial_r > 0? (hidden channel retrieval)
            apr = ans_partial_r[:, kv_h]
            mean_apr = float(np.mean(apr))
            se_apr = float(np.std(apr) / np.sqrt(n_valid))
            t_apr, p_apr = stats.ttest_1samp(apr, 0, alternative='greater')

            # Test: is answer partial_r > control partial_r?
            dp = delta_partial[:, kv_h]
            mean_dp = float(np.mean(dp))
            se_dp = float(np.std(dp) / np.sqrt(n_valid))
            if np.any(dp != 0):
                _, p_wilcox_dp = stats.wilcoxon(dp, alternative='greater')
            else:
                p_wilcox_dp = 1.0

            cpr = ctrl_partial_r[:, kv_h]

            layer_partial[f'H{kv_h}'] = {
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

        partial_results[f'L{l}'] = layer_partial
        all_results[f'L{l}']['partial_corr'] = layer_partial

        # Print
        print(f"\n  Layer {l}:")
        print(f"  {'Head':<5} {'ans_partial_r':>13} {'p(>0)':>8} "
              f"{'ctrl_partial_r':>14} {'delta':>8} {'p(Δ>0)':>8}")
        for kv_h in range(n_kv_heads):
            hname = f'H{kv_h}'
            p = layer_partial[hname]
            tag = ""
            if kv_h == 0:
                tag = " ← H0"
            elif kv_h == 5:
                tag = " ← H5"
            sig = ""
            if p['ans_partial_p'] < 0.001:
                sig = "***"
            elif p['ans_partial_p'] < 0.01:
                sig = "**"
            elif p['ans_partial_p'] < 0.05:
                sig = "*"
            print(f"  {hname:<5} {p['ans_partial_r']:>10.4f}±{p['ans_partial_se']:.4f} "
                  f"{p['ans_partial_p']:>8.4f}{sig:3s} "
                  f"{p['ctrl_partial_r']:>14.4f} {p['delta_partial']:>8.4f} "
                  f"{p['delta_partial_p']:>8.4f}{tag}")

    # ═══════════════════════════════════════════════════
    # PHASE 4: Visualization
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 4: Visualization")
    print(f"{'='*70}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bin_centers = np.array(exp084['bin_centers'])

    # ── Figure 1: Delta attention vs V|nums overlay ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, l in enumerate(layers_probe):
        ax = axes[ax_idx // 2, ax_idx % 2]
        info_key = layer_to_info[l]
        v_nums = info_landscapes[info_key]['V_nums_R']
        layer_data = all_results[f'L{l}']
        mean_delta = np.array(layer_data['mean_delta'])

        # Plot V|nums landscape (secondary axis)
        ax2 = ax.twinx()
        ax2.fill_between(bin_centers, 0, v_nums, alpha=0.15, color='blue',
                         label='V|nums (exp_084)')
        ax2.set_ylabel('V|nums R', color='blue', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(-0.1, 0.6)

        # Plot delta attention for key heads
        for kv_h, color, label in [(5, 'red', 'H5 (answer)'),
                                     (0, 'green', 'H0 (answer)'),
                                     (7, 'gray', 'H7 (dispensable)')]:
            r_val = layer_data['ecological_corr']['V_nums_R']['r'][kv_h]
            ax.plot(bin_centers, mean_delta[kv_h], color=color, linewidth=2,
                    label=f'{label} r={r_val:.2f}', marker='o', markersize=3)

        # Mean of all heads
        all_mean = mean_delta.mean(axis=0)
        r_all = float(stats.pearsonr(all_mean, v_nums)[0])
        ax.plot(bin_centers, all_mean, color='black', linewidth=1.5,
                linestyle='--', label=f'All heads r={r_all:.2f}')

        ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
        ax.set_xlabel('Chain position (%)')
        ax.set_ylabel('Δ attention (answer − control)')
        ax.set_title(f'Layer {l} (V|nums from {info_key})')
        ax.legend(fontsize=7, loc='upper left')

    fig.suptitle('Delta Attention vs V|nums Information Landscape\n'
                 'Does the answer-step attention shift align with hidden info?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'delta_attention_vs_vnums.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: Per-head ecological correlation bar chart ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, l in enumerate(layers_probe):
        ax = axes[ax_idx // 2, ax_idx % 2]
        layer_data = all_results[f'L{l}']

        # V|nums correlation with bootstrap CIs
        boot = layer_data['bootstrap_eco']['V_nums_R']
        heads = list(range(n_kv_heads))
        r_vals = [boot[f'H{h}']['r'] for h in heads]
        ci_lo = [boot[f'H{h}']['ci'][0] for h in heads]
        ci_hi = [boot[f'H{h}']['ci'][1] for h in heads]
        err_lo = [r_vals[i] - ci_lo[i] for i in range(len(heads))]
        err_hi = [ci_hi[i] - r_vals[i] for i in range(len(heads))]
        p_vals = [boot[f'H{h}']['p_boot'] for h in heads]

        colors = ['green' if h == 0 else 'red' if h == 5 else 'steelblue' for h in heads]
        bars = ax.bar(heads, r_vals, yerr=[err_lo, err_hi], color=colors,
                      capsize=3, edgecolor='black', linewidth=0.5)

        # Significance stars
        for i, p in enumerate(p_vals):
            if p < 0.001:
                ax.text(i, r_vals[i] + err_hi[i] + 0.02, '***', ha='center', fontsize=8)
            elif p < 0.01:
                ax.text(i, r_vals[i] + err_hi[i] + 0.02, '**', ha='center', fontsize=8)
            elif p < 0.05:
                ax.text(i, r_vals[i] + err_hi[i] + 0.02, '*', ha='center', fontsize=8)

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('KV Head')
        ax.set_ylabel('r(Δattn, V|nums)')
        ax.set_title(f'Layer {l}')
        ax.set_xticks(heads)
        ax.set_xticklabels([f'H{h}' for h in heads])

    fig.suptitle('Ecological Correlation: Δ Attention × V|nums\n'
                 'Green=H0 (answer), Red=H5 (answer), Blue=dispensable',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'ecological_correlation_per_head.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3: Paired delta-correlation comparison ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, l in enumerate(layers_probe):
        ax = axes[ax_idx // 2, ax_idx % 2]
        layer_data = all_results[f'L{l}']
        pr = layer_data['paired_test']

        heads = list(range(n_kv_heads))
        dc_vals = [pr[f'H{h}']['mean_delta_corr'] for h in heads]
        se_vals = [pr[f'H{h}']['se'] for h in heads]
        p_vals = [pr[f'H{h}']['wilcoxon_p'] for h in heads]

        colors = ['green' if h == 0 else 'red' if h == 5 else 'steelblue' for h in heads]
        bars = ax.bar(heads, dc_vals, yerr=[1.96 * s for s in se_vals],
                      color=colors, capsize=3, edgecolor='black', linewidth=0.5)

        for i, p in enumerate(p_vals):
            y_pos = dc_vals[i] + 1.96 * se_vals[i] + 0.005
            if dc_vals[i] < 0:
                y_pos = dc_vals[i] - 1.96 * se_vals[i] - 0.015
            if p < 0.001:
                ax.text(i, y_pos, '***', ha='center', fontsize=8)
            elif p < 0.01:
                ax.text(i, y_pos, '**', ha='center', fontsize=8)
            elif p < 0.05:
                ax.text(i, y_pos, '*', ha='center', fontsize=8)

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('KV Head')
        ax.set_ylabel('Δ r(attn, V|nums)\n(answer − control)')
        ax.set_title(f'Layer {l}: answer vs control alignment with V|nums')
        ax.set_xticks(heads)
        ax.set_xticklabels([f'H{h}' for h in heads])

    fig.suptitle('Paired Test: Is answer-step attention MORE aligned with V|nums?\n'
                 'Positive = answer step retrieves more from hidden-info positions',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'paired_delta_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 4: Multi-landscape comparison (V_R vs V|nums vs nums_R) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, l in enumerate([18, 27]):
        ax = axes[ax_idx]
        layer_data = all_results[f'L{l}']
        eco = layer_data['ecological_corr']

        landscapes = ['V_R', 'V_nums_R', 'nums_R', 'K_nums_R']
        labels = ['V_R (raw)', 'V|nums\n(hidden info)', 'nums_R\n(text numbers)', 'K|nums']
        x = np.arange(len(landscapes))
        width = 0.1

        for kv_h in range(n_kv_heads):
            color = 'green' if kv_h == 0 else 'red' if kv_h == 5 else 'lightblue'
            alpha = 1.0 if kv_h in [0, 5] else 0.4
            offset = (kv_h - n_kv_heads / 2 + 0.5) * width
            vals = [eco[ln]['r'][kv_h] for ln in landscapes]
            bars = ax.bar(x + offset, vals, width, color=color, alpha=alpha,
                          edgecolor='black', linewidth=0.3,
                          label=f'H{kv_h}' if kv_h in [0, 5, 7] else '')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_ylabel('r(Δattn, landscape)')
        ax.set_title(f'Layer {l}')
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Which information landscape does answer-step attention align with?\n'
                 'Comparing V_R, V|nums (hidden), nums_R (text), K|nums',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'multi_landscape_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 5: Answer vs control attention profiles for H5 and H0 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, l in enumerate(layers_probe):
        ax = axes[ax_idx // 2, ax_idx % 2]
        layer_data = all_results[f'L{l}']
        mean_ans = np.array(layer_data['mean_ans'])
        mean_ctrl = np.array(layer_data['mean_ctrl'])

        for kv_h, color, ls in [(5, 'red', '-'), (0, 'green', '-'),
                                  (7, 'gray', '--')]:
            name = f'H{kv_h}'
            ax.plot(bin_centers, mean_ans[kv_h], color=color, linestyle=ls,
                    linewidth=2, label=f'{name} answer', marker='o', markersize=2)
            ax.plot(bin_centers, mean_ctrl[kv_h], color=color, linestyle=':',
                    linewidth=1.5, label=f'{name} control', alpha=0.6)

        ax.set_xlabel('Chain position (%)')
        ax.set_ylabel('Attention fraction')
        ax.set_title(f'Layer {l}')
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle('Answer vs Control Attention Profiles\n'
                 'H0 (green), H5 (red), H7 (gray) — solid=answer, dotted=control',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'answer_vs_control_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 6: Partial correlation (PRIMARY RESULT) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, l in enumerate(layers_probe):
        ax = axes[ax_idx // 2, ax_idx % 2]
        pc = partial_results[f'L{l}']
        heads = list(range(n_kv_heads))

        ans_vals = [pc[f'H{h}']['ans_partial_r'] for h in heads]
        ans_se = [pc[f'H{h}']['ans_partial_se'] for h in heads]
        ctrl_vals = [pc[f'H{h}']['ctrl_partial_r'] for h in heads]
        p_vals = [pc[f'H{h}']['ans_partial_p'] for h in heads]

        x = np.arange(len(heads))
        width = 0.35
        colors_ans = ['darkgreen' if h == 0 else 'darkred' if h == 5 else 'steelblue' for h in heads]
        colors_ctrl = ['lightgreen' if h == 0 else 'lightsalmon' if h == 5 else 'lightblue' for h in heads]

        bars1 = ax.bar(x - width / 2, ans_vals, width, yerr=[1.96 * s for s in ans_se],
                        color=colors_ans, capsize=3, edgecolor='black', linewidth=0.5,
                        label='Answer step')
        bars2 = ax.bar(x + width / 2, ctrl_vals, width,
                        color=colors_ctrl, edgecolor='black', linewidth=0.5, alpha=0.6,
                        label='Control step')

        for i, p in enumerate(p_vals):
            y_pos = max(ans_vals[i] + 1.96 * ans_se[i], ctrl_vals[i]) + 0.005
            if p < 0.001:
                ax.text(i, y_pos, '***', ha='center', fontsize=8)
            elif p < 0.01:
                ax.text(i, y_pos, '**', ha='center', fontsize=8)
            elif p < 0.05:
                ax.text(i, y_pos, '*', ha='center', fontsize=8)

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('KV Head')
        ax.set_ylabel('Partial r(attn, V|nums | position)')
        ax.set_title(f'Layer {l}')
        ax.set_xticks(heads)
        ax.set_xticklabels([f'H{h}' for h in heads])
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('PRIMARY: Partial Correlation r(attention, V|nums | position)\n'
                 'Does the model attend to hidden-info positions BEYOND recency?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'partial_correlation_primary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Figures saved to", RESULTS_DIR)

    # ═══════════════════════════════════════════════════
    # PHASE 5: Summary
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Best ecological correlation (V|nums)
    print("\n  ECOLOGICAL CORRELATION: r(Δattn, V|nums) across 20 bins")
    for l in layers_probe:
        layer_data = all_results[f'L{l}']
        eco = layer_data['ecological_corr']['V_nums_R']
        boot = layer_data['bootstrap_eco']['V_nums_R']
        sig_heads = [h for h in range(n_kv_heads) if boot[f'H{h}']['p_boot'] < 0.05]
        print(f"  L{l}: mean r={eco['mean_r']:.3f}, H0 r={eco['h0_r']:.3f}, "
              f"H5 r={eco['h5_r']:.3f}, sig heads: {[f'H{h}' for h in sig_heads]}")

    print("\n  PAIRED TEST: Δr(attn, V|nums) — answer vs control")
    for l in layers_probe:
        layer_data = all_results[f'L{l}']
        pr = layer_data['paired_test']
        sig_heads = [h for h in range(n_kv_heads)
                     if pr[f'H{h}']['wilcoxon_p'] < 0.05]
        neg_heads = [h for h in range(n_kv_heads)
                     if pr[f'H{h}']['mean_delta_corr'] < 0]
        print(f"  L{l}: sig positive: {[f'H{h}' for h in sig_heads]}, "
              f"negative: {[f'H{h}' for h in neg_heads]}")

    # Key comparison: do answer heads differ from dispensable?
    print("\n  ANSWER HEADS vs DISPENSABLE HEADS:")
    for l in layers_probe:
        layer_data = all_results[f'L{l}']
        eco = layer_data['ecological_corr']['V_nums_R']
        answer_r = np.mean([eco['r'][0], eco['r'][5]])
        disp_r = np.mean([eco['r'][h] for h in range(n_kv_heads) if h not in [0, 5]])
        pr = layer_data['paired_test']
        answer_dc = np.mean([pr['H0']['mean_delta_corr'], pr['H5']['mean_delta_corr']])
        disp_dc = np.mean([pr[f'H{h}']['mean_delta_corr']
                           for h in range(n_kv_heads) if h not in [0, 5]])
        print(f"  L{l}: eco_r answer={answer_r:.3f} vs disp={disp_r:.3f} "
              f"(diff={answer_r-disp_r:+.3f}); "
              f"paired_dc answer={answer_dc:.4f} vs disp={disp_dc:.4f}")

    # PRIMARY: Partial correlation
    print("\n  *** PRIMARY RESULT: PARTIAL CORRELATION r(attn, V|nums | position) ***")
    print("  (Controls for recency — immune to positional confound)")
    for l in layers_probe:
        pc = partial_results[f'L{l}']
        sig_ans = [h for h in range(n_kv_heads) if pc[f'H{h}']['ans_partial_p'] < 0.05]
        mean_ans = np.mean([pc[f'H{h}']['ans_partial_r'] for h in range(n_kv_heads)])
        h0_r = pc['H0']['ans_partial_r']
        h5_r = pc['H5']['ans_partial_r']
        print(f"  L{l}: mean={mean_ans:.4f}, H0={h0_r:.4f}, H5={h5_r:.4f}, "
              f"sig (p<0.05): {[f'H{h}' for h in sig_ans]}")

    # V|nums vs nums_R comparison
    print("\n  HIDDEN vs TEXT INFORMATION:")
    for l in [18, 27]:
        layer_data = all_results[f'L{l}']
        v_nums_r = layer_data['ecological_corr']['V_nums_R']['mean_r']
        nums_r = layer_data['ecological_corr']['nums_R']['mean_r']
        v_r = layer_data['ecological_corr']['V_R']['mean_r']
        print(f"  L{l}: r(Δattn, V_R)={v_r:.3f}, r(Δattn, V|nums)={v_nums_r:.3f}, "
              f"r(Δattn, nums_R)={nums_r:.3f}")

    # Save results
    results_out = {
        'config': {
            'model': MODEL_NAME, 'n_problems_generated': len(generations),
            'n_correct': n_correct, 'n_valid': n_valid,
            'layers': layers_probe, 'n_bins': N_BINS,
            'n_control_steps': N_CONTROL_STEPS,
            'n_bootstrap': 1000,
        },
        'results': all_results,
        'bin_centers': bin_centers.tolist(),
        'exp084_landscapes': {k: {kk: vv.tolist() for kk, vv in v.items()}
                              for k, v in info_landscapes.items()},
    }
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results_out, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\nDone. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
