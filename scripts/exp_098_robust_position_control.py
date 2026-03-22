#!/usr/bin/env python3
"""
Experiment 098: Robust Position Control for Probe-Attention Correlation (Qwen)

Exp_097 revealed that ~80% of the linear partial correlation between attention
and V|nums is non-linear recency (on Phi). This experiment applies four position
control methods to Qwen3-4B-Base to:
1. Replicate exp_096's linear partial correlation (sanity check)
2. Apply quadratic control (new for Qwen — does it match Phi's ~80% reduction?)
3. Apply rank-based (Spearman) partial correlation — non-parametric gold standard
   that removes ALL monotonic position effects without assuming functional form
4. Run permutation null (shuffle V|nums) to establish baseline false positive rate

The key question: After the strongest possible position control, does a genuine
information-directed attention signal survive on Qwen? How does it compare to
Phi's r≈0.054 at L16?
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
N_PERMUTATIONS = 1000
# Same 4 layers as exp_096: early ramp (9), mid-plateau (18), deep plateau (27), final (35)
LAYERS_PROBE = [9, 18, 27, 35]

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_098"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-computed V|nums from exp_084 (Qwen3-4B-Base)
EXP084_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_084" / "results.json"

# ── Plain-text 8-shot exemplars (same as exp_096) ──
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


# ── FOUR POSITION CONTROL METHODS ──

def partial_corr_linear(x, y, pos):
    """Standard linear partial correlation: r(x, y | pos)."""
    from sklearn.linear_model import LinearRegression
    if len(x) < 5 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 1.0
    z = pos.reshape(-1, 1)
    x_resid = x - LinearRegression().fit(z, x).predict(z)
    y_resid = y - LinearRegression().fit(z, y).predict(z)
    if np.std(x_resid) < 1e-10 or np.std(y_resid) < 1e-10:
        return 0.0, 1.0
    return stats.pearsonr(x_resid, y_resid)


def partial_corr_quadratic(x, y, pos):
    """Quadratic partial correlation: r(x, y | pos, pos^2)."""
    from sklearn.linear_model import LinearRegression
    if len(x) < 5 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 1.0
    z = np.column_stack([pos, pos**2])
    x_resid = x - LinearRegression().fit(z, x).predict(z)
    y_resid = y - LinearRegression().fit(z, y).predict(z)
    if np.std(x_resid) < 1e-10 or np.std(y_resid) < 1e-10:
        return 0.0, 1.0
    return stats.pearsonr(x_resid, y_resid)


def partial_corr_rank(x, y, pos):
    """Rank-based (Spearman) partial correlation — non-parametric gold standard.
    Rank-transforms all variables, then computes Pearson partial correlation on ranks.
    Removes ALL monotonic position effects without assuming functional form."""
    if len(x) < 5 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 1.0
    from sklearn.linear_model import LinearRegression
    # Rank-transform all three variables
    x_rank = stats.rankdata(x).astype(np.float64)
    y_rank = stats.rankdata(y).astype(np.float64)
    pos_rank = stats.rankdata(pos).astype(np.float64)
    # Partial correlation on ranks = Spearman partial correlation
    z = pos_rank.reshape(-1, 1)
    x_resid = x_rank - LinearRegression().fit(z, x_rank).predict(z)
    y_resid = y_rank - LinearRegression().fit(z, y_rank).predict(z)
    if np.std(x_resid) < 1e-10 or np.std(y_resid) < 1e-10:
        return 0.0, 1.0
    return stats.pearsonr(x_resid, y_resid)


def main():
    t0 = time.time()

    print("=" * 70)
    print("Experiment 098: Robust Position Control — Probe-Attention Correlation")
    print("Four position control methods on Qwen3-4B-Base")
    print("=" * 70)

    # ═══════════════════════════════════════════════════
    # Load pre-computed V|nums from exp_084 (Qwen3-4B-Base)
    # ═══════════════════════════════════════════════════
    print("\nLoading V|nums information landscape from exp_084 (Qwen)...")
    with open(EXP084_PATH) as f:
        exp084 = json.load(f)

    info_landscapes = {}
    for layer_key in ['L18', 'L27']:
        res = exp084['results'][layer_key]
        info_landscapes[layer_key] = {
            'V_R': np.array(res['V_R']),
            'V_nums_R': np.array(res['V_nums_R']),
            'nums_R': np.array(res['nums_R']),
        }
        print(f"  {layer_key}: V|nums mean={np.mean(res['V_nums_R']):.3f}, "
              f"V_R mean={np.mean(res['V_R']):.3f}, "
              f"nums_R mean={np.mean(res['nums_R']):.3f}")

    # Map experiment layers to landscape layers
    layer_to_info = {9: 'L18', 18: 'L18', 27: 'L27', 35: 'L27'}
    bin_centers = np.array(exp084['bin_centers'])

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
    print("PHASE 2: Extracting per-problem attention profiles")
    print(f"{'='*70}")

    # Store per-problem, per-head, per-layer binned attention profiles
    # Shape per layer: list of dicts with 'ans_profile' and 'ctrl_profiles'
    problem_data = {l: [] for l in layers_probe}

    n_valid = 0
    for gi, g in enumerate(correct_gens):
        if time.time() - t0 > TIME_BUDGET * 0.65:
            print(f"  Time budget (65%) reached at problem {gi}")
            break

        full_ids = torch.tensor(g['full_ids'], dtype=torch.long, device=model.device).unsqueeze(0)
        prompt_len = g['prompt_len']

        ans_pos = find_answer_position(g['full_ids'], prompt_len, tokenizer)
        if ans_pos is None or ans_pos <= prompt_len + 10:
            continue

        chain_len = ans_pos - prompt_len
        # Control positions: 5 evenly spaced in 20-80% of chain
        ctrl_positions = []
        for frac in [0.25, 0.35, 0.50, 0.65, 0.75]:
            cp = prompt_len + int(frac * chain_len)
            if cp < prompt_len + 5 and cp < ans_pos - 2:
                cp = prompt_len + 5
            if cp > ans_pos - 2:
                cp = ans_pos - 2
            ctrl_positions.append(cp)
        ctrl_positions = sorted(set(ctrl_positions))
        if len(ctrl_positions) < 3:
            continue

        all_positions = ctrl_positions + [ans_pos]

        # Forward pass with attention at each target position individually
        # (to avoid memory issues from full sequence attention)
        # Actually, we can do one forward pass and extract attention at all positions
        try:
            with torch.no_grad():
                outputs = model(
                    full_ids[:, :ans_pos + 1],
                    output_attentions=True,
                    use_cache=False,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        attentions = outputs.attentions  # tuple of (1, n_q_heads, seq_len, seq_len)

        prob_data = {}
        for li in layers_probe:
            attn_matrix = attentions[li][0].float().cpu().numpy()  # (n_q_heads, seq, seq)

            # Average across Q heads within each KV group
            kv_attn = np.zeros((n_kv_heads, attn_matrix.shape[1], attn_matrix.shape[2]))
            for kv_h in range(n_kv_heads):
                q_start = kv_h * group_size
                q_end = q_start + group_size
                kv_attn[kv_h] = attn_matrix[q_start:q_end].mean(axis=0)

            # Bin attention at answer position for each KV head
            ans_profiles = np.zeros((n_kv_heads, N_BINS), dtype=np.float64)
            for kv_h in range(n_kv_heads):
                binned = bin_attention(kv_attn[kv_h, ans_pos, :ans_pos + 1],
                                       prompt_len, ans_pos, N_BINS)
                ans_profiles[kv_h] = binned[:N_BINS]

            # Bin attention at control positions
            ctrl_all = np.zeros((n_kv_heads, N_BINS), dtype=np.float64)
            for cp in ctrl_positions:
                for kv_h in range(n_kv_heads):
                    binned = bin_attention(kv_attn[kv_h, cp, :cp + 1],
                                           prompt_len, cp, N_BINS)
                    ctrl_all[kv_h] += binned[:N_BINS]
            ctrl_all /= len(ctrl_positions)

            prob_data[li] = {
                'ans_profiles': ans_profiles,   # (n_kv_heads, N_BINS)
                'ctrl_profiles': ctrl_all,      # (n_kv_heads, N_BINS)
            }

        for li in layers_probe:
            problem_data[li].append(prob_data[li])

        n_valid += 1
        del outputs, attentions
        torch.cuda.empty_cache()

        if (gi + 1) % 25 == 0:
            print(f"  {gi+1}/{len(correct_gens)} correct problems processed, "
                  f"{n_valid} valid, {time.time()-t0:.0f}s")

    print(f"\nAttention extraction complete: {n_valid} valid problems")

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ═══════════════════════════════════════════════════
    # PHASE 3: Four-method position control analysis
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 3: Four-method position control analysis")
    print(f"{'='*70}")

    all_results = {}
    pos_array = bin_centers  # position values for each bin (0.025 to 0.975)

    for li in layers_probe:
        info_key = layer_to_info[li]
        vnums = info_landscapes[info_key]['V_nums_R']
        v_r = info_landscapes[info_key]['V_R']
        nums_r = info_landscapes[info_key]['nums_R']

        print(f"\n--- Layer {li} (info from {info_key}, n_problems={len(problem_data[li])}) ---")

        layer_results = {
            'layer': li,
            'info_key': info_key,
            'n_problems': len(problem_data[li]),
            'per_head': {},
            'aggregate': {},
        }

        # Per-head analysis with all 4 methods
        for kv_h in range(n_kv_heads):
            # Collect per-problem partial correlations for this head
            ans_pcorr = {'linear': [], 'quadratic': [], 'rank': []}
            ctrl_pcorr = {'linear': [], 'quadratic': [], 'rank': []}

            for pd in problem_data[li]:
                ans_prof = pd['ans_profiles'][kv_h]  # (N_BINS,)
                ctrl_prof = pd['ctrl_profiles'][kv_h]  # (N_BINS,)

                # Apply all three partial correlation methods at answer step
                r_lin, p_lin = partial_corr_linear(ans_prof, vnums, pos_array)
                r_quad, p_quad = partial_corr_quadratic(ans_prof, vnums, pos_array)
                r_rank, p_rank = partial_corr_rank(ans_prof, vnums, pos_array)

                ans_pcorr['linear'].append(r_lin)
                ans_pcorr['quadratic'].append(r_quad)
                ans_pcorr['rank'].append(r_rank)

                # Control step
                r_lin_c, _ = partial_corr_linear(ctrl_prof, vnums, pos_array)
                r_quad_c, _ = partial_corr_quadratic(ctrl_prof, vnums, pos_array)
                r_rank_c, _ = partial_corr_rank(ctrl_prof, vnums, pos_array)

                ctrl_pcorr['linear'].append(r_lin_c)
                ctrl_pcorr['quadratic'].append(r_quad_c)
                ctrl_pcorr['rank'].append(r_rank_c)

            head_result = {}
            for method in ['linear', 'quadratic', 'rank']:
                ans_arr = np.array(ans_pcorr[method])
                ctrl_arr = np.array(ctrl_pcorr[method])
                delta = ans_arr - ctrl_arr

                # Wilcoxon test: answer > control
                if len(ans_arr) >= 10:
                    try:
                        w_stat, w_p = stats.wilcoxon(ans_arr, ctrl_arr, alternative='greater')
                    except ValueError:
                        w_stat, w_p = 0, 1.0
                else:
                    w_stat, w_p = 0, 1.0

                # One-sample t-test: mean ans_r > 0
                if len(ans_arr) >= 10 and np.std(ans_arr) > 1e-10:
                    t_stat, t_p = stats.ttest_1samp(ans_arr, 0)
                    t_p_one = t_p / 2 if t_stat > 0 else 1 - t_p / 2
                else:
                    t_stat, t_p_one = 0, 1.0

                head_result[method] = {
                    'mean_ans_r': float(np.mean(ans_arr)),
                    'std_ans_r': float(np.std(ans_arr)),
                    'mean_ctrl_r': float(np.mean(ctrl_arr)),
                    'mean_delta': float(np.mean(delta)),
                    'n_positive': int(np.sum(ans_arr > 0)),
                    'n_total': len(ans_arr),
                    'wilcoxon_p': float(w_p),
                    'ttest_p_onesided': float(t_p_one),
                    'positive': bool(np.mean(ans_arr) > 0),
                    'sig_001': bool(t_p_one < 0.001),
                }

            layer_results['per_head'][f'H{kv_h}'] = head_result

        # ── Aggregate across heads ──
        for method in ['linear', 'quadratic', 'rank']:
            head_means = [layer_results['per_head'][f'H{h}'][method]['mean_ans_r']
                          for h in range(n_kv_heads)]
            head_pos = sum(1 for m in head_means if m > 0)
            head_sig = sum(1 for h in range(n_kv_heads)
                           if layer_results['per_head'][f'H{h}'][method]['sig_001'])

            layer_results['aggregate'][method] = {
                'mean_ans_r': float(np.mean(head_means)),
                'min_head': float(np.min(head_means)),
                'max_head': float(np.max(head_means)),
                'n_positive': head_pos,
                'n_sig_001': head_sig,
                'n_heads': n_kv_heads,
            }

            print(f"  {method:10s}: mean_ans_r={np.mean(head_means):.4f}, "
                  f"positive={head_pos}/{n_kv_heads}, "
                  f"sig(p<.001)={head_sig}/{n_kv_heads}")

        # ── Permutation null distribution ──
        print(f"  Running {N_PERMUTATIONS} permutation null tests...")
        perm_means = {'linear': [], 'quadratic': [], 'rank': []}

        rng = np.random.RandomState(42)
        for perm_i in range(N_PERMUTATIONS):
            # Shuffle V|nums across bins (break position-V|nums association)
            perm_vnums = vnums.copy()
            rng.shuffle(perm_vnums)

            # Compute mean partial correlation across ALL heads and problems
            # (for efficiency, sample a subset of problems)
            sample_size = min(50, len(problem_data[li]))
            sample_idx = rng.choice(len(problem_data[li]), sample_size, replace=False)

            for method, func in [('linear', partial_corr_linear),
                                  ('quadratic', partial_corr_quadratic),
                                  ('rank', partial_corr_rank)]:
                perm_rs = []
                for si in sample_idx:
                    pd = problem_data[li][si]
                    for kv_h in range(n_kv_heads):
                        r, _ = func(pd['ans_profiles'][kv_h], perm_vnums, pos_array)
                        perm_rs.append(r)
                perm_means[method].append(float(np.mean(perm_rs)))

        for method in ['linear', 'quadratic', 'rank']:
            perm_arr = np.array(perm_means[method])
            obs_mean = layer_results['aggregate'][method]['mean_ans_r']
            perm_p = float(np.mean(perm_arr >= obs_mean))

            layer_results['aggregate'][method]['perm_null_mean'] = float(np.mean(perm_arr))
            layer_results['aggregate'][method]['perm_null_std'] = float(np.std(perm_arr))
            layer_results['aggregate'][method]['perm_null_95'] = float(np.percentile(perm_arr, 95))
            layer_results['aggregate'][method]['perm_p'] = perm_p
            layer_results['aggregate'][method]['perm_z'] = (
                float((obs_mean - np.mean(perm_arr)) / np.std(perm_arr))
                if np.std(perm_arr) > 1e-10 else 0.0
            )

            print(f"  {method:10s} perm: obs={obs_mean:.4f}, null_mean={np.mean(perm_arr):.4f}, "
                  f"null_95th={np.percentile(perm_arr, 95):.4f}, p={perm_p:.4f}, "
                  f"z={layer_results['aggregate'][method]['perm_z']:.2f}")

        # ── Ecological correlation (binned, no position control — for comparison) ──
        # Mean attention profile across all problems and heads
        mean_delta = np.zeros(N_BINS)
        for pd in problem_data[li]:
            for kv_h in range(n_kv_heads):
                mean_delta += pd['ans_profiles'][kv_h] - pd['ctrl_profiles'][kv_h]
        mean_delta /= (len(problem_data[li]) * n_kv_heads)

        eco_vnums_r, eco_vnums_p = stats.pearsonr(mean_delta, vnums)
        eco_vr_r, eco_vr_p = stats.pearsonr(mean_delta, v_r)
        eco_nums_r, eco_nums_p = stats.pearsonr(mean_delta, nums_r)

        layer_results['ecological'] = {
            'V_nums_R': {'r': float(eco_vnums_r), 'p': float(eco_vnums_p)},
            'V_R': {'r': float(eco_vr_r), 'p': float(eco_vr_p)},
            'nums_R': {'r': float(eco_nums_r), 'p': float(eco_nums_p)},
            'mean_delta': mean_delta.tolist(),
        }

        print(f"  Ecological: r(Δattn, V|nums)={eco_vnums_r:.3f} (p={eco_vnums_p:.4f}), "
              f"r(Δattn, nums_R)={eco_nums_r:.3f}")

        # ── Retention percentages (linear → quadratic → rank) ──
        lin_r = layer_results['aggregate']['linear']['mean_ans_r']
        quad_r = layer_results['aggregate']['quadratic']['mean_ans_r']
        rank_r = layer_results['aggregate']['rank']['mean_ans_r']

        if abs(lin_r) > 1e-6:
            quad_retention = 100 * quad_r / lin_r
            rank_retention = 100 * rank_r / lin_r
        else:
            quad_retention = 0
            rank_retention = 0

        layer_results['retention'] = {
            'linear_to_quadratic_pct': float(quad_retention),
            'linear_to_rank_pct': float(rank_retention),
        }
        print(f"  Retention: linear→quadratic {quad_retention:.0f}%, "
              f"linear→rank {rank_retention:.0f}%")

        all_results[f'L{li}'] = layer_results

    # ═══════════════════════════════════════════════════
    # PHASE 4: Cross-method comparison table & Phi comparison
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 4: Cross-method comparison summary")
    print(f"{'='*70}")

    print(f"\n{'Layer':<8} {'Method':<12} {'mean_r':>8} {'pos/N':>8} {'sig':>6} "
          f"{'perm_p':>8} {'perm_z':>8} {'retain%':>8}")
    print("-" * 78)

    for li in layers_probe:
        key = f'L{li}'
        lin_r = all_results[key]['aggregate']['linear']['mean_ans_r']
        for method in ['linear', 'quadratic', 'rank']:
            agg = all_results[key]['aggregate'][method]
            r = agg['mean_ans_r']
            pos = agg['n_positive']
            sig = agg['n_sig_001']
            perm_p = agg.get('perm_p', -1)
            perm_z = agg.get('perm_z', -1)
            retain = 100 * r / lin_r if abs(lin_r) > 1e-6 else 0

            print(f"L{li:<6} {method:<12} {r:>8.4f} {pos}/{agg['n_heads']:>5} "
                  f"{sig:>5} {perm_p:>8.4f} {perm_z:>8.2f} {retain:>7.0f}%")
        print()

    # Compare with Phi (exp_097) for matching layers
    print("\n--- Cross-model comparison (Qwen vs Phi at matched layers) ---")
    print(f"{'Layers':<12} {'Method':<12} {'Qwen r':>8} {'Phi r':>8} {'Ratio':>8}")
    print("-" * 55)

    # Load Phi results from exp_097
    phi_results_path = RESULTS_DIR.parent / "exp_097" / "results.json"
    if phi_results_path.exists():
        with open(phi_results_path) as f:
            phi_data = json.load(f)

        phi_pairs = {(9, 8): 'L8', (18, 16): 'L16', (27, 24): 'L24', (35, 31): 'L31'}
        for (qwen_l, phi_l), phi_key in phi_pairs.items():
            qwen_key = f'L{qwen_l}'
            if qwen_key not in all_results or phi_key not in phi_data:
                continue
            phi_layer = phi_data[phi_key]

            for method in ['linear', 'quadratic']:
                qwen_r = all_results[qwen_key]['aggregate'][method]['mean_ans_r']

                if method == 'linear':
                    phi_r = phi_layer.get('aggregate_linear', {}).get('mean_ans_r', None)
                else:
                    phi_r = phi_layer.get('aggregate_quadratic', {}).get('mean_ans_r', None)

                if phi_r is not None:
                    ratio = qwen_r / phi_r if abs(phi_r) > 1e-6 else float('inf')
                    print(f"L{qwen_l}/L{phi_l}  {method:<12} {qwen_r:>8.4f} {phi_r:>8.4f} {ratio:>8.2f}x")

    # ═══════════════════════════════════════════════════
    # PHASE 5: Figures
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 5: Generating figures")
    print(f"{'='*70}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: Per-head comparison across methods (4 panels for 4 layers)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, li in enumerate(layers_probe):
        ax = axes[idx // 2][idx % 2]
        key = f'L{li}'
        heads = list(range(n_kv_heads))

        for method, color, marker in [('linear', 'steelblue', 'o'),
                                       ('quadratic', 'darkorange', 's'),
                                       ('rank', 'seagreen', '^')]:
            vals = [all_results[key]['per_head'][f'H{h}'][method]['mean_ans_r']
                    for h in heads]
            ax.plot(heads, vals, marker=marker, markersize=5, color=color,
                    label=method, alpha=0.8, linewidth=1.5)

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_title(f'Layer {li}', fontsize=12, fontweight='bold')
        ax.set_xlabel('KV Head')
        ax.set_ylabel('Mean partial r (answer step)')
        ax.legend(fontsize=9)
        ax.set_xticks(heads)

    fig.suptitle('Exp 098: Per-Head Partial Correlation — 3 Position Control Methods\nQwen3-4B-Base',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'per_head_3methods.png', dpi=150)
    plt.close()
    print("  Saved per_head_3methods.png")

    # Figure 2: Method comparison bar chart (aggregate across heads)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(len(layers_probe))
    width = 0.25

    for i, (method, color) in enumerate([('linear', 'steelblue'),
                                          ('quadratic', 'darkorange'),
                                          ('rank', 'seagreen')]):
        vals = [all_results[f'L{li}']['aggregate'][method]['mean_ans_r']
                for li in layers_probe]
        bars = ax.bar(x + i * width, vals, width, label=method, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean partial r (answer step)', fontsize=12)
    ax.set_title('Exp 098: Position Control Method Comparison\n'
                 'Qwen3-4B-Base — Aggregate across all KV heads', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'L{li}' for li in layers_probe])
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'method_comparison_bars.png', dpi=150)
    plt.close()
    print("  Saved method_comparison_bars.png")

    # Figure 3: Retention heatmap (% of linear signal retained)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    retention_data = np.zeros((2, len(layers_probe)))
    for j, li in enumerate(layers_probe):
        key = f'L{li}'
        lin_r = all_results[key]['aggregate']['linear']['mean_ans_r']
        if abs(lin_r) > 1e-6:
            retention_data[0, j] = 100 * all_results[key]['aggregate']['quadratic']['mean_ans_r'] / lin_r
            retention_data[1, j] = 100 * all_results[key]['aggregate']['rank']['mean_ans_r'] / lin_r

    im = ax.imshow(retention_data, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=100)
    ax.set_xticks(range(len(layers_probe)))
    ax.set_xticklabels([f'L{li}' for li in layers_probe])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Quadratic', 'Rank-based'])
    for i in range(2):
        for j in range(len(layers_probe)):
            ax.text(j, i, f'{retention_data[i,j]:.0f}%', ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if retention_data[i,j] < 20 else 'black')
    plt.colorbar(im, label='% of linear signal retained')
    ax.set_title('Exp 098: Signal Retention Under Position Control\nQwen3-4B-Base',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'retention_heatmap.png', dpi=150)
    plt.close()
    print("  Saved retention_heatmap.png")

    # Figure 4: Permutation null distribution with observed values
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, li in enumerate(layers_probe):
        ax = axes[idx // 2][idx % 2]
        key = f'L{li}'

        for method, color, offset in [('linear', 'steelblue', -0.3),
                                       ('quadratic', 'darkorange', 0),
                                       ('rank', 'seagreen', 0.3)]:
            agg = all_results[key]['aggregate'][method]
            obs = agg['mean_ans_r']
            null_mean = agg.get('perm_null_mean', 0)
            null_std = agg.get('perm_null_std', 0.01)
            null_95 = agg.get('perm_null_95', 0)
            perm_z = agg.get('perm_z', 0)

            ax.errorbar(offset, null_mean, yerr=2*null_std, fmt='o', color=color,
                        markersize=6, capsize=5, label=f'{method} null')
            ax.plot(offset, obs, '*', color=color, markersize=15,
                    label=f'{method} obs (z={perm_z:.1f})')
            ax.axhline(null_95, color=color, linewidth=0.5, linestyle=':', alpha=0.5)

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_title(f'Layer {li}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean partial r')
        ax.set_xlim(-0.6, 0.6)
        ax.legend(fontsize=7, loc='best')

    fig.suptitle('Exp 098: Observed vs Permutation Null — Qwen3-4B-Base',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'permutation_null.png', dpi=150)
    plt.close()
    print("  Saved permutation_null.png")

    # Figure 5: Ecological + delta attention profiles
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, li in enumerate(layers_probe):
        ax = axes[idx // 2][idx % 2]
        key = f'L{li}'
        info_key = layer_to_info[li]

        delta = np.array(all_results[key]['ecological']['mean_delta'])
        vnums = info_landscapes[info_key]['V_nums_R']

        ax2 = ax.twinx()
        ax.bar(range(N_BINS), delta, color='steelblue', alpha=0.5, label='Δ attention')
        ax2.plot(range(N_BINS), vnums, 'r-o', markersize=4, linewidth=2, label='V|nums')

        eco_r = all_results[key]['ecological']['V_nums_R']['r']
        eco_p = all_results[key]['ecological']['V_nums_R']['p']
        ax.set_title(f'Layer {li}: r(Δattn, V|nums) = {eco_r:.3f} (p={eco_p:.4f})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Chain position bin')
        ax.set_ylabel('Δ attention (answer - control)', color='steelblue')
        ax2.set_ylabel('V|nums', color='red')
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)

    fig.suptitle('Exp 098: Delta Attention vs V|nums Landscape — Qwen3-4B-Base',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'delta_attention_vs_vnums.png', dpi=150)
    plt.close()
    print("  Saved delta_attention_vs_vnums.png")

    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    output = {
        'config': {
            'model': MODEL_NAME,
            'n_problems_generated': len(generations),
            'n_correct': n_correct,
            'n_valid': n_valid,
            'accuracy_pct': round(100 * n_correct / len(generations), 1),
            'layers': layers_probe,
            'n_bins': N_BINS,
            'n_kv_heads': n_kv_heads,
            'group_size': group_size,
            'n_permutations': N_PERMUTATIONS,
            'position_control_methods': ['linear', 'quadratic', 'rank'],
            'vnums_source': 'exp_084',
        },
    }
    for li in layers_probe:
        output[f'L{li}'] = all_results[f'L{li}']

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"DONE — Total runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
