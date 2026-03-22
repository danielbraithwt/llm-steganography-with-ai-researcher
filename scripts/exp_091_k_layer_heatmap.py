#!/usr/bin/env python3
"""
Experiment 091: K Layer × Position Heatmap — K vs V Depth Profile Comparison

Probes K|nums at ALL 36 layers × 20 position bins on Qwen3-4B-Base.
Then compares against V|nums from exp_089 to test whether Phase 1's K>V
causal hierarchy extends to probing decodability.

This is a DISCONFIRMATORY experiment: if K|nums < V|nums across layers,
it challenges the unified "K=routing=hidden channel" narrative from Phase 1.

Methodology identical to exp_089, but extracting K instead of V.
"""

import os
import json
import time
import gc
import re
import sys
import warnings

import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

T0 = time.time()
TIME_BUDGET = 6600  # 110 min
MAX_GEN = 512
MAX_SEQ_LEN = 2048
MODEL_NAME = 'Qwen/Qwen3-4B-Base'
N_PROBLEMS = 250
N_BINS = 20
N_FOLDS = 5
MAX_NUMS_DIM = 30
SHUFFLE_LAYERS = [0, 4, 9, 13, 18, 22, 27, 31, 35]

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_091"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Path to exp_089 V results for comparison
EXP089_RESULTS = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_089" / "results.json"

# ── Plain-text 8-shot exemplars (same as exp_084/089) ──
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


def answer_in_text(text, answer_str):
    clean_text = text.replace(',', '')
    clean_ans = answer_str.replace(',', '')
    if not clean_ans:
        return False
    pattern = r'(?<!\d)' + re.escape(clean_ans) + r'(?!\d)'
    return bool(re.search(pattern, clean_text))


def log_transform(x):
    return float(np.sign(x) * np.log1p(np.abs(x)))


def get_kv(cache, layer):
    if hasattr(cache, 'layers'):
        l = cache.layers[layer]
        return l.keys, l.values
    if hasattr(cache, 'key_cache'):
        return cache.key_cache[layer], cache.value_cache[layer]
    return cache[layer][0], cache[layer][1]


def find_hash_pos_in_gen(gen_ids, tokenizer):
    cum_text = ""
    for i, tid in enumerate(gen_ids):
        cum_text += tokenizer.decode([tid], skip_special_tokens=False)
        if "####" in cum_text:
            prefix = cum_text[:cum_text.index("####")]
            char_count = 0
            for j, t in enumerate(gen_ids):
                char_count += len(tokenizer.decode([t], skip_special_tokens=False))
                if char_count > len(prefix):
                    return j
            return i
    return None


def extract_numbers_from_text(text):
    nums = []
    for m in re.finditer(r'(?<![.\w])(\d+(?:\.\d+)?)(?![.\w])', text):
        try:
            val = float(m.group(1))
            nums.append(val)
        except:
            pass
    return nums


def numbers_to_features(nums, max_dim=MAX_NUMS_DIM):
    if not nums:
        return np.zeros(max_dim + 6, dtype=np.float32)
    log_nums = [np.log1p(abs(n)) * (1 if n >= 0 else -1) for n in nums]
    if len(log_nums) > max_dim:
        truncated = log_nums[-max_dim:]
    else:
        truncated = log_nums
    features = np.zeros(max_dim, dtype=np.float32)
    features[:len(truncated)] = truncated
    arr = np.array(log_nums)
    stats = np.array([
        len(log_nums), np.mean(arr), np.std(arr),
        np.max(arr), np.min(arr), np.sum(arr),
    ], dtype=np.float32)
    return np.concatenate([features, stats])


NUMS_FEAT_DIM = MAX_NUMS_DIM + 6  # 36


def main():
    t0 = time.time()

    print("=" * 70)
    print("Experiment 091: K Layer × Position Heatmap")
    print("K vs V Depth Profile Comparison (DISCONFIRMATORY)")
    print("Full K sweep: ALL 36 layers × 20 position bins")
    print("=" * 70)

    # ── Load exp_089 V results for comparison ──
    v_results = None
    if EXP089_RESULTS.exists():
        with open(EXP089_RESULTS) as f:
            v_results = json.load(f)
        print(f"Loaded exp_089 V results for comparison ({v_results['n_layers']} layers)")
    else:
        print("WARNING: exp_089 results not found — will skip K vs V comparison")

    # ── Load model ──
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // model.config.num_attention_heads)
    kv_dim = kv_heads * head_dim
    print(f"Model: {MODEL_NAME}, {n_layers} layers")
    print(f"KV heads: {kv_heads}, head_dim: {head_dim}, KV dim: {kv_dim}")

    ds = load_gsm8k()
    print(f"GSM8K test set: {len(ds)} problems")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Generate CoT traces (identical to exp_089: same seed, same model, same data)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: Generating CoT traces")
    print("=" * 70)

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
            'idx': i, 'question': question, 'gold': gold,
            'gen_text': gen_text, 'gen_ids': gen_ids,
            'pred': pred, 'correct': correct,
        })

        if (i + 1) % 50 == 0:
            n_corr = sum(g['correct'] for g in generations)
            elapsed = time.time() - t0
            print(f"  Generated {i+1} problems, {n_corr}/{len(generations)} correct "
                  f"({100*n_corr/len(generations):.1f}%) [{elapsed:.0f}s]")

    n_total = len(generations)
    n_correct = sum(g['correct'] for g in generations)
    print(f"\nPhase 1 complete: {n_total} generated, {n_correct} correct "
          f"({100*n_correct/n_total:.1f}%)")

    correct_gens = []
    for g in generations:
        if not g['correct']:
            continue
        hash_pos = find_hash_pos_in_gen(g['gen_ids'], tokenizer)
        if hash_pos is None or hash_pos < 10:
            continue
        g['hash_pos_gen'] = hash_pos
        correct_gens.append(g)

    print(f"Usable correct problems: {len(correct_gens)}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Forward pass + K extraction at ALL layers
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 2: Extracting K cache at ALL {n_layers} layers")
    print("=" * 70)

    # Shared across all layers (position, nums, answers, groups)
    shared = {
        'cumNums': [], 'rel_pos': [], 'text_reveals': [],
        'final_answer': [], 'problem_idx': [],
    }
    # Per-layer K vectors
    K_by_layer = {layer: [] for layer in range(n_layers)}

    first_reveal_positions = []
    n_extracted = 0

    for pi, gen in enumerate(correct_gens):
        if time.time() - t0 > TIME_BUDGET * 0.55:
            print(f"\n  Time budget (55%) reached at problem {pi}")
            break

        prompt_text = build_prompt(gen['question'])
        prompt_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids']
        prompt_len = prompt_ids.shape[1]

        gen_token_ids = gen['gen_ids']
        full_ids = torch.cat([
            prompt_ids[0],
            torch.tensor(gen_token_ids, dtype=torch.long)
        ]).unsqueeze(0).to(model.device)

        if full_ids.shape[1] > MAX_SEQ_LEN:
            continue

        hash_pos_gen = gen['hash_pos_gen']
        cot_length = hash_pos_gen
        if cot_length < 10:
            continue

        # Forward pass — gets KV cache at ALL layers
        with torch.no_grad():
            outputs = model(full_ids, use_cache=True)
        kv_cache = outputs.past_key_values

        cot_ids = gen_token_ids[:hash_pos_gen]
        final_answer_str = str(gen['gold']).strip()
        final_answer_log = log_transform(float(gen['gold']))
        question_text = gen['question']

        # Build cumulative text and numbers at each position
        cot_token_texts = [tokenizer.decode([tid], skip_special_tokens=False) for tid in cot_ids]
        cum_cot_text = ""
        first_reveal_rel = 1.0
        token_cumNums = []
        token_reveals = []
        question_nums = extract_numbers_from_text(question_text)

        for j, tok_text in enumerate(cot_token_texts):
            cum_cot_text += tok_text
            cot_nums = extract_numbers_from_text(cum_cot_text)
            all_nums = question_nums + cot_nums
            token_cumNums.append(numbers_to_features(all_nums))
            revealed = answer_in_text(cum_cot_text, final_answer_str)
            token_reveals.append(revealed)
            if revealed and first_reveal_rel == 1.0:
                first_reveal_rel = j / cot_length

        first_reveal_positions.append(first_reveal_rel)

        # Extract K at ALL layers for each CoT position
        for j in range(cot_length):
            abs_pos = prompt_len + j
            rel_pos = j / cot_length

            shared['cumNums'].append(token_cumNums[j])
            shared['rel_pos'].append(rel_pos)
            shared['text_reveals'].append(token_reveals[j])
            shared['final_answer'].append(final_answer_log)
            shared['problem_idx'].append(pi)

        # Extract K vectors per layer
        for layer in range(n_layers):
            K_layer, _ = get_kv(kv_cache, layer)
            for j in range(cot_length):
                abs_pos = prompt_len + j
                k_vec = K_layer[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                K_by_layer[layer].append(k_vec)

        n_extracted += 1

        del outputs, kv_cache
        torch.cuda.empty_cache()

        if (pi + 1) % 25 == 0:
            elapsed = time.time() - t0
            n_vecs = len(shared['rel_pos'])
            print(f"  Extracted {pi+1}/{len(correct_gens)} problems, "
                  f"{n_vecs} total vectors [{elapsed:.0f}s]")

    print(f"\nPhase 2 complete: {n_extracted} problems extracted")
    print(f"Total vectors per layer: {len(shared['rel_pos'])}")

    # Convert shared data to numpy
    for key in shared:
        shared[key] = np.array(shared[key])

    # Convert K arrays
    print("Converting K arrays to numpy...")
    for layer in range(n_layers):
        K_by_layer[layer] = np.array(K_by_layer[layer], dtype=np.float32)
    print(f"  K shape per layer: {K_by_layer[0].shape}")
    k_mem_gb = K_by_layer[0].nbytes * n_layers / 1e9
    print(f"  Total K memory: {k_mem_gb:.1f} GB")

    # Free model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded.")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Position-sweep probing K at ALL layers
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 3: Probing K|nums at {n_layers} layers × {N_BINS} bins")
    print("=" * 70)

    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import RidgeCV
    from scipy.stats import pearsonr

    RIDGE_ALPHAS = np.logspace(-2, 6, 50)
    bin_edges = np.linspace(0, 1, N_BINS + 1)

    rel_pos = shared['rel_pos']
    y_all = shared['final_answer']
    groups_all = shared['problem_idx']
    cumNums_all = shared['cumNums']

    # Text-reveals curve
    first_reveal_arr = np.array(first_reveal_positions[:n_extracted])
    text_reveals_curve = []
    for b in range(N_BINS):
        bin_upper = bin_edges[b + 1]
        frac = float(np.mean(first_reveal_arr <= bin_upper))
        text_reveals_curve.append(frac)

    # Precompute bin masks, nums_R, and y_resid per bin (identical to exp_089)
    print("\nPrecomputing nums_R and y_resid per bin...")
    bin_data = {}
    for b in range(N_BINS):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b < N_BINS - 1:
            mask = (rel_pos >= lo) & (rel_pos < hi)
        else:
            mask = (rel_pos >= lo) & (rel_pos <= hi)

        y_bin = y_all[mask]
        g_bin = groups_all[mask]
        nums_bin = cumNums_all[mask]
        n_samples = len(y_bin)
        n_groups = len(np.unique(g_bin))

        if n_groups < N_FOLDS + 1 or n_samples < 30:
            bin_data[b] = {'skip': True, 'n_samples': n_samples, 'n_groups': n_groups}
            print(f"  Bin {b:2d} [{lo:.2f}-{hi:.2f}]: SKIPPED (ng={n_groups}, n={n_samples})")
            continue

        gkf = GroupKFold(n_splits=min(N_FOLDS, n_groups))

        # Compute nums_R
        nums_preds = np.zeros_like(y_bin)
        for tr, te in gkf.split(nums_bin, y_bin, g_bin):
            ridge = RidgeCV(alphas=RIDGE_ALPHAS)
            ridge.fit(nums_bin[tr], y_bin[tr])
            nums_preds[te] = ridge.predict(nums_bin[te])
        nums_r, _ = pearsonr(nums_preds, y_bin)

        # Compute y_resid (residuals after nums prediction)
        y_resid = np.zeros_like(y_bin)
        for tr, te in gkf.split(nums_bin, y_bin, g_bin):
            ridge = RidgeCV(alphas=RIDGE_ALPHAS)
            ridge.fit(nums_bin[tr], y_bin[tr])
            y_resid[te] = y_bin[te] - ridge.predict(nums_bin[te])

        resid_std = np.std(y_resid)

        bin_data[b] = {
            'skip': False,
            'mask': mask,
            'y_bin': y_bin,
            'y_resid': y_resid,
            'g_bin': g_bin,
            'nums_r': float(nums_r),
            'resid_std': float(resid_std),
            'n_samples': n_samples,
            'n_groups': n_groups,
        }
        print(f"  Bin {b:2d} [{lo:.2f}-{hi:.2f}]: n={n_samples}, ng={n_groups}, "
              f"nums_R={nums_r:.3f}, resid_std={resid_std:.3f}")

    # Now probe K → answer and K → y_resid at every (layer, bin)
    print(f"\nProbing K at {n_layers} layers × {N_BINS} bins...")

    # Results matrices
    K_R_matrix = np.full((n_layers, N_BINS), np.nan)
    K_nums_R_matrix = np.full((n_layers, N_BINS), np.nan)
    nums_R_vector = np.full(N_BINS, np.nan)
    shuffle_R_matrix = np.full((n_layers, N_BINS), np.nan)

    for b in range(N_BINS):
        if bin_data[b].get('skip', True):
            continue
        nums_R_vector[b] = bin_data[b]['nums_r']

    for layer in range(n_layers):
        layer_t0 = time.time()
        K_layer = K_by_layer[layer]

        for b in range(N_BINS):
            if bin_data[b].get('skip', True):
                continue

            bd = bin_data[b]
            mask = bd['mask']
            K_bin = K_layer[mask]
            y_bin = bd['y_bin']
            y_resid = bd['y_resid']
            g_bin = bd['g_bin']

            gkf = GroupKFold(n_splits=min(N_FOLDS, bd['n_groups']))

            # K → answer (raw decodability)
            k_preds = np.zeros_like(y_bin)
            for tr, te in gkf.split(K_bin, y_bin, g_bin):
                ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                ridge.fit(K_bin[tr], y_bin[tr])
                k_preds[te] = ridge.predict(K_bin[te])
            k_r, _ = pearsonr(k_preds, y_bin)
            K_R_matrix[layer, b] = float(k_r)

            # K → y_resid (forward-looking beyond numbers)
            if bd['resid_std'] > 1e-10:
                kn_preds = np.zeros_like(y_resid)
                for tr, te in gkf.split(K_bin, y_resid, g_bin):
                    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                    ridge.fit(K_bin[tr], y_resid[tr])
                    kn_preds[te] = ridge.predict(K_bin[te])
                kn_r, _ = pearsonr(kn_preds, y_resid)
                K_nums_R_matrix[layer, b] = float(kn_r)
            else:
                K_nums_R_matrix[layer, b] = 0.0

            # Shuffle control at selected layers
            if layer in SHUFFLE_LAYERS:
                rng = np.random.RandomState(SEED + layer * 1000 + b)
                shuf_idx = rng.permutation(len(y_bin))
                y_shuf = y_bin[shuf_idx]
                g_shuf = g_bin[shuf_idx]
                s_preds = np.zeros_like(y_shuf)
                gkf_s = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(g_shuf))))
                for tr, te in gkf_s.split(K_bin, y_shuf, g_shuf):
                    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                    ridge.fit(K_bin[tr], y_shuf[tr])
                    s_preds[te] = ridge.predict(K_bin[te])
                s_r, _ = pearsonr(s_preds, y_shuf)
                shuffle_R_matrix[layer, b] = float(s_r)

        elapsed_layer = time.time() - layer_t0
        elapsed_total = time.time() - t0

        # Print summary for every 4th layer
        if layer % 4 == 0 or layer == n_layers - 1:
            valid = ~np.isnan(K_nums_R_matrix[layer])
            mean_kn = np.nanmean(K_nums_R_matrix[layer]) if valid.any() else 0
            mean_kr = np.nanmean(K_R_matrix[layer]) if valid.any() else 0
            n_pos = int(np.sum(K_nums_R_matrix[layer][valid] > 0)) if valid.any() else 0
            n_valid = int(valid.sum())
            print(f"  L{layer:2d}: K_R={mean_kr:.3f}, K|nums={mean_kn:.3f}, "
                  f"pos={n_pos}/{n_valid} [{elapsed_layer:.1f}s, total {elapsed_total:.0f}s]")

    print(f"\nPhase 3 complete. Total elapsed: {time.time()-t0:.0f}s")

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Analysis and Visualization
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: Analysis and Visualization")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 * 100  # percentage

    # ── Load V results from exp_089 ──
    V_nums_R_matrix_089 = None
    V_R_matrix_089 = None
    v_layer_means = None
    if v_results is not None:
        V_nums_R_matrix_089 = np.array(v_results['V_nums_R_matrix'])
        V_R_matrix_089 = np.array(v_results['V_R_matrix'])
        v_layer_means = v_results['layer_means']
        print(f"Loaded exp_089 V heatmaps: {V_nums_R_matrix_089.shape}")

    # ── Summary statistics ──
    print("\n--- Layer-wise summary (mean K|nums_R across bins) ---")
    k_layer_means = []
    for layer in range(n_layers):
        valid = ~np.isnan(K_nums_R_matrix[layer])
        if valid.any():
            mean_kn = float(np.nanmean(K_nums_R_matrix[layer]))
            n_pos = int(np.sum(K_nums_R_matrix[layer][valid] > 0))
            n_valid = int(valid.sum())
        else:
            mean_kn = 0.0
            n_pos = 0
            n_valid = 0
        k_layer_means.append(mean_kn)
        depth_pct = 100 * layer / (n_layers - 1)

        # Compare with V if available
        v_comp = ""
        if v_layer_means is not None and layer < len(v_layer_means):
            v_mean = v_layer_means[layer]
            diff = mean_kn - v_mean
            v_comp = f"  | V|nums={v_mean:+.3f}, K-V={diff:+.3f}"

        print(f"  L{layer:2d} ({depth_pct:5.1f}%): mean K|nums={mean_kn:+.3f}, "
              f"positive={n_pos}/{n_valid}{v_comp}")

    # Peak and emergence
    peak_layer = int(np.argmax(k_layer_means))
    peak_mean = k_layer_means[peak_layer]
    print(f"\nK peak layer: L{peak_layer} (mean K|nums = {peak_mean:.3f})")

    emergence_layer = None
    for layer in range(n_layers):
        if k_layer_means[layer] > 0.05:
            emergence_layer = layer
            break
    if emergence_layer is not None:
        print(f"K emergence layer (mean K|nums > 0.05): L{emergence_layer} "
              f"({100*emergence_layer/(n_layers-1):.0f}% depth)")

    # ── K vs V comparison statistics ──
    if v_layer_means is not None:
        print("\n--- K vs V Comparison ---")
        k_wins = 0
        v_wins = 0
        ties = 0
        k_minus_v = []
        for layer in range(n_layers):
            k_m = k_layer_means[layer]
            v_m = v_layer_means[layer]
            diff = k_m - v_m
            k_minus_v.append(diff)
            if diff > 0.01:
                k_wins += 1
            elif diff < -0.01:
                v_wins += 1
            else:
                ties += 1

        print(f"  K > V at {k_wins}/{n_layers} layers (>0.01 threshold)")
        print(f"  V > K at {v_wins}/{n_layers} layers")
        print(f"  Tied at {ties}/{n_layers} layers")
        print(f"  Mean K-V difference: {np.mean(k_minus_v):.4f}")
        print(f"  Max K advantage: {np.max(k_minus_v):.3f} at L{np.argmax(k_minus_v)}")
        print(f"  Max V advantage: {np.min(k_minus_v):.3f} at L{np.argmin(k_minus_v)}")

        # Phase-wise comparison
        ramp_layers = list(range(0, 10))
        plateau_layers = list(range(10, n_layers))
        ramp_k = np.mean([k_layer_means[l] for l in ramp_layers])
        ramp_v = np.mean([v_layer_means[l] for l in ramp_layers])
        plat_k = np.mean([k_layer_means[l] for l in plateau_layers])
        plat_v = np.mean([v_layer_means[l] for l in plateau_layers])
        print(f"\n  Ramp (L0-L9):    K|nums={ramp_k:+.3f}, V|nums={ramp_v:+.3f}, diff={ramp_k-ramp_v:+.3f}")
        print(f"  Plateau (L10-35): K|nums={plat_k:+.3f}, V|nums={plat_v:+.3f}, diff={plat_k-plat_v:+.3f}")

        # Bin-level comparison at key positions
        print("\n--- K|nums vs V|nums at key positions (per-bin) ---")
        for b_idx, b_name in [(0, "0-5% (chain start)"), (9, "45-50% (mid-chain)"),
                               (18, "90-95% (chain end)")]:
            k_col = K_nums_R_matrix[:, b_idx]
            v_col = V_nums_R_matrix_089[:, b_idx] if V_nums_R_matrix_089 is not None else None
            valid_k = ~np.isnan(k_col)
            k_mean = float(np.nanmean(k_col[valid_k])) if valid_k.any() else 0
            if v_col is not None:
                valid_v = ~np.isnan(v_col)
                v_mean = float(np.nanmean(v_col[valid_v])) if valid_v.any() else 0
                print(f"  Bin {b_idx} ({b_name}): K|nums={k_mean:.3f}, V|nums={v_mean:.3f}, "
                      f"diff={k_mean-v_mean:+.3f}")
            else:
                print(f"  Bin {b_idx} ({b_name}): K|nums={k_mean:.3f}")

    # ── Figure 1: K heatmaps (mirror of exp_089 Fig 1) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})

    ax = axes[0]
    vmin_kr = max(0, np.nanmin(K_R_matrix))
    vmax_kr = np.nanmax(K_R_matrix)
    im1 = ax.imshow(K_R_matrix, aspect='auto', origin='lower',
                     extent=[0, 100, 0, n_layers-1],
                     cmap='YlOrRd', vmin=vmin_kr, vmax=vmax_kr)
    ax.set_xlabel('Chain Position (%)', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('K → answer (raw decodability)', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax, label='Pearson R')
    for pl in [18, 27]:
        ax.axhline(y=pl, color='white', linestyle='--', alpha=0.5, linewidth=0.8)

    ax = axes[1]
    vmax_kn = max(abs(np.nanmin(K_nums_R_matrix)), np.nanmax(K_nums_R_matrix))
    vmax_kn = min(vmax_kn, 0.6)
    norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=vmax_kn)
    im2 = ax.imshow(K_nums_R_matrix, aspect='auto', origin='lower',
                     extent=[0, 100, 0, n_layers-1],
                     cmap='RdBu_r', norm=norm)
    ax.set_xlabel('Chain Position (%)', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('K|nums → answer (forward-looking)', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax, label='Pearson R (residualized)')
    for pl in [18, 27]:
        ax.axhline(y=pl, color='black', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.suptitle('Experiment 091: K Layer × Position Heatmap — Qwen3-4B-Base',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'k_layer_position_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: k_layer_position_heatmap.png")

    # ── Figure 2: K vs V heatmap comparison (MAIN FIGURE) ──
    if V_nums_R_matrix_089 is not None:
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))

        # V|nums heatmap (from exp_089)
        ax = axes[0]
        vmax_comp = 0.5
        norm_comp = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=vmax_comp)
        im1 = ax.imshow(V_nums_R_matrix_089, aspect='auto', origin='lower',
                         extent=[0, 100, 0, n_layers-1],
                         cmap='RdBu_r', norm=norm_comp)
        ax.set_xlabel('Chain Position (%)', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title('V|nums (exp_089)', fontsize=13, fontweight='bold')
        plt.colorbar(im1, ax=ax, label='R')

        # K|nums heatmap (this experiment)
        ax = axes[1]
        im2 = ax.imshow(K_nums_R_matrix, aspect='auto', origin='lower',
                         extent=[0, 100, 0, n_layers-1],
                         cmap='RdBu_r', norm=norm_comp)
        ax.set_xlabel('Chain Position (%)', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title('K|nums (exp_091)', fontsize=13, fontweight='bold')
        plt.colorbar(im2, ax=ax, label='R')

        # K - V difference heatmap
        ax = axes[2]
        diff_matrix = K_nums_R_matrix - V_nums_R_matrix_089
        # Handle NaN
        diff_valid = ~(np.isnan(K_nums_R_matrix) | np.isnan(V_nums_R_matrix_089))
        diff_display = np.where(diff_valid, diff_matrix, np.nan)
        vmax_diff = 0.25
        norm_diff = TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)
        im3 = ax.imshow(diff_display, aspect='auto', origin='lower',
                         extent=[0, 100, 0, n_layers-1],
                         cmap='PiYG', norm=norm_diff)
        ax.set_xlabel('Chain Position (%)', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title('K|nums − V|nums (green=K wins)', fontsize=13, fontweight='bold')
        plt.colorbar(im3, ax=ax, label='K−V difference')

        plt.suptitle('Exp 091: K vs V Forward-Looking Signal — Qwen3-4B-Base',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'k_vs_v_heatmap_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: k_vs_v_heatmap_comparison.png")

    # ── Figure 3: K vs V layer profiles (bar chart) ──
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(n_layers)
    width = 0.35

    bars_k = ax.bar(x - width/2, k_layer_means, width, color='indianred',
                     alpha=0.7, edgecolor='darkred', label='K|nums (exp_091)')
    if v_layer_means is not None:
        bars_v = ax.bar(x + width/2, v_layer_means, width, color='steelblue',
                         alpha=0.7, edgecolor='navy', label='V|nums (exp_089)')

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean |nums (Pearson R)', fontsize=12)
    ax.set_title('K vs V Forward-Looking Signal by Layer', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'k_vs_v_layer_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: k_vs_v_layer_profile.png")

    # ── Figure 4: K|nums at chain start vs layer (with V comparison) ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bin0_k = K_nums_R_matrix[:, 0]
    valid_k0 = ~np.isnan(bin0_k)
    ax.plot(np.arange(n_layers)[valid_k0], bin0_k[valid_k0], 'r-s', markersize=4,
            label='K|nums at 0-5% (exp_091)', linewidth=1.5)
    if V_nums_R_matrix_089 is not None:
        bin0_v = V_nums_R_matrix_089[:, 0]
        valid_v0 = ~np.isnan(bin0_v)
        ax.plot(np.arange(n_layers)[valid_v0], bin0_v[valid_v0], 'b-o', markersize=4,
                label='V|nums at 0-5% (exp_089)', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Pearson R (residualized)', fontsize=12)
    ax.set_title('K vs V Answer Decodability at Chain Start (0-5%, text=0%)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'k_vs_v_chain_start.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: k_vs_v_chain_start.png")

    # ── Figure 5: K|nums layer profile at key positions ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    position_groups = {
        '0-5% (text=0%)': [0],
        '5-15%': [1, 2],
        '25-35%': [5, 6],
        '45-55%': [9, 10],
        '65-75%': [13, 14],
        '85-95%': [17, 18],
    }
    colors = plt.cm.magma(np.linspace(0.1, 0.9, len(position_groups)))
    for (label, bins), color in zip(position_groups.items(), colors):
        profile = []
        for layer in range(n_layers):
            vals = [K_nums_R_matrix[layer, b] for b in bins
                    if not np.isnan(K_nums_R_matrix[layer, b])]
            profile.append(np.mean(vals) if vals else np.nan)
        ax.plot(range(n_layers), profile, '-o', markersize=3, color=color,
                label=label, linewidth=1.5)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('K|nums (Pearson R)', fontsize=12)
    ax.set_title('K Forward-Looking Signal by Layer at Different Chain Positions',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'k_layer_profile_by_position.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: k_layer_profile_by_position.png")

    # ── Save results ──
    results = {
        'n_layers': n_layers,
        'n_bins': N_BINS,
        'n_extracted': n_extracted,
        'n_problems_generated': n_total,
        'n_correct': n_correct,
        'accuracy': round(100 * n_correct / n_total, 1),
        'kv_dim': kv_dim,
        'K_R_matrix': K_R_matrix.tolist(),
        'K_nums_R_matrix': K_nums_R_matrix.tolist(),
        'nums_R_vector': nums_R_vector.tolist(),
        'shuffle_R_matrix': shuffle_R_matrix.tolist(),
        'text_reveals_curve': text_reveals_curve,
        'k_layer_means': k_layer_means,
        'peak_layer': peak_layer,
        'peak_mean': peak_mean,
        'emergence_layer': emergence_layer,
        'bin_centers': bin_centers.tolist(),
        'bin_info': {str(b): {
            'n_samples': int(bin_data[b]['n_samples']),
            'n_groups': int(bin_data[b]['n_groups']),
            'nums_r': float(bin_data[b].get('nums_r', 0)),
        } for b in range(N_BINS) if not bin_data[b].get('skip', True)},
    }

    # Add K vs V comparison if available
    if v_layer_means is not None:
        k_minus_v_means = [k - v for k, v in zip(k_layer_means, v_layer_means)]
        results['v_layer_means_089'] = v_layer_means
        results['k_minus_v_means'] = k_minus_v_means
        results['k_wins_count'] = sum(1 for d in k_minus_v_means if d > 0.01)
        results['v_wins_count'] = sum(1 for d in k_minus_v_means if d < -0.01)
        results['mean_k_minus_v'] = float(np.mean(k_minus_v_means))

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results.json")

    # ── Print final summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}, {n_layers} layers, KV dim {kv_dim}")
    print(f"Problems: {n_total} generated, {n_correct} correct ({100*n_correct/n_total:.1f}%)")
    print(f"Extracted: {n_extracted} problems, {len(shared['rel_pos'])} vectors/layer")
    print(f"\nK peak layer: L{peak_layer} (mean K|nums = {peak_mean:.3f})")
    if emergence_layer is not None:
        print(f"K emergence layer: L{emergence_layer} ({100*emergence_layer/(n_layers-1):.0f}% depth)")

    print(f"\nK|nums at bin 0 (0-5%, text=0%):")
    for layer in range(0, n_layers, 4):
        val = K_nums_R_matrix[layer, 0]
        if not np.isnan(val):
            v_comp = ""
            if V_nums_R_matrix_089 is not None:
                v_val = V_nums_R_matrix_089[layer, 0]
                if not np.isnan(v_val):
                    v_comp = f"  (V={v_val:+.3f}, K-V={val-v_val:+.3f})"
            print(f"  L{layer:2d}: K|nums = {val:+.3f}{v_comp}")

    if v_layer_means is not None:
        print(f"\n--- K vs V VERDICT ---")
        k_mean_all = np.mean(k_layer_means)
        v_mean_all = np.mean(v_layer_means)
        print(f"  Mean K|nums across all layers: {k_mean_all:.3f}")
        print(f"  Mean V|nums across all layers: {v_mean_all:.3f}")
        print(f"  Overall difference (K-V): {k_mean_all - v_mean_all:+.3f}")
        if k_mean_all > v_mean_all + 0.02:
            print(f"  VERDICT: K > V for forward-looking probing — consistent with Phase 1")
        elif v_mean_all > k_mean_all + 0.02:
            print(f"  VERDICT: V > K for forward-looking probing — CHALLENGES Phase 1 K>V narrative")
        else:
            print(f"  VERDICT: K ≈ V for forward-looking probing — K>V is perturbation-specific")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
