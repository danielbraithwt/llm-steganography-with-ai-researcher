#!/usr/bin/env python3
"""
Experiment 089: Layer × Position Heatmap of Forward-Looking V Signal

Probes V|nums at ALL 36 layers × 20 position bins on Qwen3-4B-Base.
Creates a heatmap showing where forward-looking answer information
emerges across both model depth and chain position.

Key efficiency: nums_R and y_resid are precomputed once per bin
(they don't depend on layer). Only V → y_resid runs per layer.
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
SHUFFLE_LAYERS = [0, 4, 9, 13, 18, 22, 27, 31, 35]  # Select layers for shuffle control

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_089"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same as exp_084) ──
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
    print("Experiment 089: Layer × Position Heatmap of Forward-Looking V Signal")
    print("Full layer sweep: ALL 36 layers × 20 position bins")
    print("=" * 70)

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
    # PHASE 1: Generate CoT traces
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
    # PHASE 2: Forward pass + V extraction at ALL layers
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 2: Extracting V cache at ALL {n_layers} layers")
    print("=" * 70)

    # Shared across all layers (position, nums, answers, groups)
    shared = {
        'cumNums': [], 'rel_pos': [], 'text_reveals': [],
        'final_answer': [], 'problem_idx': [],
    }
    # Per-layer V vectors
    V_by_layer = {layer: [] for layer in range(n_layers)}

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

        # Extract V at ALL layers for each CoT position
        for j in range(cot_length):
            abs_pos = prompt_len + j
            rel_pos = j / cot_length

            shared['cumNums'].append(token_cumNums[j])
            shared['rel_pos'].append(rel_pos)
            shared['text_reveals'].append(token_reveals[j])
            shared['final_answer'].append(final_answer_log)
            shared['problem_idx'].append(pi)

        # Extract V vectors per layer (batched for efficiency)
        for layer in range(n_layers):
            _, V_layer = get_kv(kv_cache, layer)
            for j in range(cot_length):
                abs_pos = prompt_len + j
                v_vec = V_layer[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                V_by_layer[layer].append(v_vec)

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

    # Convert V arrays — this is the big one (~2.9GB for 36 layers)
    print("Converting V arrays to numpy...")
    for layer in range(n_layers):
        V_by_layer[layer] = np.array(V_by_layer[layer], dtype=np.float32)
    print(f"  V shape per layer: {V_by_layer[0].shape}")
    v_mem_gb = V_by_layer[0].nbytes * n_layers / 1e9
    print(f"  Total V memory: {v_mem_gb:.1f} GB")

    # Free model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded.")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Position-sweep probing at ALL layers
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 3: Probing V|nums at {n_layers} layers × {N_BINS} bins")
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

    # Precompute bin masks, nums_R, and y_resid per bin
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

    # Now probe V → answer and V → y_resid at every (layer, bin)
    print(f"\nProbing V at {n_layers} layers × {N_BINS} bins...")

    # Results matrices
    V_R_matrix = np.full((n_layers, N_BINS), np.nan)
    V_nums_R_matrix = np.full((n_layers, N_BINS), np.nan)
    nums_R_vector = np.full(N_BINS, np.nan)
    shuffle_R_matrix = np.full((n_layers, N_BINS), np.nan)

    for b in range(N_BINS):
        if bin_data[b].get('skip', True):
            continue
        nums_R_vector[b] = bin_data[b]['nums_r']

    for layer in range(n_layers):
        layer_t0 = time.time()
        V_layer = V_by_layer[layer]

        for b in range(N_BINS):
            if bin_data[b].get('skip', True):
                continue

            bd = bin_data[b]
            mask = bd['mask']
            V_bin = V_layer[mask]
            y_bin = bd['y_bin']
            y_resid = bd['y_resid']
            g_bin = bd['g_bin']

            gkf = GroupKFold(n_splits=min(N_FOLDS, bd['n_groups']))

            # V → answer (raw decodability)
            v_preds = np.zeros_like(y_bin)
            for tr, te in gkf.split(V_bin, y_bin, g_bin):
                ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                ridge.fit(V_bin[tr], y_bin[tr])
                v_preds[te] = ridge.predict(V_bin[te])
            v_r, _ = pearsonr(v_preds, y_bin)
            V_R_matrix[layer, b] = float(v_r)

            # V → y_resid (forward-looking beyond numbers)
            if bd['resid_std'] > 1e-10:
                vn_preds = np.zeros_like(y_resid)
                for tr, te in gkf.split(V_bin, y_resid, g_bin):
                    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                    ridge.fit(V_bin[tr], y_resid[tr])
                    vn_preds[te] = ridge.predict(V_bin[te])
                vn_r, _ = pearsonr(vn_preds, y_resid)
                V_nums_R_matrix[layer, b] = float(vn_r)
            else:
                V_nums_R_matrix[layer, b] = 0.0

            # Shuffle control at selected layers
            if layer in SHUFFLE_LAYERS:
                rng = np.random.RandomState(SEED + layer * 1000 + b)
                shuf_idx = rng.permutation(len(y_bin))
                y_shuf = y_bin[shuf_idx]
                g_shuf = g_bin[shuf_idx]
                s_preds = np.zeros_like(y_shuf)
                gkf_s = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(g_shuf))))
                for tr, te in gkf_s.split(V_bin, y_shuf, g_shuf):
                    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                    ridge.fit(V_bin[tr], y_shuf[tr])
                    s_preds[te] = ridge.predict(V_bin[te])
                s_r, _ = pearsonr(s_preds, y_shuf)
                shuffle_R_matrix[layer, b] = float(s_r)

        elapsed_layer = time.time() - layer_t0
        elapsed_total = time.time() - t0

        # Print summary for every 4th layer
        if layer % 4 == 0 or layer == n_layers - 1:
            valid = ~np.isnan(V_nums_R_matrix[layer])
            mean_vn = np.nanmean(V_nums_R_matrix[layer]) if valid.any() else 0
            mean_vr = np.nanmean(V_R_matrix[layer]) if valid.any() else 0
            n_pos = int(np.sum(V_nums_R_matrix[layer][valid] > 0)) if valid.any() else 0
            n_valid = int(valid.sum())
            print(f"  L{layer:2d}: V_R={mean_vr:.3f}, V|nums={mean_vn:.3f}, "
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

    # ── Summary statistics ──
    print("\n--- Layer-wise summary (mean V|nums_R across bins) ---")
    layer_means = []
    for layer in range(n_layers):
        valid = ~np.isnan(V_nums_R_matrix[layer])
        if valid.any():
            mean_vn = float(np.nanmean(V_nums_R_matrix[layer]))
            n_pos = int(np.sum(V_nums_R_matrix[layer][valid] > 0))
            n_valid = int(valid.sum())
        else:
            mean_vn = 0.0
            n_pos = 0
            n_valid = 0
        layer_means.append(mean_vn)
        depth_pct = 100 * layer / (n_layers - 1)
        print(f"  L{layer:2d} ({depth_pct:5.1f}%): mean V|nums={mean_vn:+.3f}, "
              f"positive={n_pos}/{n_valid}")

    # Find peak layer
    peak_layer = int(np.argmax(layer_means))
    peak_mean = layer_means[peak_layer]
    print(f"\nPeak layer: L{peak_layer} (mean V|nums = {peak_mean:.3f})")

    # Find emergence layer (first layer where mean V|nums > 0.05)
    emergence_layer = None
    for layer in range(n_layers):
        if layer_means[layer] > 0.05:
            emergence_layer = layer
            break
    if emergence_layer is not None:
        print(f"Emergence layer (mean V|nums > 0.05): L{emergence_layer} "
              f"({100*emergence_layer/(n_layers-1):.0f}% depth)")

    # Early-bin analysis across layers
    print("\n--- V|nums at early bins (0-15%) across layers ---")
    early_bins = [0, 1, 2]  # bins 0-2 = 0-15%
    for layer in range(0, n_layers, 4):
        early_vals = [V_nums_R_matrix[layer, b] for b in early_bins
                      if not np.isnan(V_nums_R_matrix[layer, b])]
        if early_vals:
            mean_early = np.mean(early_vals)
            print(f"  L{layer:2d}: early V|nums = {mean_early:+.3f} "
                  f"(bins: {[f'{v:.3f}' for v in early_vals]})")

    # ── Figure 1: Main heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})

    # V_R heatmap
    ax = axes[0]
    vmin_vr = max(0, np.nanmin(V_R_matrix))
    vmax_vr = np.nanmax(V_R_matrix)
    im1 = ax.imshow(V_R_matrix, aspect='auto', origin='lower',
                     extent=[0, 100, 0, n_layers-1],
                     cmap='YlOrRd', vmin=vmin_vr, vmax=vmax_vr)
    ax.set_xlabel('Chain Position (%)', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('V → answer (raw decodability)', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax, label='Pearson R')
    # Mark probe layers from exp_084
    for pl in [18, 27]:
        ax.axhline(y=pl, color='white', linestyle='--', alpha=0.5, linewidth=0.8)

    # V|nums heatmap
    ax = axes[1]
    # Use diverging colormap centered at 0
    vmax_vn = max(abs(np.nanmin(V_nums_R_matrix)), np.nanmax(V_nums_R_matrix))
    vmax_vn = min(vmax_vn, 0.6)  # cap for readability
    norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=vmax_vn)
    im2 = ax.imshow(V_nums_R_matrix, aspect='auto', origin='lower',
                     extent=[0, 100, 0, n_layers-1],
                     cmap='RdBu_r', norm=norm)
    ax.set_xlabel('Chain Position (%)', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('V|nums → answer (forward-looking)', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax, label='Pearson R (residualized)')
    for pl in [18, 27]:
        ax.axhline(y=pl, color='black', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.suptitle('Experiment 089: Layer × Position Heatmap — Qwen3-4B-Base',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'layer_position_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: layer_position_heatmap.png")

    # ── Figure 2: V|nums layer profile at key positions ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    position_groups = {
        '0-5% (text=0%)': [0],
        '5-15%': [1, 2],
        '25-35%': [5, 6],
        '45-55%': [9, 10],
        '65-75%': [13, 14],
        '85-95%': [17, 18],
    }
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(position_groups)))
    for (label, bins), color in zip(position_groups.items(), colors):
        profile = []
        for layer in range(n_layers):
            vals = [V_nums_R_matrix[layer, b] for b in bins
                    if not np.isnan(V_nums_R_matrix[layer, b])]
            profile.append(np.mean(vals) if vals else np.nan)
        ax.plot(range(n_layers), profile, '-o', markersize=3, color=color,
                label=label, linewidth=1.5)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('V|nums (Pearson R)', fontsize=12)
    ax.set_title('Forward-Looking Signal by Layer at Different Chain Positions',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'layer_profile_by_position.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: layer_profile_by_position.png")

    # ── Figure 3: V_R and V|nums at key layers vs position (line plots) ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    key_layers = [0, 9, 18, 22, 27, 35]
    for idx, layer in enumerate(key_layers):
        ax = axes[idx // 3, idx % 3]
        v_r = V_R_matrix[layer]
        vn_r = V_nums_R_matrix[layer]

        ax.plot(bin_centers, v_r, 'b-o', markersize=3, label='V_R', linewidth=1.5)
        ax.plot(bin_centers, vn_r, 'r-s', markersize=3, label='V|nums', linewidth=1.5)
        ax.plot(bin_centers, nums_R_vector, 'g--', alpha=0.6, label='nums_R', linewidth=1.2)
        ax.plot(bin_centers, np.array(text_reveals_curve) * np.nanmax(v_r),
                'k:', alpha=0.4, label='text reveals', linewidth=1)

        # Shuffle control if available
        if layer in SHUFFLE_LAYERS:
            shuf = shuffle_R_matrix[layer]
            valid_s = ~np.isnan(shuf)
            if valid_s.any():
                ax.plot(bin_centers[valid_s], shuf[valid_s], 'x',
                        color='gray', markersize=4, label='shuffle', alpha=0.5)

        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        depth_pct = 100 * layer / (n_layers - 1)
        ax.set_title(f'Layer {layer} ({depth_pct:.0f}% depth)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Chain Position (%)', fontsize=9)
        ax.set_ylabel('Pearson R', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        ax.set_ylim(-0.15, 0.85)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Exp 089: V_R and V|nums at Key Layers — Qwen3-4B-Base',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'key_layers_detail.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: key_layers_detail.png")

    # ── Figure 4: Mean V|nums across all bins as function of layer ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(range(n_layers), layer_means, color='steelblue', alpha=0.7, edgecolor='navy')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    if emergence_layer is not None:
        ax.axvline(x=emergence_layer, color='red', linestyle='--', alpha=0.6,
                   label=f'Emergence (L{emergence_layer})')
    ax.axvline(x=peak_layer, color='green', linestyle='--', alpha=0.6,
               label=f'Peak (L{peak_layer})')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean V|nums (Pearson R)', fontsize=12)
    ax.set_title('Forward-Looking Signal Strength by Layer (averaged across positions)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'layer_mean_vnums.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: layer_mean_vnums.png")

    # ── Figure 5: Emergence across layers at bin 0 (0-5%, text reveals 0%) ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bin0_vals = V_nums_R_matrix[:, 0]
    bin0_vr = V_R_matrix[:, 0]
    valid_0 = ~np.isnan(bin0_vals)
    ax.plot(np.arange(n_layers)[valid_0], bin0_vr[valid_0], 'b-o', markersize=4,
            label='V_R at 0-5%', linewidth=1.5)
    ax.plot(np.arange(n_layers)[valid_0], bin0_vals[valid_0], 'r-s', markersize=4,
            label='V|nums at 0-5%', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    # Reference values from exp_084
    ax.axhline(y=0.357, color='red', linestyle=':', alpha=0.3,
               label='exp_084 L18 V|nums=0.357')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Pearson R', fontsize=12)
    ax.set_title('Answer Decodability at Chain Start (0-5%, text=0%) by Layer',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'emergence_at_chain_start.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: emergence_at_chain_start.png")

    # ── Save results ──
    results = {
        'n_layers': n_layers,
        'n_bins': N_BINS,
        'n_extracted': n_extracted,
        'n_problems_generated': n_total,
        'n_correct': n_correct,
        'accuracy': round(100 * n_correct / n_total, 1),
        'kv_dim': kv_dim,
        'V_R_matrix': V_R_matrix.tolist(),
        'V_nums_R_matrix': V_nums_R_matrix.tolist(),
        'nums_R_vector': nums_R_vector.tolist(),
        'shuffle_R_matrix': shuffle_R_matrix.tolist(),
        'text_reveals_curve': text_reveals_curve,
        'layer_means': layer_means,
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
    print(f"\nPeak layer: L{peak_layer} (mean V|nums = {peak_mean:.3f})")
    if emergence_layer is not None:
        print(f"Emergence layer: L{emergence_layer} ({100*emergence_layer/(n_layers-1):.0f}% depth)")
    print(f"\nV|nums at bin 0 (0-5%, text=0%):")
    for layer in range(0, n_layers, 4):
        val = V_nums_R_matrix[layer, 0]
        if not np.isnan(val):
            print(f"  L{layer:2d}: V|nums = {val:+.3f}")
    print(f"\nTotal runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
