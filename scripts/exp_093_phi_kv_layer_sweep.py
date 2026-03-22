#!/usr/bin/env python3
"""
Experiment 093: Cross-Model K vs V Layer Sweep — Phi-3.5-mini-Instruct

Tests whether K > V for forward-looking probing (established on Qwen in exp_091)
generalizes to a maximally different architecture: MHA (not GQA), analog encoding
(not digital), different model family.

Extracts BOTH K and V at 12 representative layers, probes K|nums and V|nums at
each (layer, bin), and directly compares K vs V.

DISCONFIRMATORY: K>V may be Qwen/GQA-specific and fail on MHA/analog models.
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
MODEL_NAME = 'microsoft/Phi-3.5-mini-instruct'
N_PROBLEMS = 200
N_BINS = 20
N_FOLDS = 5
MAX_NUMS_DIM = 30

# 12 representative layers spanning 32-layer Phi model
PROBE_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31]
SHUFFLE_LAYERS = [0, 9, 18, 31]  # 4 layers for shuffle controls
N_BOOTSTRAP = 500

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_093"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same as all Phase 2 experiments) ──
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


def probe_cv(X, y, groups, n_folds=N_FOLDS):
    """Cross-validated Ridge probe returning Pearson R of predictions."""
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import RidgeCV
    from scipy.stats import pearsonr

    RIDGE_ALPHAS = np.logspace(-2, 6, 50)
    n_groups = len(np.unique(groups))
    if n_groups < n_folds + 1:
        return 0.0

    gkf = GroupKFold(n_splits=min(n_folds, n_groups))
    preds = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups):
        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit(X[tr], y[tr])
        preds[te] = ridge.predict(X[te])

    if np.std(preds) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    r, _ = pearsonr(preds, y)
    return float(r)


def main():
    t0 = time.time()

    print("=" * 70)
    print("Experiment 093: Cross-Model K vs V Layer Sweep")
    print("Phi-3.5-mini-Instruct (MHA, analog encoding)")
    print(f"Testing K>V generalization: {len(PROBE_LAYERS)} layers, both K+V")
    print("DISCONFIRMATORY: K>V may be Qwen/GQA-specific")
    print("=" * 70)

    # ── Load model ──
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map='auto'
    )
    model.eval()

    n_layers_total = model.config.num_hidden_layers
    n_attn_heads = model.config.num_attention_heads
    kv_heads = getattr(model.config, 'num_key_value_heads', n_attn_heads)
    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // n_attn_heads)
    kv_dim = kv_heads * head_dim
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {n_layers_total}, Attn heads: {n_attn_heads}, KV heads: {kv_heads}")
    print(f"head_dim: {head_dim}, KV dim: {kv_dim}")
    print(f"Architecture: {'MHA' if kv_heads == n_attn_heads else 'GQA'}")
    print(f"Probing at layers: {PROBE_LAYERS}")

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
        if time.time() - t0 > TIME_BUDGET * 0.18:
            print(f"  Time budget (18%) reached at problem {i}")
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
    # PHASE 2: Forward pass + K and V extraction at selected layers
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 2: Extracting K+V at {len(PROBE_LAYERS)} layers")
    print("=" * 70)

    shared = {
        'cumNums': [], 'rel_pos': [], 'text_reveals': [],
        'final_answer': [], 'problem_idx': [],
    }
    K_by_layer = {layer: [] for layer in PROBE_LAYERS}
    V_by_layer = {layer: [] for layer in PROBE_LAYERS}

    first_reveal_positions = []
    n_extracted = 0

    for pi, gen in enumerate(correct_gens):
        if time.time() - t0 > TIME_BUDGET * 0.50:
            print(f"\n  Time budget (50%) reached at problem {pi}")
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

        # Forward pass — gets KV cache
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

        # Store shared data
        for j in range(cot_length):
            shared['cumNums'].append(token_cumNums[j])
            shared['rel_pos'].append(j / cot_length)
            shared['text_reveals'].append(token_reveals[j])
            shared['final_answer'].append(final_answer_log)
            shared['problem_idx'].append(pi)

        # Extract K and V vectors at selected layers
        for layer in PROBE_LAYERS:
            K_layer, V_layer = get_kv(kv_cache, layer)
            for j in range(cot_length):
                abs_pos = prompt_len + j
                k_vec = K_layer[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                v_vec = V_layer[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                K_by_layer[layer].append(k_vec)
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

    # Convert to numpy
    for key in shared:
        shared[key] = np.array(shared[key])

    print("Converting K+V arrays to numpy...")
    for layer in PROBE_LAYERS:
        K_by_layer[layer] = np.array(K_by_layer[layer], dtype=np.float32)
        V_by_layer[layer] = np.array(V_by_layer[layer], dtype=np.float32)
    sample_layer = PROBE_LAYERS[0]
    print(f"  K shape per layer: {K_by_layer[sample_layer].shape}")
    print(f"  V shape per layer: {V_by_layer[sample_layer].shape}")
    kv_mem_gb = (K_by_layer[sample_layer].nbytes + V_by_layer[sample_layer].nbytes) * len(PROBE_LAYERS) / 1e9
    print(f"  Total K+V memory: {kv_mem_gb:.1f} GB")

    # Free model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded.")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Position-sweep probing K and V at all selected layers
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 3: Probing K|nums and V|nums at {len(PROBE_LAYERS)} layers × {N_BINS} bins")
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

    # Precompute bin data: nums_R and y_resid per bin
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

        # Compute nums_R (cross-validated)
        nums_preds = np.zeros_like(y_bin)
        for tr, te in gkf.split(nums_bin, y_bin, g_bin):
            ridge = RidgeCV(alphas=RIDGE_ALPHAS)
            ridge.fit(nums_bin[tr], y_bin[tr])
            nums_preds[te] = ridge.predict(nums_bin[te])
        nums_r, _ = pearsonr(nums_preds, y_bin)

        # Compute y_resid (cross-validated residuals)
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

    # Probe K and V at each (layer, bin)
    n_probe_layers = len(PROBE_LAYERS)
    K_R_matrix = np.full((n_probe_layers, N_BINS), np.nan)
    V_R_matrix = np.full((n_probe_layers, N_BINS), np.nan)
    K_nums_R_matrix = np.full((n_probe_layers, N_BINS), np.nan)
    V_nums_R_matrix = np.full((n_probe_layers, N_BINS), np.nan)
    nums_R_vector = np.full(N_BINS, np.nan)
    shuffle_K_R_matrix = np.full((n_probe_layers, N_BINS), np.nan)
    shuffle_V_R_matrix = np.full((n_probe_layers, N_BINS), np.nan)

    for b in range(N_BINS):
        if bin_data[b].get('skip', True):
            continue
        nums_R_vector[b] = bin_data[b]['nums_r']

    for li, layer in enumerate(PROBE_LAYERS):
        layer_t0 = time.time()
        K_layer = K_by_layer[layer]
        V_layer = V_by_layer[layer]

        for b in range(N_BINS):
            if bin_data[b].get('skip', True):
                continue

            bd = bin_data[b]
            mask = bd['mask']
            K_bin = K_layer[mask]
            V_bin = V_layer[mask]
            y_bin = bd['y_bin']
            y_resid = bd['y_resid']
            g_bin = bd['g_bin']

            # K → answer
            K_R_matrix[li, b] = probe_cv(K_bin, y_bin, g_bin)

            # V → answer
            V_R_matrix[li, b] = probe_cv(V_bin, y_bin, g_bin)

            # K → y_resid (K|nums)
            if bd['resid_std'] > 1e-10:
                K_nums_R_matrix[li, b] = probe_cv(K_bin, y_resid, g_bin)
            else:
                K_nums_R_matrix[li, b] = 0.0

            # V → y_resid (V|nums)
            if bd['resid_std'] > 1e-10:
                V_nums_R_matrix[li, b] = probe_cv(V_bin, y_resid, g_bin)
            else:
                V_nums_R_matrix[li, b] = 0.0

            # Shuffle controls
            if layer in SHUFFLE_LAYERS:
                rng = np.random.RandomState(SEED + layer * 1000 + b)
                shuf_idx = rng.permutation(len(y_bin))
                y_shuf = y_bin[shuf_idx]
                g_shuf = g_bin[shuf_idx]
                shuffle_K_R_matrix[li, b] = probe_cv(K_bin, y_shuf, g_shuf)
                shuffle_V_R_matrix[li, b] = probe_cv(V_bin, y_shuf, g_shuf)

        elapsed_layer = time.time() - layer_t0
        elapsed_total = time.time() - t0

        valid = ~np.isnan(K_nums_R_matrix[li])
        mean_kn = np.nanmean(K_nums_R_matrix[li]) if valid.any() else 0
        mean_vn = np.nanmean(V_nums_R_matrix[li]) if valid.any() else 0
        n_k_wins = int(np.sum(K_nums_R_matrix[li][valid] > V_nums_R_matrix[li][valid])) if valid.any() else 0
        n_valid = int(valid.sum())

        print(f"  L{layer:2d}: K|nums={mean_kn:+.3f}, V|nums={mean_vn:+.3f}, "
              f"K>V={n_k_wins}/{n_valid} [{elapsed_layer:.1f}s, total {elapsed_total:.0f}s]")

        # Time check
        if time.time() - t0 > TIME_BUDGET * 0.90:
            print(f"\n  Time budget (90%) reached after layer {layer}")
            # Fill remaining layers with NaN
            for remaining_li in range(li + 1, n_probe_layers):
                pass  # already NaN
            break

    print(f"\nPhase 3 complete. Total elapsed: {time.time()-t0:.0f}s")

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Bootstrap significance test for K-V difference
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: Bootstrap significance test for K vs V")
    print("=" * 70)

    # For each layer, bootstrap the mean K|nums - V|nums across bins
    bootstrap_results = {}
    for li, layer in enumerate(PROBE_LAYERS):
        k_vals = K_nums_R_matrix[li]
        v_vals = V_nums_R_matrix[li]
        valid = ~(np.isnan(k_vals) | np.isnan(v_vals))
        if valid.sum() < 5:
            continue

        diffs = k_vals[valid] - v_vals[valid]
        obs_mean = float(np.mean(diffs))

        # Bootstrap
        rng = np.random.RandomState(SEED + layer)
        boot_means = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(diffs), size=len(diffs), replace=True)
            boot_means.append(float(np.mean(diffs[idx])))
        boot_means = np.array(boot_means)

        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))
        # p-value: fraction of bootstrap samples where sign flips
        if obs_mean > 0:
            p_val = float(np.mean(boot_means <= 0))
        else:
            p_val = float(np.mean(boot_means >= 0))

        bootstrap_results[layer] = {
            'obs_mean_diff': obs_mean,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'p_value': p_val,
            'n_bins': int(valid.sum()),
        }

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  L{layer:2d}: K-V = {obs_mean:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}], p={p_val:.3f} {sig}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: Analysis and Visualization
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 5: Analysis and Visualization")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 * 100

    # ── Summary statistics ──
    print("\n--- Layer-wise K|nums vs V|nums ---")
    k_layer_means = []
    v_layer_means = []
    k_wins_total = 0
    v_wins_total = 0
    for li, layer in enumerate(PROBE_LAYERS):
        k_valid = ~np.isnan(K_nums_R_matrix[li])
        v_valid = ~np.isnan(V_nums_R_matrix[li])
        both_valid = k_valid & v_valid

        mean_kn = float(np.nanmean(K_nums_R_matrix[li])) if k_valid.any() else np.nan
        mean_vn = float(np.nanmean(V_nums_R_matrix[li])) if v_valid.any() else np.nan
        k_layer_means.append(mean_kn)
        v_layer_means.append(mean_vn)

        if both_valid.any():
            k_w = int(np.sum(K_nums_R_matrix[li][both_valid] > V_nums_R_matrix[li][both_valid]))
            v_w = int(both_valid.sum()) - k_w
            k_wins_total += k_w
            v_wins_total += v_w
        else:
            k_w = v_w = 0

        n_k_pos = int(np.sum(K_nums_R_matrix[li][k_valid] > 0)) if k_valid.any() else 0
        n_v_pos = int(np.sum(V_nums_R_matrix[li][v_valid] > 0)) if v_valid.any() else 0
        n_valid = int(k_valid.sum())

        depth_pct = 100 * layer / (n_layers_total - 1)
        diff = mean_kn - mean_vn if not (np.isnan(mean_kn) or np.isnan(mean_vn)) else np.nan

        print(f"  L{layer:2d} ({depth_pct:5.1f}%): K|nums={mean_kn:+.3f}, V|nums={mean_vn:+.3f}, "
              f"diff={diff:+.3f}, K>V bins={k_w}/{n_valid}, pos_K={n_k_pos}, pos_V={n_v_pos}")

    # Overall statistics
    k_arr = np.array(k_layer_means)
    v_arr = np.array(v_layer_means)
    valid_both = ~(np.isnan(k_arr) | np.isnan(v_arr))
    mean_k_all = float(np.nanmean(k_arr[valid_both])) if valid_both.any() else 0
    mean_v_all = float(np.nanmean(v_arr[valid_both])) if valid_both.any() else 0
    k_gt_v_layers = int(np.sum(k_arr[valid_both] > v_arr[valid_both])) if valid_both.any() else 0
    n_valid_layers = int(valid_both.sum())

    print(f"\n--- Overall K vs V ---")
    print(f"  Mean K|nums: {mean_k_all:+.3f}")
    print(f"  Mean V|nums: {mean_v_all:+.3f}")
    print(f"  Mean diff (K-V): {mean_k_all - mean_v_all:+.3f}")
    print(f"  K>V at {k_gt_v_layers}/{n_valid_layers} layers")
    print(f"  K>V at {k_wins_total}/{k_wins_total+v_wins_total} (layer,bin) cells")

    # Phase-wise comparison
    ramp_indices = [i for i, l in enumerate(PROBE_LAYERS) if l < 10]
    plateau_indices = [i for i, l in enumerate(PROBE_LAYERS) if l >= 10]

    if ramp_indices:
        ramp_k = np.mean([k_layer_means[i] for i in ramp_indices if not np.isnan(k_layer_means[i])])
        ramp_v = np.mean([v_layer_means[i] for i in ramp_indices if not np.isnan(v_layer_means[i])])
        print(f"  Ramp (L<10): K|nums={ramp_k:+.3f}, V|nums={ramp_v:+.3f}, diff={ramp_k-ramp_v:+.3f}")
    if plateau_indices:
        plat_k = np.mean([k_layer_means[i] for i in plateau_indices if not np.isnan(k_layer_means[i])])
        plat_v = np.mean([v_layer_means[i] for i in plateau_indices if not np.isnan(v_layer_means[i])])
        print(f"  Plateau (L≥10): K|nums={plat_k:+.3f}, V|nums={plat_v:+.3f}, diff={plat_k-plat_v:+.3f}")

    # K and V emergence
    k_emergence = None
    v_emergence = None
    for li, layer in enumerate(PROBE_LAYERS):
        if k_emergence is None and not np.isnan(k_layer_means[li]) and k_layer_means[li] > 0.05:
            k_emergence = layer
        if v_emergence is None and not np.isnan(v_layer_means[li]) and v_layer_means[li] > 0.05:
            v_emergence = layer
    print(f"\n  K emergence (>0.05): L{k_emergence}")
    print(f"  V emergence (>0.05): L{v_emergence}")

    # K and V peaks
    k_peak_li = int(np.nanargmax(k_arr)) if valid_both.any() else 0
    v_peak_li = int(np.nanargmax(v_arr)) if valid_both.any() else 0
    print(f"  K peak: L{PROBE_LAYERS[k_peak_li]} ({k_arr[k_peak_li]:+.3f})")
    print(f"  V peak: L{PROBE_LAYERS[v_peak_li]} ({v_arr[v_peak_li]:+.3f})")

    # ── Figure 1: K|nums vs V|nums layer profile ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    ax = axes[0]
    x = np.arange(n_probe_layers)
    width = 0.35
    ax.bar(x - width/2, k_layer_means, width, color='indianred', alpha=0.7,
           edgecolor='darkred', label='K|nums')
    ax.bar(x + width/2, v_layer_means, width, color='steelblue', alpha=0.7,
           edgecolor='navy', label='V|nums')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean |nums (Pearson R)', fontsize=12)
    ax.set_title('K vs V Forward-Looking Signal by Layer', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in PROBE_LAYERS], rotation=45, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # K-V difference with bootstrap CIs
    ax = axes[1]
    diffs_mean = []
    diffs_ci_lo = []
    diffs_ci_hi = []
    sig_markers = []
    for li, layer in enumerate(PROBE_LAYERS):
        diff = k_layer_means[li] - v_layer_means[li] if not (np.isnan(k_layer_means[li]) or np.isnan(v_layer_means[li])) else 0
        diffs_mean.append(diff)
        if layer in bootstrap_results:
            br = bootstrap_results[layer]
            diffs_ci_lo.append(br['ci_lo'])
            diffs_ci_hi.append(br['ci_hi'])
            sig_markers.append(br['p_value'] < 0.05)
        else:
            diffs_ci_lo.append(diff)
            diffs_ci_hi.append(diff)
            sig_markers.append(False)

    colors = ['darkred' if d > 0 else 'navy' for d in diffs_mean]
    bars = ax.bar(x, diffs_mean, 0.6, color=colors, alpha=0.7)
    # Error bars
    yerr_lo = [diffs_mean[i] - diffs_ci_lo[i] for i in range(len(diffs_mean))]
    yerr_hi = [diffs_ci_hi[i] - diffs_mean[i] for i in range(len(diffs_mean))]
    ax.errorbar(x, diffs_mean, yerr=[yerr_lo, yerr_hi], fmt='none', ecolor='black',
                capsize=3, linewidth=1.5)
    # Significance markers
    for i, sig in enumerate(sig_markers):
        if sig:
            y_pos = max(diffs_ci_hi[i], diffs_mean[i]) + 0.005
            ax.text(i, y_pos, '*', ha='center', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('K|nums − V|nums', fontsize=12)
    ax.set_title('K−V Difference (red=K wins, blue=V wins)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in PROBE_LAYERS], rotation=45, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Exp 093: K vs V Layer Sweep — Phi-3.5-mini-Instruct (MHA)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'k_vs_v_layer_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: k_vs_v_layer_profile.png")

    # ── Figure 2: K|nums and V|nums heatmaps ──
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    vmax_comp = 0.6
    norm_comp = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=vmax_comp)

    # K|nums heatmap
    ax = axes[0]
    im1 = ax.imshow(K_nums_R_matrix, aspect='auto', origin='lower',
                     extent=[0, 100, -0.5, n_probe_layers - 0.5],
                     cmap='RdBu_r', norm=norm_comp)
    ax.set_xlabel('Chain Position (%)', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('K|nums', fontsize=13, fontweight='bold')
    ax.set_yticks(range(n_probe_layers))
    ax.set_yticklabels([f'L{l}' for l in PROBE_LAYERS], fontsize=8)
    plt.colorbar(im1, ax=ax, label='Pearson R')

    # V|nums heatmap
    ax = axes[1]
    im2 = ax.imshow(V_nums_R_matrix, aspect='auto', origin='lower',
                     extent=[0, 100, -0.5, n_probe_layers - 0.5],
                     cmap='RdBu_r', norm=norm_comp)
    ax.set_xlabel('Chain Position (%)', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('V|nums', fontsize=13, fontweight='bold')
    ax.set_yticks(range(n_probe_layers))
    ax.set_yticklabels([f'L{l}' for l in PROBE_LAYERS], fontsize=8)
    plt.colorbar(im2, ax=ax, label='Pearson R')

    # K-V difference
    ax = axes[2]
    diff_matrix = K_nums_R_matrix - V_nums_R_matrix
    diff_valid = ~(np.isnan(K_nums_R_matrix) | np.isnan(V_nums_R_matrix))
    diff_display = np.where(diff_valid, diff_matrix, np.nan)
    vmax_diff = 0.25
    norm_diff = TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)
    im3 = ax.imshow(diff_display, aspect='auto', origin='lower',
                     extent=[0, 100, -0.5, n_probe_layers - 0.5],
                     cmap='PiYG', norm=norm_diff)
    ax.set_xlabel('Chain Position (%)', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('K|nums − V|nums (green=K wins)', fontsize=13, fontweight='bold')
    ax.set_yticks(range(n_probe_layers))
    ax.set_yticklabels([f'L{l}' for l in PROBE_LAYERS], fontsize=8)
    plt.colorbar(im3, ax=ax, label='K−V')

    plt.suptitle('Exp 093: K vs V Heatmap — Phi-3.5-mini-Instruct (MHA, analog)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'k_vs_v_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: k_vs_v_heatmap.png")

    # ── Figure 3: Position sweep at key layers (K and V on same plot) ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    key_layer_indices = [0, 2, 4, 6, 8, 10]  # L0, L6, L12, L18, L24, L30
    for ax_idx, li in enumerate(key_layer_indices):
        if li >= n_probe_layers:
            break
        ax = axes.flat[ax_idx]
        layer = PROBE_LAYERS[li]

        k_vals = K_nums_R_matrix[li]
        v_vals = V_nums_R_matrix[li]
        valid_k = ~np.isnan(k_vals)
        valid_v = ~np.isnan(v_vals)

        ax.plot(bin_centers[valid_k], k_vals[valid_k], 'r-s', markersize=3,
                label='K|nums', linewidth=1.5)
        ax.plot(bin_centers[valid_v], v_vals[valid_v], 'b-o', markersize=3,
                label='V|nums', linewidth=1.5)

        # Shuffle if available
        if layer in SHUFFLE_LAYERS:
            sk = shuffle_K_R_matrix[li]
            sv = shuffle_V_R_matrix[li]
            valid_sk = ~np.isnan(sk)
            valid_sv = ~np.isnan(sv)
            if valid_sk.any():
                ax.plot(bin_centers[valid_sk], sk[valid_sk], 'r--', alpha=0.3,
                        linewidth=0.8, label='K shuffle')
            if valid_sv.any():
                ax.plot(bin_centers[valid_sv], sv[valid_sv], 'b--', alpha=0.3,
                        linewidth=0.8, label='V shuffle')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Chain Position (%)')
        ax.set_ylabel('Pearson R')
        depth_pct = 100 * layer / (n_layers_total - 1)
        ax.set_title(f'L{layer} ({depth_pct:.0f}% depth)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Exp 093: K vs V Position Sweep at Key Layers — Phi-3.5-mini',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'position_sweep_key_layers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: position_sweep_key_layers.png")

    # ── Figure 4: Cross-model comparison (Phi vs Qwen091 if available) ──
    qwen_results_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_091" / "results.json"
    if qwen_results_path.exists():
        with open(qwen_results_path) as f:
            qwen_data = json.load(f)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Qwen K and V layer means (full 36 layers)
        qwen_k = qwen_data['k_layer_means']
        qwen_v = qwen_data.get('v_layer_means_089', [])
        qwen_layers = list(range(len(qwen_k)))
        qwen_depth = [100 * l / 35 for l in qwen_layers]

        # Phi K and V layer means (12 layers)
        phi_depth = [100 * l / (n_layers_total - 1) for l in PROBE_LAYERS]

        ax.plot(qwen_depth, qwen_k, 'r-', alpha=0.5, linewidth=1.5, label='Qwen K|nums')
        if qwen_v:
            ax.plot(qwen_depth, qwen_v, 'b-', alpha=0.5, linewidth=1.5, label='Qwen V|nums')
        ax.plot(phi_depth, k_layer_means, 'r-s', markersize=6, linewidth=2, label='Phi K|nums')
        ax.plot(phi_depth, v_layer_means, 'b-o', markersize=6, linewidth=2, label='Phi V|nums')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Depth (%)', fontsize=12)
        ax.set_ylabel('Mean |nums (Pearson R)', fontsize=12)
        ax.set_title('Cross-Model: Qwen3-4B-Base (GQA) vs Phi-3.5-mini (MHA)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / 'cross_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: cross_model_comparison.png")
    else:
        print("Qwen exp_091 results not found — skipping cross-model figure")

    # ── Save results ──
    results = {
        'model': MODEL_NAME,
        'architecture': 'MHA' if kv_heads == n_attn_heads else 'GQA',
        'n_layers_total': n_layers_total,
        'n_attn_heads': n_attn_heads,
        'kv_heads': kv_heads,
        'head_dim': head_dim,
        'kv_dim': kv_dim,
        'probe_layers': PROBE_LAYERS,
        'n_bins': N_BINS,
        'n_extracted': n_extracted,
        'n_problems_generated': n_total,
        'n_correct': n_correct,
        'accuracy': round(100 * n_correct / n_total, 1),
        'K_R_matrix': K_R_matrix.tolist(),
        'V_R_matrix': V_R_matrix.tolist(),
        'K_nums_R_matrix': K_nums_R_matrix.tolist(),
        'V_nums_R_matrix': V_nums_R_matrix.tolist(),
        'nums_R_vector': nums_R_vector.tolist(),
        'shuffle_K_R_matrix': shuffle_K_R_matrix.tolist(),
        'shuffle_V_R_matrix': shuffle_V_R_matrix.tolist(),
        'text_reveals_curve': text_reveals_curve,
        'k_layer_means': k_layer_means,
        'v_layer_means': v_layer_means,
        'mean_k_all': mean_k_all,
        'mean_v_all': mean_v_all,
        'k_gt_v_layers': k_gt_v_layers,
        'n_valid_layers': n_valid_layers,
        'bootstrap_results': {str(k): v for k, v in bootstrap_results.items()},
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

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {MODEL_NAME} ({results['architecture']})")
    print(f"Layers: {n_layers_total}, KV heads: {kv_heads}, KV dim: {kv_dim}")
    print(f"Problems: {n_total} generated, {n_correct} correct ({100*n_correct/n_total:.1f}%)")
    print(f"Extracted: {n_extracted} problems, {len(shared['rel_pos'])} vectors/layer")
    print(f"\nK|nums mean: {mean_k_all:+.3f}")
    print(f"V|nums mean: {mean_v_all:+.3f}")
    print(f"K-V diff: {mean_k_all - mean_v_all:+.3f}")
    print(f"K>V at {k_gt_v_layers}/{n_valid_layers} layers ({100*k_gt_v_layers/max(n_valid_layers,1):.0f}%)")
    if k_emergence is not None:
        print(f"K emergence: L{k_emergence}")
    if v_emergence is not None:
        print(f"V emergence: L{v_emergence}")
    print(f"K peak: L{PROBE_LAYERS[k_peak_li]} ({k_arr[k_peak_li]:+.3f})")
    print(f"V peak: L{PROBE_LAYERS[v_peak_li]} ({v_arr[v_peak_li]:+.3f})")

    # Bootstrap summary
    n_sig = sum(1 for br in bootstrap_results.values() if br['p_value'] < 0.05)
    print(f"\nBootstrap: {n_sig}/{len(bootstrap_results)} layers with significant K-V diff (p<0.05)")

    # Verdict
    print("\n--- VERDICT ---")
    if k_gt_v_layers >= 8:
        print("K > V CONFIRMED on Phi (MHA, analog) — K>V probing is UNIVERSAL")
    elif k_gt_v_layers >= 4 and k_gt_v_layers < 8:
        print("K > V PARTIAL on Phi — weaker than on Qwen, architecture-modulated")
    elif abs(mean_k_all - mean_v_all) < 0.015:
        print("K ≈ V on Phi — K>V is GQA/digital-SPECIFIC, not universal")
    else:
        print(f"K vs V: {k_gt_v_layers}/{n_valid_layers} layers K>V — see details above")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
