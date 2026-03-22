#!/usr/bin/env python3
"""
Experiment 094: Mistral K vs V Layer Sweep — GQA-General or Qwen-Specific?

Tests whether K > V for forward-looking probing (established on Qwen GQA in exp_091,
REVERSED on Phi MHA in exp_093) extends to a second GQA model: Mistral-7B-v0.3.

Mistral shares GQA (8 KV heads) with Qwen but analog encoding with Phi.
If K>V on Mistral → GQA drives K>V. If V>K → Qwen-specific (digital encoding).

Extracts BOTH K and V at 12 representative layers, probes K|nums and V|nums at
each (layer, bin), and directly compares K vs V.
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
MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
N_PROBLEMS = 250  # Mistral accuracy ~44%, need more problems for sufficient correct samples
N_BINS = 20
N_FOLDS = 5
MAX_NUMS_DIM = 30

# 12 representative layers spanning 32-layer Llama model
PROBE_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31]
SHUFFLE_LAYERS = [0, 9, 18, 31]  # 4 layers for shuffle controls
N_BOOTSTRAP = 500

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_094"
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
    print("Experiment 094: Mistral K vs V Layer Sweep")
    print("Mistral-7B-v0.3 (GQA, analog encoding)")
    print(f"Testing K>V generalization: {len(PROBE_LAYERS)} layers, both K+V")
    print("CRITICAL TEST: GQA-general (K>V) or Qwen-specific?")
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
    print(f"GQA ratio: {n_attn_heads / kv_heads:.1f}x (Q heads per KV head)")
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
            break

    print(f"\nPhase 3 complete. Total elapsed: {time.time()-t0:.0f}s")

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Bootstrap significance test for K-V difference
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: Bootstrap significance test for K vs V")
    print("=" * 70)

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
        # Two-sided p-value: fraction of bootstraps on wrong side of zero
        if obs_mean > 0:
            p_val = float(np.mean(boot_means <= 0)) * 2
        else:
            p_val = float(np.mean(boot_means >= 0)) * 2
        p_val = min(p_val, 1.0)

        bootstrap_results[layer] = {
            'obs_diff': obs_mean,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'p_val': p_val,
            'n_bins': int(valid.sum()),
            'k_wins': int(np.sum(diffs > 0)),
            'v_wins': int(np.sum(diffs < 0)),
        }
        winner = "K>V" if obs_mean > 0 else "V>K"
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  L{layer:2d}: {winner} diff={obs_mean:+.4f} "
              f"CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] p={p_val:.3f} {sig} "
              f"(K>V={bootstrap_results[layer]['k_wins']}, V>K={bootstrap_results[layer]['v_wins']})")

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: Summary statistics and cross-model comparison
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 5: Summary and Cross-Model Comparison")
    print("=" * 70)

    # Overall K>V count across layers
    total_k_wins = 0
    total_v_wins = 0
    total_layers = 0
    sig_k_wins = 0
    sig_v_wins = 0

    for li, layer in enumerate(PROBE_LAYERS):
        k_mean = np.nanmean(K_nums_R_matrix[li])
        v_mean = np.nanmean(V_nums_R_matrix[li])
        if np.isnan(k_mean) or np.isnan(v_mean):
            continue
        total_layers += 1
        if k_mean > v_mean:
            total_k_wins += 1
        else:
            total_v_wins += 1

        if layer in bootstrap_results:
            br = bootstrap_results[layer]
            if br['p_val'] < 0.05:
                if br['obs_diff'] > 0:
                    sig_k_wins += 1
                else:
                    sig_v_wins += 1

    print(f"\nOverall K>V layer wins: {total_k_wins}/{total_layers} "
          f"({100*total_k_wins/total_layers if total_layers else 0:.0f}%)")
    print(f"Overall V>K layer wins: {total_v_wins}/{total_layers} "
          f"({100*total_v_wins/total_layers if total_layers else 0:.0f}%)")
    print(f"Significant (p<0.05): K>V at {sig_k_wins} layers, V>K at {sig_v_wins} layers")

    # Mean across all layers
    all_k_nums = []
    all_v_nums = []
    for li, layer in enumerate(PROBE_LAYERS):
        k_mean = np.nanmean(K_nums_R_matrix[li])
        v_mean = np.nanmean(V_nums_R_matrix[li])
        if not np.isnan(k_mean):
            all_k_nums.append(k_mean)
        if not np.isnan(v_mean):
            all_v_nums.append(v_mean)

    mean_k = np.mean(all_k_nums) if all_k_nums else 0
    mean_v = np.mean(all_v_nums) if all_v_nums else 0
    mean_diff = mean_k - mean_v
    print(f"\nMean K|nums across layers: {mean_k:+.3f}")
    print(f"Mean V|nums across layers: {mean_v:+.3f}")
    print(f"Mean K-V difference: {mean_diff:+.3f}")

    # Phase breakdown (ramp vs plateau)
    ramp_layers = [l for l in PROBE_LAYERS if l <= 9]
    plateau_layers = [l for l in PROBE_LAYERS if l > 9]

    for phase_name, phase_layers in [("Ramp (L0-L9)", ramp_layers), ("Plateau (L10-L31)", plateau_layers)]:
        k_phase = []
        v_phase = []
        for layer in phase_layers:
            li = PROBE_LAYERS.index(layer)
            k_m = np.nanmean(K_nums_R_matrix[li])
            v_m = np.nanmean(V_nums_R_matrix[li])
            if not np.isnan(k_m):
                k_phase.append(k_m)
            if not np.isnan(v_m):
                v_phase.append(v_m)
        if k_phase and v_phase:
            print(f"  {phase_name}: K|nums={np.mean(k_phase):+.3f}, V|nums={np.mean(v_phase):+.3f}, "
                  f"diff={np.mean(k_phase)-np.mean(v_phase):+.3f}")

    # Cross-model comparison table
    print("\n" + "─" * 70)
    print("CROSS-MODEL K vs V PROBING COMPARISON:")
    print("─" * 70)
    print(f"  Qwen3-4B-Base  (GQA, digital): K>V at 32/36 layers (89%), diff=+0.043")
    print(f"  Phi-3.5-mini   (MHA, analog):  V>K at 10/12 layers (83%), diff=-0.048")
    print(f"  Mistral-7B     (GQA, analog):  {'K>V' if total_k_wins > total_v_wins else 'V>K' if total_v_wins > total_k_wins else 'K≈V'}"
          f" at {max(total_k_wins, total_v_wins)}/{total_layers} layers "
          f"({100*max(total_k_wins, total_v_wins)/total_layers if total_layers else 0:.0f}%), "
          f"diff={mean_diff:+.3f}")
    print("─" * 70)

    if total_k_wins > total_v_wins * 1.5:
        print("INTERPRETATION: K>V on Mistral → GQA-GENERAL (compression drives K>V)")
    elif total_v_wins > total_k_wins * 1.5:
        print("INTERPRETATION: V>K on Mistral → K>V is Qwen-specific (digital encoding)")
    else:
        print("INTERPRETATION: K≈V on Mistral → MIXED; GQA alone insufficient for K>V")

    # ══════════════════════════════════════════════════════════════
    # PHASE 6: Generate figures
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 6: Generating figures")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: K vs V layer profile (mean |nums across bins)
    fig, ax = plt.subplots(figsize=(10, 5))
    layer_means_k = [np.nanmean(K_nums_R_matrix[li]) for li in range(n_probe_layers)]
    layer_means_v = [np.nanmean(V_nums_R_matrix[li]) for li in range(n_probe_layers)]

    ax.plot(PROBE_LAYERS[:len(layer_means_k)], layer_means_k, 'o-', color='#e74c3c',
            label='K|nums (key)', linewidth=2, markersize=6)
    ax.plot(PROBE_LAYERS[:len(layer_means_v)], layer_means_v, 's-', color='#3498db',
            label='V|nums (value)', linewidth=2, markersize=6)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean |nums (Pearson R, forward-looking)', fontsize=12)
    ax.set_title('Mistral-7B-v0.3: K vs V Forward-Looking Probing by Layer\n'
                 '(GQA, analog encoding — tests GQA-general vs Qwen-specific)', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add bootstrap significance markers
    for li, layer in enumerate(PROBE_LAYERS):
        if layer in bootstrap_results:
            br = bootstrap_results[layer]
            if br['p_val'] < 0.05:
                y_pos = max(layer_means_k[li], layer_means_v[li]) + 0.01
                ax.text(layer, y_pos, '*', fontsize=14, ha='center', color='red')

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'k_vs_v_layer_profile.png', dpi=150)
    plt.close(fig)
    print("  Saved k_vs_v_layer_profile.png")

    # Figure 2: K|nums and V|nums heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax_idx, (matrix, name, cmap) in enumerate([
        (K_nums_R_matrix, 'K|nums', 'Reds'),
        (V_nums_R_matrix, 'V|nums', 'Blues'),
    ]):
        ax = axes[ax_idx]
        valid_matrix = np.where(np.isnan(matrix), 0, matrix)
        vmax = max(np.nanmax(np.abs(K_nums_R_matrix)), np.nanmax(np.abs(V_nums_R_matrix)))
        im = ax.imshow(valid_matrix, aspect='auto', cmap=cmap,
                       vmin=-0.05, vmax=max(0.3, vmax),
                       origin='lower', interpolation='nearest')
        ax.set_xlabel('Position bin (0=start, 19=end)', fontsize=10)
        ax.set_ylabel('Layer index', fontsize=10)
        ax.set_yticks(range(n_probe_layers))
        ax.set_yticklabels([f'L{l}' for l in PROBE_LAYERS[:n_probe_layers]], fontsize=8)
        ax.set_title(f'Mistral-7B: {name}', fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('K vs V Forward-Looking Signal Heatmap (Mistral, GQA, analog)', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'k_vs_v_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved k_vs_v_heatmap.png")

    # Figure 3: Position sweep at key layers
    key_display_layers = [l for l in [6, 15, 24, 31] if l in PROBE_LAYERS]
    fig, axes = plt.subplots(1, len(key_display_layers), figsize=(4 * len(key_display_layers), 4))
    if len(key_display_layers) == 1:
        axes = [axes]
    bin_centers = [(bin_edges[b] + bin_edges[b+1]) / 2 for b in range(N_BINS)]

    for ax, layer in zip(axes, key_display_layers):
        li = PROBE_LAYERS.index(layer)
        k_vals = K_nums_R_matrix[li]
        v_vals = V_nums_R_matrix[li]
        ax.plot(bin_centers, np.where(np.isnan(k_vals), 0, k_vals),
                'o-', color='#e74c3c', label='K|nums', markersize=3, linewidth=1.5)
        ax.plot(bin_centers, np.where(np.isnan(v_vals), 0, v_vals),
                's-', color='#3498db', label='V|nums', markersize=3, linewidth=1.5)

        # Text reveals
        ax2 = ax.twinx()
        ax2.fill_between(bin_centers, text_reveals_curve, alpha=0.15, color='green', label='Text reveals')
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel('Frac text revealed', fontsize=8, color='green')

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Relative position', fontsize=9)
        ax.set_ylabel('|nums (Pearson R)', fontsize=9)
        ax.set_title(f'L{layer}', fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Mistral-7B: K vs V Position Sweep at Key Layers', fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'position_sweep_key_layers.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved position_sweep_key_layers.png")

    # Figure 4: Cross-model comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ['Qwen3-4B\n(GQA, digital)', 'Phi-3.5-mini\n(MHA, analog)', 'Mistral-7B\n(GQA, analog)']
    k_means = [0.219, 0.120, mean_k]  # From exp 091, 093, this exp
    v_means = [0.193, 0.167, mean_v]

    x = np.arange(len(models))
    width = 0.35
    bars_k = ax.bar(x - width/2, k_means, width, label='K|nums', color='#e74c3c', alpha=0.8)
    bars_v = ax.bar(x + width/2, v_means, width, label='V|nums', color='#3498db', alpha=0.8)

    ax.set_ylabel('Mean |nums (Pearson R)', fontsize=11)
    ax.set_title('K vs V Forward-Looking Probing: Cross-Model Comparison\n'
                 'GQA drives K>V? Or Qwen-specific?', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for bar in bars_k:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars_v:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'cross_model_comparison.png', dpi=150)
    plt.close(fig)
    print("  Saved cross_model_comparison.png")

    # ══════════════════════════════════════════════════════════════
    # PHASE 7: Save results JSON
    # ══════════════════════════════════════════════════════════════
    results = {
        'experiment': 94,
        'model': MODEL_NAME,
        'n_layers_total': n_layers_total,
        'n_attn_heads': n_attn_heads,
        'kv_heads': kv_heads,
        'head_dim': head_dim,
        'kv_dim': kv_dim,
        'architecture': 'GQA' if kv_heads != n_attn_heads else 'MHA',
        'gqa_ratio': n_attn_heads / kv_heads,
        'n_problems_generated': n_total,
        'n_correct': n_correct,
        'accuracy_pct': 100 * n_correct / n_total,
        'n_extracted': n_extracted,
        'probe_layers': PROBE_LAYERS,
        'n_bins': N_BINS,
        'K_nums_R_matrix': K_nums_R_matrix.tolist(),
        'V_nums_R_matrix': V_nums_R_matrix.tolist(),
        'K_R_matrix': K_R_matrix.tolist(),
        'V_R_matrix': V_R_matrix.tolist(),
        'nums_R_vector': nums_R_vector.tolist(),
        'shuffle_K_R_matrix': shuffle_K_R_matrix.tolist(),
        'shuffle_V_R_matrix': shuffle_V_R_matrix.tolist(),
        'text_reveals_curve': text_reveals_curve,
        'bootstrap_results': {str(k): v for k, v in bootstrap_results.items()},
        'summary': {
            'k_wins_layers': total_k_wins,
            'v_wins_layers': total_v_wins,
            'total_layers': total_layers,
            'sig_k_wins': sig_k_wins,
            'sig_v_wins': sig_v_wins,
            'mean_k_nums': float(mean_k),
            'mean_v_nums': float(mean_v),
            'mean_diff': float(mean_diff),
        },
        'elapsed_seconds': time.time() - t0,
    }

    results_path = RESULTS_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 094 COMPLETE")
    print("=" * 70)
    print(f"Model: {MODEL_NAME} (GQA, {kv_heads} KV heads, analog)")
    print(f"Accuracy: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")
    print(f"Extracted: {n_extracted} problems")
    print(f"\nK|nums mean: {mean_k:+.3f}")
    print(f"V|nums mean: {mean_v:+.3f}")
    print(f"K-V diff: {mean_diff:+.3f}")
    print(f"K>V layers: {total_k_wins}/{total_layers} ({100*total_k_wins/total_layers if total_layers else 0:.0f}%)")
    print(f"V>K layers: {total_v_wins}/{total_layers} ({100*total_v_wins/total_layers if total_layers else 0:.0f}%)")
    print(f"Significant (p<0.05): K>V at {sig_k_wins}, V>K at {sig_v_wins}")
    print(f"\nElapsed: {time.time()-t0:.0f}s")

    # Cross-model verdict
    print("\n" + "─" * 70)
    if total_k_wins > total_v_wins * 1.5:
        print("VERDICT: K>V on Mistral (GQA, analog) → K>V is GQA-GENERAL")
        print("GQA compression (shared KV heads) drives K information density, regardless of encoding.")
    elif total_v_wins > total_k_wins * 1.5:
        print("VERDICT: V>K on Mistral (GQA, analog) → K>V is Qwen-SPECIFIC")
        print("Both Mistral (GQA) and Phi (MHA) show V>K when analog. Digital encoding drives K>V, not GQA.")
    else:
        print("VERDICT: K≈V on Mistral (GQA, analog) → MIXED")
        print("GQA may contribute but is insufficient alone. K>V requires GQA + digital encoding (Qwen-specific combination).")
    print("─" * 70)


if __name__ == '__main__':
    main()
