#!/usr/bin/env python3
"""
Experiment 087: Position-Sweep Decodability with Numbers Control — Mistral-7B-v0.3

Boundary test: replicating the position-sweep decodability on the one model where
Phase 2 evidence previously failed (exp_081: V|nums not significant at "=" positions).

At each position bin, we train:
  - V → log(answer)          : raw V decodability
  - cumNums → log(answer)    : text-numbers baseline at that position
  - V → log(answer) | cumNums: genuine forward-looking V signal beyond numbers
  - K variants of the same
  - Shuffle control

If V|nums > 0 at early positions where text reveals 0%, the forward-looking channel
exists even on Mistral (and per-operation probing was underpowered). If V|nums ≈ 0,
Mistral is a genuine boundary case.
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
N_PROBLEMS = 350  # More problems because Mistral ~43% accuracy
PROBE_LAYERS = [16, 24]  # 50% and 75% of 32 layers
N_LAYERS_TOTAL = 32
N_BINS = 20
N_FOLDS = 5
MAX_NUMS_DIM = 30  # fixed feature dim for cumulative numbers
N_BOOTSTRAP = 300

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_087"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same as exp_082/083/084/086) ──
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
    """Check if answer appears as a standalone number in text."""
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
    """Extract all standalone numbers from text, in order of appearance."""
    nums = []
    for m in re.finditer(r'(?<![.\w])(\d+(?:\.\d+)?)(?![.\w])', text):
        try:
            val = float(m.group(1))
            nums.append(val)
        except:
            pass
    return nums


def numbers_to_features(nums, max_dim=MAX_NUMS_DIM):
    """Convert list of numbers to fixed-dim feature vector (log-transformed, padded)."""
    if not nums:
        return np.zeros(max_dim + 6, dtype=np.float32)  # +6 for summary stats
    log_nums = [np.log1p(abs(n)) * (1 if n >= 0 else -1) for n in nums]
    # Take last max_dim numbers (most recent are most relevant)
    if len(log_nums) > max_dim:
        truncated = log_nums[-max_dim:]
    else:
        truncated = log_nums
    features = np.zeros(max_dim, dtype=np.float32)
    features[:len(truncated)] = truncated
    # Add summary statistics
    arr = np.array(log_nums)
    stats = np.array([
        len(log_nums),       # count
        np.mean(arr),        # mean
        np.std(arr),         # std
        np.max(arr),         # max
        np.min(arr),         # min
        np.sum(arr),         # sum
    ], dtype=np.float32)
    return np.concatenate([features, stats])


NUMS_FEAT_DIM = MAX_NUMS_DIM + 6  # 36


def main():
    t0 = time.time()

    print("=" * 70)
    print("Experiment 087: Position-Sweep Decodability — Mistral-7B-v0.3")
    print("Boundary test: does position-sweep find signal where per-op failed?")
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
    print(f"Model: {MODEL_NAME}")

    n_layers = model.config.num_hidden_layers
    num_attn_heads = model.config.num_attention_heads
    kv_heads = getattr(model.config, 'num_key_value_heads', num_attn_heads)
    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // num_attn_heads)
    kv_dim = kv_heads * head_dim
    arch_type = 'MHA' if kv_heads == num_attn_heads else 'GQA'
    print(f"Layers: {n_layers}, KV heads: {kv_heads}, head_dim: {head_dim}, "
          f"KV dim: {kv_dim}, arch: {arch_type}")

    # Validate probe layers
    valid_probe_layers = [l for l in PROBE_LAYERS if l < n_layers]
    if valid_probe_layers != PROBE_LAYERS:
        print(f"  WARNING: Adjusted probe layers from {PROBE_LAYERS} to {valid_probe_layers}")
    PROBE_LAYERS_ACTUAL = valid_probe_layers

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
        if time.time() - t0 > TIME_BUDGET * 0.30:
            print(f"  Time budget (30%) reached at problem {i}")
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
            'idx': i,
            'question': question,
            'gold': gold,
            'gen_text': gen_text,
            'gen_ids': gen_ids,
            'pred': pred,
            'correct': correct,
        })

        if (i + 1) % 50 == 0:
            n_corr = sum(g['correct'] for g in generations)
            elapsed = time.time() - t0
            print(f"  Generated {i+1} problems, {n_corr}/{len(generations)} correct "
                  f"({100*n_corr/len(generations):.1f}%) [{elapsed:.0f}s]")

    n_total = len(generations)
    n_correct = sum(g['correct'] for g in generations)
    accuracy = 100 * n_correct / max(n_total, 1)
    print(f"\nPhase 1 complete: {n_total} generated, {n_correct} correct ({accuracy:.1f}%)")

    # Filter to correct problems with valid #### marker
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

    if len(correct_gens) < 30:
        print("ERROR: Too few correct problems for meaningful probing. Aborting.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Forward pass + KV extraction + cumulative numbers
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: Extracting KV cache + cumulative numbers at all CoT positions")
    print("=" * 70)

    data = {layer: {
        'V': [], 'K': [], 'cumNums': [],
        'rel_pos': [], 'text_reveals': [],
        'final_answer': [], 'problem_idx': [],
    } for layer in PROBE_LAYERS_ACTUAL}

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

        # Forward pass
        with torch.no_grad():
            outputs = model(full_ids, use_cache=True)
        kv_cache = outputs.past_key_values

        cot_ids = gen_token_ids[:hash_pos_gen]
        final_answer_str = str(gen['gold']).strip()
        final_answer_log = log_transform(float(gen['gold']))
        question_text = gen['question']

        # Build cumulative text and extract numbers + answer check at each position
        cot_token_texts = [tokenizer.decode([tid], skip_special_tokens=False) for tid in cot_ids]

        cum_cot_text = ""
        first_reveal_rel = 1.0
        token_cumNums = []
        token_reveals = []

        # Question numbers are always available
        question_nums = extract_numbers_from_text(question_text)

        for j, tok_text in enumerate(cot_token_texts):
            cum_cot_text += tok_text
            # Cumulative numbers: question nums + all nums in CoT so far
            cot_nums = extract_numbers_from_text(cum_cot_text)
            all_nums = question_nums + cot_nums
            token_cumNums.append(numbers_to_features(all_nums))

            revealed = answer_in_text(cum_cot_text, final_answer_str)
            token_reveals.append(revealed)
            if revealed and first_reveal_rel == 1.0:
                first_reveal_rel = j / cot_length

        first_reveal_positions.append(first_reveal_rel)

        # Extract KV at each CoT position
        for layer in PROBE_LAYERS_ACTUAL:
            K_layer, V_layer = get_kv(kv_cache, layer)

            for j in range(cot_length):
                abs_pos = prompt_len + j
                rel_pos = j / cot_length

                v_vec = V_layer[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                k_vec = K_layer[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()

                data[layer]['V'].append(v_vec)
                data[layer]['K'].append(k_vec)
                data[layer]['cumNums'].append(token_cumNums[j])
                data[layer]['rel_pos'].append(rel_pos)
                data[layer]['text_reveals'].append(token_reveals[j])
                data[layer]['final_answer'].append(final_answer_log)
                data[layer]['problem_idx'].append(pi)

        n_extracted += 1

        del outputs, kv_cache
        torch.cuda.empty_cache()

        if (pi + 1) % 25 == 0:
            elapsed = time.time() - t0
            n_vecs = len(data[PROBE_LAYERS_ACTUAL[0]]['V'])
            print(f"  Extracted {pi+1}/{len(correct_gens)} problems, "
                  f"{n_vecs} total vectors [{elapsed:.0f}s]")

    print(f"\nPhase 2 complete: {n_extracted} problems extracted")
    print(f"Total vectors per layer: {len(data[PROBE_LAYERS_ACTUAL[0]]['V'])}")

    # Convert to numpy
    for layer in PROBE_LAYERS_ACTUAL:
        for key in data[layer]:
            data[layer][key] = np.array(data[layer][key])

    # Check KV dim
    actual_kv_dim = data[PROBE_LAYERS_ACTUAL[0]]['V'].shape[1]
    print(f"Actual KV dim: {actual_kv_dim}")

    # Free model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded.")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Bin by position, train probes with nums control
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: Position-sweep probing with numbers control")
    print("=" * 70)

    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import RidgeCV
    from scipy.stats import pearsonr

    RIDGE_ALPHAS = np.logspace(-2, 6, 50)
    bin_edges = np.linspace(0, 1, N_BINS + 1)

    # Text-reveals curve (per-problem based)
    first_reveal_arr = np.array(first_reveal_positions[:n_extracted])
    text_reveals_curve = []
    for b in range(N_BINS):
        bin_upper = bin_edges[b + 1]
        frac = float(np.mean(first_reveal_arr <= bin_upper))
        text_reveals_curve.append(frac)

    results = {}

    for layer in PROBE_LAYERS_ACTUAL:
        print(f"\n--- Layer {layer} ({int(100*layer/n_layers)}% depth) ---")
        d = data[layer]
        V = d['V']
        K = d['K']
        cumNums = d['cumNums']
        y = d['final_answer']
        groups = d['problem_idx']
        rel_pos = d['rel_pos']

        layer_res = {
            'V_R': [], 'K_R': [], 'nums_R': [],
            'V_nums_R': [], 'K_nums_R': [],
            'shuffle_R': [],
            'n_samples': [], 'n_groups': [],
        }

        for b in range(N_BINS):
            if time.time() - t0 > TIME_BUDGET * 0.80:
                print(f"  Time budget (80%) reached at bin {b}")
                for key in ['V_R', 'K_R', 'nums_R', 'V_nums_R', 'K_nums_R', 'shuffle_R']:
                    layer_res[key].append(float('nan'))
                layer_res['n_samples'].append(0)
                layer_res['n_groups'].append(0)
                continue

            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b < N_BINS - 1:
                mask = (rel_pos >= lo) & (rel_pos < hi)
            else:
                mask = (rel_pos >= lo) & (rel_pos <= hi)

            V_bin = V[mask]
            K_bin = K[mask]
            nums_bin = cumNums[mask]
            y_bin = y[mask]
            g_bin = groups[mask]

            n_samples = len(y_bin)
            n_groups = len(np.unique(g_bin))
            layer_res['n_samples'].append(n_samples)
            layer_res['n_groups'].append(n_groups)

            if n_groups < N_FOLDS + 1 or n_samples < 30:
                for key in ['V_R', 'K_R', 'nums_R', 'V_nums_R', 'K_nums_R', 'shuffle_R']:
                    layer_res[key].append(float('nan'))
                print(f"  Bin {b:2d} [{lo:.2f}-{hi:.2f}]: SKIPPED (ng={n_groups}, n={n_samples})")
                continue

            gkf = GroupKFold(n_splits=min(N_FOLDS, n_groups))

            def probe_r(X, y_t, g_t):
                """Train ridge with GroupKFold, return Pearson R."""
                preds = np.zeros_like(y_t)
                for tr, te in gkf.split(X, y_t, g_t):
                    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                    ridge.fit(X[tr], y_t[tr])
                    preds[te] = ridge.predict(X[te])
                r, _ = pearsonr(preds, y_t)
                return float(r)

            def partial_r(X, y_t, Z, g_t):
                """Partial R: R(X→y | Z). Residualize y on Z, then probe X→y_resid."""
                # Step 1: get cross-validated residuals of y after removing Z
                y_resid = np.zeros_like(y_t)
                for tr, te in gkf.split(Z, y_t, g_t):
                    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                    ridge.fit(Z[tr], y_t[tr])
                    y_resid[te] = y_t[te] - ridge.predict(Z[te])
                if np.std(y_resid) < 1e-10:
                    return 0.0
                # Step 2: probe X → y_resid
                preds = np.zeros_like(y_resid)
                for tr, te in gkf.split(X, y_resid, g_t):
                    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                    ridge.fit(X[tr], y_resid[tr])
                    preds[te] = ridge.predict(X[te])
                r, _ = pearsonr(preds, y_resid)
                return float(r)

            # V → answer
            v_r = probe_r(V_bin, y_bin, g_bin)
            layer_res['V_R'].append(v_r)

            # K → answer
            k_r = probe_r(K_bin, y_bin, g_bin)
            layer_res['K_R'].append(k_r)

            # cumNums → answer
            nums_r = probe_r(nums_bin, y_bin, g_bin)
            layer_res['nums_R'].append(nums_r)

            # V → answer | cumNums
            v_nums_r = partial_r(V_bin, y_bin, nums_bin, g_bin)
            layer_res['V_nums_R'].append(v_nums_r)

            # K → answer | cumNums
            k_nums_r = partial_r(K_bin, y_bin, nums_bin, g_bin)
            layer_res['K_nums_R'].append(k_nums_r)

            # Shuffle control: permute answer labels across problems
            unique_g = np.unique(g_bin)
            answer_per_group = {g: y_bin[g_bin == g][0] for g in unique_g}
            g_list = list(answer_per_group.keys())
            a_list = list(answer_per_group.values())
            rng = np.random.RandomState(SEED + b)
            rng.shuffle(a_list)
            y_shuf = y_bin.copy()
            for g, a in zip(g_list, a_list):
                y_shuf[g_bin == g] = a
            shuf_r = probe_r(V_bin, y_shuf, g_bin)
            layer_res['shuffle_R'].append(shuf_r)

            print(f"  Bin {b:2d} [{lo:.2f}-{hi:.2f}]: V={v_r:.3f} nums={nums_r:.3f} "
                  f"V|nums={v_nums_r:.3f} K={k_r:.3f} K|nums={k_nums_r:.3f} "
                  f"shuf={shuf_r:.3f} n={n_samples} ng={n_groups}")

        results[layer] = layer_res

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Bootstrap significance for peak V|nums
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: Bootstrap significance")
    print("=" * 70)

    bootstrap_results = {}
    for layer in PROBE_LAYERS_ACTUAL:
        if time.time() - t0 > TIME_BUDGET * 0.88:
            print("  Time budget (88%) reached, skipping bootstrap")
            break

        d = data[layer]
        V = d['V']
        cumNums = d['cumNums']
        y = d['final_answer']
        groups = d['problem_idx']
        rel_pos = d['rel_pos']

        # Find peak bin for V|nums
        v_nums_rs = results[layer]['V_nums_R']
        valid = [(i, r) for i, r in enumerate(v_nums_rs) if not np.isnan(r)]
        if not valid:
            continue
        peak_bin, peak_r = max(valid, key=lambda x: x[1])

        lo, hi = bin_edges[peak_bin], bin_edges[peak_bin + 1]
        if peak_bin == N_BINS - 1:
            mask = (rel_pos >= lo) & (rel_pos <= hi)
        else:
            mask = (rel_pos >= lo) & (rel_pos < hi)

        V_bin = V[mask]
        nums_bin = cumNums[mask]
        y_bin = y[mask]
        g_bin = groups[mask]
        unique_g = np.unique(g_bin)
        ng = len(unique_g)

        boot_rs = []
        for bi in range(N_BOOTSTRAP):
            rng = np.random.RandomState(SEED + 10000 + bi)
            boot_g = rng.choice(unique_g, size=ng, replace=True)
            # Build bootstrap sample
            indices = []
            for g in boot_g:
                indices.extend(np.where(g_bin == g)[0].tolist())
            indices = np.array(indices)

            V_b = V_bin[indices]
            nums_b = nums_bin[indices]
            y_b = y_bin[indices]
            g_b = g_bin[indices]

            n_ug = len(np.unique(g_b))
            if n_ug < N_FOLDS:
                continue

            gkf = GroupKFold(n_splits=min(N_FOLDS, n_ug))

            # Residualize y on nums
            y_resid = np.zeros_like(y_b)
            for tr, te in gkf.split(nums_b, y_b, g_b):
                ridge = RidgeCV(alphas=np.logspace(-2, 6, 50))
                ridge.fit(nums_b[tr], y_b[tr])
                y_resid[te] = y_b[te] - ridge.predict(nums_b[te])

            if np.std(y_resid) < 1e-10:
                continue

            # Probe V → y_resid
            preds = np.zeros_like(y_resid)
            for tr, te in gkf.split(V_b, y_resid, g_b):
                ridge = RidgeCV(alphas=np.logspace(-2, 6, 50))
                ridge.fit(V_b[tr], y_resid[tr])
                preds[te] = ridge.predict(V_b[te])

            from scipy.stats import pearsonr as pr
            r, _ = pr(preds, y_resid)
            boot_rs.append(float(r))

        boot_rs = np.array(boot_rs)
        p_null = float(np.mean(boot_rs <= 0))
        ci_lo = float(np.percentile(boot_rs, 2.5))
        ci_hi = float(np.percentile(boot_rs, 97.5))

        bootstrap_results[layer] = {
            'peak_bin': int(peak_bin),
            'peak_R': float(peak_r),
            'boot_mean': float(np.mean(boot_rs)),
            'boot_std': float(np.std(boot_rs)),
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'p_null': p_null,
            'n_boot': len(boot_rs),
            'boot_rs': boot_rs.tolist(),
        }

        print(f"  L{layer}: peak V|nums R={peak_r:.3f} at bin {peak_bin}, "
              f"boot mean={np.mean(boot_rs):.3f}, 95%CI=[{ci_lo:.3f}, {ci_hi:.3f}], "
              f"p={p_null:.4f}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: Figures
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 5: Generating figures")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bin_centers = [(bin_edges[b] + bin_edges[b + 1]) / 2 * 100 for b in range(N_BINS)]

    # ── Figure 1: Main decodability curve with nums control (first probe layer) ──
    primary_layer = PROBE_LAYERS_ACTUAL[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    lr = results[primary_layer]

    ax.plot(bin_centers, lr['V_R'], 'b-o', label=f'V → answer (L{primary_layer})',
            markersize=4, linewidth=2)
    ax.plot(bin_centers, lr['nums_R'], color='orange', marker='s', linestyle='-',
            label=f'cumNums → answer', markersize=4, linewidth=2)
    ax.plot(bin_centers, lr['V_nums_R'], 'g-^',
            label=f'V → answer | nums (residualized)', markersize=4, linewidth=2)
    ax.plot(bin_centers, lr['shuffle_R'], color='gray', linestyle='--', alpha=0.5,
            label='Shuffle control', linewidth=1)

    ax2 = ax.twinx()
    ax2.plot(bin_centers, [f * 100 for f in text_reveals_curve], 'r--',
             label='Text reveals answer (%)', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Problems where text reveals answer (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-5, 105)

    ax.set_xlabel('Position in Chain (%)')
    ax.set_ylabel('Pearson R (probe → log answer)')
    depth_pct = int(100 * primary_layer / n_layers)
    ax.set_title(f'Position-Sweep Decodability with Numbers Control\n'
                 f'{MODEL_NAME} ({arch_type}), L{primary_layer} ({depth_pct}%), n={n_extracted} problems')
    ax.legend(loc='upper left', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'decodability_nums_control_L{primary_layer}.png', dpi=150)
    plt.close()
    print(f"  Saved decodability_nums_control_L{primary_layer}.png")

    # ── Figure 2: Both layers comparison ──
    if len(PROBE_LAYERS_ACTUAL) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax_idx, layer in enumerate(PROBE_LAYERS_ACTUAL[:2]):
            ax = axes[ax_idx]
            lr = results[layer]

            ax.plot(bin_centers, lr['V_R'], 'b-o', label='V→answer', markersize=3, linewidth=1.5)
            ax.plot(bin_centers, lr['nums_R'], color='orange', marker='s', linestyle='-',
                    label='cumNums→answer', markersize=3, linewidth=1.5)
            ax.plot(bin_centers, lr['V_nums_R'], 'g-^', label='V|nums', markersize=3, linewidth=1.5)
            ax.plot(bin_centers, lr['K_R'], 'c-d', label='K→answer', markersize=3, linewidth=1, alpha=0.6)
            ax.plot(bin_centers, lr['K_nums_R'], 'm-v', label='K|nums', markersize=3, linewidth=1, alpha=0.6)
            ax.plot(bin_centers, lr['shuffle_R'], color='gray', linestyle='--', alpha=0.4, label='Shuffle')

            ax2 = ax.twinx()
            ax2.fill_between(bin_centers, 0, [f * 100 for f in text_reveals_curve],
                             alpha=0.1, color='red')
            ax2.set_ylim(-5, 105)
            if ax_idx == 1:
                ax2.set_ylabel('Text reveals (%)', color='red')

            ax.set_xlabel('Position in Chain (%)')
            ax.set_ylabel('Pearson R')
            depth_pct = int(100 * layer / n_layers)
            ax.set_title(f'Layer {layer} ({depth_pct}% depth)')
            ax.legend(loc='upper left', fontsize=7)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.set_xlim(0, 100)

        fig.suptitle(f'Position-Sweep with Numbers Control — {MODEL_NAME} ({arch_type})', fontsize=13)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / 'decodability_both_layers.png', dpi=150)
        plt.close()
        print("  Saved decodability_both_layers.png")

    # ── Figure 3: V|nums gap (genuine forward-looking signal) ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for layer in PROBE_LAYERS_ACTUAL:
        lr = results[layer]
        gap = [v - n if not (np.isnan(v) or np.isnan(n)) else np.nan
               for v, n in zip(lr['V_R'], lr['nums_R'])]
        style = '-o' if layer == PROBE_LAYERS_ACTUAL[0] else '-s'
        ax.plot(bin_centers, gap, style,
                label=f'V_R - nums_R (L{layer})', markersize=4, linewidth=2)
        ax.plot(bin_centers, lr['V_nums_R'],
                '--^' if layer == PROBE_LAYERS_ACTUAL[0] else '--v',
                label=f'V|nums (residualized, L{layer})', markersize=3, linewidth=1.5, alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Position in Chain (%)')
    ax.set_ylabel('Gap (R)')
    ax.set_title(f'Genuine Forward-Looking Signal: V beyond Text Numbers\n{MODEL_NAME} ({arch_type})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'forward_looking_gap.png', dpi=150)
    plt.close()
    print("  Saved forward_looking_gap.png")

    # ── Figure 4: Bootstrap distribution for peak V|nums ──
    if bootstrap_results:
        fig, axes = plt.subplots(1, len(bootstrap_results), figsize=(6 * len(bootstrap_results), 5))
        if len(bootstrap_results) == 1:
            axes = [axes]
        for ax, (layer, br) in zip(axes, bootstrap_results.items()):
            boot_rs_arr = np.array(br['boot_rs'])
            ax.hist(boot_rs_arr, bins=40, alpha=0.7, color='steelblue', edgecolor='white')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Null (R=0)')
            ax.axvline(x=br['peak_R'], color='green', linestyle='-', linewidth=2,
                       label=f'Observed R={br["peak_R"]:.3f}')
            ax.axvspan(br['ci_lo'], br['ci_hi'], alpha=0.2, color='green')
            ax.set_xlabel('V|nums R (bootstrap)')
            ax.set_title(f'L{layer}: p={br["p_null"]:.4f}, 95%CI=[{br["ci_lo"]:.3f}, {br["ci_hi"]:.3f}]')
            ax.legend()
        fig.suptitle(f'Bootstrap Significance: Peak V|nums R — {MODEL_NAME}')
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / 'bootstrap_V_nums.png', dpi=150)
        plt.close()
        print("  Saved bootstrap_V_nums.png")

    # ── Figure 5: 3-model cross-comparison ──
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results"
    qwen_path = base_dir / "exp_084" / "results.json"
    phi_path = base_dir / "exp_086" / "results.json"

    if qwen_path.exists() and phi_path.exists():
        try:
            with open(qwen_path) as f:
                qwen_data = json.load(f)
            with open(phi_path) as f:
                phi_data = json.load(f)

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            # Panel 1: V|nums comparison (all 3 models)
            ax = axes[0]

            qwen_bins = qwen_data.get('bin_centers', bin_centers)
            qwen_l18 = qwen_data['results'].get('L18', {})
            qwen_vnums = qwen_l18.get('V_nums_R', [])
            if qwen_vnums:
                qwen_vnums_clean = [v if v is not None else float('nan') for v in qwen_vnums]
                ax.plot(qwen_bins, qwen_vnums_clean, 'b-o',
                        label='Qwen3-4B-Base L18', markersize=4, linewidth=2)

            phi_l16 = phi_data['results'].get('L16', {})
            phi_vnums = phi_l16.get('V_nums_R', [])
            if phi_vnums:
                phi_vnums_clean = [v if v is not None else float('nan') for v in phi_vnums]
                phi_bins = phi_data.get('bin_centers', bin_centers)
                ax.plot(phi_bins, phi_vnums_clean, 'r-s',
                        label='Phi-3.5-mini L16', markersize=4, linewidth=2)

            mistral_vnums = results[primary_layer]['V_nums_R']
            ax.plot(bin_centers, mistral_vnums, 'g-^',
                    label=f'Mistral-7B L{primary_layer}', markersize=4, linewidth=2)

            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.3)
            ax.set_xlabel('Position in Chain (%)')
            ax.set_ylabel('V|nums R (residualized)')
            ax.set_title('V|nums: Forward-Looking Signal')
            ax.legend(fontsize=9)
            ax.set_xlim(0, 100)

            # Panel 2: V_R comparison (all 3 models)
            ax = axes[1]

            qwen_vr = qwen_l18.get('V_R', [])
            if qwen_vr:
                qwen_vr_clean = [v if v is not None else float('nan') for v in qwen_vr]
                ax.plot(qwen_bins, qwen_vr_clean, 'b-o',
                        label='Qwen3-4B-Base L18', markersize=4, linewidth=2)

            phi_vr = phi_l16.get('V_R', [])
            if phi_vr:
                phi_vr_clean = [v if v is not None else float('nan') for v in phi_vr]
                ax.plot(phi_bins, phi_vr_clean, 'r-s',
                        label='Phi-3.5-mini L16', markersize=4, linewidth=2)

            mistral_vr = results[primary_layer]['V_R']
            ax.plot(bin_centers, mistral_vr, 'g-^',
                    label=f'Mistral-7B L{primary_layer}', markersize=4, linewidth=2)

            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.set_xlabel('Position in Chain (%)')
            ax.set_ylabel('V_R (raw decodability)')
            ax.set_title('Raw V Decodability')
            ax.legend(fontsize=9)
            ax.set_xlim(0, 100)

            # Panel 3: nums_R comparison (all 3 models)
            ax = axes[2]

            qwen_nums = qwen_l18.get('nums_R', [])
            if qwen_nums:
                qwen_nums_clean = [v if v is not None else float('nan') for v in qwen_nums]
                ax.plot(qwen_bins, qwen_nums_clean, 'b-o',
                        label='Qwen3-4B-Base L18', markersize=4, linewidth=2)

            phi_nums = phi_l16.get('nums_R', [])
            if phi_nums:
                phi_nums_clean = [v if v is not None else float('nan') for v in phi_nums]
                ax.plot(phi_bins, phi_nums_clean, 'r-s',
                        label='Phi-3.5-mini L16', markersize=4, linewidth=2)

            mistral_nums = results[primary_layer]['nums_R']
            ax.plot(bin_centers, mistral_nums, 'g-^',
                    label=f'Mistral-7B L{primary_layer}', markersize=4, linewidth=2)

            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.set_xlabel('Position in Chain (%)')
            ax.set_ylabel('nums_R (text numbers baseline)')
            ax.set_title('Text Numbers Baseline')
            ax.legend(fontsize=9)
            ax.set_xlim(0, 100)

            fig.suptitle('3-Model Comparison: Qwen vs Phi vs Mistral', fontsize=14)
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / 'three_model_comparison.png', dpi=150)
            plt.close()
            print("  Saved three_model_comparison.png")
        except Exception as e:
            print(f"  Could not generate 3-model comparison: {e}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 6: Save results
    # ══════════════════════════════════════════════════════════════
    save_data = {
        'config': {
            'model': MODEL_NAME,
            'n_problems': N_PROBLEMS,
            'n_generated': n_total,
            'n_correct': n_correct,
            'n_extracted': n_extracted,
            'accuracy': round(accuracy, 1),
            'probe_layers': PROBE_LAYERS_ACTUAL,
            'n_layers': n_layers,
            'kv_dim': actual_kv_dim,
            'kv_heads': kv_heads,
            'head_dim': head_dim,
            'arch_type': arch_type,
            'n_bins': N_BINS,
            'n_folds': N_FOLDS,
            'max_nums_dim': MAX_NUMS_DIM,
            'nums_feat_dim': NUMS_FEAT_DIM,
            'n_bootstrap': N_BOOTSTRAP,
            'seed': SEED,
        },
        'results': {},
        'text_reveals_curve': text_reveals_curve,
        'bootstrap': {},
    }

    for layer in PROBE_LAYERS_ACTUAL:
        lr = results[layer]
        save_data['results'][f'L{layer}'] = {
            k: [float(v) if not np.isnan(v) else None for v in vs]
            for k, vs in lr.items()
        }

    for layer, br in bootstrap_results.items():
        save_br = {k: v for k, v in br.items() if k != 'boot_rs'}
        save_data['bootstrap'][f'L{layer}'] = save_br

    save_data['bin_centers'] = [float(c) for c in bin_centers]

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    # ── Summary table ──
    print("\nSUMMARY TABLE")
    print("=" * 100)
    for layer in PROBE_LAYERS_ACTUAL:
        lr = results[layer]
        depth_pct = int(100 * layer / n_layers)
        print(f"\nLayer {layer} ({depth_pct}% depth):")
        print(f"  {'Bin%':>5s} {'V_R':>6s} {'nums_R':>7s} {'V|nums':>7s} "
              f"{'K_R':>6s} {'K|nums':>7s} {'shuf':>6s} {'text%':>6s} {'n':>5s} {'ng':>4s}")
        for b in range(N_BINS):
            pct = (bin_edges[b] + bin_edges[b + 1]) / 2 * 100
            v = lr['V_R'][b]
            nu = lr['nums_R'][b]
            vn = lr['V_nums_R'][b]
            k = lr['K_R'][b]
            kn = lr['K_nums_R'][b]
            sh = lr['shuffle_R'][b]
            txt = text_reveals_curve[b] * 100
            ns = lr['n_samples'][b]
            ng = lr['n_groups'][b]
            if np.isnan(v):
                print(f"  {pct:5.1f}% SKIPPED")
            else:
                print(f"  {pct:5.1f}% {v:6.3f} {nu:7.3f} {vn:7.3f} "
                      f"{k:6.3f} {kn:7.3f} {sh:6.3f} {txt:5.1f}% {ns:5d} {ng:4d}")

        if layer in bootstrap_results:
            br = bootstrap_results[layer]
            print(f"\n  Bootstrap peak V|nums: R={br['peak_R']:.3f}, "
                  f"95%CI=[{br['ci_lo']:.3f}, {br['ci_hi']:.3f}], p={br['p_null']:.4f}")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    for layer in PROBE_LAYERS_ACTUAL:
        lr = results[layer]
        vnums = lr['V_nums_R']
        valid_vnums = [(i, r) for i, r in enumerate(vnums) if not np.isnan(r)]
        if valid_vnums:
            positive_bins = sum(1 for _, r in valid_vnums if r > 0)
            peak_i, peak_r = max(valid_vnums, key=lambda x: x[1])
            early_bins = [(i, r) for i, r in valid_vnums if i < 6]  # 0-30%
            early_positive = sum(1 for _, r in early_bins if r > 0.05) if early_bins else 0
            early_mean = np.mean([r for _, r in early_bins]) if early_bins else 0

            # V_R at first bin
            first_vr = lr['V_R'][0] if not np.isnan(lr['V_R'][0]) else 0

            print(f"\nLayer {layer}:")
            print(f"  V_R at 2.5%: {first_vr:.3f}")
            print(f"  V|nums positive bins: {positive_bins}/{len(valid_vnums)}")
            print(f"  V|nums early (0-30%) positive (>0.05): {early_positive}/{len(early_bins)}")
            print(f"  V|nums early mean: {early_mean:.3f}")
            print(f"  V|nums peak: R={peak_r:.3f} at bin {peak_i} ({bin_centers[peak_i]:.0f}%)")

            if layer in bootstrap_results:
                br = bootstrap_results[layer]
                print(f"  Bootstrap: p={br['p_null']:.4f}, 95%CI=[{br['ci_lo']:.3f}, {br['ci_hi']:.3f}]")

    # Early Decodability Gap
    print("\nEarly Decodability Gap:")
    for layer in PROBE_LAYERS_ACTUAL:
        lr = results[layer]
        vrs = lr['V_R']
        valid_vrs = [(i, r) for i, r in enumerate(vrs) if not np.isnan(r)]
        if valid_vrs:
            peak_vr = max(r for _, r in valid_vrs)
            half_peak = peak_vr / 2
            v_half_pos = None
            for i, r in valid_vrs:
                if r >= half_peak:
                    v_half_pos = bin_centers[i]
                    break
            text_half_pos = None
            for b in range(N_BINS):
                if text_reveals_curve[b] >= 0.5:
                    text_half_pos = bin_centers[b]
                    break
            if text_half_pos is None:
                text_half_pos = 100.0
            gap = text_half_pos - (v_half_pos or 0)
            print(f"  L{layer}: V reaches 50% peak at {v_half_pos:.0f}%, "
                  f"text at {text_half_pos:.0f}%, gap = {gap:.0f}%")


if __name__ == "__main__":
    main()
