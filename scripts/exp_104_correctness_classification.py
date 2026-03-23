#!/usr/bin/env python3
"""
Experiment 104: Correctness Classification Probe from V-Cache

Core question: Can V-cache activations at intermediate CoT positions predict
whether the model will answer the problem correctly?

This tests whether V-cache encodes "computation quality" (the model's actual
computation state) beyond what the text reveals. Uses binary classification
(AUC-ROC) to avoid the regression collinearity confound from exp 103.

Analyses:
1. V-AUC at each layer × position bin (20 bins × 4 layers)
2. K-AUC for comparison (expect K > V on Qwen per exp 091)
3. Text-only baseline (numbers features → classifier)
4. Shuffle control (permuted labels → should be ~0.5)
5. V-AUC minus text-AUC gap (hidden correctness signal)
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

T0 = time.time()
TIME_BUDGET = 6600  # 110 min
MAX_GEN = 512
MAX_SEQ_LEN = 2048
MODEL_NAME = 'Qwen/Qwen3-4B-Base'
N_PROBLEMS = 600
PROBE_LAYERS = [9, 18, 27, 35]  # 25%, 50%, 75%, 97%
N_BINS = 20
N_FOLDS = 5
MAX_NUMS_DIM = 30
N_PERMUTATIONS = 200

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_104"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars ──
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


NUMS_FEAT_DIM = MAX_NUMS_DIM + 6


PCA_DIM = 64  # Reduce high-dim KV features to prevent overfitting


def cv_auc(X, y, n_folds=N_FOLDS, C=1.0, use_pca=True):
    """Stratified cross-validated Logistic Regression AUC-ROC.
    Uses balanced class weights to handle ~88/12% imbalance.
    PCA reduction applied when feature dim >> sample count."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    min_class = min(n_pos, n_neg)
    if min_class < n_folds:
        n_folds_actual = max(2, int(min_class))
    else:
        n_folds_actual = n_folds

    if min_class < 2:
        return 0.5

    skf = StratifiedKFold(n_splits=n_folds_actual, shuffle=True, random_state=SEED)
    all_proba = np.zeros(len(y))
    all_mask = np.zeros(len(y), dtype=bool)

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        # PCA for high-dim features (KV cache)
        if use_pca and X_train.shape[1] > PCA_DIM:
            n_components = min(PCA_DIM, X_train.shape[0] - 1, X_train.shape[1])
            pca = PCA(n_components=n_components, random_state=SEED)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        clf = LogisticRegression(
            C=C, class_weight='balanced', max_iter=1000,
            random_state=SEED, solver='lbfgs'
        )
        clf.fit(X_train, y[train_idx])
        all_proba[test_idx] = clf.predict_proba(X_test)[:, 1]
        all_mask[test_idx] = True

    if all_mask.sum() < len(y):
        all_proba[~all_mask] = 0.5

    try:
        return roc_auc_score(y, all_proba)
    except ValueError:
        return 0.5


def permutation_test_auc(X, y, n_perm=N_PERMUTATIONS, observed_auc=None):
    """Permutation test: p-value for observed AUC > chance (0.5)."""
    if observed_auc is None:
        observed_auc = cv_auc(X, y)

    null_aucs = []
    rng = np.random.RandomState(SEED)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        null_aucs.append(cv_auc(X, y_perm))

    null_aucs = np.array(null_aucs)
    p_value = np.mean(null_aucs >= observed_auc)
    return p_value, null_aucs


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Experiment 104: Correctness Classification Probe from V-Cache")
    print("=" * 70)

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        device_map='auto', trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Model loaded in {time.time()-T0:.0f}s")

    # ── Load GSM8K ──
    dataset = load_gsm8k()
    N = min(N_PROBLEMS, len(dataset))
    print(f"\nProcessing {N} GSM8K problems...")

    # ── Generate CoT and collect KV cache ──
    results = []
    kv_dim = None

    for idx in range(N):
        if time.time() - T0 > TIME_BUDGET * 0.45:
            print(f"\n  Time budget reached at problem {idx}/{N}")
            break

        item = dataset[idx]
        question = item['question']
        gold = extract_gold(item['answer'])
        if gold is None:
            continue

        prompt = build_prompt(question)
        enc = tokenizer(prompt, return_tensors='pt').to(model.device)
        prompt_len = enc.input_ids.shape[1]

        if prompt_len > MAX_SEQ_LEN:
            continue

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_GEN,
                do_sample=False, temperature=None, top_p=None,
                return_dict_in_generate=True, use_cache=True,
            )

        gen_ids = out.sequences[0, prompt_len:].cpu().tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_answer(gen_text)

        hash_pos = find_hash_pos_in_gen(gen_ids, tokenizer)
        if hash_pos is None or hash_pos < 5:
            continue

        # Determine correctness
        correct = False
        if pred is not None and gold is not None:
            try:
                correct = abs(float(pred) - float(gold)) < 0.01
            except:
                correct = pred.strip() == gold.strip()

        chain_len = hash_pos

        # Forward pass to get KV cache for full sequence
        full_ids = out.sequences[0, :prompt_len + chain_len].unsqueeze(0)
        with torch.no_grad():
            fwd = model(input_ids=full_ids, use_cache=True)
        cache = fwd.past_key_values

        # Extract KV features at each layer and bin
        problem_data = {
            'idx': idx,
            'correct': correct,
            'gold': gold,
            'pred': pred,
            'chain_len': chain_len,
            'gen_text': gen_text[:200],
        }

        # Store per-bin features
        for layer in PROBE_LAYERS:
            K, V = get_kv(cache, layer)
            # K, V shape: (1, n_heads, seq_len, head_dim)
            k_arr = K[0].float().cpu().numpy()  # (n_heads, seq_len, head_dim)
            v_arr = V[0].float().cpu().numpy()

            if kv_dim is None:
                kv_dim = k_arr.shape[0] * k_arr.shape[2]  # n_heads * head_dim
                print(f"  KV dim: {kv_dim} ({k_arr.shape[0]} heads × {k_arr.shape[2]} head_dim)")

            # For each position bin, take the mean KV vector
            for b in range(N_BINS):
                frac_lo = b / N_BINS
                frac_hi = (b + 1) / N_BINS
                pos_lo = prompt_len + int(frac_lo * chain_len)
                pos_hi = prompt_len + int(frac_hi * chain_len)
                if pos_hi <= pos_lo:
                    pos_hi = pos_lo + 1
                pos_hi = min(pos_hi, k_arr.shape[1])
                pos_lo = min(pos_lo, pos_hi - 1)

                k_slice = k_arr[:, pos_lo:pos_hi, :].mean(axis=1).flatten()
                v_slice = v_arr[:, pos_lo:pos_hi, :].mean(axis=1).flatten()

                problem_data[f'K_L{layer}_b{b}'] = k_slice
                problem_data[f'V_L{layer}_b{b}'] = v_slice

        # Extract numbers at each bin for text baseline
        gen_tokens = gen_ids[:chain_len]
        for b in range(N_BINS):
            frac_hi = (b + 1) / N_BINS
            tok_hi = int(frac_hi * chain_len)
            partial_text = tokenizer.decode(gen_tokens[:tok_hi], skip_special_tokens=True)
            nums = extract_numbers_from_text(partial_text)
            problem_data[f'nums_b{b}'] = numbers_to_features(nums)

        results.append(problem_data)

        del cache, fwd, out, full_ids
        torch.cuda.empty_cache()

        if (idx + 1) % 50 == 0:
            n_correct = sum(1 for r in results if r['correct'])
            n_total = len(results)
            print(f"  [{idx+1}/{N}] {n_total} valid, {n_correct} correct ({100*n_correct/max(n_total,1):.1f}%), elapsed {time.time()-T0:.0f}s")

    # ── Free model ──
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Summary ──
    n_total = len(results)
    n_correct = sum(1 for r in results if r['correct'])
    n_incorrect = n_total - n_correct
    print(f"\n{'='*70}")
    print(f"Total: {n_total} problems, {n_correct} correct ({100*n_correct/n_total:.1f}%), {n_incorrect} incorrect ({100*n_incorrect/n_total:.1f}%)")

    if n_incorrect < 5:
        print("ERROR: Too few incorrect problems for classification. Aborting.")
        sys.exit(1)

    # ── Build label array ──
    y = np.array([1 if r['correct'] else 0 for r in results])

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 1: V-cache AUC at each layer × bin
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 1: V-cache Classification AUC")
    print(f"{'='*70}")

    v_aucs = {}  # (layer, bin) -> auc
    k_aucs = {}
    text_aucs = {}

    for layer in PROBE_LAYERS:
        print(f"\n  Layer {layer}:")
        for b in range(N_BINS):
            # V-cache AUC
            X_v = np.array([r[f'V_L{layer}_b{b}'] for r in results])
            v_auc = cv_auc(X_v, y)
            v_aucs[(layer, b)] = v_auc

            # K-cache AUC
            X_k = np.array([r[f'K_L{layer}_b{b}'] for r in results])
            k_auc = cv_auc(X_k, y)
            k_aucs[(layer, b)] = k_auc

            # Text-only AUC (low-dim, no PCA needed)
            X_t = np.array([r[f'nums_b{b}'] for r in results])
            t_auc = cv_auc(X_t, y, use_pca=False)
            text_aucs[(layer, b)] = t_auc

            if b % 5 == 0 or b == N_BINS - 1:
                print(f"    Bin {b:2d}: V={v_auc:.3f}  K={k_auc:.3f}  Text={t_auc:.3f}  V-Text={v_auc-t_auc:+.3f}")

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 2: Permutation tests for significance
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 2: Permutation Tests (V-AUC > chance)")
    print(f"{'='*70}")

    # Test at a subset of bins to save time: bins 0, 5, 10, 15, 19
    test_bins = [0, 5, 10, 15, 19]
    perm_results = {}

    for layer in PROBE_LAYERS:
        print(f"\n  Layer {layer}:")
        for b in test_bins:
            if time.time() - T0 > TIME_BUDGET * 0.85:
                print("  Time budget reached, skipping remaining permutation tests")
                break

            X_v = np.array([r[f'V_L{layer}_b{b}'] for r in results])
            observed = v_aucs[(layer, b)]
            p_val, null_dist = permutation_test_auc(X_v, y, n_perm=N_PERMUTATIONS, observed_auc=observed)
            perm_results[(layer, b)] = {
                'observed': observed,
                'p': p_val,
                'null_mean': float(np.mean(null_dist)),
                'null_std': float(np.std(null_dist)),
            }
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"    Bin {b:2d}: V-AUC={observed:.3f}, p={p_val:.3f} {sig} (null={np.mean(null_dist):.3f}±{np.std(null_dist):.3f})")

        if time.time() - T0 > TIME_BUDGET * 0.85:
            break

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 3: Shuffle control
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 3: Shuffle Control (should be ~0.50)")
    print(f"{'='*70}")

    rng = np.random.RandomState(SEED + 100)
    y_shuffled = rng.permutation(y)

    shuffle_aucs = {}
    for layer in PROBE_LAYERS:
        for b in [5, 10, 15]:
            X_v = np.array([r[f'V_L{layer}_b{b}'] for r in results])
            s_auc = cv_auc(X_v, y_shuffled)
            shuffle_aucs[(layer, b)] = s_auc

    print(f"  Shuffle V-AUC (3 bins × 4 layers):")
    for layer in PROBE_LAYERS:
        vals = [shuffle_aucs[(layer, b)] for b in [5, 10, 15]]
        print(f"    L{layer}: {vals[0]:.3f}, {vals[1]:.3f}, {vals[2]:.3f}")

    mean_shuffle = np.mean(list(shuffle_aucs.values()))
    print(f"  Mean shuffle AUC: {mean_shuffle:.3f} (expect ~0.50)")

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 4: Summary statistics
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 4: Summary")
    print(f"{'='*70}")

    # For each layer, find earliest bin where V-AUC > 0.60
    for layer in PROBE_LAYERS:
        earliest_60 = None
        peak_auc = 0
        peak_bin = 0
        for b in range(N_BINS):
            auc_val = v_aucs[(layer, b)]
            if auc_val > peak_auc:
                peak_auc = auc_val
                peak_bin = b
            if earliest_60 is None and auc_val > 0.60:
                earliest_60 = b

        # V > text gap
        gaps = [v_aucs[(layer, b)] - text_aucs[(layer, b)] for b in range(N_BINS)]
        mean_gap_early = np.mean([v_aucs[(layer, b)] - text_aucs[(layer, b)] for b in range(10)])
        mean_gap_late = np.mean([v_aucs[(layer, b)] - text_aucs[(layer, b)] for b in range(10, 20)])

        # K > V comparison
        k_gt_v = sum(1 for b in range(N_BINS) if k_aucs[(layer, b)] > v_aucs[(layer, b)])

        print(f"\n  Layer {layer}:")
        print(f"    Peak V-AUC: {peak_auc:.3f} at bin {peak_bin}")
        print(f"    First V-AUC > 0.60: bin {earliest_60 if earliest_60 is not None else 'NEVER'}")
        print(f"    Mean V-text gap (early bins 0-9): {mean_gap_early:+.3f}")
        print(f"    Mean V-text gap (late bins 10-19): {mean_gap_late:+.3f}")
        print(f"    K > V at {k_gt_v}/{N_BINS} bins")

    # Cross-layer summary
    print(f"\n  Cross-layer summary:")
    for b in [0, 5, 10, 15, 19]:
        v_vals = [v_aucs[(layer, b)] for layer in PROBE_LAYERS]
        k_vals = [k_aucs[(layer, b)] for layer in PROBE_LAYERS]
        t_vals = [text_aucs[(layer, b)] for layer in PROBE_LAYERS]
        print(f"    Bin {b:2d}: V={np.mean(v_vals):.3f}  K={np.mean(k_vals):.3f}  Text={np.mean(t_vals):.3f}")

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 5: Correct vs incorrect chain length and difficulty
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 5: Potential confounds")
    print(f"{'='*70}")

    correct_lens = [r['chain_len'] for r in results if r['correct']]
    incorrect_lens = [r['chain_len'] for r in results if not r['correct']]
    print(f"  Chain length (correct):   mean={np.mean(correct_lens):.1f}, std={np.std(correct_lens):.1f}")
    print(f"  Chain length (incorrect): mean={np.mean(incorrect_lens):.1f}, std={np.std(incorrect_lens):.1f}")

    # ══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════

    output = {
        'n_total': n_total,
        'n_correct': n_correct,
        'n_incorrect': n_incorrect,
        'accuracy': n_correct / n_total,
        'v_aucs': {f'L{l}_b{b}': float(v_aucs[(l, b)]) for l, b in v_aucs},
        'k_aucs': {f'L{l}_b{b}': float(k_aucs[(l, b)]) for l, b in k_aucs},
        'text_aucs': {f'L{l}_b{b}': float(text_aucs[(l, b)]) for l, b in text_aucs},
        'shuffle_aucs': {f'L{l}_b{b}': float(shuffle_aucs[(l, b)]) for l, b in shuffle_aucs},
        'perm_results': {f'L{l}_b{b}': v for (l, b), v in perm_results.items()},
        'mean_shuffle_auc': float(mean_shuffle),
        'chain_len_correct_mean': float(np.mean(correct_lens)),
        'chain_len_incorrect_mean': float(np.mean(incorrect_lens)),
    }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: V-AUC position sweep (one line per layer)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bins = np.arange(N_BINS)
    bin_pcts = [(b + 0.5) / N_BINS * 100 for b in bins]

    for i, layer in enumerate(PROBE_LAYERS):
        ax = axes[i // 2, i % 2]
        v_vals = [v_aucs[(layer, b)] for b in bins]
        k_vals = [k_aucs[(layer, b)] for b in bins]
        t_vals = [text_aucs[(layer, b)] for b in bins]

        ax.plot(bin_pcts, v_vals, 'b-o', markersize=4, label='V-cache', linewidth=2)
        ax.plot(bin_pcts, k_vals, 'r-s', markersize=4, label='K-cache', linewidth=2)
        ax.plot(bin_pcts, t_vals, 'g--^', markersize=4, label='Text (nums)', linewidth=1.5)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Chance')
        ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.3)

        # Mark significant permutation tests
        for b in test_bins:
            if (layer, b) in perm_results and perm_results[(layer, b)]['p'] < 0.05:
                ax.plot(bin_pcts[b], v_vals[b], 'b*', markersize=12)

        ax.set_xlabel('Chain position (%)')
        ax.set_ylabel('AUC-ROC')
        ax.set_title(f'Layer {layer} ({100*layer//36}% depth)')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0.35, 0.85)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Correctness Classification: V-AUC vs K-AUC vs Text\n(stars = permutation p<0.05)', fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'auc_position_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: V-text gap heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    gap_matrix = np.zeros((len(PROBE_LAYERS), N_BINS))
    for i, layer in enumerate(PROBE_LAYERS):
        for b in range(N_BINS):
            gap_matrix[i, b] = v_aucs[(layer, b)] - text_aucs[(layer, b)]

    im = ax.imshow(gap_matrix, aspect='auto', cmap='RdBu_r', vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(0, N_BINS, 2))
    ax.set_xticklabels([f'{(b+0.5)/N_BINS*100:.0f}%' for b in range(0, N_BINS, 2)])
    ax.set_yticks(range(len(PROBE_LAYERS)))
    ax.set_yticklabels([f'L{l}' for l in PROBE_LAYERS])
    ax.set_xlabel('Chain position')
    ax.set_ylabel('Layer')
    ax.set_title('V-cache minus Text AUC gap\n(blue = V better, red = text better)')
    plt.colorbar(im, ax=ax, label='AUC gap')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'v_text_gap_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: K-V comparison bar chart at key bins
    fig, ax = plt.subplots(figsize=(10, 5))
    key_bins = [0, 5, 10, 15, 19]
    x = np.arange(len(key_bins))
    width = 0.15

    for i, layer in enumerate(PROBE_LAYERS):
        v_vals = [v_aucs[(layer, b)] for b in key_bins]
        k_vals = [k_aucs[(layer, b)] for b in key_bins]
        ax.bar(x + i * width * 2, v_vals, width, label=f'V L{layer}', color=f'C{i}', alpha=0.7)
        ax.bar(x + i * width * 2 + width, k_vals, width, label=f'K L{layer}', color=f'C{i}', alpha=0.4, hatch='//')

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    ax.set_xticks(x + width * 3)
    ax.set_xticklabels([f'Bin {b} ({(b+0.5)/N_BINS*100:.0f}%)' for b in key_bins])
    ax.set_ylabel('AUC-ROC')
    ax.set_title('V-cache vs K-cache Classification AUC at Key Positions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'kv_comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 4: Summary - mean AUC across layers
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_v = [np.mean([v_aucs[(l, b)] for l in PROBE_LAYERS]) for b in range(N_BINS)]
    mean_k = [np.mean([k_aucs[(l, b)] for l in PROBE_LAYERS]) for b in range(N_BINS)]
    mean_t = [np.mean([text_aucs[(l, b)] for l in PROBE_LAYERS]) for b in range(N_BINS)]

    ax.fill_between(bin_pcts, 0.5, mean_v, alpha=0.2, color='blue')
    ax.plot(bin_pcts, mean_v, 'b-o', markersize=5, label='V-cache (mean across layers)', linewidth=2)
    ax.plot(bin_pcts, mean_k, 'r-s', markersize=5, label='K-cache (mean across layers)', linewidth=2)
    ax.plot(bin_pcts, mean_t, 'g--^', markersize=5, label='Text (nums)', linewidth=1.5)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Chance')
    ax.set_xlabel('Chain position (%)')
    ax.set_ylabel('AUC-ROC (mean across layers)')
    ax.set_title('Correctness Classification: Mean AUC Position Sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.35, 0.85)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'mean_auc_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {RESULTS_DIR}")
    print(f"\nTotal time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f} min)")
    print("DONE")


if __name__ == '__main__':
    main()
