#!/usr/bin/env python3
"""
Experiment 105: Prompt-vs-Chain Correctness Discrimination Control

Core question: Does the V-cache correctness signal (from exp 104) come from
problem encoding in the prompt, or from computation state that builds during
chain generation?

Key comparison:
- V at last prompt token → correctness AUC (pure problem encoding)
- V at chain positions → correctness AUC (problem encoding + computation)

If prompt AUC ≈ chain AUC → signal is problem encoding (confound confirmed).
If prompt AUC < chain AUC → signal builds during computation (confound rejected).

Additionally tests:
- V at first chain token (single position, vs single prompt position)
- Concatenation of prompt + chain features (additive information test)
- K-cache comparison at all positions
- Text-only baseline at chain bins
- Permutation tests at key positions
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
PCA_DIM = 64

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_105"
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


def cv_auc(X, y, n_folds=N_FOLDS, C=1.0, use_pca=True):
    """Stratified cross-validated Logistic Regression AUC-ROC."""
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
    print("Experiment 105: Prompt-vs-Chain Correctness Discrimination Control")
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
        if time.time() - T0 > TIME_BUDGET * 0.42:
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

        # Forward pass to get KV cache for full sequence (prompt + chain)
        full_ids = out.sequences[0, :prompt_len + chain_len].unsqueeze(0)
        with torch.no_grad():
            fwd = model(input_ids=full_ids, use_cache=True)
        cache = fwd.past_key_values

        # Extract KV features
        problem_data = {
            'idx': idx,
            'correct': correct,
            'gold': gold,
            'pred': pred,
            'chain_len': chain_len,
            'prompt_len': prompt_len,
            'gen_text': gen_text[:200],
        }

        if kv_dim is None:
            K0, V0 = get_kv(cache, PROBE_LAYERS[0])
            kv_dim = K0.shape[1] * K0.shape[3]  # n_kv_heads * head_dim
            print(f"  KV dim: {kv_dim} ({K0.shape[1]} heads × {K0.shape[3]} head_dim)")
            del K0, V0

        for layer in PROBE_LAYERS:
            K, V = get_kv(cache, layer)
            # K, V shape: (1, n_kv_heads, seq_len, head_dim)
            k_arr = K[0].float().cpu().numpy()  # (n_heads, seq_len, head_dim)
            v_arr = V[0].float().cpu().numpy()

            # ── KEY EXTRACTION: Last prompt token (single position) ──
            prompt_pos = prompt_len - 1  # Last prompt token position
            k_prompt = k_arr[:, prompt_pos, :].flatten()
            v_prompt = v_arr[:, prompt_pos, :].flatten()
            problem_data[f'K_prompt_L{layer}'] = k_prompt
            problem_data[f'V_prompt_L{layer}'] = v_prompt

            # ── First chain token (single position) ──
            chain0_pos = prompt_len  # First generated token
            k_chain0 = k_arr[:, chain0_pos, :].flatten()
            v_chain0 = v_arr[:, chain0_pos, :].flatten()
            problem_data[f'K_chain0_L{layer}'] = k_chain0
            problem_data[f'V_chain0_L{layer}'] = v_chain0

            # ── Chain bins 0-19 (averaged, same as exp 104) ──
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

        # Extract text numbers at each bin
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

    mean_chain_correct = np.mean([r['chain_len'] for r in results if r['correct']])
    mean_chain_incorrect = np.mean([r['chain_len'] for r in results if not r['correct']])
    mean_prompt_len = np.mean([r['prompt_len'] for r in results])
    print(f"  Mean chain length: correct={mean_chain_correct:.1f}, incorrect={mean_chain_incorrect:.1f}")
    print(f"  Mean prompt length: {mean_prompt_len:.1f}")

    if n_incorrect < 5:
        print("ERROR: Too few incorrect problems for classification. Aborting.")
        sys.exit(1)

    # ── Build label array ──
    y = np.array([1 if r['correct'] else 0 for r in results])

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 1: CRITICAL COMPARISON — Prompt vs Chain AUC
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 1: PROMPT vs CHAIN Correctness Classification AUC")
    print("  (The critical confound test)")
    print(f"{'='*70}")

    # Store all AUCs for plotting
    all_aucs = {}  # key -> auc

    for layer in PROBE_LAYERS:
        print(f"\n  Layer {layer}:")

        # V at last prompt token
        X_v_prompt = np.array([r[f'V_prompt_L{layer}'] for r in results])
        v_auc_prompt = cv_auc(X_v_prompt, y)
        all_aucs[('V_prompt', layer)] = v_auc_prompt

        # K at last prompt token
        X_k_prompt = np.array([r[f'K_prompt_L{layer}'] for r in results])
        k_auc_prompt = cv_auc(X_k_prompt, y)
        all_aucs[('K_prompt', layer)] = k_auc_prompt

        # V at first chain token (single position)
        X_v_chain0 = np.array([r[f'V_chain0_L{layer}'] for r in results])
        v_auc_chain0_single = cv_auc(X_v_chain0, y)
        all_aucs[('V_chain0_single', layer)] = v_auc_chain0_single

        # K at first chain token
        X_k_chain0 = np.array([r[f'K_chain0_L{layer}'] for r in results])
        k_auc_chain0_single = cv_auc(X_k_chain0, y)
        all_aucs[('K_chain0_single', layer)] = k_auc_chain0_single

        # V at chain bins (averaged, replicating exp 104)
        v_auc_bin0 = None
        for b in range(N_BINS):
            X_v = np.array([r[f'V_L{layer}_b{b}'] for r in results])
            v_auc = cv_auc(X_v, y)
            all_aucs[('V_chain', layer, b)] = v_auc

            X_k = np.array([r[f'K_L{layer}_b{b}'] for r in results])
            k_auc = cv_auc(X_k, y)
            all_aucs[('K_chain', layer, b)] = k_auc

            # Text baseline
            X_t = np.array([r[f'nums_b{b}'] for r in results])
            t_auc = cv_auc(X_t, y, use_pca=False)
            all_aucs[('text', layer, b)] = t_auc

            if b == 0:
                v_auc_bin0 = v_auc

        # Print key comparison
        v_gap = v_auc_bin0 - v_auc_prompt
        print(f"    V_prompt AUC:       {v_auc_prompt:.3f}")
        print(f"    V_chain0 (single):  {v_auc_chain0_single:.3f}")
        print(f"    V_chain bin0 (avg): {v_auc_bin0:.3f}")
        print(f"    V gap (bin0-prompt): {v_gap:+.3f}")
        print(f"    K_prompt AUC:       {k_auc_prompt:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 2: Concatenation test — does chain add to prompt?
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 2: Concatenation — Does Chain Add Unique Info Beyond Prompt?")
    print(f"{'='*70}")

    concat_results = {}
    test_bins_concat = [0, 5, 10, 19]

    for layer in PROBE_LAYERS:
        print(f"\n  Layer {layer}:")
        X_v_prompt = np.array([r[f'V_prompt_L{layer}'] for r in results])

        for b in test_bins_concat:
            X_v_chain = np.array([r[f'V_L{layer}_b{b}'] for r in results])
            # Concatenate prompt + chain
            X_combined = np.concatenate([X_v_prompt, X_v_chain], axis=1)

            auc_prompt = all_aucs[('V_prompt', layer)]
            auc_chain = all_aucs[('V_chain', layer, b)]
            auc_combined = cv_auc(X_combined, y)
            all_aucs[('V_combined', layer, b)] = auc_combined

            # Unique chain info = combined - prompt
            unique_chain = auc_combined - auc_prompt
            concat_results[(layer, b)] = {
                'prompt': auc_prompt,
                'chain': auc_chain,
                'combined': auc_combined,
                'unique_chain': unique_chain,
            }
            print(f"    Bin {b:2d}: prompt={auc_prompt:.3f}  chain={auc_chain:.3f}  "
                  f"combined={auc_combined:.3f}  unique_chain={unique_chain:+.3f}")

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 3: Permutation tests at key positions
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 3: Permutation Tests at Key Positions")
    print(f"{'='*70}")

    perm_results = {}
    # Test prompt, chain0_single, bin0, bin10, bin19
    test_positions = ['prompt', 'chain0_single', 'bin0', 'bin10', 'bin19']

    for layer in PROBE_LAYERS:
        print(f"\n  Layer {layer}:")
        for pos_name in test_positions:
            if time.time() - T0 > TIME_BUDGET * 0.88:
                print("  Time budget reached, skipping remaining permutation tests")
                break

            if pos_name == 'prompt':
                X = np.array([r[f'V_prompt_L{layer}'] for r in results])
                observed = all_aucs[('V_prompt', layer)]
            elif pos_name == 'chain0_single':
                X = np.array([r[f'V_chain0_L{layer}'] for r in results])
                observed = all_aucs[('V_chain0_single', layer)]
            else:
                b = int(pos_name.replace('bin', ''))
                X = np.array([r[f'V_L{layer}_b{b}'] for r in results])
                observed = all_aucs[('V_chain', layer, b)]

            p_val, null_dist = permutation_test_auc(X, y, n_perm=N_PERMUTATIONS, observed_auc=observed)
            perm_results[(layer, pos_name)] = {
                'observed': observed,
                'p': p_val,
                'null_mean': float(np.mean(null_dist)),
                'null_std': float(np.std(null_dist)),
            }
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"    {pos_name:15s}: AUC={observed:.3f}, p={p_val:.3f} {sig}")

        if time.time() - T0 > TIME_BUDGET * 0.88:
            break

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 4: Position-by-position AUC curve (full sweep)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 4: Full AUC Position Sweep (prompt → chain bins)")
    print(f"{'='*70}")

    for layer in PROBE_LAYERS:
        print(f"\n  Layer {layer}:")
        v_prompt = all_aucs[('V_prompt', layer)]
        print(f"    PROMPT:   V={v_prompt:.3f}")
        for b in range(N_BINS):
            v = all_aucs[('V_chain', layer, b)]
            k = all_aucs[('K_chain', layer, b)]
            t = all_aucs[('text', layer, b)]
            if b % 4 == 0 or b == N_BINS - 1:
                print(f"    Bin {b:2d}:   V={v:.3f}  K={k:.3f}  Text={t:.3f}  V-prompt={v-v_prompt:+.3f}")

    # ══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════

    # Convert all AUCs to serializable format
    save_aucs = {}
    for k, v in all_aucs.items():
        save_aucs[str(k)] = float(v)

    save_perm = {}
    for k, v in perm_results.items():
        save_perm[str(k)] = v

    save_concat = {}
    for k, v in concat_results.items():
        save_concat[str(k)] = v

    output = {
        'n_total': n_total,
        'n_correct': n_correct,
        'n_incorrect': n_incorrect,
        'accuracy': n_correct / n_total,
        'mean_chain_correct': float(mean_chain_correct),
        'mean_chain_incorrect': float(mean_chain_incorrect),
        'mean_prompt_len': float(mean_prompt_len),
        'aucs': save_aucs,
        'permutation_tests': save_perm,
        'concatenation_tests': save_concat,
        'probe_layers': PROBE_LAYERS,
        'n_bins': N_BINS,
        'pca_dim': PCA_DIM,
        'n_folds': N_FOLDS,
        'seed': SEED,
    }
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Generating figures...")
    print(f"{'='*70}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Figure 1: The critical comparison — prompt vs chain AUC sweep ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Exp 105: Prompt vs Chain Correctness AUC\n(Dashed = prompt AUC, solid = chain bins)',
                 fontsize=14, fontweight='bold')

    for i, layer in enumerate(PROBE_LAYERS):
        ax = axes[i // 2, i % 2]

        # Chain bins V AUC
        v_chain_aucs = [all_aucs.get(('V_chain', layer, b), 0.5) for b in range(N_BINS)]
        k_chain_aucs = [all_aucs.get(('K_chain', layer, b), 0.5) for b in range(N_BINS)]
        text_chain_aucs = [all_aucs.get(('text', layer, b), 0.5) for b in range(N_BINS)]

        bins_x = [(b + 0.5) / N_BINS * 100 for b in range(N_BINS)]

        ax.plot(bins_x, v_chain_aucs, 'b-o', markersize=3, label='V chain', linewidth=2)
        ax.plot(bins_x, k_chain_aucs, 'r-s', markersize=3, label='K chain', linewidth=1.5, alpha=0.7)
        ax.plot(bins_x, text_chain_aucs, 'g-^', markersize=3, label='Text', linewidth=1.5, alpha=0.7)

        # Prompt AUC as horizontal dashed line
        v_prompt = all_aucs.get(('V_prompt', layer), 0.5)
        k_prompt = all_aucs.get(('K_prompt', layer), 0.5)
        ax.axhline(y=v_prompt, color='b', linestyle='--', linewidth=2, alpha=0.8, label=f'V prompt ({v_prompt:.3f})')
        ax.axhline(y=k_prompt, color='r', linestyle='--', linewidth=1.5, alpha=0.5, label=f'K prompt ({k_prompt:.3f})')

        # Single chain0 token as star marker
        v_chain0 = all_aucs.get(('V_chain0_single', layer), 0.5)
        ax.plot(0, v_chain0, 'b*', markersize=15, zorder=5, label=f'V chain0 single ({v_chain0:.3f})')

        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance')
        ax.set_xlabel('Chain position (%)')
        ax.set_ylabel('AUC-ROC')
        ax.set_title(f'Layer {layer} ({100*layer/36:.0f}% depth)')
        ax.set_ylim(0.4, 0.85)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'prompt_vs_chain_auc_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: prompt_vs_chain_auc_sweep.png")

    # ── Figure 2: Bar chart — prompt vs chain at key bins ──
    fig, ax = plt.subplots(figsize=(12, 6))

    positions = ['Prompt\n(last token)', 'Chain0\n(1st gen)', 'Bin 0\n(0-5%)',
                 'Bin 5\n(25-30%)', 'Bin 10\n(50-55%)', 'Bin 19\n(95-100%)']
    x = np.arange(len(positions))
    width = 0.18

    for i, layer in enumerate(PROBE_LAYERS):
        aucs = [
            all_aucs.get(('V_prompt', layer), 0.5),
            all_aucs.get(('V_chain0_single', layer), 0.5),
            all_aucs.get(('V_chain', layer, 0), 0.5),
            all_aucs.get(('V_chain', layer, 5), 0.5),
            all_aucs.get(('V_chain', layer, 10), 0.5),
            all_aucs.get(('V_chain', layer, 19), 0.5),
        ]
        ax.bar(x + i * width, aucs, width, label=f'L{layer}', alpha=0.8)

    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('V-cache AUC-ROC', fontsize=12)
    ax.set_title('Correctness AUC: Prompt vs Chain Positions\n(Does the signal come from problem encoding or computation?)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(positions)
    ax.legend(title='Layer')
    ax.set_ylim(0.4, 0.85)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'prompt_vs_chain_bars.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: prompt_vs_chain_bars.png")

    # ── Figure 3: Concatenation analysis ──
    fig, axes = plt.subplots(1, len(PROBE_LAYERS), figsize=(16, 5))
    fig.suptitle('Concatenation Test: Does Chain Add to Prompt?\n(Combined > Prompt = chain adds unique info)',
                 fontsize=13, fontweight='bold')

    for i, layer in enumerate(PROBE_LAYERS):
        ax = axes[i]
        bins_tested = [0, 5, 10, 19]
        prompts = [concat_results.get((layer, b), {}).get('prompt', 0.5) for b in bins_tested]
        chains = [concat_results.get((layer, b), {}).get('chain', 0.5) for b in bins_tested]
        combineds = [concat_results.get((layer, b), {}).get('combined', 0.5) for b in bins_tested]

        x_pos = np.arange(len(bins_tested))
        w = 0.25
        ax.bar(x_pos - w, prompts, w, label='Prompt only', color='steelblue', alpha=0.8)
        ax.bar(x_pos, chains, w, label='Chain only', color='coral', alpha=0.8)
        ax.bar(x_pos + w, combineds, w, label='Combined', color='forestgreen', alpha=0.8)

        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Bin {b}' for b in bins_tested])
        ax.set_title(f'Layer {layer}')
        ax.set_ylabel('AUC' if i == 0 else '')
        ax.set_ylim(0.4, 0.85)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'concatenation_test.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: concatenation_test.png")

    # ── Figure 4: Gap analysis — chain minus prompt ──
    fig, ax = plt.subplots(figsize=(10, 6))

    for layer in PROBE_LAYERS:
        v_prompt = all_aucs.get(('V_prompt', layer), 0.5)
        gaps = [all_aucs.get(('V_chain', layer, b), 0.5) - v_prompt for b in range(N_BINS)]
        bins_x = [(b + 0.5) / N_BINS * 100 for b in range(N_BINS)]
        ax.plot(bins_x, gaps, '-o', markersize=4, label=f'L{layer} (prompt={v_prompt:.3f})', linewidth=2)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Chain position (%)', fontsize=12)
    ax.set_ylabel('AUC gap: V_chain - V_prompt', fontsize=12)
    ax.set_title('Chain-Prompt AUC Gap\n(Positive = chain adds info beyond prompt encoding)',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'chain_prompt_gap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: chain_prompt_gap.png")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY: Prompt vs Chain Correctness AUC")
    print(f"{'='*70}")

    print(f"\n  {'Layer':<8} {'V_prompt':<12} {'V_chain0_s':<12} {'V_bin0':<12} {'V_bin10':<12} {'V_bin19':<12} {'Gap(b0-p)':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for layer in PROBE_LAYERS:
        vp = all_aucs.get(('V_prompt', layer), 0.5)
        vc0 = all_aucs.get(('V_chain0_single', layer), 0.5)
        vb0 = all_aucs.get(('V_chain', layer, 0), 0.5)
        vb10 = all_aucs.get(('V_chain', layer, 10), 0.5)
        vb19 = all_aucs.get(('V_chain', layer, 19), 0.5)
        gap = vb0 - vp
        print(f"  L{layer:<6} {vp:<12.3f} {vc0:<12.3f} {vb0:<12.3f} {vb10:<12.3f} {vb19:<12.3f} {gap:<+12.3f}")

    # Aggregate summary
    mean_prompt = np.mean([all_aucs.get(('V_prompt', l), 0.5) for l in PROBE_LAYERS])
    mean_bin0 = np.mean([all_aucs.get(('V_chain', l, 0), 0.5) for l in PROBE_LAYERS])
    mean_bin10 = np.mean([all_aucs.get(('V_chain', l, 10), 0.5) for l in PROBE_LAYERS])
    mean_gap = mean_bin0 - mean_prompt
    print(f"\n  Mean:   prompt={mean_prompt:.3f}  bin0={mean_bin0:.3f}  bin10={mean_bin10:.3f}  gap(b0-p)={mean_gap:+.3f}")

    # Interpret result
    print(f"\n  INTERPRETATION:")
    if mean_gap > 0.05:
        print(f"  Chain AUC exceeds prompt by {mean_gap:+.3f} → COMPUTATION STATE")
        print(f"  The correctness signal GROWS during generation. Problem encoding")
        print(f"  alone cannot explain exp 104. Confound REJECTED.")
    elif mean_gap > 0.02:
        print(f"  Chain AUC exceeds prompt by {mean_gap:+.3f} → PARTIAL computation state")
        print(f"  Some signal is from problem encoding, but chain adds meaningful info.")
    elif mean_gap > -0.02:
        print(f"  Chain AUC ≈ prompt ({mean_gap:+.3f}) → PROBLEM ENCODING confound")
        print(f"  The exp 104 signal is mostly/entirely from problem features.")
        print(f"  Exp 104's 'computation quality' interpretation is WEAKENED.")
    else:
        print(f"  Chain AUC < prompt ({mean_gap:+.3f}) → UNEXPECTED")
        print(f"  Prompt encodes correctness BETTER than chain. Unusual pattern.")

    # Permutation summary
    print(f"\n  PERMUTATION TESTS:")
    for layer in PROBE_LAYERS:
        for pos in ['prompt', 'chain0_single', 'bin0', 'bin10', 'bin19']:
            key = (layer, pos)
            if key in perm_results:
                pr = perm_results[key]
                sig = "***" if pr['p'] < 0.001 else "**" if pr['p'] < 0.01 else "*" if pr['p'] < 0.05 else "ns"
                print(f"    L{layer} {pos:15s}: AUC={pr['observed']:.3f}, p={pr['p']:.3f} {sig}")

    elapsed = time.time() - T0
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
