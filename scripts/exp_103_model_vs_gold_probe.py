#!/usr/bin/env python3
"""
Experiment 103: Model's Own Answer vs Ground Truth V-Probe

Core question: Does the V-cache encode the MODEL'S OWN predicted answer
(faithful computation channel) or just correlate with the ground truth
(general problem features)?

For incorrect problems, the model's predicted answer ≠ ground truth.
We train a probe on correct problems (where pred=gold), apply to incorrect,
and compare correlation with:
  (a) ground truth answer
  (b) model's predicted answer

If V_R(predicted) >> V_R(gold) for incorrect → V-cache encodes model's computation
If V_R(predicted) ≈ V_R(gold) → V-cache encodes general features
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

T0 = time.time()
TIME_BUDGET = 6600  # 110 min
MAX_GEN = 512
MAX_SEQ_LEN = 2048
MODEL_NAME = 'Qwen/Qwen3-4B-Base'
N_PROBLEMS = 800  # more problems to get more incorrect
PROBE_LAYERS = [9, 18, 27, 35]  # 25%, 50%, 75%, 97%
N_BINS = 20
N_FOLDS = 5
MAX_NUMS_DIM = 30
N_BOOTSTRAP = 500

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_103"
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


NUMS_FEAT_DIM = MAX_NUMS_DIM + 6


def cv_ridge_r(X, y, n_folds=N_FOLDS, alpha=1.0):
    """Cross-validated Ridge regression, returns mean Pearson R across folds."""
    if len(X) < n_folds * 2:
        return 0.0
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rs = []
    for train_idx, test_idx in kf.split(X):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = Ridge(alpha=alpha)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        if np.std(pred) < 1e-10 or np.std(y[test_idx]) < 1e-10:
            rs.append(0.0)
        else:
            r = np.corrcoef(pred, y[test_idx])[0, 1]
            rs.append(r if np.isfinite(r) else 0.0)
    return np.mean(rs) if rs else 0.0


def main():
    t0 = time.time()

    print("=" * 70)
    print("Experiment 103: Model's Own Answer vs Ground Truth V-Probe")
    print("Does V-cache encode model's computation or just problem features?")
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
    print(f"Model: {MODEL_NAME}")

    kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // model.config.num_attention_heads)
    kv_dim = kv_heads * head_dim
    print(f"KV heads: {kv_heads}, head_dim: {head_dim}, KV dim: {kv_dim}")

    ds = load_gsm8k()
    print(f"GSM8K test set: {len(ds)} problems")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Generate CoT traces — extract BOTH gold and model's predicted answer
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: Generating CoT traces")
    print("=" * 70)

    generations = []
    for i in range(min(N_PROBLEMS, len(ds))):
        if time.time() - t0 > TIME_BUDGET * 0.25:
            print(f"  Time budget (25%) reached at problem {i}")
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

        # Parse model's predicted answer as float
        pred_float = None
        if pred is not None:
            try:
                pred_float = float(pred)
            except (ValueError, TypeError):
                pass

        gold_float = None
        if gold is not None:
            try:
                gold_float = float(gold)
            except (ValueError, TypeError):
                pass

        generations.append({
            'idx': i,
            'question': question,
            'gold': gold,
            'gold_float': gold_float,
            'pred': pred,
            'pred_float': pred_float,
            'gen_text': gen_text,
            'gen_ids': gen_ids,
            'correct': correct,
        })

        if (i + 1) % 50 == 0:
            n_corr = sum(g['correct'] for g in generations)
            elapsed = time.time() - t0
            print(f"  Generated {i+1} problems, {n_corr}/{len(generations)} correct "
                  f"({100*n_corr/len(generations):.1f}%) [{elapsed:.0f}s]")

    n_total = len(generations)
    n_correct = sum(g['correct'] for g in generations)
    n_incorrect = n_total - n_correct
    print(f"\nPhase 1 complete: {n_total} generated, {n_correct} correct "
          f"({100*n_correct/n_total:.1f}%), {n_incorrect} incorrect")

    # Categorize incorrect problems
    incorrect_with_pred = [g for g in generations if not g['correct'] and g['pred_float'] is not None and g['gold_float'] is not None]
    incorrect_no_pred = [g for g in generations if not g['correct'] and (g['pred_float'] is None or g['gold_float'] is None)]
    print(f"Incorrect with parseable predicted answer: {len(incorrect_with_pred)}")
    print(f"Incorrect without parseable prediction: {len(incorrect_no_pred)}")

    # Show some example incorrect predictions
    print("\nSample incorrect predictions (first 10):")
    for g in incorrect_with_pred[:10]:
        print(f"  Problem {g['idx']}: gold={g['gold_float']}, predicted={g['pred_float']}, "
              f"ratio={g['pred_float']/g['gold_float']:.2f}" if g['gold_float'] != 0 else
              f"  Problem {g['idx']}: gold={g['gold_float']}, predicted={g['pred_float']}")

    # Find hash positions
    usable_correct = []
    usable_incorrect = []
    for g in generations:
        hash_pos = find_hash_pos_in_gen(g['gen_ids'], tokenizer)
        if g['correct']:
            if hash_pos is not None and hash_pos >= 10 and g['gold_float'] is not None:
                g['chain_len'] = hash_pos
                usable_correct.append(g)
        else:
            if g['pred_float'] is not None and g['gold_float'] is not None:
                chain_len = len(g['gen_ids'])
                if hash_pos is not None and hash_pos >= 10:
                    chain_len = hash_pos
                if chain_len >= 10:
                    g['chain_len'] = chain_len
                    usable_incorrect.append(g)

    print(f"\nUsable correct: {len(usable_correct)}, Usable incorrect: {len(usable_incorrect)}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Forward pass + KV extraction
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: Extracting V-cache at all CoT positions")
    print("=" * 70)

    all_problems_data = []
    n_extracted = 0
    all_gens = usable_correct + usable_incorrect
    np.random.shuffle(all_gens)

    for pi, gen in enumerate(all_gens):
        if time.time() - t0 > TIME_BUDGET * 0.60:
            print(f"\n  Time budget (60%) reached at problem {pi}")
            break

        chain_len = gen['chain_len']
        gen_ids = gen['gen_ids'][:chain_len]

        prompt = build_prompt(gen['question'])
        full_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
        prompt_len = full_ids.shape[1]

        full_input = torch.cat([
            full_ids,
            torch.tensor([gen_ids], dtype=torch.long)
        ], dim=1).to(model.device)

        if full_input.shape[1] > MAX_SEQ_LEN:
            continue

        with torch.no_grad():
            out = model(full_input, use_cache=True)

        cache = out.past_key_values

        bin_edges = np.linspace(0, chain_len, N_BINS + 1).astype(int)

        log_gold = log_transform(gen['gold_float'])
        log_pred = log_transform(gen['pred_float'])

        problem_data = {
            'idx': gen['idx'],
            'correct': gen['correct'],
            'gold_float': gen['gold_float'],
            'pred_float': gen['pred_float'],
            'log_gold': log_gold,
            'log_pred': log_pred,
            'bins': {},
        }

        for b in range(N_BINS):
            start = bin_edges[b]
            end = bin_edges[b + 1]
            if end <= start:
                continue

            center = (start + end) // 2
            abs_pos = prompt_len + center

            V_layers = {}
            for layer in PROBE_LAYERS:
                _, V = get_kv(cache, layer)
                v_vec = V[0, :, abs_pos, :].reshape(-1).float().cpu().numpy()
                V_layers[layer] = v_vec

            bin_tokens = gen_ids[:end]
            partial_text = tokenizer.decode(bin_tokens, skip_special_tokens=True)

            nums = extract_numbers_from_text(partial_text)
            nums_feat = numbers_to_features(nums)

            problem_data['bins'][b] = {
                'V': V_layers,
                'nums_feat': nums_feat,
            }

        all_problems_data.append(problem_data)
        n_extracted += 1

        del out, cache
        if n_extracted % 30 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if (n_extracted) % 50 == 0:
            n_c = sum(1 for p in all_problems_data if p['correct'])
            n_i = sum(1 for p in all_problems_data if not p['correct'])
            elapsed = time.time() - t0
            print(f"  Extracted {n_extracted} problems (correct={n_c}, incorrect={n_i}) [{elapsed:.0f}s]")

    # Free model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    correct_data = [p for p in all_problems_data if p['correct']]
    incorrect_data = [p for p in all_problems_data if not p['correct']]
    print(f"\nPhase 2 complete: {len(all_problems_data)} extracted "
          f"(correct={len(correct_data)}, incorrect={len(incorrect_data)})")

    if len(incorrect_data) < 5:
        print("ERROR: Too few incorrect problems for analysis. Aborting.")
        sys.exit(1)

    # Show answer distributions
    gold_incorrect = [p['gold_float'] for p in incorrect_data]
    pred_incorrect = [p['pred_float'] for p in incorrect_data]
    print(f"\nIncorrect problems answer statistics:")
    print(f"  Gold answers: mean={np.mean(gold_incorrect):.1f}, "
          f"std={np.std(gold_incorrect):.1f}, range=[{np.min(gold_incorrect):.0f}, {np.max(gold_incorrect):.0f}]")
    print(f"  Model preds:  mean={np.mean(pred_incorrect):.1f}, "
          f"std={np.std(pred_incorrect):.1f}, range=[{np.min(pred_incorrect):.0f}, {np.max(pred_incorrect):.0f}]")
    log_gold_inc = [p['log_gold'] for p in incorrect_data]
    log_pred_inc = [p['log_pred'] for p in incorrect_data]
    gold_pred_r = np.corrcoef(log_gold_inc, log_pred_inc)[0, 1] if len(log_gold_inc) > 2 else 0
    print(f"  Correlation(log_gold, log_pred) for incorrect: R={gold_pred_r:.3f}")
    print(f"  (If gold and pred are highly correlated, the experiment has less power to distinguish)")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Analysis — Gold vs Predicted answer probing
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: Analysis — V_R(gold) vs V_R(predicted) for incorrect problems")
    print("=" * 70)

    results = {
        'n_correct': len(correct_data),
        'n_incorrect': len(incorrect_data),
        'gold_pred_r_incorrect': float(gold_pred_r),
        'layers': {},
    }

    for layer in PROBE_LAYERS:
        print(f"\n{'='*50}")
        print(f"  Layer {layer}")
        print(f"{'='*50}")

        layer_results = {
            'V_R_gold_correct': [],
            'V_R_gold_incorrect': [],
            'V_R_pred_incorrect': [],
            'nums_R_gold_correct': [],
            'nums_R_gold_incorrect': [],
            'nums_R_pred_incorrect': [],
            'pct_closer_to_pred_V': [],
            'pct_closer_to_pred_nums': [],
            'mean_abs_error_V_gold_correct': [],
            'mean_abs_error_V_pred_incorrect': [],
            'mean_abs_error_V_gold_incorrect': [],
        }

        for b in range(N_BINS):
            # Gather data
            V_correct = []
            y_gold_correct = []
            nums_correct = []

            V_incorrect = []
            y_gold_incorrect = []
            y_pred_incorrect = []
            nums_incorrect = []

            for p in correct_data:
                if b not in p['bins']:
                    continue
                V_correct.append(p['bins'][b]['V'][layer])
                y_gold_correct.append(p['log_gold'])
                nums_correct.append(p['bins'][b]['nums_feat'])

            for p in incorrect_data:
                if b not in p['bins']:
                    continue
                V_incorrect.append(p['bins'][b]['V'][layer])
                y_gold_incorrect.append(p['log_gold'])
                y_pred_incorrect.append(p['log_pred'])
                nums_incorrect.append(p['bins'][b]['nums_feat'])

            V_c = np.array(V_correct) if V_correct else np.zeros((0, 1))
            y_gc = np.array(y_gold_correct)
            nums_c = np.array(nums_correct) if nums_correct else np.zeros((0, 1))

            V_i = np.array(V_incorrect) if V_incorrect else np.zeros((0, 1))
            y_gi = np.array(y_gold_incorrect)
            y_pi = np.array(y_pred_incorrect)
            nums_i = np.array(nums_incorrect) if nums_incorrect else np.zeros((0, 1))

            n_c = len(V_c)
            n_i = len(V_i)

            # --- Correct: V-probe (5-fold CV) ---
            V_R_gc = cv_ridge_r(V_c, y_gc) if n_c >= N_FOLDS * 2 else 0.0
            nums_R_gc = cv_ridge_r(nums_c, y_gc) if n_c >= N_FOLDS * 2 else 0.0

            # --- Incorrect: Train on correct, test on incorrect ---
            V_R_gi = 0.0
            V_R_pi = 0.0
            nums_R_gi = 0.0
            nums_R_pi = 0.0
            pct_closer_pred_V = 0.5
            pct_closer_pred_nums = 0.5
            mae_V_gold_c = 0.0
            mae_V_pred_i = 0.0
            mae_V_gold_i = 0.0

            if n_c >= N_FOLDS * 2 and n_i >= 3:
                # V-probe
                ridge_V = Ridge(alpha=1.0)
                ridge_V.fit(V_c, y_gc)
                v_pred_on_incorrect = ridge_V.predict(V_i)

                # Also get correct-set predictions for error comparison
                v_pred_on_correct = ridge_V.predict(V_c)  # not CV, but indicative

                # Correlation with gold
                if np.std(v_pred_on_incorrect) > 1e-10 and np.std(y_gi) > 1e-10:
                    V_R_gi = np.corrcoef(v_pred_on_incorrect, y_gi)[0, 1]
                    if not np.isfinite(V_R_gi):
                        V_R_gi = 0.0

                # Correlation with model's predicted answer
                if np.std(v_pred_on_incorrect) > 1e-10 and np.std(y_pi) > 1e-10:
                    V_R_pi = np.corrcoef(v_pred_on_incorrect, y_pi)[0, 1]
                    if not np.isfinite(V_R_pi):
                        V_R_pi = 0.0

                # Per-problem: closer to gold or predicted?
                err_gold = np.abs(v_pred_on_incorrect - y_gi)
                err_pred = np.abs(v_pred_on_incorrect - y_pi)
                closer_to_pred = (err_pred < err_gold).sum()
                ties = (np.abs(err_pred - err_gold) < 1e-10).sum()
                pct_closer_pred_V = (closer_to_pred + 0.5 * ties) / n_i

                # Mean absolute errors
                mae_V_gold_c = float(np.mean(np.abs(v_pred_on_correct - y_gc)))
                mae_V_pred_i = float(np.mean(err_pred))
                mae_V_gold_i = float(np.mean(err_gold))

                # Nums probe
                ridge_nums = Ridge(alpha=1.0)
                ridge_nums.fit(nums_c, y_gc)
                nums_pred_on_incorrect = ridge_nums.predict(nums_i)

                if np.std(nums_pred_on_incorrect) > 1e-10 and np.std(y_gi) > 1e-10:
                    nums_R_gi = np.corrcoef(nums_pred_on_incorrect, y_gi)[0, 1]
                    if not np.isfinite(nums_R_gi):
                        nums_R_gi = 0.0

                if np.std(nums_pred_on_incorrect) > 1e-10 and np.std(y_pi) > 1e-10:
                    nums_R_pi = np.corrcoef(nums_pred_on_incorrect, y_pi)[0, 1]
                    if not np.isfinite(nums_R_pi):
                        nums_R_pi = 0.0

                # Per-problem for nums
                err_gold_nums = np.abs(nums_pred_on_incorrect - y_gi)
                err_pred_nums = np.abs(nums_pred_on_incorrect - y_pi)
                closer_to_pred_nums = (err_pred_nums < err_gold_nums).sum()
                ties_nums = (np.abs(err_pred_nums - err_gold_nums) < 1e-10).sum()
                pct_closer_pred_nums = (closer_to_pred_nums + 0.5 * ties_nums) / n_i

            # Store
            layer_results['V_R_gold_correct'].append(float(V_R_gc))
            layer_results['V_R_gold_incorrect'].append(float(V_R_gi))
            layer_results['V_R_pred_incorrect'].append(float(V_R_pi))
            layer_results['nums_R_gold_correct'].append(float(nums_R_gc))
            layer_results['nums_R_gold_incorrect'].append(float(nums_R_gi))
            layer_results['nums_R_pred_incorrect'].append(float(nums_R_pi))
            layer_results['pct_closer_to_pred_V'].append(float(pct_closer_pred_V))
            layer_results['pct_closer_to_pred_nums'].append(float(pct_closer_pred_nums))
            layer_results['mean_abs_error_V_gold_correct'].append(float(mae_V_gold_c))
            layer_results['mean_abs_error_V_pred_incorrect'].append(float(mae_V_pred_i))
            layer_results['mean_abs_error_V_gold_incorrect'].append(float(mae_V_gold_i))

        # Summary for this layer
        mean_V_R_gi = np.mean(layer_results['V_R_gold_incorrect'])
        mean_V_R_pi = np.mean(layer_results['V_R_pred_incorrect'])
        mean_nums_R_gi = np.mean(layer_results['nums_R_gold_incorrect'])
        mean_nums_R_pi = np.mean(layer_results['nums_R_pred_incorrect'])
        mean_pct_V = np.mean(layer_results['pct_closer_to_pred_V'])
        mean_pct_nums = np.mean(layer_results['pct_closer_to_pred_nums'])

        print(f"\n  INCORRECT PROBLEMS — V_R comparison:")
        print(f"    V_R(gold) mean across bins:      {mean_V_R_gi:.4f}")
        print(f"    V_R(predicted) mean across bins:  {mean_V_R_pi:.4f}")
        print(f"    Δ(pred - gold):                   {mean_V_R_pi - mean_V_R_gi:+.4f}")
        print(f"    Bins where V_R(pred) > V_R(gold): {sum(1 for a, b in zip(layer_results['V_R_pred_incorrect'], layer_results['V_R_gold_incorrect']) if a > b)}/{N_BINS}")
        print(f"    Per-problem closer to pred:       {mean_pct_V:.1%}")
        print(f"\n  INCORRECT PROBLEMS — nums_R comparison:")
        print(f"    nums_R(gold) mean:                {mean_nums_R_gi:.4f}")
        print(f"    nums_R(predicted) mean:           {mean_nums_R_pi:.4f}")
        print(f"    Δ(pred - gold):                   {mean_nums_R_pi - mean_nums_R_gi:+.4f}")
        print(f"    Per-problem closer to pred:       {mean_pct_nums:.1%}")

        # Mean absolute errors
        mean_mae_gc = np.mean(layer_results['mean_abs_error_V_gold_correct'])
        mean_mae_pi = np.mean(layer_results['mean_abs_error_V_pred_incorrect'])
        mean_mae_gi = np.mean(layer_results['mean_abs_error_V_gold_incorrect'])
        print(f"\n  MEAN ABSOLUTE ERROR:")
        print(f"    Correct vs gold:     {mean_mae_gc:.4f}")
        print(f"    Incorrect vs pred:   {mean_mae_pi:.4f}")
        print(f"    Incorrect vs gold:   {mean_mae_gi:.4f}")

        results['layers'][str(layer)] = layer_results

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Bootstrap significance test
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: Bootstrap significance — V_R(pred) > V_R(gold)?")
    print("=" * 70)

    bootstrap_results = {}
    for layer in PROBE_LAYERS:
        lr = results['layers'][str(layer)]
        # Observed: mean across bins of V_R(pred) - V_R(gold)
        observed_diff = np.mean(np.array(lr['V_R_pred_incorrect']) - np.array(lr['V_R_gold_incorrect']))

        # Bootstrap: permute gold and pred labels within each incorrect problem
        # For each bootstrap iteration, randomly assign which target is "gold" and "pred"
        # This is equivalent to a paired permutation test on the per-bin differences
        diffs_per_bin = np.array(lr['V_R_pred_incorrect']) - np.array(lr['V_R_gold_incorrect'])

        count_exceed = 0
        for _ in range(N_BOOTSTRAP):
            # Randomly flip signs of per-bin differences
            signs = np.random.choice([-1, 1], size=len(diffs_per_bin))
            perm_mean = np.mean(diffs_per_bin * signs)
            if perm_mean >= observed_diff:
                count_exceed += 1

        p_val = (count_exceed + 1) / (N_BOOTSTRAP + 1)

        bootstrap_results[str(layer)] = {
            'observed_diff': float(observed_diff),
            'p_value': float(p_val),
            'n_exceed': count_exceed,
        }
        print(f"  Layer {layer}: Δ(pred-gold) = {observed_diff:+.4f}, p = {p_val:.4f}")

    results['bootstrap'] = bootstrap_results

    # Also bootstrap the per-problem closer-to-pred fraction
    print("\n  Per-problem closer-to-pred binomial test:")
    for layer in PROBE_LAYERS:
        lr = results['layers'][str(layer)]
        mean_pct = np.mean(lr['pct_closer_to_pred_V'])
        # Simple binomial: n_i problems, proportion p, test p > 0.5
        n_inc = len(incorrect_data)
        n_closer = int(round(mean_pct * n_inc))
        from scipy.stats import binomtest
        try:
            p_binom = binomtest(n_closer, n_inc, 0.5, alternative='greater').pvalue
        except Exception:
            p_binom = 1.0
        print(f"  Layer {layer}: {mean_pct:.1%} closer to pred, "
              f"binomial p = {p_binom:.4f} (n={n_inc})")
        bootstrap_results[str(layer)]['pct_closer_pred'] = float(mean_pct)
        bootstrap_results[str(layer)]['binom_p'] = float(p_binom)

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: Position-resolved analysis
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 5: Position-resolved V_R comparison")
    print("=" * 70)

    # For the best layer, show bin-by-bin comparison
    best_layer = max(PROBE_LAYERS, key=lambda l: np.mean(
        np.array(results['layers'][str(l)]['V_R_pred_incorrect']) -
        np.array(results['layers'][str(l)]['V_R_gold_incorrect'])
    ))
    lr = results['layers'][str(best_layer)]
    print(f"\n  Best layer: {best_layer}")
    print(f"  {'Bin':>4} {'V_R(gold)':>10} {'V_R(pred)':>10} {'Δ':>8} {'Closer':>8}")
    for b in range(N_BINS):
        delta = lr['V_R_pred_incorrect'][b] - lr['V_R_gold_incorrect'][b]
        pct = lr['pct_closer_to_pred_V'][b]
        print(f"  {b:>4} {lr['V_R_gold_incorrect'][b]:>10.4f} {lr['V_R_pred_incorrect'][b]:>10.4f} "
              f"{delta:>+8.4f} {pct:>7.1%}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 6: Figures
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 6: Generating figures")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bins_x = np.arange(N_BINS) * 5 + 2.5  # percentage positions

    # Figure 1: V_R(gold) vs V_R(predicted) for incorrect, all layers
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, layer in zip(axes.flatten(), PROBE_LAYERS):
        lr = results['layers'][str(layer)]
        ax.plot(bins_x, lr['V_R_gold_incorrect'], 'b-o', markersize=4, label='V_R(gold)', alpha=0.8)
        ax.plot(bins_x, lr['V_R_pred_incorrect'], 'r-s', markersize=4, label='V_R(predicted)', alpha=0.8)
        ax.plot(bins_x, lr['V_R_gold_correct'], 'g--', alpha=0.5, label='V_R(gold, correct)')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'Layer {layer} — V_R for incorrect: gold vs predicted')
        ax.set_xlabel('Position in chain (%)')
        ax.set_ylabel('Pearson R')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.5, 1.0)

        p = bootstrap_results[str(layer)]['p_value']
        diff = bootstrap_results[str(layer)]['observed_diff']
        ax.text(0.02, 0.98, f'Δ(pred-gold)={diff:+.3f}, p={p:.3f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'Exp 103: V-Probe — Gold vs Model Prediction (n_incorrect={len(incorrect_data)})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'V_R_gold_vs_predicted.png', dpi=150)
    plt.close()
    print("  Saved V_R_gold_vs_predicted.png")

    # Figure 2: Per-problem closer-to-pred fraction
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, layer in zip(axes.flatten(), PROBE_LAYERS):
        lr = results['layers'][str(layer)]
        ax.bar(bins_x - 1.5, lr['pct_closer_to_pred_V'], width=3, color='indianred',
               alpha=0.7, label='V-probe')
        ax.bar(bins_x + 1.5, lr['pct_closer_to_pred_nums'], width=3, color='steelblue',
               alpha=0.7, label='nums baseline')
        ax.axhline(0.5, color='black', linestyle='--', linewidth=1, label='chance (50%)')
        ax.set_title(f'Layer {layer} — Fraction closer to model\'s answer')
        ax.set_xlabel('Position in chain (%)')
        ax.set_ylabel('Fraction closer to pred')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.0)

    fig.suptitle(f'Exp 103: Per-Problem — V-probe closer to gold or model prediction?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'pct_closer_to_pred.png', dpi=150)
    plt.close()
    print("  Saved pct_closer_to_pred.png")

    # Figure 3: Summary bar chart — mean V_R(gold) vs V_R(pred) per layer
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    layer_labels = [f'L{l}' for l in PROBE_LAYERS]
    x = np.arange(len(PROBE_LAYERS))
    width = 0.35

    v_gold = [np.mean(results['layers'][str(l)]['V_R_gold_incorrect']) for l in PROBE_LAYERS]
    v_pred = [np.mean(results['layers'][str(l)]['V_R_pred_incorrect']) for l in PROBE_LAYERS]
    v_correct = [np.mean(results['layers'][str(l)]['V_R_gold_correct']) for l in PROBE_LAYERS]

    bars1 = ax1.bar(x - width/2, v_gold, width, label='V_R(gold)', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, v_pred, width, label='V_R(model pred)', color='indianred', alpha=0.8)
    ax1.bar(x, v_correct, width * 0.3, label='V_R(correct)', color='green', alpha=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_labels)
    ax1.set_ylabel('Mean Pearson R (incorrect problems)')
    ax1.set_title('V-probe: Gold vs Model Prediction')
    ax1.legend()
    ax1.axhline(0, color='gray', linestyle=':')

    # Add p-values
    for i, l in enumerate(PROBE_LAYERS):
        p = bootstrap_results[str(l)]['p_value']
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax1.text(i, max(v_gold[i], v_pred[i]) + 0.02, star,
                ha='center', fontsize=10, fontweight='bold')

    nums_gold = [np.mean(results['layers'][str(l)]['nums_R_gold_incorrect']) for l in PROBE_LAYERS]
    nums_pred = [np.mean(results['layers'][str(l)]['nums_R_pred_incorrect']) for l in PROBE_LAYERS]

    bars3 = ax2.bar(x - width/2, nums_gold, width, label='nums_R(gold)', color='steelblue', alpha=0.8)
    bars4 = ax2.bar(x + width/2, nums_pred, width, label='nums_R(model pred)', color='indianred', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels)
    ax2.set_ylabel('Mean Pearson R (incorrect problems)')
    ax2.set_title('Nums baseline: Gold vs Model Prediction')
    ax2.legend()
    ax2.axhline(0, color='gray', linestyle=':')

    fig.suptitle(f'Exp 103: Summary — Which target does the probe track? (n_inc={len(incorrect_data)})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'summary_bars.png', dpi=150)
    plt.close()
    print("  Saved summary_bars.png")

    # Figure 4: MAE comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    mae_gc = [np.mean(results['layers'][str(l)]['mean_abs_error_V_gold_correct']) for l in PROBE_LAYERS]
    mae_pi = [np.mean(results['layers'][str(l)]['mean_abs_error_V_pred_incorrect']) for l in PROBE_LAYERS]
    mae_gi = [np.mean(results['layers'][str(l)]['mean_abs_error_V_gold_incorrect']) for l in PROBE_LAYERS]

    ax.bar(x - width, mae_gc, width, label='Correct vs gold', color='green', alpha=0.7)
    ax.bar(x, mae_pi, width, label='Incorrect vs model pred', color='indianred', alpha=0.7)
    ax.bar(x + width, mae_gi, width, label='Incorrect vs gold', color='steelblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('V-probe prediction error by target')
    ax.legend()

    fig.suptitle(f'Exp 103: V-Probe Error — Correct vs gold, Incorrect vs pred/gold',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'mae_comparison.png', dpi=150)
    plt.close()
    print("  Saved mae_comparison.png")

    # ══════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nn_correct={len(correct_data)}, n_incorrect={len(incorrect_data)}")
    print(f"gold-pred correlation for incorrect: R={gold_pred_r:.3f}")

    print(f"\n{'Layer':>6} {'V_R(gold)':>10} {'V_R(pred)':>10} {'Δ':>8} {'p':>8} "
          f"{'%closer':>8} {'nums_R(g)':>10} {'nums_R(p)':>10}")
    print("-" * 85)

    for layer in PROBE_LAYERS:
        lr = results['layers'][str(layer)]
        br = bootstrap_results[str(layer)]
        v_g = np.mean(lr['V_R_gold_incorrect'])
        v_p = np.mean(lr['V_R_pred_incorrect'])
        n_g = np.mean(lr['nums_R_gold_incorrect'])
        n_p = np.mean(lr['nums_R_pred_incorrect'])
        pct = np.mean(lr['pct_closer_to_pred_V'])
        print(f"  L{layer:>3} {v_g:>10.4f} {v_p:>10.4f} {v_p-v_g:>+8.4f} {br['p_value']:>8.4f} "
              f"{pct:>7.1%} {n_g:>10.4f} {n_p:>10.4f}")

    # Interpretation guide
    print(f"\nInterpretation:")
    n_layers_pred_wins = sum(1 for l in PROBE_LAYERS
                            if np.mean(results['layers'][str(l)]['V_R_pred_incorrect']) >
                               np.mean(results['layers'][str(l)]['V_R_gold_incorrect']))
    n_layers_sig = sum(1 for l in PROBE_LAYERS
                       if bootstrap_results[str(l)]['p_value'] < 0.05)
    mean_pct_all = np.mean([np.mean(results['layers'][str(l)]['pct_closer_to_pred_V'])
                           for l in PROBE_LAYERS])

    if n_layers_pred_wins >= 3 and n_layers_sig >= 2:
        print(f"  → V-cache encodes MODEL'S OWN COMPUTATION (pred > gold at {n_layers_pred_wins}/4 layers, "
              f"{n_layers_sig}/4 significant)")
        print(f"  → {mean_pct_all:.1%} of incorrect problems closer to model's answer")
        print(f"  → The negative V|nums for incorrect is because V tracks model's answer, not gold")
        print(f"  → Interpretation A SUPPORTED: Hidden channel is computation-faithful")
    elif n_layers_pred_wins <= 1:
        print(f"  → V_R(pred) ≤ V_R(gold) at most layers → general features, not computation")
        print(f"  → Interpretation B SUPPORTED: V-cache encodes problem features")
    else:
        print(f"  → Mixed result: pred > gold at {n_layers_pred_wins}/4 layers, {n_layers_sig}/4 significant")
        print(f"  → Per-problem: {mean_pct_all:.1%} closer to pred")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
