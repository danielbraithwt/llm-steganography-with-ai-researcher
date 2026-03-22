#!/usr/bin/env python3
"""
Experiment 102: Cross-Model Replication of Accuracy/Difficulty-Conditional V|nums
              on Phi-3.5-mini-Instruct (MHA, analog encoding)

Replicates exp_101's functional tests on a maximally different architecture:
- Phi: MHA (32 KV heads), analog encoding, instruct-tuned
- Qwen: GQA (8 KV heads), digital encoding, base model

Two analyses (identical to exp_101):
A. CORRECT vs INCORRECT: Train V-probe on correct problems, apply to incorrect.
   If V|nums differs by accuracy, the channel carries the actual answer.
B. DIFFICULTY SPLIT: Split correct problems by ground-truth reasoning steps.
   If V|nums scales with difficulty, the channel does computational work.

PLUS additional analysis:
C. V_R DECOMPOSITION by difficulty: Report V_R and nums_R separately to
   test whether the difficulty effect is driven by V_R increase or nums_R collapse.
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
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

T0 = time.time()
TIME_BUDGET = 7000  # 116 min (from 7200s timeout)
MAX_GEN = 512
MAX_SEQ_LEN = 2048
MODEL_NAME = 'microsoft/Phi-3.5-mini-instruct'
N_PROBLEMS = 700  # aim for 300+ generated
PROBE_LAYERS = [8, 16, 24, 31]  # ~25%, 50%, 75%, 97% of 32 layers
N_BINS = 20
N_FOLDS = 5
MAX_NUMS_DIM = 30
N_BOOTSTRAP = 500
PCA_DIM = 256  # reduce Phi's 3072-dim V vectors to prevent overfitting
RIDGE_ALPHA = 10.0  # higher alpha for larger KV dim

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_102"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same as exp_101) ──
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


def count_reasoning_steps(answer_text):
    """Count reasoning steps in GSM8K ground-truth solution."""
    lines = answer_text.strip().split('\n')
    steps = 0
    for line in lines:
        line = line.strip()
        if line.startswith('####'):
            break
        if line and len(line) > 5:
            steps += 1
    return max(steps, 1)


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


def answer_in_text(text, answer_str):
    clean_text = text.replace(',', '')
    clean_ans = answer_str.replace(',', '')
    if not clean_ans:
        return False
    pattern = r'(?<!\d)' + re.escape(clean_ans) + r'(?!\d)'
    return bool(re.search(pattern, clean_text))


def cv_ridge_r(X, y, n_folds=N_FOLDS, alpha=RIDGE_ALPHA):
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
    print("Experiment 102: Phi-3.5-mini Accuracy/Difficulty-Conditional V|nums")
    print("Cross-model replication of exp_101 on MHA/analog/instruct architecture")
    print("=" * 70)

    # ── Load model ──
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
    )
    model.eval()
    print(f"Model: {MODEL_NAME}")

    n_layers = model.config.num_hidden_layers
    kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // model.config.num_attention_heads)
    kv_dim = kv_heads * head_dim
    print(f"Layers: {n_layers}, KV heads: {kv_heads}, head_dim: {head_dim}, KV dim: {kv_dim}")
    print(f"Probe layers: {PROBE_LAYERS}")
    print(f"PCA: {kv_dim}-dim → {PCA_DIM}-dim (to match Qwen's dimensionality)")

    ds = load_gsm8k()
    print(f"GSM8K test set: {len(ds)} problems")

    # Count reasoning steps
    step_counts = {}
    for i in range(len(ds)):
        step_counts[i] = count_reasoning_steps(ds[i]['answer'])

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Generate CoT traces
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: Generating CoT traces")
    print("=" * 70)

    generations = []
    gen_time_budget = TIME_BUDGET * 0.25  # 25% for generation

    for i in range(min(N_PROBLEMS, len(ds))):
        if time.time() - t0 > gen_time_budget:
            print(f"  Time budget ({gen_time_budget:.0f}s / 25%) reached at problem {i}")
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
            'gold_float': float(gold) if gold else None,
            'n_steps': step_counts[i],
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
    n_incorrect = n_total - n_correct
    print(f"\nPhase 1 complete: {n_total} generated, {n_correct} correct "
          f"({100*n_correct/n_total:.1f}%), {n_incorrect} incorrect")

    # Find chain boundaries
    usable_correct = []
    usable_incorrect = []
    for g in generations:
        hash_pos = find_hash_pos_in_gen(g['gen_ids'], tokenizer)
        if g['correct']:
            if hash_pos is not None and hash_pos >= 10:
                g['chain_len'] = hash_pos
                usable_correct.append(g)
        else:
            chain_len = len(g['gen_ids'])
            if hash_pos is not None and hash_pos >= 10:
                chain_len = hash_pos
            if chain_len >= 10:
                g['chain_len'] = chain_len
                usable_incorrect.append(g)

    print(f"Usable correct: {len(usable_correct)}, Usable incorrect: {len(usable_incorrect)}")

    # Difficulty bins (same as exp_101: tercile split by ground-truth steps)
    correct_steps = [g['n_steps'] for g in usable_correct]
    step_arr = np.array(correct_steps)
    t1 = np.percentile(step_arr, 33.3)
    t2 = np.percentile(step_arr, 66.7)
    print(f"Difficulty terciles: Easy ≤{t1:.0f} steps, Medium {t1:.0f}-{t2:.0f}, Hard >{t2:.0f}")

    for g in usable_correct:
        if g['n_steps'] <= t1:
            g['difficulty'] = 'easy'
        elif g['n_steps'] <= t2:
            g['difficulty'] = 'medium'
        else:
            g['difficulty'] = 'hard'

    for diff in ['easy', 'medium', 'hard']:
        n = sum(1 for g in usable_correct if g['difficulty'] == diff)
        mean_steps = np.mean([g['n_steps'] for g in usable_correct if g['difficulty'] == diff])
        print(f"  {diff}: {n} problems (mean {mean_steps:.1f} steps)")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Forward pass + V-cache extraction
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: Extracting V-cache at all CoT positions")
    print("=" * 70)

    all_problems_data = []
    n_extracted = 0
    all_gens = usable_correct + usable_incorrect
    np.random.shuffle(all_gens)

    extract_budget = TIME_BUDGET * 0.62  # 62% for extraction

    for pi, gen in enumerate(all_gens):
        if time.time() - t0 > extract_budget:
            print(f"\n  Time budget (62%) reached at problem {pi}")
            break

        chain_len = gen['chain_len']
        gen_ids = gen['gen_ids'][:chain_len]
        gold_float = gen.get('gold_float')
        if gold_float is None:
            continue

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
        log_answer = log_transform(gold_float)
        gen_text_so_far = ""

        problem_data = {
            'idx': gen['idx'],
            'correct': gen['correct'],
            'n_steps': gen['n_steps'],
            'difficulty': gen.get('difficulty', 'incorrect'),
            'gold_float': gold_float,
            'log_answer': log_answer,
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
            text_reveals = answer_in_text(partial_text, str(gen['gold'])) if gen['gold'] else False

            problem_data['bins'][b] = {
                'V': V_layers,
                'nums_feat': nums_feat,
                'text_reveals': text_reveals,
            }

        all_problems_data.append(problem_data)
        n_extracted += 1

        del out, cache
        if n_extracted % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if n_extracted % 50 == 0:
            n_c = sum(1 for p in all_problems_data if p['correct'])
            n_i = sum(1 for p in all_problems_data if not p['correct'])
            elapsed = time.time() - t0
            print(f"  Extracted {n_extracted} problems (correct={n_c}, incorrect={n_i}) [{elapsed:.0f}s]")

    # Free model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    n_c_final = sum(1 for p in all_problems_data if p['correct'])
    n_i_final = sum(1 for p in all_problems_data if not p['correct'])
    print(f"\nPhase 2 complete: {len(all_problems_data)} problems extracted "
          f"(correct={n_c_final}, incorrect={n_i_final})")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2.5: PCA dimensionality reduction
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 2.5: PCA reduction {kv_dim}-dim → {PCA_DIM}-dim")
    print("=" * 70)

    # Fit PCA on correct problems' V vectors (one PCA per layer)
    pca_models = {}
    correct_data = [p for p in all_problems_data if p['correct']]

    for layer in PROBE_LAYERS:
        # Gather all V vectors from correct problems
        all_v = []
        for p in correct_data:
            for b in p['bins']:
                all_v.append(p['bins'][b]['V'][layer])
        all_v = np.array(all_v)
        print(f"  L{layer}: fitting PCA on {all_v.shape[0]} vectors of dim {all_v.shape[1]}")

        pca = PCA(n_components=min(PCA_DIM, all_v.shape[1], all_v.shape[0]))
        pca.fit(all_v)
        pca_models[layer] = pca
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  L{layer}: {pca.n_components_} components explain {var_explained:.1%} variance")

    # Transform all V vectors through PCA
    for p in all_problems_data:
        for b in p['bins']:
            for layer in PROBE_LAYERS:
                v_raw = p['bins'][b]['V'][layer].reshape(1, -1)
                p['bins'][b]['V'][layer] = pca_models[layer].transform(v_raw)[0]

    print("  PCA transformation complete")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Analysis
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: Analysis — Accuracy-conditional & difficulty-conditional V|nums")
    print("=" * 70)

    correct_data = [p for p in all_problems_data if p['correct']]
    incorrect_data = [p for p in all_problems_data if not p['correct']]
    print(f"Analysis samples: {len(correct_data)} correct, {len(incorrect_data)} incorrect")

    results = {
        'model': MODEL_NAME,
        'n_generated': n_total,
        'n_correct_gen': n_correct,
        'n_incorrect_gen': n_incorrect,
        'accuracy_pct': 100 * n_correct / n_total,
        'n_correct': len(correct_data),
        'n_incorrect': len(incorrect_data),
        'pca_dim': PCA_DIM,
        'ridge_alpha': RIDGE_ALPHA,
        'layers': {},
    }

    for layer in PROBE_LAYERS:
        print(f"\n--- Layer {layer} ---")
        layer_results = {
            'correct_V_R': [], 'incorrect_V_R': [],
            'correct_nums_R': [], 'incorrect_nums_R': [],
            'correct_Vnums': [], 'incorrect_Vnums': [],
            'text_reveals_correct': [], 'text_reveals_incorrect': [],
            'difficulty_V_R': {'easy': [], 'medium': [], 'hard': []},
            'difficulty_nums_R': {'easy': [], 'medium': [], 'hard': []},
            'difficulty_Vnums': {'easy': [], 'medium': [], 'hard': []},
            'shuffle_V_R': [],
        }

        for b in range(N_BINS):
            V_correct = []
            y_correct = []
            nums_correct = []
            diff_labels = []

            V_incorrect = []
            y_incorrect = []
            nums_incorrect = []

            reveals_c = 0
            reveals_i = 0

            for p in correct_data:
                if b not in p['bins']:
                    continue
                V_correct.append(p['bins'][b]['V'][layer])
                y_correct.append(p['log_answer'])
                nums_correct.append(p['bins'][b]['nums_feat'])
                diff_labels.append(p['difficulty'])
                if p['bins'][b]['text_reveals']:
                    reveals_c += 1

            for p in incorrect_data:
                if b not in p['bins']:
                    continue
                V_incorrect.append(p['bins'][b]['V'][layer])
                y_incorrect.append(p['log_answer'])
                nums_incorrect.append(p['bins'][b]['nums_feat'])
                if p['bins'][b]['text_reveals']:
                    reveals_i += 1

            V_c = np.array(V_correct) if V_correct else np.zeros((0, 1))
            y_c = np.array(y_correct)
            nums_c = np.array(nums_correct) if nums_correct else np.zeros((0, 1))
            V_i = np.array(V_incorrect) if V_incorrect else np.zeros((0, 1))
            y_i = np.array(y_incorrect)
            nums_i = np.array(nums_incorrect) if nums_incorrect else np.zeros((0, 1))

            n_c = len(V_c)
            n_i = len(V_i)

            # --- Correct V-probe (CV) ---
            V_R_c = cv_ridge_r(V_c, y_c) if n_c >= N_FOLDS * 2 else 0.0
            nums_R_c = cv_ridge_r(nums_c, y_c) if n_c >= N_FOLDS * 2 else 0.0
            Vnums_c = max(V_R_c - nums_R_c, 0.0)

            # --- Incorrect: TRAIN on correct, PREDICT on incorrect ---
            V_R_i = 0.0
            nums_R_i = 0.0
            if n_c >= N_FOLDS * 2 and n_i >= 5:
                ridge_V = Ridge(alpha=RIDGE_ALPHA)
                ridge_V.fit(V_c, y_c)
                pred_i = ridge_V.predict(V_i)
                if np.std(pred_i) > 1e-10 and np.std(y_i) > 1e-10:
                    V_R_i = np.corrcoef(pred_i, y_i)[0, 1]
                    if not np.isfinite(V_R_i):
                        V_R_i = 0.0

                ridge_nums = Ridge(alpha=RIDGE_ALPHA)
                ridge_nums.fit(nums_c, y_c)
                pred_nums_i = ridge_nums.predict(nums_i)
                if np.std(pred_nums_i) > 1e-10 and np.std(y_i) > 1e-10:
                    nums_R_i = np.corrcoef(pred_nums_i, y_i)[0, 1]
                    if not np.isfinite(nums_R_i):
                        nums_R_i = 0.0

            Vnums_i = V_R_i - nums_R_i  # can be negative

            # --- Shuffle control ---
            shuf_r = 0.0
            if n_c >= N_FOLDS * 2:
                y_shuf = y_c.copy()
                np.random.shuffle(y_shuf)
                shuf_r = cv_ridge_r(V_c, y_shuf)

            # --- Difficulty split (among correct) ---
            for diff in ['easy', 'medium', 'hard']:
                diff_mask = [j for j, dl in enumerate(diff_labels) if dl == diff]
                if len(diff_mask) < N_FOLDS * 2:
                    layer_results['difficulty_V_R'][diff].append(0.0)
                    layer_results['difficulty_nums_R'][diff].append(0.0)
                    layer_results['difficulty_Vnums'][diff].append(0.0)
                    continue

                V_diff = V_c[diff_mask]
                y_diff = y_c[diff_mask]
                nums_diff = nums_c[diff_mask]

                diff_V_R = cv_ridge_r(V_diff, y_diff)
                diff_nums_R = cv_ridge_r(nums_diff, y_diff)
                layer_results['difficulty_V_R'][diff].append(diff_V_R)
                layer_results['difficulty_nums_R'][diff].append(diff_nums_R)
                layer_results['difficulty_Vnums'][diff].append(max(diff_V_R - diff_nums_R, 0.0))

            # Store
            layer_results['correct_V_R'].append(V_R_c)
            layer_results['incorrect_V_R'].append(V_R_i)
            layer_results['correct_nums_R'].append(nums_R_c)
            layer_results['incorrect_nums_R'].append(nums_R_i)
            layer_results['correct_Vnums'].append(Vnums_c)
            layer_results['incorrect_Vnums'].append(Vnums_i)
            layer_results['text_reveals_correct'].append(reveals_c / n_c if n_c else 0)
            layer_results['text_reveals_incorrect'].append(reveals_i / n_i if n_i else 0)
            layer_results['shuffle_V_R'].append(shuf_r)

            if b % 5 == 0 or b == N_BINS - 1:
                print(f"  Bin {b:2d}: V_R(c)={V_R_c:.3f} V_R(i)={V_R_i:.3f} "
                      f"V|nums(c)={Vnums_c:.3f} V|nums(i)={Vnums_i:.3f} "
                      f"text_reveals(c)={reveals_c/n_c:.2f}" if n_c > 0 else "")

        results['layers'][str(layer)] = layer_results

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Bootstrap significance tests
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: Bootstrap tests")
    print("=" * 70)

    bootstrap_results = {}
    for layer in PROBE_LAYERS:
        lr = results['layers'][str(layer)]
        c_vnums = np.array(lr['correct_Vnums'])
        i_vnums = np.array(lr['incorrect_Vnums'])

        # Test 1: V|nums(correct) vs V|nums(incorrect)
        observed_gap = np.mean(c_vnums) - np.mean(i_vnums)
        pooled = np.concatenate([c_vnums, i_vnums])
        n_c_bins = len(c_vnums)
        boot_gaps = []
        for _ in range(N_BOOTSTRAP):
            perm = np.random.permutation(pooled)
            boot_gaps.append(np.mean(perm[:n_c_bins]) - np.mean(perm[n_c_bins:]))
        p_accuracy = float(np.mean(np.array(boot_gaps) >= observed_gap))

        # Test 2: V|nums(hard) - V|nums(easy)
        hard_vnums = np.array(lr['difficulty_Vnums']['hard'])
        easy_vnums = np.array(lr['difficulty_Vnums']['easy'])
        observed_diff_gap = np.mean(hard_vnums) - np.mean(easy_vnums)
        pooled_diff = np.concatenate([hard_vnums, easy_vnums])
        n_h = len(hard_vnums)
        boot_diff_gaps = []
        for _ in range(N_BOOTSTRAP):
            perm = np.random.permutation(pooled_diff)
            boot_diff_gaps.append(np.mean(perm[:n_h]) - np.mean(perm[n_h:]))
        p_difficulty = float(np.mean(np.array(boot_diff_gaps) >= observed_diff_gap))

        bootstrap_results[str(layer)] = {
            'accuracy_gap': float(observed_gap),
            'accuracy_p': float(p_accuracy),
            'difficulty_gap': float(observed_diff_gap),
            'difficulty_p': float(p_difficulty),
        }

        print(f"  L{layer}: Correct-Incorrect gap = {observed_gap:.3f} (p={p_accuracy:.3f})")
        print(f"  L{layer}: Hard-Easy gap = {observed_diff_gap:.3f} (p={p_difficulty:.3f})")

    results['bootstrap'] = bootstrap_results

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: Summary & V_R decomposition
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 5: Summary")
    print("=" * 70)

    diff_stats = {}
    for diff in ['easy', 'medium', 'hard']:
        group = [p for p in correct_data if p['difficulty'] == diff]
        diff_stats[diff] = {
            'n': len(group),
            'mean_steps': float(np.mean([p['n_steps'] for p in group])),
        }
    diff_stats['incorrect'] = {'n': len(incorrect_data)}
    results['difficulty_stats'] = diff_stats

    print("\n=== ANALYSIS A: CORRECT vs INCORRECT ===")
    for layer in PROBE_LAYERS:
        lr = results['layers'][str(layer)]
        br = bootstrap_results[str(layer)]
        print(f"\nLayer {layer}:")
        print(f"  V_R    : correct mean={np.mean(lr['correct_V_R']):.3f}, "
              f"incorrect mean={np.mean(lr['incorrect_V_R']):.3f}")
        print(f"  nums_R : correct mean={np.mean(lr['correct_nums_R']):.3f}, "
              f"incorrect mean={np.mean(lr['incorrect_nums_R']):.3f}")
        print(f"  V|nums : correct mean={np.mean(lr['correct_Vnums']):.3f}, "
              f"incorrect mean={np.mean(lr['incorrect_Vnums']):.3f}, "
              f"gap={br['accuracy_gap']:.3f}, p={br['accuracy_p']:.3f}")

    print("\n=== ANALYSIS B: DIFFICULTY SPLIT (correct only) ===")
    for layer in PROBE_LAYERS:
        lr = results['layers'][str(layer)]
        br = bootstrap_results[str(layer)]
        print(f"\nLayer {layer}:")
        for diff in ['easy', 'medium', 'hard']:
            v_r_mean = np.mean(lr['difficulty_V_R'][diff])
            n_r_mean = np.mean(lr['difficulty_nums_R'][diff])
            vn_mean = np.mean(lr['difficulty_Vnums'][diff])
            n_diff = diff_stats[diff]['n']
            ms = diff_stats[diff]['mean_steps']
            print(f"  {diff:7s}: V_R={v_r_mean:.3f}, nums_R={n_r_mean:.3f}, "
                  f"V|nums={vn_mean:.3f} (n={n_diff}, {ms:.0f} steps)")
        print(f"  Hard-Easy gap = {br['difficulty_gap']:.3f} (p={br['difficulty_p']:.3f})")

    # Analysis C: V_R decomposition diagnostic
    print("\n=== ANALYSIS C: V_R DECOMPOSITION (is difficulty effect from V_R or nums_R?) ===")
    for layer in PROBE_LAYERS:
        lr = results['layers'][str(layer)]
        print(f"\nLayer {layer}:")
        for diff in ['easy', 'medium', 'hard']:
            v_r = np.mean(lr['difficulty_V_R'][diff])
            n_r = np.mean(lr['difficulty_nums_R'][diff])
            vn = np.mean(lr['difficulty_Vnums'][diff])
            # Count bins where V > nums (unclipped)
            v_r_arr = np.array(lr['difficulty_V_R'][diff])
            n_r_arr = np.array(lr['difficulty_nums_R'][diff])
            v_gt_n = np.sum(v_r_arr > n_r_arr)
            print(f"  {diff:7s}: V_R={v_r:.3f}, nums_R={n_r:.3f}, V-nums (unclipped)={v_r-n_r:.3f}, "
                  f"V>nums at {v_gt_n}/{N_BINS} bins")

    # Early decodability
    print("\n=== EARLY DECODABILITY ===")
    for layer in PROBE_LAYERS:
        lr = results['layers'][str(layer)]
        for b in range(N_BINS):
            if lr['text_reveals_correct'][b] < 0.5 and lr['correct_Vnums'][b] > 0.05:
                print(f"  L{layer}: V|nums > 0.05 at bin {b} ({5*b:.0f}% of chain), "
                      f"text reveals at {lr['text_reveals_correct'][b]:.0%}")
                break

    # Cross-model comparison summary
    print("\n=== CROSS-MODEL COMPARISON (Phi vs Qwen from exp_101) ===")
    print("Qwen exp_101 reference values:")
    print("  Accuracy effect: gap=0.036-0.164, p=0.006-0.216")
    print("  Difficulty effect: gap=0.203-0.241, p<0.001")
    print("  Hard V_R=0.145, nums_R=-0.085, V>nums 15/20 bins")
    print("  Easy V_R=0.198, nums_R=0.334, V>nums 6/20 bins")
    print("\nPhi exp_102 values:")
    for layer in PROBE_LAYERS:
        br = bootstrap_results[str(layer)]
        lr = results['layers'][str(layer)]
        print(f"  L{layer}: acc_gap={br['accuracy_gap']:.3f} (p={br['accuracy_p']:.3f}), "
              f"diff_gap={br['difficulty_gap']:.3f} (p={br['difficulty_p']:.3f})")

    # Save results
    results_file = RESULTS_DIR / 'results.json'

    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nResults saved to {results_file}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 6: Figures
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 6: Generating figures")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x_bins = np.linspace(2.5, 97.5, N_BINS)

    # ── Figure 1: Correct vs Incorrect V|nums ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, layer in enumerate(PROBE_LAYERS):
        ax = axes[ax_idx // 2][ax_idx % 2]
        lr = results['layers'][str(layer)]
        br = bootstrap_results[str(layer)]

        ax.plot(x_bins, lr['correct_Vnums'], 'b-o', markersize=4, label='Correct', linewidth=2)
        ax.plot(x_bins, lr['incorrect_Vnums'], 'r-s', markersize=4, label='Incorrect', linewidth=2)
        ax.plot(x_bins, lr['shuffle_V_R'], 'k--', alpha=0.4, label='Shuffle ctrl', linewidth=1)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        ax.fill_between(x_bins, 0, max(max(lr['correct_Vnums']), 0.1) * 1.1,
                        where=[t > 0.5 for t in lr['text_reveals_correct']],
                        alpha=0.1, color='green', label='Text reveals')

        ax.set_title(f"Layer {layer} — gap={br['accuracy_gap']:.3f}, p={br['accuracy_p']:.3f}")
        ax.set_xlabel('Position in chain (%)')
        ax.set_ylabel('V|nums (R)')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 100)

    fig.suptitle(f'Phi-3.5-mini: Correct vs Incorrect V|nums (n_c={len(correct_data)}, n_i={len(incorrect_data)})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'correct_vs_incorrect_vnums.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved correct_vs_incorrect_vnums.png")

    # ── Figure 2: Difficulty split V|nums ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, layer in enumerate(PROBE_LAYERS):
        ax = axes[ax_idx // 2][ax_idx % 2]
        lr = results['layers'][str(layer)]
        br = bootstrap_results[str(layer)]

        colors = {'easy': '#2196F3', 'medium': '#FF9800', 'hard': '#F44336'}
        for diff in ['easy', 'medium', 'hard']:
            n_diff = diff_stats[diff]['n']
            ms = diff_stats[diff]['mean_steps']
            ax.plot(x_bins, lr['difficulty_Vnums'][diff], '-o', markersize=3,
                    color=colors[diff], label=f'{diff} (n={n_diff}, {ms:.0f} steps)',
                    linewidth=2)

        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.fill_between(x_bins, 0, max(max(lr['correct_Vnums']), 0.1) * 1.1,
                        where=[t > 0.5 for t in lr['text_reveals_correct']],
                        alpha=0.1, color='green', label='Text reveals')

        ax.set_title(f"Layer {layer} — Hard-Easy gap={br['difficulty_gap']:.3f}, p={br['difficulty_p']:.3f}")
        ax.set_xlabel('Position in chain (%)')
        ax.set_ylabel('V|nums (R)')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 100)

    fig.suptitle('Phi-3.5-mini: Difficulty Split V|nums (Correct Only)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'difficulty_split_vnums.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved difficulty_split_vnums.png")

    # ── Figure 3: Raw V_R and nums_R for correct vs incorrect ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, layer in enumerate(PROBE_LAYERS):
        ax = axes[ax_idx // 2][ax_idx % 2]
        lr = results['layers'][str(layer)]

        ax.plot(x_bins, lr['correct_V_R'], 'b-o', markersize=4, label='V_R (correct, CV)', linewidth=2)
        ax.plot(x_bins, lr['incorrect_V_R'], 'r-s', markersize=4,
                label='V_R (incorrect, train-on-correct)', linewidth=2)
        ax.plot(x_bins, lr['correct_nums_R'], 'b--', alpha=0.5, label='nums_R (correct)', linewidth=1)
        ax.plot(x_bins, lr['incorrect_nums_R'], 'r--', alpha=0.5, label='nums_R (incorrect)', linewidth=1)
        ax.plot(x_bins, lr['shuffle_V_R'], 'k:', alpha=0.4, label='Shuffle ctrl', linewidth=1)

        ax.fill_between(x_bins, 0, 1.0,
                        where=[t > 0.5 for t in lr['text_reveals_correct']],
                        alpha=0.1, color='green', label='Text reveals')

        ax.set_title(f"Layer {layer}")
        ax.set_xlabel('Position in chain (%)')
        ax.set_ylabel('Pearson R')
        ax.legend(fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.3, 1.0)

    fig.suptitle('Phi-3.5-mini: Raw V_R and nums_R', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'raw_V_R_correct_vs_incorrect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved raw_V_R_correct_vs_incorrect.png")

    # ── Figure 4: Summary bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    layers_str = [f'L{l}' for l in PROBE_LAYERS]
    x = np.arange(len(PROBE_LAYERS))

    # Bar chart: accuracy effect
    ax = axes[0]
    correct_means = [np.mean(results['layers'][str(l)]['correct_Vnums']) for l in PROBE_LAYERS]
    incorrect_means = [np.mean(results['layers'][str(l)]['incorrect_Vnums']) for l in PROBE_LAYERS]
    w = 0.35
    ax.bar(x - w/2, correct_means, w, label='Correct', color='#2196F3')
    ax.bar(x + w/2, incorrect_means, w, label='Incorrect', color='#F44336')
    ax.set_xticks(x)
    ax.set_xticklabels(layers_str)
    ax.set_ylabel('Mean V|nums')
    ax.set_title('A: Accuracy Effect (Phi-3.5-mini)')
    ax.legend()
    for i, layer in enumerate(PROBE_LAYERS):
        p = bootstrap_results[str(layer)]['accuracy_p']
        ymax = max(correct_means[i], incorrect_means[i])
        ax.text(i, ymax + 0.005, f'p={p:.3f}', ha='center', fontsize=9)

    # Bar chart: difficulty effect
    ax = axes[1]
    for di, diff in enumerate(['easy', 'medium', 'hard']):
        means = [np.mean(results['layers'][str(l)]['difficulty_Vnums'][diff]) for l in PROBE_LAYERS]
        offset = (di - 1) * 0.25
        ax.bar(x + offset, means, 0.25, label=f'{diff} ({diff_stats[diff]["n"]})',
               color=['#2196F3', '#FF9800', '#F44336'][di])
    ax.set_xticks(x)
    ax.set_xticklabels(layers_str)
    ax.set_ylabel('Mean V|nums')
    ax.set_title('B: Difficulty Effect (Phi-3.5-mini, correct only)')
    ax.legend()
    for i, layer in enumerate(PROBE_LAYERS):
        p = bootstrap_results[str(layer)]['difficulty_p']
        ax.text(i, ax.get_ylim()[1] * 0.95, f'p={p:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'summary_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved summary_bars.png")

    # ── Figure 5: V_R decomposition by difficulty ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, layer in enumerate(PROBE_LAYERS):
        ax = axes[ax_idx // 2][ax_idx % 2]
        lr = results['layers'][str(layer)]

        for diff, color, marker in [('easy', '#2196F3', 'o'), ('hard', '#F44336', 's')]:
            n_d = diff_stats[diff]['n']
            ax.plot(x_bins, lr['difficulty_V_R'][diff], f'-{marker}', markersize=3,
                    color=color, label=f'V_R ({diff}, n={n_d})', linewidth=2)
            ax.plot(x_bins, lr['difficulty_nums_R'][diff], f'--{marker}', markersize=3,
                    color=color, alpha=0.5, label=f'nums_R ({diff})', linewidth=1)

        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f"Layer {layer}: V_R vs nums_R Decomposition")
        ax.set_xlabel('Position in chain (%)')
        ax.set_ylabel('Pearson R')
        ax.legend(fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.3, 1.0)

    fig.suptitle('Phi-3.5-mini: V_R Decomposition — Is difficulty effect from V_R or nums_R?',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'vr_decomposition_by_difficulty.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved vr_decomposition_by_difficulty.png")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
