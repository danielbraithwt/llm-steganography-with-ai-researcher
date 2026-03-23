#!/usr/bin/env python3
"""
Experiment 109: K-V Functional Dissociation at Intermediate Computation Positions

Tests whether K and V play different roles at arithmetic "=" positions:
- K=routing: K encodes operand features / positions for routing
- V=content: V encodes computed intermediate results

Key analyses:
1. K→result vs V→result at each layer (who carries computation?)
2. K→A vs V→A (operand encoding comparison)
3. Computation transition: (cache→result − cache→A) for K vs V across layers
4. K|MLP vs V|MLP (unique info beyond nonlinear operand baseline)
5. Offset analysis: K vs V forward-looking profiles
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
N_PROBLEMS = 400
PROBE_LAYERS = [9, 18, 27, 35]  # 25%, 50%, 75%, 97% depth
N_FOLDS = 5
MAX_NUMS_DIM = 30
POSITION_OFFSETS = [0, -5, -10]

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_109"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same as exp 106/108) ──
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


def load_gsm8k():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for row in ds:
        ans_str = row['answer'].split('####')[-1].strip().replace(',', '')
        try:
            answer = float(ans_str)
        except Exception:
            answer = None
        problems.append({
            'question': row['question'],
            'gold_answer': answer,
        })
    return problems


def signed_log(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def parse_arithmetic(text):
    """Find arithmetic expressions 'expr = result' and return with char positions."""
    num = r'[\d][\d,]*(?:\.\d+)?'
    op = r'\s*[+\-*/]\s*'
    expr = f'({num}(?:{op}{num})+)'
    result_pat = f'=\\s*\\$?\\s*({num})'
    pattern = f'{expr}\\s*{result_pat}'

    results = []
    for m in re.finditer(pattern, text):
        expr_str = m.group(1).replace(',', '').strip()
        written_str = m.group(2).replace(',', '').strip()
        try:
            written_result = float(written_str)
            sanitized = re.sub(r'[^\d+\-*/. ]', '', expr_str)
            if not sanitized.strip():
                continue
            correct_result = float(eval(sanitized))

            eq_pos_in_match = m.group(0).index('=')
            eq_char_pos = m.start() + eq_pos_in_match

            ops_found = re.findall(r'[+\-*/]', expr_str)
            if len(ops_found) == 1:
                op_type = ops_found[0]
            else:
                op_type = 'multi'

            operands = re.findall(r'[\d][\d,]*(?:\.\d+)?', expr_str)
            operands = [float(o.replace(',', '')) for o in operands]

            results.append({
                'expr_str': expr_str,
                'written_result': written_result,
                'correct_result': correct_result,
                'eq_char_pos': eq_char_pos,
                'char_start': m.start(),
                'op_type': op_type,
                'operands': operands,
            })
        except Exception:
            continue
    return results


def map_eq_to_token(gen_text, gen_ids, tokenizer, eq_char_pos):
    """Map '=' at eq_char_pos in gen_text to token index in gen_ids."""
    eq_count_target = gen_text[:eq_char_pos + 1].count('=') - 1
    if eq_count_target < 0:
        return None
    count = 0
    for i, tid in enumerate(gen_ids):
        tok = tokenizer.decode([tid])
        for ch in tok:
            if ch == '=':
                if count == eq_count_target:
                    return i
                count += 1
    return None


def get_kv(past_kv, layer_idx):
    from transformers import DynamicCache
    if isinstance(past_kv, DynamicCache):
        if hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
            return past_kv.layers[layer_idx].keys, past_kv.layers[layer_idx].values
        else:
            return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
    else:
        return past_kv[layer_idx][0], past_kv[layer_idx][1]


def operand_features(operands, op_type):
    """Create feature vector from operands and operation type."""
    A = operands[0] if len(operands) >= 1 else 0.0
    B = operands[1] if len(operands) >= 2 else 0.0
    feat = np.zeros(6)
    feat[0] = signed_log(A)
    feat[1] = signed_log(B)
    op_map = {'+': 2, '-': 3, '*': 4, '/': 5}
    if op_type in op_map:
        feat[op_map[op_type]] = 1.0
    return feat


def run_probe_cv(X, y, problem_ids, n_splits=N_FOLDS):
    """Cross-validate Ridge at problem level."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    unique_probs = np.unique(problem_ids)
    if len(unique_probs) < n_splits or X.shape[0] < 20:
        return 0.0, np.zeros(len(y))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_pred = np.full(len(y), np.nan)

    for train_prob_idx, test_prob_idx in kf.split(unique_probs):
        train_probs = set(unique_probs[train_prob_idx])
        test_probs = set(unique_probs[test_prob_idx])

        train_mask = np.array([pid in train_probs for pid in problem_ids])
        test_mask = np.array([pid in test_probs for pid in problem_ids])

        if train_mask.sum() < 10 or test_mask.sum() < 3:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])

        ridge = RidgeCV(alphas=np.logspace(-2, 6, 50))
        ridge.fit(X_train, y[train_mask])
        y_pred[test_mask] = ridge.predict(X_test)

    predicted = ~np.isnan(y_pred)
    if predicted.sum() < 10:
        return 0.0, np.zeros(len(y))
    r, _ = stats.pearsonr(y[predicted], y_pred[predicted])
    y_pred_filled = np.where(predicted, y_pred, 0.0)
    return float(r), y_pred_filled


def run_mlp_cv(X, y, problem_ids, hidden=(128, 64), n_splits=N_FOLDS):
    """Cross-validate MLP regressor at problem level."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    unique_probs = np.unique(problem_ids)
    if len(unique_probs) < n_splits or X.shape[0] < 20:
        return 0.0, np.zeros(len(y))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_pred = np.full(len(y), np.nan)

    for train_prob_idx, test_prob_idx in kf.split(unique_probs):
        train_probs = set(unique_probs[train_prob_idx])
        test_probs = set(unique_probs[test_prob_idx])

        train_mask = np.array([pid in train_probs for pid in problem_ids])
        test_mask = np.array([pid in test_probs for pid in problem_ids])

        if train_mask.sum() < 10 or test_mask.sum() < 3:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])

        mlp = MLPRegressor(
            hidden_layer_sizes=hidden, activation='relu',
            max_iter=1000, early_stopping=True, validation_fraction=0.15,
            random_state=SEED, learning_rate='adaptive', learning_rate_init=0.001,
            n_iter_no_change=20, alpha=0.001,
        )
        mlp.fit(X_train, y[train_mask])
        y_pred[test_mask] = mlp.predict(X_test)

    predicted = ~np.isnan(y_pred)
    if predicted.sum() < 10:
        return 0.0, np.zeros(len(y))
    r, _ = stats.pearsonr(y[predicted], y_pred[predicted])
    y_pred_filled = np.where(predicted, y_pred, 0.0)
    return float(r), y_pred_filled


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    kv_dim = num_kv_heads * head_dim

    print(f"Model loaded in {time.time()-t0:.1f}s")
    print(f"  {n_layers} layers, kv_dim={kv_dim}, num_kv_heads={num_kv_heads}")

    # ── Load GSM8K ──
    problems = load_gsm8k()
    np.random.seed(SEED)
    indices = np.random.permutation(len(problems))[:N_PROBLEMS]
    problems = [problems[i] for i in indices]
    print(f"  {len(problems)} problems selected")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Generate CoT and extract K+V cache at arithmetic "="
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 1: Generate CoT & extract K+V cache at arithmetic positions")
    print(f"{'='*60}")

    all_ops = []  # Each: {prob_idx, offset, k_features:{layer:vec}, v_features:{layer:vec}, target, op_A, op_B, op_type}
    n_correct = 0
    n_total = 0
    n_with_arith = 0
    gen_budget = TIME_BUDGET * 0.55

    for prob_idx, prob in enumerate(problems):
        if time.time() - t0 > gen_budget:
            print(f"  Time budget reached at problem {prob_idx}")
            break

        prompt = build_prompt(prob['question'])
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        prompt_len = prompt_ids.shape[1]

        if prompt_len > MAX_SEQ_LEN - MAX_GEN:
            continue

        with torch.no_grad():
            out = model.generate(
                prompt_ids, max_new_tokens=MAX_GEN, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids_tensor = out[0, prompt_len:]
        gen_ids_list = gen_ids_tensor.tolist()
        gen_text = tokenizer.decode(gen_ids_list, skip_special_tokens=True)

        model_answer = extract_answer(gen_text)
        gold = prob['gold_answer']
        is_correct = False
        if model_answer is not None and gold is not None:
            try:
                is_correct = abs(float(model_answer) - gold) < 0.5
            except ValueError:
                pass

        if is_correct:
            n_correct += 1
        n_total += 1

        parse_text = gen_text
        hash_pos = parse_text.find('####')
        if hash_pos > 0:
            parse_text = parse_text[:hash_pos]

        arith_ops = parse_arithmetic(parse_text)
        if len(arith_ops) == 0:
            del out
            torch.cuda.empty_cache()
            continue

        eq_positions = []
        valid_ops = []
        for op_info in arith_ops:
            tok_pos = map_eq_to_token(gen_text, gen_ids_list, tokenizer, op_info['eq_char_pos'])
            if tok_pos is not None and tok_pos < len(gen_ids_list):
                eq_positions.append(tok_pos)
                valid_ops.append(op_info)

        if len(valid_ops) == 0:
            del out
            torch.cuda.empty_cache()
            continue

        n_with_arith += 1

        # Forward pass to get KV cache
        full_ids = out[0:1, :]
        try:
            with torch.no_grad():
                model_out = model(full_ids, use_cache=True)
            past_kv = model_out.past_key_values
        except RuntimeError as e:
            print(f"  OOM at problem {prob_idx}, skipping: {e}")
            del out
            torch.cuda.empty_cache()
            continue

        for op_i, (op_info, eq_tok) in enumerate(zip(valid_ops, eq_positions)):
            full_eq_pos = prompt_len + eq_tok

            for offset in POSITION_OFFSETS:
                full_pos = full_eq_pos + offset
                chain_tok_pos = eq_tok + offset

                if chain_tok_pos < 0 or full_pos < prompt_len or full_pos >= full_ids.shape[1]:
                    continue

                k_features = {}
                v_features = {}
                for layer in PROBE_LAYERS:
                    K, V = get_kv(past_kv, layer)
                    k_vec = K[0, :, full_pos, :].reshape(-1).float().cpu().numpy()
                    v_vec = V[0, :, full_pos, :].reshape(-1).float().cpu().numpy()
                    k_features[layer] = k_vec
                    v_features[layer] = v_vec

                target = signed_log(op_info['correct_result'])
                A = op_info['operands'][0] if len(op_info['operands']) >= 1 else 0.0
                B = op_info['operands'][1] if len(op_info['operands']) >= 2 else 0.0

                all_ops.append({
                    'prob_idx': prob_idx,
                    'offset': offset,
                    'k_features': k_features,
                    'v_features': v_features,
                    'target': target,
                    'target_A': signed_log(A),
                    'target_B': signed_log(B),
                    'op_type': op_info['op_type'],
                    'operands': op_info['operands'],
                })

        # Clean up
        del past_kv, model_out
        del out
        torch.cuda.empty_cache()

        if prob_idx % 50 == 0 and prob_idx > 0:
            acc = n_correct / n_total * 100 if n_total > 0 else 0
            n_ops_so_far = len([o for o in all_ops if o['offset'] == 0])
            print(f"  Problem {prob_idx}: {n_total} total, {n_correct} correct ({acc:.1f}%), "
                  f"{n_with_arith} with arith, {n_ops_so_far} ops at offset=0")

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    acc = n_correct / n_total * 100 if n_total > 0 else 0
    print(f"\n  Generation complete: {n_total} problems, {n_correct} correct ({acc:.1f}%)")
    print(f"  {n_with_arith} problems with arithmetic, {len(all_ops)} total op×offset entries")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Probing analysis — K vs V at computation positions
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: K vs V probing at computation positions")
    print(f"{'='*60}")

    results = {}

    for offset in POSITION_OFFSETS:
        ops_at_offset = [o for o in all_ops if o['offset'] == offset]
        n_ops = len(ops_at_offset)
        print(f"\n--- Offset {offset}: {n_ops} operations ---")

        if n_ops < 50:
            print(f"  Skipping: too few operations")
            continue

        prob_ids = np.array([o['prob_idx'] for o in ops_at_offset])
        targets_result = np.array([o['target'] for o in ops_at_offset])
        targets_A = np.array([o['target_A'] for o in ops_at_offset])
        targets_B = np.array([o['target_B'] for o in ops_at_offset])

        # Operand features for MLP baseline
        op_feats = np.array([operand_features(o['operands'], o['op_type']) for o in ops_at_offset])

        # MLP baseline (same as exp 108)
        mlp_R, mlp_pred = run_mlp_cv(op_feats, targets_result, prob_ids)
        print(f"  MLP baseline (operands→result): R={mlp_R:.3f}")

        offset_results = {}

        for layer in PROBE_LAYERS:
            # Build K and V feature matrices
            K_mat = np.array([o['k_features'][layer] for o in ops_at_offset])
            V_mat = np.array([o['v_features'][layer] for o in ops_at_offset])

            # ── Probe for RESULT ──
            k_result_R, k_result_pred = run_probe_cv(K_mat, targets_result, prob_ids)
            v_result_R, v_result_pred = run_probe_cv(V_mat, targets_result, prob_ids)

            # ── Probe for OPERAND A ──
            k_A_R, _ = run_probe_cv(K_mat, targets_A, prob_ids)
            v_A_R, _ = run_probe_cv(V_mat, targets_A, prob_ids)

            # ── Probe for OPERAND B ──
            k_B_R, _ = run_probe_cv(K_mat, targets_B, prob_ids)
            v_B_R, _ = run_probe_cv(V_mat, targets_B, prob_ids)

            # ── Residualized: unique info beyond MLP baseline ──
            # K|MLP: probe K for residual(target - MLP_pred)
            residual = targets_result - mlp_pred
            if np.std(residual) > 1e-6:
                k_mlp_R, _ = run_probe_cv(K_mat, residual, prob_ids)
                v_mlp_R, _ = run_probe_cv(V_mat, residual, prob_ids)
            else:
                k_mlp_R = 0.0
                v_mlp_R = 0.0

            # Computation transition metrics
            k_transition = k_result_R - k_A_R
            v_transition = v_result_R - v_A_R

            layer_res = {
                'K_result': k_result_R,
                'V_result': v_result_R,
                'K_A': k_A_R,
                'V_A': v_A_R,
                'K_B': k_B_R,
                'V_B': v_B_R,
                'K_MLP': k_mlp_R,
                'V_MLP': v_mlp_R,
                'K_transition': k_transition,
                'V_transition': v_transition,
                'VK_result_gap': v_result_R - k_result_R,
                'VK_A_gap': v_A_R - k_A_R,
            }
            offset_results[f'L{layer}'] = layer_res

            print(f"  L{layer}: K→result={k_result_R:.3f}, V→result={v_result_R:.3f} (V-K={v_result_R-k_result_R:+.3f})")
            print(f"         K→A={k_A_R:.3f}, V→A={v_A_R:.3f} | K→B={k_B_R:.3f}, V→B={v_B_R:.3f}")
            print(f"         K|MLP={k_mlp_R:.3f}, V|MLP={v_mlp_R:.3f}")
            print(f"         K_trans={k_transition:+.3f}, V_trans={v_transition:+.3f}")

        results[f'offset_{offset}'] = {
            'n_ops': n_ops,
            'mlp_R': mlp_R,
            'layers': offset_results,
        }

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Operation-type breakdown (offset=0 only)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 3: Operation-type breakdown (K vs V)")
    print(f"{'='*60}")

    ops_off0 = [o for o in all_ops if o['offset'] == 0]
    op_types = {}
    for o in ops_off0:
        ot = o['op_type']
        if ot not in op_types:
            op_types[ot] = []
        op_types[ot].append(o)

    op_type_results = {}
    for ot in sorted(op_types.keys()):
        ops = op_types[ot]
        n = len(ops)
        if n < 30:
            print(f"  {ot}: n={n}, too few — skipping")
            continue

        prob_ids = np.array([o['prob_idx'] for o in ops])
        targets = np.array([o['target'] for o in ops])

        layer = 27  # Use L27 (best layer from exp 108)
        K_mat = np.array([o['k_features'][layer] for o in ops])
        V_mat = np.array([o['v_features'][layer] for o in ops])

        k_R, _ = run_probe_cv(K_mat, targets, prob_ids)
        v_R, _ = run_probe_cv(V_mat, targets, prob_ids)

        # MLP baseline for this op type
        op_feats = np.array([operand_features(o['operands'], o['op_type']) for o in ops])
        mlp_R, _ = run_mlp_cv(op_feats, targets, prob_ids)

        op_type_results[ot] = {'n': n, 'K_R': k_R, 'V_R': v_R, 'MLP_R': mlp_R, 'VK_gap': v_R - k_R}
        print(f"  {ot}: n={n}, K={k_R:.3f}, V={v_R:.3f}, MLP={mlp_R:.3f}, V-K={v_R-k_R:+.3f}")

    results['op_type_breakdown'] = op_type_results

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Summary statistics and save
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Summary table for offset=0
    off0 = results.get('offset_0', {})
    off0_layers = off0.get('layers', {})

    print(f"\n{'='*60}")
    print("Offset=0 K vs V comparison:")
    print(f"{'='*60}")
    print(f"{'Layer':>6} | {'K→res':>7} {'V→res':>7} {'V-K':>7} | {'K→A':>7} {'V→A':>7} | {'K|MLP':>7} {'V|MLP':>7} | {'K_tr':>7} {'V_tr':>7}")
    print("-" * 95)

    vk_result_gaps = []
    k_transitions = []
    v_transitions = []
    v_gt_k_result_count = 0

    for layer in PROBE_LAYERS:
        key = f'L{layer}'
        if key not in off0_layers:
            continue
        d = off0_layers[key]
        gap = d['VK_result_gap']
        vk_result_gaps.append(gap)
        k_transitions.append(d['K_transition'])
        v_transitions.append(d['V_transition'])
        if gap > 0:
            v_gt_k_result_count += 1
        print(f"  L{layer:>3} | {d['K_result']:>7.3f} {d['V_result']:>7.3f} {gap:>+7.3f} | "
              f"{d['K_A']:>7.3f} {d['V_A']:>7.3f} | {d['K_MLP']:>7.3f} {d['V_MLP']:>7.3f} | "
              f"{d['K_transition']:>+7.3f} {d['V_transition']:>+7.3f}")

    if vk_result_gaps:
        mean_vk_gap = np.mean(vk_result_gaps)
        print(f"\n  Mean V-K result gap: {mean_vk_gap:+.3f}")
        print(f"  V>K result at {v_gt_k_result_count}/{len(vk_result_gaps)} layers")

    # Offset comparison
    print(f"\n{'='*60}")
    print("Offset comparison (L27, result probing):")
    print(f"{'='*60}")
    for offset in POSITION_OFFSETS:
        key = f'offset_{offset}'
        if key in results and 'layers' in results[key] and 'L27' in results[key]['layers']:
            d = results[key]['layers']['L27']
            print(f"  Offset {offset:>3}: K→res={d['K_result']:.3f}, V→res={d['V_result']:.3f}, "
                  f"V-K={d['VK_result_gap']:+.3f}")

    # ── Save results ──
    save_results = {
        'model': MODEL_NAME,
        'n_problems_total': n_total,
        'n_correct': n_correct,
        'accuracy': acc,
        'n_with_arith': n_with_arith,
        'probe_layers': PROBE_LAYERS,
        'offsets': POSITION_OFFSETS,
        'kv_dim': kv_dim,
        'results': {},
        'op_type_breakdown': op_type_results,
    }
    # Convert results for JSON (nested)
    for off_key, off_val in results.items():
        if off_key == 'op_type_breakdown':
            continue
        save_results['results'][off_key] = off_val

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Figures
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 5: Generating figures")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Figure 1: K vs V result probing across layers (offset=0) ──
    if off0_layers:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        layers_plot = [l for l in PROBE_LAYERS if f'L{l}' in off0_layers]
        k_results = [off0_layers[f'L{l}']['K_result'] for l in layers_plot]
        v_results = [off0_layers[f'L{l}']['V_result'] for l in layers_plot]
        mlp_val = off0.get('mlp_R', 0)

        x = np.arange(len(layers_plot))
        w = 0.35

        # Panel A: Result probing
        ax = axes[0]
        bars_k = ax.bar(x - w/2, k_results, w, label='K→result', color='#2196F3', alpha=0.8)
        bars_v = ax.bar(x + w/2, v_results, w, label='V→result', color='#F44336', alpha=0.8)
        ax.axhline(mlp_val, color='gray', linestyle='--', alpha=0.7, label=f'MLP baseline ({mlp_val:.3f})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers_plot])
        ax.set_ylabel('Pearson R')
        ax.set_title('K vs V: Result Probing at "=" (offset=0)')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)

        # Panel B: Operand probing
        ax = axes[1]
        k_A = [off0_layers[f'L{l}']['K_A'] for l in layers_plot]
        v_A = [off0_layers[f'L{l}']['V_A'] for l in layers_plot]
        k_B = [off0_layers[f'L{l}']['K_B'] for l in layers_plot]
        v_B = [off0_layers[f'L{l}']['V_B'] for l in layers_plot]

        ax.plot(x, k_A, 'o-', color='#2196F3', label='K→A', linewidth=2)
        ax.plot(x, v_A, 's-', color='#F44336', label='V→A', linewidth=2)
        ax.plot(x, k_B, 'o--', color='#2196F3', alpha=0.5, label='K→B')
        ax.plot(x, v_B, 's--', color='#F44336', alpha=0.5, label='V→B')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers_plot])
        ax.set_ylabel('Pearson R')
        ax.set_title('K vs V: Operand Probing at "=" (offset=0)')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'kv_result_operand_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved kv_result_operand_comparison.png")

    # ── Figure 2: Computation transition K vs V ──
    if off0_layers:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        layers_plot = [l for l in PROBE_LAYERS if f'L{l}' in off0_layers]

        # Panel A: K transition (K→result vs K→A)
        ax = axes[0]
        k_res = [off0_layers[f'L{l}']['K_result'] for l in layers_plot]
        k_a = [off0_layers[f'L{l}']['K_A'] for l in layers_plot]
        ax.plot(range(len(layers_plot)), k_res, 'o-', color='#2196F3', linewidth=2, label='K→result')
        ax.plot(range(len(layers_plot)), k_a, 's-', color='#64B5F6', linewidth=2, label='K→A')
        ax.fill_between(range(len(layers_plot)), k_a, k_res, alpha=0.2, color='#2196F3')
        ax.set_xticks(range(len(layers_plot)))
        ax.set_xticklabels([f'L{l}' for l in layers_plot])
        ax.set_ylabel('Pearson R')
        ax.set_title('K: Computation Transition')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)

        # Panel B: V transition (V→result vs V→A)
        ax = axes[1]
        v_res = [off0_layers[f'L{l}']['V_result'] for l in layers_plot]
        v_a = [off0_layers[f'L{l}']['V_A'] for l in layers_plot]
        ax.plot(range(len(layers_plot)), v_res, 'o-', color='#F44336', linewidth=2, label='V→result')
        ax.plot(range(len(layers_plot)), v_a, 's-', color='#EF9A9A', linewidth=2, label='V→A')
        ax.fill_between(range(len(layers_plot)), v_a, v_res, alpha=0.2, color='#F44336')
        ax.set_xticks(range(len(layers_plot)))
        ax.set_xticklabels([f'L{l}' for l in layers_plot])
        ax.set_ylabel('Pearson R')
        ax.set_title('V: Computation Transition')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'computation_transition_kv.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved computation_transition_kv.png")

    # ── Figure 3: K|MLP vs V|MLP (unique info beyond operands) ──
    if off0_layers:
        fig, ax = plt.subplots(figsize=(8, 5))

        layers_plot = [l for l in PROBE_LAYERS if f'L{l}' in off0_layers]
        k_mlp = [off0_layers[f'L{l}']['K_MLP'] for l in layers_plot]
        v_mlp = [off0_layers[f'L{l}']['V_MLP'] for l in layers_plot]

        x = np.arange(len(layers_plot))
        w = 0.35
        ax.bar(x - w/2, k_mlp, w, label='K|MLP', color='#2196F3', alpha=0.8)
        ax.bar(x + w/2, v_mlp, w, label='V|MLP', color='#F44336', alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers_plot])
        ax.set_ylabel('Pearson R (residualized)')
        ax.set_title('Unique Info Beyond Nonlinear Operand Baseline')
        ax.legend(fontsize=10)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'kv_unique_info_mlp.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved kv_unique_info_mlp.png")

    # ── Figure 4: Offset profiles K vs V ──
    offsets_with_data = [o for o in POSITION_OFFSETS if f'offset_{o}' in results and 'layers' in results[f'offset_{o}']]
    if len(offsets_with_data) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))

        for layer in [27, 35]:
            k_vals = []
            v_vals = []
            offs = []
            for offset in offsets_with_data:
                key = f'offset_{offset}'
                lkey = f'L{layer}'
                if lkey in results[key]['layers']:
                    k_vals.append(results[key]['layers'][lkey]['K_result'])
                    v_vals.append(results[key]['layers'][lkey]['V_result'])
                    offs.append(offset)

            if len(offs) > 1:
                ax.plot(offs, k_vals, 'o-', color='#2196F3', linewidth=2,
                        label=f'K→result L{layer}', alpha=0.7 if layer == 35 else 1.0)
                ax.plot(offs, v_vals, 's-', color='#F44336', linewidth=2,
                        label=f'V→result L{layer}', alpha=0.7 if layer == 35 else 1.0)

        ax.set_xlabel('Offset from "="')
        ax.set_ylabel('Pearson R')
        ax.set_title('K vs V Forward-Looking Profiles')
        ax.legend(fontsize=9)
        ax.invert_xaxis()

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'kv_offset_profiles.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved kv_offset_profiles.png")

    # ── Figure 5: Op-type K vs V ──
    if op_type_results:
        fig, ax = plt.subplots(figsize=(10, 5))

        op_labels = sorted(op_type_results.keys())
        k_vals = [op_type_results[ot]['K_R'] for ot in op_labels]
        v_vals = [op_type_results[ot]['V_R'] for ot in op_labels]
        mlp_vals = [op_type_results[ot]['MLP_R'] for ot in op_labels]
        ns = [op_type_results[ot]['n'] for ot in op_labels]

        x = np.arange(len(op_labels))
        w = 0.25
        ax.bar(x - w, k_vals, w, label='K→result', color='#2196F3', alpha=0.8)
        ax.bar(x, v_vals, w, label='V→result', color='#F44336', alpha=0.8)
        ax.bar(x + w, mlp_vals, w, label='MLP baseline', color='gray', alpha=0.6)
        ax.set_xticks(x)

        display_labels = []
        for ot, n in zip(op_labels, ns):
            sym = {'+': '+', '-': '-', '*': '*', '/': '/', 'multi': 'multi'}
            display_labels.append(f"{sym.get(ot, ot)}\n(n={n})")
        ax.set_xticklabels(display_labels)
        ax.set_ylabel('Pearson R (L27)')
        ax.set_title('K vs V by Operation Type (L27, offset=0)')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'kv_op_type_breakdown.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved kv_op_type_breakdown.png")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE. Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
