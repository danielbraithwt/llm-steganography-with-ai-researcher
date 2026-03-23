#!/usr/bin/env python3
"""
Experiment 108: Nonlinear Operand Baseline Challenge for Intermediate Value Probe

CHALLENGE EXPERIMENT: Tests whether V at "=" merely encodes operands (from which
a probe can compute the result) or carries pre-computed results beyond operand info.

Key analyses:
1. Baseline ladder: text_linear < operand_linear < operand_MLP < oracle vs V
2. V|operand_MLP: V's unique info beyond what operands+op can provide
3. Operand probing: V→A, V→B vs V→result
4. Offset analysis: operand visibility at -5/-10
5. Op-type breakdown of MLP vs V
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
POSITION_OFFSETS = [0, -5, -10, -20]

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_108"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same as exp 106) ──
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

            # Determine operation type
            ops_found = re.findall(r'[+\-*/]', expr_str)
            if len(ops_found) == 1:
                op_type = ops_found[0]
            else:
                op_type = 'multi'

            # Extract operands
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


def extract_chain_numbers(text):
    """Extract all numbers from text."""
    nums = re.findall(r'\d[\d,]*(?:\.\d+)?', text)
    return [float(n.replace(',', '')) for n in nums]


def numbers_to_features(numbers, max_dim=MAX_NUMS_DIM):
    """Convert list of numbers to fixed-size feature vector (log-transformed)."""
    log_nums = [signed_log(n) for n in numbers[-max_dim:]]
    feat = np.zeros(max_dim)
    feat[:len(log_nums)] = log_nums
    return feat


def operand_features(operands, op_type):
    """Create feature vector from operands and operation type.
    Features: [signed_log(A), signed_log(B), op_+, op_-, op_*, op_/]
    """
    A = operands[0] if len(operands) >= 1 else 0.0
    B = operands[1] if len(operands) >= 2 else 0.0

    feat = np.zeros(6)
    feat[0] = signed_log(A)
    feat[1] = signed_log(B)
    op_map = {'+': 2, '-': 3, '*': 4, '/': 5}
    if op_type in op_map:
        feat[op_map[op_type]] = 1.0
    return feat


def oracle_features(operands, op_type):
    """Oracle features: includes pre-computed results for all operations.
    Features: [signed_log(A), signed_log(B), op_onehot(4),
               signed_log(A+B), signed_log(A-B), signed_log(A*B), signed_log(A/B)]
    """
    A = operands[0] if len(operands) >= 1 else 0.0
    B = operands[1] if len(operands) >= 2 else 0.0

    feat = np.zeros(10)
    feat[0] = signed_log(A)
    feat[1] = signed_log(B)
    op_map = {'+': 2, '-': 3, '*': 4, '/': 5}
    if op_type in op_map:
        feat[op_map[op_type]] = 1.0

    feat[6] = signed_log(A + B)
    feat[7] = signed_log(A - B)
    feat[8] = signed_log(A * B)
    if abs(B) > 1e-10:
        feat[9] = signed_log(A / B)
    return feat


def run_probe_problem_cv(X, y, problem_ids, n_splits=N_FOLDS):
    """Cross-validate Ridge at problem level to prevent leakage."""
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


def run_mlp_problem_cv(X, y, problem_ids, hidden=(128, 64), n_splits=N_FOLDS):
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


def check_operand_visibility(gen_text, gen_ids_list, tokenizer, chain_tok_pos, operands, char_start):
    """Check which operands are visible in text up to chain_tok_pos.
    Returns (A_visible, B_visible) booleans.
    """
    if chain_tok_pos < 0:
        return False, False

    visible_text = tokenizer.decode(gen_ids_list[:chain_tok_pos + 1], skip_special_tokens=True)

    # The expression starts at char_start in gen_text.
    # We need to check if the expression text is within the visible text.
    # Since visible_text corresponds to tokens 0..chain_tok_pos, we check
    # whether operand numbers appear in the visible text near the expected position.

    A = operands[0] if len(operands) >= 1 else None
    B = operands[1] if len(operands) >= 2 else None

    # Check if the expression start is within visible text
    # We use a heuristic: search for the operand values in the visible text
    # near the end (within last 100 chars to avoid false matches from earlier)
    search_region = visible_text  # search full visible text

    A_visible = False
    B_visible = False

    if A is not None:
        # Format A as it would appear in text
        a_str = str(int(A)) if A == int(A) else str(A)
        # Also check with commas (e.g. "80,000")
        a_str_comma = "{:,}".format(int(A)) if A == int(A) and A >= 1000 else None
        if a_str in search_region or (a_str_comma and a_str_comma in search_region):
            A_visible = True

    if B is not None:
        b_str = str(int(B)) if B == int(B) else str(B)
        b_str_comma = "{:,}".format(int(B)) if B == int(B) and B >= 1000 else None
        if b_str in search_region or (b_str_comma and b_str_comma in search_region):
            B_visible = True

    return A_visible, B_visible


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
    print(f"  {n_layers} layers, kv_dim={kv_dim}")

    # ── Load GSM8K ──
    problems = load_gsm8k()
    np.random.seed(SEED)
    indices = np.random.permutation(len(problems))[:N_PROBLEMS]
    problems = [problems[i] for i in indices]
    print(f"  {len(problems)} problems selected")

    # ═════════════════════════════════════════════════════════════════
    # PHASE 1: Generate CoT and extract V-cache at arithmetic positions
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 1: Generate CoT & extract V-cache at arithmetic positions")
    print(f"{'='*60}")

    all_ops = []
    prompt_v_cache = {}
    n_correct = 0
    n_total = 0
    n_with_arith = 0
    gen_budget = TIME_BUDGET * 0.65

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

        # Extract V at last prompt token (prompt control)
        prompt_v = {}
        for layer in PROBE_LAYERS:
            K, V = get_kv(past_kv, layer)
            v_vec = V[0, :, prompt_len - 1, :].reshape(-1).float().cpu().numpy()
            prompt_v[layer] = v_vec
        prompt_v_cache[prob_idx] = prompt_v

        for op_i, (op_info, eq_tok) in enumerate(zip(valid_ops, eq_positions)):
            full_eq_pos = prompt_len + eq_tok

            for offset in POSITION_OFFSETS:
                full_pos = full_eq_pos + offset
                chain_tok_pos = eq_tok + offset

                if chain_tok_pos < 0 or full_pos < prompt_len or full_pos >= full_ids.shape[1]:
                    continue

                v_features = {}
                for layer in PROBE_LAYERS:
                    K, V = get_kv(past_kv, layer)
                    v_vec = V[0, :, full_pos, :].reshape(-1).float().cpu().numpy()
                    v_features[layer] = v_vec

                # Text baseline: numbers visible in chain up to this position
                visible_text = tokenizer.decode(gen_ids_list[:chain_tok_pos + 1], skip_special_tokens=True)
                visible_nums = extract_chain_numbers(visible_text)
                text_feat = numbers_to_features(visible_nums)

                # Operand visibility check
                A_vis, B_vis = check_operand_visibility(
                    gen_text, gen_ids_list, tokenizer, chain_tok_pos,
                    op_info['operands'], op_info['char_start'])

                all_ops.append({
                    'problem_idx': prob_idx,
                    'op_idx': op_i,
                    'offset': offset,
                    'written_result': op_info['written_result'],
                    'correct_result': op_info['correct_result'],
                    'op_type': op_info['op_type'],
                    'operands': op_info['operands'],
                    'gold_answer': gold if gold is not None else 0.0,
                    'is_correct': is_correct,
                    'chain_fraction': chain_tok_pos / max(len(gen_ids_list), 1),
                    'v_features': v_features,
                    'text_features': text_feat,
                    'A_visible': A_vis,
                    'B_visible': B_vis,
                })

        del past_kv, model_out, out
        torch.cuda.empty_cache()

        if (prob_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            n_ops_at_eq = sum(1 for o in all_ops if o['offset'] == 0)
            print(f"  Problem {prob_idx+1}: {n_total} gen, {n_correct} correct "
                  f"({100*n_correct/max(n_total,1):.1f}%), {n_with_arith} w/arith, "
                  f"{n_ops_at_eq} ops@eq, {elapsed:.0f}s")

    print(f"\nGeneration complete: {n_total} problems, {n_correct} correct "
          f"({100*n_correct/max(n_total,1):.1f}%)")
    n_ops_at_eq = sum(1 for o in all_ops if o['offset'] == 0)
    print(f"Problems with arithmetic: {n_with_arith}")
    print(f"Total operation-offset entries: {len(all_ops)}")
    print(f"Operations at offset=0: {n_ops_at_eq}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    if n_ops_at_eq < 50:
        print("ERROR: Too few arithmetic operations. Aborting.")
        sys.exit(1)

    # ═════════════════════════════════════════════════════════════════
    # PHASE 2: Probing Analysis with Nonlinear Baselines
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: Probing Analysis with Nonlinear Operand Baselines")
    print(f"{'='*60}")

    from scipy import stats

    # ── Main analysis at offset=0 ──
    ops_eq = [o for o in all_ops if o['offset'] == 0]
    targets = np.array([signed_log(o['written_result']) for o in ops_eq])
    problem_ids = np.array([o['problem_idx'] for o in ops_eq])
    op_types = [o['op_type'] for o in ops_eq]

    # Text baseline (exp 106 replication)
    text_X = np.vstack([o['text_features'] for o in ops_eq])
    text_R, text_pred = run_probe_problem_cv(text_X, targets, problem_ids)
    print(f"\n  text_linear_R:   {text_R:.3f}")

    # Operand linear baseline
    operand_X = np.vstack([operand_features(o['operands'], o['op_type']) for o in ops_eq])
    op_linear_R, op_linear_pred = run_probe_problem_cv(operand_X, targets, problem_ids)
    print(f"  operand_linear_R: {op_linear_R:.3f}")

    # Operand MLP baseline
    op_mlp_R, op_mlp_pred = run_mlp_problem_cv(operand_X, targets, problem_ids, hidden=(128, 64))
    print(f"  operand_MLP_R:    {op_mlp_R:.3f}")

    # Oracle baseline (has pre-computed results)
    oracle_X = np.vstack([oracle_features(o['operands'], o['op_type']) for o in ops_eq])
    oracle_R, oracle_pred = run_probe_problem_cv(oracle_X, targets, problem_ids)
    print(f"  oracle_R:         {oracle_R:.3f}")

    # Operand A target (for operand probing)
    targets_A = np.array([signed_log(o['operands'][0]) if len(o['operands']) >= 1 else 0.0
                          for o in ops_eq])
    targets_B = np.array([signed_log(o['operands'][1]) if len(o['operands']) >= 2 else 0.0
                          for o in ops_eq])

    results = {
        'text_R': text_R, 'operand_linear_R': op_linear_R,
        'operand_MLP_R': op_mlp_R, 'oracle_R': oracle_R,
        'n_ops': len(ops_eq), 'n_problems': len(np.unique(problem_ids)),
        'n_correct': n_correct, 'n_total': n_total,
    }
    layer_results = {}

    print(f"\n  N operations at offset=0: {len(ops_eq)}")
    print(f"  N unique problems: {len(np.unique(problem_ids))}")

    print(f"\n{'='*60}")
    print("LAYER-WISE V PROBING vs BASELINES at offset=0")
    print(f"{'='*60}")
    print(f"{'Layer':>6} | {'V_R':>6} | {'text':>6} | {'op_lin':>6} | {'op_MLP':>6} | {'oracle':>6} | {'V|MLP':>6} | {'V→A':>6} | {'V→B':>6}")
    print("-" * 75)

    for layer in PROBE_LAYERS:
        V_X = np.vstack([o['v_features'][layer] for o in ops_eq])

        # V → result
        V_R, V_pred = run_probe_problem_cv(V_X, targets, problem_ids)

        # V|text (replication)
        resid_text = targets - text_pred
        V_resid_text_R, _ = run_probe_problem_cv(V_X, resid_text, problem_ids)

        # V|operand_linear
        resid_op_lin = targets - op_linear_pred
        V_resid_op_lin_R, _ = run_probe_problem_cv(V_X, resid_op_lin, problem_ids)

        # V|operand_MLP — KEY METRIC
        resid_op_mlp = targets - op_mlp_pred
        V_resid_op_mlp_R, _ = run_probe_problem_cv(V_X, resid_op_mlp, problem_ids)

        # V|oracle
        resid_oracle = targets - oracle_pred
        V_resid_oracle_R, _ = run_probe_problem_cv(V_X, resid_oracle, problem_ids)

        # Prompt control
        prompt_V_X = np.vstack([prompt_v_cache[o['problem_idx']][layer] for o in ops_eq])
        prompt_V_R, _ = run_probe_problem_cv(prompt_V_X, targets, problem_ids)

        # OPERAND PROBING: V → A, V → B
        V_to_A_R, _ = run_probe_problem_cv(V_X, targets_A, problem_ids)
        V_to_B_R, _ = run_probe_problem_cv(V_X, targets_B, problem_ids)

        print(f"  L{layer:>2}  | {V_R:>6.3f} | {text_R:>6.3f} | {op_linear_R:>6.3f} | "
              f"{op_mlp_R:>6.3f} | {oracle_R:>6.3f} | {V_resid_op_mlp_R:>6.3f} | "
              f"{V_to_A_R:>6.3f} | {V_to_B_R:>6.3f}")

        layer_results[str(layer)] = {
            'V_R': V_R, 'V_resid_text': V_resid_text_R,
            'V_resid_op_linear': V_resid_op_lin_R,
            'V_resid_op_MLP': V_resid_op_mlp_R,
            'V_resid_oracle': V_resid_oracle_R,
            'prompt_V_R': prompt_V_R,
            'V_to_A_R': V_to_A_R, 'V_to_B_R': V_to_B_R,
        }

    results['layers'] = layer_results

    # ═════════════════════════════════════════════════════════════════
    # OPERATION-TYPE BREAKDOWN: MLP vs V
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("OPERATION-TYPE BREAKDOWN: MLP vs V at offset=0")
    print(f"{'='*60}")

    op_type_results = {}
    for ot in ['+', '-', '*', '/', 'multi']:
        mask = np.array([o['op_type'] == ot for o in ops_eq])
        n_ot = mask.sum()
        if n_ot < 20:
            continue

        sub_targets = targets[mask]
        sub_pids = problem_ids[mask]
        sub_text_X = text_X[mask]
        sub_op_X = operand_X[mask]

        sub_text_R, _ = run_probe_problem_cv(sub_text_X, sub_targets, sub_pids)

        sub_op_lin_R, _ = run_probe_problem_cv(sub_op_X, sub_targets, sub_pids)
        sub_op_mlp_R, _ = run_mlp_problem_cv(sub_op_X, sub_targets, sub_pids, hidden=(64, 32))

        # V probe per operation
        layer = 35  # use deepest layer
        sub_V_X = np.vstack([ops_eq[i]['v_features'][layer] for i in range(len(ops_eq)) if mask[i]])
        sub_V_R, _ = run_probe_problem_cv(sub_V_X, sub_targets, sub_pids)

        print(f"  {ot:>5} (n={n_ot:>4}): V_R={sub_V_R:.3f}, text_R={sub_text_R:.3f}, "
              f"op_lin_R={sub_op_lin_R:.3f}, op_MLP_R={sub_op_mlp_R:.3f}, "
              f"V−MLP={sub_V_R - sub_op_mlp_R:+.3f}")

        op_type_results[ot] = {
            'n': int(n_ot), 'V_R': sub_V_R, 'text_R': sub_text_R,
            'op_linear_R': sub_op_lin_R, 'op_MLP_R': sub_op_mlp_R,
        }

    results['op_types'] = op_type_results

    # ═════════════════════════════════════════════════════════════════
    # OFFSET ANALYSIS: Operand visibility and V vs MLP
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("OFFSET ANALYSIS: Operand visibility & V vs MLP")
    print(f"{'='*60}")

    offset_results = {}
    for offset in POSITION_OFFSETS:
        ops_off = [o for o in all_ops if o['offset'] == offset]
        if len(ops_off) < 50:
            continue

        n_ops = len(ops_off)
        n_A_vis = sum(1 for o in ops_off if o['A_visible'])
        n_B_vis = sum(1 for o in ops_off if o['B_visible'])
        n_both_vis = sum(1 for o in ops_off if o['A_visible'] and o['B_visible'])

        off_targets = np.array([signed_log(o['written_result']) for o in ops_off])
        off_pids = np.array([o['problem_idx'] for o in ops_off])
        off_text_X = np.vstack([o['text_features'] for o in ops_off])

        off_text_R, off_text_pred = run_probe_problem_cv(off_text_X, off_targets, off_pids)

        # Full operand baseline (uses all operands — oracle-like at this offset)
        off_op_X = np.vstack([operand_features(o['operands'], o['op_type']) for o in ops_off])
        off_op_mlp_R, off_op_mlp_pred = run_mlp_problem_cv(off_op_X, off_targets, off_pids, hidden=(128, 64))

        # Visible-operand MLP: only use operands that are actually visible
        off_vis_X = []
        for o in ops_off:
            feat = np.zeros(6)
            if o['A_visible'] and len(o['operands']) >= 1:
                feat[0] = signed_log(o['operands'][0])
            if o['B_visible'] and len(o['operands']) >= 2:
                feat[1] = signed_log(o['operands'][1])
            # Op type visible if A is visible (op follows A in expression)
            if o['A_visible']:
                op_map = {'+': 2, '-': 3, '*': 4, '/': 5}
                if o['op_type'] in op_map:
                    feat[op_map[o['op_type']]] = 1.0
            off_vis_X.append(feat)
        off_vis_X = np.vstack(off_vis_X)
        off_vis_mlp_R, off_vis_pred = run_mlp_problem_cv(off_vis_X, off_targets, off_pids, hidden=(128, 64))

        # V probe at this offset (use layer 27 — best from exp 106)
        layer = 27
        off_V_X = np.vstack([o['v_features'][layer] for o in ops_off])
        off_V_R, _ = run_probe_problem_cv(off_V_X, off_targets, off_pids)

        # V|MLP_visible: V unique info beyond visible operands
        resid_vis = off_targets - off_vis_pred
        off_V_resid_vis_R, _ = run_probe_problem_cv(off_V_X, resid_vis, off_pids)

        print(f"  offset={offset:>3}: n={n_ops:>4}, A_vis={100*n_A_vis/n_ops:.0f}%, "
              f"B_vis={100*n_B_vis/n_ops:.0f}%, both={100*n_both_vis/n_ops:.0f}%")
        print(f"           V_R={off_V_R:.3f}, text_R={off_text_R:.3f}, "
              f"MLP_all={off_op_mlp_R:.3f}, MLP_vis={off_vis_mlp_R:.3f}, "
              f"V|MLP_vis={off_V_resid_vis_R:.3f}")

        offset_results[str(offset)] = {
            'n': n_ops, 'A_visible_pct': n_A_vis / n_ops,
            'B_visible_pct': n_B_vis / n_ops, 'both_visible_pct': n_both_vis / n_ops,
            'V_R': off_V_R, 'text_R': off_text_R,
            'MLP_all_R': off_op_mlp_R, 'MLP_visible_R': off_vis_mlp_R,
            'V_resid_MLP_vis_R': off_V_resid_vis_R,
        }

    results['offsets'] = offset_results

    # ═════════════════════════════════════════════════════════════════
    # COMPUTATION vs ENCODING ANALYSIS
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("COMPUTATION vs ENCODING: V→result vs V→operandA vs V→operandB")
    print(f"{'='*60}")
    print(f"  If V encodes operands: V→A ≈ V→B ≈ V→result (all high)")
    print(f"  If V has computed: V→result >> V→A, V→B (operands compressed)")
    print()

    comp_results = {}
    for layer in PROBE_LAYERS:
        V_R_result = layer_results[str(layer)]['V_R']
        V_to_A = layer_results[str(layer)]['V_to_A_R']
        V_to_B = layer_results[str(layer)]['V_to_B_R']

        interpretation = "ENCODING" if min(V_to_A, V_to_B) > V_R_result * 0.85 else "COMPUTATION"
        print(f"  L{layer:>2}: V→result={V_R_result:.3f}, V→A={V_to_A:.3f}, "
              f"V→B={V_to_B:.3f} → {interpretation}")

        comp_results[str(layer)] = {
            'V_to_result': V_R_result, 'V_to_A': V_to_A, 'V_to_B': V_to_B,
        }

    results['computation_vs_encoding'] = comp_results

    # ═════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═════════════════════════════════════════════════════════════════
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

    # ═════════════════════════════════════════════════════════════════
    # FIGURES
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("GENERATING FIGURES")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: Baseline Ladder at offset=0
    fig, axes = plt.subplots(1, len(PROBE_LAYERS), figsize=(16, 5), sharey=True)
    fig.suptitle("Baseline Ladder: V vs Text vs Operand Baselines at '=' (offset=0)", fontsize=13)

    for ax_i, layer in enumerate(PROBE_LAYERS):
        lr = layer_results[str(layer)]
        labels = ['text_lin', 'op_lin', 'op_MLP', 'oracle', 'V']
        values = [text_R, op_linear_R, op_mlp_R, oracle_R, lr['V_R']]
        colors = ['#4472C4', '#5B9BD5', '#70AD47', '#FFC000', '#C00000']

        bars = axes[ax_i].bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)
        axes[ax_i].set_title(f'L{layer}', fontsize=11)
        axes[ax_i].set_ylim(0, 1.05)
        if ax_i == 0:
            axes[ax_i].set_ylabel('Pearson R → result', fontsize=10)
        axes[ax_i].axhline(y=lr['V_R'], color='red', linestyle='--', alpha=0.3)
        axes[ax_i].tick_params(axis='x', rotation=30, labelsize=8)
        for bar, val in zip(bars, values):
            axes[ax_i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'baseline_ladder.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved baseline_ladder.png")

    # Figure 2: V|baseline residuals
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(PROBE_LAYERS))
    width = 0.18

    for i, (label, key) in enumerate([
        ('V|text_lin', 'V_resid_text'),
        ('V|op_lin', 'V_resid_op_linear'),
        ('V|op_MLP', 'V_resid_op_MLP'),
        ('V|oracle', 'V_resid_oracle'),
    ]):
        vals = [layer_results[str(l)][key] for l in PROBE_LAYERS]
        bars = ax.bar(x_pos + i * width, vals, width, label=label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Layer')
    ax.set_ylabel('V|baseline Pearson R')
    ax.set_title('V Residualized Against Each Baseline (V unique info)')
    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels([f'L{l}' for l in PROBE_LAYERS])
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=0.10, color='red', linestyle='--', alpha=0.4, label='threshold')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'v_residualized.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved v_residualized.png")

    # Figure 3: Operand probing (computation vs encoding)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(PROBE_LAYERS))
    width = 0.25

    vals_result = [layer_results[str(l)]['V_R'] for l in PROBE_LAYERS]
    vals_A = [layer_results[str(l)]['V_to_A_R'] for l in PROBE_LAYERS]
    vals_B = [layer_results[str(l)]['V_to_B_R'] for l in PROBE_LAYERS]

    bars1 = ax.bar(x_pos - width, vals_result, width, label='V→result', color='#C00000')
    bars2 = ax.bar(x_pos, vals_A, width, label='V→operand_A', color='#4472C4')
    bars3 = ax.bar(x_pos + width, vals_B, width, label='V→operand_B', color='#70AD47')

    for bars, vals in [(bars1, vals_result), (bars2, vals_A), (bars3, vals_B)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Pearson R')
    ax.set_title('V→result vs V→operands (Computation vs Encoding Test)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'L{l}' for l in PROBE_LAYERS])
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'computation_vs_encoding.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved computation_vs_encoding.png")

    # Figure 4: Offset analysis (V vs MLP_visible)
    if len(offset_results) > 0:
        offsets_sorted = sorted([int(k) for k in offset_results.keys()])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: V_R, MLP_all, MLP_visible, text at each offset
        v_vals = [offset_results[str(o)]['V_R'] for o in offsets_sorted]
        mlp_all = [offset_results[str(o)]['MLP_all_R'] for o in offsets_sorted]
        mlp_vis = [offset_results[str(o)]['MLP_visible_R'] for o in offsets_sorted]
        text_vals = [offset_results[str(o)]['text_R'] for o in offsets_sorted]

        axes[0].plot(offsets_sorted, v_vals, 'rs-', label='V (L27)', linewidth=2, markersize=8)
        axes[0].plot(offsets_sorted, mlp_all, 'g^-', label='MLP (all ops)', linewidth=2, markersize=8)
        axes[0].plot(offsets_sorted, mlp_vis, 'bD-', label='MLP (visible ops)', linewidth=2, markersize=8)
        axes[0].plot(offsets_sorted, text_vals, 'ko--', label='text_linear', linewidth=1.5, markersize=6)
        axes[0].set_xlabel('Offset from "="')
        axes[0].set_ylabel('Pearson R → result')
        axes[0].set_title('V vs Baselines at Each Offset')
        axes[0].legend(fontsize=9)
        axes[0].set_ylim(0, 1.05)

        # Right: Operand visibility at each offset
        a_vis = [100 * offset_results[str(o)]['A_visible_pct'] for o in offsets_sorted]
        b_vis = [100 * offset_results[str(o)]['B_visible_pct'] for o in offsets_sorted]
        both_vis = [100 * offset_results[str(o)]['both_visible_pct'] for o in offsets_sorted]

        axes[1].plot(offsets_sorted, a_vis, 'bs-', label='A visible', linewidth=2)
        axes[1].plot(offsets_sorted, b_vis, 'gs-', label='B visible', linewidth=2)
        axes[1].plot(offsets_sorted, both_vis, 'rs-', label='Both visible', linewidth=2)
        axes[1].set_xlabel('Offset from "="')
        axes[1].set_ylabel('% of operations')
        axes[1].set_title('Operand Visibility by Offset')
        axes[1].legend(fontsize=9)
        axes[1].set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'offset_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved offset_analysis.png")

    # Figure 5: Op-type breakdown (V vs MLP)
    if len(op_type_results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ops = sorted(op_type_results.keys())
        x_pos = np.arange(len(ops))
        width = 0.2

        for i, (label, key, color) in enumerate([
            ('V_R (L35)', 'V_R', '#C00000'),
            ('text_linear', 'text_R', '#4472C4'),
            ('op_linear', 'op_linear_R', '#5B9BD5'),
            ('op_MLP', 'op_MLP_R', '#70AD47'),
        ]):
            vals = [op_type_results[o].get(key, 0) for o in ops]
            bars = ax.bar(x_pos + i * width, vals, width, label=label, color=color)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=7)

        ns = [op_type_results[o]['n'] for o in ops]
        labels = [f'{o}\n(n={n})' for o, n in zip(ops, ns)]
        ax.set_xticks(x_pos + 1.5 * width)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Pearson R → result')
        ax.set_title('V vs Baselines by Operation Type (L35)')
        ax.legend(fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'op_type_v_vs_mlp.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved op_type_v_vs_mlp.png")

    # ═════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nBaseline ladder at offset=0:")
    print(f"  text_linear:     R = {text_R:.3f}")
    print(f"  operand_linear:  R = {op_linear_R:.3f}")
    print(f"  operand_MLP:     R = {op_mlp_R:.3f}")
    print(f"  oracle:          R = {oracle_R:.3f}")
    for layer in PROBE_LAYERS:
        lr = layer_results[str(layer)]
        print(f"  V at L{layer}:        R = {lr['V_R']:.3f}  |  V|op_MLP = {lr['V_resid_op_MLP']:.3f}  |  "
              f"V→A = {lr['V_to_A_R']:.3f}  |  V→B = {lr['V_to_B_R']:.3f}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
