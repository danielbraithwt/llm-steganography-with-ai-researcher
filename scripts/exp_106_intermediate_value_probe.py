#!/usr/bin/env python3
"""
Experiment 106: Intermediate Value Decodability from V-Cache

Core question: Does V-cache at arithmetic '=' positions encode the computation
result BEFORE it appears in the generated text?

For each arithmetic expression "A OP B = C" in CoT traces:
1. Extract V-cache at the "=" token position (before result C is written)
2. Also extract at offsets -5, -10, -20 tokens before "=" (forward-looking test)
3. Train linear probe to predict signed_log(C) from V-cache
4. Compare to text-only baseline (cumulative chain numbers up to position)

Novel: ALL prior Phase 2 probing targeted the FINAL answer. This probes for
INTERMEDIATE step results at exact computation positions.
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
N_PROBLEMS = 600
PROBE_LAYERS = [9, 18, 27, 35]  # 25%, 50%, 75%, 97% depth
N_FOLDS = 5
N_BOOTSTRAP = 200
MAX_NUMS_DIM = 30
POSITION_OFFSETS = [0, -5, -10, -20]  # relative to "=" token

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_106"
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
    # Fill NaN with 0 for downstream residualization
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

    all_ops = []  # each entry: {problem_idx, offset, v_features, k_features, text_features, ...}
    prompt_v_cache = {}  # prob_idx -> {layer: v_vec}
    n_correct = 0
    n_total = 0
    n_with_arith = 0
    gen_budget = TIME_BUDGET * 0.70

    for prob_idx, prob in enumerate(problems):
        if time.time() - t0 > gen_budget:
            print(f"  Time budget reached at problem {prob_idx}")
            break

        prompt = build_prompt(prob['question'])
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        prompt_len = prompt_ids.shape[1]

        if prompt_len > MAX_SEQ_LEN - MAX_GEN:
            continue

        # Generate
        with torch.no_grad():
            out = model.generate(
                prompt_ids, max_new_tokens=MAX_GEN, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids_tensor = out[0, prompt_len:]
        gen_ids_list = gen_ids_tensor.tolist()
        gen_text = tokenizer.decode(gen_ids_list, skip_special_tokens=True)

        # Check correctness
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

        # Parse arithmetic (truncate at ####)
        parse_text = gen_text
        hash_pos = parse_text.find('####')
        if hash_pos > 0:
            parse_text = parse_text[:hash_pos]

        arith_ops = parse_arithmetic(parse_text)
        if len(arith_ops) == 0:
            del out
            torch.cuda.empty_cache()
            continue

        # Map eq positions to token indices
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

        # Forward pass on full sequence to get KV cache
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

        # For each arithmetic operation, extract features at each offset
        for op_i, (op_info, eq_tok) in enumerate(zip(valid_ops, eq_positions)):
            full_eq_pos = prompt_len + eq_tok  # position in full sequence

            for offset in POSITION_OFFSETS:
                full_pos = full_eq_pos + offset
                chain_tok_pos = eq_tok + offset

                # Clip to valid chain range
                if chain_tok_pos < 0 or full_pos < prompt_len or full_pos >= full_ids.shape[1]:
                    continue

                v_features = {}
                k_features = {}
                for layer in PROBE_LAYERS:
                    K, V = get_kv(past_kv, layer)
                    v_vec = V[0, :, full_pos, :].reshape(-1).float().cpu().numpy()
                    k_vec = K[0, :, full_pos, :].reshape(-1).float().cpu().numpy()
                    v_features[layer] = v_vec
                    k_features[layer] = k_vec

                # Text baseline: numbers visible in chain up to this position
                visible_text = tokenizer.decode(gen_ids_list[:chain_tok_pos + 1], skip_special_tokens=True)
                visible_nums = extract_chain_numbers(visible_text)
                text_feat = numbers_to_features(visible_nums)

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
                    'k_features': k_features,
                    'text_features': text_feat,
                })

        # Clean up
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

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    if n_ops_at_eq < 50:
        print("ERROR: Too few arithmetic operations. Aborting.")
        sys.exit(1)

    # ═════════════════════════════════════════════════════════════════
    # PHASE 2: Probing Analysis
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: Probing Analysis")
    print(f"{'='*60}")

    from scipy import stats

    # ── Main analysis: V_R, K_R, text_R at "=" (offset=0) ──
    ops_eq = [o for o in all_ops if o['offset'] == 0]
    targets = np.array([signed_log(o['written_result']) for o in ops_eq])
    problem_ids = np.array([o['problem_idx'] for o in ops_eq])
    gold_answers = np.array([signed_log(o['gold_answer']) for o in ops_eq])
    text_X = np.vstack([o['text_features'] for o in ops_eq])
    op_types = [o['op_type'] for o in ops_eq]

    n_unique_probs = len(np.unique(problem_ids))
    print(f"\nMain analysis at offset=0:")
    print(f"  N operations: {len(ops_eq)}")
    print(f"  N unique problems: {n_unique_probs}")
    print(f"  Target range: {targets.min():.2f} to {targets.max():.2f}")
    print(f"  Target std: {targets.std():.3f}")

    # Text baseline
    text_R, text_pred = run_probe_problem_cv(text_X, targets, problem_ids)
    print(f"\n  Text baseline R: {text_R:.3f}")

    # Gold answer baseline
    gold_X = gold_answers.reshape(-1, 1)
    gold_R, gold_pred = run_probe_problem_cv(gold_X, targets, problem_ids)
    print(f"  Gold answer R: {gold_R:.3f}")

    results = {'text_R': text_R, 'gold_R': gold_R, 'n_ops': len(ops_eq),
               'n_problems': n_unique_probs, 'n_correct': n_correct, 'n_total': n_total}
    layer_results = {}

    for layer in PROBE_LAYERS:
        V_X = np.vstack([o['v_features'][layer] for o in ops_eq])
        K_X = np.vstack([o['k_features'][layer] for o in ops_eq])

        V_R, V_pred = run_probe_problem_cv(V_X, targets, problem_ids)
        K_R, K_pred = run_probe_problem_cv(K_X, targets, problem_ids)

        # V|text: residualize targets against text predictions
        resid_text = targets - text_pred
        V_resid_text_R, _ = run_probe_problem_cv(V_X, resid_text, problem_ids)

        # V|gold: residualize targets against gold answer predictions
        resid_gold = targets - gold_pred
        V_resid_gold_R, _ = run_probe_problem_cv(V_X, resid_gold, problem_ids)

        # Prompt control: V at last prompt token
        prompt_V_X = np.vstack([prompt_v_cache[o['problem_idx']][layer] for o in ops_eq])
        prompt_V_R, _ = run_probe_problem_cv(prompt_V_X, targets, problem_ids)

        layer_results[layer] = {
            'V_R': V_R, 'K_R': K_R, 'V_resid_text': V_resid_text_R,
            'V_resid_gold': V_resid_gold_R, 'prompt_V_R': prompt_V_R,
        }

        print(f"\n  Layer {layer}:")
        print(f"    V_R = {V_R:.3f}   K_R = {K_R:.3f}   text_R = {text_R:.3f}")
        print(f"    V|text = {V_resid_text_R:.3f}   V|gold = {V_resid_gold_R:.3f}")
        print(f"    V_eq - text = {V_R - text_R:+.3f}   V_eq - V_prompt = {V_R - prompt_V_R:+.3f}")
        print(f"    prompt_V_R = {prompt_V_R:.3f}")

    results['layers'] = layer_results

    # ── Operation type breakdown ──
    print(f"\n{'='*60}")
    print("OPERATION TYPE BREAKDOWN (at offset=0)")
    print(f"{'='*60}")

    op_type_results = {}
    for otype in ['+', '-', '*', '/', 'multi']:
        mask = np.array([o == otype for o in op_types])
        n_otype = mask.sum()
        if n_otype < 20:
            print(f"  {otype}: only {n_otype} ops, skipping")
            continue

        t_y = targets[mask]
        t_pids = problem_ids[mask]
        t_text_X = text_X[mask]
        t_text_R, _ = run_probe_problem_cv(t_text_X, t_y, t_pids)

        print(f"\n  Operation '{otype}' (n={n_otype}):")
        otype_layer_results = {}
        for layer in PROBE_LAYERS:
            t_V_X = np.vstack([o['v_features'][layer] for o, m in zip(ops_eq, mask) if m])
            t_V_R, _ = run_probe_problem_cv(t_V_X, t_y, t_pids)
            gap = t_V_R - t_text_R
            print(f"    L{layer}: V_R={t_V_R:.3f}  text_R={t_text_R:.3f}  gap={gap:+.3f}")
            otype_layer_results[layer] = {'V_R': t_V_R, 'text_R': t_text_R, 'gap': gap}
        op_type_results[otype] = {'n': int(n_otype), 'layers': otype_layer_results}

    results['op_types'] = op_type_results

    # ── Position sweep: forward-looking analysis ──
    print(f"\n{'='*60}")
    print("POSITION SWEEP: Forward-looking analysis")
    print(f"{'='*60}")

    sweep_results = {}
    for offset in POSITION_OFFSETS:
        ops_off = [o for o in all_ops if o['offset'] == offset]
        n_off = len(ops_off)
        if n_off < 50:
            print(f"  Offset {offset}: only {n_off} ops, skipping")
            continue

        off_targets = np.array([signed_log(o['written_result']) for o in ops_off])
        off_pids = np.array([o['problem_idx'] for o in ops_off])
        off_text_X = np.vstack([o['text_features'] for o in ops_off])
        off_text_R, _ = run_probe_problem_cv(off_text_X, off_targets, off_pids)

        print(f"\n  Offset {offset} (n={n_off}):")
        offset_layer_results = {}
        for layer in PROBE_LAYERS:
            V_X = np.vstack([o['v_features'][layer] for o in ops_off])
            off_V_R, _ = run_probe_problem_cv(V_X, off_targets, off_pids)
            print(f"    L{layer}: V_R={off_V_R:.3f}  text_R={off_text_R:.3f}  gap={off_V_R-off_text_R:+.3f}")
            offset_layer_results[layer] = {'V_R': off_V_R, 'text_R': off_text_R}
        sweep_results[offset] = {'n': n_off, 'text_R': off_text_R, 'layers': offset_layer_results}

    results['position_sweep'] = sweep_results

    # ── Bootstrap significance at offset=0 ──
    print(f"\n{'='*60}")
    print("BOOTSTRAP SIGNIFICANCE (offset=0)")
    print(f"{'='*60}")

    if time.time() - t0 < TIME_BUDGET * 0.90:
        boot_results = {}
        for layer in PROBE_LAYERS:
            if time.time() - t0 > TIME_BUDGET * 0.95:
                print(f"  Time budget reached, stopping bootstrap at L{layer}")
                break

            V_X = np.vstack([o['v_features'][layer] for o in ops_eq])
            observed_R = layer_results[layer]['V_R']

            null_Rs = []
            for b in range(N_BOOTSTRAP):
                if time.time() - t0 > TIME_BUDGET * 0.95:
                    break
                perm = np.random.permutation(len(targets))
                perm_R, _ = run_probe_problem_cv(V_X, targets[perm], problem_ids)
                null_Rs.append(perm_R)

            if len(null_Rs) > 10:
                p_val = np.mean([nr >= observed_R for nr in null_Rs])
                null_mean = np.mean(null_Rs)
                null_std = np.std(null_Rs)
                z_score = (observed_R - null_mean) / max(null_std, 1e-6)
                print(f"  L{layer}: observed={observed_R:.3f}, null_mean={null_mean:.3f}±{null_std:.3f}, "
                      f"p={p_val:.4f}, z={z_score:.1f} ({len(null_Rs)} perms)")
                boot_results[layer] = {
                    'observed': observed_R, 'null_mean': null_mean, 'null_std': null_std,
                    'p': p_val, 'z': z_score, 'n_perms': len(null_Rs),
                }
            else:
                print(f"  L{layer}: insufficient permutations ({len(null_Rs)})")

        results['bootstrap'] = boot_results
    else:
        print("  Skipped — time budget")

    # ═════════════════════════════════════════════════════════════════
    # PHASE 3: Figures
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 3: Generating Figures")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: V_R vs text_R vs K_R at offset=0, by layer
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(PROBE_LAYERS))
    w = 0.22
    v_vals = [layer_results[l]['V_R'] for l in PROBE_LAYERS]
    k_vals = [layer_results[l]['K_R'] for l in PROBE_LAYERS]
    text_vals = [text_R] * len(PROBE_LAYERS)
    prompt_vals = [layer_results[l]['prompt_V_R'] for l in PROBE_LAYERS]

    bars_v = ax.bar(x - 1.5*w, v_vals, w, label='V at "="', color='#2196F3')
    bars_k = ax.bar(x - 0.5*w, k_vals, w, label='K at "="', color='#FF9800')
    bars_t = ax.bar(x + 0.5*w, text_vals, w, label='Text (nums)', color='#4CAF50')
    bars_p = ax.bar(x + 1.5*w, prompt_vals, w, label='V at prompt', color='#9C27B0')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Pearson R with intermediate result')
    ax.set_title(f'Intermediate Value Decodability at "=" Position (n={len(ops_eq)} ops)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in PROBE_LAYERS])
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0, color='black', linewidth=0.5)

    for bars in [bars_v, bars_k, bars_t, bars_p]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v_vs_text_at_eq.png', dpi=150)
    plt.close(fig)
    print("  Saved v_vs_text_at_eq.png")

    # Figure 2: Position sweep (forward-looking)
    fig, ax = plt.subplots(figsize=(10, 6))
    offsets_with_data = sorted([o for o in sweep_results.keys()], reverse=False)
    for layer in PROBE_LAYERS:
        vr_vals = []
        valid_offsets = []
        for off in offsets_with_data:
            if layer in sweep_results[off]['layers']:
                vr_vals.append(sweep_results[off]['layers'][layer]['V_R'])
                valid_offsets.append(off)
        if valid_offsets:
            ax.plot(valid_offsets, vr_vals, 'o-', label=f'V L{layer}', linewidth=2)

    # Text baseline
    text_sweep = [sweep_results[off]['text_R'] for off in offsets_with_data]
    ax.plot(offsets_with_data, text_sweep, 's--', color='gray', label='Text (nums)',
            linewidth=2, markersize=8)

    ax.set_xlabel('Token offset relative to "="')
    ax.set_ylabel('Pearson R with intermediate result')
    ax.set_title('Forward-Looking: V_R at positions before "=" sign')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='red', linewidth=0.5, linestyle=':', label='"=" position')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'position_sweep.png', dpi=150)
    plt.close(fig)
    print("  Saved position_sweep.png")

    # Figure 3: Operation type breakdown
    op_types_with_data = [o for o in ['+', '-', '*', '/', 'multi'] if o in op_type_results]
    if len(op_types_with_data) >= 2:
        # Use best layer (highest V_R at offset=0)
        best_layer = max(PROBE_LAYERS, key=lambda l: layer_results[l]['V_R'])
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(op_types_with_data))
        w = 0.35
        v_vals = [op_type_results[o]['layers'][best_layer]['V_R'] for o in op_types_with_data]
        t_vals = [op_type_results[o]['layers'][best_layer]['text_R'] for o in op_types_with_data]
        counts = [op_type_results[o]['n'] for o in op_types_with_data]

        bars_v = ax.bar(x - w/2, v_vals, w, label=f'V (L{best_layer})', color='#2196F3')
        bars_t = ax.bar(x + w/2, t_vals, w, label='Text (nums)', color='#4CAF50')

        ax.set_xlabel('Operation Type')
        ax.set_ylabel('Pearson R')
        ax.set_title(f'V vs Text by Operation Type (L{best_layer})')
        labels = [f'{o}\n(n={c})' for o, c in zip(op_types_with_data, counts)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.0)

        for bars in [bars_v, bars_t]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.02:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                            ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'op_type_breakdown.png', dpi=150)
        plt.close(fig)
        print("  Saved op_type_breakdown.png")

    # Figure 4: Residualized signal
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(PROBE_LAYERS))
    w = 0.3
    vt_vals = [layer_results[l]['V_resid_text'] for l in PROBE_LAYERS]
    vg_vals = [layer_results[l]['V_resid_gold'] for l in PROBE_LAYERS]

    ax.bar(x - w/2, vt_vals, w, label='V|text', color='#2196F3')
    ax.bar(x + w/2, vg_vals, w, label='V|gold_answer', color='#FF5722')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Pearson R (residualized)')
    ax.set_title('Residualized V: info beyond text & final answer')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in PROBE_LAYERS])
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'residualized_signal.png', dpi=150)
    plt.close(fig)
    print("  Saved residualized_signal.png")

    # Figure 5: Prompt control
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(PROBE_LAYERS))
    w = 0.3
    veq_vals = [layer_results[l]['V_R'] for l in PROBE_LAYERS]
    vprompt_vals = [layer_results[l]['prompt_V_R'] for l in PROBE_LAYERS]

    ax.bar(x - w/2, veq_vals, w, label='V at "=" (computation)', color='#2196F3')
    ax.bar(x + w/2, vprompt_vals, w, label='V at prompt (encoding)', color='#9C27B0')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Pearson R with intermediate result')
    ax.set_title('Computation vs Problem Encoding')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in PROBE_LAYERS])
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'prompt_control.png', dpi=150)
    plt.close(fig)
    print("  Saved prompt_control.png")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\nAt '=' position (offset=0, n={len(ops_eq)} operations, {n_unique_probs} problems):")
    print(f"  {'Layer':<8} {'V_R':<8} {'K_R':<8} {'text_R':<8} {'V|text':<8} {'V|gold':<8} {'V_prompt':<8} {'V-text':<8} {'V-prompt':<8}")
    for layer in PROBE_LAYERS:
        lr = layer_results[layer]
        print(f"  L{layer:<6} {lr['V_R']:<8.3f} {lr['K_R']:<8.3f} {text_R:<8.3f} "
              f"{lr['V_resid_text']:<8.3f} {lr['V_resid_gold']:<8.3f} "
              f"{lr['prompt_V_R']:<8.3f} {lr['V_R']-text_R:+<8.3f} {lr['V_R']-lr['prompt_V_R']:+<8.3f}")

    # Operation type summary
    if op_type_results:
        best_layer = max(PROBE_LAYERS, key=lambda l: layer_results[l]['V_R'])
        print(f"\nOperation type breakdown (L{best_layer}):")
        for otype in op_types_with_data:
            otr = op_type_results[otype]['layers'][best_layer]
            print(f"  {otype}: V_R={otr['V_R']:.3f}, text_R={otr['text_R']:.3f}, gap={otr['gap']:+.3f} (n={op_type_results[otype]['n']})")

    # Position sweep summary
    print(f"\nForward-looking position sweep:")
    best_layer = max(PROBE_LAYERS, key=lambda l: layer_results[l]['V_R'])
    for off in sorted(sweep_results.keys()):
        if best_layer in sweep_results[off]['layers']:
            sr = sweep_results[off]['layers'][best_layer]
            print(f"  Offset {off:+3d}: V_R(L{best_layer})={sr['V_R']:.3f}, "
                  f"text_R={sweep_results[off]['text_R']:.3f}")

    # ── Save results ──
    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        return obj

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
