#!/usr/bin/env python3
"""
Experiment 111: Residual Stream WRRA Probing

Exp_099 found K-probe (52-64%) and V-probe (40-52%) at WRRA positions were at
chance — neither significantly predicted the correct value when the model wrote
wrong arithmetic but got the right final answer.

Sun et al. (EMNLP 2025) achieved >90% detection probing the RESIDUAL STREAM
(full hidden states, not K/V separately). Their hypothesis: the correct answer
is encoded in the full hidden representation but may be lost when projected into
K and V subspaces.

This experiment probes the residual stream (dim=2560) at arithmetic "=" positions
alongside K (dim=640) and V (dim=640) to test whether:
1. The residual stream carries more information about intermediate results than
   K or V alone (on correct positions)
2. At WRRA error positions, the residual stream predicts the correct value even
   when K/V probes fail
3. The exp_099 null result was due to probing the wrong representation, not due
   to absence of hidden computation

Runs on full GSM8K (1319 problems) to maximize WRRA cases.
"""

import os
import json
import time
import gc
import re
import sys

import numpy as np
import torch
from pathlib import Path

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

T0 = time.time()
TIME_BUDGET = 6600  # 110 min
MAX_GEN = 512
MAX_SEQ_LEN = 2048
MODEL_NAME = 'Qwen/Qwen3-4B-Base'
N_PROBLEMS = 1319  # Full GSM8K test set
PROBE_LAYERS = [9, 18, 27, 35]  # 25%, 50%, 75%, 97% depth

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_111"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same as exp_078/099) ──
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
    """Extract the final answer from #### tag or 'the answer is' pattern."""
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '')
    m = re.search(r'[Tt]he answer is\s*\$?(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '')
    return None


def parse_arithmetic(text):
    """Find all arithmetic expressions in plain text."""
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

            # Error check with tolerance
            if abs(correct_result) > 100:
                is_error = abs(written_result - correct_result) / max(abs(correct_result), 1) > 0.005
            else:
                is_error = abs(written_result - correct_result) > 0.5

            # Confidence filter
            if is_error and correct_result != 0:
                ratio = max(abs(written_result), 0.01) / max(abs(correct_result), 0.01)
                error_confidence = 'high' if 0.1 <= ratio <= 10 else 'low'
            elif is_error:
                error_confidence = 'high' if abs(written_result) < 10 else 'low'
            else:
                error_confidence = None

            eq_pos_in_match = m.group(0).index('=')
            eq_char_pos = m.start() + eq_pos_in_match

            results.append({
                'expr_str': expr_str,
                'written_result': written_result,
                'correct_result': correct_result,
                'is_error': is_error,
                'error_confidence': error_confidence,
                'eq_char_pos': eq_char_pos,
                'char_start': m.start(),
                'char_end': m.end(),
                'full_match': m.group(0),
            })
        except Exception:
            continue

    return results


def load_gsm8k():
    """Load GSM8K test set."""
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
            'gold_answer_str': ans_str,
        })
    return problems


def get_kv(past_kv, layer_idx):
    """Extract K, V from model output's past_key_values."""
    from transformers import DynamicCache
    if isinstance(past_kv, DynamicCache):
        if hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
            return past_kv.layers[layer_idx].keys, past_kv.layers[layer_idx].values
        else:
            return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
    else:
        return past_kv[layer_idx][0], past_kv[layer_idx][1]


def signed_log(x):
    """Log-transform preserving sign: sign(x) * log(|x| + 1)"""
    return np.sign(x) * np.log(np.abs(x) + 1)


def run_probe_cv(X, y, n_splits=5, problem_ids=None):
    """Train ridge probe with problem-level cross-validation, return Pearson R and predictions."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, GroupKFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    if X.shape[0] < max(n_splits, 10):
        return 0.0, np.zeros_like(y, dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    alphas = np.logspace(-2, 6, 50)

    if problem_ids is not None:
        unique_ids = np.unique(problem_ids)
        if len(unique_ids) >= n_splits:
            kf = GroupKFold(n_splits=n_splits)
            splits = list(kf.split(X_scaled, y, groups=problem_ids))
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            splits = list(kf.split(X_scaled))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = list(kf.split(X_scaled))

    y_pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in splits:
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(X_scaled[train_idx], y[train_idx])
        y_pred[test_idx] = ridge.predict(X_scaled[test_idx])

    r, _ = stats.pearsonr(y, y_pred)
    return float(r), y_pred


def train_probe_full(X, y):
    """Train ridge probe on ALL data, return fitted scaler + model for applying to error positions."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    alphas = np.logspace(-2, 6, 50)
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(X_scaled, y)
    return scaler, ridge


def map_eq_to_token(gen_text, gen_ids, tokenizer, eq_char_pos):
    """Map '=' at eq_char_pos in gen_text to token position in gen_ids."""
    eq_idx = gen_text[:eq_char_pos + 1].count('=') - 1
    if eq_idx < 0:
        return None

    count = 0
    for i, tid in enumerate(gen_ids):
        tok = tokenizer.decode([tid])
        for ch in tok:
            if ch == '=':
                if count == eq_idx:
                    return i
                count += 1
    return None


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
    hidden_size = model.config.hidden_size
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = getattr(model.config, 'head_dim', hidden_size // model.config.num_attention_heads)
    kv_dim = num_kv_heads * head_dim

    print(f"Model loaded in {time.time()-t0:.1f}s")
    print(f"  {n_layers} layers, hidden_size={hidden_size}, kv_dim={kv_dim}")
    print(f"  Probe layers: {PROBE_LAYERS}")

    # ── Load GSM8K ──
    print(f"\nLoading GSM8K...")
    problems = load_gsm8k()
    # Use same seed/permutation as exp_099 for comparability on first 800
    np.random.seed(SEED)
    indices = np.random.permutation(len(problems))[:N_PROBLEMS]
    problems = [problems[i] for i in indices]
    print(f"  {len(problems)} problems selected")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Generate plain-text CoT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 1: Generating plain-text CoT")
    print(f"{'='*60}")

    gen_data = []
    n_correct = 0
    gen_budget = TIME_BUDGET * 0.45  # 45% for generation

    for i, prob in enumerate(problems):
        if time.time() - t0 > gen_budget:
            print(f"  Generation time budget reached at problem {i}")
            break

        prompt = build_prompt(prob['question'])
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

        if input_ids.shape[1] > MAX_SEQ_LEN - MAX_GEN:
            continue

        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=MAX_GEN, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = out[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

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

        # Parse arithmetic
        arith_ops = parse_arithmetic(gen_text)

        entry = {
            'problem_idx': i,
            'question': prob['question'],
            'gold_answer': gold,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'gen_text': gen_text,
            'gen_ids': gen_ids.cpu().tolist(),
            'prompt_len': input_ids.shape[1],
            'arith_ops': arith_ops,
        }
        gen_data.append(entry)

        if (i + 1) % 100 == 0:
            n_gen = len(gen_data)
            acc = n_correct / max(n_gen, 1)
            n_err = sum(1 for e in gen_data for op in e['arith_ops'] if op['is_error'])
            elapsed = time.time() - t0
            print(f"  [{i+1}/{N_PROBLEMS}] Generated: {n_gen}, correct: {n_correct} ({acc:.1%}), "
                  f"arith errors so far: {n_err}, time: {elapsed:.0f}s")

    n_generated = len(gen_data)
    accuracy = n_correct / max(n_generated, 1)
    print(f"\nGeneration complete: {n_generated} problems, {n_correct} correct ({accuracy:.1%})")

    # ── Parse errors ──
    all_ops = []
    for entry in gen_data:
        for op in entry['arith_ops']:
            op['problem_idx'] = entry['problem_idx']
            op['is_correct_problem'] = entry['is_correct']
            op['gold_answer'] = entry['gold_answer']
            all_ops.append(op)

    n_total_ops = len(all_ops)
    n_errors = sum(1 for op in all_ops if op['is_error'])
    n_high_conf = sum(1 for op in all_ops if op['is_error'] and op['error_confidence'] == 'high')
    n_wrra = sum(1 for op in all_ops if op['is_error'] and op['is_correct_problem'])
    n_wrwa = sum(1 for op in all_ops if op['is_error'] and not op['is_correct_problem'])
    n_correct_ops = sum(1 for op in all_ops if not op['is_error'])

    print(f"\nArithmetic analysis:")
    print(f"  Total operations: {n_total_ops}")
    print(f"  Errors: {n_errors} ({100*n_errors/max(n_total_ops,1):.2f}%)")
    print(f"  High-confidence errors: {n_high_conf}")
    print(f"  WRRA cases (error + correct final): {n_wrra}")
    print(f"  WRWA cases (error + wrong final): {n_wrwa}")
    print(f"  Correct operations: {n_correct_ops}")

    if n_wrra < 5:
        print(f"\nWARNING: Only {n_wrra} WRRA cases found. Experiment may be underpowered.")
        print("Continuing with available data...")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Extract hidden states + KV at "=" positions
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: Hidden state + KV extraction at arithmetic positions")
    print(f"{'='*60}")

    # Only process problems that have arithmetic operations
    problems_with_ops = [e for e in gen_data if len(e['arith_ops']) > 0]
    print(f"  Problems with arithmetic: {len(problems_with_ops)}")

    # Storage for features
    probe_data = []

    extract_budget = TIME_BUDGET * 0.75  # 75% total for gen+extract

    for pi, entry in enumerate(problems_with_ops):
        if time.time() - t0 > extract_budget:
            print(f"  Extraction time budget reached at problem {pi}/{len(problems_with_ops)}")
            break

        # Rebuild full sequence
        prompt = build_prompt(entry['question'])
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        gen_ids_tensor = torch.tensor([entry['gen_ids']], device=model.device)
        full_ids = torch.cat([prompt_ids, gen_ids_tensor], dim=1)

        if full_ids.shape[1] > MAX_SEQ_LEN:
            continue

        # Forward pass: get BOTH hidden states and KV cache
        with torch.no_grad():
            out = model(full_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        hidden_states = out.hidden_states  # tuple of (1, seq_len, hidden_size) per layer

        prompt_len = prompt_ids.shape[1]
        gen_text = entry['gen_text']
        gen_ids = entry['gen_ids']

        for op in entry['arith_ops']:
            # Map "=" character position to token position
            tok_pos = map_eq_to_token(gen_text, gen_ids, tokenizer, op['eq_char_pos'])
            if tok_pos is None:
                continue

            # Adjust to full sequence position
            abs_tok_pos = prompt_len + tok_pos

            if abs_tok_pos >= full_ids.shape[1]:
                continue

            feats = {
                'problem_idx': entry['problem_idx'],
                'is_error': op['is_error'],
                'error_confidence': op.get('error_confidence'),
                'is_correct_problem': entry['is_correct'],
                'written_result': op['written_result'],
                'correct_result': op['correct_result'],
                'gold_answer': entry['gold_answer'],
                'expr_str': op['expr_str'],
                'full_match': op['full_match'],
                'is_wrra': op['is_error'] and entry['is_correct'],
                'is_wrwa': op['is_error'] and not entry['is_correct'],
                'tok_pos': abs_tok_pos,
                'rel_pos': tok_pos / max(len(gen_ids), 1),
                'h': {},  # hidden states (residual stream)
                'k': {},  # K-cache
                'v': {},  # V-cache
            }

            for layer in PROBE_LAYERS:
                # Hidden state at this position (residual stream output of layer)
                # hidden_states[0] = embedding, hidden_states[l+1] = output of layer l
                h_vec = hidden_states[layer + 1][0, abs_tok_pos, :].cpu().float().numpy()
                feats['h'][layer] = h_vec

                # K/V cache
                k_cache, v_cache = get_kv(past_kv, layer)
                k_vec = k_cache[0, :, abs_tok_pos, :].reshape(-1).cpu().float().numpy()
                v_vec = v_cache[0, :, abs_tok_pos, :].reshape(-1).cpu().float().numpy()
                feats['k'][layer] = k_vec
                feats['v'][layer] = v_vec

            probe_data.append(feats)

        # Free memory
        del out, past_kv, hidden_states
        torch.cuda.empty_cache()

        if (pi + 1) % 100 == 0:
            elapsed = time.time() - t0
            n_ext = len(probe_data)
            n_wrra_ext = sum(1 for d in probe_data if d['is_wrra'])
            print(f"  [{pi+1}/{len(problems_with_ops)}] Extracted {n_ext} positions "
                  f"(WRRA: {n_wrra_ext}), time: {elapsed:.0f}s")

    # Free model from GPU
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nExtraction complete: {len(probe_data)} arithmetic positions")

    # Separate by category
    correct_positions = [d for d in probe_data if not d['is_error']]
    error_positions = [d for d in probe_data if d['is_error']]
    wrra_positions = [d for d in probe_data if d['is_wrra']]
    wrwa_positions = [d for d in probe_data if d['is_wrwa']]

    print(f"  Correct positions: {len(correct_positions)}")
    print(f"  Error positions: {len(error_positions)}")
    print(f"  WRRA positions: {len(wrra_positions)}")
    print(f"  WRWA positions: {len(wrwa_positions)}")

    n_wrra_final = len(wrra_positions)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Probing — Residual Stream vs K vs V
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 3: Residual stream vs K vs V probing")
    print(f"{'='*60}")

    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = {}
    wrra_results_all = {}

    for layer in PROBE_LAYERS:
        print(f"\n--- Layer {layer} ({100*layer/n_layers:.0f}% depth) ---")

        # Build feature matrices for CORRECT positions
        H_correct = np.array([d['h'][layer] for d in correct_positions])
        K_correct = np.array([d['k'][layer] for d in correct_positions])
        V_correct = np.array([d['v'][layer] for d in correct_positions])
        y_local = signed_log(np.array([d['correct_result'] for d in correct_positions]))
        prob_ids = np.array([d['problem_idx'] for d in correct_positions])

        print(f"  Training data: {len(correct_positions)} correct positions, "
              f"H dim={H_correct.shape[1]}, K dim={K_correct.shape[1]}, V dim={V_correct.shape[1]}")

        # ── Cross-validated probing on CORRECT positions ──
        h_r, h_pred = run_probe_cv(H_correct, y_local, problem_ids=prob_ids)
        k_r, k_pred = run_probe_cv(K_correct, y_local, problem_ids=prob_ids)
        v_r, v_pred = run_probe_cv(V_correct, y_local, problem_ids=prob_ids)

        # Shuffle control
        np.random.seed(SEED + layer)
        y_shuffled = y_local.copy()
        np.random.shuffle(y_shuffled)
        h_r_shuf, _ = run_probe_cv(H_correct, y_shuffled, problem_ids=prob_ids)
        k_r_shuf, _ = run_probe_cv(K_correct, y_shuffled, problem_ids=prob_ids)
        v_r_shuf, _ = run_probe_cv(V_correct, y_shuffled, problem_ids=prob_ids)

        print(f"  H→local R = {h_r:.3f} (shuffle: {h_r_shuf:.3f})")
        print(f"  K→local R = {k_r:.3f} (shuffle: {k_r_shuf:.3f})")
        print(f"  V→local R = {v_r:.3f} (shuffle: {v_r_shuf:.3f})")

        results[layer] = {
            'h_r': h_r, 'k_r': k_r, 'v_r': v_r,
            'h_r_shuf': h_r_shuf, 'k_r_shuf': k_r_shuf, 'v_r_shuf': v_r_shuf,
        }

        # ── Train probes on ALL correct data for WRRA application ──
        h_scaler, h_model = train_probe_full(H_correct, y_local)
        k_scaler, k_model = train_probe_full(K_correct, y_local)
        v_scaler, v_model = train_probe_full(V_correct, y_local)

        # ── Apply probes at WRRA positions ──
        if len(wrra_positions) > 0:
            H_wrra = np.array([d['h'][layer] for d in wrra_positions])
            K_wrra = np.array([d['k'][layer] for d in wrra_positions])
            V_wrra = np.array([d['v'][layer] for d in wrra_positions])

            h_pred_wrra = h_model.predict(h_scaler.transform(H_wrra))
            k_pred_wrra = k_model.predict(k_scaler.transform(K_wrra))
            v_pred_wrra = v_model.predict(v_scaler.transform(V_wrra))

            correct_values = signed_log(np.array([d['correct_result'] for d in wrra_positions]))
            written_values = signed_log(np.array([d['written_result'] for d in wrra_positions]))

            # Correct-alignment: is probe prediction closer to correct or written?
            h_closer_correct = np.abs(h_pred_wrra - correct_values) < np.abs(h_pred_wrra - written_values)
            k_closer_correct = np.abs(k_pred_wrra - correct_values) < np.abs(k_pred_wrra - written_values)
            v_closer_correct = np.abs(v_pred_wrra - correct_values) < np.abs(v_pred_wrra - written_values)

            h_align = np.sum(h_closer_correct)
            k_align = np.sum(k_closer_correct)
            v_align = np.sum(v_closer_correct)
            n_wrra_test = len(wrra_positions)

            # Binomial test (H0: 50% chance)
            from scipy.stats import binomtest
            h_binom = binomtest(h_align, n_wrra_test, 0.5, alternative='greater')
            k_binom = binomtest(k_align, n_wrra_test, 0.5, alternative='greater')
            v_binom = binomtest(v_align, n_wrra_test, 0.5, alternative='greater')

            print(f"\n  WRRA correct-alignment (n={n_wrra_test}):")
            print(f"    H: {h_align}/{n_wrra_test} = {h_align/n_wrra_test:.1%} (p={h_binom.pvalue:.4f})")
            print(f"    K: {k_align}/{n_wrra_test} = {k_align/n_wrra_test:.1%} (p={k_binom.pvalue:.4f})")
            print(f"    V: {v_align}/{n_wrra_test} = {v_align/n_wrra_test:.1%} (p={v_binom.pvalue:.4f})")

            # McNemar tests: H vs K, H vs V, K vs V
            def mcnemar_test(a, b):
                """Compare two binary vectors using McNemar's test."""
                from scipy.stats import binomtest as bt
                both_a_only = np.sum(a & ~b)
                both_b_only = np.sum(~a & b)
                n_discordant = both_a_only + both_b_only
                if n_discordant == 0:
                    return 1.0
                return bt(both_a_only, n_discordant, 0.5).pvalue

            hk_p = mcnemar_test(h_closer_correct, k_closer_correct)
            hv_p = mcnemar_test(h_closer_correct, v_closer_correct)
            kv_p = mcnemar_test(k_closer_correct, v_closer_correct)
            print(f"    McNemar H vs K: p={hk_p:.4f}")
            print(f"    McNemar H vs V: p={hv_p:.4f}")
            print(f"    McNemar K vs V: p={kv_p:.4f}")

            # Probe distance analysis: how far are predictions from correct/written?
            h_dist_correct = np.abs(h_pred_wrra - correct_values)
            h_dist_written = np.abs(h_pred_wrra - written_values)
            k_dist_correct = np.abs(k_pred_wrra - correct_values)
            k_dist_written = np.abs(k_pred_wrra - written_values)
            v_dist_correct = np.abs(v_pred_wrra - correct_values)
            v_dist_written = np.abs(v_pred_wrra - written_values)

            print(f"    H mean |pred-correct|={np.mean(h_dist_correct):.3f}, |pred-written|={np.mean(h_dist_written):.3f}")
            print(f"    K mean |pred-correct|={np.mean(k_dist_correct):.3f}, |pred-written|={np.mean(k_dist_written):.3f}")
            print(f"    V mean |pred-correct|={np.mean(v_dist_correct):.3f}, |pred-written|={np.mean(v_dist_written):.3f}")

            wrra_results_all[layer] = {
                'h_align': int(h_align), 'k_align': int(k_align), 'v_align': int(v_align),
                'n': n_wrra_test,
                'h_rate': float(h_align / n_wrra_test),
                'k_rate': float(k_align / n_wrra_test),
                'v_rate': float(v_align / n_wrra_test),
                'h_p': float(h_binom.pvalue), 'k_p': float(k_binom.pvalue), 'v_p': float(v_binom.pvalue),
                'hk_mcnemar_p': float(hk_p), 'hv_mcnemar_p': float(hv_p), 'kv_mcnemar_p': float(kv_p),
                'h_dist_correct': float(np.mean(h_dist_correct)),
                'h_dist_written': float(np.mean(h_dist_written)),
                'k_dist_correct': float(np.mean(k_dist_correct)),
                'k_dist_written': float(np.mean(k_dist_written)),
                'v_dist_correct': float(np.mean(v_dist_correct)),
                'v_dist_written': float(np.mean(v_dist_written)),
                'per_case': [],
            }

            # Per-case details for WRRA positions
            for i, d in enumerate(wrra_positions):
                wrra_results_all[layer]['per_case'].append({
                    'problem_idx': d['problem_idx'],
                    'expr': d['expr_str'],
                    'full_match': d['full_match'],
                    'correct': d['correct_result'],
                    'written': d['written_result'],
                    'h_pred': float(np.exp(np.abs(h_pred_wrra[i])) - 1) * np.sign(h_pred_wrra[i]),
                    'k_pred': float(np.exp(np.abs(k_pred_wrra[i])) - 1) * np.sign(k_pred_wrra[i]),
                    'v_pred': float(np.exp(np.abs(v_pred_wrra[i])) - 1) * np.sign(v_pred_wrra[i]),
                    'h_correct': bool(h_closer_correct[i]),
                    'k_correct': bool(k_closer_correct[i]),
                    'v_correct': bool(v_closer_correct[i]),
                })

        # ── Also apply at WRWA positions for comparison ──
        if len(wrwa_positions) > 0:
            H_wrwa = np.array([d['h'][layer] for d in wrwa_positions])
            K_wrwa = np.array([d['k'][layer] for d in wrwa_positions])
            V_wrwa = np.array([d['v'][layer] for d in wrwa_positions])

            h_pred_wrwa = h_model.predict(h_scaler.transform(H_wrwa))
            k_pred_wrwa = k_model.predict(k_scaler.transform(K_wrwa))
            v_pred_wrwa = v_model.predict(v_scaler.transform(V_wrwa))

            correct_wrwa = signed_log(np.array([d['correct_result'] for d in wrwa_positions]))
            written_wrwa = signed_log(np.array([d['written_result'] for d in wrwa_positions]))

            h_closer_c_wrwa = np.abs(h_pred_wrwa - correct_wrwa) < np.abs(h_pred_wrwa - written_wrwa)
            k_closer_c_wrwa = np.abs(k_pred_wrwa - correct_wrwa) < np.abs(k_pred_wrwa - written_wrwa)
            v_closer_c_wrwa = np.abs(v_pred_wrwa - correct_wrwa) < np.abs(v_pred_wrwa - written_wrwa)

            n_wrwa_test = len(wrwa_positions)
            print(f"\n  WRWA correct-alignment (n={n_wrwa_test}, control — expect ~50%):")
            print(f"    H: {np.sum(h_closer_c_wrwa)}/{n_wrwa_test} = {np.mean(h_closer_c_wrwa):.1%}")
            print(f"    K: {np.sum(k_closer_c_wrwa)}/{n_wrwa_test} = {np.mean(k_closer_c_wrwa):.1%}")
            print(f"    V: {np.sum(v_closer_c_wrwa)}/{n_wrwa_test} = {np.mean(v_closer_c_wrwa):.1%}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Additional analyses
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 4: Additional analyses")
    print(f"{'='*60}")

    # ── Analysis 1: Combined H+K+V features ──
    print("\n--- Concatenated (H+K+V) probing at WRRA ---")
    if len(wrra_positions) > 0:
        for layer in PROBE_LAYERS:
            H_c = np.array([d['h'][layer] for d in correct_positions])
            K_c = np.array([d['k'][layer] for d in correct_positions])
            V_c = np.array([d['v'][layer] for d in correct_positions])
            HKV_c = np.concatenate([H_c, K_c, V_c], axis=1)
            y_local = signed_log(np.array([d['correct_result'] for d in correct_positions]))

            # Train on all correct
            hkv_scaler, hkv_model = train_probe_full(HKV_c, y_local)

            # Apply at WRRA
            H_w = np.array([d['h'][layer] for d in wrra_positions])
            K_w = np.array([d['k'][layer] for d in wrra_positions])
            V_w = np.array([d['v'][layer] for d in wrra_positions])
            HKV_w = np.concatenate([H_w, K_w, V_w], axis=1)

            hkv_pred = hkv_model.predict(hkv_scaler.transform(HKV_w))
            correct_vals = signed_log(np.array([d['correct_result'] for d in wrra_positions]))
            written_vals = signed_log(np.array([d['written_result'] for d in wrra_positions]))

            closer = np.abs(hkv_pred - correct_vals) < np.abs(hkv_pred - written_vals)
            n_align = np.sum(closer)
            binom_p = binomtest(n_align, len(wrra_positions), 0.5, alternative='greater').pvalue

            print(f"  L{layer}: HKV align {n_align}/{len(wrra_positions)} = {n_align/len(wrra_positions):.1%} (p={binom_p:.4f})")

            if layer in wrra_results_all:
                wrra_results_all[layer]['hkv_align'] = int(n_align)
                wrra_results_all[layer]['hkv_rate'] = float(n_align / len(wrra_positions))
                wrra_results_all[layer]['hkv_p'] = float(binom_p)

    # ── Analysis 2: Error type breakdown (high vs low confidence) ──
    print("\n--- Error confidence breakdown ---")
    high_conf_wrra = [d for d in wrra_positions if d['error_confidence'] == 'high']
    low_conf_wrra = [d for d in wrra_positions if d['error_confidence'] == 'low']
    print(f"  High-confidence WRRA: {len(high_conf_wrra)}")
    print(f"  Low-confidence WRRA: {len(low_conf_wrra)}")

    if len(high_conf_wrra) >= 5:
        print("  High-confidence WRRA alignment (best layer only):")
        best_layer = max(PROBE_LAYERS, key=lambda l: results[l]['h_r'])
        H_hc = np.array([d['h'][best_layer] for d in high_conf_wrra])
        H_c = np.array([d['h'][best_layer] for d in correct_positions])
        y_local = signed_log(np.array([d['correct_result'] for d in correct_positions]))
        sc, md = train_probe_full(H_c, y_local)
        h_pred_hc = md.predict(sc.transform(H_hc))
        correct_hc = signed_log(np.array([d['correct_result'] for d in high_conf_wrra]))
        written_hc = signed_log(np.array([d['written_result'] for d in high_conf_wrra]))
        closer_hc = np.abs(h_pred_hc - correct_hc) < np.abs(h_pred_hc - written_hc)
        n_hc = len(high_conf_wrra)
        hc_align = np.sum(closer_hc)
        hc_p = binomtest(hc_align, n_hc, 0.5, alternative='greater').pvalue
        print(f"    H at L{best_layer}: {hc_align}/{n_hc} = {hc_align/n_hc:.1%} (p={hc_p:.4f})")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Figures
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 5: Generating figures")
    print(f"{'='*60}")

    # ── Figure 1: H vs K vs V correct-alignment at WRRA ──
    if len(wrra_positions) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        layers_str = [f"L{l}" for l in PROBE_LAYERS]
        h_rates = [wrra_results_all[l]['h_rate'] for l in PROBE_LAYERS]
        k_rates = [wrra_results_all[l]['k_rate'] for l in PROBE_LAYERS]
        v_rates = [wrra_results_all[l]['v_rate'] for l in PROBE_LAYERS]

        x = np.arange(len(PROBE_LAYERS))
        w = 0.25
        ax = axes[0]
        bars_h = ax.bar(x - w, h_rates, w, label='H (residual)', color='#2196F3', alpha=0.85)
        bars_k = ax.bar(x, k_rates, w, label='K-cache', color='#FF9800', alpha=0.85)
        bars_v = ax.bar(x + w, v_rates, w, label='V-cache', color='#4CAF50', alpha=0.85)

        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance (50%)')
        ax.set_xticks(x)
        ax.set_xticklabels(layers_str)
        ax.set_ylabel('Correct-alignment rate')
        ax.set_title(f'WRRA Correct-Alignment: H vs K vs V (n={n_wrra_final})')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.0)

        # Add p-value annotations
        for i, l in enumerate(PROBE_LAYERS):
            h_p = wrra_results_all[l]['h_p']
            sig_h = '*' if h_p < 0.05 else ('†' if h_p < 0.10 else '')
            if sig_h:
                ax.annotate(sig_h, (i - w, h_rates[i] + 0.02), ha='center', fontsize=12, color='#2196F3')

        # ── Figure 1b: Probe R on correct positions ──
        ax2 = axes[1]
        h_rs = [results[l]['h_r'] for l in PROBE_LAYERS]
        k_rs = [results[l]['k_r'] for l in PROBE_LAYERS]
        v_rs = [results[l]['v_r'] for l in PROBE_LAYERS]
        h_sh = [results[l]['h_r_shuf'] for l in PROBE_LAYERS]

        ax2.plot(layers_str, h_rs, 'o-', color='#2196F3', label='H (residual)', linewidth=2)
        ax2.plot(layers_str, k_rs, 's-', color='#FF9800', label='K-cache', linewidth=2)
        ax2.plot(layers_str, v_rs, '^-', color='#4CAF50', label='V-cache', linewidth=2)
        ax2.plot(layers_str, h_sh, 'x--', color='gray', label='H shuffle', linewidth=1)
        ax2.set_ylabel('Pearson R (CV)')
        ax2.set_title('Probe R on Correct Positions (→ local result)')
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 1.05)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'wrra_residual_stream_alignment.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: wrra_residual_stream_alignment.png")

    # ── Figure 2: Per-case predictions scatter ──
    if len(wrra_positions) > 0:
        best_layer = max(PROBE_LAYERS, key=lambda l: wrra_results_all[l]['h_rate'])
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax_idx, (label, cache_key, color) in enumerate([
            ('H (residual)', 'h', '#2196F3'),
            ('K-cache', 'k', '#FF9800'),
            ('V-cache', 'v', '#4CAF50'),
        ]):
            ax = axes[ax_idx]
            cases = wrra_results_all[best_layer]['per_case']
            correct_vals = [c['correct'] for c in cases]
            written_vals = [c['written'] for c in cases]
            pred_vals = [c[f'{cache_key}_pred'] for c in cases]
            is_correct = [c[f'{cache_key}_correct'] for c in cases]

            for j in range(len(cases)):
                marker = 'o' if is_correct[j] else 'x'
                alpha = 0.8 if is_correct[j] else 0.4
                ec = 'black' if is_correct[j] else 'none'
                ax.scatter(correct_vals[j], pred_vals[j], marker=marker, color=color,
                          alpha=alpha, s=60, edgecolors=ec, linewidths=0.5)

            # Reference lines
            all_vals = correct_vals + written_vals + pred_vals
            vmin, vmax = min(all_vals) * 0.9, max(all_vals) * 1.1
            if vmin > 0:
                vmin = min(all_vals) * 0.8
            ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.3, label='pred=correct')
            ax.set_xlabel('Correct result')
            ax.set_ylabel('Probe prediction')
            n_correct_cache = sum(is_correct)
            ax.set_title(f'{label} at L{best_layer}\n{n_correct_cache}/{len(cases)} correct-aligned')
            ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'wrra_per_case_scatter.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: wrra_per_case_scatter.png")

    # ── Figure 3: Distance comparison (probe pred to correct vs written) ──
    if len(wrra_positions) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(PROBE_LAYERS))
        w = 0.12
        colors_correct = ['#2196F3', '#FF9800', '#4CAF50']
        colors_written = ['#90CAF9', '#FFE0B2', '#A5D6A7']

        for idx, (label, key) in enumerate([('H', 'h'), ('K', 'k'), ('V', 'v')]):
            dc = [wrra_results_all[l][f'{key}_dist_correct'] for l in PROBE_LAYERS]
            dw = [wrra_results_all[l][f'{key}_dist_written'] for l in PROBE_LAYERS]
            ax.bar(x + (2*idx - 2) * w, dc, w, label=f'{label}→correct', color=colors_correct[idx])
            ax.bar(x + (2*idx - 1) * w, dw, w, label=f'{label}→written', color=colors_written[idx],
                   hatch='//')

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in PROBE_LAYERS])
        ax.set_ylabel('Mean |pred - target| (log-space)')
        ax.set_title(f'WRRA: Probe Distance to Correct vs Written Value (n={n_wrra_final})')
        ax.legend(fontsize=8, ncol=3, loc='upper left')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'wrra_distance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: wrra_distance_comparison.png")

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    summary = {
        'experiment': 'exp_111_residual_stream_wrra',
        'model': MODEL_NAME,
        'n_generated': n_generated,
        'n_correct': n_correct,
        'accuracy': accuracy,
        'n_total_ops': n_total_ops,
        'n_errors': n_errors,
        'n_high_conf_errors': n_high_conf,
        'n_wrra': n_wrra,
        'n_wrwa': n_wrwa,
        'n_correct_ops': n_correct_ops,
        'n_extracted': len(probe_data),
        'n_wrra_extracted': len(wrra_positions),
        'n_wrwa_extracted': len(wrwa_positions),
        'probe_results_correct': {},
        'wrra_results': {},
        'runtime_seconds': time.time() - t0,
    }

    for l in PROBE_LAYERS:
        summary['probe_results_correct'][str(l)] = results[l]
        if l in wrra_results_all:
            wrra_copy = {k: v for k, v in wrra_results_all[l].items() if k != 'per_case'}
            summary['wrra_results'][str(l)] = wrra_copy

    # Save per-case data separately (can be large)
    per_case_data = {}
    for l in PROBE_LAYERS:
        if l in wrra_results_all:
            per_case_data[str(l)] = wrra_results_all[l]['per_case']

    # JSON-safe conversion
    def make_json_safe(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        return obj

    with open(RESULTS_DIR / 'summary.json', 'w') as f:
        json.dump(make_json_safe(summary), f, indent=2)
    with open(RESULTS_DIR / 'wrra_per_case.json', 'w') as f:
        json.dump(make_json_safe(per_case_data), f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Generated: {n_generated} problems, {n_correct} correct ({accuracy:.1%})")
    print(f"Arithmetic: {n_total_ops} ops, {n_errors} errors ({100*n_errors/max(n_total_ops,1):.2f}%)")
    print(f"WRRA cases: {n_wrra} (extracted: {len(wrra_positions)})")
    print(f"\nProbe R on correct positions:")
    for l in PROBE_LAYERS:
        print(f"  L{l}: H={results[l]['h_r']:.3f}, K={results[l]['k_r']:.3f}, V={results[l]['v_r']:.3f}")
    if len(wrra_positions) > 0:
        print(f"\nWRRA correct-alignment (n={len(wrra_positions)}):")
        for l in PROBE_LAYERS:
            wr = wrra_results_all[l]
            print(f"  L{l}: H={wr['h_rate']:.1%} (p={wr['h_p']:.4f}), "
                  f"K={wr['k_rate']:.1%} (p={wr['k_p']:.4f}), "
                  f"V={wr['v_rate']:.1%} (p={wr['v_p']:.4f})")
    print(f"\nTotal runtime: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
