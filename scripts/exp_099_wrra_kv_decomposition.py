#!/usr/bin/env python3
"""
Experiment 099: K vs V Probing at WRRA Error Positions

Extends exp_078 (WRRA plain-text CoT) with:
1. Scaled up to 800 problems for more WRRA cases (~50 expected)
2. K-probe vs V-probe comparison at WRRA error positions (exp_078 only tested V)
3. McNemar test for K vs V difference at error positions
4. Per-layer K/V correct-alignment comparison

This is our unique contribution vs Sun et al. (EMNLP 2025) who probed
residual stream only. The K/V decomposition at error positions tests whether
K-routing or V-content (or both) carries the correct intermediate value when
the text is wrong.

Phase 1 prediction: on Qwen (digital, K>V probing), K-probe should predict
correct values at error positions at LEAST as well as V-probe.
Disconfirmation: if K is at chance at error positions, K-routing doesn't carry
hidden computation (challenges the K-routing narrative).
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
N_PROBLEMS = 800
PROBE_LAYERS = [9, 18, 27, 35]  # 25%, 50%, 75%, 97% depth

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_099"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same as exp_078) ──
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


def run_probe_cv(X, y, n_splits=5):
    """Train ridge probe with cross-validation, return Pearson R and predictions."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    if X.shape[0] < max(n_splits, 10):
        return 0.0, np.zeros_like(y, dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    alphas = np.logspace(-2, 6, 50)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    y_pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in kf.split(X_scaled):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(X_scaled[train_idx], y[train_idx])
        y_pred[test_idx] = ridge.predict(X_scaled[test_idx])

    r, _ = stats.pearsonr(y, y_pred)
    return float(r), y_pred


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
    embed_fn = model.get_input_embeddings()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = hidden_size // model.config.num_attention_heads
    kv_dim = num_kv_heads * head_dim

    print(f"Model loaded in {time.time()-t0:.1f}s")
    print(f"  {n_layers} layers, hidden_size={hidden_size}, kv_dim={kv_dim}")
    print(f"  Probe layers: {PROBE_LAYERS}")

    # ── Load GSM8K ──
    print(f"\nLoading GSM8K...")
    problems = load_gsm8k()
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
    gen_budget = TIME_BUDGET * 0.50  # 50% for generation

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
    n_correct_ops = sum(1 for op in all_ops if not op['is_error'])

    print(f"\nArithmetic analysis:")
    print(f"  Total operations: {n_total_ops}")
    print(f"  Errors: {n_errors} ({100*n_errors/max(n_total_ops,1):.2f}%)")
    print(f"  High-confidence errors: {n_high_conf}")
    print(f"  WRRA cases (error + correct final): {n_wrra}")
    print(f"  Correct operations: {n_correct_ops}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: KV extraction at "=" positions
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: KV extraction at arithmetic positions")
    print(f"{'='*60}")

    # Collect all problems that have arithmetic operations
    problems_with_ops = [e for e in gen_data if len(e['arith_ops']) > 0]
    print(f"  Problems with arithmetic: {len(problems_with_ops)}")

    # Storage for KV features
    kv_data = []  # list of dicts: {k_feats, v_feats, text_feats, ...} per layer per op

    extract_budget = TIME_BUDGET * 0.75  # 75% total budget for gen+extract

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

        # Forward pass to get KV cache
        with torch.no_grad():
            out = model(full_ids, use_cache=True)
        past_kv = out.past_key_values

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

            # Extract K, V, text embedding at this position for each probe layer
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
                'tok_pos': abs_tok_pos,
                'rel_pos': tok_pos / max(len(gen_ids), 1),
                'k': {},
                'v': {},
            }

            for layer in PROBE_LAYERS:
                k_cache, v_cache = get_kv(past_kv, layer)
                # Shape: (1, n_kv_heads, seq_len, head_dim)
                k_vec = k_cache[0, :, abs_tok_pos, :].reshape(-1).cpu().float().numpy()
                v_vec = v_cache[0, :, abs_tok_pos, :].reshape(-1).cpu().float().numpy()
                feats['k'][layer] = k_vec
                feats['v'][layer] = v_vec

            # Text embedding (detach from grad graph)
            tok_id = full_ids[0, abs_tok_pos]
            text_emb = embed_fn(tok_id.unsqueeze(0)).squeeze(0).detach().cpu().float().numpy()
            feats['text_emb'] = text_emb

            kv_data.append(feats)

        # Free memory
        del out, past_kv
        torch.cuda.empty_cache()

        if (pi + 1) % 100 == 0:
            elapsed = time.time() - t0
            n_ext = len(kv_data)
            print(f"  [{pi+1}/{len(problems_with_ops)}] Extracted {n_ext} positions, "
                  f"time: {elapsed:.0f}s")

    print(f"\nExtraction complete: {len(kv_data)} arithmetic positions")

    # Separate correct vs error vs WRRA
    correct_positions = [d for d in kv_data if not d['is_error']]
    error_positions = [d for d in kv_data if d['is_error']]
    wrra_positions = [d for d in kv_data if d['is_wrra']]
    wrwa_positions = [d for d in kv_data if d['is_error'] and not d['is_correct_problem']]

    print(f"  Correct positions: {len(correct_positions)}")
    print(f"  Error positions: {len(error_positions)}")
    print(f"  WRRA positions: {len(wrra_positions)}")
    print(f"  WRWA positions: {len(wrwa_positions)}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Probing — K vs V comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 3: K vs V probing at arithmetic positions")
    print(f"{'='*60}")

    from scipy import stats

    results = {}

    for layer in PROBE_LAYERS:
        print(f"\n--- Layer {layer} ({100*layer/n_layers:.0f}% depth) ---")

        # Build feature matrices for CORRECT positions (training data)
        K_correct = np.array([d['k'][layer] for d in correct_positions])
        V_correct = np.array([d['v'][layer] for d in correct_positions])
        y_local_correct = signed_log(np.array([d['correct_result'] for d in correct_positions]))
        y_final_correct = signed_log(np.array([d['gold_answer'] for d in correct_positions
                                                if d['gold_answer'] is not None]))

        # Filter out None gold answers
        valid_final = [i for i, d in enumerate(correct_positions) if d['gold_answer'] is not None]
        K_correct_final = K_correct[valid_final]
        V_correct_final = V_correct[valid_final]

        print(f"  Correct positions: {len(correct_positions)} (local), {len(valid_final)} (final)")

        # ── Probe 1: K/V → local result (training on correct) ──
        k_r_local, k_pred_local = run_probe_cv(K_correct, y_local_correct)
        v_r_local, v_pred_local = run_probe_cv(V_correct, y_local_correct)
        print(f"  K→local R = {k_r_local:.3f}")
        print(f"  V→local R = {v_r_local:.3f}")

        # ── Probe 2: K/V → final answer (training on correct) ──
        k_r_final, k_pred_final = run_probe_cv(K_correct_final, y_final_correct)
        v_r_final, v_pred_final = run_probe_cv(V_correct_final, y_final_correct)
        print(f"  K→final R = {k_r_final:.3f}")
        print(f"  V→final R = {v_r_final:.3f}")

        # ── Probe 3: Partial correlation (K/V → final | local) ──
        # We need probes trained on full data for predictions at error positions
        # Train K and V probes for LOCAL result on ALL correct positions
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        alphas = np.logspace(-2, 6, 50)

        # K probe for local result
        scaler_k = StandardScaler()
        K_scaled = scaler_k.fit_transform(K_correct)
        k_probe_local = RidgeCV(alphas=alphas)
        k_probe_local.fit(K_scaled, y_local_correct)

        # V probe for local result
        scaler_v = StandardScaler()
        V_scaled = scaler_v.fit_transform(V_correct)
        v_probe_local = RidgeCV(alphas=alphas)
        v_probe_local.fit(V_scaled, y_local_correct)

        # ── WRRA Analysis: K vs V correct-alignment at error positions ──
        if len(wrra_positions) >= 3:
            print(f"\n  WRRA Analysis ({len(wrra_positions)} error positions):")

            K_error = np.array([d['k'][layer] for d in wrra_positions])
            V_error = np.array([d['v'][layer] for d in wrra_positions])
            y_correct_at_error = signed_log(np.array([d['correct_result'] for d in wrra_positions]))
            y_written_at_error = signed_log(np.array([d['written_result'] for d in wrra_positions]))

            # Apply trained probes to error positions
            K_error_scaled = scaler_k.transform(K_error)
            V_error_scaled = scaler_v.transform(V_error)
            k_pred_error = k_probe_local.predict(K_error_scaled)
            v_pred_error = v_probe_local.predict(V_error_scaled)

            # Correct-alignment: is probe closer to correct or written?
            k_dist_correct = np.abs(k_pred_error - y_correct_at_error)
            k_dist_written = np.abs(k_pred_error - y_written_at_error)
            k_correct_aligned = k_dist_correct < k_dist_written

            v_dist_correct = np.abs(v_pred_error - y_correct_at_error)
            v_dist_written = np.abs(v_pred_error - y_written_at_error)
            v_correct_aligned = v_dist_correct < v_dist_written

            n_wrra = len(wrra_positions)
            k_align_rate = np.mean(k_correct_aligned)
            v_align_rate = np.mean(v_correct_aligned)

            # Binomial test for each
            k_p = stats.binomtest(int(np.sum(k_correct_aligned)), n_wrra, 0.5).pvalue
            v_p = stats.binomtest(int(np.sum(v_correct_aligned)), n_wrra, 0.5).pvalue

            # McNemar test for K vs V difference
            # Construct 2x2 table: K-correct&V-wrong, K-wrong&V-correct
            both_correct = np.sum(k_correct_aligned & v_correct_aligned)
            k_only = np.sum(k_correct_aligned & ~v_correct_aligned)
            v_only = np.sum(~k_correct_aligned & v_correct_aligned)
            both_wrong = np.sum(~k_correct_aligned & ~v_correct_aligned)

            if k_only + v_only > 0:
                # McNemar's chi-squared (without continuity correction for small N)
                mcnemar_chi2 = (k_only - v_only) ** 2 / (k_only + v_only)
                mcnemar_p = 1 - stats.chi2.cdf(mcnemar_chi2, 1)
            else:
                mcnemar_chi2 = 0.0
                mcnemar_p = 1.0

            print(f"    K correct-alignment: {int(np.sum(k_correct_aligned))}/{n_wrra} "
                  f"= {k_align_rate:.3f} (p={k_p:.4f})")
            print(f"    V correct-alignment: {int(np.sum(v_correct_aligned))}/{n_wrra} "
                  f"= {v_align_rate:.3f} (p={v_p:.4f})")
            print(f"    McNemar K vs V: chi2={mcnemar_chi2:.3f}, p={mcnemar_p:.4f}")
            print(f"    Concordance: both-correct={int(both_correct)}, K-only={int(k_only)}, "
                  f"V-only={int(v_only)}, both-wrong={int(both_wrong)}")

            # Per-case details for the first few
            print(f"\n    Per-case details (first 10):")
            for j, wp in enumerate(wrra_positions[:10]):
                print(f"      [{j}] '{wp['full_match'][:50]}' "
                      f"correct={wp['correct_result']:.1f} written={wp['written_result']:.1f} | "
                      f"K_pred={k_pred_error[j]:.3f}→{'C' if k_correct_aligned[j] else 'W'} "
                      f"V_pred={v_pred_error[j]:.3f}→{'C' if v_correct_aligned[j] else 'W'}")

            results[layer] = {
                'k_r_local': k_r_local,
                'v_r_local': v_r_local,
                'k_r_final': k_r_final,
                'v_r_final': v_r_final,
                'n_wrra': n_wrra,
                'k_align_count': int(np.sum(k_correct_aligned)),
                'k_align_rate': float(k_align_rate),
                'k_p': float(k_p),
                'v_align_count': int(np.sum(v_correct_aligned)),
                'v_align_rate': float(v_align_rate),
                'v_p': float(v_p),
                'both_correct': int(both_correct),
                'k_only': int(k_only),
                'v_only': int(v_only),
                'both_wrong': int(both_wrong),
                'mcnemar_chi2': float(mcnemar_chi2),
                'mcnemar_p': float(mcnemar_p),
            }
        else:
            print(f"  Too few WRRA cases ({len(wrra_positions)}) for analysis")
            results[layer] = {
                'k_r_local': k_r_local,
                'v_r_local': v_r_local,
                'k_r_final': k_r_final,
                'v_r_final': v_r_final,
                'n_wrra': len(wrra_positions),
            }

    # ── Also test ALL error positions (not just WRRA) ──
    print(f"\n{'='*60}")
    print("ALL error positions analysis (error, regardless of final answer)")
    print(f"{'='*60}")

    all_error_results = {}
    if len(error_positions) >= 3:
        for layer in PROBE_LAYERS:
            K_all_err = np.array([d['k'][layer] for d in error_positions])
            V_all_err = np.array([d['v'][layer] for d in error_positions])
            y_correct_all = signed_log(np.array([d['correct_result'] for d in error_positions]))
            y_written_all = signed_log(np.array([d['written_result'] for d in error_positions]))

            # Reuse probes trained on correct positions
            scaler_k_tmp = StandardScaler()
            K_correct_tmp = np.array([d['k'][layer] for d in correct_positions])
            scaler_k_tmp.fit(K_correct_tmp)
            scaler_v_tmp = StandardScaler()
            V_correct_tmp = np.array([d['v'][layer] for d in correct_positions])
            scaler_v_tmp.fit(V_correct_tmp)

            # Need to re-train probes (or reuse from above — but above only stored last layer)
            # Let's re-train cleanly
            y_local_c = signed_log(np.array([d['correct_result'] for d in correct_positions]))
            k_probe = RidgeCV(alphas=alphas)
            k_probe.fit(scaler_k_tmp.transform(K_correct_tmp), y_local_c)
            v_probe = RidgeCV(alphas=alphas)
            v_probe.fit(scaler_v_tmp.transform(V_correct_tmp), y_local_c)

            k_pred_all = k_probe.predict(scaler_k_tmp.transform(K_all_err))
            v_pred_all = v_probe.predict(scaler_v_tmp.transform(V_all_err))

            k_dc = np.abs(k_pred_all - y_correct_all) < np.abs(k_pred_all - y_written_all)
            v_dc = np.abs(v_pred_all - y_correct_all) < np.abs(v_pred_all - y_written_all)

            n_err = len(error_positions)
            k_rate = np.mean(k_dc)
            v_rate = np.mean(v_dc)
            k_p_val = stats.binomtest(int(np.sum(k_dc)), n_err, 0.5).pvalue
            v_p_val = stats.binomtest(int(np.sum(v_dc)), n_err, 0.5).pvalue

            print(f"  L{layer}: K-align {int(np.sum(k_dc))}/{n_err}={k_rate:.3f} (p={k_p_val:.4f}) "
                  f"| V-align {int(np.sum(v_dc))}/{n_err}={v_rate:.3f} (p={v_p_val:.4f})")

            all_error_results[layer] = {
                'n_errors': n_err,
                'k_align_rate': float(k_rate),
                'k_p': float(k_p_val),
                'v_align_rate': float(v_rate),
                'v_p': float(v_p_val),
            }

    # ── Shuffle control ──
    print(f"\n{'='*60}")
    print("SHUFFLE CONTROL: random label pairing")
    print(f"{'='*60}")

    shuffle_results = {}
    for layer in PROBE_LAYERS:
        K_correct_arr = np.array([d['k'][layer] for d in correct_positions])
        V_correct_arr = np.array([d['v'][layer] for d in correct_positions])
        y_local_c = signed_log(np.array([d['correct_result'] for d in correct_positions]))

        # Shuffle y
        rng = np.random.RandomState(SEED + layer)
        y_shuffled = rng.permutation(y_local_c)

        k_r_shuf, _ = run_probe_cv(K_correct_arr, y_shuffled)
        v_r_shuf, _ = run_probe_cv(V_correct_arr, y_shuffled)
        print(f"  L{layer}: K→local(shuffled) R={k_r_shuf:.3f}, V→local(shuffled) R={v_r_shuf:.3f}")
        shuffle_results[layer] = {'k_shuffle': k_r_shuf, 'v_shuffle': v_r_shuf}

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Figures
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 4: Generating figures")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers_pct = [100 * l / n_layers for l in PROBE_LAYERS]
    layer_labels = [f"L{l}\n({p:.0f}%)" for l, p in zip(PROBE_LAYERS, layers_pct)]

    # ── Figure 1: K vs V correct-alignment at WRRA positions ──
    if any('k_align_rate' in results[l] for l in PROBE_LAYERS):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: WRRA correct-alignment rates
        ax = axes[0]
        k_rates = [results[l].get('k_align_rate', 0.5) for l in PROBE_LAYERS]
        v_rates = [results[l].get('v_align_rate', 0.5) for l in PROBE_LAYERS]
        x = np.arange(len(PROBE_LAYERS))
        w = 0.35

        bars_k = ax.bar(x - w/2, k_rates, w, label='K-probe', color='#e74c3c', alpha=0.8)
        bars_v = ax.bar(x + w/2, v_rates, w, label='V-probe', color='#3498db', alpha=0.8)
        ax.axhline(0.5, color='gray', linestyle='--', label='Chance (50%)')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_ylabel('Correct-alignment rate')
        ax.set_title(f'WRRA: K vs V Correct-Alignment\n(n={results[PROBE_LAYERS[0]].get("n_wrra", "?")})')
        ax.legend()
        ax.set_ylim(0, 1)

        # Add p-values
        for i, l in enumerate(PROBE_LAYERS):
            k_p_val = results[l].get('k_p', 1.0)
            v_p_val = results[l].get('v_p', 1.0)
            star_k = '***' if k_p_val < 0.001 else '**' if k_p_val < 0.01 else '*' if k_p_val < 0.05 else ''
            star_v = '***' if v_p_val < 0.001 else '**' if v_p_val < 0.01 else '*' if v_p_val < 0.05 else ''
            ax.text(i - w/2, k_rates[i] + 0.02, star_k, ha='center', fontsize=10, color='#e74c3c')
            ax.text(i + w/2, v_rates[i] + 0.02, star_v, ha='center', fontsize=10, color='#3498db')

        # Right: McNemar concordance
        ax2 = axes[1]
        # Use the best layer for concordance visualization
        best_layer = max(PROBE_LAYERS, key=lambda l: results[l].get('k_align_rate', 0) +
                         results[l].get('v_align_rate', 0))
        r = results[best_layer]
        if 'both_correct' in r:
            categories = ['Both\ncorrect', 'K-only\ncorrect', 'V-only\ncorrect', 'Both\nwrong']
            counts = [r['both_correct'], r['k_only'], r['v_only'], r['both_wrong']]
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
            ax2.bar(categories, counts, color=colors, alpha=0.8)
            ax2.set_ylabel('Count')
            ax2.set_title(f'K vs V Concordance at L{best_layer}\n'
                          f'McNemar p={r["mcnemar_p"]:.3f}')
            for i, (cat, cnt) in enumerate(zip(categories, counts)):
                ax2.text(i, cnt + 0.3, str(cnt), ha='center', fontweight='bold')

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'wrra_kv_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved wrra_kv_comparison.png")

    # ── Figure 2: K vs V probe R comparison (local and final) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    k_locals = [results[l]['k_r_local'] for l in PROBE_LAYERS]
    v_locals = [results[l]['v_r_local'] for l in PROBE_LAYERS]
    k_shuf = [shuffle_results[l]['k_shuffle'] for l in PROBE_LAYERS]
    v_shuf = [shuffle_results[l]['v_shuffle'] for l in PROBE_LAYERS]
    x = np.arange(len(PROBE_LAYERS))
    ax.plot(x, k_locals, 'o-', color='#e74c3c', label='K→local', linewidth=2, markersize=8)
    ax.plot(x, v_locals, 's-', color='#3498db', label='V→local', linewidth=2, markersize=8)
    ax.plot(x, k_shuf, 'x--', color='#e74c3c', label='K→shuffle', alpha=0.5)
    ax.plot(x, v_shuf, '+--', color='#3498db', label='V→shuffle', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel('Pearson R')
    ax.set_title('K vs V → Local Result')
    ax.legend()
    ax.set_ylim(-0.2, 1.1)

    ax2 = axes[1]
    k_finals = [results[l]['k_r_final'] for l in PROBE_LAYERS]
    v_finals = [results[l]['v_r_final'] for l in PROBE_LAYERS]
    ax2.plot(x, k_finals, 'o-', color='#e74c3c', label='K→final', linewidth=2, markersize=8)
    ax2.plot(x, v_finals, 's-', color='#3498db', label='V→final', linewidth=2, markersize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels)
    ax2.set_ylabel('Pearson R')
    ax2.set_title('K vs V → Final Answer')
    ax2.legend()
    ax2.set_ylim(-0.2, 1.1)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'kv_probe_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved kv_probe_comparison.png")

    # ── Figure 3: Error examples with K vs V predictions ──
    if len(wrra_positions) >= 3:
        fig, ax = plt.subplots(figsize=(12, 6))
        best_layer = PROBE_LAYERS[-2] if len(PROBE_LAYERS) > 2 else PROBE_LAYERS[-1]  # L27

        # Re-compute predictions for the best layer
        K_err = np.array([d['k'][best_layer] for d in wrra_positions])
        V_err = np.array([d['v'][best_layer] for d in wrra_positions])
        K_c = np.array([d['k'][best_layer] for d in correct_positions])
        V_c = np.array([d['v'][best_layer] for d in correct_positions])
        y_lc = signed_log(np.array([d['correct_result'] for d in correct_positions]))

        sc_k = StandardScaler()
        sc_k.fit(K_c)
        pr_k = RidgeCV(alphas=alphas)
        pr_k.fit(sc_k.transform(K_c), y_lc)

        sc_v = StandardScaler()
        sc_v.fit(V_c)
        pr_v = RidgeCV(alphas=alphas)
        pr_v.fit(sc_v.transform(V_c), y_lc)

        k_preds = pr_k.predict(sc_k.transform(K_err))
        v_preds = pr_v.predict(sc_v.transform(V_err))

        y_corr = signed_log(np.array([d['correct_result'] for d in wrra_positions]))
        y_writ = signed_log(np.array([d['written_result'] for d in wrra_positions]))

        n_show = min(len(wrra_positions), 20)
        x_pos = np.arange(n_show)

        ax.scatter(x_pos, y_corr[:n_show], marker='*', s=200, color='#2ecc71', zorder=5,
                   label='Correct value')
        ax.scatter(x_pos, y_writ[:n_show], marker='X', s=150, color='#e74c3c', zorder=5,
                   label='Written (wrong)')
        ax.scatter(x_pos - 0.15, k_preds[:n_show], marker='o', s=80, color='#e74c3c', alpha=0.7,
                   label='K-probe pred')
        ax.scatter(x_pos + 0.15, v_preds[:n_show], marker='s', s=80, color='#3498db', alpha=0.7,
                   label='V-probe pred')

        ax.set_xlabel('WRRA case index')
        ax.set_ylabel('signed_log(value)')
        ax.set_title(f'K vs V Predictions at WRRA Error Positions (L{best_layer})')
        ax.legend(loc='best')

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'wrra_predictions_scatter.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved wrra_predictions_scatter.png")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Save results
    # ═══════════════════════════════════════════════════════════════
    summary = {
        'n_problems_generated': n_generated,
        'n_correct': n_correct,
        'accuracy': float(accuracy),
        'n_total_ops': n_total_ops,
        'n_errors': n_errors,
        'n_high_conf_errors': n_high_conf,
        'error_rate': float(n_errors / max(n_total_ops, 1)),
        'n_wrra': n_wrra,
        'n_wrwa': len(wrwa_positions),
        'n_correct_ops_extracted': len(correct_positions),
        'n_error_ops_extracted': len(error_positions),
        'n_wrra_extracted': len(wrra_positions),
        'per_layer': {str(l): results[l] for l in PROBE_LAYERS},
        'all_error': {str(l): all_error_results.get(l, {}) for l in PROBE_LAYERS},
        'shuffle': {str(l): shuffle_results[l] for l in PROBE_LAYERS},
        'elapsed_seconds': time.time() - t0,
    }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Generated: {n_generated} problems, {n_correct} correct ({accuracy:.1%})")
    print(f"Arithmetic: {n_total_ops} ops, {n_errors} errors ({100*n_errors/max(n_total_ops,1):.2f}%)")
    print(f"WRRA cases: {len(wrra_positions)} extracted")
    print(f"\nK vs V at WRRA positions:")
    for l in PROBE_LAYERS:
        r = results[l]
        if 'k_align_rate' in r:
            print(f"  L{l}: K-align {r['k_align_count']}/{r['n_wrra']}={r['k_align_rate']:.3f} "
                  f"(p={r['k_p']:.4f}) | "
                  f"V-align {r['v_align_count']}/{r['n_wrra']}={r['v_align_rate']:.3f} "
                  f"(p={r['v_p']:.4f})")
    print(f"\nK vs V probe R:")
    for l in PROBE_LAYERS:
        r = results[l]
        print(f"  L{l}: K→local={r['k_r_local']:.3f} V→local={r['v_r_local']:.3f} | "
              f"K→final={r['k_r_final']:.3f} V→final={r['v_r_final']:.3f}")
    print(f"\nTotal elapsed: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
