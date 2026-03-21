#!/usr/bin/env python3
"""
Experiment 078: WRRA Plain-Text CoT + Forward-Looking Computation Probing

Two analyses at arithmetic "=" positions:
1. WRRA rate estimation: Does plain-text format (no <<EXPR=RESULT>>) produce more
   arithmetic errors than the calculator format (exp_071 found 0.15%)?
2. Forward-looking probing: At computation positions, does V predict the FINAL
   answer (not just the local result)? Tests whether the hidden channel carries
   forward-looking computation info.

Method:
- Generate 8-shot CoT with plain-text arithmetic exemplars
- Parse arithmetic from generated text
- Forward pass to extract KV at "=" positions
- Train probes: V→local_result (replicate exp_071), V→final_answer (NEW),
  text→final_answer (baseline)
- Partial correlation: V→final_answer controlling for local_result
- If WRRA errors found: correct-alignment analysis
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
N_PROBLEMS = 400
PROBE_LAYERS = [9, 18, 27, 35]  # 25%, 50%, 75%, 97% depth

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_078"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same questions as exp_071, NO <<...>> tags) ──
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
    """
    Find all arithmetic expressions in plain text.
    Matches: "16 - 3 - 4 = 9", "9 * 2 = 18", "200 * 0.4 / 2 = 40"
    Returns list of dicts with expression details and character positions.
    """
    # Number: digits with optional commas and decimals
    num = r'[\d][\d,]*(?:\.\d+)?'
    op = r'\s*[+\-*/]\s*'
    # Expression: at least one operator between numbers
    expr = f'({num}(?:{op}{num})+)'
    # Result after "="
    result = f'=\\s*\\$?\\s*({num})'
    pattern = f'{expr}\\s*{result}'

    results = []
    for m in re.finditer(pattern, text):
        expr_str = m.group(1).replace(',', '').strip()
        written_str = m.group(2).replace(',', '').strip()

        try:
            written_result = float(written_str)
            # Safe eval: only allow digits, operators, spaces, dots
            sanitized = re.sub(r'[^\d+\-*/. ]', '', expr_str)
            if not sanitized.strip():
                continue
            correct_result = float(eval(sanitized))

            # Error check with tolerance
            if abs(correct_result) > 100:
                is_error = abs(written_result - correct_result) / max(abs(correct_result), 1) > 0.005
            else:
                is_error = abs(written_result - correct_result) > 0.5

            # Confidence: ratio check to filter parsing artifacts
            # E.g., "3/4 = 90" where 3/4 is used as a fraction of something, not 3÷4
            if is_error and correct_result != 0:
                ratio = max(abs(written_result), 0.01) / max(abs(correct_result), 0.01)
                # High confidence: written is within 10x of correct (genuine arithmetic error)
                # Low confidence: > 10x off (likely parsing artifact / contextual fraction)
                error_confidence = 'high' if 0.1 <= ratio <= 10 else 'low'
            elif is_error:
                error_confidence = 'high' if abs(written_result) < 10 else 'low'
            else:
                error_confidence = None

            # Find the "=" position within this match
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
    """Train ridge probe with cross-validation, return Pearson R."""
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


def run_probe_cv_r_only(X, y, n_splits=5):
    """Same as run_probe_cv but returns only R (for convenience)."""
    result = run_probe_cv(X, y, n_splits)
    return result[0]


def map_eq_to_token(gen_text, gen_ids, tokenizer, eq_char_pos):
    """
    Map '=' at eq_char_pos in gen_text to token position in gen_ids.
    Uses occurrence counting (robust to whitespace differences between
    joint decode and individual token decode).
    """
    # Count which '=' occurrence this is (0-indexed)
    eq_idx = gen_text[:eq_char_pos + 1].count('=') - 1
    if eq_idx < 0:
        return None

    # Find the same-numbered '=' in token stream
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
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_GEN,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        gen_ids = output[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Extract answer
        pred_ans_str = extract_answer(gen_text)
        pred_ans = None
        if pred_ans_str:
            try:
                pred_ans = float(pred_ans_str)
            except Exception:
                pass

        is_correct = (pred_ans is not None and prob['gold_answer'] is not None and
                      abs(pred_ans - prob['gold_answer']) < 0.5)
        if is_correct:
            n_correct += 1

        # Parse arithmetic
        arith_ops = parse_arithmetic(gen_text)

        gen_data.append({
            'prob_idx': i,
            'question': prob['question'],
            'gold_answer': prob['gold_answer'],
            'pred_answer': pred_ans,
            'is_correct': is_correct,
            'gen_text': gen_text,
            'gen_ids': gen_ids.cpu().tolist(),
            'prompt_len': input_ids.shape[1],
            'arith_ops': arith_ops,
        })

        if (i + 1) % 50 == 0:
            n_with_ops = sum(1 for d in gen_data if d['arith_ops'])
            total_ops = sum(len(d['arith_ops']) for d in gen_data)
            total_errors = sum(sum(1 for op in d['arith_ops'] if op['is_error']) for d in gen_data)
            print(f"  {i+1}/{N_PROBLEMS} | correct: {n_correct}/{len(gen_data)} "
                  f"({100*n_correct/max(len(gen_data),1):.1f}%) | ops: {total_ops} | "
                  f"errors: {total_errors} | {time.time()-t0:.0f}s")

    n_gen = len(gen_data)
    acc = 100 * n_correct / max(n_gen, 1)
    print(f"\n  Generated: {n_gen} problems")
    print(f"  Accuracy: {n_correct}/{n_gen} ({acc:.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Arithmetic error analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: Arithmetic error analysis")
    print(f"{'='*60}")

    all_ops = []
    errors = []
    wrra = []
    wrwa = []

    for d in gen_data:
        for op in d['arith_ops']:
            op['prob_idx'] = d['prob_idx']
            op['prob_correct'] = d['is_correct']
            op['gold_answer'] = d['gold_answer']
            op['pred_answer'] = d['pred_answer']
            all_ops.append(op)
            if op['is_error']:
                errors.append(op)
                if d['is_correct']:
                    wrra.append(op)
                else:
                    wrwa.append(op)

    n_ops = len(all_ops)
    n_errors = len(errors)
    n_wrra = len(wrra)
    n_wrwa = len(wrwa)
    error_rate = 100 * n_errors / max(n_ops, 1)

    print(f"  Total arithmetic operations: {n_ops}")
    print(f"  Errors: {n_errors} ({error_rate:.2f}%)")
    print(f"  WRRA (wrong calc, right answer): {n_wrra}")
    print(f"  WRWA (wrong calc, wrong answer): {n_wrwa}")
    print(f"  Comparison to exp_071 (calculator format): 2/1339 = 0.15%")
    print(f"  Format effect: {error_rate:.2f}% plain-text vs 0.15% calculator")

    # Separate high/low confidence errors
    errors_high = [e for e in errors if e.get('error_confidence') == 'high']
    errors_low = [e for e in errors if e.get('error_confidence') == 'low']
    n_errors_high = len(errors_high)
    n_errors_low = len(errors_low)
    wrra_high = [e for e in wrra if e.get('error_confidence') == 'high']
    wrwa_high = [e for e in wrwa if e.get('error_confidence') == 'high']

    print(f"  High-confidence errors: {n_errors_high} (genuine arithmetic mistakes)")
    print(f"  Low-confidence errors: {n_errors_low} (likely parsing artifacts / contextual fractions)")
    print(f"  High-conf WRRA: {len(wrra_high)}")
    print(f"  High-conf WRWA: {len(wrwa_high)}")

    if errors:
        print(f"\n  Error details (up to 30):")
        for j, err in enumerate(errors[:30]):
            # Get context around the error
            prob_data = gen_data[err['prob_idx']] if err['prob_idx'] < len(gen_data) else None
            ctx = ''
            if prob_data:
                text = prob_data['gen_text']
                start = max(0, err['char_start'] - 30)
                end = min(len(text), err['char_end'] + 30)
                ctx = f"  ctx: ...{text[start:end]}..."
            conf = err.get('error_confidence', '?')
            print(f"    {j+1}. [{conf}] '{err['full_match']}' → written={err['written_result']}, "
                  f"correct={err['correct_result']}, final_correct={err['prob_correct']}")
            if ctx:
                print(f"       {ctx}")

    # Save Phase 2 results
    phase2_results = {
        'n_problems': n_gen,
        'n_correct': n_correct,
        'accuracy': acc,
        'n_operations': n_ops,
        'n_errors': n_errors,
        'n_errors_high_confidence': n_errors_high,
        'n_errors_low_confidence': n_errors_low,
        'error_rate_pct': error_rate,
        'error_rate_high_conf_pct': 100 * n_errors_high / max(n_ops, 1),
        'n_wrra': n_wrra,
        'n_wrra_high_conf': len(wrra_high),
        'n_wrwa': n_wrwa,
        'n_wrwa_high_conf': len(wrwa_high),
        'exp071_error_rate_pct': 0.15,
        'errors': [{'expr': e['full_match'], 'written': e['written_result'],
                     'correct': e['correct_result'], 'final_correct': e['prob_correct'],
                     'confidence': e.get('error_confidence', '?')}
                    for e in errors],
    }
    with open(RESULTS_DIR / "error_analysis.json", 'w') as f:
        json.dump(phase2_results, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Forward pass + KV extraction at computation positions
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 3: KV extraction at computation positions")
    print(f"{'='*60}")

    # Only process problems with arithmetic operations AND correct final answers
    # (for clean probing data — need to know the actual final answer)
    probe_problems = [d for d in gen_data if d['arith_ops'] and d['gold_answer'] is not None]
    print(f"  Problems with arithmetic: {len(probe_problems)}")

    # Collect features at each "=" position
    # For each operation: K, V at each layer + text embedding + metadata
    features = {layer: {'K': [], 'V': []} for layer in PROBE_LAYERS}
    text_features = []
    local_results = []
    final_answers = []
    is_error_flags = []
    written_results_list = []
    correct_results_list = []
    problem_indices = []

    extraction_budget = TIME_BUDGET * 0.75  # 75% of total budget
    n_extracted = 0

    for pi, d in enumerate(probe_problems):
        if time.time() - t0 > extraction_budget:
            print(f"  Extraction budget reached at problem {pi}")
            break

        # Map arithmetic "=" positions to token positions
        ops_with_token_pos = []
        for op in d['arith_ops']:
            tok_pos = map_eq_to_token(d['gen_text'], d['gen_ids'], tokenizer, op['eq_char_pos'])
            if tok_pos is not None:
                ops_with_token_pos.append((op, tok_pos))

        if not ops_with_token_pos:
            continue

        # Forward pass on full sequence (prompt + gen)
        full_ids = torch.tensor(
            tokenizer.encode(build_prompt(d['question'])) + d['gen_ids'],
            dtype=torch.long
        ).unsqueeze(0).to(model.device)

        if full_ids.shape[1] > MAX_SEQ_LEN:
            continue

        with torch.no_grad():
            out = model(full_ids, use_cache=True)

        # Extract features at each "=" position
        prompt_len = d['prompt_len']
        for op, tok_pos in ops_with_token_pos:
            abs_pos = prompt_len + tok_pos  # absolute position in full sequence

            if abs_pos >= full_ids.shape[1] or abs_pos < prompt_len:
                continue

            # Text embedding at "=" position
            token_id = full_ids[0, abs_pos]
            with torch.no_grad():
                text_emb = embed_fn(token_id.unsqueeze(0)).squeeze(0).cpu().float().numpy()
            text_features.append(text_emb)

            # KV at each probe layer
            for layer in PROBE_LAYERS:
                k, v = get_kv(out.past_key_values, layer)
                # k, v shape: [1, num_kv_heads, seq_len, head_dim]
                k_vec = k[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                v_vec = v[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                features[layer]['K'].append(k_vec)
                features[layer]['V'].append(v_vec)

            local_results.append(op['written_result'] if not op['is_error'] else op['correct_result'])
            final_answers.append(d['gold_answer'])
            is_error_flags.append(op['is_error'])
            written_results_list.append(op['written_result'])
            correct_results_list.append(op['correct_result'])
            problem_indices.append(d['prob_idx'])

        n_extracted += 1

        # Clean up
        del out
        gc.collect()
        torch.cuda.empty_cache()

        if (pi + 1) % 50 == 0:
            n_feat = len(local_results)
            print(f"  {pi+1}/{len(probe_problems)} problems | {n_feat} operation features | "
                  f"{time.time()-t0:.0f}s")

    # Convert to arrays
    for layer in PROBE_LAYERS:
        features[layer]['K'] = np.array(features[layer]['K'])
        features[layer]['V'] = np.array(features[layer]['V'])
    text_features = np.array(text_features)
    local_results = np.array(local_results)
    final_answers = np.array(final_answers)
    is_error_flags = np.array(is_error_flags)
    written_results_arr = np.array(written_results_list)
    correct_results_arr = np.array(correct_results_list)

    n_total = len(local_results)
    n_correct_ops = int((~is_error_flags).sum())
    n_error_ops = int(is_error_flags.sum())

    print(f"\n  Total operation features extracted: {n_total}")
    print(f"  Correct operations: {n_correct_ops}")
    print(f"  Error operations: {n_error_ops}")
    print(f"  KV dim: {features[PROBE_LAYERS[0]]['V'].shape[1]}")
    print(f"  Text dim: {text_features.shape[1]}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Forward-looking probing
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 4: Forward-looking probing at computation positions")
    print(f"{'='*60}")

    # Targets (log-transformed)
    y_local = signed_log(local_results)
    y_final = signed_log(final_answers)

    # Use ONLY correct operations for training (clean signal)
    correct_mask = ~is_error_flags
    if correct_mask.sum() < 20:
        print("  ERROR: Too few correct operations for probing. Aborting.")
        sys.exit(1)

    # Shuffle target for controls
    np.random.seed(SEED + 999)
    shuffle_idx = np.random.permutation(correct_mask.sum())
    y_final_shuffled = y_final[correct_mask][shuffle_idx]

    results_table = {}

    for layer in PROBE_LAYERS:
        layer_pct = round(layer / (n_layers - 1) * 100)
        print(f"\n  Layer L{layer} ({layer_pct}% depth):")

        V = features[layer]['V'][correct_mask]
        K = features[layer]['K'][correct_mask]
        T = text_features[correct_mask]
        yl = y_local[correct_mask]
        yf = y_final[correct_mask]

        # 1. V → local result (replicate exp_071)
        r_v_local, pred_v_local = run_probe_cv(V, yl)
        print(f"    V → local result:    R = {r_v_local:.3f}")

        # 2. V → final answer (NEW)
        r_v_final, pred_v_final = run_probe_cv(V, yf)
        print(f"    V → final answer:    R = {r_v_final:.3f}")

        # 3. K → final answer
        r_k_final, _ = run_probe_cv(K, yf)
        print(f"    K → final answer:    R = {r_k_final:.3f}")

        # 4. Text → final answer (baseline)
        r_t_final, _ = run_probe_cv(T, yf)
        print(f"    Text → final answer: R = {r_t_final:.3f}")

        # 5. Shuffle control
        r_v_shuffle = run_probe_cv_r_only(V, y_final_shuffled)
        print(f"    V → shuffle:         R = {r_v_shuffle:.3f}")

        # 6. Partial correlation: V → final_answer | local_result
        # Residualize y_final against y_local, then probe V → residual
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(yl.reshape(-1, 1), yf)
        residual_yf = yf - lr.predict(yl.reshape(-1, 1))
        r_v_partial, _ = run_probe_cv(V, residual_yf)
        print(f"    V → final|local:     R = {r_v_partial:.3f}  (partial)")

        # 7. K → final | local
        r_k_partial, _ = run_probe_cv(K, residual_yf)
        print(f"    K → final|local:     R = {r_k_partial:.3f}  (partial)")

        # 8. Text → final | local
        r_t_partial, _ = run_probe_cv(T, residual_yf)
        print(f"    Text → final|local:  R = {r_t_partial:.3f}  (partial)")

        # 9. Correlation between local result and final answer
        from scipy import stats
        r_local_final, _ = stats.pearsonr(yl, yf)
        print(f"    local↔final corr:    R = {r_local_final:.3f}")

        results_table[f'L{layer}'] = {
            'layer_pct': layer_pct,
            'V_local': r_v_local,
            'V_final': r_v_final,
            'K_final': r_k_final,
            'Text_final': r_t_final,
            'V_shuffle': r_v_shuffle,
            'V_partial': r_v_partial,
            'K_partial': r_k_partial,
            'Text_partial': r_t_partial,
            'local_final_corr': float(r_local_final),
        }

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: WRRA analysis (if errors found)
    # ═══════════════════════════════════════════════════════════════
    wrra_results = None
    if n_error_ops >= 2:
        print(f"\n{'='*60}")
        print(f"PHASE 5: WRRA analysis ({n_error_ops} error positions)")
        print(f"{'='*60}")

        # Train probe on correct operations, apply to error positions
        for layer in PROBE_LAYERS:
            layer_pct = round(layer / (n_layers - 1) * 100)
            V_train = features[layer]['V'][correct_mask]
            y_train = signed_log(local_results[correct_mask])

            # Train probe
            from sklearn.linear_model import RidgeCV as RidgeCVClass
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            V_train_s = scaler.fit_transform(V_train)
            alphas = np.logspace(-2, 6, 50)
            ridge = RidgeCVClass(alphas=alphas)
            ridge.fit(V_train_s, y_train)

            # Apply to error positions
            V_err = features[layer]['V'][is_error_flags]
            V_err_s = scaler.transform(V_err)
            pred_err = ridge.predict(V_err_s)

            # Compare predictions to correct vs written values
            correct_vals = signed_log(correct_results_arr[is_error_flags])
            written_vals = signed_log(written_results_arr[is_error_flags])

            dist_to_correct = np.abs(pred_err - correct_vals)
            dist_to_written = np.abs(pred_err - written_vals)
            alignment = (dist_to_correct < dist_to_written).mean()

            print(f"\n  L{layer} ({layer_pct}%):")
            print(f"    Error positions: {n_error_ops}")
            print(f"    Correct-alignment rate: {alignment:.3f} (0.5 = chance)")
            for ei in range(min(n_error_ops, 5)):
                print(f"      Error {ei+1}: pred={pred_err[ei]:.3f}, "
                      f"correct={correct_vals[ei]:.3f}, written={written_vals[ei]:.3f}")

        wrra_results = {
            'n_errors': n_error_ops,
            'note': 'See stdout for detailed alignment analysis'
        }
    else:
        print(f"\n  WRRA analysis skipped: only {n_error_ops} error positions (need ≥2)")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 6: Figures
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 6: Generating figures")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = sorted(results_table.keys(), key=lambda x: int(x[1:]))
    layer_pcts = [results_table[l]['layer_pct'] for l in layers]

    # Figure 1: Forward-looking probes by layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Final answer probes
    ax = axes[0]
    ax.plot(layer_pcts, [results_table[l]['V_final'] for l in layers], 'o-', label='V → final', color='blue', linewidth=2)
    ax.plot(layer_pcts, [results_table[l]['K_final'] for l in layers], 's--', label='K → final', color='green', linewidth=2)
    ax.plot(layer_pcts, [results_table[l]['Text_final'] for l in layers], '^--', label='Text → final', color='red', linewidth=2)
    ax.plot(layer_pcts, [results_table[l]['V_shuffle'] for l in layers], 'x:', label='V → shuffle', color='gray', linewidth=1)
    ax.set_xlabel('Layer Depth (%)')
    ax.set_ylabel('Pearson R (5-fold CV)')
    ax.set_title('Final Answer Probing at Computation Positions')
    ax.legend(fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Right: Partial correlation (controlling for local result)
    ax = axes[1]
    ax.plot(layer_pcts, [results_table[l]['V_partial'] for l in layers], 'o-', label='V → final|local', color='blue', linewidth=2)
    ax.plot(layer_pcts, [results_table[l]['K_partial'] for l in layers], 's--', label='K → final|local', color='green', linewidth=2)
    ax.plot(layer_pcts, [results_table[l]['Text_partial'] for l in layers], '^--', label='Text → final|local', color='red', linewidth=2)
    ax.set_xlabel('Layer Depth (%)')
    ax.set_ylabel('Pearson R (5-fold CV)')
    ax.set_title('Partial: Final Answer | Local Result')
    ax.legend(fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'forward_looking_probes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved forward_looking_probes.png")

    # Figure 2: V→local vs V→final comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(layers))
    width = 0.25
    ax.bar(x - width, [results_table[l]['V_local'] for l in layers], width, label='V → local result', color='steelblue')
    ax.bar(x, [results_table[l]['V_final'] for l in layers], width, label='V → final answer', color='coral')
    ax.bar(x + width, [results_table[l]['V_partial'] for l in layers], width, label='V → final|local (partial)', color='gold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l[1:]} ({results_table[l]["layer_pct"]}%)' for l in layers])
    ax.set_ylabel('Pearson R')
    ax.set_title('V-Probe: Local vs Final Answer vs Partial')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v_probe_local_vs_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved v_probe_local_vs_final.png")

    # Figure 3: Error rate comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['Calculator\n(exp_071)', 'Plain text\n(exp_078)'],
                  [0.15, error_rate],
                  color=['steelblue', 'coral'],
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title(f'Arithmetic Error Rate: Calculator vs Plain Text\n'
                 f'(exp_071: 2/{1339}, exp_078: {n_errors}/{n_ops})')
    for bar, val in zip(bars, [0.15, error_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(error_rate, 0.5) * 1.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'error_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved error_rate_comparison.png")

    # ═══════════════════════════════════════════════════════════════
    # Save all results
    # ═══════════════════════════════════════════════════════════════
    all_results = {
        'model': MODEL_NAME,
        'n_problems_generated': n_gen,
        'n_correct': n_correct,
        'accuracy': acc,
        'n_operations': n_ops,
        'n_errors': n_errors,
        'error_rate_pct': error_rate,
        'n_wrra': n_wrra,
        'n_wrwa': n_wrwa,
        'n_operation_features': n_total,
        'n_correct_ops': n_correct_ops,
        'kv_dim': int(kv_dim),
        'text_dim': int(hidden_size),
        'probe_layers': PROBE_LAYERS,
        'forward_looking_probes': results_table,
        'wrra_analysis': wrra_results,
        'elapsed_seconds': time.time() - t0,
    }
    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'results.json'}")
    print(f"  Total elapsed: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
