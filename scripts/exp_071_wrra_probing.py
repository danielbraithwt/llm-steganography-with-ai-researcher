#!/usr/bin/env python3
"""
Experiment 071: Wrong Reasoning, Right Answer (WRRA) — KV Cache Probing

Probes whether the KV cache encodes the CORRECT intermediate value at positions
where the model writes WRONG arithmetic, providing direct evidence that the
hidden channel carries computation independent of the text.

Method:
1. Generate CoT for GSM8K problems, collecting KV cache
2. Parse all arithmetic operations (<<EXPR=RESULT>> format)
3. Identify correct vs error positions; classify WRRA vs WRWA
4. Train ridge probe on correct positions: KV features → value
5. Apply to error positions: does it predict correct or written value?
6. Compare K-probe vs V-probe (expect V > K per Phase 2 findings)
"""

import os
import json
import time
import gc
import re
import sys
import bisect

import numpy as np
import torch
from pathlib import Path

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

TIME_BUDGET = 5400  # 90 min (generous — need many problems to find errors)
MAX_GEN = 512
MAX_SEQ_LEN = 2048
MODEL_NAME = 'Qwen/Qwen3-4B-Base'
N_PROBLEMS_MAX = 600  # process up to this many to find enough errors

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_071"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 8-shot exemplars (same as Phase 1/exp_068) ──
EXEMPLARS = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
     "a": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "a": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total bolts needed is 2+1=<<2+1=3>>3\n#### 3"},
    {"q": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
     "a": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*150%=$<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"},
    {"q": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "a": "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624"},
    {"q": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
     "a": "If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.\nIf she feeds the flock 15 cups in the morning, and 25 cups in the afternoon, then the carry-over to the final meal would be 60-15-25=<<60-15-25=20>>20 cups.\n#### 20"},
    {"q": "Kylar went to the store to get water and some apples. The store sold apples for $1 each and water for $3 per bottle. Kylar wanted to buy one bag of apples and 2 bottles of water. How much would Kylar spend if each bag has 6 apples?",
     "a": "A bag has 6 apples and each apple costs $1, so a bag costs 6*1=$<<6*1=6>>6\nKylar wants 2 bottles of water so that would cost 2*3=$<<2*3=6>>6\nAltogether, Kylar would spend 6+6=$<<6+6=12>>12\n#### 12"},
    {"q": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
     "a": "If Seattle has 20 sheep, Charleston has 4 * 20 = <<4*20=80>>80 sheep\nToulouse has 2 * 80 = <<2*80=160>>160 sheep\nTogether, they have 20 + 80 + 160 = <<20+80+160=260>>260 sheep\n#### 260"},
    {"q": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
     "a": "First find how long it takes to download 40% of the file: 200 GB * 0.4 / 2 GB/minute = <<200*0.4/2=40>>40 minutes\nThen find how long it takes to download the whole file once the restart is complete: 200 GB / 2 GB/minute = <<200/2=100>>100 minutes\nThen add the time to download 40% of the file, the restart time, and the time to download the whole file: 40 + 20 + 100 = <<40+20+100=160>>160 minutes\n#### 160"},
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


def safe_eval(expr_str):
    """Safely evaluate simple arithmetic expression."""
    cleaned = expr_str.replace(',', '').strip()
    # Only allow: digits, +, -, *, /, ., (, ), spaces
    if not re.match(r'^[\d\s+\-*/().]+$', cleaned):
        return None
    try:
        code = compile(cleaned, '<string>', 'eval')
        if code.co_names:
            return None
        result = eval(code, {"__builtins__": {}}, {})
        return float(result)
    except:
        return None


def parse_arithmetic(text):
    """Find all <<EXPR=RESULT>> operations in generated text."""
    ops = []
    for m in re.finditer(r'<<([^>]+?)=(-?[\d,]+(?:\.\d+)?)>>', text):
        expr_str = m.group(1)
        written_str = m.group(2).replace(',', '')
        correct = safe_eval(expr_str)
        if correct is None:
            continue
        try:
            written = float(written_str)
        except:
            continue
        # Position of "=" in the match (last char before result)
        eq_char_pos = m.start() + len('<<') + len(expr_str)
        ops.append({
            'expr': expr_str,
            'written': written,
            'correct': correct,
            'is_correct': abs(correct - written) < 0.5,
            'eq_char_pos': eq_char_pos,
            'match_text': m.group(0),
        })
    return ops


def build_token_starts(gen_ids, tokenizer):
    """Build cumulative character positions for each generated token."""
    token_starts = []
    cum = 0
    for tid in gen_ids:
        token_starts.append(cum)
        decoded = tokenizer.decode([tid], skip_special_tokens=True)
        cum += len(decoded)
    return token_starts, cum


def char_to_token(char_pos, token_starts):
    """Map character position to token index using binary search."""
    idx = bisect.bisect_right(token_starts, char_pos) - 1
    return max(0, idx)


def get_kv(past_kv, layer_idx):
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
    """Train ridge probe with cross-validation, return (r, p)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    if X.shape[0] < max(n_splits, 5):
        return 0.0, 1.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    alphas = np.logspace(-2, 6, 50)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    y_pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in kf.split(X_scaled):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(X_scaled[train_idx], y[train_idx])
        y_pred[test_idx] = ridge.predict(X_scaled[test_idx])

    r, p = stats.pearsonr(y, y_pred)
    return float(r), float(p)


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map='auto', trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    probe_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    print(f"Model: {n_layers} layers, hidden_size={hidden_size}")
    print(f"Probe layers: {probe_layers}")

    ds = load_gsm8k()
    print(f"GSM8K: {len(ds)} test problems")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Generate CoT & extract KV features at arithmetic positions
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 1: Generate CoT and extract features")
    print(f"{'='*60}")

    # Storage per probe layer
    all_K = {l: [] for l in probe_layers}
    all_V = {l: [] for l in probe_layers}
    all_text_emb = []
    all_correct_vals = []
    all_written_vals = []
    all_meta = []

    n_attempted = 0
    n_generated = 0
    n_with_ops = 0
    n_ops_total = 0
    n_ops_correct = 0
    n_ops_error = 0
    n_final_correct = 0
    n_final_wrong = 0
    n_wrra = 0
    n_wrwa = 0
    error_examples = []

    for prob_idx in range(min(N_PROBLEMS_MAX, len(ds))):
        elapsed = time.time() - t0
        if elapsed > TIME_BUDGET * 0.50:
            print(f"  Time budget (50%): stopping at {n_attempted} problems ({elapsed:.0f}s)")
            break

        question = ds[prob_idx]['question']
        gold = extract_gold(ds[prob_idx]['answer'])
        if gold is None:
            continue

        prompt = build_prompt(question)
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        prompt_len = inputs.input_ids.shape[1]

        if prompt_len > MAX_SEQ_LEN - MAX_GEN:
            continue

        n_attempted += 1

        # Generate CoT
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_GEN, do_sample=False,
                temperature=1.0, return_dict_in_generate=True, use_cache=True,
            )

        gen_ids = output.sequences[0][prompt_len:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred_answer = extract_answer(gen_text)
        final_correct = (pred_answer is not None and
                         str(pred_answer).strip() == str(gold).strip())

        # Truncate at #### to avoid parsing arithmetic from continued generation
        answer_marker = gen_text.find('####')
        if answer_marker >= 0:
            gen_text_for_ops = gen_text[:answer_marker]
        else:
            gen_text_for_ops = gen_text

        n_generated += 1
        if final_correct:
            n_final_correct += 1
        else:
            n_final_wrong += 1

        # Parse arithmetic operations in generated text (before #### only)
        ops = parse_arithmetic(gen_text_for_ops)
        if len(ops) == 0:
            del output
            gc.collect()
            torch.cuda.empty_cache()
            continue

        n_with_ops += 1

        # Build token-to-char mapping
        token_starts, total_chars = build_token_starts(gen_ids, tokenizer)

        # Forward pass to get full KV cache
        full_ids = output.sequences[0:1]
        with torch.no_grad():
            out = model(full_ids, use_cache=True)
        past_kv = out.past_key_values

        # Token embeddings for text baseline
        with torch.no_grad():
            gen_token_ids = full_ids[0, prompt_len:]
            text_emb = model.model.embed_tokens(gen_token_ids).cpu().float()

        # Extract features at each arithmetic position
        for op in ops:
            tok_pos_gen = char_to_token(op['eq_char_pos'], token_starts)
            tok_pos_full = tok_pos_gen + prompt_len

            if tok_pos_full >= full_ids.shape[1] or tok_pos_gen >= text_emb.shape[0]:
                continue

            # KV features at this position
            for layer_idx in probe_layers:
                k, v = get_kv(past_kv, layer_idx)
                k_feat = k[0, :, tok_pos_full, :].cpu().float().reshape(-1).numpy()
                v_feat = v[0, :, tok_pos_full, :].cpu().float().reshape(-1).numpy()
                all_K[layer_idx].append(k_feat)
                all_V[layer_idx].append(v_feat)

            # Text features
            t_feat = text_emb[tok_pos_gen].numpy()
            all_text_emb.append(t_feat)

            all_correct_vals.append(op['correct'])
            all_written_vals.append(op['written'])

            is_err = not op['is_correct']
            meta = {
                'prob_idx': prob_idx,
                'is_correct': op['is_correct'],
                'final_correct': final_correct,
                'expr': op['expr'],
                'written': op['written'],
                'correct': op['correct'],
                'wrra': is_err and final_correct,
                'wrwa': is_err and (not final_correct),
            }
            all_meta.append(meta)

            n_ops_total += 1
            if op['is_correct']:
                n_ops_correct += 1
            else:
                n_ops_error += 1
                if final_correct:
                    n_wrra += 1
                else:
                    n_wrwa += 1
                # Save example for display
                if len(error_examples) < 20:
                    error_examples.append({
                        'prob_idx': prob_idx,
                        'expr': op['expr'],
                        'written': op['written'],
                        'correct': op['correct'],
                        'final_correct': final_correct,
                        'context': gen_text[max(0, op['eq_char_pos']-60):op['eq_char_pos']+40],
                    })

        # Free memory
        del output, out, past_kv, text_emb
        gc.collect()
        torch.cuda.empty_cache()

        if n_attempted % 50 == 0:
            print(f"  {n_attempted} attempted, {n_generated} generated, "
                  f"{n_ops_total} ops ({n_ops_correct} correct, {n_ops_error} errors, "
                  f"{n_wrra} WRRA, {n_wrwa} WRWA)")

    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")
    print(f"Problems: {n_attempted} attempted, {n_generated} generated")
    print(f"  Final correct: {n_final_correct} ({100*n_final_correct/max(n_generated,1):.1f}%)")
    print(f"  Final wrong: {n_final_wrong} ({100*n_final_wrong/max(n_generated,1):.1f}%)")
    print(f"Problems with arithmetic: {n_with_ops}")
    print(f"Operations: {n_ops_total} total")
    print(f"  Correct: {n_ops_correct} ({100*n_ops_correct/max(n_ops_total,1):.1f}%)")
    print(f"  Errors: {n_ops_error} ({100*n_ops_error/max(n_ops_total,1):.1f}%)")
    print(f"  WRRA (error + correct final): {n_wrra}")
    print(f"  WRWA (error + wrong final): {n_wrwa}")

    if n_ops_error == 0:
        print("\nNO ARITHMETIC ERRORS FOUND — cannot run WRRA analysis.")
        print("This is itself informative: the model's arithmetic is perfect on these problems.")
        # Save results and exit
        results = {
            'n_attempted': n_attempted, 'n_generated': n_generated,
            'n_ops_total': n_ops_total, 'n_ops_correct': n_ops_correct,
            'n_ops_error': 0, 'n_wrra': 0, 'n_wrwa': 0,
            'error_rate': 0.0,
            'conclusion': 'No arithmetic errors found. Cannot perform WRRA analysis.',
        }
        with open(RESULTS_DIR / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return

    # Print example errors
    print(f"\n--- Example arithmetic errors ---")
    for ex in error_examples[:10]:
        label = "WRRA" if ex['final_correct'] else "WRWA"
        print(f"  [{label}] {ex['expr']} = {ex['correct']:.1f} (wrote {ex['written']:.1f})")
        print(f"    Context: ...{ex['context']}...")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Convert to arrays and prepare for probing
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 2: Probe analysis")
    print(f"{'='*60}")

    for l in probe_layers:
        all_K[l] = np.array(all_K[l])
        all_V[l] = np.array(all_V[l])
    all_text_emb = np.array(all_text_emb)
    correct_vals = np.array(all_correct_vals)
    written_vals = np.array(all_written_vals)

    is_correct = np.array([m['is_correct'] for m in all_meta])
    is_wrra = np.array([m['wrra'] for m in all_meta])
    is_wrwa = np.array([m['wrwa'] for m in all_meta])
    is_error = ~is_correct

    # Log-transform for regression
    y_correct = signed_log(correct_vals)
    y_written = signed_log(written_vals)

    print(f"Feature shapes: K={all_K[probe_layers[0]].shape}, "
          f"text={all_text_emb.shape}")
    print(f"Correct positions: {is_correct.sum()}, Error positions: {is_error.sum()}")
    print(f"WRRA: {is_wrra.sum()}, WRWA: {is_wrwa.sum()}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Train probes on correct positions, test on error positions
    # ═══════════════════════════════════════════════════════════════

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from scipy import stats

    train_mask = is_correct
    test_mask = is_error

    results = {
        'n_attempted': n_attempted, 'n_generated': n_generated,
        'n_ops_total': n_ops_total, 'n_ops_correct': int(n_ops_correct),
        'n_ops_error': int(n_ops_error), 'n_wrra': int(n_wrra), 'n_wrwa': int(n_wrwa),
        'error_rate': float(n_ops_error / n_ops_total),
        'probe_results': {},
    }

    alphas = np.logspace(-2, 6, 50)
    probe_data = {}  # store predictions for figures

    for layer_idx in probe_layers:
        for feat_name, feat_all in [('K', all_K[layer_idx]), ('V', all_V[layer_idx])]:
            train_X = feat_all[train_mask]
            train_y = y_correct[train_mask]  # = y_written for correct positions
            test_X = feat_all[test_mask]

            # Cross-validated training accuracy
            train_r, train_p = run_probe_cv(train_X, train_y, n_splits=5)

            # Train on ALL correct positions, predict on error positions
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(train_X)
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(X_train_sc, train_y)

            if test_X.shape[0] > 0:
                X_test_sc = scaler.transform(test_X)
                preds = ridge.predict(X_test_sc)

                test_correct_y = y_correct[test_mask]
                test_written_y = y_written[test_mask]

                # Alignment: is prediction closer to correct or written?
                dist_correct = np.abs(preds - test_correct_y)
                dist_written = np.abs(preds - test_written_y)
                correct_aligned = dist_correct < dist_written
                alignment_rate = float(np.mean(correct_aligned))

                # Correlation with correct vs written
                r_correct = float(np.corrcoef(preds, test_correct_y)[0, 1]) if len(preds) > 2 else float('nan')
                r_written = float(np.corrcoef(preds, test_written_y)[0, 1]) if len(preds) > 2 else float('nan')

                # Mean absolute error
                mae_correct = float(np.mean(dist_correct))
                mae_written = float(np.mean(dist_written))

                # WRRA subset
                wrra_mask_in_test = is_wrra[test_mask]
                if wrra_mask_in_test.sum() > 0:
                    wrra_preds = preds[wrra_mask_in_test]
                    wrra_correct_y = test_correct_y[wrra_mask_in_test]
                    wrra_written_y = test_written_y[wrra_mask_in_test]
                    wrra_dist_c = np.abs(wrra_preds - wrra_correct_y)
                    wrra_dist_w = np.abs(wrra_preds - wrra_written_y)
                    wrra_alignment = float(np.mean(wrra_dist_c < wrra_dist_w))
                else:
                    wrra_alignment = float('nan')

                # WRWA subset
                wrwa_mask_in_test = is_wrwa[test_mask]
                if wrwa_mask_in_test.sum() > 0:
                    wrwa_preds = preds[wrwa_mask_in_test]
                    wrwa_correct_y = test_correct_y[wrwa_mask_in_test]
                    wrwa_written_y = test_written_y[wrwa_mask_in_test]
                    wrwa_dist_c = np.abs(wrwa_preds - wrwa_correct_y)
                    wrwa_dist_w = np.abs(wrwa_preds - wrwa_written_y)
                    wrwa_alignment = float(np.mean(wrwa_dist_c < wrwa_dist_w))
                else:
                    wrwa_alignment = float('nan')
            else:
                alignment_rate = float('nan')
                r_correct = float('nan')
                r_written = float('nan')
                mae_correct = float('nan')
                mae_written = float('nan')
                wrra_alignment = float('nan')
                wrwa_alignment = float('nan')
                preds = np.array([])

            key = f'L{layer_idx}_{feat_name}'
            results['probe_results'][key] = {
                'train_r': train_r,
                'train_p': train_p,
                'n_train': int(train_X.shape[0]),
                'n_test': int(test_X.shape[0]),
                'correct_alignment_rate': alignment_rate,
                'r_correct': r_correct,
                'r_written': r_written,
                'mae_correct': mae_correct,
                'mae_written': mae_written,
                'wrra_alignment': wrra_alignment,
                'wrwa_alignment': wrwa_alignment,
            }
            probe_data[key] = preds

            print(f"  {key}: train_r={train_r:.3f}, alignment={alignment_rate:.3f}, "
                  f"r_correct={r_correct:.3f}, r_written={r_written:.3f}, "
                  f"wrra_align={wrra_alignment:.3f}")

    # Text baseline
    train_X_text = all_text_emb[train_mask]
    train_y_text = y_correct[train_mask]
    test_X_text = all_text_emb[test_mask]

    text_train_r, text_train_p = run_probe_cv(train_X_text, train_y_text, n_splits=5)

    scaler_text = StandardScaler()
    X_train_text_sc = scaler_text.fit_transform(train_X_text)
    ridge_text = RidgeCV(alphas=alphas)
    ridge_text.fit(X_train_text_sc, train_y_text)

    if test_X_text.shape[0] > 0:
        X_test_text_sc = scaler_text.transform(test_X_text)
        text_preds = ridge_text.predict(X_test_text_sc)
        test_correct_y = y_correct[test_mask]
        test_written_y = y_written[test_mask]

        text_dist_c = np.abs(text_preds - test_correct_y)
        text_dist_w = np.abs(text_preds - test_written_y)
        text_alignment = float(np.mean(text_dist_c < text_dist_w))
        text_r_correct = float(np.corrcoef(text_preds, test_correct_y)[0, 1]) if len(text_preds) > 2 else float('nan')
        text_r_written = float(np.corrcoef(text_preds, test_written_y)[0, 1]) if len(text_preds) > 2 else float('nan')

        wrra_mask_in_test = is_wrra[test_mask]
        if wrra_mask_in_test.sum() > 0:
            text_wrra_preds = text_preds[wrra_mask_in_test]
            text_wrra_c = y_correct[test_mask][wrra_mask_in_test]
            text_wrra_w = y_written[test_mask][wrra_mask_in_test]
            text_wrra_align = float(np.mean(np.abs(text_wrra_preds - text_wrra_c) <
                                             np.abs(text_wrra_preds - text_wrra_w)))
        else:
            text_wrra_align = float('nan')
    else:
        text_alignment = float('nan')
        text_r_correct = float('nan')
        text_r_written = float('nan')
        text_wrra_align = float('nan')
        text_preds = np.array([])

    results['probe_results']['text'] = {
        'train_r': text_train_r,
        'train_p': text_train_p,
        'n_train': int(train_X_text.shape[0]),
        'n_test': int(test_X_text.shape[0]),
        'correct_alignment_rate': text_alignment,
        'r_correct': text_r_correct,
        'r_written': text_r_written,
        'wrra_alignment': text_wrra_align,
    }
    print(f"  text: train_r={text_train_r:.3f}, alignment={text_alignment:.3f}, "
          f"wrra_align={text_wrra_align:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Shuffle control — train probe on correct positions
    # with shuffled value labels → should give ~50% alignment (chance)
    # ═══════════════════════════════════════════════════════════════
    print(f"\nShuffle control:")
    best_layer = probe_layers[-2]  # 75% depth
    for feat_name, feat_all in [('K', all_K[best_layer]), ('V', all_V[best_layer])]:
        train_X = feat_all[train_mask]
        test_X = feat_all[test_mask]
        shuffled_y = train_y_text.copy()
        np.random.shuffle(shuffled_y)

        shuf_train_r, _ = run_probe_cv(train_X, shuffled_y, n_splits=5)

        scaler_s = StandardScaler()
        X_tr_s = scaler_s.fit_transform(train_X)
        ridge_s = RidgeCV(alphas=alphas)
        ridge_s.fit(X_tr_s, shuffled_y)

        if test_X.shape[0] > 0:
            X_te_s = scaler_s.transform(test_X)
            shuf_preds = ridge_s.predict(X_te_s)
            shuf_dist_c = np.abs(shuf_preds - y_correct[test_mask])
            shuf_dist_w = np.abs(shuf_preds - y_written[test_mask])
            shuf_align = float(np.mean(shuf_dist_c < shuf_dist_w))
        else:
            shuf_align = float('nan')

        results['probe_results'][f'shuffle_L{best_layer}_{feat_name}'] = {
            'train_r': shuf_train_r,
            'correct_alignment_rate': shuf_align,
        }
        print(f"  shuffle_L{best_layer}_{feat_name}: train_r={shuf_train_r:.3f}, alignment={shuf_align:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Permutation test for alignment rate significance
    # ═══════════════════════════════════════════════════════════════
    n_perm = 1000
    if is_error.sum() >= 5:
        print(f"\nPermutation test (n={n_perm}):")
        best_feat_name = None
        best_alignment = 0
        for key, res in results['probe_results'].items():
            if 'shuffle' not in key and 'text' not in key:
                if not np.isnan(res.get('correct_alignment_rate', float('nan'))):
                    if res['correct_alignment_rate'] > best_alignment:
                        best_alignment = res['correct_alignment_rate']
                        best_feat_name = key

        if best_feat_name is not None:
            # Observed alignment rate
            obs_rate = results['probe_results'][best_feat_name]['correct_alignment_rate']

            # Pre-compute predictions once (training data doesn't change)
            preds_best = probe_data[best_feat_name]
            test_correct_perm = y_correct[test_mask]
            test_written_perm = y_written[test_mask]

            # Under null: shuffle correct/written labels
            n_errors = int(is_error.sum())
            null_rates = []
            for _ in range(n_perm):
                # Randomly assign which is "correct" and which is "written"
                swap = np.random.binomial(1, 0.5, n_errors).astype(bool)
                null_correct = test_correct_perm.copy()
                null_written = test_written_perm.copy()
                null_correct[swap], null_written[swap] = null_written[swap], null_correct[swap]

                dist_c = np.abs(preds_best - null_correct)
                dist_w = np.abs(preds_best - null_written)
                null_rates.append(np.mean(dist_c < dist_w))

            p_value = float(np.mean(np.array(null_rates) >= obs_rate))
            results['permutation_test'] = {
                'feature': best_feat_name,
                'observed_alignment': obs_rate,
                'null_mean': float(np.mean(null_rates)),
                'null_std': float(np.std(null_rates)),
                'p_value': p_value,
                'n_permutations': n_perm,
            }
            print(f"  {best_feat_name}: observed={obs_rate:.3f}, "
                  f"null={np.mean(null_rates):.3f}±{np.std(null_rates):.3f}, "
                  f"p={p_value:.4f}")

    # Save error examples
    results['error_examples'] = error_examples

    # Save metadata for each operation
    results['all_metadata'] = all_meta

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 6: Generate figures
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 6: Generating figures")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: Alignment rate comparison across layers and features
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    layer_labels = [f'L{l}' for l in probe_layers]
    k_aligns = [results['probe_results'].get(f'L{l}_K', {}).get('correct_alignment_rate', float('nan'))
                for l in probe_layers]
    v_aligns = [results['probe_results'].get(f'L{l}_V', {}).get('correct_alignment_rate', float('nan'))
                for l in probe_layers]
    text_align_val = results['probe_results'].get('text', {}).get('correct_alignment_rate', float('nan'))

    x = np.arange(len(probe_layers))
    width = 0.25
    bars_k = ax.bar(x - width, k_aligns, width, label='K-probe', color='#2196F3', alpha=0.8)
    bars_v = ax.bar(x, v_aligns, width, label='V-probe', color='#FF9800', alpha=0.8)
    bars_t = ax.bar(x + width, [text_align_val]*len(probe_layers), width,
                    label='Text-probe', color='#4CAF50', alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax.set_xlabel('Probe Layer')
    ax.set_ylabel('Correct-Alignment Rate')
    ax.set_title(f'WRRA: Does KV cache encode CORRECT value at arithmetic error positions?\n'
                 f'(n_errors={n_ops_error}, n_wrra={n_wrra}, n_wrwa={n_wrwa})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}\n({100*l/n_layers:.0f}%)' for l in probe_layers])
    ax.legend()
    ax.set_ylim(0, 1.05)
    for bars in [bars_k, bars_v, bars_t]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.2f}',
                        ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'alignment_by_layer.png', dpi=150)
    plt.close()

    # Figure 2: Training R (probe quality) comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    k_train_r = [results['probe_results'].get(f'L{l}_K', {}).get('train_r', float('nan'))
                 for l in probe_layers]
    v_train_r = [results['probe_results'].get(f'L{l}_V', {}).get('train_r', float('nan'))
                 for l in probe_layers]
    text_train_r_val = results['probe_results'].get('text', {}).get('train_r', float('nan'))

    ax.bar(x - width, k_train_r, width, label='K-probe', color='#2196F3', alpha=0.8)
    ax.bar(x, v_train_r, width, label='V-probe', color='#FF9800', alpha=0.8)
    ax.bar(x + width, [text_train_r_val]*len(probe_layers), width,
           label='Text-probe', color='#4CAF50', alpha=0.8)
    ax.set_xlabel('Probe Layer')
    ax.set_ylabel('Training R (cross-validated)')
    ax.set_title('Probe quality: Prediction of arithmetic values from KV/text features\n'
                 '(trained on correct-arithmetic positions)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}\n({100*l/n_layers:.0f}%)' for l in probe_layers])
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'training_r_by_layer.png', dpi=150)
    plt.close()

    # Figure 3: WRRA vs WRWA alignment comparison (if both exist)
    if n_wrra > 0 and n_wrwa > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        wrra_aligns = [results['probe_results'].get(f'L{l}_V', {}).get('wrra_alignment', float('nan'))
                       for l in probe_layers]
        wrwa_aligns = [results['probe_results'].get(f'L{l}_V', {}).get('wrwa_alignment', float('nan'))
                       for l in probe_layers]
        ax.bar(x - width/2, wrra_aligns, width, label='WRRA (correct final)', color='#4CAF50', alpha=0.8)
        ax.bar(x + width/2, wrwa_aligns, width, label='WRWA (wrong final)', color='#f44336', alpha=0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Probe Layer')
        ax.set_ylabel('Correct-Alignment Rate (V-probe)')
        ax.set_title(f'V-probe alignment: WRRA ({n_wrra} ops) vs WRWA ({n_wrwa} ops)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in probe_layers])
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'wrra_vs_wrwa_alignment.png', dpi=150)
        plt.close()

    # Figure 4: Scatter plot — predicted vs actual at error positions (best layer V-probe)
    best_layer_v = probe_layers[-2]  # 75% depth
    v_key = f'L{best_layer_v}_V'
    if v_key in probe_data and len(probe_data[v_key]) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        test_correct_y = y_correct[test_mask]
        test_written_y = y_written[test_mask]
        preds_v = probe_data[v_key]

        # Left: predicted vs correct
        ax = axes[0]
        wrra_m = is_wrra[test_mask]
        wrwa_m = is_wrwa[test_mask]
        if wrra_m.sum() > 0:
            ax.scatter(test_correct_y[wrra_m], preds_v[wrra_m], c='green', alpha=0.6, label=f'WRRA (n={wrra_m.sum()})')
        if wrwa_m.sum() > 0:
            ax.scatter(test_correct_y[wrwa_m], preds_v[wrwa_m], c='red', alpha=0.6, label=f'WRWA (n={wrwa_m.sum()})')
        lims = [min(test_correct_y.min(), preds_v.min()), max(test_correct_y.max(), preds_v.max())]
        ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
        ax.set_xlabel('Correct value (log-transformed)')
        ax.set_ylabel('Probe prediction')
        ax.set_title(f'V-probe (L{best_layer_v}) vs CORRECT value')
        ax.legend()

        # Right: predicted vs written
        ax = axes[1]
        if wrra_m.sum() > 0:
            ax.scatter(test_written_y[wrra_m], preds_v[wrra_m], c='green', alpha=0.6, label=f'WRRA (n={wrra_m.sum()})')
        if wrwa_m.sum() > 0:
            ax.scatter(test_written_y[wrwa_m], preds_v[wrwa_m], c='red', alpha=0.6, label=f'WRWA (n={wrwa_m.sum()})')
        lims = [min(test_written_y.min(), preds_v.min()), max(test_written_y.max(), preds_v.max())]
        ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
        ax.set_xlabel('Written value (log-transformed)')
        ax.set_ylabel('Probe prediction')
        ax.set_title(f'V-probe (L{best_layer_v}) vs WRITTEN value')
        ax.legend()

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'scatter_pred_vs_actual.png', dpi=150)
        plt.close()

    # Figure 5: Summary bar chart — key metrics
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    best_k_align = max([results['probe_results'].get(f'L{l}_K', {}).get('correct_alignment_rate', 0)
                        for l in probe_layers])
    best_v_align = max([results['probe_results'].get(f'L{l}_V', {}).get('correct_alignment_rate', 0)
                        for l in probe_layers])
    categories = ['K-probe\n(best layer)', 'V-probe\n(best layer)', 'Text-probe']
    values = [best_k_align, best_v_align, text_align_val]
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylabel('Correct-Alignment Rate')
    ax.set_title(f'At arithmetic errors, does probe predict correct or written value?\n'
                 f'(n={n_ops_error} errors across {n_with_ops} problems)')
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}',
                    ha='center', va='bottom', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'summary_alignment.png', dpi=150)
    plt.close()

    print(f"Figures saved to {RESULTS_DIR}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 7: Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Arithmetic operations: {n_ops_total} ({n_ops_correct} correct, {n_ops_error} errors)")
    print(f"Error rate: {100*n_ops_error/n_ops_total:.1f}%")
    print(f"WRRA: {n_wrra}, WRWA: {n_wrwa}")
    print(f"\nBest alignment rates (correct-aligned at error positions):")
    print(f"  K-probe (best): {best_k_align:.3f}")
    print(f"  V-probe (best): {best_v_align:.3f}")
    print(f"  Text-probe:     {text_align_val:.3f}")
    print(f"  Chance:         0.500")
    if 'permutation_test' in results:
        pt = results['permutation_test']
        print(f"\nPermutation test: p={pt['p_value']:.4f} (observed={pt['observed_alignment']:.3f}, "
              f"null={pt['null_mean']:.3f}±{pt['null_std']:.3f})")
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
