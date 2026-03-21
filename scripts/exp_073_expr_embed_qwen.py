#!/usr/bin/env python3
"""
Experiment 073: Expression-Embedding Retroactive Baseline on Qwen + Permutation Significance

Pre-registered retroactive test: does expression-mean embedding narrow the KV advantage
at computation positions on Qwen3-4B-Base? Exp_072 showed it narrows Mistral from +0.85
to +0.125. If the same happens on Qwen, our strongest Phase 2 finding weakens.

Also adds 1000-shuffle permutation tests for statistical significance of all probes.

Method:
1. Generate 8-shot CoT on GSM8K with <<EXPR=RESULT>> format
2. Parse arithmetic operations, extract KV + text features at "=" positions
3. Train ridge probes: K, V, text-at-"=", expression-mean-embedding -> log(result)
4. Permutation test (1000 shuffles) for significance of each probe and V-expr difference
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

TIME_BUDGET = 6000  # 100 min
MAX_GEN = 512
MAX_SEQ_LEN = 2048
MODEL_NAME = 'Qwen/Qwen3-4B-Base'
N_PROBLEMS_MAX = 500
N_PERM = 1000  # permutation shuffles

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_073"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 8-shot exemplars (same as exp_071/072) ──
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
        eq_char_pos = m.start() + len('<<') + len(expr_str)
        expr_start_char = m.start() + len('<<')
        expr_end_char = eq_char_pos
        ops.append({
            'expr': expr_str,
            'written': written,
            'correct': correct,
            'is_correct': abs(correct - written) < 0.5,
            'eq_char_pos': eq_char_pos,
            'expr_start_char': expr_start_char,
            'expr_end_char': expr_end_char,
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


def permutation_test(X, y, n_perm=1000, n_splits=5):
    """Permutation test: shuffle y, refit probes, build null distribution of R.
    Returns (observed_r, p_value, null_distribution)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    if X.shape[0] < max(n_splits, 5):
        return 0.0, 1.0, []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    alphas = np.logspace(-2, 6, 50)

    # Observed R
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in kf.split(X_scaled):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(X_scaled[train_idx], y[train_idx])
        y_pred[test_idx] = ridge.predict(X_scaled[test_idx])
    obs_r, _ = stats.pearsonr(y, y_pred)

    # Null distribution
    rng = np.random.RandomState(SEED + 1)
    null_rs = []
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        kf_shuf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED + 2 + i)
        y_pred_shuf = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in kf_shuf.split(X_scaled):
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(X_scaled[train_idx], y_shuf[train_idx])
            y_pred_shuf[test_idx] = ridge.predict(X_scaled[test_idx])
        r_shuf, _ = stats.pearsonr(y_shuf, y_pred_shuf)
        null_rs.append(float(r_shuf))

    p_value = float(np.mean(np.array(null_rs) >= obs_r))
    return float(obs_r), p_value, null_rs


def paired_permutation_test(X1, X2, y, n_perm=1000, n_splits=5):
    """Test if X1 probe is significantly better than X2 probe.
    Shuffles assignment of (X1, X2) to features, measures R1-R2 under null.
    Returns (obs_diff, p_value)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    if X1.shape[0] < max(n_splits, 5):
        return 0.0, 1.0

    # Observed R difference
    r1, _ = run_probe_cv(X1, y, n_splits)
    r2, _ = run_probe_cv(X2, y, n_splits)
    obs_diff = r1 - r2

    # Under null: randomly swap which features go to "probe 1" vs "probe 2"
    rng = np.random.RandomState(SEED + 100)
    null_diffs = []
    for i in range(n_perm):
        swap = rng.random(X1.shape[0]) > 0.5
        X1_perm = X1.copy()
        X2_perm = X2.copy()
        X1_perm[swap] = X2[swap]
        X2_perm[swap] = X1[swap]
        r1_p, _ = run_probe_cv(X1_perm, y, n_splits)
        r2_p, _ = run_probe_cv(X2_perm, y, n_splits)
        null_diffs.append(r1_p - r2_p)

    p_value = float(np.mean(np.array(null_diffs) >= obs_diff))
    return obs_diff, p_value


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
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    probe_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    print(f"Model: {n_layers} layers, hidden_size={hidden_size}")
    print(f"Probe layers: {probe_layers}")

    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = hidden_size // model.config.num_attention_heads
    kv_dim = num_kv_heads * head_dim
    print(f"KV heads: {num_kv_heads}, head_dim: {head_dim}, KV dim: {kv_dim}")
    print(f"Text dim: {hidden_size}")

    ds = load_gsm8k()
    print(f"GSM8K: {len(ds)} test problems")

    # Determine embed_tokens
    if hasattr(model.model, 'embed_tokens'):
        embed_fn = model.model.embed_tokens
    elif hasattr(model, 'get_input_embeddings'):
        embed_fn = model.get_input_embeddings()
    else:
        raise RuntimeError("Cannot find embedding layer")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Generate CoT & extract features
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 1: Generate CoT and extract features")
    print(f"{'='*60}")

    all_K = {l: [] for l in probe_layers}
    all_V = {l: [] for l in probe_layers}
    all_text_emb = []       # token embedding at "=" (weak baseline)
    all_expr_emb = []       # mean of expression token embeddings (strong baseline)
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

        ops = parse_arithmetic(gen_text_for_ops)
        if len(ops) == 0:
            del output
            gc.collect()
            torch.cuda.empty_cache()
            continue

        n_with_ops += 1

        token_starts, total_chars = build_token_starts(gen_ids, tokenizer)

        # Forward pass to get full KV cache
        full_ids = output.sequences[0:1]
        with torch.no_grad():
            out = model(full_ids, use_cache=True)
        past_kv = out.past_key_values

        # Token embeddings
        with torch.no_grad():
            gen_token_ids = full_ids[0, prompt_len:]
            text_emb_all = embed_fn(gen_token_ids).cpu().float()

        for op in ops:
            tok_pos_gen_eq = char_to_token(op['eq_char_pos'], token_starts)
            tok_pos_full_eq = tok_pos_gen_eq + prompt_len

            if tok_pos_full_eq >= full_ids.shape[1] or tok_pos_gen_eq >= text_emb_all.shape[0]:
                continue

            # KV features at "="
            for layer_idx in probe_layers:
                k, v = get_kv(past_kv, layer_idx)
                k_feat = k[0, :, tok_pos_full_eq, :].cpu().float().reshape(-1).numpy()
                v_feat = v[0, :, tok_pos_full_eq, :].cpu().float().reshape(-1).numpy()
                all_K[layer_idx].append(k_feat)
                all_V[layer_idx].append(v_feat)

            # Weak text baseline: token embedding at "="
            t_feat = text_emb_all[tok_pos_gen_eq].numpy()
            all_text_emb.append(t_feat)

            # Strong text baseline: mean of expression token embeddings
            tok_pos_gen_start = char_to_token(op['expr_start_char'], token_starts)
            tok_pos_gen_end = tok_pos_gen_eq
            if tok_pos_gen_start < tok_pos_gen_end and tok_pos_gen_end <= text_emb_all.shape[0]:
                expr_tokens_emb = text_emb_all[tok_pos_gen_start:tok_pos_gen_end]
                expr_mean = expr_tokens_emb.mean(dim=0).numpy()
            else:
                expr_mean = t_feat.copy()
            all_expr_emb.append(expr_mean)

            all_correct_vals.append(op['correct'])
            all_written_vals.append(op['written'])

            meta = {
                'prob_idx': prob_idx,
                'is_correct': op['is_correct'],
                'final_correct': final_correct,
                'expr': op['expr'],
                'written': op['written'],
                'correct': op['correct'],
            }
            all_meta.append(meta)

            n_ops_total += 1
            if op['is_correct']:
                n_ops_correct += 1
            else:
                n_ops_error += 1

        del output, out, past_kv, text_emb_all
        gc.collect()
        torch.cuda.empty_cache()

        if n_attempted % 50 == 0:
            print(f"  {n_attempted} attempted, {n_generated} generated, "
                  f"{n_ops_total} ops ({n_ops_correct} correct, {n_ops_error} errors)")

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

    if n_ops_total < 10:
        print("INSUFFICIENT DATA — fewer than 10 operations.")
        results = {'n_ops_total': n_ops_total, 'conclusion': 'Insufficient data'}
        with open(RESULTS_DIR / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return

    # Free model memory before probing
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Model freed, starting probes ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Probe analysis with permutation tests
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 2: Probe analysis + permutation tests ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")

    for l in probe_layers:
        all_K[l] = np.array(all_K[l])
        all_V[l] = np.array(all_V[l])
    all_text_emb = np.array(all_text_emb)
    all_expr_emb = np.array(all_expr_emb)
    correct_vals = np.array(all_correct_vals)

    is_correct_arr = np.array([m['is_correct'] for m in all_meta])
    train_mask = is_correct_arr
    y_train = signed_log(correct_vals[train_mask])

    print(f"Feature shapes: K={all_K[probe_layers[0]].shape}, "
          f"text={all_text_emb.shape}, expr={all_expr_emb.shape}")
    print(f"Correct positions (training): {train_mask.sum()}")

    results = {
        'n_attempted': n_attempted, 'n_generated': n_generated,
        'n_ops_total': int(n_ops_total), 'n_ops_correct': int(n_ops_correct),
        'n_ops_error': int(n_ops_error),
        'n_final_correct': int(n_final_correct), 'n_final_wrong': int(n_final_wrong),
        'accuracy': float(n_final_correct / max(n_generated, 1)),
        'error_rate': float(n_ops_error / n_ops_total) if n_ops_total > 0 else 0.0,
        'probe_results': {},
        'permutation_tests': {},
    }

    # ── Standard probe R (same as exp_071/072 for comparability) ──
    print(f"\n--- Cross-validated probe R (on correct-arithmetic positions) ---")
    print(f"{'Layer':>8} {'K R':>8} {'V R':>8} {'Text R':>8} {'Expr R':>8} {'Shuf K':>8} {'Shuf V':>8}")

    best_layer = None
    best_v_r = -1

    for layer_idx in probe_layers:
        layer_pct = int(100 * (layer_idx + 1) / n_layers)

        k_r, k_p = run_probe_cv(all_K[layer_idx][train_mask], y_train)
        v_r, v_p = run_probe_cv(all_V[layer_idx][train_mask], y_train)
        t_r, t_p = run_probe_cv(all_text_emb[train_mask], y_train)
        e_r, e_p = run_probe_cv(all_expr_emb[train_mask], y_train)

        # Shuffle controls
        rng = np.random.RandomState(SEED + layer_idx)
        y_shuf = rng.permutation(y_train)
        sk_r, _ = run_probe_cv(all_K[layer_idx][train_mask], y_shuf)
        sv_r, _ = run_probe_cv(all_V[layer_idx][train_mask], y_shuf)

        print(f"L{layer_idx:>3} ({layer_pct:>2}%) {k_r:>+8.3f} {v_r:>+8.3f} {t_r:>+8.3f} "
              f"{e_r:>+8.3f} {sk_r:>+8.3f} {sv_r:>+8.3f}")

        results['probe_results'][f'L{layer_idx}'] = {
            'layer_pct': layer_pct,
            'K_r': k_r, 'V_r': v_r, 'text_r': t_r, 'expr_r': e_r,
            'shuffle_K_r': sk_r, 'shuffle_V_r': sv_r,
            'V_minus_expr': v_r - e_r,
            'V_minus_text': v_r - t_r,
            'K_minus_expr': k_r - e_r,
        }

        if v_r > best_v_r:
            best_v_r = v_r
            best_layer = layer_idx

    print(f"\nBest layer: L{best_layer} ({int(100*(best_layer+1)/n_layers)}% depth)")
    best_res = results['probe_results'][f'L{best_layer}']
    print(f"  V-probe:   R = {best_res['V_r']:.3f}")
    print(f"  K-probe:   R = {best_res['K_r']:.3f}")
    print(f"  Expr-embed: R = {best_res['expr_r']:.3f}")
    print(f"  Token-embed: R = {best_res['text_r']:.3f}")
    print(f"  V - expr gap: {best_res['V_minus_expr']:+.3f}")
    print(f"  V - text gap: {best_res['V_minus_text']:+.3f}")

    results['best_layer'] = best_layer
    results['best_layer_pct'] = int(100 * (best_layer + 1) / n_layers)

    # ── Permutation significance tests (at best layer) ──
    print(f"\n--- Permutation tests (N={N_PERM}) at best layer L{best_layer} ---")
    t_perm_start = time.time()

    # Check time budget — permutation tests are expensive
    elapsed = time.time() - t0
    remaining = TIME_BUDGET - elapsed
    # Estimate: each permutation ≈ 0.1s for ridge on ~1000 samples
    est_time_per_probe = N_PERM * 0.15  # conservative
    n_probes_to_test = 4  # V, K, expr, text
    est_total = est_time_per_probe * n_probes_to_test

    if remaining < est_total + 120:  # need 2 min buffer for figures
        # Reduce permutation count
        n_perm_actual = max(100, int((remaining - 120) / (0.15 * n_probes_to_test)))
        print(f"  Time budget tight: reducing permutations to {n_perm_actual}")
    else:
        n_perm_actual = N_PERM

    # V-probe permutation test
    print(f"  Testing V-probe...")
    v_obs, v_perm_p, v_null = permutation_test(
        all_V[best_layer][train_mask], y_train, n_perm=n_perm_actual)
    print(f"    V-probe: R={v_obs:.3f}, p={v_perm_p:.4f} (n_perm={n_perm_actual})")

    # K-probe permutation test
    print(f"  Testing K-probe...")
    k_obs, k_perm_p, k_null = permutation_test(
        all_K[best_layer][train_mask], y_train, n_perm=n_perm_actual)
    print(f"    K-probe: R={k_obs:.3f}, p={k_perm_p:.4f}")

    # Expression-embed permutation test
    print(f"  Testing expression-embed...")
    e_obs, e_perm_p, e_null = permutation_test(
        all_expr_emb[train_mask], y_train, n_perm=n_perm_actual)
    print(f"    Expr-embed: R={e_obs:.3f}, p={e_perm_p:.4f}")

    # Token-embed permutation test
    print(f"  Testing token-embed...")
    t_obs, t_perm_p, t_null = permutation_test(
        all_text_emb[train_mask], y_train, n_perm=n_perm_actual)
    print(f"    Token-embed: R={t_obs:.3f}, p={t_perm_p:.4f}")

    results['permutation_tests'] = {
        'n_perm': n_perm_actual,
        'V_probe': {'R': v_obs, 'p': v_perm_p},
        'K_probe': {'R': k_obs, 'p': k_perm_p},
        'expr_embed': {'R': e_obs, 'p': e_perm_p},
        'token_embed': {'R': t_obs, 'p': t_perm_p},
    }

    # Paired permutation test: V-probe vs expression-embed
    elapsed = time.time() - t0
    remaining = TIME_BUDGET - elapsed
    if remaining > 300:  # need 5 min
        n_paired = min(200, n_perm_actual)
        print(f"\n  Paired permutation test: V-probe vs expr-embed (N={n_paired})...")
        v_expr_diff, v_expr_p = paired_permutation_test(
            all_V[best_layer][train_mask], all_expr_emb[train_mask], y_train,
            n_perm=n_paired)
        print(f"    V - expr diff: {v_expr_diff:+.3f}, p={v_expr_p:.4f}")
        results['permutation_tests']['V_minus_expr'] = {
            'diff': v_expr_diff, 'p': v_expr_p, 'n_perm': n_paired
        }
    else:
        print(f"\n  Skipping paired test (time budget: {remaining:.0f}s remaining)")

    perm_time = time.time() - t_perm_start
    print(f"\nPermutation tests completed in {perm_time:.0f}s")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Figures
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 3: Generating figures ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: Main result — probe R by layer (matches exp_072 format)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    layers_pct = [int(100 * (l + 1) / n_layers) for l in probe_layers]

    k_rs = [results['probe_results'][f'L{l}']['K_r'] for l in probe_layers]
    v_rs = [results['probe_results'][f'L{l}']['V_r'] for l in probe_layers]
    t_rs = [results['probe_results'][f'L{l}']['text_r'] for l in probe_layers]
    e_rs = [results['probe_results'][f'L{l}']['expr_r'] for l in probe_layers]
    sk_rs = [results['probe_results'][f'L{l}']['shuffle_K_r'] for l in probe_layers]

    ax.plot(layers_pct, v_rs, 'o-', color='#e74c3c', linewidth=2.5, markersize=10, label='V-probe', zorder=5)
    ax.plot(layers_pct, k_rs, 's-', color='#3498db', linewidth=2.5, markersize=10, label='K-probe', zorder=5)
    ax.plot(layers_pct, e_rs, '^-', color='#2ecc71', linewidth=2.5, markersize=10, label='Expr-embed (strong baseline)')
    ax.plot(layers_pct, t_rs, 'D-', color='#95a5a6', linewidth=2, markersize=8, label='Token-embed (weak baseline)')
    ax.plot(layers_pct, sk_rs, 'x--', color='#bdc3c7', linewidth=1.5, markersize=8, label='Shuffle control')

    ax.set_xlabel('Layer depth (%)', fontsize=13)
    ax.set_ylabel('Pearson R (5-fold CV)', fontsize=13)
    ax.set_title(f'Exp 073: Computation-Position Probing — Qwen3-4B-Base\n'
                 f'Expression-Embedding Retroactive Baseline (n={train_mask.sum()} ops)', fontsize=14)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Annotate V-expr gap at best layer
    bl_pct = int(100 * (best_layer + 1) / n_layers)
    bl_v = best_res['V_r']
    bl_e = best_res['expr_r']
    ax.annotate(f'V-expr gap: {bl_v - bl_e:+.3f}',
                xy=(bl_pct, (bl_v + bl_e) / 2), fontsize=11, fontweight='bold',
                ha='left', va='center',
                xytext=(bl_pct + 3, (bl_v + bl_e) / 2),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'probe_r_by_layer.png', dpi=150)
    plt.close(fig)
    print(f"  Saved probe_r_by_layer.png")

    # Figure 2: Cross-model comparison (Qwen exp_073 vs Mistral exp_072)
    # Load exp_072 results if available
    exp072_path = RESULTS_DIR.parent / 'exp_072' / 'results.json'
    if exp072_path.exists():
        with open(exp072_path) as f:
            exp072 = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Qwen (this experiment)
        ax = axes[0]
        bar_labels = ['V-probe', 'K-probe', 'Expr-embed', 'Token-embed']
        bar_vals = [best_res['V_r'], best_res['K_r'], best_res['expr_r'], best_res['text_r']]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']
        bars = ax.bar(bar_labels, bar_vals, color=colors, edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_title(f'Qwen3-4B-Base (exp_073)\nBest layer: L{best_layer} ({bl_pct}%)', fontsize=13)
        ax.set_ylabel('Pearson R', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

        # Right: Mistral (exp_072)
        ax = axes[1]
        # Find best layer from exp_072
        mistral_best_v = -1
        mistral_best_key = None
        for key, vals in exp072.get('probe_results', {}).items():
            if vals['V_r'] > mistral_best_v:
                mistral_best_v = vals['V_r']
                mistral_best_key = key
        if mistral_best_key:
            m_res = exp072['probe_results'][mistral_best_key]
            m_vals = [m_res['V_r'], m_res['K_r'], m_res['expr_r'], m_res['text_r']]
            bars = ax.bar(bar_labels, m_vals, color=colors, edgecolor='black', linewidth=0.8)
            for bar, val in zip(bars, m_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax.set_title(f'Mistral-7B-v0.3 (exp_072)\nBest layer: {mistral_best_key} ({m_res["layer_pct"]}%)', fontsize=13)
        else:
            ax.set_title('Mistral-7B-v0.3 (exp_072)\n(data not available)', fontsize=13)
        ax.set_ylabel('Pearson R', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

        fig.suptitle('Cross-Model: Expression-Embedding Baseline at Computation Positions', fontsize=15, y=1.02)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'cross_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved cross_model_comparison.png")

    # Figure 3: Permutation test null distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    probe_data = [
        ('V-probe', v_obs, v_null, v_perm_p, '#e74c3c'),
        ('K-probe', k_obs, k_null, k_perm_p, '#3498db'),
        ('Expr-embed', e_obs, e_null, e_perm_p, '#2ecc71'),
        ('Token-embed', t_obs, t_null, t_perm_p, '#95a5a6'),
    ]

    for ax, (name, obs_r, null_dist, p_val, color) in zip(axes.flat, probe_data):
        if null_dist:
            ax.hist(null_dist, bins=50, color=color, alpha=0.6, edgecolor='black', linewidth=0.5)
            ax.axvline(obs_r, color='red', linewidth=2.5, linestyle='--', label=f'Observed R={obs_r:.3f}')
            ax.set_title(f'{name}: p={p_val:.4f} (n_perm={n_perm_actual})', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(name, fontsize=12)
        ax.set_xlabel('Pearson R under null', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.legend(fontsize=10)

    fig.suptitle('Exp 073: Permutation Test Null Distributions (Qwen3-4B-Base)', fontsize=14)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'permutation_tests.png', dpi=150)
    plt.close(fig)
    print(f"  Saved permutation_tests.png")

    # Figure 4: V-expr advantage bar chart with significance
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    gaps = []
    gap_labels = []
    for l in probe_layers:
        lp = results['probe_results'][f'L{l}']
        gaps.append(lp['V_minus_expr'])
        gap_labels.append(f'L{l} ({lp["layer_pct"]}%)')

    bar_colors = ['#e74c3c' if g > 0.15 else '#f39c12' if g > 0 else '#95a5a6' for g in gaps]
    bars = ax.bar(gap_labels, gaps, color=bar_colors, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:+.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.axhline(0, color='gray', linewidth=1)
    ax.axhline(0.125, color='orange', linewidth=1.5, linestyle='--',
               label='Mistral V-expr gap (+0.125)')
    ax.set_ylabel('V-probe R minus Expr-embed R', fontsize=12)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_title(f'Exp 073: KV Advantage Over Expression Embedding\n'
                 f'Qwen3-4B-Base (n={train_mask.sum()} correct-arithmetic ops)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v_expr_advantage.png', dpi=150)
    plt.close(fig)
    print(f"  Saved v_expr_advantage.png")

    # Save results
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 073 COMPLETE ({total_time:.0f}s = {total_time/60:.1f} min)")
    print(f"{'='*60}")

    # Summary
    print(f"\n=== KEY FINDINGS ===")
    print(f"Best layer: L{best_layer} ({results['best_layer_pct']}%)")
    print(f"  V-probe R:    {best_res['V_r']:.3f}")
    print(f"  K-probe R:    {best_res['K_r']:.3f}")
    print(f"  Expr-embed R: {best_res['expr_r']:.3f}")
    print(f"  Token-embed R: {best_res['text_r']:.3f}")
    print(f"  V - expr gap: {best_res['V_minus_expr']:+.3f}")
    print(f"  V - text gap: {best_res['V_minus_text']:+.3f}")
    print(f"\nPermutation p-values:")
    print(f"  V-probe: p={v_perm_p:.4f}")
    print(f"  K-probe: p={k_perm_p:.4f}")
    print(f"  Expr-embed: p={e_perm_p:.4f}")
    print(f"  Token-embed: p={t_perm_p:.4f}")
    if 'V_minus_expr' in results.get('permutation_tests', {}):
        ve = results['permutation_tests']['V_minus_expr']
        print(f"  V - expr paired: diff={ve['diff']:+.3f}, p={ve['p']:.4f}")


if __name__ == '__main__':
    main()
