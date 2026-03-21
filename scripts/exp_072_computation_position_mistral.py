#!/usr/bin/env python3
"""
Experiment 072: Computation-Position KV Probing — Mistral Cross-Model Replication

Pre-registered replication of exp_071's computation-position finding on Mistral-7B-v0.3.
Adds enhanced text baseline (average expression token embeddings).

Method:
1. Generate 8-shot CoT on GSM8K with <<EXPR=RESULT>> format
2. Parse arithmetic operations, extract KV features at "=" positions
3. Train ridge probes: K, V, text-at-"=", expression-mean-embedding → log(result)
4. Compare probe R values across 4 layers
5. If enough errors, run WRRA alignment analysis
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
MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
N_PROBLEMS_MAX = 400

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_072"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 8-shot exemplars (same as exp_071) ──
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
        # Also store expression span for expression-embedding baseline
        expr_start_char = m.start() + len('<<')
        expr_end_char = eq_char_pos  # up to but not including "="
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

    # Determine KV dimension
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = hidden_size // model.config.num_attention_heads
    kv_dim = num_kv_heads * head_dim
    print(f"KV heads: {num_kv_heads}, head_dim: {head_dim}, KV dim: {kv_dim}")
    print(f"Text dim: {hidden_size}")

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
    n_wrra = 0
    n_wrwa = 0
    error_examples = []

    # Determine embed_tokens path for Mistral
    if hasattr(model.model, 'embed_tokens'):
        embed_fn = model.model.embed_tokens
    elif hasattr(model, 'get_input_embeddings'):
        embed_fn = model.get_input_embeddings()
    else:
        raise RuntimeError("Cannot find embedding layer")

    for prob_idx in range(min(N_PROBLEMS_MAX, len(ds))):
        elapsed = time.time() - t0
        if elapsed > TIME_BUDGET * 0.55:
            print(f"  Time budget (55%): stopping at {n_attempted} problems ({elapsed:.0f}s)")
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

        # Truncate at #### for parsing
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

        # Parse arithmetic operations
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

        # Token embeddings for text baselines
        with torch.no_grad():
            gen_token_ids = full_ids[0, prompt_len:]
            text_emb_all = embed_fn(gen_token_ids).cpu().float()

        # Extract features at each arithmetic position
        for op in ops:
            tok_pos_gen_eq = char_to_token(op['eq_char_pos'], token_starts)
            tok_pos_full_eq = tok_pos_gen_eq + prompt_len

            if tok_pos_full_eq >= full_ids.shape[1] or tok_pos_gen_eq >= text_emb_all.shape[0]:
                continue

            # KV features at "=" position
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
            tok_pos_gen_end = tok_pos_gen_eq  # "=" position (exclusive upper bound for expr)
            if tok_pos_gen_start < tok_pos_gen_end and tok_pos_gen_end <= text_emb_all.shape[0]:
                expr_tokens_emb = text_emb_all[tok_pos_gen_start:tok_pos_gen_end]
                expr_mean = expr_tokens_emb.mean(dim=0).numpy()
            else:
                # Fallback: just use the "=" embedding
                expr_mean = t_feat.copy()
            all_expr_emb.append(expr_mean)

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
        del output, out, past_kv, text_emb_all
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
    print(f"  WRRA: {n_wrra}, WRWA: {n_wrwa}")

    if n_ops_total == 0:
        print("NO OPERATIONS FOUND — Mistral may not follow <<EXPR=RESULT>> format.")
        results = {
            'n_attempted': n_attempted, 'n_generated': n_generated,
            'n_ops_total': 0, 'error_rate': None,
            'conclusion': 'No arithmetic operations found. Model may not follow format.',
        }
        with open(RESULTS_DIR / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return

    if error_examples:
        print(f"\n--- Example arithmetic errors ---")
        for ex in error_examples[:10]:
            label = "WRRA" if ex['final_correct'] else "WRWA"
            print(f"  [{label}] {ex['expr']} = {ex['correct']:.1f} (wrote {ex['written']:.1f})")
            print(f"    Context: ...{ex['context']}...")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Probe analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 2: Probe analysis ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")

    for l in probe_layers:
        all_K[l] = np.array(all_K[l])
        all_V[l] = np.array(all_V[l])
    all_text_emb = np.array(all_text_emb)
    all_expr_emb = np.array(all_expr_emb)
    correct_vals = np.array(all_correct_vals)
    written_vals = np.array(all_written_vals)

    is_correct_arr = np.array([m['is_correct'] for m in all_meta])
    is_wrra = np.array([m['wrra'] for m in all_meta])
    is_wrwa = np.array([m['wrwa'] for m in all_meta])
    is_error = ~is_correct_arr

    y_correct = signed_log(correct_vals)
    y_written = signed_log(written_vals)

    print(f"Feature shapes: K={all_K[probe_layers[0]].shape}, "
          f"text={all_text_emb.shape}, expr={all_expr_emb.shape}")
    print(f"Correct positions: {is_correct_arr.sum()}, Error positions: {is_error.sum()}")
    print(f"WRRA: {is_wrra.sum()}, WRWA: {is_wrwa.sum()}")

    # ── Train probes on ALL positions (training R comparison) ──
    # Use all positions where is_correct (= written matches correct)
    # For the primary analysis, train on correct-arithmetic positions
    train_mask = is_correct_arr
    y_train = y_correct[train_mask]

    results = {
        'n_attempted': n_attempted, 'n_generated': n_generated,
        'n_ops_total': int(n_ops_total), 'n_ops_correct': int(n_ops_correct),
        'n_ops_error': int(n_ops_error), 'n_wrra': int(n_wrra), 'n_wrwa': int(n_wrwa),
        'error_rate': float(n_ops_error / n_ops_total) if n_ops_total > 0 else 0.0,
        'n_final_correct': int(n_final_correct), 'n_final_wrong': int(n_final_wrong),
        'accuracy': float(n_final_correct / max(n_generated, 1)),
        'probe_results': {},
    }

    print(f"\n--- Cross-validated probe R (on correct-arithmetic positions) ---")
    print(f"{'Layer':>8} {'K R':>8} {'V R':>8} {'Text R':>8} {'Expr R':>8} {'Shuf K':>8} {'Shuf V':>8}")

    for layer_idx in probe_layers:
        layer_pct = int(100 * (layer_idx + 1) / n_layers)

        # K-probe
        k_r, k_p = run_probe_cv(all_K[layer_idx][train_mask], y_train)
        # V-probe
        v_r, v_p = run_probe_cv(all_V[layer_idx][train_mask], y_train)
        # Text-at-"=" (weak baseline)
        text_r, text_p = run_probe_cv(all_text_emb[train_mask], y_train)
        # Expression-mean (strong baseline)
        expr_r, expr_p = run_probe_cv(all_expr_emb[train_mask], y_train)
        # Shuffle control: K features paired with shuffled y
        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        shuf_k_r, _ = run_probe_cv(all_K[layer_idx][train_mask], y_shuffled)
        shuf_v_r, _ = run_probe_cv(all_V[layer_idx][train_mask], y_shuffled)

        print(f"L{layer_idx:2d} ({layer_pct:3d}%) {k_r:+.3f}   {v_r:+.3f}   {text_r:+.3f}   "
              f"{expr_r:+.3f}   {shuf_k_r:+.3f}   {shuf_v_r:+.3f}")

        results['probe_results'][f'L{layer_idx}'] = {
            'layer_idx': layer_idx,
            'layer_pct': layer_pct,
            'K_r': k_r, 'K_p': k_p,
            'V_r': v_r, 'V_p': v_p,
            'text_r': text_r, 'text_p': text_p,
            'expr_r': expr_r, 'expr_p': expr_p,
            'shuffle_K_r': shuf_k_r,
            'shuffle_V_r': shuf_v_r,
            'K_minus_text': k_r - text_r,
            'V_minus_text': v_r - text_r,
            'K_minus_expr': k_r - expr_r,
            'V_minus_expr': v_r - expr_r,
            'V_minus_K': v_r - k_r,
        }

    # ── WRRA alignment analysis (if enough errors) ──
    wrra_results = None
    if n_ops_error >= 5:
        print(f"\n--- WRRA alignment analysis (n_errors={n_ops_error}) ---")
        wrra_results = {}

        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        alphas = np.logspace(-2, 6, 50)

        test_mask = is_error

        for layer_idx in probe_layers:
            layer_pct = int(100 * (layer_idx + 1) / n_layers)
            layer_wrra = {}

            for feat_name, feat_all in [('K', all_K[layer_idx]), ('V', all_V[layer_idx]),
                                         ('text', all_text_emb), ('expr', all_expr_emb)]:
                train_X = feat_all[train_mask]
                train_y = y_correct[train_mask]
                test_X = feat_all[test_mask]

                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(train_X)
                ridge = RidgeCV(alphas=alphas)
                ridge.fit(X_train_sc, train_y)

                X_test_sc = scaler.transform(test_X)
                preds = ridge.predict(X_test_sc)

                test_correct_y = y_correct[test_mask]
                test_written_y = y_written[test_mask]

                dist_correct = np.abs(preds - test_correct_y)
                dist_written = np.abs(preds - test_written_y)
                correct_aligned = dist_correct < dist_written
                alignment_rate = float(np.mean(correct_aligned))

                layer_wrra[feat_name] = {
                    'alignment_rate': alignment_rate,
                    'n_test': int(test_mask.sum()),
                    'preds': preds.tolist(),
                    'correct_vals': test_correct_y.tolist(),
                    'written_vals': test_written_y.tolist(),
                }

                print(f"  L{layer_idx} ({layer_pct}%) {feat_name:>5}: "
                      f"alignment={alignment_rate:.3f} (n={test_mask.sum()})")

            wrra_results[f'L{layer_idx}'] = layer_wrra

        results['wrra_results'] = wrra_results
    elif n_ops_error > 0:
        print(f"\nToo few errors ({n_ops_error}) for meaningful WRRA analysis.")
        results['wrra_note'] = f'Only {n_ops_error} errors found; WRRA underpowered'

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Generate figures
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 3: Generate figures ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: Training R by layer — main result
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    layers = sorted(results['probe_results'].keys(), key=lambda x: results['probe_results'][x]['layer_idx'])
    layer_labels = [f"L{results['probe_results'][l]['layer_idx']} ({results['probe_results'][l]['layer_pct']}%)"
                    for l in layers]

    k_vals = [results['probe_results'][l]['K_r'] for l in layers]
    v_vals = [results['probe_results'][l]['V_r'] for l in layers]
    text_vals = [results['probe_results'][l]['text_r'] for l in layers]
    expr_vals = [results['probe_results'][l]['expr_r'] for l in layers]
    shuf_k_vals = [results['probe_results'][l]['shuffle_K_r'] for l in layers]

    x = np.arange(len(layers))
    width = 0.17
    ax.bar(x - 1.5*width, k_vals, width, label='K-probe', color='#2196F3')
    ax.bar(x - 0.5*width, v_vals, width, label='V-probe', color='#4CAF50')
    ax.bar(x + 0.5*width, expr_vals, width, label='Expr-embed (strong text)', color='#FF9800')
    ax.bar(x + 1.5*width, text_vals, width, label='Token-embed (weak text)', color='#9E9E9E', alpha=0.7)
    # Shuffle as error-bar line
    ax.scatter(x - 1.5*width, shuf_k_vals, marker='x', color='red', s=80, zorder=5, label='Shuffle control')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Cross-validated R (Pearson)')
    ax.set_title(f'Computation-Position Probing: Mistral-7B-v0.3\n'
                 f'KV features vs text baselines at arithmetic "=" positions (n={n_ops_correct} ops)')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.legend(loc='lower right')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylim(-0.3, 1.1)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'training_r_by_layer.png', dpi=150)
    plt.close(fig)
    print(f"  Saved training_r_by_layer.png")

    # Figure 2: Cross-model comparison (Qwen exp_071 vs Mistral exp_072)
    qwen_results = {
        'L9': {'K_r': 0.936, 'V_r': 0.948, 'text_r': 0.108},
        'L18': {'K_r': 0.936, 'V_r': 0.955, 'text_r': 0.108},
        'L27': {'K_r': 0.954, 'V_r': 0.971, 'text_r': 0.108},
        'L35': {'K_r': 0.961, 'V_r': 0.975, 'text_r': 0.108},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Qwen panel
    ax = axes[0]
    qwen_layers = ['L9', 'L18', 'L27', 'L35']
    x_q = np.arange(len(qwen_layers))
    ax.bar(x_q - width, [qwen_results[l]['K_r'] for l in qwen_layers], width, label='K', color='#2196F3')
    ax.bar(x_q, [qwen_results[l]['V_r'] for l in qwen_layers], width, label='V', color='#4CAF50')
    ax.bar(x_q + width, [qwen_results[l]['text_r'] for l in qwen_layers], width, label='Text', color='#9E9E9E')
    ax.set_title('Qwen3-4B-Base (exp_071)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cross-validated R')
    ax.set_xticks(x_q)
    ax.set_xticklabels(['25%', '50%', '75%', '97%'])
    ax.legend()
    ax.set_ylim(-0.1, 1.1)

    # Mistral panel
    ax = axes[1]
    m_layers = sorted(results['probe_results'].keys(), key=lambda x: results['probe_results'][x]['layer_idx'])
    x_m = np.arange(len(m_layers))
    ax.bar(x_m - 1.5*width, [results['probe_results'][l]['K_r'] for l in m_layers], width, label='K', color='#2196F3')
    ax.bar(x_m - 0.5*width, [results['probe_results'][l]['V_r'] for l in m_layers], width, label='V', color='#4CAF50')
    ax.bar(x_m + 0.5*width, [results['probe_results'][l]['expr_r'] for l in m_layers], width, label='Expr-embed', color='#FF9800')
    ax.bar(x_m + 1.5*width, [results['probe_results'][l]['text_r'] for l in m_layers], width, label='Token-embed', color='#9E9E9E')
    ax.set_title('Mistral-7B-v0.3 (exp_072)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cross-validated R')
    ax.set_xticks(x_m)
    ax.set_xticklabels([f"{results['probe_results'][l]['layer_pct']}%" for l in m_layers])
    ax.legend()
    ax.set_ylim(-0.1, 1.1)

    fig.suptitle('Computation-Position Probing: Cross-Model Comparison', fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'cross_model_comparison.png', dpi=150)
    plt.close(fig)
    print(f"  Saved cross_model_comparison.png")

    # Figure 3: KV advantage over text baselines
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    k_minus_text = [results['probe_results'][l]['K_minus_text'] for l in layers]
    v_minus_text = [results['probe_results'][l]['V_minus_text'] for l in layers]
    k_minus_expr = [results['probe_results'][l]['K_minus_expr'] for l in layers]
    v_minus_expr = [results['probe_results'][l]['V_minus_expr'] for l in layers]

    ax.bar(x - 1.5*width, v_minus_text, width, label='V - token-embed', color='#4CAF50')
    ax.bar(x - 0.5*width, k_minus_text, width, label='K - token-embed', color='#2196F3')
    ax.bar(x + 0.5*width, v_minus_expr, width, label='V - expr-embed', color='#81C784')
    ax.bar(x + 1.5*width, k_minus_expr, width, label='K - expr-embed', color='#64B5F6')

    ax.set_xlabel('Layer')
    ax.set_ylabel('R advantage (KV minus text)')
    ax.set_title('KV Advantage Over Text Baselines at Computation Positions')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'kv_advantage_over_text.png', dpi=150)
    plt.close(fig)
    print(f"  Saved kv_advantage_over_text.png")

    # Save results
    # Convert any numpy values
    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    results = sanitize(results)
    results['error_examples'] = error_examples[:10]
    results['elapsed_seconds'] = time.time() - t0

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    best_layer = max(results['probe_results'].keys(),
                     key=lambda l: results['probe_results'][l]['V_r'])
    bl = results['probe_results'][best_layer]
    print(f"Best layer: {best_layer} ({bl['layer_pct']}% depth)")
    print(f"  V-probe R: {bl['V_r']:.3f}")
    print(f"  K-probe R: {bl['K_r']:.3f}")
    print(f"  Expr-embed R: {bl['expr_r']:.3f}")
    print(f"  Token-embed R: {bl['text_r']:.3f}")
    print(f"  V - expr gap: {bl['V_minus_expr']:+.3f}")
    print(f"  V - token gap: {bl['V_minus_text']:+.3f}")
    print(f"  Shuffle K: {bl['shuffle_K_r']:.3f}")
    print(f"\nArithmetic error rate: {results['error_rate']:.4f} ({n_ops_error}/{n_ops_total})")
    print(f"Model accuracy: {results['accuracy']:.1%}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
