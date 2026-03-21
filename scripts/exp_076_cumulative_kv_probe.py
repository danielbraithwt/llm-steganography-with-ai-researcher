#!/usr/bin/env python3
"""
Experiment 076: Cumulative KV Probe — Fair Apples-to-Apples Early Decodability

Fixes the exp_075 design flaw: single-position KV vs cumulative text.

Probes at each decile (10%–100% of CoT):
  1. cum_K: cumulative mean of K-vectors from CoT start to position P
  2. cum_V: cumulative mean of V-vectors from CoT start to position P
  3. cum_text: cumulative mean of token embeddings from CoT start to position P
  4. hidden_state: hidden state at position P (full transformer output at L35)
  5. single_V: V-vector at position P only (for comparison with exp_075)
  6. shuffle: permuted answers for validation

Target: log(|final_answer| + 1) × sign(answer)
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

TIME_BUDGET = 6000  # 100 min
MAX_GEN = 512
MAX_SEQ_LEN = 2048
MODEL_NAME = 'Qwen/Qwen3-4B-Base'
N_PROBLEMS_MAX = 300
N_DECILES = 10

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_076"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 8-shot exemplars (same as exp_071-075) ──
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
    """Train ridge probe with cross-validation, return Pearson R."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    if X.shape[0] < max(n_splits, 10):
        return 0.0

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
    return float(r)


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
    probe_layer = n_layers - 1  # L35 (100% depth) — focus on most computed layer
    print(f"Model: {n_layers} layers, hidden_size={hidden_size}")
    print(f"Probe layer: L{probe_layer} ({round(probe_layer/(n_layers-1)*100)}% depth)")

    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim_actual = hidden_size // model.config.num_attention_heads
    print(f"KV heads: {num_kv_heads}, head_dim: {head_dim_actual}")

    # Get embedding function
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_fn = model.model.embed_tokens
    elif hasattr(model, 'get_input_embeddings'):
        embed_fn = model.get_input_embeddings()
    else:
        raise RuntimeError("Cannot find embedding layer")

    ds = load_gsm8k()
    print(f"GSM8K: {len(ds)} test problems")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Generate CoT for all problems
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 1: Generate CoT")
    print(f"{'='*60}")

    gen_results = []
    for i, row in enumerate(ds):
        if i >= N_PROBLEMS_MAX:
            break
        if time.time() - t0 > TIME_BUDGET * 0.35:
            print(f"Time budget for generation: stopping at {i} problems")
            break

        question = row['question']
        gold = extract_gold(row['answer'])
        prompt = build_prompt(question)

        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN)
        prompt_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            gen = model.generate(
                **{k: v.to(model.device) for k, v in inputs.items()},
                max_new_tokens=MAX_GEN,
                do_sample=False,
                temperature=1.0,
            )

        gen_ids = gen[0][prompt_len:].cpu()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_answer(gen_text)

        correct = False
        if pred is not None and gold is not None:
            try:
                correct = abs(float(pred) - float(gold)) < 0.5
            except:
                pass

        gen_results.append({
            'idx': i,
            'question': question,
            'gold': gold,
            'pred': pred,
            'correct': correct,
            'gen_text': gen_text,
            'prompt_len': prompt_len,
            'gen_ids': gen_ids,
        })

        if (i + 1) % 50 == 0:
            n_corr = sum(1 for r in gen_results if r['correct'])
            elapsed = time.time() - t0
            print(f"  [{i+1}/{N_PROBLEMS_MAX}] Correct: {n_corr}/{i+1} "
                  f"({n_corr/(i+1)*100:.1f}%) [{elapsed:.0f}s]")

    n_total = len(gen_results)
    correct_results = [r for r in gen_results if r['correct']]
    n_correct = len(correct_results)
    print(f"\nGeneration complete: {n_correct}/{n_total} correct "
          f"({n_correct/n_total*100:.1f}%) [{time.time()-t0:.0f}s]")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Forward pass to extract CUMULATIVE KV + hidden states
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 2: Extract cumulative KV + hidden state features at {N_DECILES} deciles")
    print(f"{'='*60}")

    decile_fracs = np.linspace(0.1, 1.0, N_DECILES)

    # Determine KV dim from a test forward pass
    test_prompt = build_prompt(correct_results[0]['question'])
    test_text = test_prompt + correct_results[0]['gen_text']
    test_inputs = tokenizer(test_text, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN)
    with torch.no_grad():
        test_out = model(test_inputs['input_ids'].to(model.device), use_cache=True,
                         output_hidden_states=True)
    k_test, v_test = get_kv(test_out.past_key_values, probe_layer)
    kv_dim = k_test.shape[1] * k_test.shape[3]  # n_kv_heads * head_dim
    hs_dim = test_out.hidden_states[probe_layer + 1].shape[-1]  # hidden state dim
    print(f"KV dim: {kv_dim} ({k_test.shape[1]} heads x {k_test.shape[3]} head_dim)")
    print(f"Hidden state dim: {hs_dim}")
    print(f"Text embedding dim: {hidden_size}")
    del test_out, k_test, v_test
    torch.cuda.empty_cache()

    # Allocate storage for features at each decile
    cum_K_feat = np.zeros((n_correct, N_DECILES, kv_dim), dtype=np.float32)
    cum_V_feat = np.zeros((n_correct, N_DECILES, kv_dim), dtype=np.float32)
    cum_text_feat = np.zeros((n_correct, N_DECILES, hidden_size), dtype=np.float32)
    hs_feat = np.zeros((n_correct, N_DECILES, hs_dim), dtype=np.float32)
    single_V_feat = np.zeros((n_correct, N_DECILES, kv_dim), dtype=np.float32)
    answers = np.zeros(n_correct)
    cot_lengths = np.zeros(n_correct, dtype=int)
    valid_mask = np.ones(n_correct, dtype=bool)

    t_phase2 = time.time()
    for pi, r in enumerate(correct_results):
        if time.time() - t0 > TIME_BUDGET * 0.80:
            print(f"Time budget: stopping feature extraction at {pi}/{n_correct}")
            valid_mask[pi:] = False
            break

        prompt = build_prompt(r['question'])
        full_text = prompt + r['gen_text']

        inputs = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = inputs['input_ids'].to(model.device)
        seq_len = input_ids.shape[1]
        prompt_len = r['prompt_len']

        cot_start = prompt_len
        cot_end = seq_len
        cot_len = cot_end - cot_start

        if cot_len < 10:
            valid_mask[pi] = False
            continue

        cot_lengths[pi] = cot_len
        try:
            answers[pi] = signed_log(float(r['gold']))
        except:
            valid_mask[pi] = False
            continue

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True, output_hidden_states=True)
            past_kv = outputs.past_key_values

            # Hidden states: outputs.hidden_states is a tuple of (n_layers+1) tensors
            # Index [layer+1] gives the output of that layer (index [0] is embeddings)
            hs_layer = outputs.hidden_states[probe_layer + 1]  # [1, seq_len, hs_dim]
            hs_cot = hs_layer[0, cot_start:cot_end, :].float().cpu().numpy()  # [cot_len, hs_dim]

            # Token embeddings for CoT positions
            cot_ids = input_ids[0, cot_start:cot_end]
            token_embeds = embed_fn(cot_ids).float().cpu().numpy()  # [cot_len, hidden_size]

        # Extract KV at probe layer for CoT positions
        k_cache, v_cache = get_kv(past_kv, probe_layer)
        # Shape: [1, n_kv_heads, seq_len, head_dim]
        k_cot = k_cache[0, :, cot_start:cot_end, :].permute(1, 0, 2).reshape(cot_len, -1).float().cpu().numpy()
        v_cot = v_cache[0, :, cot_start:cot_end, :].permute(1, 0, 2).reshape(cot_len, -1).float().cpu().numpy()

        # Compute cumulative sums for cumulative mean computation
        k_cumsum = np.cumsum(k_cot, axis=0)  # [cot_len, kv_dim]
        v_cumsum = np.cumsum(v_cot, axis=0)
        text_cumsum = np.cumsum(token_embeds, axis=0)

        # Extract features at each decile position
        for di, frac in enumerate(decile_fracs):
            pos = min(int(frac * cot_len) - 1, cot_len - 1)
            pos = max(0, pos)

            # Cumulative means (avg of vectors 0:pos)
            cum_K_feat[pi, di] = k_cumsum[pos] / (pos + 1)
            cum_V_feat[pi, di] = v_cumsum[pos] / (pos + 1)
            cum_text_feat[pi, di] = text_cumsum[pos] / (pos + 1)

            # Hidden state at this position
            hs_feat[pi, di] = hs_cot[pos]

            # Single-position V (for exp_075 comparison)
            single_V_feat[pi, di] = v_cot[pos]

        # Free GPU memory
        del outputs, past_kv, k_cache, v_cache, hs_layer, hs_cot
        torch.cuda.empty_cache()

        if (pi + 1) % 25 == 0:
            elapsed = time.time() - t_phase2
            rate = elapsed / (pi + 1)
            remaining = rate * (n_correct - pi - 1)
            print(f"  [{pi+1}/{n_correct}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    # Filter to valid problems
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)
    answers_valid = answers[valid_idx]
    cot_lengths_valid = cot_lengths[valid_idx]
    print(f"\nFeature extraction complete: {n_valid} valid problems [{time.time()-t0:.0f}s]")
    print(f"Mean CoT length: {cot_lengths_valid.mean():.1f} tokens")
    print(f"Answer range: [{np.exp(np.abs(answers_valid).min())-1:.0f}, "
          f"{np.exp(np.abs(answers_valid).max())-1:.0f}]")

    # Free model memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Train probes at each decile
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 3: Train probes at each decile")
    print(f"{'='*60}")

    probe_results = {
        'cum_K': [], 'cum_V': [], 'cum_text': [],
        'hidden_state': [], 'single_V': [], 'shuffle': [],
    }

    print(f"\n  {'Decile':>6} {'cumK':>7} {'cumV':>7} {'cumTxt':>7} {'HidSt':>7} "
          f"{'singleV':>8} {'Shuffle':>8}")
    print(f"  {'-'*56}")

    for di in range(N_DECILES):
        X_ck = cum_K_feat[valid_idx, di]
        X_cv = cum_V_feat[valid_idx, di]
        X_ct = cum_text_feat[valid_idx, di]
        X_hs = hs_feat[valid_idx, di]
        X_sv = single_V_feat[valid_idx, di]
        y = answers_valid

        r_ck = run_probe_cv(X_ck, y)
        r_cv = run_probe_cv(X_cv, y)
        r_ct = run_probe_cv(X_ct, y)
        r_hs = run_probe_cv(X_hs, y)
        r_sv = run_probe_cv(X_sv, y)

        # Shuffle control
        rng = np.random.RandomState(SEED + di)
        y_shuf = rng.permutation(y)
        r_shuf = run_probe_cv(X_hs, y_shuf)  # shuffle on strongest probe

        probe_results['cum_K'].append(r_ck)
        probe_results['cum_V'].append(r_cv)
        probe_results['cum_text'].append(r_ct)
        probe_results['hidden_state'].append(r_hs)
        probe_results['single_V'].append(r_sv)
        probe_results['shuffle'].append(r_shuf)

        pct = int(decile_fracs[di] * 100)
        print(f"  {pct:>5}% {r_ck:>7.3f} {r_cv:>7.3f} {r_ct:>7.3f} {r_hs:>7.3f} "
              f"{r_sv:>8.3f} {r_shuf:>8.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3.5: Early decodability analysis
    # ═══════════════════════════════════════════════════════════════
    R_THRESHOLD = 0.3
    print(f"\n{'='*60}")
    print(f"EARLY DECODABILITY ANALYSIS (threshold R>{R_THRESHOLD})")
    print(f"{'='*60}")

    for probe_name in ['cum_K', 'cum_V', 'cum_text', 'hidden_state', 'single_V']:
        rs = probe_results[probe_name]
        first_above = None
        for di, r_val in enumerate(rs):
            if r_val >= R_THRESHOLD:
                first_above = di
                break
        if first_above is not None:
            print(f"  {probe_name:>14}: first R>{R_THRESHOLD} at decile {first_above+1} "
                  f"({int(decile_fracs[first_above]*100)}%), max R={max(rs):.3f}")
        else:
            print(f"  {probe_name:>14}: never reaches R>{R_THRESHOLD} (max R={max(rs):.3f})")

    # Gap computations
    def first_above_threshold(name):
        for di, r_val in enumerate(probe_results[name]):
            if r_val >= R_THRESHOLD:
                return di
        return None

    t_first = first_above_threshold('cum_text')
    for comp_name in ['cum_V', 'hidden_state']:
        c_first = first_above_threshold(comp_name)
        if c_first is not None and t_first is not None:
            gap = t_first - c_first
            print(f"\n  {comp_name} vs cum_text gap: {gap} deciles ({gap*10}% of chain)")
        elif c_first is not None:
            print(f"\n  {comp_name} vs cum_text gap: text NEVER reaches threshold — "
                  f"{comp_name} leads by >={N_DECILES - c_first} deciles")
        else:
            print(f"\n  {comp_name} vs cum_text gap: {comp_name} never reaches threshold")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3.75: Advantage analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"ADVANTAGE ANALYSIS (R differences)")
    print(f"{'='*60}")

    print(f"\n  {'Decile':>6} {'cumV-cumTxt':>12} {'HS-cumTxt':>10} {'cumV-singleV':>13}")
    print(f"  {'-'*45}")

    for di in range(N_DECILES):
        pct = int(decile_fracs[di] * 100)
        cv = probe_results['cum_V'][di]
        ct = probe_results['cum_text'][di]
        hs = probe_results['hidden_state'][di]
        sv = probe_results['single_V'][di]
        print(f"  {pct:>5}% {cv-ct:>+12.3f} {hs-ct:>+10.3f} {cv-sv:>+13.3f}")

    # Summary stats
    cv_arr = np.array(probe_results['cum_V'])
    ct_arr = np.array(probe_results['cum_text'])
    hs_arr = np.array(probe_results['hidden_state'])
    sv_arr = np.array(probe_results['single_V'])

    n_cv_leads = np.sum(cv_arr > ct_arr)
    n_hs_leads = np.sum(hs_arr > ct_arr)
    mean_cv_adv = np.mean(cv_arr - ct_arr)
    mean_hs_adv = np.mean(hs_arr - ct_arr)
    mean_agg_adv = np.mean(cv_arr - sv_arr)

    print(f"\n  cumV > cumText at {n_cv_leads}/{N_DECILES} positions, mean advantage: {mean_cv_adv:+.3f}")
    print(f"  hiddenState > cumText at {n_hs_leads}/{N_DECILES} positions, mean advantage: {mean_hs_adv:+.3f}")
    print(f"  cumV > singleV: mean aggregation improvement: {mean_agg_adv:+.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Figures
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 4: Generate figures")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    positions_pct = [int(f * 100) for f in decile_fracs]

    # Figure 1: All probes — R vs position
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(positions_pct, probe_results['hidden_state'], 'o-', color='#e41a1c',
            linewidth=2.5, markersize=7, label='Hidden state (L35)', zorder=6)
    ax.plot(positions_pct, probe_results['cum_V'], 's-', color='#ff7f00',
            linewidth=2, markersize=6, label='Cumulative V', zorder=5)
    ax.plot(positions_pct, probe_results['cum_K'], 'D-', color='#377eb8',
            linewidth=2, markersize=6, label='Cumulative K', zorder=5)
    ax.plot(positions_pct, probe_results['cum_text'], '^-', color='#4daf4a',
            linewidth=2, markersize=6, label='Cumulative text')
    ax.plot(positions_pct, probe_results['single_V'], 'v--', color='#984ea3',
            linewidth=1.5, markersize=5, label='Single-position V (exp_075)', alpha=0.7)
    ax.plot(positions_pct, probe_results['shuffle'], 'x--', color='gray',
            linewidth=1, markersize=5, label='Shuffle control', alpha=0.5)

    ax.axhline(y=R_THRESHOLD, color='black', linestyle=':', alpha=0.3,
               label=f'R={R_THRESHOLD} threshold')
    ax.set_xlabel('Position in CoT chain (%)', fontsize=12)
    ax.set_ylabel('Probe R (Pearson correlation)', fontsize=12)
    ax.set_title(f'Cumulative KV vs Text — Early Answer Decodability\n'
                 f'Qwen3-4B-Base, L{probe_layer} (n={n_valid})', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(5, 105)
    ax.set_ylim(-0.3, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'cumulative_probes_vs_position.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: cumulative_probes_vs_position.png")

    # Figure 2: Advantage of cumV and hidden_state over cumulative text
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.array(positions_pct)
    ax.bar(x - 2, cv_arr - ct_arr, width=4, color='#ff7f00', alpha=0.8,
           label='Cumulative V - Cumulative text')
    ax.bar(x + 2, hs_arr - ct_arr, width=4, color='#e41a1c', alpha=0.8,
           label='Hidden state - Cumulative text')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Position in CoT chain (%)', fontsize=12)
    ax.set_ylabel('R advantage over cumulative text', fontsize=12)
    ax.set_title(f'KV/Hidden-State Advantage Over Text Baseline\n'
                 f'Qwen3-4B-Base, L{probe_layer} (n={n_valid})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'advantage_over_text.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: advantage_over_text.png")

    # Figure 3: Aggregation improvement (cumulative V vs single V)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(positions_pct, probe_results['cum_V'], 's-', color='#ff7f00',
            linewidth=2, markersize=7, label='Cumulative V (mean of V[0:P])')
    ax.plot(positions_pct, probe_results['single_V'], 'v--', color='#984ea3',
            linewidth=2, markersize=7, label='Single V (V[P] only)')
    ax.plot(positions_pct, probe_results['cum_text'], '^-', color='#4daf4a',
            linewidth=2, markersize=7, label='Cumulative text (reference)')
    ax.fill_between(positions_pct, probe_results['single_V'], probe_results['cum_V'],
                     alpha=0.2, color='#ff7f00', label='Aggregation gain')
    ax.axhline(y=R_THRESHOLD, color='black', linestyle=':', alpha=0.3)
    ax.set_xlabel('Position in CoT chain (%)', fontsize=12)
    ax.set_ylabel('Probe R (Pearson correlation)', fontsize=12)
    ax.set_title(f'Aggregation Improvement: Cumulative vs Single-Position V\n'
                 f'Qwen3-4B-Base, L{probe_layer} (n={n_valid})', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(5, 105)
    ax.set_ylim(-0.3, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'aggregation_improvement.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: aggregation_improvement.png")

    # ═══════════════════════════════════════════════════════════════
    # Save results JSON
    # ═══════════════════════════════════════════════════════════════
    results_json = {
        'model': MODEL_NAME,
        'n_generated': n_total,
        'n_correct': n_correct,
        'n_valid': n_valid,
        'accuracy': n_correct / n_total if n_total > 0 else 0,
        'mean_cot_length': float(cot_lengths_valid.mean()),
        'probe_layer': probe_layer,
        'probe_layer_pct': round(probe_layer / (n_layers - 1) * 100),
        'decile_fracs': decile_fracs.tolist(),
        'kv_dim': kv_dim,
        'hs_dim': hs_dim,
        'text_dim': hidden_size,
        'probes': {
            'cum_K': probe_results['cum_K'],
            'cum_V': probe_results['cum_V'],
            'cum_text': probe_results['cum_text'],
            'hidden_state': probe_results['hidden_state'],
            'single_V': probe_results['single_V'],
            'shuffle': probe_results['shuffle'],
        },
        'advantages': {
            'cumV_over_cumText': (cv_arr - ct_arr).tolist(),
            'hs_over_cumText': (hs_arr - ct_arr).tolist(),
            'cumV_over_singleV': (cv_arr - sv_arr).tolist(),
        },
        'summary': {
            'n_cumV_leads_cumText': int(n_cv_leads),
            'n_hs_leads_cumText': int(n_hs_leads),
            'mean_cumV_advantage': float(mean_cv_adv),
            'mean_hs_advantage': float(mean_hs_adv),
            'mean_aggregation_improvement': float(mean_agg_adv),
        },
        'early_decodability': {},
        'runtime_s': time.time() - t0,
    }

    # Early decodability gaps
    for comp_name in ['cum_V', 'hidden_state']:
        c_first = first_above_threshold(comp_name)
        results_json['early_decodability'][comp_name] = {
            'first_above_threshold': c_first,
            'text_first_above_threshold': t_first,
            'gap_deciles': (t_first - c_first) if (c_first is not None and t_first is not None) else None,
            'threshold': R_THRESHOLD,
        }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: results.json")

    # ═══════════════════════════════════════════════════════════════
    # Print summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Probe layer: L{probe_layer} ({results_json['probe_layer_pct']}% depth)")
    print(f"  cum_V R range: [{min(probe_results['cum_V']):.3f}, {max(probe_results['cum_V']):.3f}]")
    print(f"  cum_K R range: [{min(probe_results['cum_K']):.3f}, {max(probe_results['cum_K']):.3f}]")
    print(f"  cum_text R range: [{min(probe_results['cum_text']):.3f}, {max(probe_results['cum_text']):.3f}]")
    print(f"  hidden_state R range: [{min(probe_results['hidden_state']):.3f}, {max(probe_results['hidden_state']):.3f}]")
    print(f"  single_V R range: [{min(probe_results['single_V']):.3f}, {max(probe_results['single_V']):.3f}]")
    print(f"  shuffle R range: [{min(probe_results['shuffle']):.3f}, {max(probe_results['shuffle']):.3f}]")

    print(f"\n  KEY COMPARISONS:")
    for pct_idx, pct_val in [(2, '30%'), (4, '50%'), (9, '100%')]:
        cv = probe_results['cum_V'][pct_idx]
        ct = probe_results['cum_text'][pct_idx]
        hs = probe_results['hidden_state'][pct_idx]
        print(f"    At {pct_val}: cumV={cv:.3f}, cumText={ct:.3f}, HS={hs:.3f}, "
              f"cumV-cumText={cv-ct:+.3f}, HS-cumText={hs-ct:+.3f}")

    print(f"\n  cumV > cumText at {n_cv_leads}/{N_DECILES} positions")
    print(f"  hiddenState > cumText at {n_hs_leads}/{N_DECILES} positions")
    print(f"  Mean cumV advantage: {mean_cv_adv:+.3f}")
    print(f"  Mean HS advantage: {mean_hs_adv:+.3f}")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
