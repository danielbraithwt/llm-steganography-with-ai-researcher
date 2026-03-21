#!/usr/bin/env python3
"""
Experiment 077: Cross-Model Cumulative KV Probe — Mistral-7B-v0.3

Replicates exp_076 on Mistral-7B-v0.3 with three enhancements:
1. Cross-model replication (Llama vs Qwen) — does cumV > cumText hold universally?
2. Multi-layer sweep (L8, L16, L24, L31) — does the advantage grow with depth?
3. Paired bootstrap significance test for cumV-cumText advantage

If cumV > cumText replicates on Llama AND the advantage grows with layer depth,
that's strong evidence the KV cache accumulates computation beyond what text
tokens encode — not just an artifact of richer representations.

Target: log(|final_answer| + 1) * sign(answer)
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
MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
N_PROBLEMS_MAX = 250
N_DECILES = 10
N_BOOTSTRAP = 2000

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_077"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 8-shot exemplars (same as exp_071-076) ──
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
    return np.sign(x) * np.log(np.abs(x) + 1)


def run_probe_cv(X, y, n_splits=5, return_preds=False):
    """Train ridge probe with cross-validation, return Pearson R (and optionally predictions)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    if X.shape[0] < max(n_splits, 10):
        if return_preds:
            return 0.0, np.zeros_like(y)
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
    if return_preds:
        return float(r), y_pred
    return float(r)


def bootstrap_r_difference(y, preds_a, preds_b, n_boot=2000, seed=42):
    """Bootstrap test for R(a) - R(b). Returns observed diff, CI, p-value."""
    from scipy import stats
    obs_r_a = stats.pearsonr(y, preds_a)[0]
    obs_r_b = stats.pearsonr(y, preds_b)[0]
    obs_diff = obs_r_a - obs_r_b

    diffs = np.zeros(n_boot)
    n = len(y)
    rng = np.random.RandomState(seed)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        # Avoid degenerate samples
        if len(np.unique(y[idx])) < 3:
            diffs[b] = 0.0
            continue
        r_a = np.corrcoef(y[idx], preds_a[idx])[0, 1]
        r_b = np.corrcoef(y[idx], preds_b[idx])[0, 1]
        if np.isnan(r_a):
            r_a = 0.0
        if np.isnan(r_b):
            r_b = 0.0
        diffs[b] = r_a - r_b

    ci_lo = np.percentile(diffs, 2.5)
    ci_hi = np.percentile(diffs, 97.5)
    p_val = np.mean(diffs <= 0)  # fraction where V does NOT lead
    return float(obs_diff), float(ci_lo), float(ci_hi), float(p_val)


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
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim_actual = hidden_size // model.config.num_attention_heads
    print(f"Model: {n_layers} layers, hidden_size={hidden_size}")
    print(f"KV heads: {num_kv_heads}, head_dim: {head_dim_actual}")

    # Probe at 4 layer depths: 25%, 50%, 75%, ~97%
    probe_layers = [
        n_layers // 4,           # 25% depth
        n_layers // 2,           # 50% depth
        3 * n_layers // 4,       # 75% depth
        n_layers - 1,            # ~97% depth
    ]
    probe_layer_pcts = [round(l / (n_layers - 1) * 100) for l in probe_layers]
    print(f"Probe layers: {['L'+str(l)+' ('+str(p)+'%)' for l, p in zip(probe_layers, probe_layer_pcts)]}")

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
    print(f"PHASE 1: Generate CoT ({MODEL_NAME})")
    print(f"{'='*60}")

    gen_results = []
    for i, row in enumerate(ds):
        if i >= N_PROBLEMS_MAX:
            break
        if time.time() - t0 > TIME_BUDGET * 0.40:
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

        if (i + 1) % 25 == 0:
            n_corr = sum(1 for r in gen_results if r['correct'])
            elapsed = time.time() - t0
            print(f"  [{i+1}/{N_PROBLEMS_MAX}] Correct: {n_corr}/{i+1} "
                  f"({n_corr/(i+1)*100:.1f}%) [{elapsed:.0f}s]")

    n_total = len(gen_results)
    correct_results = [r for r in gen_results if r['correct']]
    n_correct = len(correct_results)
    gen_acc = n_correct / n_total * 100 if n_total > 0 else 0
    print(f"\nGeneration complete: {n_correct}/{n_total} correct "
          f"({gen_acc:.1f}%) [{time.time()-t0:.0f}s]")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Forward pass to extract CUMULATIVE KV features at 4 layers
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 2: Extract cumulative KV features at {len(probe_layers)} layers x {N_DECILES} deciles")
    print(f"{'='*60}")

    decile_fracs = np.linspace(0.1, 1.0, N_DECILES)

    # Determine KV dim from a test forward pass
    test_prompt = build_prompt(correct_results[0]['question'])
    test_text = test_prompt + correct_results[0]['gen_text']
    test_inputs = tokenizer(test_text, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN)
    with torch.no_grad():
        test_out = model(test_inputs['input_ids'].to(model.device), use_cache=True)
    k_test, v_test = get_kv(test_out.past_key_values, probe_layers[0])
    kv_dim = k_test.shape[1] * k_test.shape[3]  # n_kv_heads * head_dim
    print(f"KV dim: {kv_dim} ({k_test.shape[1]} heads x {k_test.shape[3]} head_dim)")
    print(f"Text embedding dim: {hidden_size}")
    del test_out, k_test, v_test
    torch.cuda.empty_cache()

    n_probe_layers = len(probe_layers)
    # Storage: [n_problems, n_layers, n_deciles, dim]
    cum_K_feat = np.zeros((n_correct, n_probe_layers, N_DECILES, kv_dim), dtype=np.float32)
    cum_V_feat = np.zeros((n_correct, n_probe_layers, N_DECILES, kv_dim), dtype=np.float32)
    cum_text_feat = np.zeros((n_correct, N_DECILES, hidden_size), dtype=np.float32)
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
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values

            # Token embeddings for CoT positions
            cot_ids = input_ids[0, cot_start:cot_end]
            token_embeds = embed_fn(cot_ids).float().cpu().numpy()  # [cot_len, hidden_size]

        # Compute text cumulative sums
        text_cumsum = np.cumsum(token_embeds, axis=0)

        # Extract KV at each probe layer
        for li, layer_idx in enumerate(probe_layers):
            k_cache, v_cache = get_kv(past_kv, layer_idx)
            # Shape: [1, n_kv_heads, seq_len, head_dim]
            k_cot = k_cache[0, :, cot_start:cot_end, :].permute(1, 0, 2).reshape(cot_len, -1).float().cpu().numpy()
            v_cot = v_cache[0, :, cot_start:cot_end, :].permute(1, 0, 2).reshape(cot_len, -1).float().cpu().numpy()

            k_cumsum = np.cumsum(k_cot, axis=0)
            v_cumsum = np.cumsum(v_cot, axis=0)

            for di, frac in enumerate(decile_fracs):
                pos = min(int(frac * cot_len) - 1, cot_len - 1)
                pos = max(0, pos)
                cum_K_feat[pi, li, di] = k_cumsum[pos] / (pos + 1)
                cum_V_feat[pi, li, di] = v_cumsum[pos] / (pos + 1)

        # Text features (same across layers — just input embeddings)
        for di, frac in enumerate(decile_fracs):
            pos = min(int(frac * cot_len) - 1, cot_len - 1)
            pos = max(0, pos)
            cum_text_feat[pi, di] = text_cumsum[pos] / (pos + 1)

        # Free GPU memory
        del outputs, past_kv
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
    if len(answers_valid) > 0:
        print(f"Answer range: [{np.exp(np.abs(answers_valid).min())-1:.0f}, "
              f"{np.exp(np.abs(answers_valid).max())-1:.0f}]")

    # Free model memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Train probes at each layer x decile
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 3: Train probes at {n_probe_layers} layers x {N_DECILES} deciles")
    print(f"{'='*60}")

    # Store results per layer
    all_probe_results = {}
    all_cv_preds = {}  # for bootstrap test
    all_ct_preds = {}

    for li, (layer_idx, layer_pct) in enumerate(zip(probe_layers, probe_layer_pcts)):
        layer_name = f"L{layer_idx}"
        print(f"\n  --- Layer {layer_name} ({layer_pct}% depth) ---")
        print(f"  {'Decile':>6} {'cumK':>7} {'cumV':>7} {'cumTxt':>7} {'Shuffle':>8}")
        print(f"  {'-'*38}")

        probe_res = {'cum_K': [], 'cum_V': [], 'cum_text': [], 'shuffle': []}
        cv_preds_layer = {}
        ct_preds_layer = {}

        for di in range(N_DECILES):
            X_ck = cum_K_feat[valid_idx, li, di]
            X_cv = cum_V_feat[valid_idx, li, di]
            X_ct = cum_text_feat[valid_idx, di]
            y = answers_valid

            r_ck = run_probe_cv(X_ck, y)
            r_cv, preds_cv = run_probe_cv(X_cv, y, return_preds=True)
            r_ct, preds_ct = run_probe_cv(X_ct, y, return_preds=True)

            cv_preds_layer[di] = preds_cv
            ct_preds_layer[di] = preds_ct

            # Shuffle control (on cumV)
            rng = np.random.RandomState(SEED + li * 100 + di)
            y_shuf = rng.permutation(y)
            r_shuf = run_probe_cv(X_cv, y_shuf)

            probe_res['cum_K'].append(r_ck)
            probe_res['cum_V'].append(r_cv)
            probe_res['cum_text'].append(r_ct)
            probe_res['shuffle'].append(r_shuf)

            pct = int(decile_fracs[di] * 100)
            print(f"  {pct:>5}% {r_ck:>7.3f} {r_cv:>7.3f} {r_ct:>7.3f} {r_shuf:>8.3f}")

        all_probe_results[layer_name] = probe_res
        all_cv_preds[layer_name] = cv_preds_layer
        all_ct_preds[layer_name] = ct_preds_layer

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3.5: Advantage analysis + Bootstrap significance test
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 3.5: Advantage analysis + Bootstrap significance test")
    print(f"{'='*60}")

    bootstrap_results = {}

    for li, (layer_idx, layer_pct) in enumerate(zip(probe_layers, probe_layer_pcts)):
        layer_name = f"L{layer_idx}"
        pr = all_probe_results[layer_name]
        cv_arr = np.array(pr['cum_V'])
        ct_arr = np.array(pr['cum_text'])
        ck_arr = np.array(pr['cum_K'])

        n_cv_leads = int(np.sum(cv_arr > ct_arr))
        mean_cv_adv = float(np.mean(cv_arr - ct_arr))
        mean_ck_adv = float(np.mean(ck_arr - ct_arr))

        print(f"\n  --- {layer_name} ({layer_pct}% depth) ---")
        print(f"  {'Decile':>6} {'cumV-cumTxt':>12} {'cumK-cumTxt':>12}")
        print(f"  {'-'*32}")
        for di in range(N_DECILES):
            pct = int(decile_fracs[di] * 100)
            print(f"  {pct:>5}% {cv_arr[di]-ct_arr[di]:>+12.3f} {ck_arr[di]-ct_arr[di]:>+12.3f}")
        print(f"  Mean cumV advantage: {mean_cv_adv:+.4f} ({n_cv_leads}/{N_DECILES} positions)")
        print(f"  Mean cumK advantage: {mean_ck_adv:+.4f}")

        # Bootstrap significance test for mean advantage across all deciles
        boot_diffs = np.zeros(N_BOOTSTRAP)
        rng = np.random.RandomState(SEED + 5000 + li)
        for b in range(N_BOOTSTRAP):
            idx = rng.choice(n_valid, size=n_valid, replace=True)
            adv_sum = 0.0
            for di in range(N_DECILES):
                y_boot = answers_valid[idx]
                if len(np.unique(y_boot)) < 3:
                    continue
                r_cv_b = np.corrcoef(y_boot, all_cv_preds[layer_name][di][idx])[0, 1]
                r_ct_b = np.corrcoef(y_boot, all_ct_preds[layer_name][di][idx])[0, 1]
                if np.isnan(r_cv_b):
                    r_cv_b = 0.0
                if np.isnan(r_ct_b):
                    r_ct_b = 0.0
                adv_sum += (r_cv_b - r_ct_b)
            boot_diffs[b] = adv_sum / N_DECILES

        ci_lo = float(np.percentile(boot_diffs, 2.5))
        ci_hi = float(np.percentile(boot_diffs, 97.5))
        p_val = float(np.mean(boot_diffs <= 0))

        bootstrap_results[layer_name] = {
            'observed_mean_advantage': mean_cv_adv,
            'ci_95_lower': ci_lo,
            'ci_95_upper': ci_hi,
            'p_value': p_val,
            'n_bootstrap': N_BOOTSTRAP,
            'n_positions_V_leads': n_cv_leads,
        }

        print(f"\n  BOOTSTRAP TEST (n={N_BOOTSTRAP}):")
        print(f"    Observed mean cumV-cumText advantage: {mean_cv_adv:+.4f}")
        print(f"    95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    p-value (cumV <= cumText): {p_val:.4f}")
        sig = "YES" if p_val < 0.05 else "NO"
        print(f"    Significant at alpha=0.05: {sig}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3.75: Layer sweep summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"LAYER SWEEP SUMMARY — Does advantage grow with depth?")
    print(f"{'='*60}")

    layer_advantages = []
    print(f"\n  {'Layer':>8} {'Depth%':>7} {'MeanAdvR':>9} {'Positions':>10} {'CI_lo':>8} {'CI_hi':>8} {'p-val':>8}")
    print(f"  {'-'*62}")
    for li, (layer_idx, layer_pct) in enumerate(zip(probe_layers, probe_layer_pcts)):
        layer_name = f"L{layer_idx}"
        br = bootstrap_results[layer_name]
        adv = br['observed_mean_advantage']
        layer_advantages.append(adv)
        print(f"  {layer_name:>8} {layer_pct:>6}% {adv:>+9.4f} {br['n_positions_V_leads']:>5}/{N_DECILES}"
              f" {br['ci_95_lower']:>+8.4f} {br['ci_95_upper']:>+8.4f} {br['p_value']:>8.4f}")

    # Test if advantage grows with depth (Spearman correlation)
    from scipy import stats
    if len(layer_advantages) >= 4:
        rho, p_growth = stats.spearmanr(probe_layer_pcts, layer_advantages)
        print(f"\n  Layer depth vs advantage: Spearman rho={rho:.3f}, p={p_growth:.4f}")
        if rho > 0.5 and p_growth < 0.1:
            print(f"  FINDING: Advantage GROWS with depth — consistent with computation accumulation")
        elif rho < -0.5 and p_growth < 0.1:
            print(f"  FINDING: Advantage SHRINKS with depth — inconsistent with computation hypothesis")
        else:
            print(f"  FINDING: No clear depth trend — advantage is layer-independent")

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

    # Figure 1: All layers — cumV, cumK, cumText R vs position
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for li, (layer_idx, layer_pct) in enumerate(zip(probe_layers, probe_layer_pcts)):
        ax = axes[li // 2, li % 2]
        layer_name = f"L{layer_idx}"
        pr = all_probe_results[layer_name]

        ax.plot(positions_pct, pr['cum_V'], 's-', color='#ff7f00',
                linewidth=2, markersize=6, label='Cumulative V')
        ax.plot(positions_pct, pr['cum_K'], 'D-', color='#377eb8',
                linewidth=2, markersize=6, label='Cumulative K')
        ax.plot(positions_pct, pr['cum_text'], '^-', color='#4daf4a',
                linewidth=2, markersize=6, label='Cumulative text')
        ax.plot(positions_pct, pr['shuffle'], 'x--', color='gray',
                linewidth=1, markersize=5, label='Shuffle', alpha=0.5)

        ax.axhline(y=0.3, color='black', linestyle=':', alpha=0.3)
        ax.set_xlabel('Position in CoT (%)')
        ax.set_ylabel('Probe R')
        br = bootstrap_results[layer_name]
        ax.set_title(f'{layer_name} ({layer_pct}% depth) — adv={br["observed_mean_advantage"]:+.3f}, '
                     f'p={br["p_value"]:.3f}')
        ax.legend(fontsize=8, loc='best')
        ax.set_xlim(5, 105)
        ax.set_ylim(-0.3, 1.0)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Cross-Model Replication: Cumulative KV vs Text\n'
                 f'Llama-3.1-8B-Instruct (n={n_valid})', fontsize=14)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'layer_sweep_probes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: layer_sweep_probes.png")

    # Figure 2: Layer sweep — mean advantage vs depth
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#377eb8', '#ff7f00', '#e41a1c', '#984ea3']
    for li, (layer_idx, layer_pct) in enumerate(zip(probe_layers, probe_layer_pcts)):
        layer_name = f"L{layer_idx}"
        br = bootstrap_results[layer_name]
        ax.bar(layer_pct, br['observed_mean_advantage'], width=8, color=colors[li],
               alpha=0.8, label=f'{layer_name}')
        ax.errorbar(layer_pct, br['observed_mean_advantage'],
                    yerr=[[br['observed_mean_advantage'] - br['ci_95_lower']],
                          [br['ci_95_upper'] - br['observed_mean_advantage']]],
                    fmt='none', color='black', capsize=5, linewidth=2)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Layer Depth (%)', fontsize=12)
    ax.set_ylabel('Mean cumV - cumText R', fontsize=12)
    ax.set_title(f'KV Advantage Over Text vs Layer Depth\n'
                 f'Llama-3.1-8B-Instruct (n={n_valid}, {N_BOOTSTRAP} bootstrap)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'layer_depth_advantage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: layer_depth_advantage.png")

    # Figure 3: Cross-model comparison (deepest layer) with exp_076 Qwen data
    fig, ax = plt.subplots(figsize=(10, 6))
    # Llama (this experiment) — deepest layer
    deepest = f"L{probe_layers[-1]}"
    pr_llama = all_probe_results[deepest]
    ax.plot(positions_pct, pr_llama['cum_V'], 's-', color='#ff7f00',
            linewidth=2, markersize=7, label=f'Llama cumV ({deepest})')
    ax.plot(positions_pct, pr_llama['cum_text'], '^-', color='#4daf4a',
            linewidth=2, markersize=7, label='Llama cumText')

    # Load exp_076 Qwen data for comparison if available
    qwen_results_path = RESULTS_DIR.parent / 'exp_076' / 'results.json'
    if qwen_results_path.exists():
        with open(qwen_results_path) as f:
            qwen_data = json.load(f)
        ax.plot(positions_pct, qwen_data['probes']['cum_V'], 's--', color='#ff7f00',
                linewidth=1.5, markersize=5, alpha=0.6, label='Qwen cumV (exp_076)')
        ax.plot(positions_pct, qwen_data['probes']['cum_text'], '^--', color='#4daf4a',
                linewidth=1.5, markersize=5, alpha=0.6, label='Qwen cumText (exp_076)')

    ax.axhline(y=0.3, color='black', linestyle=':', alpha=0.3)
    ax.set_xlabel('Position in CoT chain (%)', fontsize=12)
    ax.set_ylabel('Probe R (Pearson correlation)', fontsize=12)
    ax.set_title(f'Cross-Model Comparison: Llama vs Qwen\n'
                 f'Cumulative V vs Cumulative Text at deepest layer', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(5, 105)
    ax.set_ylim(-0.3, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'cross_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: cross_model_comparison.png")

    # ═══════════════════════════════════════════════════════════════
    # Save results JSON
    # ═══════════════════════════════════════════════════════════════
    results_json = {
        'model': MODEL_NAME,
        'n_generated': n_total,
        'n_correct': n_correct,
        'n_valid': n_valid,
        'accuracy': gen_acc,
        'mean_cot_length': float(cot_lengths_valid.mean()) if len(cot_lengths_valid) > 0 else 0,
        'probe_layers': {f"L{l}": p for l, p in zip(probe_layers, probe_layer_pcts)},
        'kv_dim': kv_dim,
        'text_dim': hidden_size,
        'n_deciles': N_DECILES,
        'n_bootstrap': N_BOOTSTRAP,
        'probes_by_layer': {},
        'bootstrap_by_layer': bootstrap_results,
        'layer_sweep': {
            'depths_pct': probe_layer_pcts,
            'mean_advantages': layer_advantages,
        },
        'runtime_s': time.time() - t0,
    }

    for li, (layer_idx, layer_pct) in enumerate(zip(probe_layers, probe_layer_pcts)):
        layer_name = f"L{layer_idx}"
        pr = all_probe_results[layer_name]
        cv_arr = np.array(pr['cum_V'])
        ct_arr = np.array(pr['cum_text'])
        results_json['probes_by_layer'][layer_name] = {
            'cum_K': pr['cum_K'],
            'cum_V': pr['cum_V'],
            'cum_text': pr['cum_text'],
            'shuffle': pr['shuffle'],
            'advantages_cumV_over_cumText': (cv_arr - ct_arr).tolist(),
        }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: results.json")

    # ═══════════════════════════════════════════════════════════════
    # Print final summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")

    print(f"\nModel: {MODEL_NAME}")
    print(f"Accuracy: {n_correct}/{n_total} ({gen_acc:.1f}%)")
    print(f"Valid problems: {n_valid}")

    any_sig = False
    for li, (layer_idx, layer_pct) in enumerate(zip(probe_layers, probe_layer_pcts)):
        layer_name = f"L{layer_idx}"
        br = bootstrap_results[layer_name]
        pr = all_probe_results[layer_name]
        n_leads = br['n_positions_V_leads']
        sig_str = "***" if br['p_value'] < 0.001 else ("**" if br['p_value'] < 0.01 else ("*" if br['p_value'] < 0.05 else ""))
        if br['p_value'] < 0.05:
            any_sig = True
        print(f"\n  {layer_name} ({layer_pct}%): cumV adv = {br['observed_mean_advantage']:+.4f} "
              f"({n_leads}/{N_DECILES} pos) "
              f"[{br['ci_95_lower']:+.4f}, {br['ci_95_upper']:+.4f}] "
              f"p={br['p_value']:.4f} {sig_str}")
        print(f"    cumV range: [{min(pr['cum_V']):.3f}, {max(pr['cum_V']):.3f}]")
        print(f"    cumText range: [{min(pr['cum_text']):.3f}, {max(pr['cum_text']):.3f}]")

    # Cross-model comparison
    print(f"\n  CROSS-MODEL COMPARISON (deepest layer):")
    deepest = f"L{probe_layers[-1]}"
    br_llama = bootstrap_results[deepest]
    print(f"    Llama mean advantage: {br_llama['observed_mean_advantage']:+.4f} "
          f"(p={br_llama['p_value']:.4f})")
    if qwen_results_path.exists():
        with open(qwen_results_path) as f:
            qwen_data = json.load(f)
        qwen_adv = qwen_data['summary']['mean_cumV_advantage']
        print(f"    Qwen  mean advantage: {qwen_adv:+.4f} (exp_076)")
        if br_llama['observed_mean_advantage'] > 0 and qwen_adv > 0:
            print(f"    REPLICATION: Both models show cumV > cumText!")
        else:
            print(f"    DIVERGENCE: Models disagree on cumV vs cumText direction")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
