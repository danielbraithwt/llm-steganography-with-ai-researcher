#!/usr/bin/env python3
"""
Experiment 075: Early Answer Decodability — Linear Probe Along CoT Chain

Core question: Does the KV cache "know" the final answer before the text reveals it?

Method:
1. Generate 8-shot CoT on GSM8K (~250 problems) with Qwen3-4B-Base
2. For correctly solved problems (~200), forward pass to extract:
   - K, V vectors at layers [L18, L35] for each CoT token position
   - Token embeddings for each CoT token position
3. At 10 position fractions (10%, 20%, ..., 100% of CoT length):
   - Train ridge regression probes to predict log(|final_answer| + 1)
   - K-probe, V-probe, text-cumulative-mean probe, shuffle control
4. Plot R vs position for each probe type

The "early decodability gap" = how many position deciles earlier KV-probe R
exceeds 0.3 compared to text-probe R exceeding 0.3.
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
N_PROBLEMS_MAX = 300
N_DECILES = 10

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_075"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 8-shot exemplars (same as exp_071-074) ──
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
    """Train ridge probe with cross-validation, return R."""
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
    # Probe at 50% depth and 100% depth
    probe_layers = [n_layers // 2, n_layers - 1]
    print(f"Model: {n_layers} layers, hidden_size={hidden_size}")
    print(f"Probe layers: {probe_layers} ({[round(l/(n_layers-1)*100) for l in probe_layers]}% depth)")

    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = hidden_size // model.config.num_attention_heads
    print(f"KV heads: {num_kv_heads}, head_dim: {head_dim}")

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
    # PHASE 2: Forward pass to extract KV + text features at deciles
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 2: Extract KV features at {N_DECILES} decile positions")
    print(f"{'='*60}")

    decile_fracs = np.linspace(0.1, 1.0, N_DECILES)  # 10%, 20%, ..., 100%

    # First, determine KV dim from a test forward pass
    test_prompt = build_prompt(correct_results[0]['question'])
    test_text = test_prompt + correct_results[0]['gen_text']
    test_inputs = tokenizer(test_text, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN)
    with torch.no_grad():
        test_out = model(test_inputs['input_ids'].to(model.device), use_cache=True)
    k_test, v_test = get_kv(test_out.past_key_values, probe_layers[0])
    kv_dim = k_test.shape[1] * k_test.shape[3]  # n_kv_heads * head_dim
    print(f"KV dim: {kv_dim} ({k_test.shape[1]} heads x {k_test.shape[3]} head_dim)")
    print(f"Text dim: {hidden_size}")
    del test_out, k_test, v_test
    torch.cuda.empty_cache()

    # Allocate storage
    K_feat = {l: np.zeros((n_correct, N_DECILES, kv_dim), dtype=np.float32) for l in probe_layers}
    V_feat = {l: np.zeros((n_correct, N_DECILES, kv_dim), dtype=np.float32) for l in probe_layers}
    text_cum_feat = np.zeros((n_correct, N_DECILES, hidden_size), dtype=np.float32)
    text_tok_feat = np.zeros((n_correct, N_DECILES, hidden_size), dtype=np.float32)
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

        # CoT token range
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
            outputs = model(input_ids, use_cache=True, output_hidden_states=False)
            past_kv = outputs.past_key_values

            # Token embeddings for CoT positions
            cot_ids = input_ids[0, cot_start:cot_end]
            with torch.no_grad():
                token_embeds = embed_fn(cot_ids).float().cpu().numpy()  # [cot_len, hidden_size]

        # Extract KV at probe layers and compute decile features
        for layer in probe_layers:
            k_cache, v_cache = get_kv(past_kv, layer)
            # Shape: [1, n_kv_heads, seq_len, head_dim]
            # Extract CoT positions and flatten heads
            k_cot = k_cache[0, :, cot_start:cot_end, :].permute(1, 0, 2).reshape(cot_len, -1).float().cpu().numpy()
            v_cot = v_cache[0, :, cot_start:cot_end, :].permute(1, 0, 2).reshape(cot_len, -1).float().cpu().numpy()

            for di, frac in enumerate(decile_fracs):
                pos = min(int(frac * cot_len) - 1, cot_len - 1)
                pos = max(0, pos)
                K_feat[layer][pi, di] = k_cot[pos]
                V_feat[layer][pi, di] = v_cot[pos]

        # Text features: cumulative mean and current-token
        cum_sum = np.cumsum(token_embeds, axis=0)
        for di, frac in enumerate(decile_fracs):
            pos = min(int(frac * cot_len) - 1, cot_len - 1)
            pos = max(0, pos)
            text_cum_feat[pi, di] = cum_sum[pos] / (pos + 1)
            text_tok_feat[pi, di] = token_embeds[pos]

        # Free GPU memory
        del outputs, past_kv, k_cache, v_cache
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
    print(f"Answer range: [{np.exp(np.abs(answers_valid).min())-1:.0f}, {np.exp(np.abs(answers_valid).max())-1:.0f}]")

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

    probe_results = {}
    for layer in probe_layers:
        layer_pct = round(layer / (n_layers - 1) * 100)
        probe_results[layer] = {
            'K': [], 'V': [], 'text_cum': [], 'text_tok': [], 'shuffle_V': [],
            'layer_pct': layer_pct,
        }

        print(f"\nLayer {layer} ({layer_pct}% depth):")
        print(f"  {'Decile':>6} {'K-probe':>8} {'V-probe':>8} {'Text-cum':>9} {'Text-tok':>9} {'Shuffle':>8}")
        print(f"  {'-'*50}")

        for di in range(N_DECILES):
            X_k = K_feat[layer][valid_idx, di]
            X_v = V_feat[layer][valid_idx, di]
            X_tc = text_cum_feat[valid_idx, di]
            X_tt = text_tok_feat[valid_idx, di]
            y = answers_valid

            r_k = run_probe_cv(X_k, y)
            r_v = run_probe_cv(X_v, y)
            r_tc = run_probe_cv(X_tc, y)
            r_tt = run_probe_cv(X_tt, y)

            # Shuffle control: permute answers
            rng = np.random.RandomState(SEED + di)
            y_shuf = rng.permutation(y)
            r_shuf = run_probe_cv(X_v, y_shuf)

            probe_results[layer]['K'].append(r_k)
            probe_results[layer]['V'].append(r_v)
            probe_results[layer]['text_cum'].append(r_tc)
            probe_results[layer]['text_tok'].append(r_tt)
            probe_results[layer]['shuffle_V'].append(r_shuf)

            pct = int(decile_fracs[di] * 100)
            print(f"  {pct:>5}% {r_k:>8.3f} {r_v:>8.3f} {r_tc:>9.3f} {r_tt:>9.3f} {r_shuf:>8.3f}")

    # Compute early decodability gaps
    R_THRESHOLD = 0.3
    print(f"\n{'='*60}")
    print(f"EARLY DECODABILITY ANALYSIS (threshold R>{R_THRESHOLD})")
    print(f"{'='*60}")

    for layer in probe_layers:
        layer_pct = probe_results[layer]['layer_pct']
        print(f"\nLayer {layer} ({layer_pct}%):")

        for probe_name in ['K', 'V', 'text_cum']:
            rs = probe_results[layer][probe_name]
            first_above = None
            for di, r in enumerate(rs):
                if r >= R_THRESHOLD:
                    first_above = di
                    break
            if first_above is not None:
                print(f"  {probe_name:>8}: first R>{R_THRESHOLD} at decile {first_above+1} "
                      f"({int(decile_fracs[first_above]*100)}%)")
            else:
                print(f"  {probe_name:>8}: never reaches R>{R_THRESHOLD} (max R={max(rs):.3f})")

        # Gap calculation
        v_first = None
        t_first = None
        for di, r in enumerate(probe_results[layer]['V']):
            if r >= R_THRESHOLD:
                v_first = di
                break
        for di, r in enumerate(probe_results[layer]['text_cum']):
            if r >= R_THRESHOLD:
                t_first = di
                break

        if v_first is not None and t_first is not None:
            gap = t_first - v_first
            print(f"  V-text gap: {gap} deciles ({gap*10}% of chain)")
        elif v_first is not None:
            print(f"  V-text gap: text NEVER reaches threshold — V leads by >={N_DECILES - v_first} deciles")
        else:
            print(f"  V-text gap: V never reaches threshold — no early decodability")

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

    # Figure 1: R vs position for each layer
    fig, axes = plt.subplots(1, len(probe_layers), figsize=(7 * len(probe_layers), 5), sharey=True)
    if len(probe_layers) == 1:
        axes = [axes]

    for ax_i, layer in enumerate(probe_layers):
        ax = axes[ax_i]
        layer_pct = probe_results[layer]['layer_pct']

        ax.plot(positions_pct, probe_results[layer]['V'], 'o-', color='#e41a1c',
                linewidth=2, markersize=6, label='V-probe', zorder=5)
        ax.plot(positions_pct, probe_results[layer]['K'], 's-', color='#377eb8',
                linewidth=2, markersize=6, label='K-probe', zorder=5)
        ax.plot(positions_pct, probe_results[layer]['text_cum'], '^-', color='#4daf4a',
                linewidth=2, markersize=6, label='Text (cumulative mean)')
        ax.plot(positions_pct, probe_results[layer]['text_tok'], 'D-', color='#984ea3',
                linewidth=1.5, markersize=5, label='Text (current token)', alpha=0.7)
        ax.plot(positions_pct, probe_results[layer]['shuffle_V'], 'x--', color='gray',
                linewidth=1, markersize=5, label='Shuffle control', alpha=0.5)

        ax.axhline(y=R_THRESHOLD, color='black', linestyle=':', alpha=0.3,
                   label=f'R={R_THRESHOLD} threshold')
        ax.set_xlabel('Position in CoT chain (%)', fontsize=12)
        if ax_i == 0:
            ax.set_ylabel('Probe R (Pearson correlation)', fontsize=12)
        ax.set_title(f'Layer {layer} ({layer_pct}% depth)', fontsize=13)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(5, 105)
        ax.set_ylim(-0.2, 1.0)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Early Answer Decodability — Qwen3-4B-Base (n={n_valid})', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'early_decodability_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: early_decodability_by_layer.png")

    # Figure 2: KV advantage over text at best layer
    best_layer = probe_layers[-1]  # last layer
    v_rs = np.array(probe_results[best_layer]['V'])
    k_rs = np.array(probe_results[best_layer]['K'])
    tc_rs = np.array(probe_results[best_layer]['text_cum'])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.array(positions_pct) - 1.5, v_rs - tc_rs, width=3, color='#e41a1c',
           alpha=0.7, label='V-probe advantage over text')
    ax.bar(np.array(positions_pct) + 1.5, k_rs - tc_rs, width=3, color='#377eb8',
           alpha=0.7, label='K-probe advantage over text')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Position in CoT chain (%)', fontsize=12)
    ax.set_ylabel('R(KV) - R(text cumulative)', fontsize=12)
    ax.set_title(f'KV Advantage Over Text — Layer {best_layer} '
                 f'({probe_results[best_layer]["layer_pct"]}%)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(RESULTS_DIR / 'kv_advantage_over_text.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: kv_advantage_over_text.png")

    # Figure 3: Heatmap of R across layers and positions
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax_i, (name, color_label) in enumerate([('V', 'V-probe R'), ('K', 'K-probe R'),
                                                  ('text_cum', 'Text-cumulative R')]):
        data = np.array([probe_results[l][name] for l in probe_layers])
        ax = axes[ax_i]
        im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r', vmin=-0.1, vmax=0.8)
        ax.set_xticks(range(N_DECILES))
        ax.set_xticklabels([f'{p}%' for p in positions_pct], fontsize=8)
        ax.set_yticks(range(len(probe_layers)))
        ax.set_yticklabels([f'L{l} ({probe_results[l]["layer_pct"]}%)' for l in probe_layers])
        ax.set_xlabel('Position in CoT')
        ax.set_title(color_label)
        plt.colorbar(im, ax=ax, shrink=0.8)
        # Annotate cells with R values
        for yi in range(data.shape[0]):
            for xi in range(data.shape[1]):
                ax.text(xi, yi, f'{data[yi, xi]:.2f}', ha='center', va='center',
                       fontsize=7, color='white' if data[yi, xi] > 0.4 else 'black')

    fig.suptitle(f'Probe R Across Layers and Positions — Qwen3-4B-Base (n={n_valid})', fontsize=13)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'probe_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: probe_heatmap.png")

    # ═══════════════════════════════════════════════════════════════
    # Save results JSON
    # ═══════════════════════════════════════════════════════════════
    # Convert results to serializable format
    results_json = {
        'model': MODEL_NAME,
        'n_generated': n_total,
        'n_correct': n_correct,
        'n_valid': n_valid,
        'accuracy': n_correct / n_total if n_total > 0 else 0,
        'mean_cot_length': float(cot_lengths_valid.mean()),
        'probe_layers': probe_layers,
        'decile_fracs': decile_fracs.tolist(),
        'kv_dim': kv_dim,
        'text_dim': hidden_size,
        'probes': {},
        'early_decodability': {},
        'runtime_s': time.time() - t0,
    }

    for layer in probe_layers:
        layer_key = f'L{layer}'
        results_json['probes'][layer_key] = {
            'K': probe_results[layer]['K'],
            'V': probe_results[layer]['V'],
            'text_cum': probe_results[layer]['text_cum'],
            'text_tok': probe_results[layer]['text_tok'],
            'shuffle_V': probe_results[layer]['shuffle_V'],
        }

        # Early decodability for this layer
        v_first = t_first = None
        for di, r in enumerate(probe_results[layer]['V']):
            if r >= R_THRESHOLD:
                v_first = di
                break
        for di, r in enumerate(probe_results[layer]['text_cum']):
            if r >= R_THRESHOLD:
                t_first = di
                break

        gap = None
        if v_first is not None and t_first is not None:
            gap = t_first - v_first
        elif v_first is not None:
            gap = N_DECILES - v_first  # text never reaches threshold

        results_json['early_decodability'][layer_key] = {
            'V_first_above_threshold': v_first,
            'text_first_above_threshold': t_first,
            'gap_deciles': gap,
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
    best = probe_layers[-1]
    best_pct = probe_results[best]['layer_pct']
    print(f"Best layer: L{best} ({best_pct}% depth)")
    print(f"  V-probe R range: [{min(probe_results[best]['V']):.3f}, {max(probe_results[best]['V']):.3f}]")
    print(f"  K-probe R range: [{min(probe_results[best]['K']):.3f}, {max(probe_results[best]['K']):.3f}]")
    print(f"  Text-cum R range: [{min(probe_results[best]['text_cum']):.3f}, {max(probe_results[best]['text_cum']):.3f}]")
    print(f"  Shuffle R range: [{min(probe_results[best]['shuffle_V']):.3f}, {max(probe_results[best]['shuffle_V']):.3f}]")

    v_100 = probe_results[best]['V'][-1]
    t_100 = probe_results[best]['text_cum'][-1]
    print(f"\n  At 100%: V={v_100:.3f}, text={t_100:.3f}, gap={v_100-t_100:+.3f}")
    v_30 = probe_results[best]['V'][2]  # 30%
    t_30 = probe_results[best]['text_cum'][2]
    print(f"  At  30%: V={v_30:.3f}, text={t_30:.3f}, gap={v_30-t_30:+.3f}")

    ld = results_json['early_decodability'][f'L{best}']
    if ld['gap_deciles'] is not None:
        print(f"\n  Early decodability gap: {ld['gap_deciles']} deciles "
              f"({ld['gap_deciles']*10}% of chain)")
    else:
        print(f"\n  Early decodability gap: could not compute")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
