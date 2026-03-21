#!/usr/bin/env python3
"""
Experiment 069: Early Answer Decodability — Mistral-7B-v0.3 Cross-Model Replication

Cross-model replication of exp_068 (Qwen3-4B-Base) on Mistral-7B-v0.3.
Tests whether KV cache probes exceed text baseline during normal generation
on an architecturally different model (analog encoding, different family).

Key differences from Qwen:
- Mistral uses analog K-encoding (vs Qwen's digital)
- Mistral is a base model (same as Qwen)
- Mistral is 7B params (vs 4B) — use bfloat16 to fit in 32GB
- Mistral has 32 layers, 8 KV heads, 128 head_dim (GQA)
- We use the same 8-shot CoT format for fair comparison
- Llama was first choice but is gated/inaccessible without auth token
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

TIME_BUDGET = 5400  # seconds — 8B model generates slower than 4B
MAX_GEN = 512
MAX_SEQ_LEN = 2048
MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
N_PROBLEMS_TARGET = 80
N_POSITION_BINS = 10

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_069"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 8-shot exemplars (same as Phase 1 / exp_068) ──
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
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return ds


def get_kv(past_kv, layer_idx):
    from transformers import DynamicCache
    if isinstance(past_kv, DynamicCache):
        if hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
            return past_kv.layers[layer_idx].keys, past_kv.layers[layer_idx].values
        else:
            return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
    else:
        return past_kv[layer_idx][0], past_kv[layer_idx][1]


def run_probe(X, y, n_splits=5):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    if X.shape[0] < n_splits:
        return 0.0, 1.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas = np.logspace(-2, 6, 50)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    y_pred = np.zeros_like(y)
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

    # Use same fractional layer indices as exp_068 for fair comparison
    probe_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    print(f"Model: {n_layers} layers, hidden_size={hidden_size}")
    print(f"Probe layers: {probe_layers}")

    ds = load_gsm8k()
    print(f"GSM8K: {len(ds)} test problems")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1: Generate CoT and extract features
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 1: Generate CoT and extract features")
    print(f"{'='*60}")

    from transformers import DynamicCache

    problems_data = []
    n_attempted = 0

    for i in range(min(N_PROBLEMS_TARGET * 3, len(ds))):
        if len(problems_data) >= N_PROBLEMS_TARGET:
            break
        elapsed = time.time() - t0
        if elapsed > TIME_BUDGET * 0.60:
            print(f"  Time budget: stopping at {len(problems_data)} problems ({elapsed:.0f}s)")
            break

        question = ds[i]['question']
        gold = extract_gold(ds[i]['answer'])
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

        generated_ids = output.sequences[0][prompt_len:]
        gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        pred_answer = extract_answer(gen_text)

        if pred_answer is None or str(pred_answer).strip() != str(gold).strip():
            del output; gc.collect(); torch.cuda.empty_cache()
            continue

        full_ids = output.sequences[0:1]
        total_len = full_ids.shape[1]
        reasoning_len = total_len - prompt_len

        if reasoning_len < 30:
            del output; gc.collect(); torch.cuda.empty_cache()
            continue

        # Forward pass to get KV cache for all positions
        with torch.no_grad():
            out = model(full_ids, use_cache=True)
        past_kv = out.past_key_values

        # Token embeddings for text baseline
        with torch.no_grad():
            reasoning_ids = full_ids[0, prompt_len:]
            token_embeddings = model.model.embed_tokens(reasoning_ids).cpu().float()

        k_sample, _ = get_kv(past_kv, 0)
        n_kv_heads = k_sample.shape[1]
        head_dim = k_sample.shape[3]
        kv_dim = n_kv_heads * head_dim

        problem = {
            'answer': float(gold),
            'reasoning_len': int(reasoning_len),
        }

        for layer_idx in probe_layers:
            k_full, v_full = get_kv(past_kv, layer_idx)
            k_r = k_full[0, :, prompt_len:, :].cpu().float()
            v_r = v_full[0, :, prompt_len:, :].cpu().float()

            rl = k_r.shape[1]
            k_flat = k_r.permute(1, 0, 2).reshape(rl, kv_dim)
            v_flat = v_r.permute(1, 0, 2).reshape(rl, kv_dim)

            for bin_idx in range(N_POSITION_BINS):
                start = int(bin_idx * rl / N_POSITION_BINS)
                end = int((bin_idx + 1) * rl / N_POSITION_BINS)
                end = max(end, start + 1)

                # WITHIN-BIN features
                k_bin = k_flat[start:end].mean(dim=0).numpy()
                v_bin = v_flat[start:end].mean(dim=0).numpy()
                problem[f'K_bin_L{layer_idx}_b{bin_idx}'] = k_bin.astype(np.float32)
                problem[f'V_bin_L{layer_idx}_b{bin_idx}'] = v_bin.astype(np.float32)

                # CUMULATIVE features (from start to end of this bin)
                k_cum = k_flat[:end].mean(dim=0).numpy()
                v_cum = v_flat[:end].mean(dim=0).numpy()
                problem[f'K_cum_L{layer_idx}_b{bin_idx}'] = k_cum.astype(np.float32)
                problem[f'V_cum_L{layer_idx}_b{bin_idx}'] = v_cum.astype(np.float32)

        # Text features — both within-bin and cumulative
        for bin_idx in range(N_POSITION_BINS):
            start = int(bin_idx * reasoning_len / N_POSITION_BINS)
            end_pos = int((bin_idx + 1) * reasoning_len / N_POSITION_BINS)
            end_pos = max(end_pos, 1)
            start = max(start, 0)
            if start >= end_pos:
                start = end_pos - 1

            # Within-bin text
            text_bin = token_embeddings[start:end_pos].mean(dim=0).numpy()
            problem[f'text_bin_b{bin_idx}'] = text_bin.astype(np.float32)

            # Cumulative text
            text_cum = token_embeddings[:end_pos].mean(dim=0).numpy()
            problem[f'text_cum_b{bin_idx}'] = text_cum.astype(np.float32)

        problems_data.append(problem)

        del past_kv, out, output, token_embeddings, k_r, v_r, k_flat, v_flat
        gc.collect(); torch.cuda.empty_cache()

        if len(problems_data) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {len(problems_data)}/{N_PROBLEMS_TARGET} valid | "
                  f"{n_attempted} attempted | {elapsed:.0f}s elapsed")

    n_valid = len(problems_data)
    gen_time = time.time() - t0
    print(f"\nGeneration complete: {n_valid} valid problems from {n_attempted} attempted ({gen_time:.0f}s)")
    print(f"  KV dim: {n_kv_heads} heads x {head_dim} dim = {kv_dim}")
    print(f"  Text dim: {hidden_size}")

    if n_valid < 20:
        print("ERROR: Too few valid problems. Aborting.")
        sys.exit(1)

    del model; gc.collect(); torch.cuda.empty_cache()
    print("Model freed from GPU.")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2: Linear Probing — BOTH cumulative and within-bin
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 2: Training linear probes (cumulative + within-bin)")
    print(f"{'='*60}")

    answers = np.array([p['answer'] for p in problems_data])
    log_answers = np.sign(answers) * np.log1p(np.abs(answers))

    print(f"Answer range: {answers.min():.0f} to {answers.max():.0f}")
    print(f"Log-answer range: {log_answers.min():.2f} to {log_answers.max():.2f}")
    print(f"Unique answers: {len(np.unique(answers))}")

    probe_results = {}

    # For each comparison type: cumulative and within-bin
    for mode in ['cum', 'bin']:
        print(f"\n  === {mode.upper()} features ===")
        for layer_idx in probe_layers:
            print(f"\n  Layer {layer_idx} ({mode}):")
            for bin_idx in range(N_POSITION_BINS):
                frac = (bin_idx + 1) / N_POSITION_BINS

                # K-probe
                X_k = np.stack([p[f'K_{mode}_L{layer_idx}_b{bin_idx}'] for p in problems_data])
                r_k, p_k = run_probe(X_k, log_answers)
                probe_results[(f'K_{mode}', layer_idx, bin_idx)] = {'r': r_k, 'p': p_k}

                # V-probe
                X_v = np.stack([p[f'V_{mode}_L{layer_idx}_b{bin_idx}'] for p in problems_data])
                r_v, p_v = run_probe(X_v, log_answers)
                probe_results[(f'V_{mode}', layer_idx, bin_idx)] = {'r': r_v, 'p': p_v}

                # Shuffle control
                rng = np.random.RandomState(SEED + bin_idx)
                shuffled = log_answers.copy()
                rng.shuffle(shuffled)
                r_s, p_s = run_probe(X_k, shuffled)
                probe_results[(f'shuffle_{mode}', layer_idx, bin_idx)] = {'r': r_s, 'p': p_s}

                if (bin_idx + 1) % 5 == 0:
                    print(f"    bin {frac:.0%}: K r={r_k:.3f}  V r={r_v:.3f}  shuf r={r_s:.3f}")

        # Text baseline (matching mode)
        print(f"\n  Text baseline ({mode}):")
        for bin_idx in range(N_POSITION_BINS):
            frac = (bin_idx + 1) / N_POSITION_BINS
            X_text = np.stack([p[f'text_{mode}_b{bin_idx}'] for p in problems_data])
            r_t, p_t = run_probe(X_text, log_answers)
            probe_results[(f'text_{mode}', 'all', bin_idx)] = {'r': r_t, 'p': p_t}
            if (bin_idx + 1) % 5 == 0:
                print(f"    bin {frac:.0%}: text r={r_t:.3f}")

    probe_time = time.time() - t0 - gen_time
    print(f"\nProbing complete ({probe_time:.0f}s)")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 3: Analysis & Visualization
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 3: Analysis & Visualization")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    positions = [(b + 1) / N_POSITION_BINS for b in range(N_POSITION_BINS)]

    # Find best layer (highest mean cumulative K r)
    best_layer = max(probe_layers,
                     key=lambda l: np.mean([probe_results[('K_cum', l, b)]['r']
                                            for b in range(N_POSITION_BINS)]))

    # ── Figure 1: CUMULATIVE comparison (fair) — best layer ──
    fig, ax = plt.subplots(figsize=(12, 7))

    k_cum_rs = [probe_results[('K_cum', best_layer, b)]['r'] for b in range(N_POSITION_BINS)]
    v_cum_rs = [probe_results[('V_cum', best_layer, b)]['r'] for b in range(N_POSITION_BINS)]
    text_cum_rs = [probe_results[('text_cum', 'all', b)]['r'] for b in range(N_POSITION_BINS)]
    shuf_cum_rs = [probe_results[('shuffle_cum', best_layer, b)]['r'] for b in range(N_POSITION_BINS)]

    ax.plot(positions, k_cum_rs, 'b-o', label=f'K-probe cumulative (L{best_layer})',
            linewidth=2.5, markersize=8)
    ax.plot(positions, v_cum_rs, 'r-s', label=f'V-probe cumulative (L{best_layer})',
            linewidth=2.5, markersize=8)
    ax.plot(positions, text_cum_rs, 'g-^', label='Text baseline cumulative',
            linewidth=2.5, markersize=8)
    ax.plot(positions, shuf_cum_rs, 'k--x', label='Shuffle control', linewidth=1.5,
            markersize=6, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # Mark early decodability gap
    threshold = 0.3
    k_dec = next((p for p, r in zip(positions, k_cum_rs) if r >= threshold), None)
    text_dec = next((p for p, r in zip(positions, text_cum_rs) if r >= threshold), None)

    if k_dec is not None:
        ax.axvline(x=k_dec, color='blue', linestyle=':', alpha=0.4)
        ax.text(k_dec + 0.01, 0.92, f'K: {k_dec:.0%}', color='blue', fontsize=10)
    if text_dec is not None:
        ax.axvline(x=text_dec, color='green', linestyle=':', alpha=0.4)
        ax.text(text_dec + 0.01, 0.87, f'Text: {text_dec:.0%}', color='green', fontsize=10)

    if k_dec is not None and text_dec is not None:
        gap = text_dec - k_dec
        ax.axhline(y=threshold, color='purple', linestyle=':', alpha=0.3)
        ax.annotate(f'Early decodability gap: {gap:.0%} of CoT',
                    xy=(0.5, threshold + 0.03), fontsize=12, color='purple',
                    ha='center', fontweight='bold')
    elif k_dec is not None and text_dec is None:
        ax.axhline(y=threshold, color='purple', linestyle=':', alpha=0.3)
        ax.annotate(f'K reaches r>={threshold}, text NEVER does',
                    xy=(0.5, threshold + 0.03), fontsize=12, color='purple',
                    ha='center', fontweight='bold')

    ax.set_xlabel('Normalized position in CoT', fontsize=13)
    ax.set_ylabel('Pearson r (cross-validated)', fontsize=13)
    ax.set_title(f'Early Answer Decodability — CUMULATIVE Features (Fair Comparison)\n'
                 f'{MODEL_NAME} | Best layer: {best_layer} | n={n_valid}', fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(-0.4, 1.0)
    ax.set_xlim(0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'cumulative_decodability_main.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cumulative_decodability_main.png")

    # ── Figure 2: WITHIN-BIN comparison — best layer ──
    best_layer_bin = max(probe_layers,
                         key=lambda l: np.mean([probe_results[('K_bin', l, b)]['r']
                                                for b in range(N_POSITION_BINS)]))
    fig, ax = plt.subplots(figsize=(12, 7))

    k_bin_rs = [probe_results[('K_bin', best_layer_bin, b)]['r'] for b in range(N_POSITION_BINS)]
    v_bin_rs = [probe_results[('V_bin', best_layer_bin, b)]['r'] for b in range(N_POSITION_BINS)]
    text_bin_rs = [probe_results[('text_bin', 'all', b)]['r'] for b in range(N_POSITION_BINS)]
    shuf_bin_rs = [probe_results[('shuffle_bin', best_layer_bin, b)]['r'] for b in range(N_POSITION_BINS)]

    ax.plot(positions, k_bin_rs, 'b-o', label=f'K-probe within-bin (L{best_layer_bin})',
            linewidth=2.5, markersize=8)
    ax.plot(positions, v_bin_rs, 'r-s', label=f'V-probe within-bin (L{best_layer_bin})',
            linewidth=2.5, markersize=8)
    ax.plot(positions, text_bin_rs, 'g-^', label='Text baseline within-bin',
            linewidth=2.5, markersize=8)
    ax.plot(positions, shuf_bin_rs, 'k--x', label='Shuffle control', linewidth=1.5,
            markersize=6, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Normalized position in CoT', fontsize=13)
    ax.set_ylabel('Pearson r (cross-validated)', fontsize=13)
    ax.set_title(f'Answer Decodability — WITHIN-BIN Features (Local Comparison)\n'
                 f'{MODEL_NAME} | Best layer: {best_layer_bin} | n={n_valid}', fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(-0.5, 1.0)
    ax.set_xlim(0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'withinbin_decodability_main.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: withinbin_decodability_main.png")

    # ── Figure 3: All layers cumulative (2x2) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Cumulative Decodability by Layer — {MODEL_NAME} (n={n_valid})', fontsize=14)

    for ax_idx, layer_idx in enumerate(probe_layers):
        ax = axes[ax_idx // 2][ax_idx % 2]

        k_rs = [probe_results[('K_cum', layer_idx, b)]['r'] for b in range(N_POSITION_BINS)]
        v_rs = [probe_results[('V_cum', layer_idx, b)]['r'] for b in range(N_POSITION_BINS)]
        text_rs = [probe_results[('text_cum', 'all', b)]['r'] for b in range(N_POSITION_BINS)]
        shuf_rs = [probe_results[('shuffle_cum', layer_idx, b)]['r'] for b in range(N_POSITION_BINS)]

        ax.plot(positions, k_rs, 'b-o', label='K-probe', linewidth=2, markersize=5)
        ax.plot(positions, v_rs, 'r-s', label='V-probe', linewidth=2, markersize=5)
        ax.plot(positions, text_rs, 'g-^', label='Text baseline', linewidth=2, markersize=5)
        ax.plot(positions, shuf_rs, 'k--x', label='Shuffle ctrl', linewidth=1, markersize=5, alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        depth_pct = int(100 * layer_idx / (n_layers - 1))
        ax.set_title(f'Layer {layer_idx} ({depth_pct}% depth)')
        ax.set_xlabel('Position in CoT')
        ax.set_ylabel('Pearson r')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.4, 1.0)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'cumulative_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cumulative_by_layer.png")

    # ── Figure 4: K-V advantage heatmaps (cumulative and within-bin) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, mode in enumerate(['cum', 'bin']):
        kv_diff = np.array([[probe_results[(f'K_{mode}', l, b)]['r'] -
                              probe_results[(f'V_{mode}', l, b)]['r']
                              for b in range(N_POSITION_BINS)]
                             for l in probe_layers])

        vabs = max(abs(kv_diff.min()), abs(kv_diff.max()), 0.1)
        im = axes[ax_idx].imshow(kv_diff, aspect='auto', cmap='RdBu_r',
                                  vmin=-vabs, vmax=vabs)
        axes[ax_idx].set_yticks(range(len(probe_layers)))
        axes[ax_idx].set_yticklabels([f'L{l}' for l in probe_layers])
        axes[ax_idx].set_xticks(range(N_POSITION_BINS))
        axes[ax_idx].set_xticklabels([f'{int(100*(b+1)/N_POSITION_BINS)}%'
                                       for b in range(N_POSITION_BINS)], fontsize=8)
        axes[ax_idx].set_xlabel('Position in CoT')
        axes[ax_idx].set_ylabel('Layer')
        axes[ax_idx].set_title(f'K minus V ({mode}ulative)')
        plt.colorbar(im, ax=axes[ax_idx])

        for i in range(len(probe_layers)):
            for j in range(N_POSITION_BINS):
                color = 'white' if abs(kv_diff[i, j]) > vabs * 0.5 else 'black'
                axes[ax_idx].text(j, i, f'{kv_diff[i,j]:.2f}', ha='center',
                                   va='center', fontsize=7, color=color)

    fig.suptitle(f'K minus V Probe Advantage — {MODEL_NAME}', fontsize=13)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'kv_advantage_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: kv_advantage_heatmap.png")

    # ── Figure 5: K-Text advantage (cumulative) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for layer_idx in probe_layers:
        k_rs = [probe_results[('K_cum', layer_idx, b)]['r'] for b in range(N_POSITION_BINS)]
        text_rs = [probe_results[('text_cum', 'all', b)]['r'] for b in range(N_POSITION_BINS)]
        diff = [k - t for k, t in zip(k_rs, text_rs)]
        depth_pct = int(100 * layer_idx / (n_layers - 1))
        ax.plot(positions, diff, '-o', label=f'L{layer_idx} ({depth_pct}%)',
                linewidth=2, markersize=5)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.7)
    ax.fill_between([0, 1.1], 0, 1, alpha=0.05, color='blue', label='K > Text')
    ax.fill_between([0, 1.1], -1, 0, alpha=0.05, color='green', label='Text > K')
    ax.set_xlabel('Normalized position in CoT', fontsize=13)
    ax.set_ylabel('K-probe r minus Text-baseline r (cumulative)', fontsize=13)
    ax.set_title(f'K-Cache vs Text: Answer Information Advantage\n'
                 f'{MODEL_NAME} | n={n_valid} | Positive = hidden channel carries MORE info',
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 1.05)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'k_minus_text_cumulative.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: k_minus_text_cumulative.png")

    # ── Figure 6: Cross-model comparison (if exp_068 results exist) ──
    qwen_results_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_068" / "results.json"
    if qwen_results_path.exists():
        with open(qwen_results_path) as f:
            qwen_res = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Cumulative comparison at best layers
        qwen_best_layer = qwen_res['best_layer_cum']

        # Qwen K, V, Text
        qwen_k = [qwen_res['probe_results'][f'K_cum_L{qwen_best_layer}_b{b}']['r']
                   for b in range(N_POSITION_BINS)]
        qwen_text = [qwen_res['probe_results'][f'text_cum_Lall_b{b}']['r']
                     for b in range(N_POSITION_BINS)]

        # Llama K, V, Text
        llama_k = k_cum_rs
        llama_text = text_cum_rs

        axes[0].plot(positions, qwen_k, 'b-o', label=f'Qwen K (L{qwen_best_layer})',
                     linewidth=2, markersize=6)
        axes[0].plot(positions, qwen_text, 'b--^', label='Qwen Text',
                     linewidth=1.5, markersize=5, alpha=0.7)
        axes[0].plot(positions, llama_k, 'r-o', label=f'Llama K (L{best_layer})',
                     linewidth=2, markersize=6)
        axes[0].plot(positions, llama_text, 'r--^', label='Llama Text',
                     linewidth=1.5, markersize=5, alpha=0.7)
        axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        axes[0].set_xlabel('Position in CoT', fontsize=12)
        axes[0].set_ylabel('Pearson r', fontsize=12)
        axes[0].set_title('Cross-Model: K-probe vs Text Baseline', fontsize=13)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-0.4, 1.0)

        # K-Text advantage comparison
        qwen_diff = [k - t for k, t in zip(qwen_k, qwen_text)]
        llama_diff = [k - t for k, t in zip(llama_k, llama_text)]

        axes[1].plot(positions, qwen_diff, 'b-o', label='Qwen (K - Text)',
                     linewidth=2, markersize=6)
        axes[1].plot(positions, llama_diff, 'r-o', label='Llama (K - Text)',
                     linewidth=2, markersize=6)
        axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.7)
        axes[1].fill_between([0, 1.1], 0, 0.5, alpha=0.05, color='blue')
        axes[1].fill_between([0, 1.1], -0.5, 0, alpha=0.05, color='green')
        axes[1].set_xlabel('Position in CoT', fontsize=12)
        axes[1].set_ylabel('K-probe r minus Text r', fontsize=12)
        axes[1].set_title('Cross-Model: Hidden Channel Advantage', fontsize=13)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle('Cross-Model Replication: Qwen3-4B-Base vs Llama-3.1-8B-Instruct', fontsize=14)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'cross_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: cross_model_comparison.png")
    else:
        print("Qwen exp_068 results not found — skipping cross-model comparison figure")

    # ═══════════════════════════════════════════════════════════════════
    # Results Summary
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")

    print(f"\nModel: {MODEL_NAME}")
    print(f"Valid problems: {n_valid}")

    for mode in ['cum', 'bin']:
        bl = best_layer if mode == 'cum' else best_layer_bin
        mode_name = 'CUMULATIVE' if mode == 'cum' else 'WITHIN-BIN'
        print(f"\n--- {mode_name} — Best layer: {bl} ---")
        print(f"{'Pos':>5}  {'K':>7}  {'V':>7}  {'Text':>7}  {'Shuf':>7}  {'K-V':>7}  {'K-T':>7}")
        for b in range(N_POSITION_BINS):
            frac = (b + 1) / N_POSITION_BINS
            rk = probe_results[(f'K_{mode}', bl, b)]['r']
            rv = probe_results[(f'V_{mode}', bl, b)]['r']
            rt = probe_results[(f'text_{mode}', 'all', b)]['r']
            rs = probe_results[(f'shuffle_{mode}', bl, b)]['r']
            print(f"  {frac:.0%}  {rk:+.3f}  {rv:+.3f}  {rt:+.3f}  {rs:+.3f}  "
                  f"{rk-rv:+.3f}  {rk-rt:+.3f}")

    # Early decodability gaps
    threshold_r = 0.3
    print(f"\n--- Early decodability gap (cumulative, r >= {threshold_r}) ---")
    for layer_idx in probe_layers:
        k_rs_l = [probe_results[('K_cum', layer_idx, b)]['r'] for b in range(N_POSITION_BINS)]
        text_rs_l = [probe_results[('text_cum', 'all', b)]['r'] for b in range(N_POSITION_BINS)]

        k_pos = next((p for p, r in zip(positions, k_rs_l) if r >= threshold_r), None)
        t_pos = next((p for p, r in zip(positions, text_rs_l) if r >= threshold_r), None)

        if k_pos is not None and t_pos is not None:
            gap = t_pos - k_pos
            print(f"  Layer {layer_idx}: K@{k_pos:.0%}, Text@{t_pos:.0%}, gap={gap:.0%}")
        elif k_pos is not None and t_pos is None:
            print(f"  Layer {layer_idx}: K@{k_pos:.0%}, Text=NEVER, gap=TEXT NEVER REACHES")
        elif k_pos is None and t_pos is not None:
            print(f"  Layer {layer_idx}: K=NEVER, Text@{t_pos:.0%}, gap=NEGATIVE")
        else:
            print(f"  Layer {layer_idx}: NEITHER reaches threshold")

    # K vs V summary
    print(f"\n--- K vs V mean r (cumulative) ---")
    for layer_idx in probe_layers:
        mean_k = np.mean([probe_results[('K_cum', layer_idx, b)]['r'] for b in range(N_POSITION_BINS)])
        mean_v = np.mean([probe_results[('V_cum', layer_idx, b)]['r'] for b in range(N_POSITION_BINS)])
        print(f"  Layer {layer_idx}: K={mean_k:.3f}  V={mean_v:.3f}  K-V={mean_k-mean_v:+.3f}")

    mean_text_cum = np.mean([probe_results[('text_cum', 'all', b)]['r'] for b in range(N_POSITION_BINS)])
    print(f"  Text baseline (cum): {mean_text_cum:.3f}")

    # ── Save JSON ──
    results_json = {
        'model': MODEL_NAME,
        'n_problems': n_valid,
        'n_attempted': n_attempted,
        'n_position_bins': N_POSITION_BINS,
        'probe_layers': probe_layers,
        'n_layers': n_layers,
        'hidden_size': hidden_size,
        'kv_dim': kv_dim,
        'n_kv_heads': n_kv_heads,
        'head_dim': head_dim,
        'best_layer_cum': best_layer,
        'best_layer_bin': best_layer_bin,
        'answer_stats': {
            'min': float(answers.min()),
            'max': float(answers.max()),
            'mean': float(answers.mean()),
            'std': float(answers.std()),
            'n_unique': int(len(np.unique(answers))),
        },
        'probe_results': {},
        'early_decodability_gap_cum': {},
        'generation_time_s': gen_time,
        'probe_time_s': probe_time,
        'total_time_s': time.time() - t0,
    }

    for (cond, layer, bin_idx), res in probe_results.items():
        key = f"{cond}_L{layer}_b{bin_idx}"
        results_json['probe_results'][key] = {'r': round(res['r'], 4), 'p': round(res['p'], 6)}

    for layer_idx in probe_layers:
        k_rs_l = [probe_results[('K_cum', layer_idx, b)]['r'] for b in range(N_POSITION_BINS)]
        text_rs_l = [probe_results[('text_cum', 'all', b)]['r'] for b in range(N_POSITION_BINS)]
        k_pos = next((p for p, r in zip(positions, k_rs_l) if r >= threshold_r), None)
        t_pos = next((p for p, r in zip(positions, text_rs_l) if r >= threshold_r), None)
        gap = (t_pos - k_pos) if (k_pos is not None and t_pos is not None) else None
        results_json['early_decodability_gap_cum'][f'L{layer_idx}'] = {
            'k_threshold_pos': k_pos, 'text_threshold_pos': t_pos, 'gap': gap,
        }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
