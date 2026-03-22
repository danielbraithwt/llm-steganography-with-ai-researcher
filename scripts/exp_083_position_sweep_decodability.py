#!/usr/bin/env python3
"""
Experiment 083: Position-Sweep KV Decodability (Full Experiment A)

Does the KV cache know the answer before the text reveals it?

Method:
1. Generate plain-text CoT on Qwen3-4B-Base (250 problems, greedy)
2. For each correct problem, forward pass to extract V and K at all CoT
   token positions (layers 18, 27 = 50%, 75% depth)
3. Bin tokens by relative position in chain (20 bins: 0-5%, 5-10%, ..., 95-100%)
4. At each bin, train ridge probe on V → log(final_answer) with GroupKFold
5. Compute text-reveals-answer fraction at each bin (cumulative: has the answer
   appeared anywhere in the text by this position?)
6. Plot decodability curve: V-probe R vs text-reveals-answer fraction

Key metric: Early Decodability Gap — how much earlier does V-probe decode
the answer compared to when the text explicitly reveals it.

Controls:
- K-probe alongside V-probe (compare K vs V decodability)
- Shuffle control at each bin (random answer labels → R ≈ 0)
- Layer comparison (L18 vs L27)
- Per-problem first-reveal position analysis
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
N_PROBLEMS = 250
PROBE_LAYERS = [18, 27]  # 50% and 75% of 36 layers
N_BINS = 20
N_FOLDS = 5

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_083"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (from exp_078, no <<EXPR=RESULT>>) ──
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


def answer_in_text(text, answer_str):
    """Check if answer appears as a standalone number in text."""
    clean_text = text.replace(',', '')
    clean_ans = answer_str.replace(',', '')
    if not clean_ans:
        return False
    # Word boundary check: answer not part of a larger number
    pattern = r'(?<!\d)' + re.escape(clean_ans) + r'(?!\d)'
    return bool(re.search(pattern, clean_text))


def log_transform(x):
    """Log-transform for regression target."""
    return float(np.sign(x) * np.log1p(np.abs(x)))


def get_kv(cache, layer):
    """Extract K, V from cache (handles DynamicCache and tuple formats)."""
    if hasattr(cache, 'layers'):
        # New DynamicCache with .layers[i].keys/.values
        l = cache.layers[layer]
        return l.keys, l.values
    if hasattr(cache, 'key_cache'):
        return cache.key_cache[layer], cache.value_cache[layer]
    return cache[layer][0], cache[layer][1]


def find_hash_pos_in_gen(gen_ids, tokenizer):
    """Find the token index where #### starts in generated tokens."""
    cum_text = ""
    for i, tid in enumerate(gen_ids):
        cum_text += tokenizer.decode([tid], skip_special_tokens=False)
        if "####" in cum_text:
            prefix = cum_text[:cum_text.index("####")]
            char_count = 0
            for j, t in enumerate(gen_ids):
                char_count += len(tokenizer.decode([t], skip_special_tokens=False))
                if char_count > len(prefix):
                    return j
            return i
    return None


def main():
    t0 = time.time()

    print("=" * 60)
    print("Experiment 083: Position-Sweep KV Decodability")
    print("Full Experiment A: Does the KV cache know the answer")
    print("before the text reveals it?")
    print("=" * 60)

    # ── Load model ──
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
    )
    model.eval()
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {model.config.num_hidden_layers}")

    kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    # Use model.config.head_dim if available (Qwen uses 128, not hidden_size/num_heads)
    if hasattr(model.config, 'head_dim'):
        head_dim = model.config.head_dim
    else:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    kv_dim = kv_heads * head_dim
    print(f"KV heads: {kv_heads}, head_dim: {head_dim}, KV dim: {kv_dim}")

    ds = load_gsm8k()
    print(f"GSM8K test set: {len(ds)} problems")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Generate CoT traces
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: Generating CoT traces")
    print("=" * 60)

    generations = []
    for i in range(min(N_PROBLEMS, len(ds))):
        if time.time() - t0 > TIME_BUDGET * 0.25:
            print(f"  Time budget (25%) reached at problem {i}")
            break

        question = ds[i]['question']
        gold = extract_gold(ds[i]['answer'])
        prompt = build_prompt(question)

        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        prompt_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_GEN, do_sample=False,
                temperature=1.0, use_cache=True
            )

        gen_ids = output[0][prompt_len:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_answer(gen_text)
        correct = pred is not None and gold is not None and str(pred).strip() == str(gold).strip()

        generations.append({
            'idx': i,
            'question': question,
            'gold': gold,
            'gen_text': gen_text,
            'gen_ids': gen_ids,
            'pred': pred,
            'correct': correct,
        })

        if (i + 1) % 50 == 0:
            n_corr = sum(g['correct'] for g in generations)
            elapsed = time.time() - t0
            print(f"  Generated {i+1} problems, {n_corr}/{len(generations)} correct "
                  f"({100*n_corr/len(generations):.1f}%) [{elapsed:.0f}s]")

    n_total = len(generations)
    n_correct = sum(g['correct'] for g in generations)
    print(f"\nPhase 1 complete: {n_total} generated, {n_correct} correct "
          f"({100*n_correct/n_total:.1f}%)")

    # Filter to correct problems with valid #### marker
    correct_gens = []
    for g in generations:
        if not g['correct']:
            continue
        hash_pos = find_hash_pos_in_gen(g['gen_ids'], tokenizer)
        if hash_pos is None or hash_pos < 10:
            continue
        g['hash_pos_gen'] = hash_pos
        correct_gens.append(g)

    print(f"Usable correct problems: {len(correct_gens)}")

    # Save generation info
    gen_summary = {
        'n_total': n_total,
        'n_correct': n_correct,
        'accuracy': 100 * n_correct / n_total,
        'n_usable': len(correct_gens),
    }
    with open(RESULTS_DIR / 'generation_summary.json', 'w') as f:
        json.dump(gen_summary, f, indent=2)

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Forward pass + KV extraction at all CoT positions
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: Extracting KV cache at all CoT positions")
    print("=" * 60)

    # Data collectors
    data = {layer: {
        'V': [], 'K': [],
        'rel_pos': [], 'text_reveals': [],
        'final_answer': [], 'problem_idx': [],
    } for layer in PROBE_LAYERS}

    # Per-problem metadata for text-reveals analysis
    first_reveal_positions = []  # relative position where answer first appears
    problem_cot_lengths = []
    problem_answers = []

    n_extracted = 0

    for pi, gen in enumerate(correct_gens):
        if time.time() - t0 > TIME_BUDGET * 0.60:
            print(f"\n  Time budget (60%) reached at problem {pi}")
            break

        # Build full sequence: prompt + generated CoT + #### + answer
        prompt_text = build_prompt(gen['question'])
        prompt_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids']
        prompt_len = prompt_ids.shape[1]

        gen_token_ids = gen['gen_ids']
        full_ids = torch.cat([
            prompt_ids[0],
            torch.tensor(gen_token_ids, dtype=torch.long)
        ]).unsqueeze(0).to(model.device)

        if full_ids.shape[1] > MAX_SEQ_LEN:
            continue

        hash_pos_gen = gen['hash_pos_gen']
        cot_length = hash_pos_gen  # CoT is gen_ids[0:hash_pos_gen]

        if cot_length < 10:
            continue

        # Forward pass to get KV cache
        with torch.no_grad():
            outputs = model(full_ids, use_cache=True)

        kv_cache = outputs.past_key_values

        # Decode CoT tokens for text-reveals-answer check
        cot_ids = gen_token_ids[:hash_pos_gen]
        final_answer_str = str(gen['gold']).strip()
        final_answer_log = log_transform(float(gen['gold']))

        cot_token_texts = [tokenizer.decode([tid], skip_special_tokens=False) for tid in cot_ids]

        # Build cumulative text and check answer at each position
        cum_text = ""
        reveals_at = []
        first_reveal_rel = 1.0  # default: never revealed (shouldn't happen)
        for j, tok_text in enumerate(cot_token_texts):
            cum_text += tok_text
            revealed = answer_in_text(cum_text, final_answer_str)
            reveals_at.append(revealed)
            if revealed and first_reveal_rel == 1.0:
                first_reveal_rel = j / cot_length

        first_reveal_positions.append(first_reveal_rel)
        problem_cot_lengths.append(cot_length)
        problem_answers.append(gen['gold'])

        # Extract V and K at CoT positions (absolute pos = prompt_len + j)
        for layer in PROBE_LAYERS:
            K_layer, V_layer = get_kv(kv_cache, layer)
            # Shape: (1, n_kv_heads, seq_len, head_dim)

            for j in range(cot_length):
                abs_pos = prompt_len + j
                rel_pos = j / cot_length

                v_vec = V_layer[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                k_vec = K_layer[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()

                data[layer]['V'].append(v_vec)
                data[layer]['K'].append(k_vec)
                data[layer]['rel_pos'].append(rel_pos)
                data[layer]['text_reveals'].append(reveals_at[j])
                data[layer]['final_answer'].append(final_answer_log)
                data[layer]['problem_idx'].append(pi)

        n_extracted += 1

        # Free memory
        del outputs, kv_cache
        torch.cuda.empty_cache()

        if (pi + 1) % 25 == 0:
            elapsed = time.time() - t0
            n_vecs = len(data[PROBE_LAYERS[0]]['V'])
            print(f"  Extracted {pi+1}/{len(correct_gens)} problems, "
                  f"{n_vecs} total vectors [{elapsed:.0f}s]")

    print(f"\nPhase 2 complete: {n_extracted} problems extracted")
    print(f"Total V vectors per layer: {len(data[PROBE_LAYERS[0]]['V'])}")

    # Free model to save memory for probing
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded.")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Bin by relative position, train probes
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 3: Position-sweep probing")
    print("=" * 60)

    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge

    bin_edges = np.linspace(0, 1, N_BINS + 1)

    # Compute text-reveals-answer cumulative curve (per-problem based)
    # For each bin center c: fraction of problems whose answer first appears by position c
    first_reveal_arr = np.array(first_reveal_positions[:n_extracted])
    text_reveals_curve = []
    for b in range(N_BINS):
        bin_upper = bin_edges[b + 1]
        frac = np.mean(first_reveal_arr <= bin_upper)
        text_reveals_curve.append({
            'bin_center': float((bin_edges[b] + bin_edges[b + 1]) / 2),
            'fraction': float(frac),
        })

    print("\nText-reveals-answer curve:")
    for entry in text_reveals_curve:
        print(f"  Position {entry['bin_center']:.2f}: {100*entry['fraction']:.1f}% problems revealed")

    # Probe at each bin for each layer
    probe_results = {}

    for layer in PROBE_LAYERS:
        print(f"\n--- Layer {layer} ---")

        V_all = np.array(data[layer]['V'])
        K_all = np.array(data[layer]['K'])
        answers = np.array(data[layer]['final_answer'])
        rel_pos = np.array(data[layer]['rel_pos'])
        problem_idx = np.array(data[layer]['problem_idx'])

        n_total_vecs = len(V_all)
        print(f"Total vectors: {n_total_vecs}, dim: {V_all.shape[1]}")

        bin_indices = np.digitize(rel_pos, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, N_BINS - 1)

        layer_results = {}

        for b in range(N_BINS):
            mask = bin_indices == b
            n_samples = mask.sum()
            unique_probs = np.unique(problem_idx[mask])
            n_problems = len(unique_probs)

            if n_problems < 10:
                print(f"  Bin {b} ({(bin_edges[b]+bin_edges[b+1])/2:.2f}): "
                      f"SKIPPED (only {n_problems} problems)")
                continue

            V_bin = V_all[mask]
            K_bin = K_all[mask]
            ans_bin = answers[mask]
            prob_bin = problem_idx[mask]

            bin_center = (bin_edges[b] + bin_edges[b + 1]) / 2

            # GroupKFold
            n_splits = min(N_FOLDS, n_problems)
            gkf = GroupKFold(n_splits=n_splits)

            v_preds = np.zeros(n_samples)
            k_preds = np.zeros(n_samples)

            for train_idx, test_idx in gkf.split(V_bin, ans_bin, prob_bin):
                # V probe
                ridge_v = Ridge(alpha=1.0)
                ridge_v.fit(V_bin[train_idx], ans_bin[train_idx])
                v_preds[test_idx] = ridge_v.predict(V_bin[test_idx])

                # K probe
                ridge_k = Ridge(alpha=1.0)
                ridge_k.fit(K_bin[train_idx], ans_bin[train_idx])
                k_preds[test_idx] = ridge_k.predict(K_bin[test_idx])

            # Shuffle control: permute answers at problem level
            rng = np.random.RandomState(SEED + layer * 100 + b)
            prob_answer_map = {}
            for idx in range(len(ans_bin)):
                prob_answer_map[prob_bin[idx]] = ans_bin[idx]
            shuffled_probs = rng.permutation(unique_probs)
            shuffle_map = {orig: shuffled_probs[i % len(shuffled_probs)]
                          for i, orig in enumerate(unique_probs)}
            ans_shuffled = np.array([prob_answer_map[shuffle_map[p]] for p in prob_bin])

            v_shuf_preds = np.zeros(n_samples)
            for train_idx, test_idx in gkf.split(V_bin, ans_shuffled, prob_bin):
                ridge_s = Ridge(alpha=1.0)
                ridge_s.fit(V_bin[train_idx], ans_shuffled[train_idx])
                v_shuf_preds[test_idx] = ridge_s.predict(V_bin[test_idx])

            # Compute Pearson R
            def safe_corr(a, b):
                if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                    return 0.0
                return float(np.corrcoef(a, b)[0, 1])

            v_r = safe_corr(ans_bin, v_preds)
            k_r = safe_corr(ans_bin, k_preds)
            v_shuf_r = safe_corr(ans_shuffled, v_shuf_preds)

            # R² (can be negative for bad fits)
            def r_squared(y_true, y_pred):
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            v_r2 = r_squared(ans_bin, v_preds)
            k_r2 = r_squared(ans_bin, k_preds)

            layer_results[b] = {
                'bin_center': float(bin_center),
                'n_samples': int(n_samples),
                'n_problems': int(n_problems),
                'V_R': v_r,
                'K_R': k_r,
                'V_shuf_R': v_shuf_r,
                'V_R2': v_r2,
                'K_R2': k_r2,
                'text_reveals_frac': text_reveals_curve[b]['fraction'],
            }

            print(f"  Bin {b} ({bin_center:.2f}): V_R={v_r:.3f}, K_R={k_r:.3f}, "
                  f"shuf={v_shuf_r:.3f}, text_frac={text_reveals_curve[b]['fraction']:.3f} "
                  f"(n={n_samples}, probs={n_problems})")

        probe_results[f'L{layer}'] = layer_results

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Compute Early Decodability Gap
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 4: Early Decodability Gap")
    print("=" * 60)

    gap_results = {}
    for layer in PROBE_LAYERS:
        layer_key = f'L{layer}'
        lr = probe_results[layer_key]
        if not lr:
            continue

        # Get V_R values and bin centers
        bins_sorted = sorted(lr.keys())
        bin_centers = [lr[b]['bin_center'] for b in bins_sorted]
        v_rs = [lr[b]['V_R'] for b in bins_sorted]
        text_fracs = [lr[b]['text_reveals_frac'] for b in bins_sorted]

        max_v_r = max(v_rs) if v_rs else 0
        threshold = 0.5 * max_v_r  # 50% of max decodability

        # Find position where V-probe first exceeds threshold
        v_reach_pos = 1.0
        if max_v_r > 0.05:  # Only meaningful if max R > 0.05
            for bc, vr in zip(bin_centers, v_rs):
                if vr >= threshold:
                    v_reach_pos = bc
                    break

        # Find position where text-reveals-answer crosses 50%
        text_reach_pos = 1.0
        for bc, tf in zip(bin_centers, text_fracs):
            if tf >= 0.50:
                text_reach_pos = bc
                break

        gap = text_reach_pos - v_reach_pos  # positive = V decodes earlier

        gap_results[layer_key] = {
            'max_V_R': float(max_v_r),
            'threshold_50pct': float(threshold),
            'V_reaches_threshold_at': float(v_reach_pos),
            'text_reaches_50pct_at': float(text_reach_pos),
            'early_decodability_gap': float(gap),
        }

        print(f"\n{layer_key}:")
        print(f"  Max V_R: {max_v_r:.3f}")
        print(f"  V-probe reaches 50% of max ({threshold:.3f}) at position: {v_reach_pos:.2f}")
        print(f"  Text reveals answer to 50% problems at position: {text_reach_pos:.2f}")
        print(f"  Early Decodability Gap: {gap:.2f} ({100*gap:.0f}% of chain)")

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: Bootstrap significance for key results
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 5: Bootstrap significance")
    print("=" * 60)

    # Bootstrap the peak V_R to get confidence interval
    best_layer = max(PROBE_LAYERS, key=lambda l: gap_results.get(f'L{l}', {}).get('max_V_R', 0))
    best_layer_key = f'L{best_layer}'
    lr = probe_results[best_layer_key]
    peak_bin = max(lr.keys(), key=lambda b: lr[b]['V_R'])

    # Bootstrap: resample problems and re-run probe
    mask_peak = np.digitize(
        np.array(data[best_layer]['rel_pos']), bin_edges
    ) - 1 == peak_bin
    mask_peak = np.clip(mask_peak.astype(int), 0, 1).astype(bool)

    V_peak = np.array(data[best_layer]['V'])[mask_peak]
    ans_peak = np.array(data[best_layer]['final_answer'])[mask_peak]
    prob_peak = np.array(data[best_layer]['problem_idx'])[mask_peak]

    n_boot = 500
    boot_rs = []
    unique_probs_peak = np.unique(prob_peak)

    if len(unique_probs_peak) >= 10:
        for boot_i in range(n_boot):
            if time.time() - t0 > TIME_BUDGET * 0.85:
                print(f"  Bootstrap interrupted at {boot_i}/{n_boot}")
                break

            rng_boot = np.random.RandomState(SEED + boot_i)
            # Resample problems with replacement
            boot_probs = rng_boot.choice(unique_probs_peak, size=len(unique_probs_peak), replace=True)

            # Collect data for resampled problems
            boot_V = []
            boot_ans = []
            boot_grp = []
            for new_idx, orig_prob in enumerate(boot_probs):
                pmask = prob_peak == orig_prob
                boot_V.append(V_peak[pmask])
                boot_ans.append(ans_peak[pmask])
                boot_grp.append(np.full(pmask.sum(), new_idx))

            boot_V = np.concatenate(boot_V)
            boot_ans = np.concatenate(boot_ans)
            boot_grp = np.concatenate(boot_grp)

            if len(np.unique(boot_grp)) < 5:
                continue

            # GroupKFold probe
            gkf_boot = GroupKFold(n_splits=min(5, len(np.unique(boot_grp))))
            preds = np.zeros(len(boot_ans))
            try:
                for tr, te in gkf_boot.split(boot_V, boot_ans, boot_grp):
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(boot_V[tr], boot_ans[tr])
                    preds[te] = ridge.predict(boot_V[te])
                r_val = np.corrcoef(boot_ans, preds)[0, 1] if np.std(preds) > 1e-10 else 0
                boot_rs.append(r_val)
            except Exception:
                pass

    boot_rs = np.array(boot_rs)
    if len(boot_rs) > 0:
        ci_lower = float(np.percentile(boot_rs, 2.5))
        ci_upper = float(np.percentile(boot_rs, 97.5))
        boot_mean = float(np.mean(boot_rs))
        p_value = float(np.mean(boot_rs <= 0))
        print(f"\nPeak V_R bootstrap ({best_layer_key}, bin {peak_bin}):")
        print(f"  Mean R: {boot_mean:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"  P(R ≤ 0): {p_value:.4f}")
    else:
        ci_lower, ci_upper, boot_mean, p_value = 0, 0, 0, 1
        print("  Bootstrap: insufficient data")

    bootstrap_results = {
        'layer': best_layer_key,
        'bin': peak_bin,
        'n_boot': len(boot_rs),
        'mean_R': boot_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
    }

    # ══════════════════════════════════════════════════════════════
    # PHASE 6: Figures
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 6: Generating figures")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Figure 1: Main decodability curve (V-probe R + text-reveals) ──
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    colors_layer = {'L18': '#3498db', 'L27': '#e74c3c'}

    for layer in PROBE_LAYERS:
        layer_key = f'L{layer}'
        lr = probe_results[layer_key]
        if not lr:
            continue
        bins_sorted = sorted(lr.keys())
        bin_centers = [lr[b]['bin_center'] for b in bins_sorted]
        v_rs = [lr[b]['V_R'] for b in bins_sorted]
        shuf_rs = [lr[b]['V_shuf_R'] for b in bins_sorted]

        ax1.plot(bin_centers, v_rs, 'o-', color=colors_layer[layer_key],
                 label=f'V-probe R ({layer_key})', linewidth=2, markersize=5)
        ax1.plot(bin_centers, shuf_rs, 'x--', color=colors_layer[layer_key],
                 alpha=0.4, label=f'Shuffle ctrl ({layer_key})', linewidth=1)

    # Text-reveals curve
    text_centers = [tc['bin_center'] for tc in text_reveals_curve]
    text_fracs = [tc['fraction'] for tc in text_reveals_curve]
    ax1.plot(text_centers, text_fracs, 's-', color='#2ecc71', linewidth=2.5,
             markersize=6, label='Text reveals answer (fraction)', zorder=5)

    # Mark the decodability gap if present
    for layer in PROBE_LAYERS:
        layer_key = f'L{layer}'
        if layer_key in gap_results:
            gr = gap_results[layer_key]
            if gr['early_decodability_gap'] > 0.05:
                ax1.axvspan(gr['V_reaches_threshold_at'], gr['text_reaches_50pct_at'],
                           alpha=0.1, color=colors_layer[layer_key],
                           label=f'Gap ({100*gr["early_decodability_gap"]:.0f}% chain, {layer_key})')

    ax1.set_xlabel('Relative Position in CoT Chain', fontsize=13)
    ax1.set_ylabel('Probe R / Text Fraction', fontsize=13)
    ax1.set_title('Position-Sweep KV Decodability: When Does the KV Cache\n'
                   'Know the Answer vs When Does the Text Reveal It?',
                   fontsize=14)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.1, 1.05)
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'decodability_curve.png', dpi=150)
    plt.close()
    print("  Saved decodability_curve.png")

    # ── Figure 2: K vs V decodability comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, layer in enumerate(PROBE_LAYERS):
        layer_key = f'L{layer}'
        lr = probe_results[layer_key]
        if not lr:
            continue
        bins_sorted = sorted(lr.keys())
        bin_centers = [lr[b]['bin_center'] for b in bins_sorted]
        v_rs = [lr[b]['V_R'] for b in bins_sorted]
        k_rs = [lr[b]['K_R'] for b in bins_sorted]

        axes[ax_idx].plot(bin_centers, v_rs, 'o-', color='#e74c3c',
                          label='V-probe R', linewidth=2, markersize=5)
        axes[ax_idx].plot(bin_centers, k_rs, 's-', color='#3498db',
                          label='K-probe R', linewidth=2, markersize=5)
        axes[ax_idx].plot(text_centers, text_fracs, '^-', color='#2ecc71',
                          label='Text reveals', linewidth=1.5, markersize=4, alpha=0.7)
        axes[ax_idx].set_xlabel('Relative Position in CoT', fontsize=11)
        axes[ax_idx].set_ylabel('Probe R / Text Fraction', fontsize=11)
        axes[ax_idx].set_title(f'{layer_key} (depth={100*layer/36:.0f}%)', fontsize=12)
        axes[ax_idx].legend(fontsize=9)
        axes[ax_idx].set_xlim(0, 1)
        axes[ax_idx].set_ylim(-0.1, 1.05)
        axes[ax_idx].grid(True, alpha=0.3)

    plt.suptitle('K vs V Decodability by Position', fontsize=14)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'kv_comparison.png', dpi=150)
    plt.close()
    print("  Saved kv_comparison.png")

    # ── Figure 3: First-reveal position histogram ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(first_reveal_arr, bins=20, range=(0, 1), color='#2ecc71',
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Relative Position Where Answer First Appears in Text', fontsize=12)
    ax.set_ylabel('Number of Problems', fontsize=12)
    ax.set_title(f'Distribution of Answer First-Reveal Positions\n'
                  f'(n={n_extracted} problems, median={np.median(first_reveal_arr):.2f})',
                  fontsize=13)
    ax.axvline(x=np.median(first_reveal_arr), color='red', linestyle='--',
               label=f'Median: {np.median(first_reveal_arr):.2f}')
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'first_reveal_histogram.png', dpi=150)
    plt.close()
    print("  Saved first_reveal_histogram.png")

    # ── Figure 4: Bootstrap distribution ──
    if len(boot_rs) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(boot_rs, bins=40, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='R=0 (null)')
        ax.axvline(x=boot_mean, color='blue', linestyle='-', linewidth=2,
                   label=f'Mean R={boot_mean:.3f}')
        ax.axvline(x=ci_lower, color='blue', linestyle=':', alpha=0.5)
        ax.axvline(x=ci_upper, color='blue', linestyle=':', alpha=0.5)
        ax.set_xlabel('Probe R (bootstrapped)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Bootstrap Distribution of Peak V-probe R\n'
                      f'{best_layer_key}, bin {peak_bin} | '
                      f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}] | p={p_value:.4f}',
                      fontsize=12)
        ax.legend(fontsize=11)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'bootstrap_peak_R.png', dpi=150)
        plt.close()
        print("  Saved bootstrap_peak_R.png")

    # ══════════════════════════════════════════════════════════════
    # Save all results
    # ══════════════════════════════════════════════════════════════
    full_results = {
        'generation': gen_summary,
        'n_extracted': n_extracted,
        'probe_layers': PROBE_LAYERS,
        'n_bins': N_BINS,
        'kv_dim': kv_dim,
        'probe_results': probe_results,
        'text_reveals_curve': text_reveals_curve,
        'gap_results': gap_results,
        'bootstrap': bootstrap_results,
        'first_reveal_stats': {
            'median': float(np.median(first_reveal_arr)),
            'mean': float(np.mean(first_reveal_arr)),
            'std': float(np.std(first_reveal_arr)),
            'min': float(np.min(first_reveal_arr)),
            'max': float(np.max(first_reveal_arr)),
        },
        'elapsed_seconds': time.time() - t0,
    }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Problems extracted: {n_extracted}")
    print(f"Median answer first-reveal position: {np.median(first_reveal_arr):.2f}")
    for layer in PROBE_LAYERS:
        lk = f'L{layer}'
        if lk in gap_results:
            gr = gap_results[lk]
            print(f"\n{lk} (depth={100*layer/36:.0f}%):")
            print(f"  Peak V_R: {gr['max_V_R']:.3f}")
            print(f"  V reaches half-peak at: {gr['V_reaches_threshold_at']:.2f}")
            print(f"  Text 50% at: {gr['text_reaches_50pct_at']:.2f}")
            print(f"  EARLY DECODABILITY GAP: {100*gr['early_decodability_gap']:.0f}% of chain")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
