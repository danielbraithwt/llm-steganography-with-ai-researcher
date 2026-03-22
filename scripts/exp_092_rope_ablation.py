#!/usr/bin/env python3
"""
Experiment 092: RoPE Ablation — Is K>V Probing a Positional Encoding Artifact?

Compares probing performance of:
- K_pre:  K after k_proj + k_norm, BEFORE RoPE
- K_post: K from KV cache, AFTER RoPE (standard)
- V:      Value vectors from KV cache

At 8 selected layers (4 ramp + 4 plateau) × 20 position bins.

DISCONFIRMATORY: If K_pre|nums < V|nums, then K>V from exp_091 was a RoPE artifact.
"""

import os
import json
import time
import gc
import re
import sys
import warnings

import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

T0 = time.time()
TIME_BUDGET = 6600  # 110 min
MAX_GEN = 512
MAX_SEQ_LEN = 2048
MODEL_NAME = 'Qwen/Qwen3-4B-Base'
N_PROBLEMS = 250
N_BINS = 20
N_FOLDS = 5
MAX_NUMS_DIM = 30

# 8 layers: 4 ramp (L0, L3, L6, L9) + 4 plateau (L12, L18, L24, L30)
TARGET_LAYERS = [0, 3, 6, 9, 12, 18, 24, 30]
SHUFFLE_LAYERS = [0, 9, 18, 30]

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_092"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Path to exp_091 K results for sanity check
EXP091_RESULTS = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_091" / "results.json"

# ── Plain-text 8-shot exemplars (identical to exp_084/089/091) ──
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
    clean_text = text.replace(',', '')
    clean_ans = answer_str.replace(',', '')
    if not clean_ans:
        return False
    pattern = r'(?<!\d)' + re.escape(clean_ans) + r'(?!\d)'
    return bool(re.search(pattern, clean_text))


def log_transform(x):
    return float(np.sign(x) * np.log1p(np.abs(x)))


def get_kv(cache, layer):
    if hasattr(cache, 'layers'):
        l = cache.layers[layer]
        return l.keys, l.values
    if hasattr(cache, 'key_cache'):
        return cache.key_cache[layer], cache.value_cache[layer]
    return cache[layer][0], cache[layer][1]


def find_hash_pos_in_gen(gen_ids, tokenizer):
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


def extract_numbers_from_text(text):
    nums = []
    for m in re.finditer(r'(?<![.\w])(\d+(?:\.\d+)?)(?![.\w])', text):
        try:
            val = float(m.group(1))
            nums.append(val)
        except:
            pass
    return nums


def numbers_to_features(nums, max_dim=MAX_NUMS_DIM):
    if not nums:
        return np.zeros(max_dim + 6, dtype=np.float32)
    log_nums = [np.log1p(abs(n)) * (1 if n >= 0 else -1) for n in nums]
    if len(log_nums) > max_dim:
        truncated = log_nums[-max_dim:]
    else:
        truncated = log_nums
    features = np.zeros(max_dim, dtype=np.float32)
    features[:len(truncated)] = truncated
    arr = np.array(log_nums)
    stats = np.array([
        len(log_nums), np.mean(arr), np.std(arr),
        np.max(arr), np.min(arr), np.sum(arr),
    ], dtype=np.float32)
    return np.concatenate([features, stats])


NUMS_FEAT_DIM = MAX_NUMS_DIM + 6  # 36


def main():
    t0 = time.time()

    print("=" * 70)
    print("Experiment 092: RoPE Ablation — Pre-RoPE K vs Post-RoPE K vs V")
    print("DISCONFIRMATORY: Testing whether K>V probing is a RoPE artifact")
    print(f"Target layers: {TARGET_LAYERS}")
    print("=" * 70)

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

    n_layers = model.config.num_hidden_layers
    kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // model.config.num_attention_heads)
    kv_dim = kv_heads * head_dim
    print(f"Model: {MODEL_NAME}, {n_layers} layers")
    print(f"KV heads: {kv_heads}, head_dim: {head_dim}, KV dim: {kv_dim}")

    # Verify k_norm exists at target layers
    for layer in TARGET_LAYERS:
        attn = model.model.layers[layer].self_attn
        assert hasattr(attn, 'k_norm'), f"Layer {layer} has no k_norm!"
    print("k_norm verified at all target layers")

    ds = load_gsm8k()
    print(f"GSM8K test set: {len(ds)} problems")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Generate CoT traces
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: Generating CoT traces")
    print("=" * 70)

    generations = []
    for i in range(min(N_PROBLEMS, len(ds))):
        if time.time() - t0 > TIME_BUDGET * 0.20:
            print(f"  Time budget (20%) reached at problem {i}")
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
            'idx': i, 'question': question, 'gold': gold,
            'gen_text': gen_text, 'gen_ids': gen_ids,
            'pred': pred, 'correct': correct,
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

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Forward pass with hooks — extract K_pre, K_post, V
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 2: Extracting K_pre + K_post + V at {len(TARGET_LAYERS)} layers")
    print("=" * 70)

    # Shared data (same across K_pre, K_post, V)
    shared = {
        'cumNums': [], 'rel_pos': [], 'text_reveals': [],
        'final_answer': [], 'problem_idx': [],
    }

    # Per-layer vectors for 3 sources
    K_pre_by_layer = {layer: [] for layer in TARGET_LAYERS}
    K_post_by_layer = {layer: [] for layer in TARGET_LAYERS}
    V_by_layer = {layer: [] for layer in TARGET_LAYERS}

    first_reveal_positions = []
    n_extracted = 0

    for pi, gen in enumerate(correct_gens):
        if time.time() - t0 > TIME_BUDGET * 0.55:
            print(f"\n  Time budget (55%) reached at problem {pi}")
            break

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
        cot_length = hash_pos_gen
        if cot_length < 10:
            continue

        # Register hooks to capture pre-RoPE K
        k_pre_storage = {}

        def make_k_norm_hook(layer_idx):
            def hook(module, input, output):
                # k_norm output: [batch, seq_len, kv_heads, head_dim]
                # Transpose to match KV cache format: [batch, kv_heads, seq_len, head_dim]
                k_pre_storage[layer_idx] = output.detach().transpose(1, 2)
            return hook

        handles = []
        for layer in TARGET_LAYERS:
            handle = model.model.layers[layer].self_attn.k_norm.register_forward_hook(
                make_k_norm_hook(layer)
            )
            handles.append(handle)

        # Forward pass — gets KV cache (post-RoPE K and V) + triggers hooks (pre-RoPE K)
        with torch.no_grad():
            outputs = model(full_ids, use_cache=True)
        kv_cache = outputs.past_key_values

        # Remove hooks
        for h in handles:
            h.remove()

        cot_ids = gen_token_ids[:hash_pos_gen]
        final_answer_str = str(gen['gold']).strip()
        final_answer_log = log_transform(float(gen['gold']))
        question_text = gen['question']

        # Build cumulative text and numbers
        cot_token_texts = [tokenizer.decode([tid], skip_special_tokens=False) for tid in cot_ids]
        cum_cot_text = ""
        first_reveal_rel = 1.0
        token_cumNums = []
        token_reveals = []
        question_nums = extract_numbers_from_text(question_text)

        for j, tok_text in enumerate(cot_token_texts):
            cum_cot_text += tok_text
            cot_nums = extract_numbers_from_text(cum_cot_text)
            all_nums = question_nums + cot_nums
            token_cumNums.append(numbers_to_features(all_nums))
            revealed = answer_in_text(cum_cot_text, final_answer_str)
            token_reveals.append(revealed)
            if revealed and first_reveal_rel == 1.0:
                first_reveal_rel = j / cot_length

        first_reveal_positions.append(first_reveal_rel)

        # Store shared data
        for j in range(cot_length):
            shared['cumNums'].append(token_cumNums[j])
            shared['rel_pos'].append(j / cot_length)
            shared['text_reveals'].append(token_reveals[j])
            shared['final_answer'].append(final_answer_log)
            shared['problem_idx'].append(pi)

        # Extract vectors at target layers
        for layer in TARGET_LAYERS:
            K_post_layer, V_layer = get_kv(kv_cache, layer)
            K_pre_layer = k_pre_storage[layer]

            for j in range(cot_length):
                abs_pos = prompt_len + j

                # K_pre: [batch, kv_heads, seq_len, head_dim]
                k_pre_vec = K_pre_layer[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                K_pre_by_layer[layer].append(k_pre_vec)

                # K_post: [batch, kv_heads, seq_len, head_dim]
                k_post_vec = K_post_layer[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                K_post_by_layer[layer].append(k_post_vec)

                # V: [batch, kv_heads, seq_len, head_dim]
                v_vec = V_layer[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                V_by_layer[layer].append(v_vec)

        n_extracted += 1

        del outputs, kv_cache, k_pre_storage
        torch.cuda.empty_cache()

        if (pi + 1) % 25 == 0:
            elapsed = time.time() - t0
            n_vecs = len(shared['rel_pos'])
            print(f"  Extracted {pi+1}/{len(correct_gens)} problems, "
                  f"{n_vecs} total vectors [{elapsed:.0f}s]")

    print(f"\nPhase 2 complete: {n_extracted} problems extracted")
    print(f"Total vectors per layer: {len(shared['rel_pos'])}")

    # Convert to numpy
    for key in shared:
        shared[key] = np.array(shared[key])

    print("Converting vector arrays to numpy...")
    for layer in TARGET_LAYERS:
        K_pre_by_layer[layer] = np.array(K_pre_by_layer[layer], dtype=np.float32)
        K_post_by_layer[layer] = np.array(K_post_by_layer[layer], dtype=np.float32)
        V_by_layer[layer] = np.array(V_by_layer[layer], dtype=np.float32)

    print(f"  K_pre shape per layer: {K_pre_by_layer[TARGET_LAYERS[0]].shape}")
    print(f"  K_post shape per layer: {K_post_by_layer[TARGET_LAYERS[0]].shape}")
    print(f"  V shape per layer: {V_by_layer[TARGET_LAYERS[0]].shape}")
    mem_gb = (K_pre_by_layer[TARGET_LAYERS[0]].nbytes * len(TARGET_LAYERS) * 3) / 1e9
    print(f"  Total vector memory: {mem_gb:.1f} GB")

    # ── Sanity check: K_pre vs K_post should differ by RoPE ──
    print("\nSanity check: K_pre vs K_post cosine similarity...")
    for layer in [TARGET_LAYERS[0], TARGET_LAYERS[-1]]:
        kpre = K_pre_by_layer[layer][:100]
        kpost = K_post_by_layer[layer][:100]
        # Cosine similarity
        norms_pre = np.linalg.norm(kpre, axis=1, keepdims=True) + 1e-10
        norms_post = np.linalg.norm(kpost, axis=1, keepdims=True) + 1e-10
        cos_sim = np.sum((kpre / norms_pre) * (kpost / norms_post), axis=1)
        print(f"  L{layer}: mean cos_sim(K_pre, K_post) = {cos_sim.mean():.4f} "
              f"(should be <1.0 if RoPE changes K; std={cos_sim.std():.4f})")
        # K_pre vs V
        v = V_by_layer[layer][:100]
        norms_v = np.linalg.norm(v, axis=1, keepdims=True) + 1e-10
        cos_kv = np.sum((kpre / norms_pre) * (v / norms_v), axis=1)
        print(f"  L{layer}: mean cos_sim(K_pre, V) = {cos_kv.mean():.4f}")

    # Free model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model freed from GPU")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Probing — K_pre|nums, K_post|nums, V|nums
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: Probing at 8 layers × 20 bins × 3 sources")
    print("=" * 70)

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    y_all = shared['final_answer']
    cumNums_all = shared['cumNums']
    rel_pos_all = shared['rel_pos']
    groups_all = shared['problem_idx']
    n_total_vecs = len(y_all)

    # Assign bins
    bin_edges = np.linspace(0, 1.0, N_BINS + 1)
    bin_idx = np.digitize(rel_pos_all, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, N_BINS - 1)

    alphas = np.logspace(-2, 6, 50)
    gkf = GroupKFold(n_splits=N_FOLDS)

    # Results storage
    results = {
        'layers': TARGET_LAYERS,
        'n_bins': N_BINS,
        'n_problems': n_extracted,
        'n_total_vecs': n_total_vecs,
        'kv_dim': kv_dim,
        'K_pre': {str(l): {} for l in TARGET_LAYERS},
        'K_post': {str(l): {} for l in TARGET_LAYERS},
        'V': {str(l): {} for l in TARGET_LAYERS},
        'shuffle': {str(l): {} for l in SHUFFLE_LAYERS},
    }

    def probe_cv(X, y, groups):
        """Ridge regression with GroupKFold CV, return mean R across folds."""
        if len(np.unique(groups)) < N_FOLDS:
            return float('nan')
        rs = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(X[train_idx], y[train_idx])
            pred = ridge.predict(X[test_idx])
            ss_res = np.sum((y[test_idx] - pred) ** 2)
            ss_tot = np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
            r = 1 - ss_res / (ss_tot + 1e-10)
            rs.append(r)
        return float(np.mean(rs))

    total_probes = len(TARGET_LAYERS) * N_BINS * 3  # K_pre, K_post, V
    probe_count = 0

    for li, layer in enumerate(TARGET_LAYERS):
        if time.time() - t0 > TIME_BUDGET * 0.92:
            print(f"\n  Time budget (92%) reached at layer {layer}")
            break

        print(f"\n  Layer {layer} ({li+1}/{len(TARGET_LAYERS)})...")
        k_pre_data = K_pre_by_layer[layer]
        k_post_data = K_post_by_layer[layer]
        v_data = V_by_layer[layer]

        for b in range(N_BINS):
            mask = bin_idx == b
            if mask.sum() < 20:
                for source_name in ['K_pre', 'K_post', 'V']:
                    results[source_name][str(layer)][str(b)] = {
                        'R': float('nan'), 'nums_R': float('nan'), 'resid_R': float('nan'),
                        'n': int(mask.sum()),
                    }
                continue

            y_bin = y_all[mask]
            nums_bin = cumNums_all[mask]
            groups_bin = groups_all[mask]

            # Compute nums_R and y_resid (shared across K_pre, K_post, V)
            nums_R = probe_cv(nums_bin, y_bin, groups_bin)

            # Compute y_resid: residuals after removing nums prediction
            ridge_nums = RidgeCV(alphas=alphas)
            ridge_nums.fit(nums_bin, y_bin)
            y_resid = y_bin - ridge_nums.predict(nums_bin)

            # Probe each source
            for source_name, source_data in [
                ('K_pre', k_pre_data),
                ('K_post', k_post_data),
                ('V', v_data),
            ]:
                X_bin = source_data[mask]
                raw_R = probe_cv(X_bin, y_bin, groups_bin)
                resid_R = probe_cv(X_bin, y_resid, groups_bin)

                results[source_name][str(layer)][str(b)] = {
                    'R': float(raw_R),
                    'nums_R': float(nums_R),
                    'resid_R': float(resid_R),  # This is the |nums metric
                    'n': int(mask.sum()),
                }
                probe_count += 1

            if b == 0 or b == N_BINS - 1:
                print(f"    Bin {b}: K_pre|nums={results['K_pre'][str(layer)][str(b)]['resid_R']:.3f}, "
                      f"K_post|nums={results['K_post'][str(layer)][str(b)]['resid_R']:.3f}, "
                      f"V|nums={results['V'][str(layer)][str(b)]['resid_R']:.3f}, "
                      f"n={int(mask.sum())}")

        # Shuffle control at selected layers
        if layer in SHUFFLE_LAYERS:
            print(f"    Running shuffle control...")
            # Pick 3 bins for shuffle
            for b in [0, N_BINS // 2, N_BINS - 1]:
                mask = bin_idx == b
                if mask.sum() < 20:
                    continue
                y_bin = y_all[mask]
                nums_bin = cumNums_all[mask]
                groups_bin = groups_all[mask]

                ridge_nums = RidgeCV(alphas=alphas)
                ridge_nums.fit(nums_bin, y_bin)
                y_resid = y_bin - ridge_nums.predict(nums_bin)

                # Shuffle: randomize y_resid across problems
                rng = np.random.RandomState(SEED + layer * 100 + b)
                unique_groups = np.unique(groups_bin)
                perm = rng.permutation(len(unique_groups))
                group_map = dict(zip(unique_groups, unique_groups[perm]))
                y_shuffled = np.zeros_like(y_resid)
                for g_orig, g_new in group_map.items():
                    orig_mask = groups_bin == g_orig
                    new_mask = groups_bin == g_new
                    n_min = min(orig_mask.sum(), new_mask.sum())
                    y_shuffled[orig_mask][:n_min] = y_resid[new_mask][:n_min]

                for source_name, source_data in [('K_post', k_post_data)]:
                    X_bin = source_data[mask]
                    shuf_R = probe_cv(X_bin, y_shuffled, groups_bin)
                    results['shuffle'][str(layer)][str(b)] = {
                        'R': float(shuf_R),
                    }

        # Print layer summary
        layer_kpre = [results['K_pre'][str(layer)][str(b)]['resid_R']
                      for b in range(N_BINS) if str(b) in results['K_pre'][str(layer)]
                      and not np.isnan(results['K_pre'][str(layer)][str(b)]['resid_R'])]
        layer_kpost = [results['K_post'][str(layer)][str(b)]['resid_R']
                       for b in range(N_BINS) if str(b) in results['K_post'][str(layer)]
                       and not np.isnan(results['K_post'][str(layer)][str(b)]['resid_R'])]
        layer_v = [results['V'][str(layer)][str(b)]['resid_R']
                   for b in range(N_BINS) if str(b) in results['V'][str(layer)]
                   and not np.isnan(results['V'][str(layer)][str(b)]['resid_R'])]

        if layer_kpre and layer_kpost and layer_v:
            mean_kpre = np.mean(layer_kpre)
            mean_kpost = np.mean(layer_kpost)
            mean_v = np.mean(layer_v)
            kpre_wins = sum(1 for a, b in zip(layer_kpre, layer_v) if a > b)
            kpost_wins = sum(1 for a, b in zip(layer_kpost, layer_v) if a > b)
            print(f"    SUMMARY L{layer}: K_pre|nums={mean_kpre:.3f}, "
                  f"K_post|nums={mean_kpost:.3f}, V|nums={mean_v:.3f}")
            print(f"    K_pre>V: {kpre_wins}/{len(layer_v)} bins, "
                  f"K_post>V: {kpost_wins}/{len(layer_v)} bins")
            elapsed = time.time() - t0
            print(f"    [{elapsed:.0f}s elapsed, {probe_count}/{total_probes} probes done]")

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Analysis and Visualization
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: Analysis")
    print("=" * 70)

    # Compute summary statistics
    summary = {'layers': {}, 'overall': {}}

    for layer in TARGET_LAYERS:
        sl = str(layer)
        if sl not in results['K_pre'] or not results['K_pre'][sl]:
            continue

        kpre_vals = []
        kpost_vals = []
        v_vals = []
        for b in range(N_BINS):
            sb = str(b)
            if sb in results['K_pre'][sl] and not np.isnan(results['K_pre'][sl][sb]['resid_R']):
                kpre_vals.append(results['K_pre'][sl][sb]['resid_R'])
                kpost_vals.append(results['K_post'][sl][sb]['resid_R'])
                v_vals.append(results['V'][sl][sb]['resid_R'])

        if not kpre_vals:
            continue

        kpre_arr = np.array(kpre_vals)
        kpost_arr = np.array(kpost_vals)
        v_arr = np.array(v_vals)

        kpre_mean = float(np.mean(kpre_arr))
        kpost_mean = float(np.mean(kpost_arr))
        v_mean = float(np.mean(v_arr))
        kpre_gt_v = int(np.sum(kpre_arr > v_arr))
        kpost_gt_v = int(np.sum(kpost_arr > v_arr))
        kpre_gt_kpost = int(np.sum(kpre_arr > kpost_arr))
        rope_contrib = kpost_mean - kpre_mean  # RoPE contribution
        intrinsic_gap = kpre_mean - v_mean  # Intrinsic K>V gap

        summary['layers'][sl] = {
            'K_pre_mean': kpre_mean,
            'K_post_mean': kpost_mean,
            'V_mean': v_mean,
            'K_pre_gt_V': kpre_gt_v,
            'K_post_gt_V': kpost_gt_v,
            'K_pre_gt_K_post': kpre_gt_kpost,
            'rope_contribution': rope_contrib,
            'intrinsic_K_gt_V': intrinsic_gap,
            'n_bins': len(kpre_vals),
            'depth_pct': round(100 * layer / (n_layers - 1), 1),
        }

        print(f"  L{layer} ({100*layer/(n_layers-1):.0f}%): K_pre={kpre_mean:+.3f}, "
              f"K_post={kpost_mean:+.3f}, V={v_mean:+.3f} | "
              f"RoPE_contrib={rope_contrib:+.3f}, intrinsic_K>V={intrinsic_gap:+.3f} | "
              f"K_pre>V: {kpre_gt_v}/{len(kpre_vals)}, K_post>V: {kpost_gt_v}/{len(kpre_vals)}")

    # Overall summary
    all_kpre = []
    all_kpost = []
    all_v = []
    for sl in summary['layers']:
        all_kpre.append(summary['layers'][sl]['K_pre_mean'])
        all_kpost.append(summary['layers'][sl]['K_post_mean'])
        all_v.append(summary['layers'][sl]['V_mean'])

    if all_kpre:
        overall_kpre = np.mean(all_kpre)
        overall_kpost = np.mean(all_kpost)
        overall_v = np.mean(all_v)
        overall_rope = overall_kpost - overall_kpre
        overall_intrinsic = overall_kpre - overall_v
        kpre_wins_layers = sum(1 for a, b in zip(all_kpre, all_v) if a > b)
        kpost_wins_layers = sum(1 for a, b in zip(all_kpost, all_v) if a > b)

        summary['overall'] = {
            'K_pre_mean': float(overall_kpre),
            'K_post_mean': float(overall_kpost),
            'V_mean': float(overall_v),
            'rope_contribution': float(overall_rope),
            'intrinsic_K_gt_V': float(overall_intrinsic),
            'K_pre_wins_layers': kpre_wins_layers,
            'K_post_wins_layers': kpost_wins_layers,
            'total_layers': len(all_kpre),
            'fraction_rope': float(overall_rope / (overall_kpost - overall_v + 1e-10)),
            'fraction_intrinsic': float(overall_intrinsic / (overall_kpost - overall_v + 1e-10)),
        }

        print(f"\n{'='*70}")
        print(f"OVERALL SUMMARY:")
        print(f"  K_pre|nums  (no RoPE): {overall_kpre:+.4f}")
        print(f"  K_post|nums (w/ RoPE): {overall_kpost:+.4f}")
        print(f"  V|nums:                {overall_v:+.4f}")
        print(f"  ---")
        print(f"  RoPE contribution:     {overall_rope:+.4f} "
              f"({100*summary['overall']['fraction_rope']:.1f}% of K>V gap)")
        print(f"  Intrinsic K>V gap:     {overall_intrinsic:+.4f} "
              f"({100*summary['overall']['fraction_intrinsic']:.1f}% of K>V gap)")
        print(f"  K_pre > V at {kpre_wins_layers}/{len(all_kpre)} layers")
        print(f"  K_post > V at {kpost_wins_layers}/{len(all_kpost)} layers")
        print(f"{'='*70}")

        # Ramp vs plateau breakdown
        ramp_layers = [l for l in TARGET_LAYERS if l <= 9]
        plat_layers = [l for l in TARGET_LAYERS if l > 9]

        for phase_name, phase_layers in [("RAMP (L0-L9)", ramp_layers), ("PLATEAU (L12-L30)", plat_layers)]:
            phase_kpre = [summary['layers'][str(l)]['K_pre_mean'] for l in phase_layers if str(l) in summary['layers']]
            phase_kpost = [summary['layers'][str(l)]['K_post_mean'] for l in phase_layers if str(l) in summary['layers']]
            phase_v = [summary['layers'][str(l)]['V_mean'] for l in phase_layers if str(l) in summary['layers']]
            if phase_kpre:
                print(f"\n  {phase_name}:")
                print(f"    K_pre={np.mean(phase_kpre):+.4f}, K_post={np.mean(phase_kpost):+.4f}, V={np.mean(phase_v):+.4f}")
                print(f"    RoPE contrib: {np.mean(phase_kpost)-np.mean(phase_kpre):+.4f}, "
                      f"Intrinsic K>V: {np.mean(phase_kpre)-np.mean(phase_v):+.4f}")

    results['summary'] = summary

    # Save results
    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: Visualization
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 5: Creating figures")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Figure 1: Bar chart — K_pre vs K_post vs V by layer ──
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(len(TARGET_LAYERS))
    width = 0.25

    kpre_means = [summary['layers'].get(str(l), {}).get('K_pre_mean', 0) for l in TARGET_LAYERS]
    kpost_means = [summary['layers'].get(str(l), {}).get('K_post_mean', 0) for l in TARGET_LAYERS]
    v_means = [summary['layers'].get(str(l), {}).get('V_mean', 0) for l in TARGET_LAYERS]

    bars1 = ax.bar(x - width, kpre_means, width, label='K_pre (no RoPE)', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x, kpost_means, width, label='K_post (w/ RoPE)', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, v_means, width, label='V', color='#3498db', alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean |nums (forward-looking signal)', fontsize=12)
    ax.set_title('RoPE Ablation: K_pre vs K_post vs V Forward-Looking Signal\nQwen3-4B-Base', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in TARGET_LAYERS])
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.grid(axis='y', alpha=0.3)

    # Add phase labels
    ax.axvspan(-0.5, 3.5, alpha=0.05, color='orange', label='_')
    ax.axvspan(3.5, 7.5, alpha=0.05, color='blue', label='_')
    ax.text(1.5, ax.get_ylim()[1]*0.95, 'RAMP', ha='center', fontsize=10, style='italic', color='orange')
    ax.text(5.5, ax.get_ylim()[1]*0.95, 'PLATEAU', ha='center', fontsize=10, style='italic', color='blue')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "rope_ablation_by_layer.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved rope_ablation_by_layer.png")

    # ── Figure 2: Stacked decomposition — RoPE vs Intrinsic contribution ──
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    rope_contribs = [summary['layers'].get(str(l), {}).get('rope_contribution', 0) for l in TARGET_LAYERS]
    intrinsic_gaps = [summary['layers'].get(str(l), {}).get('intrinsic_K_gt_V', 0) for l in TARGET_LAYERS]

    colors_intrinsic = ['#2ecc71' if g > 0 else '#e74c3c' for g in intrinsic_gaps]
    colors_rope = ['#f39c12' if r > 0 else '#9b59b6' for r in rope_contribs]

    ax.bar(x, intrinsic_gaps, width=0.35, label='Intrinsic K>V (K_pre − V)', color='#2ecc71', alpha=0.8)
    ax.bar(x, rope_contribs, width=0.35, bottom=intrinsic_gaps, label='RoPE contribution (K_post − K_pre)', color='#f39c12', alpha=0.8)

    # Add V baseline as horizontal markers
    for i, v in enumerate(v_means):
        ax.plot([i-0.2, i+0.2], [0, 0], 'k-', linewidth=1)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('|nums contribution', fontsize=12)
    ax.set_title('Decomposition of K>V Gap: Intrinsic Content vs RoPE Position\nQwen3-4B-Base', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in TARGET_LAYERS])
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "rope_vs_intrinsic_decomposition.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved rope_vs_intrinsic_decomposition.png")

    # ── Figure 3: Position sweep comparison at 2 key layers ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    example_layers = [TARGET_LAYERS[1], TARGET_LAYERS[-1]]  # L3 (ramp) and L30 (plateau)

    for ax, layer in zip(axes, example_layers):
        sl = str(layer)
        if sl not in results['K_pre']:
            continue

        bins = list(range(N_BINS))
        kpre_r = [results['K_pre'][sl].get(str(b), {}).get('resid_R', float('nan')) for b in bins]
        kpost_r = [results['K_post'][sl].get(str(b), {}).get('resid_R', float('nan')) for b in bins]
        v_r = [results['V'][sl].get(str(b), {}).get('resid_R', float('nan')) for b in bins]
        bin_centers = [(b + 0.5) / N_BINS * 100 for b in bins]

        ax.plot(bin_centers, kpre_r, 'r-o', label='K_pre (no RoPE)', markersize=4, linewidth=1.5)
        ax.plot(bin_centers, kpost_r, 'g-s', label='K_post (w/ RoPE)', markersize=4, linewidth=1.5)
        ax.plot(bin_centers, v_r, 'b-^', label='V', markersize=4, linewidth=1.5)

        ax.set_xlabel('Position in chain (%)', fontsize=11)
        ax.set_ylabel('|nums (forward-looking signal)', fontsize=11)
        ax.set_title(f'Layer {layer} ({100*layer/(n_layers-1):.0f}% depth)', fontsize=12)
        ax.legend(fontsize=10)
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(alpha=0.3)

    fig.suptitle('Position-wise RoPE Ablation: Ramp vs Plateau Layer', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "position_sweep_rope_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved position_sweep_rope_comparison.png")

    # ── Figure 4: K_pre vs K_post scatter (per-bin values) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    all_kpre_bins = []
    all_kpost_bins = []
    all_v_bins = []
    all_layer_labels = []

    for layer in TARGET_LAYERS:
        sl = str(layer)
        if sl not in results['K_pre']:
            continue
        for b in range(N_BINS):
            sb = str(b)
            if sb in results['K_pre'][sl]:
                kp = results['K_pre'][sl][sb]['resid_R']
                kpo = results['K_post'][sl][sb]['resid_R']
                vv = results['V'][sl][sb]['resid_R']
                if not (np.isnan(kp) or np.isnan(kpo) or np.isnan(vv)):
                    all_kpre_bins.append(kp)
                    all_kpost_bins.append(kpo)
                    all_v_bins.append(vv)
                    all_layer_labels.append(layer)

    all_kpre_bins = np.array(all_kpre_bins)
    all_kpost_bins = np.array(all_kpost_bins)
    all_v_bins = np.array(all_v_bins)
    all_layer_labels = np.array(all_layer_labels)

    # Scatter: K_pre vs K_post
    ax = axes[0]
    scatter = ax.scatter(all_kpost_bins, all_kpre_bins, c=all_layer_labels,
                         cmap='viridis', alpha=0.6, s=20)
    lims = [min(all_kpre_bins.min(), all_kpost_bins.min()) - 0.05,
            max(all_kpre_bins.max(), all_kpost_bins.max()) + 0.05]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='y=x')
    ax.set_xlabel('K_post|nums (with RoPE)', fontsize=11)
    ax.set_ylabel('K_pre|nums (no RoPE)', fontsize=11)
    ax.set_title('K_pre vs K_post (each dot = 1 bin)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Layer')

    # Scatter: K_pre vs V
    ax = axes[1]
    scatter = ax.scatter(all_v_bins, all_kpre_bins, c=all_layer_labels,
                         cmap='viridis', alpha=0.6, s=20)
    lims = [min(all_kpre_bins.min(), all_v_bins.min()) - 0.05,
            max(all_kpre_bins.max(), all_v_bins.max()) + 0.05]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='y=x')
    ax.set_xlabel('V|nums', fontsize=11)
    ax.set_ylabel('K_pre|nums (no RoPE)', fontsize=11)
    ax.set_title('K_pre vs V (each dot = 1 bin)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Layer')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "kpre_scatter_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved kpre_scatter_comparison.png")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\nDone!")


if __name__ == "__main__":
    main()
