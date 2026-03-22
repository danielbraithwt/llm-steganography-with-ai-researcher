#!/usr/bin/env python3
"""
Experiment 082: Cross-Model Forward-Looking Probing — Phi-3.5-mini-Instruct (MHA)

Replicates exp_079's residualization hierarchy on Phi-3.5-mini-Instruct using
ONLY GroupKFold (no standard KFold — known to leak within-problem data).

Core question: Does the V forward-looking signal (V→final|nums R=0.24 on Qwen)
replicate on a THIRD model family with MHA (not GQA) architecture?

Key differences from Qwen3-4B-Base and Mistral-7B-v0.3:
- Phi-3.5-mini uses MHA (32 KV heads, not GQA 8)
- Different family (Microsoft vs Alibaba/Mistral)
- Instruction-tuned (not base)
- Smaller model (3.8B) but specialized for reasoning
- kv_dim = 3072 (32 heads × 96 head_dim) — 3x larger than GQA models

Method:
1. Generate plain-text CoT for 200 GSM8K problems on Phi-3.5-mini-Instruct
2. Parse arithmetic operations, filter to correct problems
3. Forward pass: extract K and V at "=" positions at L8, L16, L24, L31
4. Extract problem numbers (log-transformed) and problem embeddings
5. GroupKFold probes: full residualization hierarchy (same as exp_079/081)
6. K probes alongside V (compare K vs V for forward-looking)
7. Bootstrap significance for key GroupKFold results
8. WRRA analysis if enough errors found
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
MODEL_NAME = 'microsoft/Phi-3.5-mini-instruct'
N_PROBLEMS = 200
N_LAYERS_TOTAL = 32
PROBE_LAYERS = [8, 16, 24, 31]  # 25%, 50%, 75%, 97%

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_082"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same as exp_078/079/081) ──
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
    m = re.search(r'[Tt]he answer is\s*\$?(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '')
    return None


def parse_arithmetic(text):
    """Find arithmetic expressions in plain text. Returns list of dicts."""
    num = r'[\d][\d,]*(?:\.\d+)?'
    op = r'\s*[+\-*/]\s*'
    expr = f'({num}(?:{op}{num})+)'
    result = r'=\s*\$?\s*(' + num + r')'
    pattern = f'{expr}\\s*{result}'

    results = []
    for m in re.finditer(pattern, text):
        expr_str = m.group(1).replace(',', '').strip()
        written_str = m.group(2).replace(',', '').strip()
        try:
            written_result = float(written_str)
            sanitized = re.sub(r'[^\d+\-*/. ]', '', expr_str)
            if not sanitized.strip():
                continue
            correct_result = float(eval(sanitized))
            if abs(correct_result) > 100:
                is_error = abs(written_result - correct_result) / max(abs(correct_result), 1) > 0.005
            else:
                is_error = abs(written_result - correct_result) > 0.5

            eq_pos_in_match = m.group(0).index('=')
            eq_char_pos = m.start() + eq_pos_in_match

            results.append({
                'expr_str': expr_str,
                'written_result': written_result,
                'correct_result': correct_result,
                'is_error': is_error,
                'eq_char_pos': eq_char_pos,
            })
        except Exception:
            continue
    return results


def extract_problem_numbers(question_text):
    """Extract all numeric values from a GSM8K question → fixed-length log-transformed vector."""
    MAX_NUMS = 20
    nums = re.findall(r'(?<![.\d])(\d[\d,]*(?:\.\d+)?)(?!\d)', question_text)
    values = []
    for n in nums:
        try:
            v = float(n.replace(',', ''))
            values.append(v)
        except Exception:
            continue
    features = np.zeros(MAX_NUMS, dtype=np.float32)
    for i, v in enumerate(values[:MAX_NUMS]):
        features[i] = np.sign(v) * np.log(abs(v) + 1)
    return features, values


def load_gsm8k():
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
    """Extract K, V tensors from past_key_values at a given layer."""
    # transformers 5.x DynamicCache with .layers attribute
    if hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
        return past_kv.layers[layer_idx].keys, past_kv.layers[layer_idx].values
    # transformers 4.x DynamicCache with key_cache/value_cache
    if hasattr(past_kv, 'key_cache'):
        return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
    # Tuple-based cache
    return past_kv[layer_idx][0], past_kv[layer_idx][1]


def signed_log(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def map_eq_to_token(gen_text, gen_ids, tokenizer, eq_char_pos):
    """Map character-level '=' position to token index in gen_ids."""
    eq_idx = gen_text[:eq_char_pos + 1].count('=') - 1
    if eq_idx < 0:
        return None
    count = 0
    for i, tid in enumerate(gen_ids):
        tok = tokenizer.decode([tid])
        for ch in tok:
            if ch == '=':
                if count == eq_idx:
                    return i
                count += 1
    return None


# ── GroupKFold probing functions ──

def run_probe_group_cv(X, y, groups, n_splits=5):
    """Ridge probe with GroupKFold — no same-problem leakage."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        return 0.0, np.zeros_like(y, dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    alphas = np.logspace(-2, 6, 50)
    gkf = GroupKFold(n_splits=n_splits)

    y_pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in gkf.split(X_scaled, y, groups):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(X_scaled[train_idx], y[train_idx])
        y_pred[test_idx] = ridge.predict(X_scaled[test_idx])

    r, _ = stats.pearsonr(y, y_pred)
    return float(r), y_pred


def partial_r_group(X, y, X_confound, groups, n_splits=5):
    """Partial correlation with GroupKFold."""
    _, y_confound_pred = run_probe_group_cv(X_confound, y, groups, n_splits)
    y_resid = y - y_confound_pred
    if np.std(y_resid) < 1e-10:
        return 0.0, np.zeros_like(y, dtype=float)
    r, preds = run_probe_group_cv(X, y_resid, groups, n_splits)
    return float(r), preds


def bootstrap_r_significance(y, y_pred, n_boot=2000, seed=42):
    """Bootstrap test: is R significantly > 0? Returns mean_R, CI, p-value."""
    from scipy import stats
    rng = np.random.RandomState(seed)
    n = len(y)
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if np.std(y[idx]) < 1e-10 or np.std(y_pred[idx]) < 1e-10:
            continue
        r, _ = stats.pearsonr(y[idx], y_pred[idx])
        boot_rs.append(r)
    boot_rs = np.array(boot_rs)
    return float(np.mean(boot_rs)), (float(np.percentile(boot_rs, 2.5)), float(np.percentile(boot_rs, 97.5))), float(np.mean(boot_rs <= 0))


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
    )
    model.eval()
    embed_fn = model.get_input_embeddings()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = getattr(model.config, 'head_dim', hidden_size // model.config.num_attention_heads)
    kv_dim = num_kv_heads * head_dim

    print(f"Model loaded in {time.time()-t0:.1f}s")
    print(f"  {n_layers} layers, hidden_size={hidden_size}, kv_dim={kv_dim}")
    print(f"  num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    num_attn_heads = model.config.num_attention_heads
    arch_type = 'MHA' if num_kv_heads == num_attn_heads else 'GQA'
    print(f"  Architecture: {arch_type}")

    # Validate probe layers
    valid_probe_layers = [l for l in PROBE_LAYERS if l < n_layers]
    if valid_probe_layers != PROBE_LAYERS:
        print(f"  WARNING: Adjusted probe layers from {PROBE_LAYERS} to {valid_probe_layers}")
    PROBE_LAYERS_ACTUAL = valid_probe_layers

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
    gen_budget = TIME_BUDGET * 0.40  # 40% for generation

    for i, prob in enumerate(problems):
        if time.time() - t0 > gen_budget:
            print(f"  Generation time budget reached at problem {i}")
            break

        prompt = build_prompt(prob['question'])
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

        if input_ids.shape[1] > MAX_SEQ_LEN - MAX_GEN:
            continue

        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_GEN,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        gen_ids = output[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred_ans_str = extract_answer(gen_text)
        pred_ans = None
        if pred_ans_str:
            try:
                pred_ans = float(pred_ans_str)
            except Exception:
                pass

        gold = prob['gold_answer']
        correct = pred_ans is not None and gold is not None and abs(pred_ans - gold) < 0.5

        if correct:
            n_correct += 1

        # Parse arithmetic
        ops = parse_arithmetic(gen_text)

        # Extract problem numbers
        prob_nums_feat, prob_nums_raw = extract_problem_numbers(prob['question'])

        gen_data.append({
            'question': prob['question'],
            'gen_text': gen_text,
            'gen_ids': gen_ids.cpu().tolist(),
            'prompt_len': input_ids.shape[1],
            'gold_answer': gold,
            'pred_answer': pred_ans,
            'correct': correct,
            'operations': ops,
            'prob_nums_feat': prob_nums_feat,
            'prob_nums_raw': prob_nums_raw,
        })

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(problems)}] correct: {n_correct}/{i+1} ({100*n_correct/(i+1):.1f}%)")

        del output
        torch.cuda.empty_cache()

    n_generated = len(gen_data)
    print(f"\n  Total generated: {n_generated}")
    print(f"  Correct: {n_correct} ({100*n_correct/max(n_generated,1):.1f}%)")

    # Filter to correct problems with operations
    valid = [d for d in gen_data if d['correct'] and len(d['operations']) > 0]
    print(f"  Valid (correct + has operations): {len(valid)}")

    if len(valid) < 20:
        print("ERROR: Too few valid problems for probing. Aborting.")
        with open(RESULTS_DIR / 'results.json', 'w') as f:
            json.dump({'error': 'too_few_valid', 'n_correct': n_correct, 'n_generated': n_generated}, f, indent=2)
        return

    total_ops = sum(len(d['operations']) for d in valid)
    correct_ops = sum(sum(1 for o in d['operations'] if not o['is_error']) for d in valid)
    error_ops = total_ops - correct_ops
    print(f"  Total operations: {total_ops}, correct: {correct_ops}, errors: {error_ops} ({100*error_ops/max(total_ops,1):.2f}%)")

    # Count WRRA cases
    wrra_count = 0
    for d in gen_data:
        if d['correct']:
            for op in d['operations']:
                if op['is_error']:
                    wrra_count += 1
    print(f"  WRRA cases (correct problem + wrong arithmetic): {wrra_count}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Forward pass + KV extraction + problem embeddings
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: Forward pass + KV extraction + problem embeddings")
    print(f"{'='*60}")

    # Per-operation data keyed by layer
    probe_data = {layer: {
        'K': [], 'V': [], 'prob_nums': [], 'prob_embed': [], 'local_result': [],
        'final_answer': [], 'problem_idx': [], 'is_error': [], 'written_result': [],
        'correct_result': [],
    } for layer in PROBE_LAYERS_ACTUAL}

    extract_budget = TIME_BUDGET * 0.70
    n_extracted = 0

    for idx, d in enumerate(valid):
        if time.time() - t0 > extract_budget:
            print(f"  Extraction time budget reached at problem {idx}")
            break

        gen_ids_t = torch.tensor(d['gen_ids'], device=model.device).unsqueeze(0)
        prompt_ids = tokenizer.encode(build_prompt(d['question']), return_tensors='pt').to(model.device)
        full_ids = torch.cat([prompt_ids, gen_ids_t], dim=1)

        # Forward pass with KV cache
        full_mask = torch.ones_like(full_ids)
        with torch.no_grad():
            out = model(full_ids, attention_mask=full_mask, use_cache=True)

        prompt_len = prompt_ids.shape[1]

        # Problem embedding: mean of prompt token embeddings
        with torch.no_grad():
            prompt_embeds = embed_fn(prompt_ids[0]).cpu().float().numpy()
        prob_embed = prompt_embeds.mean(axis=0)

        gen_text = d['gen_text']
        gen_ids_list = d['gen_ids']

        for op in d['operations']:
            eq_char_pos = op['eq_char_pos']
            tok_pos = map_eq_to_token(gen_text, gen_ids_list, tokenizer, eq_char_pos)
            if tok_pos is None:
                continue

            abs_pos = prompt_len + tok_pos
            if abs_pos >= full_ids.shape[1]:
                continue

            for layer in PROBE_LAYERS_ACTUAL:
                k_cache, v_cache = get_kv(out.past_key_values, layer)
                k_vec = k_cache[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()
                v_vec = v_cache[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()

                probe_data[layer]['K'].append(k_vec)
                probe_data[layer]['V'].append(v_vec)
                probe_data[layer]['prob_nums'].append(d['prob_nums_feat'])
                probe_data[layer]['prob_embed'].append(prob_embed)
                probe_data[layer]['local_result'].append(signed_log(op['correct_result']))
                probe_data[layer]['final_answer'].append(signed_log(d['gold_answer']))
                probe_data[layer]['problem_idx'].append(idx)
                probe_data[layer]['is_error'].append(op['is_error'])
                probe_data[layer]['written_result'].append(signed_log(op['written_result']))
                probe_data[layer]['correct_result'].append(signed_log(op['correct_result']))

        n_extracted += 1
        del out
        torch.cuda.empty_cache()

        if (idx + 1) % 25 == 0:
            n_ops = len(probe_data[PROBE_LAYERS_ACTUAL[0]]['V'])
            print(f"  [{idx+1}/{len(valid)}] extracted {n_ops} ops from {n_extracted} problems")

    # Free model memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n  Extraction complete. {n_extracted} problems processed.")
    for layer in PROBE_LAYERS_ACTUAL:
        print(f"  Layer {layer}: {len(probe_data[layer]['V'])} operation positions")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: GroupKFold Probes (the ONLY valid methodology)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 3: GroupKFold probes (no within-problem leakage)")
    print(f"{'='*60}")

    results = {}

    for layer in PROBE_LAYERS_ACTUAL:
        pd_layer = probe_data[layer]
        # Filter to correct operations for probing (errors go to WRRA)
        correct_mask = np.array([not e for e in pd_layer['is_error']])
        n_correct_ops = int(correct_mask.sum())

        if n_correct_ops < 30:
            print(f"  Layer {layer}: only {n_correct_ops} correct ops, skipping")
            continue

        V = np.array(pd_layer['V'])[correct_mask]
        K = np.array(pd_layer['K'])[correct_mask]
        prob_nums = np.array(pd_layer['prob_nums'])[correct_mask]
        prob_embed = np.array(pd_layer['prob_embed'])[correct_mask]
        local = np.array(pd_layer['local_result'])[correct_mask].reshape(-1, 1)
        y_final = np.array(pd_layer['final_answer'])[correct_mask]
        groups = np.array(pd_layer['problem_idx'])[correct_mask]

        n_problems = len(np.unique(groups))
        pct = 100 * layer / N_LAYERS_TOTAL
        print(f"\n  Layer L{layer} ({pct:.0f}% depth), n={n_correct_ops} ops from {n_problems} problems")
        print(f"    V dim={V.shape[1]}, K dim={K.shape[1]}, prob_embed dim={prob_embed.shape[1]}")

        lr = {}

        # ── Core probes ──
        r, pred_v_final = run_probe_group_cv(V, y_final, groups)
        lr['V_final'] = r
        print(f"    V → final:           R = {r:.3f}")

        r, pred_k_final = run_probe_group_cv(K, y_final, groups)
        lr['K_final'] = r
        print(f"    K → final:           R = {r:.3f}")

        r, _ = run_probe_group_cv(prob_nums, y_final, groups)
        lr['nums_final'] = r
        print(f"    prob_nums → final:   R = {r:.3f}")

        r, _ = run_probe_group_cv(prob_embed, y_final, groups)
        lr['embed_final'] = r
        print(f"    prob_embed → final:  R = {r:.3f}")

        r, _ = run_probe_group_cv(local, y_final, groups)
        lr['local_final'] = r
        print(f"    local → final:       R = {r:.3f}")

        # ── Partial correlations: V ──
        r, pred_v_partial_nums = partial_r_group(V, y_final, prob_nums, groups)
        lr['V_partial_nums'] = r
        print(f"    V → final | nums:    R = {r:.3f}  ← KEY CHALLENGE")

        r, _ = partial_r_group(V, y_final, prob_embed, groups)
        lr['V_partial_embed'] = r
        print(f"    V → final | embed:   R = {r:.3f}  ← STRICTEST")

        r, _ = partial_r_group(V, y_final, local, groups)
        lr['V_partial_local'] = r
        print(f"    V → final | local:   R = {r:.3f}")

        nums_local = np.hstack([prob_nums, local])
        r, _ = partial_r_group(V, y_final, nums_local, groups)
        lr['V_partial_nums_local'] = r
        print(f"    V → final | n+l:     R = {r:.3f}")

        embed_local = np.hstack([prob_embed, local])
        r, _ = partial_r_group(V, y_final, embed_local, groups)
        lr['V_partial_embed_local'] = r
        print(f"    V → final | e+l:     R = {r:.3f}  ← STRICTEST COMBINED")

        # ── Partial correlations: K ──
        r, _ = partial_r_group(K, y_final, prob_nums, groups)
        lr['K_partial_nums'] = r
        print(f"    K → final | nums:    R = {r:.3f}")

        r, _ = partial_r_group(K, y_final, prob_embed, groups)
        lr['K_partial_embed'] = r
        print(f"    K → final | embed:   R = {r:.3f}")

        r, _ = partial_r_group(K, y_final, local, groups)
        lr['K_partial_local'] = r
        print(f"    K → final | local:   R = {r:.3f}")

        r, _ = partial_r_group(K, y_final, embed_local, groups)
        lr['K_partial_embed_local'] = r
        print(f"    K → final | e+l:     R = {r:.3f}")

        # ── Shuffle control ──
        y_shuffle = y_final.copy()
        np.random.seed(SEED + layer + 200)
        np.random.shuffle(y_shuffle)
        r, _ = run_probe_group_cv(V, y_shuffle, groups)
        lr['V_shuffle'] = r
        print(f"    V → shuffle:         R = {r:.3f}")

        # ── Bootstrap significance for key results ──
        print(f"\n    Bootstrap significance (2000 samples)...")

        # V → final significance
        mean_r, ci, p = bootstrap_r_significance(y_final, pred_v_final, n_boot=2000, seed=SEED)
        lr['bootstrap_V_final'] = {'mean_r': mean_r, 'ci': list(ci), 'p': p}
        print(f"    V→final: mean_R={mean_r:.3f}, CI=[{ci[0]:.3f},{ci[1]:.3f}], p={p:.4f}")

        # V → final | nums significance (bootstrap on partial residuals)
        _, y_nums_pred = run_probe_group_cv(prob_nums, y_final, groups)
        y_resid_nums = y_final - y_nums_pred
        if np.std(y_resid_nums) > 1e-10:
            _, pred_v_resid = run_probe_group_cv(V, y_resid_nums, groups)
            mean_r, ci, p = bootstrap_r_significance(y_resid_nums, pred_v_resid, n_boot=2000, seed=SEED+1)
            lr['bootstrap_V_partial_nums'] = {'mean_r': mean_r, 'ci': list(ci), 'p': p}
            print(f"    V|nums: mean_R={mean_r:.3f}, CI=[{ci[0]:.3f},{ci[1]:.3f}], p={p:.4f}")

        # V → final | embed significance
        _, y_embed_pred = run_probe_group_cv(prob_embed, y_final, groups)
        y_resid_embed = y_final - y_embed_pred
        if np.std(y_resid_embed) > 1e-10:
            _, pred_v_resid_e = run_probe_group_cv(V, y_resid_embed, groups)
            mean_r, ci, p = bootstrap_r_significance(y_resid_embed, pred_v_resid_e, n_boot=2000, seed=SEED+2)
            lr['bootstrap_V_partial_embed'] = {'mean_r': mean_r, 'ci': list(ci), 'p': p}
            print(f"    V|embed: mean_R={mean_r:.3f}, CI=[{ci[0]:.3f},{ci[1]:.3f}], p={p:.4f}")

        results[f'L{layer}'] = lr

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3.5: WRRA Analysis (if enough errors)
    # ═══════════════════════════════════════════════════════════════
    wrra_results = {}
    # Use L24 (75% depth — equivalent to Qwen L27 where WRRA peaked)
    wrra_layer = 24 if 24 in PROBE_LAYERS_ACTUAL else PROBE_LAYERS_ACTUAL[-1]

    pd_wrra = probe_data[wrra_layer]
    error_mask = np.array(pd_wrra['is_error'])
    correct_mask_wrra = ~error_mask
    n_errors = int(error_mask.sum())

    print(f"\n{'='*60}")
    print(f"PHASE 3.5: WRRA Analysis at L{wrra_layer} (n_errors={n_errors})")
    print(f"{'='*60}")

    if n_errors >= 5:
        # Train probe on correct operations: V → local_result
        V_correct = np.array(pd_wrra['V'])[correct_mask_wrra]
        y_local_correct = np.array(pd_wrra['local_result'])[correct_mask_wrra]

        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        V_correct_scaled = scaler.fit_transform(V_correct)
        ridge = RidgeCV(alphas=np.logspace(-2, 6, 50))
        ridge.fit(V_correct_scaled, y_local_correct)

        # Apply to error operations
        V_error = np.array(pd_wrra['V'])[error_mask]
        V_error_scaled = scaler.transform(V_error)
        predicted_at_errors = ridge.predict(V_error_scaled)

        written_at_errors = np.array(pd_wrra['written_result'])[error_mask]
        correct_at_errors = np.array(pd_wrra['correct_result'])[error_mask]

        # Alignment: does probe predict closer to correct or written value?
        dist_to_correct = np.abs(predicted_at_errors - correct_at_errors)
        dist_to_written = np.abs(predicted_at_errors - written_at_errors)
        aligned_correct = dist_to_correct < dist_to_written

        n_aligned = int(aligned_correct.sum())
        alignment_rate = n_aligned / n_errors

        from scipy.stats import binomtest
        binom = binomtest(n_aligned, n_errors, 0.5, alternative='greater')

        wrra_results = {
            'layer': wrra_layer,
            'n_errors': n_errors,
            'n_aligned_correct': n_aligned,
            'alignment_rate': alignment_rate,
            'p_value': binom.pvalue,
            'significant': binom.pvalue < 0.05,
        }

        print(f"  WRRA alignment: {n_aligned}/{n_errors} = {alignment_rate:.3f}")
        print(f"  Binomial p = {binom.pvalue:.4f} (H0: chance = 0.5)")
        print(f"  Significant: {binom.pvalue < 0.05}")

        # Also try all probe layers
        for layer in PROBE_LAYERS_ACTUAL:
            pd_l = probe_data[layer]
            em = np.array(pd_l['is_error'])
            cm = ~em
            if em.sum() < 5:
                continue

            V_c = np.array(pd_l['V'])[cm]
            y_c = np.array(pd_l['local_result'])[cm]
            sc = StandardScaler()
            V_c_s = sc.fit_transform(V_c)
            r = RidgeCV(alphas=np.logspace(-2, 6, 50))
            r.fit(V_c_s, y_c)

            V_e = np.array(pd_l['V'])[em]
            V_e_s = sc.transform(V_e)
            pred = r.predict(V_e_s)

            wr = np.array(pd_l['written_result'])[em]
            cr = np.array(pd_l['correct_result'])[em]
            dc = np.abs(pred - cr)
            dw = np.abs(pred - wr)
            al = dc < dw
            na = int(al.sum())
            bt = binomtest(na, int(em.sum()), 0.5, alternative='greater')
            print(f"    L{layer}: {na}/{int(em.sum())} = {na/int(em.sum()):.3f} (p={bt.pvalue:.4f})")
    else:
        print(f"  Only {n_errors} errors — insufficient for WRRA analysis")
        wrra_results = {'n_errors': n_errors, 'message': 'insufficient errors'}

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Figures
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 4: Generating figures")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: V and K forward-looking probes by layer (GroupKFold)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    layers_with_results = sorted([l for l in PROBE_LAYERS_ACTUAL if f'L{l}' in results])
    layer_labels = [f'L{l} ({100*l//N_LAYERS_TOTAL}%)' for l in layers_with_results]
    x = np.arange(len(layers_with_results))

    if len(layers_with_results) > 0:
        # Panel 1: Raw probes
        v_finals = [results[f'L{l}']['V_final'] for l in layers_with_results]
        k_finals = [results[f'L{l}']['K_final'] for l in layers_with_results]
        nums_finals = [results[f'L{l}']['nums_final'] for l in layers_with_results]
        embed_finals = [results[f'L{l}']['embed_final'] for l in layers_with_results]
        local_finals = [results[f'L{l}']['local_final'] for l in layers_with_results]
        shuffles = [results[f'L{l}']['V_shuffle'] for l in layers_with_results]

        ax1.plot(x, v_finals, 'o-', label='V→final', linewidth=2, markersize=8, color='blue')
        ax1.plot(x, k_finals, 's-', label='K→final', linewidth=2, markersize=8, color='red')
        ax1.plot(x, [nums_finals[0]] * len(x), '--', label='nums→final', color='orange', alpha=0.7)
        ax1.plot(x, [embed_finals[0]] * len(x), '--', label='embed→final', color='green', alpha=0.7)
        ax1.plot(x, [local_finals[0]] * len(x), '--', label='local→final', color='purple', alpha=0.7)
        ax1.plot(x, shuffles, 'x-', label='V→shuffle', color='gray', alpha=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_labels)
        ax1.set_ylabel('Pearson R (GroupKFold)')
        ax1.set_title('Raw Forward-Looking Probes')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5)

        # Panel 2: Partial correlations (residualized)
        v_partial_nums = [results[f'L{l}']['V_partial_nums'] for l in layers_with_results]
        v_partial_embed = [results[f'L{l}']['V_partial_embed'] for l in layers_with_results]
        v_partial_local = [results[f'L{l}']['V_partial_local'] for l in layers_with_results]
        v_partial_el = [results[f'L{l}']['V_partial_embed_local'] for l in layers_with_results]
        k_partial_embed = [results[f'L{l}']['K_partial_embed'] for l in layers_with_results]

        ax2.plot(x, v_partial_nums, 'o-', label='V|nums', linewidth=2, markersize=8, color='blue')
        ax2.plot(x, v_partial_embed, 's-', label='V|embed', linewidth=2, markersize=8, color='cyan')
        ax2.plot(x, v_partial_local, '^-', label='V|local', linewidth=2, markersize=8, color='green')
        ax2.plot(x, v_partial_el, 'D-', label='V|embed+local', linewidth=2, markersize=8, color='red')
        ax2.plot(x, k_partial_embed, 'v-', label='K|embed', linewidth=1.5, markersize=7, color='orange')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_labels)
        ax2.set_ylabel('Partial R (GroupKFold)')
        ax2.set_title('Residualized Forward-Looking Probes')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        ax2.axhline(y=0, color='k', linewidth=0.5)

    fig.suptitle(f'Exp 082: Phi-3.5-mini Forward-Looking Probing (GroupKFold, MHA)\nn={n_correct} correct problems, {n_extracted} extracted', fontsize=13)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'forward_looking_probes_phi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved forward_looking_probes_phi.png")

    # Figure 2: Three-model comparison bar chart (Qwen vs Mistral vs Phi)
    fig, ax = plt.subplots(figsize=(14, 6))

    probe_names = ['V→final', 'K→final', 'V|nums', 'V|embed', 'V|local', 'V|e+l', 'shuffle']
    # Best layer for Phi (L24 = 75% depth, comparable across models)
    best_layer = 24 if 'L24' in results else (layers_with_results[-1] if layers_with_results else 24)
    bl_key = f'L{best_layer}'
    if bl_key in results:
        phi_vals = [
            results[bl_key].get('V_final', 0),
            results[bl_key].get('K_final', 0),
            results[bl_key].get('V_partial_nums', 0),
            results[bl_key].get('V_partial_embed', 0),
            results[bl_key].get('V_partial_local', 0),
            results[bl_key].get('V_partial_embed_local', 0),
            results[bl_key].get('V_shuffle', 0),
        ]
    else:
        phi_vals = [0] * len(probe_names)

    # Qwen values from exp_079 GroupKFold (L27 = 77% depth)
    qwen_vals = [0.487, 0, 0.242, 0.221, 0.299, 0.215, -0.017]
    # Mistral values from exp_081 (L24 = 75% depth)
    mistral_vals = [0.231, 0.195, 0.057, -0.010, 0.098, -0.010, -0.083]

    x = np.arange(len(probe_names))
    width = 0.25
    ax.bar(x - width, qwen_vals, width, label='Qwen3-4B-Base (L27)', alpha=0.8, color='steelblue')
    ax.bar(x, mistral_vals, width, label='Mistral-7B (L24)', alpha=0.8, color='coral')
    ax.bar(x + width, phi_vals, width, label=f'Phi-3.5-mini (L{best_layer})', alpha=0.8, color='forestgreen')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, rotation=30, ha='right')
    ax.set_ylabel('Pearson R (GroupKFold)')
    ax.set_title('Three-Model Comparison: Forward-Looking V Probes')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)

    for i, (q, m, p) in enumerate(zip(qwen_vals, mistral_vals, phi_vals)):
        ax.text(i - width, q + 0.01, f'{q:.2f}', ha='center', va='bottom', fontsize=7, color='steelblue')
        ax.text(i, m + 0.01, f'{m:.2f}', ha='center', va='bottom', fontsize=7, color='coral')
        ax.text(i + width, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=7, color='forestgreen')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'three_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved three_model_comparison.png")

    # Figure 3: Residualization waterfall (Phi only)
    if layers_with_results:
        fig, axes = plt.subplots(1, len(layers_with_results), figsize=(5 * len(layers_with_results), 5))
        if len(layers_with_results) == 1:
            axes = [axes]

        for ax, layer in zip(axes, layers_with_results):
            r = results[f'L{layer}']
            stages = ['V→final', '- nums', '- local', '- embed+local']
            values = [
                r['V_final'],
                r['V_partial_nums'],
                r['V_partial_local'],
                r['V_partial_embed_local'],
            ]
            colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

            bars = ax.bar(range(len(stages)), values, color=colors, alpha=0.8, edgecolor='black')
            ax.set_xticks(range(len(stages)))
            ax.set_xticklabels(stages, rotation=30, ha='right')
            ax.set_ylabel('Pearson R')
            ax.set_title(f'L{layer} ({100*layer//N_LAYERS_TOTAL}%)')
            ax.axhline(y=0, color='k', linewidth=0.5)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, max(val + 0.01, 0.01),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        fig.suptitle('Phi-3.5-mini: How Much V→Final Signal Survives Each Control?', fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'waterfall_controls_phi.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved waterfall_controls_phi.png")

    # Figure 4: Accuracy vs nums→final across 3 models (tests accuracy confound)
    fig, ax = plt.subplots(figsize=(8, 6))
    model_data = [
        ('Qwen3-4B-Base', 0.87, 0.153, 0.242, 'steelblue'),
        ('Mistral-7B', 0.43, 0.390, 0.057, 'coral'),
        ('Phi-3.5-mini', n_correct/max(n_generated,1),
         results.get(bl_key, {}).get('nums_final', 0),
         results.get(bl_key, {}).get('V_partial_nums', 0),
         'forestgreen'),
    ]
    for name, acc, nums_r, v_nums_r, color in model_data:
        ax.scatter(acc, v_nums_r, s=200, c=color, label=f'{name} (V|nums={v_nums_r:.3f})', zorder=3)
        ax.annotate(f'nums→final={nums_r:.3f}', (acc, v_nums_r),
                   textcoords="offset points", xytext=(10, 10), fontsize=9)

    ax.set_xlabel('Model Accuracy on GSM8K')
    ax.set_ylabel('V → final | nums (Partial R)')
    ax.set_title('Accuracy Confound Test: Does Higher Accuracy → Stronger V|nums?')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'accuracy_confound_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved accuracy_confound_test.png")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Save Results
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 5: Saving results")
    print(f"{'='*60}")

    summary = {
        'model': MODEL_NAME,
        'architecture': arch_type,
        'n_problems_generated': n_generated,
        'n_correct': n_correct,
        'accuracy': n_correct / max(n_generated, 1),
        'n_valid_with_ops': len(valid),
        'n_extracted': n_extracted,
        'total_ops': total_ops,
        'correct_ops': correct_ops,
        'error_ops': error_ops,
        'error_rate': error_ops / max(total_ops, 1),
        'wrra_count': wrra_count,
        'kv_dim': kv_dim,
        'hidden_size': hidden_size,
        'n_layers': n_layers,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'probe_layers': PROBE_LAYERS_ACTUAL,
        'results_groupkfold': results,
        'wrra_results': wrra_results,
        'runtime_seconds': time.time() - t0,
    }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved results.json")

    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Architecture: MHA ({num_kv_heads} KV heads)")
    print(f"Accuracy: {n_correct}/{n_generated} ({100*n_correct/max(n_generated,1):.1f}%)")
    print(f"Valid problems: {len(valid)}, Extracted: {n_extracted}")
    print(f"Operations: {total_ops} total, {error_ops} errors ({100*error_ops/max(total_ops,1):.1f}%)")
    print(f"WRRA cases: {wrra_count}")
    print(f"\nGroupKFold probe results:")
    for layer in layers_with_results:
        r = results[f'L{layer}']
        print(f"  L{layer} ({100*layer//N_LAYERS_TOTAL}%):")
        print(f"    V→final={r['V_final']:.3f}  K→final={r['K_final']:.3f}")
        print(f"    V|nums={r['V_partial_nums']:.3f}  V|embed={r['V_partial_embed']:.3f}  V|e+l={r['V_partial_embed_local']:.3f}")
        print(f"    K|embed={r['K_partial_embed']:.3f}  shuffle={r['V_shuffle']:.3f}")
        if 'bootstrap_V_partial_nums' in r:
            b = r['bootstrap_V_partial_nums']
            print(f"    V|nums bootstrap: R={b['mean_r']:.3f}, CI=[{b['ci'][0]:.3f},{b['ci'][1]:.3f}], p={b['p']:.4f}")

    print(f"\nTotal runtime: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
