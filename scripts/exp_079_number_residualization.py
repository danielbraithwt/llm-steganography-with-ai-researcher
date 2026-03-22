#!/usr/bin/env python3
"""
Experiment 079: Problem-Number Residualization — Challenge Experiment

CHALLENGE: The last 3 experiments (076-078) all confirmed the hypothesis.
This experiment directly attacks the biggest confound in exp_078's strong
finding (V→final partial R=0.520): the problem-number confound.

Core question: Does V at computation positions predict the final answer
BEYOND what problem numbers alone predict? Or is V just an efficient
encoder of input numbers?

Method:
1. Generate plain-text CoT (same as exp_078)
2. Extract V at "=" positions at L27, L35
3. Extract problem numbers from each question (raw numeric values)
4. Compute "problem embedding": cumulative mean of token embeddings at CoT start
5. Residualization probes:
   a. V → final (replicate exp_078)
   b. prob_numbers → final (how predictive are raw numbers?)
   c. prob_embed → final (how predictive is the full problem context?)
   d. V → final | prob_numbers (partial: V beyond problem numbers)
   e. V → final | prob_embed (partial: V beyond problem context)
   f. V → final | local (replicate exp_078 partial)
   g. V → final | prob_numbers + local (combined control)
   h. V → final | prob_embed + local (strictest control)
   i. local → final | prob_numbers (does local computation add beyond numbers?)
   j. Shuffle controls
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
N_PROBLEMS = 400
PROBE_LAYERS = [27, 35]  # Focus on the two strongest layers from exp_078

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_079"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same as exp_078) ──
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
    result = f'=\\s*\\$?\\s*({num})'
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
    """Extract all numeric values from a GSM8K question.
    Returns a fixed-length feature vector of log-transformed numbers."""
    MAX_NUMS = 20
    # Find all numbers in the question (lookahead allows period-then-non-digit for sentence endings)
    nums = re.findall(r'(?<![.\d])(\d[\d,]*(?:\.\d+)?)(?!\d)', question_text)
    values = []
    for n in nums:
        try:
            v = float(n.replace(',', ''))
            values.append(v)
        except Exception:
            continue
    # Log-transform and zero-pad to fixed length
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


def run_probe_cv(X, y, n_splits=5):
    """Train ridge probe with cross-validation, return Pearson R and predictions."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    if X.shape[0] < max(n_splits, 10):
        return 0.0, np.zeros_like(y, dtype=float)

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
    return float(r), y_pred


def partial_r(X, y, X_confound, n_splits=5):
    """Partial correlation: R of X→y after regressing out X_confound from y.
    Returns: partial_R, predictions."""
    # Step 1: Get confound predictions of y
    _, y_confound_pred = run_probe_cv(X_confound, y, n_splits)
    # Step 2: Residualize y
    y_resid = y - y_confound_pred
    # Check variance
    if np.std(y_resid) < 1e-10:
        return 0.0, np.zeros_like(y, dtype=float)
    # Step 3: Probe X → residual
    r, preds = run_probe_cv(X, y_resid, n_splits)
    return float(r), preds


def map_eq_to_token(gen_text, gen_ids, tokenizer, eq_char_pos):
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


def bootstrap_significance(y, y_pred_a, y_pred_b, n_boot=2000, seed=42):
    """Bootstrap test for whether probe A has higher R than probe B.
    Returns: mean_diff, 95% CI, p-value (one-sided: A > B)."""
    from scipy import stats
    rng = np.random.RandomState(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        y_b = y[idx]
        if np.std(y_b) < 1e-10:
            continue
        r_a, _ = stats.pearsonr(y_b, y_pred_a[idx])
        r_b, _ = stats.pearsonr(y_b, y_pred_b[idx])
        diffs.append(r_a - r_b)
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci_lo = np.percentile(diffs, 2.5)
    ci_hi = np.percentile(diffs, 97.5)
    p_value = np.mean(diffs <= 0)  # P(A ≤ B)
    return mean_diff, (ci_lo, ci_hi), p_value


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
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
    gen_budget = TIME_BUDGET * 0.45

    for i, prob in enumerate(problems):
        if time.time() - t0 > gen_budget:
            print(f"  Generation time budget reached at problem {i}")
            break

        prompt = build_prompt(prob['question'])
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

        if input_ids.shape[1] > MAX_SEQ_LEN - MAX_GEN:
            continue

        with torch.no_grad():
            output = model.generate(
                input_ids,
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

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(problems)}] correct: {n_correct}/{i+1} ({100*n_correct/(i+1):.1f}%)")

        del output
        torch.cuda.empty_cache()

    print(f"\n  Total generated: {len(gen_data)}")
    print(f"  Correct: {n_correct} ({100*n_correct/len(gen_data):.1f}%)")

    # Filter to correct problems with operations
    valid = [d for d in gen_data if d['correct'] and len(d['operations']) > 0]
    print(f"  Valid (correct + has operations): {len(valid)}")

    total_ops = sum(len(d['operations']) for d in valid)
    correct_ops = sum(sum(1 for o in d['operations'] if not o['is_error']) for d in valid)
    error_ops = total_ops - correct_ops
    print(f"  Total operations: {total_ops}, errors: {error_ops} ({100*error_ops/total_ops:.2f}%)")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Forward pass + KV extraction + problem embeddings
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: Forward pass + KV extraction + problem embeddings")
    print(f"{'='*60}")

    # Collect data for probing
    # Per operation: V_features, problem_numbers, problem_embedding, local_result, final_answer
    probe_data = {layer: {
        'V': [], 'prob_nums': [], 'prob_embed': [], 'local_result': [],
        'final_answer': [], 'problem_idx': []
    } for layer in PROBE_LAYERS}

    extract_budget = TIME_BUDGET * 0.75
    n_extracted = 0

    for idx, d in enumerate(valid):
        if time.time() - t0 > extract_budget:
            print(f"  Extraction time budget reached at problem {idx}")
            break

        gen_ids_t = torch.tensor(d['gen_ids'], device=model.device).unsqueeze(0)
        prompt_ids = tokenizer.encode(build_prompt(d['question']), return_tensors='pt').to(model.device)
        full_ids = torch.cat([prompt_ids, gen_ids_t], dim=1)

        # Forward pass with KV cache
        with torch.no_grad():
            out = model(full_ids, use_cache=True)

        prompt_len = prompt_ids.shape[1]

        # Compute problem embedding: mean of token embeddings for the QUESTION part
        # (last question in the prompt, after "Q: ")
        # Use the full prompt embeddings for a strong confound baseline
        with torch.no_grad():
            prompt_embeds = embed_fn(prompt_ids[0]).cpu().float().numpy()  # [prompt_len, hidden_size]
        prob_embed = prompt_embeds.mean(axis=0)  # [hidden_size]

        # Map "=" positions and extract V
        gen_text = d['gen_text']
        gen_ids_list = d['gen_ids']

        for op in d['operations']:
            if op['is_error']:
                continue  # Only probe correct operations (as in exp_078)

            eq_char_pos = op['eq_char_pos']
            tok_pos = map_eq_to_token(gen_text, gen_ids_list, tokenizer, eq_char_pos)
            if tok_pos is None:
                continue

            abs_pos = prompt_len + tok_pos  # absolute position in full sequence

            if abs_pos >= full_ids.shape[1]:
                continue

            for layer in PROBE_LAYERS:
                _, v_cache = get_kv(out.past_key_values, layer)
                v_vec = v_cache[0, :, abs_pos, :].reshape(-1).cpu().float().numpy()  # [kv_dim]

                probe_data[layer]['V'].append(v_vec)
                probe_data[layer]['prob_nums'].append(d['prob_nums_feat'])
                probe_data[layer]['prob_embed'].append(prob_embed)
                probe_data[layer]['local_result'].append(signed_log(op['correct_result']))
                probe_data[layer]['final_answer'].append(signed_log(d['gold_answer']))
                probe_data[layer]['problem_idx'].append(idx)

        n_extracted += 1
        del out
        torch.cuda.empty_cache()

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(valid)}] extracted {sum(len(probe_data[l]['V']) for l in PROBE_LAYERS[:1])} ops")

    # Free model memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n  Extraction complete. {n_extracted} problems processed.")
    for layer in PROBE_LAYERS:
        print(f"  Layer {layer}: {len(probe_data[layer]['V'])} operation positions")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Residualization Probes
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 3: Residualization probes")
    print(f"{'='*60}")

    results = {}

    for layer in PROBE_LAYERS:
        pd = probe_data[layer]
        n = len(pd['V'])
        if n < 20:
            print(f"  Layer {layer}: only {n} samples, skipping")
            continue

        V = np.array(pd['V'])
        prob_nums = np.array(pd['prob_nums'])
        prob_embed = np.array(pd['prob_embed'])
        local = np.array(pd['local_result']).reshape(-1, 1)
        y_final = np.array(pd['final_answer'])

        print(f"\n  Layer L{layer} ({100*layer/36:.0f}% depth), n={n} operations")
        print(f"    V dim={V.shape[1]}, prob_nums dim={prob_nums.shape[1]}, prob_embed dim={prob_embed.shape[1]}")

        layer_results = {}

        # ── A: V → final (replicate exp_078) ──
        r_v_final, pred_v_final = run_probe_cv(V, y_final)
        layer_results['V_final'] = r_v_final
        print(f"    V → final:           R = {r_v_final:.3f}")

        # ── B: prob_numbers → final ──
        r_nums_final, pred_nums_final = run_probe_cv(prob_nums, y_final)
        layer_results['nums_final'] = r_nums_final
        print(f"    prob_nums → final:   R = {r_nums_final:.3f}")

        # ── C: prob_embed → final ──
        r_embed_final, pred_embed_final = run_probe_cv(prob_embed, y_final)
        layer_results['embed_final'] = r_embed_final
        print(f"    prob_embed → final:  R = {r_embed_final:.3f}")

        # ── D: local → final ──
        r_local_final, pred_local_final = run_probe_cv(local, y_final)
        layer_results['local_final'] = r_local_final
        print(f"    local → final:       R = {r_local_final:.3f}")

        # ── E: V → final | prob_numbers (KEY CHALLENGE TEST) ──
        r_v_partial_nums, pred_v_partial_nums = partial_r(V, y_final, prob_nums)
        layer_results['V_final_partial_nums'] = r_v_partial_nums
        print(f"    V → final | nums:    R = {r_v_partial_nums:.3f}  ← KEY CHALLENGE")

        # ── F: V → final | prob_embed (STRICTEST CONTEXT CONTROL) ──
        r_v_partial_embed, pred_v_partial_embed = partial_r(V, y_final, prob_embed)
        layer_results['V_final_partial_embed'] = r_v_partial_embed
        print(f"    V → final | embed:   R = {r_v_partial_embed:.3f}  ← STRICTEST")

        # ── G: V → final | local (replicate exp_078) ──
        r_v_partial_local, pred_v_partial_local = partial_r(V, y_final, local)
        layer_results['V_final_partial_local'] = r_v_partial_local
        print(f"    V → final | local:   R = {r_v_partial_local:.3f}  (replicate exp_078)")

        # ── H: V → final | nums + local (combined) ──
        nums_local = np.hstack([prob_nums, local])
        r_v_partial_nums_local, _ = partial_r(V, y_final, nums_local)
        layer_results['V_final_partial_nums_local'] = r_v_partial_nums_local
        print(f"    V → final | n+l:     R = {r_v_partial_nums_local:.3f}")

        # ── I: V → final | embed + local (strictest combined) ──
        embed_local = np.hstack([prob_embed, local])
        r_v_partial_embed_local, _ = partial_r(V, y_final, embed_local)
        layer_results['V_final_partial_embed_local'] = r_v_partial_embed_local
        print(f"    V → final | e+l:     R = {r_v_partial_embed_local:.3f}  ← STRICTEST COMBINED")

        # ── J: local → final | nums (does computation add beyond numbers?) ──
        r_local_partial_nums, _ = partial_r(local, y_final, prob_nums)
        layer_results['local_final_partial_nums'] = r_local_partial_nums
        print(f"    local → final | nums: R = {r_local_partial_nums:.3f}")

        # ── K: Shuffle control ──
        y_shuffle = y_final.copy()
        np.random.seed(SEED + layer)
        np.random.shuffle(y_shuffle)
        r_shuffle, _ = run_probe_cv(V, y_shuffle)
        layer_results['V_shuffle'] = r_shuffle
        print(f"    V → shuffle:         R = {r_shuffle:.3f}")

        # ── L: Bootstrap significance for V vs prob_embed ──
        print(f"\n    Bootstrap significance tests (2000 samples)...")
        # Test: V → final vs prob_embed → final
        _, pred_v_final_bs = run_probe_cv(V, y_final)
        _, pred_embed_final_bs = run_probe_cv(prob_embed, y_final)
        mean_diff_ve, ci_ve, p_ve = bootstrap_significance(y_final, pred_v_final_bs, pred_embed_final_bs)
        layer_results['bootstrap_V_vs_embed'] = {
            'mean_diff': mean_diff_ve, 'ci': ci_ve, 'p': p_ve
        }
        print(f"    V vs embed: diff={mean_diff_ve:.3f}, CI=[{ci_ve[0]:.3f},{ci_ve[1]:.3f}], p={p_ve:.3f}")

        # Test: is V → final | nums significantly > 0?
        # Use bootstrap on partial correlation residuals
        _, y_nums_pred = run_probe_cv(prob_nums, y_final)
        y_resid_nums = y_final - y_nums_pred
        if np.std(y_resid_nums) > 1e-10:
            _, pred_v_resid = run_probe_cv(V, y_resid_nums)
            # Bootstrap: is R > 0?
            from scipy import stats as sp_stats
            rng = np.random.RandomState(SEED)
            boot_rs = []
            for _ in range(2000):
                bidx = rng.choice(n, n, replace=True)
                if np.std(y_resid_nums[bidx]) < 1e-10:
                    continue
                br, _ = sp_stats.pearsonr(y_resid_nums[bidx], pred_v_resid[bidx])
                boot_rs.append(br)
            boot_rs = np.array(boot_rs)
            ci_partial = (np.percentile(boot_rs, 2.5), np.percentile(boot_rs, 97.5))
            p_partial = np.mean(boot_rs <= 0)
            layer_results['bootstrap_V_partial_nums'] = {
                'mean_r': np.mean(boot_rs), 'ci': ci_partial, 'p': p_partial
            }
            print(f"    V|nums bootstrap: mean_R={np.mean(boot_rs):.3f}, CI=[{ci_partial[0]:.3f},{ci_partial[1]:.3f}], p={p_partial:.4f}")

        results[f'L{layer}'] = layer_results

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3.5: Problem-Level Probing (fixes within-problem data leakage)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 3.5: Problem-level probing (GroupKFold — no within-problem leakage)")
    print(f"{'='*60}")

    from sklearn.model_selection import GroupKFold

    def run_probe_group_cv(X, y, groups, n_splits=5):
        """Ridge probe with GroupKFold — no same-problem leakage."""
        from sklearn.linear_model import RidgeCV
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

    results_group = {}

    for layer in PROBE_LAYERS:
        pd = probe_data[layer]
        n = len(pd['V'])
        if n < 20:
            continue

        V = np.array(pd['V'])
        prob_nums = np.array(pd['prob_nums'])
        prob_embed = np.array(pd['prob_embed'])
        local = np.array(pd['local_result']).reshape(-1, 1)
        y_final = np.array(pd['final_answer'])
        groups = np.array(pd['problem_idx'])

        n_problems = len(np.unique(groups))
        print(f"\n  Layer L{layer}, n={n} ops from {n_problems} problems (GroupKFold)")

        gl = {}

        # Core probes with GroupKFold
        r, _ = run_probe_group_cv(V, y_final, groups)
        gl['V_final'] = r
        print(f"    V → final:           R = {r:.3f}")

        r, _ = run_probe_group_cv(prob_nums, y_final, groups)
        gl['nums_final'] = r
        print(f"    prob_nums → final:   R = {r:.3f}")

        r, _ = run_probe_group_cv(prob_embed, y_final, groups)
        gl['embed_final'] = r
        print(f"    prob_embed → final:  R = {r:.3f}  ← was 0.977 with KFold (leakage?)")

        r, _ = run_probe_group_cv(local, y_final, groups)
        gl['local_final'] = r
        print(f"    local → final:       R = {r:.3f}")

        # Key partial correlations with GroupKFold
        r, _ = partial_r_group(V, y_final, prob_nums, groups)
        gl['V_final_partial_nums'] = r
        print(f"    V → final | nums:    R = {r:.3f}  ← KEY (GroupKFold)")

        r, _ = partial_r_group(V, y_final, prob_embed, groups)
        gl['V_final_partial_embed'] = r
        print(f"    V → final | embed:   R = {r:.3f}  ← STRICTEST (GroupKFold)")

        r, _ = partial_r_group(V, y_final, local, groups)
        gl['V_final_partial_local'] = r
        print(f"    V → final | local:   R = {r:.3f}")

        embed_local = np.hstack([prob_embed, local])
        r, _ = partial_r_group(V, y_final, embed_local, groups)
        gl['V_final_partial_embed_local'] = r
        print(f"    V → final | e+l:     R = {r:.3f}  ← STRICTEST COMBINED (GroupKFold)")

        nums_local = np.hstack([prob_nums, local])
        r, _ = partial_r_group(V, y_final, nums_local, groups)
        gl['V_final_partial_nums_local'] = r
        print(f"    V → final | n+l:     R = {r:.3f}")

        # Shuffle control
        y_shuffle = y_final.copy()
        np.random.seed(SEED + layer + 100)
        np.random.shuffle(y_shuffle)
        r, _ = run_probe_group_cv(V, y_shuffle, groups)
        gl['V_shuffle'] = r
        print(f"    V → shuffle:         R = {r:.3f}")

        results_group[f'L{layer}'] = gl

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Figures
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 4: Generating figures")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers_str = [f'L{l}' for l in PROBE_LAYERS]
    layers_pct = [f'L{l} ({100*l/36:.0f}%)' for l in PROBE_LAYERS]

    # Figure 1: Residualization hierarchy — bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    probe_names = ['V→final', 'nums→final', 'embed→final', 'local→final',
                   'V|nums', 'V|embed', 'V|local', 'V|n+l', 'V|e+l', 'shuffle']
    result_keys = ['V_final', 'nums_final', 'embed_final', 'local_final',
                   'V_final_partial_nums', 'V_final_partial_embed',
                   'V_final_partial_local', 'V_final_partial_nums_local',
                   'V_final_partial_embed_local', 'V_shuffle']

    x = np.arange(len(probe_names))
    width = 0.35
    for i, layer in enumerate(layers_str):
        if layer not in results:
            continue
        vals = [results[layer].get(k, 0) for k in result_keys]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=layers_pct[i], alpha=0.8)
        # Highlight challenge tests
        for j, v in enumerate(vals):
            if probe_names[j] in ['V|nums', 'V|embed', 'V|e+l']:
                bars[j].set_edgecolor('red')
                bars[j].set_linewidth(2)

    ax.set_ylabel('Pearson R')
    ax.set_title('Exp 079: Problem-Number Residualization\n(Red border = challenge tests)')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'residualization_hierarchy.png', dpi=150)
    plt.close()
    print("  Saved residualization_hierarchy.png")

    # Figure 2: Waterfall chart — how much R survives each control
    fig, axes = plt.subplots(1, len(PROBE_LAYERS), figsize=(6*len(PROBE_LAYERS), 5))
    if len(PROBE_LAYERS) == 1:
        axes = [axes]

    for ax, layer, lpct in zip(axes, layers_str, layers_pct):
        if layer not in results:
            continue
        r = results[layer]
        stages = ['V→final', '- nums', '- local', '- embed+local']
        values = [
            r['V_final'],
            r['V_final_partial_nums'],
            r['V_final_partial_local'],
            r['V_final_partial_embed_local'],
        ]
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

        bars = ax.bar(range(len(stages)), values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=30, ha='right')
        ax.set_ylabel('Pearson R')
        ax.set_title(f'{lpct}')
        ax.axhline(y=0, color='k', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle('How Much V→Final Signal Survives Each Control?', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'waterfall_controls.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved waterfall_controls.png")

    # Figure 3: Confound decomposition — what fraction of V's signal comes from each source?
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (layer, lpct) in enumerate(zip(layers_str, layers_pct)):
        if layer not in results:
            continue
        r = results[layer]
        total = r['V_final']
        from_nums = total - r['V_final_partial_nums']
        from_local = r['V_final_partial_nums'] - r['V_final_partial_nums_local']
        residual = r['V_final_partial_nums_local']

        # Stacked bar
        ax.bar(i, from_nums, color='#FF9800', alpha=0.8, label='From problem numbers' if i==0 else '')
        ax.bar(i, from_local, bottom=from_nums, color='#4CAF50', alpha=0.8, label='From local result' if i==0 else '')
        ax.bar(i, residual, bottom=from_nums+from_local, color='#F44336', alpha=0.8, label='Residual (hidden computation?)' if i==0 else '')
        ax.text(i, total + 0.01, f'R={total:.3f}', ha='center', fontsize=10)

    ax.set_xticks(range(len(layers_str)))
    ax.set_xticklabels(layers_pct)
    ax.set_ylabel('Pearson R contribution')
    ax.set_title('Decomposition: Where Does V→Final Signal Come From?')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'confound_decomposition.png', dpi=150)
    plt.close()
    print("  Saved confound_decomposition.png")

    # Figure 4: KFold vs GroupKFold comparison
    if results_group:
        fig, ax = plt.subplots(figsize=(10, 6))
        probe_keys_compare = [
            ('V_final', 'V→final'),
            ('embed_final', 'embed→final'),
            ('V_final_partial_nums', 'V|nums'),
            ('V_final_partial_embed', 'V|embed'),
            ('V_final_partial_local', 'V|local'),
            ('V_final_partial_embed_local', 'V|e+l'),
        ]
        x = np.arange(len(probe_keys_compare))
        width = 0.2
        for i, layer in enumerate(layers_str):
            if layer not in results or layer not in results_group:
                continue
            kfold_vals = [results[layer].get(k, 0) for k, _ in probe_keys_compare]
            group_vals = [results_group[layer].get(k, 0) for k, _ in probe_keys_compare]
            ax.bar(x + (2*i-1)*width, kfold_vals, width, label=f'{layers_pct[i]} KFold', alpha=0.7, hatch='/')
            ax.bar(x + (2*i)*width, group_vals, width, label=f'{layers_pct[i]} GroupKFold', alpha=0.9)

        ax.set_ylabel('Pearson R')
        ax.set_title('KFold vs GroupKFold (fixing within-problem leakage)')
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in probe_keys_compare], rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'kfold_vs_groupkfold.png', dpi=150)
        plt.close()
        print("  Saved kfold_vs_groupkfold.png")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Save results
    # ═══════════════════════════════════════════════════════════════
    summary = {
        'experiment': 'exp_079_number_residualization',
        'model': MODEL_NAME,
        'n_generated': len(gen_data),
        'n_correct': n_correct,
        'accuracy_pct': round(100 * n_correct / len(gen_data), 1),
        'n_valid': len(valid),
        'n_extracted': n_extracted,
        'total_ops': total_ops,
        'error_ops': error_ops,
        'kv_dim': kv_dim,
        'hidden_size': hidden_size,
        'prob_nums_dim': 20,
        'probe_layers': PROBE_LAYERS,
        'results': {},
        'runtime_s': round(time.time() - t0, 1),
    }
    # Convert numpy values for JSON
    for layer_key, layer_res in results.items():
        summary['results'][layer_key] = {}
        for k, v in layer_res.items():
            if isinstance(v, dict):
                summary['results'][layer_key][k] = {
                    kk: (list(vv) if isinstance(vv, tuple) else round(float(vv), 4) if isinstance(vv, (float, np.floating)) else vv)
                    for kk, vv in v.items()
                }
            elif isinstance(v, (float, np.floating)):
                summary['results'][layer_key][k] = round(float(v), 4)
            else:
                summary['results'][layer_key][k] = v

    # Add GroupKFold results
    summary['results_groupkfold'] = {}
    for layer_key, layer_res in results_group.items():
        summary['results_groupkfold'][layer_key] = {
            k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
            for k, v in layer_res.items()
        }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'results.json'}")

    # ── Print final summary ──
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Problems: {len(gen_data)} generated, {n_correct} correct ({summary['accuracy_pct']}%)")
    print(f"  Operations: {total_ops} total, {error_ops} errors ({100*error_ops/total_ops:.2f}%)")
    print()

    for layer in layers_str:
        if layer not in results:
            continue
        r = results[layer]
        print(f"  {layer}:")
        print(f"    V → final:               R = {r['V_final']:.3f}")
        print(f"    prob_nums → final:        R = {r['nums_final']:.3f}")
        print(f"    prob_embed → final:       R = {r['embed_final']:.3f}")
        print(f"    V → final | nums:         R = {r['V_final_partial_nums']:.3f}  ← KEY CHALLENGE")
        print(f"    V → final | embed:        R = {r['V_final_partial_embed']:.3f}")
        print(f"    V → final | local:        R = {r['V_final_partial_local']:.3f}")
        print(f"    V → final | nums+local:   R = {r['V_final_partial_nums_local']:.3f}")
        print(f"    V → final | embed+local:  R = {r['V_final_partial_embed_local']:.3f}  ← STRICTEST")
        print(f"    local → final | nums:     R = {r['local_final_partial_nums']:.3f}")
        print(f"    Shuffle:                  R = {r['V_shuffle']:.3f}")
        if 'bootstrap_V_partial_nums' in r:
            bp = r['bootstrap_V_partial_nums']
            ci = bp['ci'] if isinstance(bp['ci'], (list, tuple)) else bp['ci']
            print(f"    Bootstrap V|nums: R={bp['mean_r']:.3f}, CI=[{ci[0]:.3f},{ci[1]:.3f}], p={bp['p']:.4f}")
        print()

    # GroupKFold results
    if results_group:
        print(f"\n  --- GroupKFold (proper problem-level CV) ---")
        for layer in layers_str:
            if layer not in results_group:
                continue
            r = results_group[layer]
            print(f"  {layer} (GroupKFold):")
            print(f"    V → final:               R = {r['V_final']:.3f}")
            print(f"    prob_nums → final:        R = {r['nums_final']:.3f}")
            print(f"    prob_embed → final:       R = {r['embed_final']:.3f}")
            print(f"    V → final | nums:         R = {r['V_final_partial_nums']:.3f}  ← KEY (GroupKFold)")
            print(f"    V → final | embed:        R = {r['V_final_partial_embed']:.3f}  ← STRICTEST (GroupKFold)")
            print(f"    V → final | local:        R = {r['V_final_partial_local']:.3f}")
            print(f"    V → final | embed+local:  R = {r['V_final_partial_embed_local']:.3f}")
            print(f"    V → final | nums+local:   R = {r['V_final_partial_nums_local']:.3f}")
            print(f"    Shuffle:                  R = {r['V_shuffle']:.3f}")
            print()

    print(f"\n  Total runtime: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
