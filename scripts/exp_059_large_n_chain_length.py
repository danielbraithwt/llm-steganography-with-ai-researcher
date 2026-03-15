#!/usr/bin/env python3
"""
Experiment 059: Large-N Compression Replication + Chain-Length Dependence + Hybrid Strategy

Tests whether "recent" universality (100% at all budgets) holds at large N,
whether chain-length modulates strategy effectiveness, and whether hybrid
strategies (k_norm + recency) outperform pure recent.

Strategies:
1. recent    — sinks (first 4) + most recent N positions
2. k_norm    — highest K-norm L2 across layers
3. random    — random position selection
4. hybrid_50_50 — sinks + 50% k_norm + 50% recent
5. hybrid_70_30 — sinks + 70% recent + 30% k_norm

Budget: 33% under masking (where differences are largest)
Models: Qwen3-4B-Base + Llama-3.1-8B-Instruct
"""

import os
import json
import time
import random as pyrandom
import gc
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
TIME_BUDGET = 1750  # seconds total for both models
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 2048
BUDGET_FRAC = 0.33
N_SINKS = 4
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_059")

MODELS = [
    ("Qwen/Qwen3-4B-Base", 20),   # (model_name, n_target)
    ("meta-llama/Llama-3.1-8B-Instruct", 40),
]

STRATEGIES = ['recent', 'k_norm', 'random', 'hybrid_50_50', 'hybrid_70_30']

# Chain-length buckets
LENGTH_BUCKETS = {
    'short': (0, 100),
    'medium': (100, 200),
    'long': (200, float('inf')),
}

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
    if "\nQ:" in text:
        text = text[:text.index("\nQ:")]
    if "####" in text:
        after = text.split("####")[-1].strip()
        m = re.search(r'-?[\d,]+\.?\d*', after)
        return m.group(0).replace(',', '') if m else ""
    m = re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    return nums[-1].replace(',', '') if nums else ""


def normalize_answer(ans):
    ans = ans.strip().replace(",", "").replace("$", "")
    try:
        val = float(ans)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return ans


def extract_evicted_answer(text):
    """Extract the answer number from eviction-generated text.

    The model generates after '####' prefix. The answer should be the
    first number, but the model may continue generating past it.
    """
    text = text.strip()
    # Take only text before first newline (answer should be on first line)
    if '\n' in text:
        text = text[:text.index('\n')]
    text = text.strip()
    # Extract first number
    m = re.match(r'\$?\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(",", "")
    return text


def find_truncation_point(reasoning_ids, tokenizer):
    ids_list = reasoning_ids[0].tolist()
    text = tokenizer.decode(ids_list, skip_special_tokens=True)
    if "####" in text:
        prefix = text[:text.index("####")]
        prefix_toks = tokenizer.encode(prefix, add_special_tokens=False)
        pos = len(prefix_toks)
        if pos >= 10:
            return pos
    m = re.search(r'[Tt]he (?:final )?answer is', text)
    if m:
        prefix = text[:m.start()]
        prefix_toks = tokenizer.encode(prefix, add_special_tokens=False)
        pos = len(prefix_toks)
        if pos >= 10:
            return pos
    return None


def wilson_ci(n_success, n_total, z=1.96):
    if n_total == 0:
        return (0.0, 0.0)
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return (max(0, center - spread), min(1, center + spread))


def select_positions(strategy, budget_frac, reasoning_len, prompt_len, k_norm_importance):
    """Select which reasoning positions to keep based on strategy."""
    n_keep = max(1, int(budget_frac * reasoning_len))
    reason_positions = list(range(prompt_len, prompt_len + reasoning_len))
    reason_k_norms = k_norm_importance[prompt_len:prompt_len + reasoning_len]

    if strategy == "random":
        rng = np.random.RandomState(SEED)
        keep_idx = set(rng.choice(reasoning_len, size=min(n_keep, reasoning_len),
                                  replace=False).tolist())

    elif strategy == "k_norm":
        keep_idx = set(np.argsort(reason_k_norms)[-n_keep:].tolist())

    elif strategy == "recent":
        n_sinks = min(N_SINKS, reasoning_len)
        n_recent = max(0, n_keep - n_sinks)
        if n_recent <= 0:
            keep_list = list(range(min(n_keep, reasoning_len)))
        else:
            sink_set = list(range(n_sinks))
            recent_set = list(range(max(n_sinks, reasoning_len - n_recent), reasoning_len))
            keep_list = sorted(set(sink_set + recent_set))
        keep_idx = set(keep_list[:n_keep]) if len(keep_list) > n_keep else set(keep_list)

    elif strategy == "hybrid_50_50":
        keep_idx = _hybrid_select(reason_k_norms, reasoning_len, n_keep, 0.50)

    elif strategy == "hybrid_70_30":
        keep_idx = _hybrid_select(reason_k_norms, reasoning_len, n_keep, 0.70)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    keep_pos = sorted(reason_positions[i] for i in keep_idx)
    evict_pos = sorted(reason_positions[i] for i in range(reasoning_len) if i not in keep_idx)
    return keep_pos, evict_pos, keep_idx


def _hybrid_select(reason_k_norms, reasoning_len, n_keep, recent_frac):
    """Hybrid strategy: sinks + recent_frac of budget as recent + rest as k_norm."""
    n_sinks = min(N_SINKS, reasoning_len)
    n_remaining = max(0, n_keep - n_sinks)
    n_recent = max(0, int(recent_frac * n_remaining))
    n_knorm = max(0, n_remaining - n_recent)

    # Start with sinks
    selected = set(range(n_sinks))

    # Add recent positions (from the end)
    if n_recent > 0:
        recent_start = max(n_sinks, reasoning_len - n_recent)
        for i in range(recent_start, reasoning_len):
            selected.add(i)

    # Add highest k_norm NOT already selected
    if n_knorm > 0:
        remaining_indices = [i for i in range(reasoning_len) if i not in selected]
        if remaining_indices:
            remaining_scores = [(reason_k_norms[i], i) for i in remaining_indices]
            remaining_scores.sort(reverse=True)
            for _, idx in remaining_scores[:n_knorm]:
                selected.add(idx)

    # Trim to n_keep if needed
    if len(selected) > n_keep:
        selected = set(sorted(selected)[:n_keep])

    return selected


@torch.no_grad()
def generate_and_build_cache(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS):
    """Generate CoT trace and build KV cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
    )
    full_ids = gen_out.sequences[0]
    gen_ids = full_ids[prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del gen_out
    gc.collect(); torch.cuda.empty_cache()

    reasoning_ids = gen_ids.unsqueeze(0)
    trunc_point = find_truncation_point(reasoning_ids, tokenizer)
    if trunc_point is None or trunc_point < 10:
        return None

    full_seq = torch.cat([inputs.input_ids[0], gen_ids[:trunc_point]])
    total_len = full_seq.shape[0]
    if total_len > MAX_SEQ_LEN:
        return None

    outputs = model(input_ids=full_seq.unsqueeze(0), use_cache=True)
    full_cache = outputs.past_key_values
    del outputs

    num_layers = model.config.num_hidden_layers
    reasoning_len = trunc_point

    # Compute K-norm importance
    k_norm_importance = np.zeros(total_len)
    for l in range(num_layers):
        k = full_cache.layers[l].keys[0]
        k_norms = k.float().norm(dim=-1).cpu().numpy()
        k_norm_importance += k_norms.mean(axis=0)

    return {
        'gen_text': gen_text,
        'full_seq': full_seq,
        'full_cache': full_cache,
        'prompt_len': prompt_len,
        'reasoning_len': reasoning_len,
        'k_norm_importance': k_norm_importance,
        'total_len': total_len,
    }


@torch.no_grad()
def evict_and_generate(model, tokenizer, full_cache, prompt_len, reasoning_len,
                       evict_positions, num_layers, max_answer_tokens=64):
    """Apply masking eviction and generate answer."""
    total_len = prompt_len + reasoning_len
    device = model.device
    evicted_cache = DynamicCache()

    # Build eviction mask
    evict_tensor = torch.zeros(total_len, dtype=torch.bool, device=device)
    if evict_positions:
        evict_idx = torch.tensor(evict_positions, dtype=torch.long, device=device)
        evict_tensor[evict_idx] = True

    for l in range(num_layers):
        k = full_cache.layers[l].keys[:, :, :total_len, :].clone()
        v = full_cache.layers[l].values[:, :, :total_len, :].clone()
        evicted_cache.update(k, v, l)

    # Attention mask with 0 at evicted positions
    attn_mask = torch.ones(1, total_len, dtype=torch.long, device=device)
    attn_mask[0, evict_tensor] = 0

    # Generate answer — feed " ####" prefix then decode
    answer_prefix = tokenizer.encode(" ####", add_special_tokens=False)
    cache_len = total_len

    out = None
    for i in range(len(answer_prefix)):
        tok = torch.tensor([[answer_prefix[i]]], device=device)
        attn_mask = torch.cat([attn_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
        out = model(input_ids=tok, past_key_values=evicted_cache, use_cache=True,
                    attention_mask=attn_mask,
                    position_ids=torch.tensor([[cache_len]], device=device))
        evicted_cache = out.past_key_values
        cache_len += 1

    gen_tokens = []
    next_logits = out.logits[:, -1, :]
    # Build stop set: newlines + Q token (model may start next question)
    newline_ids = set()
    for nl in ["\n", "\r", "\r\n"]:
        newline_ids.update(tokenizer.encode(nl, add_special_tokens=False))
    q_ids = set(tokenizer.encode("\nQ", add_special_tokens=False))

    for step in range(max_answer_tokens):
        next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
        token_id = next_tok[0, 0].item()
        if token_id == tokenizer.eos_token_id or token_id in newline_ids:
            break
        # Also stop if we've got at least one token and hit a letter (answer should be numeric)
        if gen_tokens and token_id in q_ids:
            break
        gen_tokens.append(token_id)
        attn_mask = torch.cat([attn_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
        out = model(input_ids=next_tok, past_key_values=evicted_cache, use_cache=True,
                    attention_mask=attn_mask,
                    position_ids=torch.tensor([[cache_len]], device=device))
        evicted_cache = out.past_key_values
        next_logits = out.logits[:, -1, :]
        cache_len += 1

    answer_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    del evicted_cache
    return answer_text


def get_length_bucket(reasoning_len):
    for name, (lo, hi) in LENGTH_BUCKETS.items():
        if lo <= reasoning_len < hi:
            return name
    return 'long'


def run_model(model_name, n_target, gsm8k_data, global_start, time_remaining):
    """Run all strategies for one model."""
    short_name = model_name.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"  Model: {short_name}  (target {n_target} valid problems)")
    print(f"  Time remaining: {time_remaining:.0f}s")
    print(f"{'='*60}")

    model_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"  Model loaded in {time.time() - model_start:.1f}s  ({num_layers} layers)")

    np.random.seed(SEED)
    pyrandom.seed(SEED)
    torch.manual_seed(SEED)

    # Shuffle problems for variety across runs
    problem_indices = list(range(len(gsm8k_data)))
    rng = pyrandom.Random(SEED + hash(model_name))
    rng.shuffle(problem_indices)

    results = []
    n_valid = 0
    n_attempted = 0

    for prob_idx in problem_indices:
        if n_valid >= n_target:
            break
        elapsed = time.time() - global_start
        if elapsed > time_remaining:
            print(f"  Time budget exhausted after {n_valid} valid problems")
            break

        problem = gsm8k_data[prob_idx]
        question = problem['question']
        true_answer = normalize_answer(problem['answer'].split("####")[-1].strip())

        n_attempted += 1
        prompt = build_prompt(question)

        try:
            cache_data = generate_and_build_cache(model, tokenizer, prompt)
        except Exception as e:
            print(f"  Problem {prob_idx}: generation failed ({e})")
            continue

        if cache_data is None:
            continue

        gen_text = cache_data['gen_text']
        reasoning_len = cache_data['reasoning_len']
        prompt_len = cache_data['prompt_len']
        full_cache = cache_data['full_cache']
        k_norm_importance = cache_data['k_norm_importance']

        # Check baseline accuracy
        gen_answer = normalize_answer(extract_answer(gen_text))
        if gen_answer != true_answer:
            del cache_data['full_cache']
            gc.collect(); torch.cuda.empty_cache()
            continue

        n_valid += 1
        length_bucket = get_length_bucket(reasoning_len)

        prob_result = {
            'problem_idx': prob_idx,
            'true_answer': true_answer,
            'reasoning_len': reasoning_len,
            'length_bucket': length_bucket,
            'strategies': {},
        }

        # Test each strategy
        for strategy in STRATEGIES:
            try:
                keep_pos, evict_pos, keep_idx = select_positions(
                    strategy, BUDGET_FRAC, reasoning_len, prompt_len, k_norm_importance)

                answer = evict_and_generate(
                    model, tokenizer, full_cache, prompt_len, reasoning_len,
                    evict_pos, num_layers)

                pred_answer = normalize_answer(extract_evicted_answer(answer))
                correct = (pred_answer == true_answer)

                # Record position analysis
                keep_list = sorted(keep_idx)
                if reasoning_len > 0:
                    mean_pos = np.mean(keep_list) / reasoning_len if keep_list else 0.0
                    q1_frac = sum(1 for x in keep_list if x < reasoning_len * 0.2) / max(len(keep_list), 1)
                    q5_frac = sum(1 for x in keep_list if x >= reasoning_len * 0.8) / max(len(keep_list), 1)
                else:
                    mean_pos = q1_frac = q5_frac = 0.0

                prob_result['strategies'][strategy] = {
                    'answer': pred_answer,
                    'correct': correct,
                    'n_keep': len(keep_pos),
                    'n_evict': len(evict_pos),
                    'mean_pos': float(mean_pos),
                    'q1_frac': float(q1_frac),
                    'q5_frac': float(q5_frac),
                }

            except Exception as e:
                print(f"    Strategy {strategy} failed: {e}")
                prob_result['strategies'][strategy] = {
                    'answer': '', 'correct': False,
                    'n_keep': 0, 'n_evict': 0,
                    'mean_pos': 0, 'q1_frac': 0, 'q5_frac': 0,
                    'error': str(e),
                }

        results.append(prob_result)
        del cache_data['full_cache']
        gc.collect(); torch.cuda.empty_cache()

        # Progress report every 5 valid problems
        if n_valid % 5 == 0:
            elapsed_model = time.time() - model_start
            strat_summary = {s: sum(1 for r in results if r['strategies'].get(s, {}).get('correct', False))
                           for s in STRATEGIES}
            print(f"  [{n_valid}/{n_target}] {elapsed_model:.0f}s  "
                  f"Acc: " + " | ".join(f"{s}={strat_summary[s]}/{n_valid}" for s in STRATEGIES))

    # Final summary
    elapsed_model = time.time() - model_start
    print(f"\n  {short_name}: {n_valid} valid / {n_attempted} attempted in {elapsed_model:.0f}s")

    # Clean up model
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    return {
        'model': short_name,
        'n_valid': n_valid,
        'n_attempted': n_attempted,
        'elapsed': elapsed_model,
        'results': results,
    }


def analyze_and_plot(all_results):
    """Analyze results and generate figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for model_data in all_results:
        model_name = model_data['model']
        results = model_data['results']
        n_valid = model_data['n_valid']

        if n_valid == 0:
            continue

        print(f"\n{'='*60}")
        print(f"  Analysis: {model_name} (n={n_valid})")
        print(f"{'='*60}")

        # ── Overall accuracy by strategy ──
        strategy_acc = {}
        for strategy in STRATEGIES:
            n_correct = sum(1 for r in results if r['strategies'].get(strategy, {}).get('correct', False))
            acc = n_correct / n_valid if n_valid > 0 else 0
            lo, hi = wilson_ci(n_correct, n_valid)
            strategy_acc[strategy] = {
                'n_correct': n_correct, 'n_total': n_valid,
                'accuracy': acc, 'ci_lo': lo, 'ci_hi': hi,
            }
            print(f"  {strategy:15s}: {n_correct}/{n_valid} = {acc*100:.1f}%  "
                  f"[{lo*100:.1f}, {hi*100:.1f}]")

        # ── Chain-length distribution ──
        length_dist = {}
        for bucket_name in LENGTH_BUCKETS:
            bucket_results = [r for r in results if r['length_bucket'] == bucket_name]
            n_bucket = len(bucket_results)
            if n_bucket == 0:
                continue

            length_dist[bucket_name] = {'n': n_bucket, 'strategies': {}}
            print(f"\n  {bucket_name} chains (n={n_bucket}):")

            for strategy in STRATEGIES:
                n_correct = sum(1 for r in bucket_results
                              if r['strategies'].get(strategy, {}).get('correct', False))
                acc = n_correct / n_bucket if n_bucket > 0 else 0
                lo, hi = wilson_ci(n_correct, n_bucket)
                length_dist[bucket_name]['strategies'][strategy] = {
                    'n_correct': n_correct, 'n_total': n_bucket,
                    'accuracy': acc, 'ci_lo': lo, 'ci_hi': hi,
                }
                print(f"    {strategy:15s}: {n_correct}/{n_bucket} = {acc*100:.1f}%  "
                      f"[{lo*100:.1f}, {hi*100:.1f}]")

        # ── Chain-length statistics ──
        chain_lengths = [r['reasoning_len'] for r in results]
        print(f"\n  Chain length stats: min={min(chain_lengths)}, max={max(chain_lengths)}, "
              f"median={np.median(chain_lengths):.0f}, mean={np.mean(chain_lengths):.0f}")

        # ── Per-problem analysis: which problems does each strategy get wrong? ──
        print(f"\n  Per-problem failure analysis:")
        for strategy in STRATEGIES:
            failures = [r for r in results if not r['strategies'].get(strategy, {}).get('correct', False)]
            if failures:
                fail_lengths = [r['reasoning_len'] for r in failures]
                print(f"    {strategy}: {len(failures)} failures, "
                      f"chain lengths = {sorted(fail_lengths)}")
            else:
                print(f"    {strategy}: 0 failures (100%)")

        # ── Correlation: chain length vs accuracy (per-strategy) ──
        print(f"\n  Chain length × accuracy correlation:")
        for strategy in STRATEGIES:
            correct_list = [1 if r['strategies'].get(strategy, {}).get('correct', False) else 0
                          for r in results]
            if len(set(correct_list)) > 1:  # Need variance
                try:
                    from scipy.stats import pointbiserialr
                    r_pb, p_val = pointbiserialr(chain_lengths, correct_list)
                    print(f"    {strategy}: r_pb={r_pb:.3f}, p={p_val:.4f}")
                except ImportError:
                    # Fallback: compute manually
                    cl_arr = np.array(chain_lengths)
                    co_arr = np.array(correct_list)
                    mean_1 = cl_arr[co_arr == 1].mean() if co_arr.sum() > 0 else 0
                    mean_0 = cl_arr[co_arr == 0].mean() if (1 - co_arr).sum() > 0 else 0
                    print(f"    {strategy}: mean_len(correct)={mean_1:.0f}, mean_len(wrong)={mean_0:.0f}")
            else:
                print(f"    {strategy}: no variance (all {'correct' if correct_list[0] else 'incorrect'})")

        # ── Position profile by strategy ──
        print(f"\n  Mean position profile:")
        for strategy in STRATEGIES:
            mean_positions = [r['strategies'].get(strategy, {}).get('mean_pos', 0) for r in results]
            q1_fracs = [r['strategies'].get(strategy, {}).get('q1_frac', 0) for r in results]
            q5_fracs = [r['strategies'].get(strategy, {}).get('q5_frac', 0) for r in results]
            print(f"    {strategy:15s}: mean_pos={np.mean(mean_positions):.3f}, "
                  f"Q1={np.mean(q1_fracs)*100:.1f}%, Q5={np.mean(q5_fracs)*100:.1f}%")

        # ── FIGURE 1: Overall accuracy bar chart ──
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(STRATEGIES))
        accs = [strategy_acc[s]['accuracy'] * 100 for s in STRATEGIES]
        ci_los = [strategy_acc[s]['ci_lo'] * 100 for s in STRATEGIES]
        ci_his = [strategy_acc[s]['ci_hi'] * 100 for s in STRATEGIES]
        errors_lo = [a - l for a, l in zip(accs, ci_los)]
        errors_hi = [h - a for a, h in zip(accs, ci_his)]

        colors = ['#2ecc71', '#3498db', '#95a5a6', '#e67e22', '#e74c3c']
        bars = ax.bar(x, accs, yerr=[errors_lo, errors_hi], capsize=5,
                      color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in STRATEGIES], fontsize=9)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{model_name} — KV Eviction at 33% Budget (Masking)\nn={n_valid} problems')
        ax.set_ylim(0, 110)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3, label='Baseline')

        for bar, acc_val, n_c in zip(bars, accs, [strategy_acc[s]['n_correct'] for s in STRATEGIES]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{acc_val:.1f}%\n({n_c}/{n_valid})', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, f'overall_accuracy_{model_name}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

        # ── FIGURE 2: Chain-length breakdown ──
        buckets_with_data = [b for b in LENGTH_BUCKETS if b in length_dist]
        if len(buckets_with_data) >= 2:
            fig, axes = plt.subplots(1, len(buckets_with_data), figsize=(5*len(buckets_with_data), 6),
                                     sharey=True)
            if len(buckets_with_data) == 1:
                axes = [axes]

            for ax_idx, bucket_name in enumerate(buckets_with_data):
                ax = axes[ax_idx]
                bd = length_dist[bucket_name]
                n_bucket = bd['n']
                x = np.arange(len(STRATEGIES))
                accs_b = [bd['strategies'].get(s, {}).get('accuracy', 0) * 100 for s in STRATEGIES]
                ci_los_b = [bd['strategies'].get(s, {}).get('ci_lo', 0) * 100 for s in STRATEGIES]
                ci_his_b = [bd['strategies'].get(s, {}).get('ci_hi', 0) * 100 for s in STRATEGIES]
                errors_lo_b = [max(0, a - l) for a, l in zip(accs_b, ci_los_b)]
                errors_hi_b = [max(0, h - a) for a, h in zip(accs_b, ci_his_b)]

                ax.bar(x, accs_b, yerr=[errors_lo_b, errors_hi_b], capsize=4,
                       color=colors, edgecolor='black', linewidth=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels([s.replace('_', '\n') for s in STRATEGIES], fontsize=7)
                ax.set_title(f'{bucket_name} (n={n_bucket})')
                ax.set_ylim(0, 110)
                ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
                if ax_idx == 0:
                    ax.set_ylabel('Accuracy (%)')

            plt.suptitle(f'{model_name} — Strategy × Chain Length at 33% Budget', fontsize=12)
            plt.tight_layout()
            fig_path = os.path.join(RESULTS_DIR, f'chain_length_{model_name}.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {fig_path}")

        # ── FIGURE 3: Chain-length histogram ──
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(chain_lengths, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        for bucket_name, (lo, hi) in LENGTH_BUCKETS.items():
            if hi == float('inf'):
                hi = max(chain_lengths) + 10
            ax.axvline(x=lo, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Reasoning Chain Length (tokens)')
        ax.set_ylabel('Count')
        ax.set_title(f'{model_name} — Chain Length Distribution (n={n_valid})')

        # Add bucket labels
        for bucket_name, (lo, hi) in LENGTH_BUCKETS.items():
            n_in = sum(1 for cl in chain_lengths if lo <= cl < hi)
            mid = lo + min(hi, max(chain_lengths) + 10) / 2 if lo == 0 else (lo + min(hi, max(chain_lengths))) / 2
            if n_in > 0:
                ax.text(mid, ax.get_ylim()[1] * 0.9, f'{bucket_name}\n(n={n_in})',
                        ha='center', fontsize=9, color='red')

        plt.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, f'chain_length_dist_{model_name}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

        # ── FIGURE 4: Scatter — chain length vs per-strategy correctness ──
        fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(4*len(STRATEGIES), 4), sharey=True)
        for s_idx, strategy in enumerate(STRATEGIES):
            ax = axes[s_idx]
            for r in results:
                c = r['strategies'].get(strategy, {}).get('correct', False)
                ax.scatter(r['reasoning_len'], 1 if c else 0,
                          color='green' if c else 'red', alpha=0.4, s=30)
            ax.set_xlabel('Chain Length')
            ax.set_title(strategy.replace('_', '\n'), fontsize=9)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Wrong', 'Correct'])
        plt.suptitle(f'{model_name} — Per-Problem Correctness vs Chain Length', fontsize=11)
        plt.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, f'scatter_correctness_{model_name}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

        # ── Save per-model results JSON ──
        save_data = {
            'model': model_name,
            'n_valid': n_valid,
            'n_attempted': model_data['n_attempted'],
            'elapsed': model_data['elapsed'],
            'strategy_accuracy': strategy_acc,
            'length_distribution': length_dist,
            'chain_length_stats': {
                'min': int(min(chain_lengths)),
                'max': int(max(chain_lengths)),
                'median': float(np.median(chain_lengths)),
                'mean': float(np.mean(chain_lengths)),
            },
            'per_problem_results': results,
        }
        json_path = os.path.join(RESULTS_DIR, f'results_{model_name}.json')
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"  Saved: {json_path}")


def main():
    global_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load GSM8K
    print("Loading GSM8K dataset...")
    from datasets import load_dataset
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_data = list(gsm8k)
    print(f"  Loaded {len(gsm8k_data)} problems")

    all_results = []

    for model_name, n_target in MODELS:
        elapsed = time.time() - global_start
        remaining = TIME_BUDGET - elapsed
        if remaining < 120:
            print(f"\nSkipping {model_name} — only {remaining:.0f}s remaining")
            continue

        model_results = run_model(model_name, n_target, gsm8k_data,
                                  global_start, TIME_BUDGET)
        all_results.append(model_results)

    # Analysis and plotting
    print("\n" + "="*60)
    print("  ANALYSIS AND VISUALIZATION")
    print("="*60)
    analyze_and_plot(all_results)

    elapsed_total = time.time() - global_start
    print(f"\nTotal runtime: {elapsed_total:.0f}s")


if __name__ == "__main__":
    main()
