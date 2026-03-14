#!/usr/bin/env python3
"""
Experiment 013: Selectivity-Based Protection + Positional Analysis

Tests whether selectivity (AC-TC rank difference) rescues the AC/TC framework
after exp_012 showed that raw AC-based protection fails on Llama.

Key additions over exp_012:
1. SEL (selectivity) protection strategy: protect most answer-selective positions
2. Positional analysis: where in the sequence does each strategy noise?
3. Position-controlled protection: test selectivity within position quartiles

Method:
  For each problem:
  1. Generate clean CoT trace, verify correct
  2. Forward pass with output_attentions → compute AC, H2O, TC, SEL scores
  3. Forward pass with use_cache → build base KV cache
  4. For each (noise_fraction, strategy):
     - Select positions to NOISE (lowest by strategy score)
     - Record positional statistics of noise set
     - Apply norm-matched noise, generate answer, check accuracy
"""

import os
import sys
import json
import time
import random
import gc
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
from scipy import stats as scipy_stats

# ── Config ──────────────────────────────────────────────────────────────
NUM_PROBLEMS = 35
NOISE_FRACTIONS = [0.01, 0.03, 0.05, 0.07, 0.10]
STRATEGIES = ["sel", "ac", "h2o", "tc", "random"]
ATTENTION_LAYERS_AC = [-1, -2, -3, -4]  # last 4 layers for AC/TC
MAX_GEN_TOKENS = 64
MAX_SEQ_LEN = 1536
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_013")

os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 8-shot GSM8K exemplars (same as prior experiments) ──────────────────
GSM8K_EXEMPLARS = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
     "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total bolts needed is 2+1=<<2+1=3>>3\n#### 3"},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
     "answer": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*150%=$<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"},
    {"question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "answer": "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624"},
    {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
     "answer": "If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.\nIf she feeds the flock 15 cups in the morning, and 25 cups in the afternoon, then the carry-over to the final meal would be 60-15-25=<<60-15-25=20>>20 cups.\n#### 20"},
    {"question": "Kylar went to the store to get water and some apples. The store sold apples for $1 each and water for $3 per bottle. Kylar wanted to buy one bag of apples and 2 bottles of water. How much would Kylar spend if each bag has 6 apples?",
     "answer": "A bag has 6 apples and each apple costs $1, so a bag costs 6*1=$<<6*1=6>>6\nKylar wants 2 bottles of water so that would cost 2*3=$<<2*3=6>>6\nAltogether, Kylar would spend 6+6=$<<6+6=12>>12\n#### 12"},
    {"question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
     "answer": "If Seattle has 20 sheep, Charleston has 4 * 20 = <<4*20=80>>80 sheep\nToulouse has 2 * 80 = <<2*80=160>>160 sheep\nTogether, they have 20 + 80 + 160 = <<20+80+160=260>>260 sheep\n#### 260"},
    {"question": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
     "answer": "First find how long it takes to download 40% of the file: 200 GB * 0.4 / 2 GB/minute = <<200*0.4/2=40>>40 minutes\nThen find how long it takes to download the whole file once the restart is complete: 200 GB / 2 GB/minute = <<200/2=100>>100 minutes\nThen add the time to download 40% of the file, the restart time, and the time to download the whole file: 40 + 20 + 100 = <<40+20+100=160>>160 minutes\n#### 160"},
]


def build_prompt(question):
    prompt = ""
    for ex in GSM8K_EXEMPLARS:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def extract_answer(text):
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = ans.replace(",", "").replace("$", "").strip()
        match = re.match(r'^-?[\d.]+', ans)
        if match:
            return match.group(0)
        return ans
    m = re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(",", "")
    return ""


def normalize_answer(ans):
    ans = ans.strip().replace(",", "").replace("$", "")
    try:
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans


@torch.no_grad()
def generate_trace(model, tokenizer, prompt, max_tokens=512):
    """Generate a full CoT trace via greedy autoregressive decoding."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = []
    past_kv = None
    current_input = inputs.input_ids

    for step in range(max_tokens):
        if past_kv is not None:
            outputs = model(input_ids=current_input, past_key_values=past_kv, use_cache=True)
        else:
            outputs = model(**inputs, use_cache=True)

        past_kv = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        token_id = next_token[0, 0].item()
        generated_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

        current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "####" in current_text:
            after = current_text.split("####")[-1]
            if re.search(r'\d+', after):
                break

        if re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?\s*-?[\d,]+', current_text):
            break

        if "\nQ:" in current_text or "\n\nQ:" in current_text:
            idx = current_text.find("\nQ:")
            if idx > 0:
                truncated = current_text[:idx]
                generated_ids = tokenizer.encode(truncated, add_special_tokens=False)
            break

        current_input = next_token

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    del past_kv, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return generated_text


@torch.no_grad()
def compute_importance_scores(model, tokenizer, prompt, trace_text):
    """
    Forward pass with output_attentions to compute per-position importance scores:
    1. H2O: cumulative attention from ALL subsequent tokens (all layers)
    2. AC: attention from last (answer) token (last 4 layers)
    3. TC: average attention from later reasoning tokens (last 4 layers)
    4. SEL: selectivity = rank(AC) - rank(TC)
    """
    full_text = prompt + trace_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    reasoning_len = seq_len - prompt_len

    if reasoning_len < 10:
        return None

    # Truncate if too long
    if seq_len > MAX_SEQ_LEN:
        max_reasoning = MAX_SEQ_LEN - prompt_len
        if max_reasoning < 10:
            return None
        trace_tokens = tokenizer(trace_text, return_tensors="pt").input_ids[0][:max_reasoning]
        trace_text = tokenizer.decode(trace_tokens, skip_special_tokens=True)
        full_text = prompt + trace_text
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]
        reasoning_len = seq_len - prompt_len

    outputs = model(**inputs, output_attentions=True, use_cache=False)

    num_layers = len(outputs.attentions)
    last_token_idx = seq_len - 1
    ac_layer_indices = [num_layers + i for i in ATTENTION_LAYERS_AC]

    h2o_score = torch.zeros(reasoning_len, device=model.device, dtype=torch.float32)
    ac_score = torch.zeros(reasoning_len, device=model.device, dtype=torch.float32)
    tc_score = torch.zeros(reasoning_len, device=model.device, dtype=torch.float32)

    num_heads = outputs.attentions[0].shape[1]

    for li in range(num_layers):
        attn = outputs.attentions[li][0].float()  # (num_heads, seq_len, seq_len)

        # H2O: cumulative attention received from subsequent tokens
        full_col_sums = attn[:, :, prompt_len:seq_len].sum(dim=1)  # (heads, reasoning_len)
        reasoning_block = attn[:, prompt_len:seq_len, prompt_len:seq_len]
        diag = torch.diagonal(reasoning_block, dim1=1, dim2=2)  # (heads, reasoning_len)
        h2o_layer = (full_col_sums - diag).sum(dim=0)  # (reasoning_len,)
        h2o_score += h2o_layer

        if li in ac_layer_indices:
            # AC: attention from last token to each reasoning position
            ac_score += attn[:, last_token_idx, prompt_len:seq_len].sum(dim=0)

            # TC: average attention from later reasoning tokens
            reasoning_attn_sum = reasoning_block.sum(dim=0)
            mask = torch.tril(
                torch.ones(reasoning_len, reasoning_len, device=model.device),
                diagonal=-1
            )
            weighted = reasoning_attn_sum * mask
            col_sums = weighted.sum(dim=0)
            col_counts = mask.sum(dim=0).clamp(min=1)
            tc_score += col_sums / col_counts
            del reasoning_attn_sum, mask, weighted

        del attn, reasoning_block

    num_ac_layers = len(ac_layer_indices)
    h2o_score = h2o_score / (num_layers * num_heads)
    ac_score = ac_score / (num_ac_layers * num_heads)
    tc_score = tc_score / (num_ac_layers * num_heads)

    # Compute selectivity: rank(AC) - rank(TC)
    ac_np = ac_score.cpu().numpy()
    tc_np = tc_score.cpu().numpy()
    h2o_np = h2o_score.cpu().numpy()

    ac_ranks = scipy_stats.rankdata(ac_np)
    tc_ranks = scipy_stats.rankdata(tc_np)
    sel_score = ac_ranks - tc_ranks  # positive = more answer-selective

    last_token_id = inputs.input_ids[0, -1].item()

    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "h2o_score": h2o_np,
        "ac_score": ac_np,
        "tc_score": tc_np,
        "sel_score": sel_score,
        "prompt_len": prompt_len,
        "reasoning_len": reasoning_len,
        "seq_len": seq_len,
        "last_token_id": last_token_id,
        "truncated_trace": trace_text,
    }


@torch.no_grad()
def build_kv_cache(model, tokenizer, prompt, trace_text):
    """Forward pass with use_cache=True to build the base KV cache."""
    full_text = prompt + trace_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    outputs = model(**inputs, use_cache=True, output_attentions=False)
    kv_cache = outputs.past_key_values
    seq_len = inputs.input_ids.shape[1]
    last_token_id = inputs.input_ids[0, -1].item()
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    return kv_cache, last_token_id, seq_len


def get_kv(cache, layer_idx):
    if hasattr(cache, 'layers') and len(cache.layers) > layer_idx:
        return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
    elif hasattr(cache, 'key_cache'):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    else:
        return cache[layer_idx][0], cache[layer_idx][1]


def set_kv(cache, layer_idx, keys, values):
    if hasattr(cache, 'layers') and len(cache.layers) > layer_idx:
        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values
    elif hasattr(cache, 'key_cache'):
        cache.key_cache[layer_idx] = keys
        cache.value_cache[layer_idx] = values


def clone_kv_cache(kv_cache, num_layers):
    cloned = DynamicCache()
    for li in range(num_layers):
        keys, values = get_kv(kv_cache, li)
        cloned.update(keys.clone(), values.clone(), li)
    return cloned


@torch.no_grad()
def noise_and_generate(model, tokenizer, base_kv, positions_to_noise,
                       prompt_len, seq_len, num_layers, last_token_id):
    """
    Clone KV cache, apply norm-matched noise to specified reasoning positions,
    truncate to seq_len-1, re-process last token, and generate answer.
    """
    cloned = clone_kv_cache(base_kv, num_layers)

    abs_positions = [prompt_len + p for p in positions_to_noise]
    for li in range(num_layers):
        keys, values = get_kv(cloned, li)
        for pos in abs_positions:
            if pos >= keys.shape[2]:
                continue
            k_norm = keys[:, :, pos, :].norm().item()
            k_noise = torch.randn_like(keys[:, :, pos, :])
            k_noise = k_noise * (k_norm / (k_noise.norm().item() + 1e-8))
            keys[:, :, pos, :] = k_noise
            v_norm = values[:, :, pos, :].norm().item()
            v_noise = torch.randn_like(values[:, :, pos, :])
            v_noise = v_noise * (v_norm / (v_noise.norm().item() + 1e-8))
            values[:, :, pos, :] = v_noise

    for li in range(num_layers):
        keys, values = get_kv(cloned, li)
        set_kv(cloned, li,
               keys[:, :, :seq_len - 1, :].clone(),
               values[:, :, :seq_len - 1, :].clone())

    last_token = torch.tensor([[last_token_id]], device=model.device)
    outputs = model(input_ids=last_token, past_key_values=cloned, use_cache=True)

    generated_ids = []
    for step in range(MAX_GEN_TOKENS):
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        token_id = next_token[0, 0].item()
        generated_ids.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "####" in current_text:
            after = current_text.split("####")[-1]
            if re.search(r'\d+', after):
                break
        if re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?\s*-?[\d,]+', current_text):
            break
        if "\nQ:" in current_text:
            break
        outputs = model(input_ids=next_token, past_key_values=cloned, use_cache=True)

    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    extracted = extract_answer(answer_text)

    del cloned, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return answer_text, extracted


def select_positions_to_noise(scores, reasoning_len, noise_fraction, strategy):
    """
    Select reasoning positions to NOISE (the ones NOT protected).
    Protection strategy protects the TOP positions by score; noises the BOTTOM.
    """
    n_noise = max(1, int(reasoning_len * noise_fraction))

    if strategy == "random":
        noised = sorted(random.sample(range(reasoning_len), min(n_noise, reasoning_len)))
    elif strategy in ["ac", "h2o", "tc", "sel"]:
        score = scores[f"{strategy}_score"]
        # Ascending sort: first n_noise are the LOWEST scored (least important by this metric)
        sorted_positions = np.argsort(score)
        noised = sorted(sorted_positions[:n_noise].tolist())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return noised


def main():
    start_time = time.time()
    print(f"{'='*70}")
    print(f"Experiment 013: Selectivity-Based Protection + Positional Analysis")
    print(f"Strategies: {STRATEGIES}")
    print(f"Noise fractions: {NOISE_FRACTIONS}")
    print(f"{'='*70}\n")

    # Load GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(dataset)))
    random.seed(SEED)
    random.shuffle(indices)
    selected = indices[:NUM_PROBLEMS]

    # Load model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    print(f"Loaded: {num_layers} layers, {num_heads} heads, {kv_heads} KV heads")

    # ── Phase 1: Generate traces + compute importance scores ──
    print("\n=== Phase 1: Trace generation + score computation ===")
    valid_problems = []

    for prob_idx, ds_idx in enumerate(selected):
        elapsed = time.time() - start_time
        if elapsed > 600:  # 10 min for phase 1
            print(f"Phase 1 time limit at problem {prob_idx}")
            break

        problem = dataset[ds_idx]
        question = problem["question"]
        true_answer = normalize_answer(
            problem["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
        )
        prompt = build_prompt(question)

        print(f"\nProblem {prob_idx+1}/{len(selected)} (#{ds_idx}), true={true_answer}")

        trace_text = generate_trace(model, tokenizer, prompt)
        gen_answer = extract_answer(trace_text)
        gen_norm = normalize_answer(gen_answer) if gen_answer else ""
        correct = (gen_norm == true_answer)
        print(f"  Generated: '{gen_answer}' (correct: {correct})")

        if not correct:
            print("  SKIP: baseline incorrect")
            continue

        # Extract reasoning (before ####)
        if "####" in trace_text:
            reasoning_text = trace_text[:trace_text.index("####")]
        else:
            reasoning_text = trace_text

        try:
            scores = compute_importance_scores(model, tokenizer, prompt, reasoning_text)
        except torch.cuda.OutOfMemoryError:
            print("  SKIP: OOM during attention extraction")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        if scores is None:
            print("  SKIP: trace too short")
            continue

        reasoning_text = scores["truncated_trace"]

        # Quick sanity: print score correlations
        ac_h2o_r = np.corrcoef(scores["ac_score"], scores["h2o_score"])[0, 1]
        sel_h2o_r = np.corrcoef(scores["sel_score"], scores["h2o_score"])[0, 1]
        print(f"  Valid! R={scores['reasoning_len']}, AC-H2O r={ac_h2o_r:.3f}, SEL-H2O r={sel_h2o_r:.3f}")

        valid_problems.append({
            "ds_idx": ds_idx,
            "true_answer": true_answer,
            "prompt": prompt,
            "reasoning_text": reasoning_text,
            "scores": scores,
        })

    print(f"\n{len(valid_problems)} valid problems")

    if len(valid_problems) < 5:
        print("ERROR: Too few valid problems!")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return

    # ── Positional analysis of scores ──
    print("\n=== Positional Analysis ===")
    all_positions = []
    all_ac = []
    all_h2o = []
    all_sel = []
    all_tc = []

    for prob in valid_problems:
        s = prob["scores"]
        r_len = s["reasoning_len"]
        rel_pos = np.arange(r_len) / (r_len - 1) if r_len > 1 else np.array([0.5])
        all_positions.extend(rel_pos.tolist())
        all_ac.extend(s["ac_score"].tolist())
        all_h2o.extend(s["h2o_score"].tolist())
        all_sel.extend(s["sel_score"].tolist())
        all_tc.extend(s["tc_score"].tolist())

    all_positions = np.array(all_positions)
    all_ac = np.array(all_ac)
    all_h2o = np.array(all_h2o)
    all_sel = np.array(all_sel)
    all_tc = np.array(all_tc)

    pos_ac_rho, _ = scipy_stats.spearmanr(all_positions, all_ac)
    pos_h2o_rho, _ = scipy_stats.spearmanr(all_positions, all_h2o)
    pos_sel_rho, _ = scipy_stats.spearmanr(all_positions, all_sel)
    pos_tc_rho, _ = scipy_stats.spearmanr(all_positions, all_tc)

    print(f"Position-score Spearman correlations (n={len(all_positions)}):")
    print(f"  Position vs AC:  rho={pos_ac_rho:.3f}")
    print(f"  Position vs H2O: rho={pos_h2o_rho:.3f}")
    print(f"  Position vs SEL: rho={pos_sel_rho:.3f}")
    print(f"  Position vs TC:  rho={pos_tc_rho:.3f}")

    positional_analysis = {
        "n_positions": len(all_positions),
        "pos_ac_rho": float(pos_ac_rho),
        "pos_h2o_rho": float(pos_h2o_rho),
        "pos_sel_rho": float(pos_sel_rho),
        "pos_tc_rho": float(pos_tc_rho),
    }

    # ── Phase 2: Protection strategy sweep ──
    print("\n=== Phase 2: Protection strategy comparison ===")

    condition_results = {}
    for nf in NOISE_FRACTIONS:
        for strat in STRATEGIES:
            key = f"{strat}_{nf}"
            condition_results[key] = {"correct": 0, "total": 0, "per_problem": []}
    condition_results["baseline"] = {"correct": 0, "total": 0, "per_problem": []}
    condition_results["full_noise"] = {"correct": 0, "total": 0, "per_problem": []}

    # Track noise set statistics
    noise_set_stats = {
        nf: {strat: [] for strat in STRATEGIES}
        for nf in NOISE_FRACTIONS
    }

    for pi, prob in enumerate(valid_problems):
        elapsed = time.time() - start_time
        if elapsed > 1650:  # 27.5 min hard limit
            print(f"Time limit at problem {pi}")
            break

        print(f"\n  Problem {pi+1}/{len(valid_problems)} (#{prob['ds_idx']})")

        try:
            base_kv, last_token_id, seq_len = build_kv_cache(
                model, tokenizer, prob["prompt"], prob["reasoning_text"]
            )
        except torch.cuda.OutOfMemoryError:
            print("  SKIP: OOM during KV cache build")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        scores = prob["scores"]
        prompt_len = scores["prompt_len"]
        reasoning_len = scores["reasoning_len"]

        if seq_len != scores["seq_len"]:
            print(f"  WARN: seq_len mismatch (kv={seq_len}, scores={scores['seq_len']})")
            seq_len = min(seq_len, scores["seq_len"])

        # ── Baseline ──
        _, baseline_answer = noise_and_generate(
            model, tokenizer, base_kv, [],
            prompt_len, seq_len, num_layers, last_token_id
        )
        baseline_norm = normalize_answer(baseline_answer) if baseline_answer else ""
        baseline_correct = (baseline_norm == prob["true_answer"])
        condition_results["baseline"]["correct"] += int(baseline_correct)
        condition_results["baseline"]["total"] += 1
        condition_results["baseline"]["per_problem"].append({
            "ds_idx": prob["ds_idx"], "correct": baseline_correct, "answer": baseline_answer
        })
        print(f"    Baseline: '{baseline_answer}' (correct: {baseline_correct})")

        if not baseline_correct:
            print("    WARN: KV-cache baseline incorrect — skip all conditions")
            for nf in NOISE_FRACTIONS:
                for strat in STRATEGIES:
                    key = f"{strat}_{nf}"
                    condition_results[key]["total"] += 1
                    condition_results[key]["per_problem"].append({
                        "ds_idx": prob["ds_idx"], "correct": False,
                        "answer": "", "note": "baseline_failed"
                    })
            condition_results["full_noise"]["total"] += 1
            condition_results["full_noise"]["per_problem"].append({
                "ds_idx": prob["ds_idx"], "correct": False,
                "answer": "", "note": "baseline_failed"
            })
            del base_kv
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # ── Full noise ──
        all_positions_list = list(range(reasoning_len))
        _, full_noise_answer = noise_and_generate(
            model, tokenizer, base_kv, all_positions_list,
            prompt_len, seq_len, num_layers, last_token_id
        )
        full_norm = normalize_answer(full_noise_answer) if full_noise_answer else ""
        full_correct = (full_norm == prob["true_answer"])
        condition_results["full_noise"]["correct"] += int(full_correct)
        condition_results["full_noise"]["total"] += 1
        condition_results["full_noise"]["per_problem"].append({
            "ds_idx": prob["ds_idx"], "correct": full_correct, "answer": full_noise_answer
        })

        # ── Strategy × noise fraction sweep ──
        for nf in NOISE_FRACTIONS:
            for strat in STRATEGIES:
                if strat == "random":
                    random.seed(SEED + prob["ds_idx"] * 100 + int(nf * 1000))

                key = f"{strat}_{nf}"
                positions_to_noise = select_positions_to_noise(
                    scores, reasoning_len, nf, strat
                )

                # Track noise set composition and position statistics
                noised_ac = scores["ac_score"][positions_to_noise]
                noised_h2o = scores["h2o_score"][positions_to_noise]
                noised_sel = scores["sel_score"][positions_to_noise]
                rel_positions = np.array(positions_to_noise) / (reasoning_len - 1) if reasoning_len > 1 else np.array([0.5])

                noise_set_stats[nf][strat].append({
                    "mean_ac": float(noised_ac.mean()),
                    "mean_h2o": float(noised_h2o.mean()),
                    "mean_sel": float(noised_sel.mean()),
                    "mean_rel_position": float(rel_positions.mean()),
                    "n_noised": len(positions_to_noise),
                    "position_std": float(rel_positions.std()) if len(rel_positions) > 1 else 0.0,
                })

                _, answer = noise_and_generate(
                    model, tokenizer, base_kv, positions_to_noise,
                    prompt_len, seq_len, num_layers, last_token_id
                )
                ans_norm = normalize_answer(answer) if answer else ""
                is_correct = (ans_norm == prob["true_answer"])
                condition_results[key]["correct"] += int(is_correct)
                condition_results[key]["total"] += 1
                condition_results[key]["per_problem"].append({
                    "ds_idx": prob["ds_idx"],
                    "correct": is_correct,
                    "answer": answer,
                    "n_noised": len(positions_to_noise),
                })

            # Print row for this noise fraction
            row = []
            for strat in STRATEGIES:
                key = f"{strat}_{nf}"
                cr = condition_results[key]
                acc = cr["correct"] / cr["total"] * 100 if cr["total"] > 0 else 0
                row.append(f"{strat}:{acc:.0f}%")
            print(f"    NF={nf}: {', '.join(row)}")

        del base_kv
        gc.collect()
        torch.cuda.empty_cache()

    # ── Compute summary ──
    accuracy_summary = {}
    for key, cr in condition_results.items():
        if cr["total"] > 0:
            accuracy_summary[key] = {
                "accuracy": cr["correct"] / cr["total"],
                "n_correct": cr["correct"],
                "n_total": cr["total"],
            }

    # Noise set composition summary
    noise_composition = {}
    for nf in NOISE_FRACTIONS:
        noise_composition[str(nf)] = {}
        for strat in STRATEGIES:
            if noise_set_stats[nf][strat]:
                stats = noise_set_stats[nf][strat]
                noise_composition[str(nf)][strat] = {
                    "mean_ac_of_noised": float(np.mean([s["mean_ac"] for s in stats])),
                    "mean_h2o_of_noised": float(np.mean([s["mean_h2o"] for s in stats])),
                    "mean_sel_of_noised": float(np.mean([s["mean_sel"] for s in stats])),
                    "mean_rel_position": float(np.mean([s["mean_rel_position"] for s in stats])),
                    "mean_position_std": float(np.mean([s["position_std"] for s in stats])),
                }

    # ── Print summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY (n={len(valid_problems)})")
    print(f"{'='*70}")
    print(f"{'Condition':<20} {'Accuracy':>10} {'n_correct':>10} {'n_total':>10}")
    print("-" * 50)
    for key in ["baseline", "full_noise"] + [f"{s}_{nf}" for nf in NOISE_FRACTIONS for s in STRATEGIES]:
        if key in accuracy_summary:
            v = accuracy_summary[key]
            print(f"{key:<20} {v['accuracy']:>9.1%} {v['n_correct']:>10} {v['n_total']:>10}")

    print("\nNoise set mean relative position (0=start, 1=end of reasoning):")
    for nf in NOISE_FRACTIONS:
        row = []
        for strat in STRATEGIES:
            nc = noise_composition.get(str(nf), {}).get(strat, {})
            if nc:
                row.append(f"{strat}:{nc['mean_rel_position']:.3f}")
        print(f"  NF={nf}: {', '.join(row)}")

    print("\nNoise set mean AC score:")
    for nf in NOISE_FRACTIONS:
        row = []
        for strat in STRATEGIES:
            nc = noise_composition.get(str(nf), {}).get(strat, {})
            if nc:
                row.append(f"{strat}:{nc['mean_ac_of_noised']:.5f}")
        print(f"  NF={nf}: {', '.join(row)}")

    print("\nNoise set mean selectivity:")
    for nf in NOISE_FRACTIONS:
        row = []
        for strat in STRATEGIES:
            nc = noise_composition.get(str(nf), {}).get(strat, {})
            if nc:
                row.append(f"{strat}:{nc['mean_sel_of_noised']:.1f}")
        print(f"  NF={nf}: {', '.join(row)}")

    # ── Save results ──
    results_data = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "n_valid": len(valid_problems),
        "accuracy_by_condition": accuracy_summary,
        "noise_composition": noise_composition,
        "positional_analysis": positional_analysis,
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(results_data, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
    print(f"\nResults saved to {RESULTS_DIR}/summary.json")

    # ── Generate figures ──
    print("\n=== Generating figures ===")
    generate_figures(results_data, noise_composition, positional_analysis, valid_problems)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")


def generate_figures(results_data, noise_composition, positional_analysis, valid_problems):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc_data = results_data["accuracy_by_condition"]
    colors = {"sel": "#9b59b6", "ac": "#e74c3c", "h2o": "#3498db", "tc": "#2ecc71", "random": "#95a5a6"}
    markers = {"sel": "P", "ac": "o", "h2o": "s", "tc": "^", "random": "D"}

    # ── Figure 1: Accuracy vs noise fraction by strategy ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for strat in STRATEGIES:
        fractions = []
        accuracies = []
        for nf in NOISE_FRACTIONS:
            key = f"{strat}_{nf}"
            if key in acc_data:
                fractions.append(nf * 100)
                accuracies.append(acc_data[key]["accuracy"] * 100)
        if fractions:
            label = strat.upper()
            if strat == "sel":
                label = "SEL (selectivity)"
            ax.plot(fractions, accuracies, f"{markers[strat]}-", color=colors[strat],
                    label=f"{label} protect", linewidth=2.5 if strat == "sel" else 2, markersize=10)

    if "baseline" in acc_data:
        ax.axhline(y=acc_data["baseline"]["accuracy"] * 100, color="black",
                   linestyle="--", linewidth=1, label="Baseline (clean)", alpha=0.5)
    if "full_noise" in acc_data:
        ax.axhline(y=acc_data["full_noise"]["accuracy"] * 100, color="gray",
                   linestyle=":", linewidth=1, label="Full noise", alpha=0.5)

    ax.set_xlabel("Noise Fraction (% of reasoning positions noised)", fontsize=12)
    ax.set_ylabel("Answer Accuracy (%)", fontsize=12)
    n_valid = results_data.get("n_valid", "?")
    ax.set_title(f"Llama-3.1-8B: Protection Strategy Comparison (n={n_valid})\nDoes selectivity rescue the AC/TC framework?", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "protection_accuracy_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: protection_accuracy_sweep.png")

    # ── Figure 2: Grouped bar chart at key noise fractions ──
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    key_fractions = [0.01, 0.03, 0.05]
    x = np.arange(len(key_fractions))
    width = 0.8 / len(STRATEGIES)

    for si, strat in enumerate(STRATEGIES):
        accuracies = []
        for nf in key_fractions:
            key = f"{strat}_{nf}"
            if key in acc_data:
                accuracies.append(acc_data[key]["accuracy"] * 100)
            else:
                accuracies.append(0)
        offset = (si - len(STRATEGIES) / 2 + 0.5) * width
        label = strat.upper()
        if strat == "sel":
            label = "SEL (selectivity)"
        ax2.bar(x + offset, accuracies, width, label=label,
                color=colors[strat], edgecolor="black", linewidth=0.5)

    ax2.set_xlabel("Noise Fraction", fontsize=12)
    ax2.set_ylabel("Answer Accuracy (%)", fontsize=12)
    ax2.set_title(f"Llama-3.1-8B: Strategy Comparison at Key Noise Fractions (n={n_valid})", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{nf*100:.0f}%" for nf in key_fractions])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_ylim(0, 105)
    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "strategy_comparison_bars.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: strategy_comparison_bars.png")

    # ── Figure 3: Positional analysis — where does each strategy noise? ──
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for strat in STRATEGIES:
        fractions = []
        positions = []
        for nf in NOISE_FRACTIONS:
            nc = noise_composition.get(str(nf), {}).get(strat, {})
            if nc:
                fractions.append(nf * 100)
                positions.append(nc["mean_rel_position"])
        if fractions:
            label = strat.upper()
            if strat == "sel":
                label = "SEL (selectivity)"
            ax3.plot(fractions, positions, f"{markers[strat]}-", color=colors[strat],
                    label=label, linewidth=2, markersize=8)

    ax3.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Middle of sequence")
    ax3.set_xlabel("Noise Fraction (%)", fontsize=12)
    ax3.set_ylabel("Mean Relative Position of Noised Positions\n(0=start, 1=end of reasoning)", fontsize=11)
    ax3.set_title("Where in the Sequence Does Each Strategy Noise?", fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.2)
    ax3.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig3.savefig(os.path.join(RESULTS_DIR, "noise_position_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: noise_position_analysis.png")

    # ── Figure 4: Score vs position scatter (pooled across problems) ──
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))

    # Subsample for plotting (max 2000 points)
    all_pos = []
    all_scores = {"ac": [], "h2o": [], "tc": [], "sel": []}
    for prob in valid_problems:
        s = prob["scores"]
        r_len = s["reasoning_len"]
        rel_pos = np.arange(r_len) / (r_len - 1) if r_len > 1 else np.array([0.5])
        all_pos.extend(rel_pos.tolist())
        all_scores["ac"].extend(s["ac_score"].tolist())
        all_scores["h2o"].extend(s["h2o_score"].tolist())
        all_scores["tc"].extend(s["tc_score"].tolist())
        all_scores["sel"].extend(s["sel_score"].tolist())

    n_pts = len(all_pos)
    max_pts = 2000
    if n_pts > max_pts:
        idx = np.random.choice(n_pts, max_pts, replace=False)
    else:
        idx = np.arange(n_pts)

    all_pos = np.array(all_pos)
    for (ax_r, score_name, title, rho_val) in [
        (axes4[0, 0], "ac", "AC Score", positional_analysis["pos_ac_rho"]),
        (axes4[0, 1], "h2o", "H2O Score", positional_analysis["pos_h2o_rho"]),
        (axes4[1, 0], "tc", "TC Score", positional_analysis["pos_tc_rho"]),
        (axes4[1, 1], "sel", "Selectivity", positional_analysis["pos_sel_rho"]),
    ]:
        score_arr = np.array(all_scores[score_name])
        ax_r.scatter(all_pos[idx], score_arr[idx], alpha=0.15, s=8, color=colors.get(score_name, "gray"))
        ax_r.set_xlabel("Relative Position (0=start, 1=end)")
        ax_r.set_ylabel(title)
        ax_r.set_title(f"{title} vs Position (rho={rho_val:.3f})")
        ax_r.grid(True, alpha=0.2)

    plt.suptitle("Score-Position Relationships (Positional Confound Analysis)", fontsize=14)
    plt.tight_layout()
    fig4.savefig(os.path.join(RESULTS_DIR, "score_position_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: score_position_scatter.png")

    # ── Figure 5: SEL advantage over other strategies ──
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    for compare_strat in ["ac", "h2o", "tc", "random"]:
        fractions = []
        diffs = []
        for nf in NOISE_FRACTIONS:
            sel_key = f"sel_{nf}"
            cmp_key = f"{compare_strat}_{nf}"
            if sel_key in acc_data and cmp_key in acc_data:
                fractions.append(nf * 100)
                diff = (acc_data[sel_key]["accuracy"] - acc_data[cmp_key]["accuracy"]) * 100
                diffs.append(diff)
        if fractions:
            ax5.plot(fractions, diffs, f"{markers[compare_strat]}-", color=colors[compare_strat],
                    label=f"SEL - {compare_strat.upper()}", linewidth=2, markersize=8)

    ax5.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax5.set_xlabel("Noise Fraction (%)", fontsize=12)
    ax5.set_ylabel("SEL Advantage (percentage points)", fontsize=12)
    ax5.set_title("Selectivity Protection Advantage Over Other Strategies", fontsize=13)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.2)
    plt.tight_layout()
    fig5.savefig(os.path.join(RESULTS_DIR, "sel_advantage.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: sel_advantage.png")


if __name__ == "__main__":
    main()
