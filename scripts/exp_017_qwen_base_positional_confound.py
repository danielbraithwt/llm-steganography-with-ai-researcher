#!/usr/bin/env python3
"""
Experiment 017: Positional Confound Test on Qwen3-4B-Base

Tests whether the +23.5pp destruction dissociation found in exp_004 is a genuine
spatial structure effect or a positional confound (as exp_013/016 showed on instruct
models).

Key question (Q49): Does the original destruction dissociation on Qwen-Base survive
positional controls?

Design:
1. Replicate exp_004 destruction (SelAC, SelTC, Random) at 5%, 10% noise
2. Add positional strategies (POS_EARLY, POS_LATE)
3. Position-controlled within-half analysis (SelAC vs SelTC within early/late halves)
4. Position-score correlations and mean noise position recording
5. Uses SAME model, seed, problems, and noise method as exp_004 for direct comparison

Noise method: REPLACEMENT (norm-matched Gaussian) — same as exp_004
Model: Qwen/Qwen3-4B-Base
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
MODEL_NAME = "Qwen/Qwen3-4B-Base"
NUM_PROBLEMS = 40  # aim for ~34 valid (matching exp_004)
NOISE_FRACTIONS = [0.05, 0.10]  # 5% is where exp_004 found +23.5pp; 10% for robustness
ATTENTION_LAYERS = [-1, -2, -3, -4]  # last 4 layers (same as exp_004)
MAX_GEN_TOKENS = 768
SEED = 42  # same seed as exp_004
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_017")
TIME_LIMIT = 1700  # seconds — leave margin for analysis/figures

os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 8-shot GSM8K exemplars (same as exp_004) ──────────────────────────
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
def generate_trace(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS):
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
            if re.search(r'\d+\s*\n', after):
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
def teacher_force_with_attention(model, tokenizer, prompt, trace_text):
    full_text = prompt + trace_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    reasoning_len = seq_len - prompt_len

    if reasoning_len < 5:
        return None

    if seq_len > 2048:
        max_reasoning = 2048 - prompt_len
        trace_tokens = tokenizer(trace_text, return_tensors="pt").input_ids[0][:max_reasoning]
        trace_text = tokenizer.decode(trace_tokens, skip_special_tokens=True)
        full_text = prompt + trace_text
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]
        reasoning_len = seq_len - prompt_len

    outputs = model(**inputs, output_attentions=True, use_cache=False)

    num_layers = len(outputs.attentions)
    layer_indices = [num_layers + i for i in ATTENTION_LAYERS]
    last_token_idx = seq_len - 1

    answer_coupling = torch.zeros(reasoning_len, device=model.device)
    text_coupling = torch.zeros(reasoning_len, device=model.device)

    for li in layer_indices:
        attn = outputs.attentions[li][0]  # (num_heads, seq_len, seq_len)

        # Answer coupling: attention from last token to each reasoning position
        answer_coupling += attn[:, last_token_idx, prompt_len:seq_len].sum(dim=0)

        # Text coupling: average attention from later reasoning tokens to each position
        reasoning_attn = attn[:, prompt_len:seq_len, prompt_len:seq_len]
        reasoning_attn_sum = reasoning_attn.sum(dim=0)
        mask = torch.tril(torch.ones(reasoning_len, reasoning_len, device=model.device), diagonal=-1)
        weighted = reasoning_attn_sum * mask
        col_sums = weighted.sum(dim=0)
        col_counts = mask.sum(dim=0).clamp(min=1)
        text_coupling += col_sums / col_counts

    num_layers_used = len(layer_indices)
    num_heads = attn.shape[0]
    answer_coupling = answer_coupling / (num_layers_used * num_heads)
    text_coupling = text_coupling / (num_layers_used * num_heads)

    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "answer_coupling": answer_coupling.cpu().numpy(),
        "text_coupling": text_coupling.cpu().numpy(),
        "prompt_len": prompt_len,
        "reasoning_len": reasoning_len,
        "seq_len": seq_len,
    }


def compute_selectivity(answer_coupling, text_coupling):
    n = len(answer_coupling)
    if n <= 1:
        return np.zeros(n)
    ac_ranks = np.argsort(np.argsort(answer_coupling)).astype(float) / (n - 1)
    tc_ranks = np.argsort(np.argsort(text_coupling)).astype(float) / (n - 1)
    selectivity = ac_ranks - tc_ranks
    return selectivity


@torch.no_grad()
def ablated_answer_generation(model, tokenizer, prompt, reasoning_text,
                               positions_to_ablate, prompt_len):
    """
    Teacher-force prompt + reasoning, REPLACE specific KV positions with norm-matched
    Gaussian noise (same method as exp_004), generate answer.
    """
    # Truncate reasoning at "####" if present
    if "####" in reasoning_text:
        reasoning_text = reasoning_text[:reasoning_text.index("####")]

    full_text = prompt + reasoning_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]

    outputs = model(**inputs, use_cache=True, output_attentions=False)
    past_kv = outputs.past_key_values
    num_layers = len(past_kv.layers)

    reasoning_len = seq_len - prompt_len
    lookback = min(20, reasoning_len)
    lookback_start = seq_len - lookback

    # Ablate KV cache — REPLACEMENT noise (norm-matched)
    abs_positions = [prompt_len + p for p in positions_to_ablate if prompt_len + p < seq_len]

    if abs_positions:
        for layer_idx in range(num_layers):
            layer = past_kv.layers[layer_idx]
            for pos in abs_positions:
                # Replacement: replace with random noise scaled to same norm
                k_norm = layer.keys[:, :, pos, :].norm().item()
                v_norm = layer.values[:, :, pos, :].norm().item()
                k_shape = layer.keys[:, :, pos, :].shape
                v_shape = layer.values[:, :, pos, :].shape
                k_noise = torch.randn(k_shape, device=layer.keys.device, dtype=layer.keys.dtype)
                v_noise = torch.randn(v_shape, device=layer.values.device, dtype=layer.values.dtype)
                k_noise = k_noise * (k_norm / (k_noise.norm().item() + 1e-8))
                v_noise = v_noise * (v_norm / (v_noise.norm().item() + 1e-8))
                layer.keys[:, :, pos, :] = k_noise
                layer.values[:, :, pos, :] = v_noise

    # Re-process lookback tokens through ablated cache
    lookback_tokens = inputs.input_ids[:, lookback_start:seq_len]

    trunc_kv = DynamicCache()
    for layer_idx in range(num_layers):
        layer = past_kv.layers[layer_idx]
        key = layer.keys[:, :, :lookback_start, :].clone()
        value = layer.values[:, :, :lookback_start, :].clone()
        trunc_kv.update(key, value, layer_idx)

    lookback_outputs = model(
        input_ids=lookback_tokens,
        past_key_values=trunc_kv,
        use_cache=True,
        output_attentions=False,
    )

    # Generate answer
    gen_kv = lookback_outputs.past_key_values
    next_token_logits = lookback_outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_ids = [next_token[0, 0].item()]

    for _ in range(150):
        gen_out = model(input_ids=next_token, past_key_values=gen_kv, use_cache=True)
        gen_kv = gen_out.past_key_values
        next_token = torch.argmax(gen_out.logits[:, -1, :], dim=-1, keepdim=True)
        token_id = next_token[0, 0].item()
        generated_ids.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "####" in decoded:
            after = decoded.split("####")[-1]
            if re.search(r'\d+\s*\n', after):
                break
        if "\nQ:" in decoded:
            break

    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    del outputs, lookback_outputs, past_kv, gen_kv, trunc_kv
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "answer_text": answer_text,
        "answer": extract_answer(answer_text),
    }


def get_strategy_positions(strategy, selectivity, reasoning_len, n_prune):
    """
    Return (positions_list, mean_relative_position) for a given strategy.
    """
    positions = np.arange(reasoning_len)
    rel_positions = positions / max(reasoning_len - 1, 1)

    if strategy == "selac":
        # Top selectivity (most answer-coupled relative to text-coupled)
        idx = np.argsort(selectivity)[-n_prune:]
    elif strategy == "seltc":
        # Bottom selectivity (most text-coupled relative to answer-coupled)
        idx = np.argsort(selectivity)[:n_prune]
    elif strategy == "random":
        idx = np.array(random.sample(range(reasoning_len), n_prune))
    elif strategy == "pos_early":
        idx = np.arange(n_prune)
    elif strategy == "pos_late":
        idx = np.arange(reasoning_len - n_prune, reasoning_len)
    elif strategy == "early_half_selac":
        half = reasoning_len // 2
        if half < n_prune:
            idx = np.arange(half)
        else:
            early_sel = selectivity[:half]
            local_idx = np.argsort(early_sel)[-n_prune:]
            idx = local_idx  # already in [0, half) range
    elif strategy == "early_half_seltc":
        half = reasoning_len // 2
        if half < n_prune:
            idx = np.arange(half)
        else:
            early_sel = selectivity[:half]
            local_idx = np.argsort(early_sel)[:n_prune]
            idx = local_idx
    elif strategy == "late_half_selac":
        half = reasoning_len // 2
        start = reasoning_len - half
        if half < n_prune:
            idx = np.arange(start, reasoning_len)
        else:
            late_sel = selectivity[start:]
            local_idx = np.argsort(late_sel)[-n_prune:]
            idx = local_idx + start  # offset to global position
    elif strategy == "late_half_seltc":
        half = reasoning_len // 2
        start = reasoning_len - half
        if half < n_prune:
            idx = np.arange(start, reasoning_len)
        else:
            late_sel = selectivity[start:]
            local_idx = np.argsort(late_sel)[:n_prune]
            idx = local_idx + start
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    idx = np.array(idx)
    mean_rel_pos = float(np.mean(rel_positions[idx])) if len(idx) > 0 else 0.5
    return idx.tolist(), mean_rel_pos


def main():
    start_time = time.time()
    print(f"{'='*70}")
    print(f"Experiment 017: Positional Confound Test on Qwen3-4B-Base")
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {NUM_PROBLEMS}")
    print(f"Noise fractions: {NOISE_FRACTIONS}")
    print(f"Noise type: REPLACEMENT (norm-matched, same as exp_004)")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded in {time.time() - start_time:.1f}s")

    # Load GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(dataset)))
    random.shuffle(indices)  # same shuffle as exp_004 with SEED=42
    selected = indices[:NUM_PROBLEMS]

    # ── Phase 1: Generate traces and compute scores ─────────────────────
    print("\n=== Phase 1: Generating traces and computing AC/TC/selectivity ===")
    problems_data = []

    for prob_idx, ds_idx in enumerate(selected):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.5:  # use half time for phase 1
            print(f"Phase 1 time budget reached at problem {prob_idx}")
            break

        problem = dataset[ds_idx]
        question = problem["question"]
        true_answer = normalize_answer(problem["answer"].split("####")[-1].strip()
                                       .replace(",", "").replace("$", ""))
        prompt = build_prompt(question)

        print(f"\nProblem {prob_idx+1}/{NUM_PROBLEMS} (#{ds_idx}), true={true_answer}")

        trace_text = generate_trace(model, tokenizer, prompt)
        gen_answer = extract_answer(trace_text)
        gen_norm = normalize_answer(gen_answer) if gen_answer else ""
        correct = (gen_norm == true_answer)
        print(f"  Generated: '{gen_answer}' (correct: {correct})")

        if not correct:
            print("  SKIP: baseline incorrect")
            continue

        if "####" in trace_text:
            reasoning_text = trace_text[:trace_text.index("####")]
        else:
            reasoning_text = trace_text

        try:
            attn_info = teacher_force_with_attention(model, tokenizer, prompt, reasoning_text)
        except torch.cuda.OutOfMemoryError:
            print("  SKIP: OOM during attention extraction")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        if attn_info is None:
            print("  SKIP: trace too short")
            continue

        selectivity = compute_selectivity(attn_info["answer_coupling"], attn_info["text_coupling"])

        # Compute position-score correlations for this problem
        positions = np.arange(attn_info["reasoning_len"])
        if len(positions) > 5:
            sel_pos_rho = scipy_stats.spearmanr(positions, selectivity).statistic
            ac_pos_rho = scipy_stats.spearmanr(positions, attn_info["answer_coupling"]).statistic
            tc_pos_rho = scipy_stats.spearmanr(positions, attn_info["text_coupling"]).statistic
        else:
            sel_pos_rho = ac_pos_rho = tc_pos_rho = 0.0

        problems_data.append({
            "ds_idx": ds_idx,
            "true_answer": true_answer,
            "prompt": prompt,
            "reasoning_text": reasoning_text,
            "answer_coupling": attn_info["answer_coupling"],
            "text_coupling": attn_info["text_coupling"],
            "selectivity": selectivity,
            "prompt_len": attn_info["prompt_len"],
            "reasoning_len": attn_info["reasoning_len"],
            "sel_pos_rho": sel_pos_rho,
            "ac_pos_rho": ac_pos_rho,
            "tc_pos_rho": tc_pos_rho,
        })
        print(f"  Valid! reasoning_len={attn_info['reasoning_len']}, "
              f"SEL-pos rho={sel_pos_rho:.3f}, AC-pos rho={ac_pos_rho:.3f}, TC-pos rho={tc_pos_rho:.3f}")

    n_valid = len(problems_data)
    print(f"\n=== Phase 1 complete: {n_valid} valid problems ===")

    if n_valid < 3:
        print("ERROR: Too few valid problems!")
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump({"error": "too_few_valid", "n_valid": n_valid}, f, indent=2)
        return

    # Report aggregate position-score correlations
    all_positions = []
    all_sel = []
    all_ac = []
    all_tc = []
    for pd in problems_data:
        n_r = pd["reasoning_len"]
        pos = np.arange(n_r) / max(n_r - 1, 1)  # normalize to [0, 1]
        all_positions.extend(pos.tolist())
        all_sel.extend(pd["selectivity"].tolist())
        all_ac.extend(pd["answer_coupling"].tolist())
        all_tc.extend(pd["text_coupling"].tolist())

    pooled_sel_rho = scipy_stats.spearmanr(all_positions, all_sel).statistic
    pooled_ac_rho = scipy_stats.spearmanr(all_positions, all_ac).statistic
    pooled_tc_rho = scipy_stats.spearmanr(all_positions, all_tc).statistic

    print(f"\nPooled position-score correlations (n={len(all_positions)} positions):")
    print(f"  SEL-position rho = {pooled_sel_rho:+.3f}")
    print(f"  AC-position rho  = {pooled_ac_rho:+.3f}")
    print(f"  TC-position rho  = {pooled_tc_rho:+.3f}")

    # Per-problem summary
    mean_sel_rho = np.mean([pd["sel_pos_rho"] for pd in problems_data])
    mean_ac_rho = np.mean([pd["ac_pos_rho"] for pd in problems_data])
    mean_tc_rho = np.mean([pd["tc_pos_rho"] for pd in problems_data])
    print(f"\nPer-problem mean position-score correlations:")
    print(f"  Mean SEL-position rho = {mean_sel_rho:+.3f}")
    print(f"  Mean AC-position rho  = {mean_ac_rho:+.3f}")
    print(f"  Mean TC-position rho  = {mean_tc_rho:+.3f}")

    # ── Phase 2: Control test (baseline pipeline check) ─────────────────
    print("\n=== Phase 2: Pipeline control test (3 problems, no noise) ===")
    control_correct = 0
    control_total = min(3, n_valid)
    for ci in range(control_total):
        pd = problems_data[ci]
        result = ablated_answer_generation(
            model, tokenizer, pd["prompt"], pd["reasoning_text"],
            [],  # no positions to ablate
            pd["prompt_len"],
        )
        ans_norm = normalize_answer(result["answer"]) if result["answer"] else ""
        is_correct = (ans_norm == pd["true_answer"])
        control_correct += int(is_correct)
        print(f"  Control {ci+1}: expected={pd['true_answer']}, got={result['answer']}, correct={is_correct}")

    pipeline_baseline = control_correct / control_total if control_total > 0 else 0
    print(f"Pipeline control: {control_correct}/{control_total} ({pipeline_baseline:.0%})")

    # ── Phase 3: Ablation sweep with all strategies ─────────────────────
    print("\n=== Phase 3: Ablation sweep with positional controls ===")

    STRATEGIES = [
        "selac", "seltc", "random",
        "pos_early", "pos_late",
        "early_half_selac", "early_half_seltc",
        "late_half_selac", "late_half_seltc",
    ]

    sweep_results = {}

    for frac in NOISE_FRACTIONS:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT:
            print(f"Time budget reached before fraction {frac}")
            break

        frac_key = f"noise_{frac}"
        print(f"\n--- REPLACEMENT NOISE at {frac:.0%} ---")
        sweep_results[frac_key] = {}

        for strategy in STRATEGIES:
            sweep_results[frac_key][strategy] = []

        for pi, pd in enumerate(problems_data):
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT:
                print(f"Time budget reached at problem {pi}")
                break

            n_prune = max(1, int(pd["reasoning_len"] * frac))

            for strategy in STRATEGIES:
                positions, mean_pos = get_strategy_positions(
                    strategy, pd["selectivity"], pd["reasoning_len"], n_prune
                )
                try:
                    result = ablated_answer_generation(
                        model, tokenizer, pd["prompt"], pd["reasoning_text"],
                        positions, pd["prompt_len"],
                    )
                    ans_norm = normalize_answer(result["answer"]) if result["answer"] else ""
                    correct = (ans_norm == pd["true_answer"])
                    sweep_results[frac_key][strategy].append({
                        "ds_idx": pd["ds_idx"],
                        "correct": correct,
                        "answer": result["answer"],
                        "mean_pos": mean_pos,
                    })
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    sweep_results[frac_key][strategy].append({
                        "ds_idx": pd["ds_idx"],
                        "error": "OOM",
                        "mean_pos": mean_pos,
                    })

            if (pi + 1) % 5 == 0 or pi == len(problems_data) - 1:
                parts = []
                for s in STRATEGIES[:5]:  # show first 5
                    s_correct = sum(1 for r in sweep_results[frac_key][s] if r.get("correct"))
                    parts.append(f"{s}={s_correct}/{pi+1}")
                print(f"  [{pi+1}/{n_valid}] {frac:.0%}: {', '.join(parts)}")

        # Print summary for this fraction
        print(f"\n  Summary at {frac:.0%}:")
        for s in STRATEGIES:
            valid = [r for r in sweep_results[frac_key][s] if "error" not in r]
            correct_count = sum(1 for r in valid if r["correct"])
            n = len(valid)
            acc = correct_count / n if n > 0 else 0
            mean_pos = np.mean([r["mean_pos"] for r in sweep_results[frac_key][s]]) if sweep_results[frac_key][s] else 0
            print(f"    {s:<20s}: {correct_count}/{n} ({acc:.1%}), mean_pos={mean_pos:.3f}")

    # ── Analysis ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    summary = {
        "experiment": "exp_017_qwen_base_positional_confound",
        "model": MODEL_NAME,
        "noise_type": "replacement (norm-matched)",
        "num_valid": n_valid,
        "num_attempted": NUM_PROBLEMS,
        "pipeline_control": f"{control_correct}/{control_total}",
        "noise_fractions": NOISE_FRACTIONS,
        "strategies": STRATEGIES,
        "position_correlations": {
            "pooled": {
                "n_positions": len(all_positions),
                "sel_pos_rho": float(pooled_sel_rho),
                "ac_pos_rho": float(pooled_ac_rho),
                "tc_pos_rho": float(pooled_tc_rho),
            },
            "per_problem_mean": {
                "sel_pos_rho": float(mean_sel_rho),
                "ac_pos_rho": float(mean_ac_rho),
                "tc_pos_rho": float(mean_tc_rho),
            },
        },
        "sweep": {},
    }

    for frac_key in sweep_results:
        summary["sweep"][frac_key] = {}
        for strategy in STRATEGIES:
            valid = [r for r in sweep_results[frac_key][strategy] if "error" not in r]
            correct_count = sum(1 for r in valid if r["correct"])
            n = len(valid)
            acc = correct_count / n if n > 0 else 0
            mean_pos = float(np.mean([r["mean_pos"] for r in sweep_results[frac_key][strategy]])) if sweep_results[frac_key][strategy] else 0
            summary["sweep"][frac_key][strategy] = {
                "accuracy": acc,
                "correct": correct_count,
                "total": n,
                "mean_noise_position": mean_pos,
            }

    # Dissociation analysis for each fraction
    print("\nDISSOCIATION ANALYSIS:")
    for frac_key in summary["sweep"]:
        data = summary["sweep"][frac_key]
        if "selac" in data and "seltc" in data:
            selac_acc = data["selac"]["accuracy"]
            seltc_acc = data["seltc"]["accuracy"]
            pos_early_acc = data.get("pos_early", {}).get("accuracy", -1)
            pos_late_acc = data.get("pos_late", {}).get("accuracy", -1)

            gap_unconstrained = seltc_acc - selac_acc  # positive = AC hurts more
            gap_positional = pos_late_acc - pos_early_acc if pos_early_acc >= 0 else None

            # Within-half gaps
            eh_selac = data.get("early_half_selac", {}).get("accuracy", -1)
            eh_seltc = data.get("early_half_seltc", {}).get("accuracy", -1)
            lh_selac = data.get("late_half_selac", {}).get("accuracy", -1)
            lh_seltc = data.get("late_half_seltc", {}).get("accuracy", -1)

            gap_early_half = (eh_seltc - eh_selac) if (eh_selac >= 0 and eh_seltc >= 0) else None
            gap_late_half = (lh_seltc - lh_selac) if (lh_selac >= 0 and lh_seltc >= 0) else None

            print(f"\n  {frac_key}:")
            print(f"    Unconstrained:   SelTC-SelAC = {gap_unconstrained:+.1%} (positive = AC more destructive)")
            print(f"    SelAC mean_pos = {data['selac']['mean_noise_position']:.3f}")
            print(f"    SelTC mean_pos = {data['seltc']['mean_noise_position']:.3f}")
            if gap_positional is not None:
                print(f"    Positional:      POS_LATE-POS_EARLY = {gap_positional:+.1%}")
            if gap_early_half is not None:
                print(f"    Early half:      SelTC-SelAC = {gap_early_half:+.1%}")
                print(f"      Early+SelAC mean_pos = {data['early_half_selac']['mean_noise_position']:.3f}")
                print(f"      Early+SelTC mean_pos = {data['early_half_seltc']['mean_noise_position']:.3f}")
            if gap_late_half is not None:
                print(f"    Late half:       SelTC-SelAC = {gap_late_half:+.1%}")
                print(f"      Late+SelAC mean_pos = {data['late_half_selac']['mean_noise_position']:.3f}")
                print(f"      Late+SelTC mean_pos = {data['late_half_seltc']['mean_noise_position']:.3f}")

            # Determine confound verdict
            if gap_positional is not None and abs(gap_positional) > 0.15:
                if gap_early_half is not None and abs(gap_early_half) < 0.10:
                    verdict = "POSITIONAL CONFOUND — within-half gap collapsed"
                elif gap_early_half is not None and abs(gap_early_half) >= 0.10:
                    verdict = f"GENUINE EFFECT — within-half gap persists ({gap_early_half:+.1%})"
                else:
                    verdict = "POSITIONAL CONFOUND LIKELY — large positional gap"
            elif gap_positional is not None and abs(gap_positional) < 0.10:
                verdict = "NO POSITIONAL CONFOUND — position alone doesn't predict accuracy"
            else:
                verdict = "INCONCLUSIVE"

            print(f"    VERDICT: {verdict}")

            summary["sweep"][frac_key]["dissociation"] = {
                "unconstrained_gap": float(gap_unconstrained),
                "positional_gap": float(gap_positional) if gap_positional is not None else None,
                "early_half_gap": float(gap_early_half) if gap_early_half is not None else None,
                "late_half_gap": float(gap_late_half) if gap_late_half is not None else None,
                "verdict": verdict,
            }

    summary["elapsed_seconds"] = time.time() - start_time

    # Save results
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_convert)

    # Save raw results (without numpy arrays)
    raw_results = {}
    for fk in sweep_results:
        raw_results[fk] = {}
        for s in sweep_results[fk]:
            raw_results[fk][s] = sweep_results[fk][s]
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(raw_results, f, indent=2, default=_convert)

    # ── Generate figures ────────────────────────────────────────────────
    generate_figures(summary, RESULTS_DIR)

    total_time = time.time() - start_time
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")


def generate_figures(summary, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = summary["sweep"]
    if not sweep:
        print("No data for figures")
        return

    n = summary["num_valid"]

    # Figure 1: Strategy comparison bars at 5% noise
    target_key = "noise_0.05"
    if target_key in sweep:
        data = sweep[target_key]
        strategies = list(data.keys())
        strategies = [s for s in strategies if s != "dissociation" and isinstance(data[s], dict) and "accuracy" in data[s]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        accs = [data[s]["accuracy"] * 100 for s in strategies]
        positions = [data[s]["mean_noise_position"] for s in strategies]

        colors = []
        for s in strategies:
            if s in ("selac", "early_half_selac", "late_half_selac"):
                colors.append("#e74c3c")  # red for AC
            elif s in ("seltc", "early_half_seltc", "late_half_seltc"):
                colors.append("#3498db")  # blue for TC
            elif s in ("pos_early",):
                colors.append("#e67e22")  # orange
            elif s in ("pos_late",):
                colors.append("#2ecc71")  # green
            else:
                colors.append("#95a5a6")  # gray for random

        x = np.arange(len(strategies))
        ax1.bar(x, accs, color=colors, edgecolor="black", linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Answer Accuracy (%)")
        ax1.set_title(f"Strategy Comparison at 5% Noise (n={n})")
        ax1.set_ylim(0, 110)
        ax1.grid(True, alpha=0.3)
        for i, v in enumerate(accs):
            ax1.text(i, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

        # Panel 2: Mean noise position vs accuracy (scatter)
        ax2.scatter(positions, accs, c=colors, s=100, edgecolor="black", linewidth=0.5, zorder=5)
        for i, s in enumerate(strategies):
            ax2.annotate(s, (positions[i], accs[i]), textcoords="offset points",
                        xytext=(5, 5), fontsize=7)
        ax2.set_xlabel("Mean Noise Position (0=start, 1=end)")
        ax2.set_ylabel("Answer Accuracy (%)")
        ax2.set_title("Position vs Accuracy")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Exp 017: Positional Confound Test — Qwen3-4B-Base", fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, "strategy_comparison_5pct.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Figure saved: strategy_comparison_5pct.png")

    # Figure 2: Within-half dissociation analysis
    fig2, axes = plt.subplots(1, len(sweep), figsize=(6 * len(sweep), 6))
    if len(sweep) == 1:
        axes = [axes]

    for ax, frac_key in zip(axes, sorted(sweep.keys())):
        data = sweep[frac_key]
        if "dissociation" not in data:
            continue
        diss = data["dissociation"]

        labels = ["Unconstrained\n(SelTC-SelAC)", "Positional\n(Late-Early)",
                  "Within Early\n(SelTC-SelAC)", "Within Late\n(SelTC-SelAC)"]
        gaps = [
            diss["unconstrained_gap"] * 100 if diss["unconstrained_gap"] is not None else 0,
            diss["positional_gap"] * 100 if diss["positional_gap"] is not None else 0,
            diss["early_half_gap"] * 100 if diss["early_half_gap"] is not None else 0,
            diss["late_half_gap"] * 100 if diss["late_half_gap"] is not None else 0,
        ]
        bar_colors = ["#9b59b6", "#e67e22", "#3498db", "#2ecc71"]

        x = np.arange(len(labels))
        bars = ax.bar(x, gaps, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Gap (percentage points)")
        frac_val = frac_key.split("_")[1]
        ax.set_title(f"{frac_key} — Dissociation Gaps")
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.2)
        for i, v in enumerate(gaps):
            ax.text(i, v + (1 if v >= 0 else -3), f"{v:+.1f}pp", ha="center", fontsize=9)

    fig2.suptitle(f"Exp 017: Dissociation Analysis (n={n})", fontsize=12)
    plt.tight_layout()
    fig2.savefig(os.path.join(results_dir, "dissociation_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: dissociation_analysis.png")

    # Figure 3: Position-score correlations
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    corr_data = summary["position_correlations"]["pooled"]
    metrics = ["sel_pos_rho", "ac_pos_rho", "tc_pos_rho"]
    metric_labels = ["SEL vs Position", "AC vs Position", "TC vs Position"]
    values = [corr_data[m] for m in metrics]
    bar_colors = ["#9b59b6", "#e74c3c", "#3498db"]

    x = np.arange(len(metrics))
    bars = ax3.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_labels, fontsize=10)
    ax3.set_ylabel("Spearman rho with Position")
    ax3.set_title(f"Position-Score Correlations (n={corr_data['n_positions']} positions)")
    ax3.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax3.grid(True, alpha=0.2)
    for i, v in enumerate(values):
        ax3.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:+.3f}", ha="center", fontsize=10)
    ax3.set_ylim(-1, 1)

    plt.tight_layout()
    fig3.savefig(os.path.join(results_dir, "position_score_correlations.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: position_score_correlations.png")

    # Figure 4: Accuracy sweep across fractions (if multiple fractions)
    if len(sweep) > 1:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        frac_keys_sorted = sorted(sweep.keys())
        fracs_vals = []
        strategy_accs = {s: [] for s in ["selac", "seltc", "random", "pos_early", "pos_late"]}

        for fk in frac_keys_sorted:
            frac_val = float(fk.split("_")[1]) * 100
            fracs_vals.append(frac_val)
            for s in strategy_accs:
                if s in sweep[fk] and isinstance(sweep[fk][s], dict):
                    strategy_accs[s].append(sweep[fk][s]["accuracy"] * 100)
                else:
                    strategy_accs[s].append(None)

        style_map = {
            "selac": ("o-", "#e74c3c", "SelAC"),
            "seltc": ("s-", "#3498db", "SelTC"),
            "random": ("^-", "#95a5a6", "Random"),
            "pos_early": ("D-", "#e67e22", "POS_EARLY"),
            "pos_late": ("v-", "#2ecc71", "POS_LATE"),
        }

        for s, (style, color, label) in style_map.items():
            vals = strategy_accs[s]
            valid_fracs = [f for f, v in zip(fracs_vals, vals) if v is not None]
            valid_vals = [v for v in vals if v is not None]
            if valid_vals:
                ax4.plot(valid_fracs, valid_vals, style, color=color, label=label, lw=2, ms=8)

        ax4.set_xlabel("Noise Fraction (%)")
        ax4.set_ylabel("Answer Accuracy (%)")
        ax4.set_title(f"Accuracy vs Noise Fraction (n={n})")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-5, 110)

        plt.tight_layout()
        fig4.savefig(os.path.join(results_dir, "accuracy_sweep.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Figure saved: accuracy_sweep.png")


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    main()
