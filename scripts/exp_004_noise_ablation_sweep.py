#!/usr/bin/env python3
"""
Experiment 004: Noise Ablation Sweep with Selectivity-Based Classification

Key improvements over exp_002/003:
1. TWO ablation methods at different regimes:
   - NOISE injection at low fractions (20-70%) — noise is very destructive so the
     interesting range is lower. At 90% noise, ALL conditions hit 0% accuracy.
   - ZEROING at high fractions (80-99%) — zeroing is weak so the interesting range
     is higher. At 50% zeroing, ALL conditions maintain 100% accuracy.
2. SELECTIVITY-BASED classification — separates positions that are exclusively
   answer-coupled (high AC, low TC) from exclusively text-coupled (high TC, low AC).
   This addresses the overlap problem from exp_002 where AC and TC top-50% heavily
   overlapped, making the distinction meaningless.
3. Fraction sweep to find degradation cliff for each method
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

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B-Base"
NUM_PROBLEMS = 35  # aim for ~30 valid at ~91% baseline accuracy
# Noise is VERY destructive (even 30% = total collapse), so test very low fractions
NOISE_FRACTIONS = [0.05, 0.10, 0.15, 0.20]
# Zeroing is weak (50% = no effect), so test very high fractions
ZERO_FRACTIONS = [0.90, 0.95, 0.99]
ATTENTION_LAYERS = [-1, -2, -3, -4]  # last 4 layers
MAX_GEN_TOKENS = 768
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_004")

os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 8-shot GSM8K exemplars ─────────────────────────────────────────────
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
    """
    Compute selectivity scores for each position.

    Selectivity = percentile_rank(AC) - percentile_rank(TC)
    Positive = selectively answer-coupled
    Negative = selectively text-coupled

    Returns selectivity array (same length as inputs).
    """
    n = len(answer_coupling)
    ac_ranks = np.argsort(np.argsort(answer_coupling)).astype(float) / (n - 1)  # percentile ranks [0,1]
    tc_ranks = np.argsort(np.argsort(text_coupling)).astype(float) / (n - 1)
    selectivity = ac_ranks - tc_ranks  # range [-1, 1]
    return selectivity


@torch.no_grad()
def ablated_answer_generation(model, tokenizer, prompt, reasoning_text,
                               positions_to_ablate, prompt_len, ablation_mode="noise"):
    """
    Teacher-force prompt + reasoning, ablate specific KV positions, generate answer.
    ablation_mode: "zero" or "noise" (norm-matched Gaussian)
    """
    full_text = prompt + reasoning_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]

    outputs = model(**inputs, use_cache=True, output_attentions=False)
    past_kv = outputs.past_key_values
    logits = outputs.logits
    num_layers = len(past_kv.layers)

    reasoning_len = seq_len - prompt_len
    lookback = min(20, reasoning_len)
    lookback_start = seq_len - lookback

    # Pre-ablation text loss
    orig_text_losses = []
    for t in range(lookback - 1):
        pos = lookback_start + t
        target_id = inputs.input_ids[0, pos + 1].item()
        log_probs = torch.log_softmax(logits[0, pos], dim=-1)
        orig_text_losses.append(-log_probs[target_id].item())

    # Ablate KV cache
    abs_positions = [prompt_len + p for p in positions_to_ablate if prompt_len + p < seq_len]

    if abs_positions:
        for layer_idx in range(num_layers):
            layer = past_kv.layers[layer_idx]
            for pos in abs_positions:
                if ablation_mode == "zero":
                    layer.keys[:, :, pos, :] = 0.0
                    layer.values[:, :, pos, :] = 0.0
                elif ablation_mode == "noise":
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

    pruned_text_losses = []
    for t in range(lookback - 1):
        target_id = lookback_tokens[0, t + 1].item()
        log_probs = torch.log_softmax(lookback_outputs.logits[0, t], dim=-1)
        pruned_text_losses.append(-log_probs[target_id].item())

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

    mean_orig = float(np.mean(orig_text_losses)) if orig_text_losses else 0.0
    mean_pruned = float(np.mean(pruned_text_losses)) if pruned_text_losses else 0.0

    del outputs, lookback_outputs, past_kv, gen_kv, trunc_kv
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "answer_text": answer_text,
        "answer": extract_answer(answer_text),
        "mean_orig_text_loss": mean_orig,
        "mean_pruned_text_loss": mean_pruned,
        "text_loss_change": mean_pruned - mean_orig,
    }


def main():
    start_time = time.time()
    print(f"{'='*70}")
    print(f"Experiment 004: Noise Ablation Sweep with Selectivity Classification")
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {NUM_PROBLEMS}")
    print(f"Noise fractions: {NOISE_FRACTIONS}")
    print(f"Zero fractions: {ZERO_FRACTIONS}")
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
    random.shuffle(indices)
    selected = indices[:NUM_PROBLEMS]

    # ── Phase 1: Generate traces and extract attention ──────────────────
    print("\n=== Phase 1: Generating traces and classifying positions ===")
    problems_data = []

    for prob_idx, ds_idx in enumerate(selected):
        elapsed = time.time() - start_time
        if elapsed > 900:  # 15 min budget for phase 1
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

        # Compute selectivity scores
        selectivity = compute_selectivity(attn_info["answer_coupling"], attn_info["text_coupling"])

        # Report overlap between standard AC and TC top positions at 50%
        n_half = attn_info["reasoning_len"] // 2
        ac_top = set(np.argsort(attn_info["answer_coupling"])[-n_half:])
        tc_top = set(np.argsort(attn_info["text_coupling"])[-n_half:])
        overlap_frac = len(ac_top & tc_top) / n_half if n_half > 0 else 0

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
            "overlap_at_50pct": overlap_frac,
        })
        print(f"  Valid! reasoning_len={attn_info['reasoning_len']}, "
              f"AC/TC overlap@50%={overlap_frac:.0%}, "
              f"selectivity range=[{selectivity.min():.2f}, {selectivity.max():.2f}]")

    print(f"\n=== Phase 1 complete: {len(problems_data)} valid problems ===")

    if len(problems_data) < 3:
        print("ERROR: Too few valid problems!")
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump({"error": "too_few_valid", "n_valid": len(problems_data)}, f, indent=2)
        return

    # Report mean overlap
    mean_overlap = np.mean([pd["overlap_at_50pct"] for pd in problems_data])
    print(f"Mean AC/TC overlap at 50%: {mean_overlap:.0%}")

    # ── Phase 2: Ablation sweep ───────────────────────────────────────
    print("\n=== Phase 2: Ablation sweep (selectivity-based) ===")
    print("Noise injection at low fractions (destructive), zeroing at high fractions (weak)")

    sweep_results = {}
    conditions_base = ["selective_AC", "selective_TC", "random"]

    # Build the sweep plan: (fraction, method) pairs
    sweep_plan = [(f, "noise") for f in NOISE_FRACTIONS] + [(f, "zero") for f in ZERO_FRACTIONS]

    for frac, method in sweep_plan:
        elapsed = time.time() - start_time
        if elapsed > 1500:  # 25 min hard cutoff
            print(f"Time budget reached at {method} fraction {frac}")
            break

        sweep_key = f"{method}_{frac}"
        print(f"\n--- {method.upper()} at {frac:.0%} ---")

        sweep_results[sweep_key] = {c: [] for c in conditions_base}

        for pi, pd in enumerate(problems_data):
            n_prune = max(1, int(pd["reasoning_len"] * frac))

            # Selectivity-based positions
            sel_ac_positions = np.argsort(pd["selectivity"])[-n_prune:].tolist()
            sel_tc_positions = np.argsort(pd["selectivity"])[:n_prune].tolist()
            rnd_positions = random.sample(range(pd["reasoning_len"]), n_prune)

            position_map = {
                "selective_AC": sel_ac_positions,
                "selective_TC": sel_tc_positions,
                "random": rnd_positions,
            }

            for cond_name in conditions_base:
                positions = position_map[cond_name]
                try:
                    result = ablated_answer_generation(
                        model, tokenizer, pd["prompt"], pd["reasoning_text"],
                        positions, pd["prompt_len"], ablation_mode=method
                    )
                    ans_norm = normalize_answer(result["answer"]) if result["answer"] else ""
                    correct = (ans_norm == pd["true_answer"])
                    sweep_results[sweep_key][cond_name].append({
                        "ds_idx": pd["ds_idx"],
                        "correct": correct,
                        "answer": result["answer"],
                        "text_loss_change": result["text_loss_change"],
                    })
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    sweep_results[sweep_key][cond_name].append({
                        "ds_idx": pd["ds_idx"],
                        "error": "OOM",
                    })

            if (pi + 1) % 10 == 0 or pi == len(problems_data) - 1:
                n = pi + 1
                parts = []
                for c in conditions_base:
                    c_correct = sum(1 for r in sweep_results[sweep_key][c] if r.get("correct"))
                    parts.append(f"{c}={c_correct}/{n}")
                print(f"  [{n}/{len(problems_data)}] {method}@{frac:.0%}: {', '.join(parts)}")

        # Print summary for this condition
        for c in conditions_base:
            valid = [r for r in sweep_results[sweep_key][c] if "error" not in r]
            correct_count = sum(1 for r in valid if r["correct"])
            n = len(valid)
            acc = correct_count / n if n > 0 else 0
            tlc = [r["text_loss_change"] for r in valid]
            mean_tlc = np.mean(tlc) if tlc else 0
            print(f"  {c}: accuracy={correct_count}/{n} ({acc:.0%}), "
                  f"text_loss_change={mean_tlc:+.4f}")

    # ── Analysis ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    summary = {
        "experiment": "exp_004_noise_ablation_sweep",
        "model": MODEL_NAME,
        "num_valid": len(problems_data),
        "mean_overlap_at_50pct": float(mean_overlap),
        "noise_fractions": NOISE_FRACTIONS,
        "zero_fractions": ZERO_FRACTIONS,
        "sweep": {},
    }

    print(f"\nValid problems: {len(problems_data)}")
    print(f"Mean AC/TC overlap at 50%: {mean_overlap:.0%}")
    print(f"\n{'Method+Frac':<16} {'Sel_AC Acc':<14} {'Sel_TC Acc':<14} {'Rnd Acc':<12} {'Sel_AC TLC':<14} {'Sel_TC TLC':<14} {'Rnd TLC':<12}")
    print("-" * 96)

    for sweep_key in sweep_results:
        frac_summary = {}
        for c in sweep_results[sweep_key]:
            valid = [r for r in sweep_results[sweep_key][c] if "error" not in r]
            correct_count = sum(1 for r in valid if r["correct"])
            n = len(valid)
            acc = correct_count / n if n > 0 else 0
            tlc_vals = [r["text_loss_change"] for r in valid]
            mean_tlc = float(np.mean(tlc_vals)) if tlc_vals else 0
            std_tlc = float(np.std(tlc_vals)) if len(tlc_vals) > 1 else 0
            frac_summary[c] = {
                "accuracy": acc,
                "correct": correct_count,
                "total": n,
                "mean_text_loss_change": mean_tlc,
                "std_text_loss_change": std_tlc,
            }
        summary["sweep"][sweep_key] = frac_summary

        if all(c in frac_summary for c in conditions_base):
            sac = frac_summary["selective_AC"]
            stc = frac_summary["selective_TC"]
            rnd = frac_summary["random"]
            print(f"{sweep_key:<16} {sac['accuracy']:<14.1%} {stc['accuracy']:<14.1%} {rnd['accuracy']:<12.1%} "
                  f"{sac['mean_text_loss_change']:<+14.4f} {stc['mean_text_loss_change']:<+14.4f} {rnd['mean_text_loss_change']:<+12.4f}")

    # ── Dissociation analysis ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DISSOCIATION ANALYSIS")
    print(f"{'='*70}")

    for sweep_key, frac_data in summary["sweep"].items():
        if "selective_AC" not in frac_data or "selective_TC" not in frac_data:
            continue
        sac_acc = frac_data["selective_AC"]["accuracy"]
        stc_acc = frac_data["selective_TC"]["accuracy"]
        rnd_acc = frac_data["random"]["accuracy"]
        sac_tlc = frac_data["selective_AC"]["mean_text_loss_change"]
        stc_tlc = frac_data["selective_TC"]["mean_text_loss_change"]

        acc_dissoc = sac_acc < stc_acc  # AC ablation hurts accuracy more
        text_dissoc = stc_tlc > sac_tlc  # TC ablation hurts text more
        double = acc_dissoc and text_dissoc

        acc_effect = stc_acc - sac_acc
        tlc_effect = stc_tlc - sac_tlc

        print(f"\n{sweep_key}:")
        print(f"  SelAC_acc={sac_acc:.1%} SelTC_acc={stc_acc:.1%} Rnd_acc={rnd_acc:.1%}")
        print(f"  SelAC_tlc={sac_tlc:+.4f} SelTC_tlc={stc_tlc:+.4f}")
        print(f"  acc_dissoc={acc_dissoc} (effect={acc_effect:+.3f})")
        print(f"  text_dissoc={text_dissoc} (effect={tlc_effect:+.4f})")
        print(f"  DOUBLE DISSOCIATION: {double}")

    summary["elapsed_seconds"] = time.time() - start_time

    # Save results
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_convert)

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(sweep_results, f, indent=2, default=_convert)

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

    # Parse sweep keys into (method, fraction)
    def parse_key(k):
        parts = k.rsplit("_", 1)
        return parts[0], float(parts[1])

    # Separate noise and zero results
    noise_keys = sorted([k for k in sweep if k.startswith("noise_")], key=lambda k: parse_key(k)[1])
    zero_keys = sorted([k for k in sweep if k.startswith("zero_")], key=lambda k: parse_key(k)[1])

    # Figure 1: Two-panel — noise sweep (left) and zero sweep (right) for accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for keys, ax, method_name in [(noise_keys, axes[0], "Noise Injection"),
                                   (zero_keys, axes[1], "Zeroing")]:
        if not keys:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        fracs = [parse_key(k)[1] * 100 for k in keys]
        sac_accs = [sweep[k]["selective_AC"]["accuracy"] for k in keys if "selective_AC" in sweep[k]]
        stc_accs = [sweep[k]["selective_TC"]["accuracy"] for k in keys if "selective_TC" in sweep[k]]
        rnd_accs = [sweep[k]["random"]["accuracy"] for k in keys if "random" in sweep[k]]

        ax.plot(fracs[:len(sac_accs)], sac_accs, "o-", color="#e74c3c", label="Sel. AC", lw=2, ms=8)
        ax.plot(fracs[:len(stc_accs)], stc_accs, "s-", color="#3498db", label="Sel. TC", lw=2, ms=8)
        ax.plot(fracs[:len(rnd_accs)], rnd_accs, "^-", color="#95a5a6", label="Random", lw=2, ms=8)
        ax.set_xlabel("Ablation Fraction (%)")
        ax.set_ylabel("Answer Accuracy")
        ax.set_title(f"{method_name}: Accuracy vs Fraction")
        ax.set_ylim(-0.05, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Exp 004: Ablation Sweep — Accuracy (n={n}, Qwen3-4B-Base)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "ablation_accuracy_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: ablation_accuracy_sweep.png")

    # Figure 2: Text loss change sweep (same layout)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    for keys, ax, method_name in [(noise_keys, axes2[0], "Noise Injection"),
                                   (zero_keys, axes2[1], "Zeroing")]:
        if not keys:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        fracs = [parse_key(k)[1] * 100 for k in keys]
        sac_tlcs = [sweep[k]["selective_AC"]["mean_text_loss_change"] for k in keys if "selective_AC" in sweep[k]]
        stc_tlcs = [sweep[k]["selective_TC"]["mean_text_loss_change"] for k in keys if "selective_TC" in sweep[k]]
        rnd_tlcs = [sweep[k]["random"]["mean_text_loss_change"] for k in keys if "random" in sweep[k]]

        ax.plot(fracs[:len(sac_tlcs)], sac_tlcs, "o-", color="#e74c3c", label="Sel. AC", lw=2, ms=8)
        ax.plot(fracs[:len(stc_tlcs)], stc_tlcs, "s-", color="#3498db", label="Sel. TC", lw=2, ms=8)
        ax.plot(fracs[:len(rnd_tlcs)], rnd_tlcs, "^-", color="#95a5a6", label="Random", lw=2, ms=8)
        ax.set_xlabel("Ablation Fraction (%)")
        ax.set_ylabel("Text Loss Change (nats)")
        ax.set_title(f"{method_name}: Text Loss vs Fraction")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)

    fig2.suptitle(f"Exp 004: Ablation Sweep — Text Loss (n={n}, Qwen3-4B-Base)", fontsize=12)
    plt.tight_layout()
    fig2.savefig(os.path.join(results_dir, "ablation_textloss_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: ablation_textloss_sweep.png")

    # Figure 3: Combined dissociation effect (all conditions on one plot)
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    all_keys = noise_keys + zero_keys
    if all_keys:
        labels = []
        acc_effects = []
        tlc_effects = []
        colors = []
        for k in all_keys:
            if "selective_AC" not in sweep[k] or "selective_TC" not in sweep[k]:
                continue
            method, frac = parse_key(k)
            labels.append(f"{method[:1].upper()}{int(frac*100)}%")
            acc_effects.append(sweep[k]["selective_TC"]["accuracy"] - sweep[k]["selective_AC"]["accuracy"])
            tlc_effects.append(sweep[k]["selective_TC"]["mean_text_loss_change"] - sweep[k]["selective_AC"]["mean_text_loss_change"])
            colors.append("#e67e22" if method == "noise" else "#2ecc71")

        x = np.arange(len(labels))
        axes3[0].bar(x, acc_effects, color=colors, edgecolor="black", linewidth=0.5)
        axes3[0].set_xticks(x)
        axes3[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        axes3[0].set_ylabel("Accuracy Effect (SelTC - SelAC)")
        axes3[0].set_title("Accuracy Dissociation\n(positive = AC hurts more)")
        axes3[0].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        axes3[0].grid(True, alpha=0.2)

        axes3[1].bar(x, tlc_effects, color=colors, edgecolor="black", linewidth=0.5)
        axes3[1].set_xticks(x)
        axes3[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        axes3[1].set_ylabel("Text Loss Effect (SelTC - SelAC)")
        axes3[1].set_title("Text Loss Dissociation\n(positive = TC hurts more)")
        axes3[1].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        axes3[1].grid(True, alpha=0.2)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#e67e22", edgecolor="black", label="Noise"),
                           Patch(facecolor="#2ecc71", edgecolor="black", label="Zero")]
        axes3[0].legend(handles=legend_elements)

    fig3.suptitle(f"Exp 004: Dissociation Effect Size (n={n})", fontsize=12)
    plt.tight_layout()
    fig3.savefig(os.path.join(results_dir, "dissociation_effects.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: dissociation_effects.png")


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()
