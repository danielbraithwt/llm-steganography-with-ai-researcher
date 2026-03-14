#!/usr/bin/env python3
"""
Experiment 003: Pruning Fraction Sweep — Diagnosing the Double Dissociation Failure

Exp_002 found 100% accuracy at 50% pruning for ALL position types (AC, TC, random).
This experiment sweeps pruning fraction from 50% to 95% to find the accuracy cliff
and tests whether it differs between answer-coupled and text-coupled positions.

Also tests noise injection (replacing with Gaussian noise) as an alternative to zeroing,
which prevents the "route around zeros" confound.

Method:
1. Generate 8-shot CoT traces for GSM8K problems
2. Teacher-force to extract attention and classify positions (AC vs TC)
3. For each pruning fraction [0.5, 0.7, 0.8, 0.9, 0.95]:
   - Zero AC positions -> measure accuracy
   - Zero TC positions -> measure accuracy
   - Zero random positions -> measure accuracy
4. At 90% fraction, also test noise injection at AC vs TC positions
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
NUM_PROBLEMS = 50  # aim for ~45 valid at 91% baseline accuracy
PRUNE_FRACTIONS = [0.50, 0.70, 0.80, 0.90, 0.95]
NOISE_FRACTION = 0.90  # fraction for noise injection test
ATTENTION_LAYERS = [-1, -2, -3, -4]  # last 4 layers
MAX_GEN_TOKENS = 768
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_003")

os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 8-shot GSM8K exemplars (same as exp_002) ─────────────────────────
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

    # OOM guard
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


@torch.no_grad()
def pruned_answer_generation(model, tokenizer, prompt, reasoning_text,
                              positions_to_ablate, prompt_len, ablation_mode="zero"):
    """
    Teacher-force prompt + reasoning, ablate specific KV positions, generate answer.

    ablation_mode: "zero" = set KV to 0, "noise" = replace with Gaussian noise (matched norm)

    Returns: answer text, answer correctness metrics
    """
    full_text = prompt + reasoning_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]

    # Forward pass to build KV cache
    outputs = model(**inputs, use_cache=True, output_attentions=False)
    past_kv = outputs.past_key_values
    logits = outputs.logits
    num_layers = len(past_kv.layers)

    # Compute text loss on last 20 reasoning tokens (before ablation, for reference)
    reasoning_len = seq_len - prompt_len
    lookback = min(20, reasoning_len)
    lookback_start = seq_len - lookback

    orig_text_losses = []
    for t in range(lookback - 1):
        pos = lookback_start + t
        target_id = inputs.input_ids[0, pos + 1].item()
        log_probs = torch.log_softmax(logits[0, pos], dim=-1)
        orig_text_losses.append(-log_probs[target_id].item())

    # Ablate KV cache at specified positions
    abs_positions = [prompt_len + p for p in positions_to_ablate if prompt_len + p < seq_len]

    if abs_positions:
        for layer_idx in range(num_layers):
            layer = past_kv.layers[layer_idx]
            for pos in abs_positions:
                if ablation_mode == "zero":
                    layer.keys[:, :, pos, :] = 0.0
                    layer.values[:, :, pos, :] = 0.0
                elif ablation_mode == "noise":
                    # Replace with Gaussian noise matched to the norm of the original
                    k_norm = layer.keys[:, :, pos, :].norm().item()
                    v_norm = layer.values[:, :, pos, :].norm().item()
                    k_shape = layer.keys[:, :, pos, :].shape
                    v_shape = layer.values[:, :, pos, :].shape
                    k_noise = torch.randn(k_shape, device=layer.keys.device, dtype=layer.keys.dtype)
                    v_noise = torch.randn(v_shape, device=layer.values.device, dtype=layer.values.dtype)
                    # Normalize noise to match original norms
                    k_noise = k_noise * (k_norm / (k_noise.norm().item() + 1e-8))
                    v_noise = v_noise * (v_norm / (v_noise.norm().item() + 1e-8))
                    layer.keys[:, :, pos, :] = k_noise
                    layer.values[:, :, pos, :] = v_noise

    # Re-process last tokens through ablated cache to measure text loss change
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
    print(f"Experiment 003: Pruning Fraction Sweep")
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {NUM_PROBLEMS}")
    print(f"Prune fractions: {PRUNE_FRACTIONS}")
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
        if elapsed > 1200:  # 20 min budget for full run
            print(f"Time budget reached at problem {prob_idx}")
            break

        problem = dataset[ds_idx]
        question = problem["question"]
        true_answer = normalize_answer(problem["answer"].split("####")[-1].strip()
                                       .replace(",", "").replace("$", ""))
        prompt = build_prompt(question)

        print(f"\nProblem {prob_idx+1}/{NUM_PROBLEMS} (#{ds_idx}), true={true_answer}")

        # Generate trace
        trace_text = generate_trace(model, tokenizer, prompt)
        gen_answer = extract_answer(trace_text)
        gen_norm = normalize_answer(gen_answer) if gen_answer else ""
        correct = (gen_norm == true_answer)
        print(f"  Generated: '{gen_answer}' (correct: {correct})")

        if not correct:
            print("  SKIP: baseline incorrect")
            continue

        # Split trace
        if "####" in trace_text:
            reasoning_text = trace_text[:trace_text.index("####")]
        else:
            reasoning_text = trace_text

        # Extract attention
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

        problems_data.append({
            "ds_idx": ds_idx,
            "true_answer": true_answer,
            "prompt": prompt,
            "reasoning_text": reasoning_text,
            "answer_coupling": attn_info["answer_coupling"],
            "text_coupling": attn_info["text_coupling"],
            "prompt_len": attn_info["prompt_len"],
            "reasoning_len": attn_info["reasoning_len"],
        })
        print(f"  Valid! reasoning_len={attn_info['reasoning_len']}, "
              f"AC mean={attn_info['answer_coupling'].mean():.4f}, "
              f"TC mean={attn_info['text_coupling'].mean():.4f}")

    print(f"\n=== Phase 1 complete: {len(problems_data)} valid problems ===")

    if len(problems_data) < 3:
        print("ERROR: Too few valid problems!")
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump({"error": "too_few_valid", "n_valid": len(problems_data)}, f, indent=2)
        return

    # ── Phase 2: Pruning fraction sweep ─────────────────────────────────
    print("\n=== Phase 2: Pruning fraction sweep ===")

    # Structure: results[fraction][position_type] = list of per-problem results
    sweep_results = {}

    for frac in PRUNE_FRACTIONS:
        elapsed = time.time() - start_time
        if elapsed > 1500:  # 25 min hard cutoff
            print(f"Time budget reached at fraction {frac}")
            break

        print(f"\n--- Pruning fraction: {frac:.0%} ---")
        sweep_results[frac] = {"answer_coupled": [], "text_coupled": [], "random": []}

        for pi, pd in enumerate(problems_data):
            n_prune = max(1, int(pd["reasoning_len"] * frac))
            ac_positions = np.argsort(pd["answer_coupling"])[-n_prune:].tolist()
            tc_positions = np.argsort(pd["text_coupling"])[-n_prune:].tolist()
            rnd_positions = random.sample(range(pd["reasoning_len"]), n_prune)

            for pos_type, positions in [("answer_coupled", ac_positions),
                                         ("text_coupled", tc_positions),
                                         ("random", rnd_positions)]:
                try:
                    result = pruned_answer_generation(
                        model, tokenizer, pd["prompt"], pd["reasoning_text"],
                        positions, pd["prompt_len"], ablation_mode="zero"
                    )
                    ans_norm = normalize_answer(result["answer"]) if result["answer"] else ""
                    correct = (ans_norm == pd["true_answer"])
                    sweep_results[frac][pos_type].append({
                        "ds_idx": pd["ds_idx"],
                        "correct": correct,
                        "answer": result["answer"],
                        "text_loss_change": result["text_loss_change"],
                    })
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    sweep_results[frac][pos_type].append({
                        "ds_idx": pd["ds_idx"],
                        "error": "OOM",
                    })

            if (pi + 1) % 5 == 0:
                ac_correct = sum(1 for r in sweep_results[frac]["answer_coupled"] if r.get("correct"))
                tc_correct = sum(1 for r in sweep_results[frac]["text_coupled"] if r.get("correct"))
                rnd_correct = sum(1 for r in sweep_results[frac]["random"] if r.get("correct"))
                n = pi + 1
                print(f"  [{n}/{len(problems_data)}] frac={frac:.0%}: "
                      f"AC={ac_correct}/{n}, TC={tc_correct}/{n}, Rnd={rnd_correct}/{n}")

        # Print fraction summary
        for pos_type in ["answer_coupled", "text_coupled", "random"]:
            results_list = sweep_results[frac][pos_type]
            valid = [r for r in results_list if "error" not in r]
            correct = sum(1 for r in valid if r["correct"])
            n = len(valid)
            acc = correct / n if n > 0 else 0
            tlc = [r["text_loss_change"] for r in valid]
            mean_tlc = np.mean(tlc) if tlc else 0
            print(f"  {pos_type}: accuracy={correct}/{n} ({acc:.0%}), "
                  f"text_loss_change={mean_tlc:+.4f}")

    # ── Phase 3: Noise injection at 90% ─────────────────────────────────
    elapsed = time.time() - start_time
    noise_results = {"answer_coupled": [], "text_coupled": []}

    if elapsed < 1400:  # only if time permits
        print(f"\n=== Phase 3: Noise injection at {NOISE_FRACTION:.0%} ===")

        for pi, pd in enumerate(problems_data):
            elapsed = time.time() - start_time
            if elapsed > 1500:
                print("Time budget reached during noise injection")
                break

            n_prune = max(1, int(pd["reasoning_len"] * NOISE_FRACTION))
            ac_positions = np.argsort(pd["answer_coupling"])[-n_prune:].tolist()
            tc_positions = np.argsort(pd["text_coupling"])[-n_prune:].tolist()

            for pos_type, positions in [("answer_coupled", ac_positions),
                                         ("text_coupled", tc_positions)]:
                try:
                    result = pruned_answer_generation(
                        model, tokenizer, pd["prompt"], pd["reasoning_text"],
                        positions, pd["prompt_len"], ablation_mode="noise"
                    )
                    ans_norm = normalize_answer(result["answer"]) if result["answer"] else ""
                    correct = (ans_norm == pd["true_answer"])
                    noise_results[pos_type].append({
                        "ds_idx": pd["ds_idx"],
                        "correct": correct,
                        "answer": result["answer"],
                        "text_loss_change": result["text_loss_change"],
                    })
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    noise_results[pos_type].append({"ds_idx": pd["ds_idx"], "error": "OOM"})

            if (pi + 1) % 5 == 0:
                ac_c = sum(1 for r in noise_results["answer_coupled"] if r.get("correct"))
                tc_c = sum(1 for r in noise_results["text_coupled"] if r.get("correct"))
                n = pi + 1
                print(f"  [{n}/{len(problems_data)}] noise@{NOISE_FRACTION:.0%}: "
                      f"AC={ac_c}/{n}, TC={tc_c}/{n}")

        for pos_type in ["answer_coupled", "text_coupled"]:
            valid = [r for r in noise_results[pos_type] if "error" not in r]
            correct = sum(1 for r in valid if r["correct"])
            n = len(valid)
            acc = correct / n if n > 0 else 0
            print(f"  noise_{pos_type}: accuracy={correct}/{n} ({acc:.0%})")
    else:
        print("\n=== Phase 3 skipped (time budget) ===")

    # ── Analysis ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    # Build summary table
    summary = {
        "experiment": "exp_003_pruning_sweep",
        "model": MODEL_NAME,
        "num_valid": len(problems_data),
        "prune_fractions": PRUNE_FRACTIONS,
        "sweep": {},
        "noise_injection": {},
    }

    print(f"\nValid problems: {len(problems_data)}")
    print(f"\n{'Fraction':<10} {'AC Acc':<12} {'TC Acc':<12} {'Rnd Acc':<12} {'AC TLC':<12} {'TC TLC':<12} {'Rnd TLC':<12}")
    print("-" * 82)

    for frac in PRUNE_FRACTIONS:
        if frac not in sweep_results:
            continue
        frac_summary = {}
        for pos_type in ["answer_coupled", "text_coupled", "random"]:
            valid = [r for r in sweep_results[frac][pos_type] if "error" not in r]
            correct = sum(1 for r in valid if r["correct"])
            n = len(valid)
            acc = correct / n if n > 0 else 0
            tlc_vals = [r["text_loss_change"] for r in valid]
            mean_tlc = float(np.mean(tlc_vals)) if tlc_vals else 0
            std_tlc = float(np.std(tlc_vals)) if len(tlc_vals) > 1 else 0
            frac_summary[pos_type] = {
                "accuracy": acc,
                "correct": correct,
                "total": n,
                "mean_text_loss_change": mean_tlc,
                "std_text_loss_change": std_tlc,
            }
        summary["sweep"][str(frac)] = frac_summary

        ac = frac_summary["answer_coupled"]
        tc = frac_summary["text_coupled"]
        rnd = frac_summary["random"]
        print(f"{frac:<10.0%} {ac['accuracy']:<12.1%} {tc['accuracy']:<12.1%} {rnd['accuracy']:<12.1%} "
              f"{ac['mean_text_loss_change']:<+12.4f} {tc['mean_text_loss_change']:<+12.4f} {rnd['mean_text_loss_change']:<+12.4f}")

    # Noise injection results
    if noise_results["answer_coupled"]:
        for pos_type in ["answer_coupled", "text_coupled"]:
            valid = [r for r in noise_results[pos_type] if "error" not in r]
            correct = sum(1 for r in valid if r["correct"])
            n = len(valid)
            acc = correct / n if n > 0 else 0
            tlc_vals = [r["text_loss_change"] for r in valid]
            mean_tlc = float(np.mean(tlc_vals)) if tlc_vals else 0
            summary["noise_injection"][pos_type] = {
                "accuracy": acc,
                "correct": correct,
                "total": n,
                "mean_text_loss_change": mean_tlc,
            }
        ac_noise = summary["noise_injection"]["answer_coupled"]
        tc_noise = summary["noise_injection"]["text_coupled"]
        print(f"\nNoise injection at {NOISE_FRACTION:.0%}:")
        print(f"  AC: accuracy={ac_noise['accuracy']:.1%}, text_loss_change={ac_noise['mean_text_loss_change']:+.4f}")
        print(f"  TC: accuracy={tc_noise['accuracy']:.1%}, text_loss_change={tc_noise['mean_text_loss_change']:+.4f}")

    # ── Dissociation analysis ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DISSOCIATION ANALYSIS")
    print(f"{'='*70}")

    for frac_str, frac_data in summary["sweep"].items():
        ac_acc = frac_data["answer_coupled"]["accuracy"]
        tc_acc = frac_data["text_coupled"]["accuracy"]
        rnd_acc = frac_data["random"]["accuracy"]
        ac_tlc = frac_data["answer_coupled"]["mean_text_loss_change"]
        tc_tlc = frac_data["text_coupled"]["mean_text_loss_change"]

        acc_dissoc = ac_acc < tc_acc
        text_dissoc = tc_tlc > ac_tlc
        double = acc_dissoc and text_dissoc
        print(f"\nFraction {frac_str}: AC_acc={ac_acc:.1%} TC_acc={tc_acc:.1%} "
              f"| acc_dissoc={acc_dissoc} text_dissoc={text_dissoc} "
              f"| DOUBLE={double}")

    summary["elapsed_seconds"] = time.time() - start_time

    # Save results
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_convert)

    # Save raw sweep data
    raw_data = {}
    for frac in PRUNE_FRACTIONS:
        if frac in sweep_results:
            raw_data[str(frac)] = sweep_results[frac]
    raw_data["noise_injection"] = noise_results
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(raw_data, f, indent=2, default=_convert)

    # ── Generate figures ────────────────────────────────────────────────
    generate_figures(summary, RESULTS_DIR)

    total_time = time.time() - start_time
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")


def generate_figures(summary, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = sorted([float(f) for f in summary["sweep"].keys()])
    if not fracs:
        print("No data for figures")
        return

    ac_accs = [summary["sweep"][str(f)]["answer_coupled"]["accuracy"] for f in fracs]
    tc_accs = [summary["sweep"][str(f)]["text_coupled"]["accuracy"] for f in fracs]
    rnd_accs = [summary["sweep"][str(f)]["random"]["accuracy"] for f in fracs]

    ac_tlcs = [summary["sweep"][str(f)]["answer_coupled"]["mean_text_loss_change"] for f in fracs]
    tc_tlcs = [summary["sweep"][str(f)]["text_coupled"]["mean_text_loss_change"] for f in fracs]
    rnd_tlcs = [summary["sweep"][str(f)]["random"]["mean_text_loss_change"] for f in fracs]

    frac_pcts = [f * 100 for f in fracs]

    # Figure 1: Accuracy vs pruning fraction
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(frac_pcts, ac_accs, "o-", color="#e74c3c", label="Answer-coupled", linewidth=2, markersize=8)
    axes[0].plot(frac_pcts, tc_accs, "s-", color="#3498db", label="Text-coupled", linewidth=2, markersize=8)
    axes[0].plot(frac_pcts, rnd_accs, "^-", color="#95a5a6", label="Random", linewidth=2, markersize=8)
    axes[0].set_xlabel("Pruning Fraction (%)")
    axes[0].set_ylabel("Answer Accuracy")
    axes[0].set_title("Answer Accuracy vs Pruning Fraction")
    axes[0].set_ylim(-0.05, 1.1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frac_pcts, ac_tlcs, "o-", color="#e74c3c", label="Answer-coupled", linewidth=2, markersize=8)
    axes[1].plot(frac_pcts, tc_tlcs, "s-", color="#3498db", label="Text-coupled", linewidth=2, markersize=8)
    axes[1].plot(frac_pcts, rnd_tlcs, "^-", color="#95a5a6", label="Random", linewidth=2, markersize=8)
    axes[1].set_xlabel("Pruning Fraction (%)")
    axes[1].set_ylabel("Text Loss Change (nats)")
    axes[1].set_title("Text Loss Change vs Pruning Fraction")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=0.5)

    n = summary["num_valid"]
    fig.suptitle(f"Exp 003: Pruning Fraction Sweep (n={n}, Qwen3-4B-Base)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "pruning_sweep_accuracy_vs_fraction.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: pruning_sweep_accuracy_vs_fraction.png")

    # Figure 2: Dissociation effect size vs fraction
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    acc_effects = [tc - ac for ac, tc in zip(ac_accs, tc_accs)]
    tlc_effects = [tc - ac for ac, tc in zip(ac_tlcs, tc_tlcs)]

    ax2_twin = ax2.twinx()
    ax2.bar([f - 1 for f in frac_pcts], acc_effects, width=2, color="#2ecc71", alpha=0.7,
            label="Accuracy effect (TC - AC)")
    ax2_twin.bar([f + 1 for f in frac_pcts], tlc_effects, width=2, color="#9b59b6", alpha=0.7,
                 label="Text loss effect (TC - AC)")

    ax2.set_xlabel("Pruning Fraction (%)")
    ax2.set_ylabel("Accuracy Effect (TC - AC)", color="#2ecc71")
    ax2_twin.set_ylabel("Text Loss Effect (TC - AC)", color="#9b59b6")
    ax2.set_title("Dissociation Effect Size vs Pruning Fraction")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig2.savefig(os.path.join(results_dir, "dissociation_effect_vs_fraction.png"),
                 dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: dissociation_effect_vs_fraction.png")

    # Figure 3: Noise injection comparison (if available)
    if summary.get("noise_injection") and summary["noise_injection"].get("answer_coupled"):
        fig3, axes3 = plt.subplots(1, 2, figsize=(10, 5))

        # Compare zero vs noise at 90%
        frac_key = str(NOISE_FRACTION)
        if frac_key in summary["sweep"]:
            zero_ac_acc = summary["sweep"][frac_key]["answer_coupled"]["accuracy"]
            zero_tc_acc = summary["sweep"][frac_key]["text_coupled"]["accuracy"]
        else:
            zero_ac_acc = zero_tc_acc = 0

        noise_ac_acc = summary["noise_injection"]["answer_coupled"]["accuracy"]
        noise_tc_acc = summary["noise_injection"]["text_coupled"]["accuracy"]

        x = np.arange(2)
        width = 0.3
        axes3[0].bar(x - width/2, [zero_ac_acc, zero_tc_acc], width,
                     label="Zero", color=["#e74c3c", "#3498db"], alpha=0.5, edgecolor="black")
        axes3[0].bar(x + width/2, [noise_ac_acc, noise_tc_acc], width,
                     label="Noise", color=["#e74c3c", "#3498db"], alpha=1.0, edgecolor="black")
        axes3[0].set_xticks(x)
        axes3[0].set_xticklabels(["Answer-coupled", "Text-coupled"])
        axes3[0].set_ylabel("Accuracy")
        axes3[0].set_title(f"Zero vs Noise at {NOISE_FRACTION:.0%} Pruning")
        axes3[0].set_ylim(0, 1.1)
        axes3[0].legend()

        # Text loss comparison
        if frac_key in summary["sweep"]:
            zero_ac_tlc = summary["sweep"][frac_key]["answer_coupled"]["mean_text_loss_change"]
            zero_tc_tlc = summary["sweep"][frac_key]["text_coupled"]["mean_text_loss_change"]
        else:
            zero_ac_tlc = zero_tc_tlc = 0

        noise_ac_tlc = summary["noise_injection"]["answer_coupled"]["mean_text_loss_change"]
        noise_tc_tlc = summary["noise_injection"]["text_coupled"]["mean_text_loss_change"]

        axes3[1].bar(x - width/2, [zero_ac_tlc, zero_tc_tlc], width,
                     label="Zero", color=["#e74c3c", "#3498db"], alpha=0.5, edgecolor="black")
        axes3[1].bar(x + width/2, [noise_ac_tlc, noise_tc_tlc], width,
                     label="Noise", color=["#e74c3c", "#3498db"], alpha=1.0, edgecolor="black")
        axes3[1].set_xticks(x)
        axes3[1].set_xticklabels(["Answer-coupled", "Text-coupled"])
        axes3[1].set_ylabel("Text Loss Change (nats)")
        axes3[1].set_title(f"Text Loss: Zero vs Noise at {NOISE_FRACTION:.0%}")
        axes3[1].legend()

        plt.tight_layout()
        fig3.savefig(os.path.join(results_dir, "zero_vs_noise_comparison.png"),
                     dpi=150, bbox_inches="tight")
        plt.close()
        print("Figure saved: zero_vs_noise_comparison.png")


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
