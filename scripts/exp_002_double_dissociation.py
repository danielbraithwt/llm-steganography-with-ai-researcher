#!/usr/bin/env python3
"""
Experiment 002: Double Dissociation via KV Position Pruning (retry of exp_001)

Fixes from exp_001:
- Uses Qwen/Qwen3-4B-Base (base model, not instruct)
- Adds stop criteria to prevent model from continuing past the answer
- More robust answer extraction

Tests whether answer-coupled and text-coupled KV cache positions are
functionally separable under NORMAL inference (no adversarial machinery).

Method:
1. Generate 8-shot CoT traces for GSM8K problems
2. Teacher-force traces to extract attention patterns
3. Classify positions as answer-coupled or text-coupled
4. Prune each position type from KV cache before answer generation
5. Measure answer accuracy and reasoning-token prediction loss

Prediction (if hypothesis TRUE):
- Pruning answer-coupled positions: accuracy drops, text loss unchanged
- Pruning text-coupled positions: accuracy preserved, text loss increases

Prediction (if hypothesis FALSE):
- Both pruning conditions affect accuracy and text loss similarly
"""

import os
import sys
import json
import time
import random
import gc
import traceback
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B-Base"
NUM_PROBLEMS = 90  # aim for ~30 valid (baseline correct) given ~33% accuracy
PRUNE_FRACTION = 0.50  # prune top half — aggressive to test functional impact
ATTENTION_LAYERS = [-1, -2, -3, -4]  # last 4 layers for attention analysis
MAX_GEN_TOKENS = 768
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_002")

os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 8-shot GSM8K exemplars ──────────────────────────────────────────────
GSM8K_EXEMPLARS = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
        "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total bolts needed is 2+1=<<2+1=3>>3\n#### 3"
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "answer": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*150%=$<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624"
    },
    {
        "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
        "answer": "If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.\nIf she feeds the flock 15 cups in the morning, and 25 cups in the afternoon, then the carry-over to the final meal would be 60-15-25=<<60-15-25=20>>20 cups.\n#### 20"
    },
    {
        "question": "Kylar went to the store to get water and some apples. The store sold apples for $1 each and water for $3 per bottle. Kylar wanted to buy one bag of apples and 2 bottles of water. How much would Kylar spend if each bag has 6 apples?",
        "answer": "A bag has 6 apples and each apple costs $1, so a bag costs 6*1=$<<6*1=6>>6\nKylar wants 2 bottles of water so that would cost 2*3=$<<2*3=6>>6\nAltogether, Kylar would spend 6+6=$<<6+6=12>>12\n#### 12"
    },
    {
        "question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
        "answer": "If Seattle has 20 sheep, Charleston has 4 * 20 = <<4*20=80>>80 sheep\nToulouse has 2 * 80 = <<2*80=160>>160 sheep\nTogether, they have 20 + 80 + 160 = <<20+80+160=260>>260 sheep\n#### 260"
    },
    {
        "question": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
        "answer": "First find how long it takes to download 40% of the file: 200 GB * 0.4 / 2 GB/minute = <<200*0.4/2=40>>40 minutes\nThen find how long it takes to download the whole file once the restart is complete: 200 GB / 2 GB/minute = <<200/2=100>>100 minutes\nThen add the time to download 40% of the file, the restart time, and the time to download the whole file: 40 + 20 + 100 = <<40+20+100=160>>160 minutes\n#### 160"
    },
]


def build_prompt(question: str) -> str:
    """Build 8-shot CoT prompt for a GSM8K question."""
    prompt = ""
    for ex in GSM8K_EXEMPLARS:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def extract_answer(text: str) -> str:
    """Extract the numeric answer from generated text.

    Tries #### first, then falls back to finding the last number.
    """
    # Try #### format first
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = ans.replace(",", "").replace("$", "").strip()
        # Extract just the number
        match = re.match(r'^-?[\d.]+', ans)
        if match:
            return match.group(0)
        return ans
    return ""


def parse_gsm8k_answer(answer_text: str) -> str:
    """Parse ground truth answer from GSM8K."""
    if "####" in answer_text:
        ans = answer_text.split("####")[-1].strip()
        ans = ans.replace(",", "").replace("$", "").strip()
        return ans
    return ""


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison (handle decimals, etc.)."""
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
    """Generate a CoT trace with proper stop criteria."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Generate token by token with stop criteria
    generated_ids = []
    past_kv = None
    current_input = inputs.input_ids

    for step in range(max_tokens):
        if past_kv is not None:
            outputs = model(input_ids=current_input, past_key_values=past_kv, use_cache=True)
        else:
            outputs = model(**inputs, use_cache=True)

        past_kv = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        token_id = next_token[0, 0].item()
        generated_ids.append(token_id)

        # Check stop conditions
        if token_id == tokenizer.eos_token_id:
            break

        # Decode current generation to check for stop patterns
        current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Stop if we've produced #### followed by a number and a newline
        # (require \n, not $, to avoid cutting off multi-digit numbers)
        if "####" in current_text:
            after_hash = current_text.split("####")[-1]
            if re.search(r'\d+\s*\n', after_hash):
                break

        # Stop if the model starts a new question
        if "\nQ:" in current_text or "\n\nQ:" in current_text:
            # Remove the "Q:..." part
            idx = current_text.find("\nQ:")
            if idx > 0:
                # Re-tokenize just the part before the new question
                truncated = current_text[:idx]
                generated_ids = tokenizer.encode(truncated, add_special_tokens=False)
            break

        current_input = next_token

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_ids_tensor = torch.tensor(generated_ids, device=model.device)

    del past_kv, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return generated_text, generated_ids_tensor


@torch.no_grad()
def teacher_force_with_attention(model, tokenizer, prompt, trace_text):
    """
    Teacher-force prompt + trace, return attention weights from last 4 layers.
    Returns per-position attention from the final token and from reasoning tokens.
    """
    full_text = prompt + trace_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    reasoning_len = seq_len - prompt_len

    if reasoning_len < 5:
        return None

    # Check if sequence is too long for attention extraction (OOM guard)
    if seq_len > 2048:
        print(f"    Sequence too long ({seq_len} tokens), truncating trace for attention")
        # Truncate the trace to keep seq_len under 2048
        max_reasoning = 2048 - prompt_len
        trace_tokens = tokenizer(trace_text, return_tensors="pt").input_ids[0][:max_reasoning]
        trace_text = tokenizer.decode(trace_tokens, skip_special_tokens=True)
        full_text = prompt + trace_text
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]
        reasoning_len = seq_len - prompt_len

    # Forward pass with attention
    outputs = model(
        **inputs,
        output_attentions=True,
        use_cache=False,
    )

    # Extract attention from last 4 layers
    num_layers = len(outputs.attentions)
    layer_indices = [num_layers + i for i in ATTENTION_LAYERS]

    last_token_idx = seq_len - 1

    answer_coupling = torch.zeros(reasoning_len, device=model.device)
    text_coupling = torch.zeros(reasoning_len, device=model.device)

    for li in layer_indices:
        attn = outputs.attentions[li][0]  # (num_heads, seq_len, seq_len)

        # Answer coupling: attention from last token to each reasoning pos
        answer_coupling += attn[:, last_token_idx, prompt_len:seq_len].sum(dim=0)

        # Text coupling: for each reasoning position, average attention FROM
        # later reasoning tokens TO this position
        reasoning_attn = attn[:, prompt_len:seq_len, prompt_len:seq_len]  # (H, R, R)
        reasoning_attn_sum = reasoning_attn.sum(dim=0)  # (R, R)
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
def pruned_generation(model, tokenizer, prompt, reasoning_text, positions_to_prune, prompt_len):
    """
    Teacher-force prompt + reasoning (before ####), prune specific reasoning
    positions from KV cache, then let model generate the answer.

    reasoning_text: the reasoning portion BEFORE #### (model must generate #### and answer)
    positions_to_prune: reasoning-relative positions (0-indexed from start of reasoning)

    Returns: generated answer text, text prediction loss change
    """
    full_text = prompt + reasoning_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    reasoning_len = seq_len - prompt_len

    # Forward pass to build KV cache
    outputs = model(
        **inputs,
        use_cache=True,
        output_attentions=False,
    )

    past_kv = outputs.past_key_values
    logits = outputs.logits
    num_layers = len(past_kv.layers)

    # Compute per-token cross-entropy loss for reasoning tokens (original/unpruned)
    reasoning_losses = []
    for t in range(reasoning_len - 1):
        pos = prompt_len + t
        target_id = inputs.input_ids[0, pos + 1].item()
        log_probs = torch.log_softmax(logits[0, pos], dim=-1)
        loss = -log_probs[target_id].item()
        reasoning_losses.append(loss)

    # Prune KV cache at specified positions
    abs_positions = [prompt_len + p for p in positions_to_prune if prompt_len + p < seq_len]

    if abs_positions:
        for layer_idx in range(num_layers):
            layer = past_kv.layers[layer_idx]
            for pos in abs_positions:
                layer.keys[:, :, pos, :] = 0.0
                layer.values[:, :, pos, :] = 0.0

    # Measure text loss under pruning by re-processing last N reasoning tokens
    lookback = min(20, reasoning_len)
    lookback_start = seq_len - lookback
    lookback_tokens = inputs.input_ids[:, lookback_start:seq_len]

    # Build truncated DynamicCache (everything before lookback_start)
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
        loss = -log_probs[target_id].item()
        pruned_text_losses.append(loss)

    # Generate answer from the end of the reasoning with pruned cache
    gen_kv = lookback_outputs.past_key_values
    next_token_logits = lookback_outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_ids = [next_token[0, 0].item()]

    for _ in range(150):
        gen_out = model(
            input_ids=next_token,
            past_key_values=gen_kv,
            use_cache=True,
        )
        gen_kv = gen_out.past_key_values
        next_token_logits = gen_out.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
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
    orig_text_losses = reasoning_losses[-(lookback - 1):] if len(reasoning_losses) >= lookback - 1 else reasoning_losses

    del outputs, lookback_outputs, past_kv, gen_kv
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "answer_text": answer_text,
        "answer": extract_answer(answer_text),
        "orig_text_losses": orig_text_losses,
        "pruned_text_losses": pruned_text_losses,
        "mean_orig_text_loss": float(np.mean(orig_text_losses)) if orig_text_losses else 0,
        "mean_pruned_text_loss": float(np.mean(pruned_text_losses)) if pruned_text_losses else 0,
    }


def main():
    start_time = time.time()
    print(f"{'='*60}")
    print(f"Experiment 002: Double Dissociation via KV Position Pruning")
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {NUM_PROBLEMS}")
    print(f"Prune fraction: {PRUNE_FRACTION}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # needed for output_attentions=True
    )
    model.eval()
    print(f"Model loaded in {time.time() - start_time:.1f}s")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GSM8K
    print("Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    # Select problems (deterministic with seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected = indices[:NUM_PROBLEMS]

    all_results = []
    n_correct_baseline = 0
    n_attempted = 0

    for prob_idx, ds_idx in enumerate(selected):
        prob_start = time.time()
        problem = dataset[ds_idx]
        question = problem["question"]
        true_answer = parse_gsm8k_answer(problem["answer"])
        prompt = build_prompt(question)

        print(f"\n--- Problem {prob_idx + 1}/{NUM_PROBLEMS} (GSM8K #{ds_idx}) ---")
        print(f"True answer: {true_answer}")

        # Check time budget (25 min experiment budget within 30 min max)
        elapsed = time.time() - start_time
        if elapsed > 1500:  # 25 minutes
            print(f"\nTime budget reached ({elapsed:.0f}s). Stopping.")
            break

        n_attempted += 1

        # Phase 1: Generate CoT trace
        print("  Generating trace...")
        trace_text, trace_ids = generate_trace(model, tokenizer, prompt)
        generated_answer = extract_answer(trace_text)
        gen_norm = normalize_answer(generated_answer) if generated_answer else ""
        true_norm = normalize_answer(true_answer)
        correct_baseline = (gen_norm == true_norm)
        print(f"  Generated answer: '{generated_answer}' (correct: {correct_baseline})")
        if not correct_baseline:
            print(f"  Trace preview: {trace_text[:200]}...")

        if correct_baseline:
            n_correct_baseline += 1

        if not correct_baseline:
            print("  SKIP: baseline incorrect")
            all_results.append({
                "problem_idx": ds_idx,
                "true_answer": true_answer,
                "baseline_answer": generated_answer,
                "baseline_correct": False,
                "skipped": True,
            })
            continue

        # Split trace into reasoning (before ####) and answer
        if "####" in trace_text:
            reasoning_text = trace_text[:trace_text.index("####")]
        else:
            reasoning_text = trace_text
        print(f"  Reasoning text length: {len(reasoning_text)} chars")

        # Phase 2: Extract attention patterns (using full trace for attention)
        print("  Extracting attention patterns...")
        try:
            attn_info = teacher_force_with_attention(model, tokenizer, prompt, reasoning_text)
        except torch.cuda.OutOfMemoryError:
            print("  SKIP: OOM during attention extraction")
            gc.collect()
            torch.cuda.empty_cache()
            all_results.append({
                "problem_idx": ds_idx,
                "true_answer": true_answer,
                "baseline_answer": generated_answer,
                "baseline_correct": True,
                "skipped": True,
                "skip_reason": "OOM",
            })
            continue

        if attn_info is None:
            print("  SKIP: trace too short")
            all_results.append({
                "problem_idx": ds_idx,
                "baseline_correct": True,
                "skipped": True,
                "skip_reason": "trace_too_short",
            })
            continue

        answer_coupling = attn_info["answer_coupling"]
        text_coupling = attn_info["text_coupling"]
        prompt_len = attn_info["prompt_len"]
        reasoning_len = attn_info["reasoning_len"]

        print(f"  Reasoning length: {reasoning_len} tokens")
        print(f"  Answer coupling: mean={answer_coupling.mean():.4f}, max={answer_coupling.max():.4f}")
        print(f"  Text coupling: mean={text_coupling.mean():.4f}, max={text_coupling.max():.4f}")

        # Check overlap between top answer-coupled and text-coupled positions
        n_prune = max(1, int(reasoning_len * PRUNE_FRACTION))
        answer_coupled_positions = np.argsort(answer_coupling)[-n_prune:].tolist()
        text_coupled_positions = np.argsort(text_coupling)[-n_prune:].tolist()
        overlap = len(set(answer_coupled_positions) & set(text_coupled_positions))
        print(f"  Pruning {n_prune} positions per condition (overlap: {overlap})")

        all_positions = list(range(reasoning_len))
        random_positions = random.sample(all_positions, n_prune)

        # Phase 3: Pruning experiments
        result = {
            "problem_idx": ds_idx,
            "true_answer": true_answer,
            "baseline_answer": generated_answer,
            "baseline_correct": True,
            "skipped": False,
            "reasoning_len": reasoning_len,
            "n_pruned": n_prune,
            "position_overlap": overlap,
            "answer_coupling_stats": {
                "mean": float(answer_coupling.mean()),
                "std": float(answer_coupling.std()),
                "max": float(answer_coupling.max()),
            },
            "text_coupling_stats": {
                "mean": float(text_coupling.mean()),
                "std": float(text_coupling.std()),
                "max": float(text_coupling.max()),
            },
        }

        conditions = {
            "no_pruning": [],
            "prune_answer_coupled": answer_coupled_positions,
            "prune_text_coupled": text_coupled_positions,
            "prune_random": random_positions,
        }

        for cond_name, positions in conditions.items():
            print(f"  Running condition: {cond_name} ({len(positions)} positions)...")
            try:
                cond_result = pruned_generation(
                    model, tokenizer, prompt, reasoning_text,
                    positions, prompt_len
                )
                cond_answer = cond_result["answer"]
                cond_norm = normalize_answer(cond_answer) if cond_answer else ""
                cond_correct = (cond_norm == true_norm)
                print(f"    Answer: '{cond_answer}' (correct: {cond_correct})")
                print(f"    Text loss: orig={cond_result['mean_orig_text_loss']:.3f}, pruned={cond_result['mean_pruned_text_loss']:.3f}")

                result[cond_name] = {
                    "answer": cond_answer,
                    "correct": cond_correct,
                    "mean_orig_text_loss": cond_result["mean_orig_text_loss"],
                    "mean_pruned_text_loss": cond_result["mean_pruned_text_loss"],
                    "text_loss_change": cond_result["mean_pruned_text_loss"] - cond_result["mean_orig_text_loss"],
                }
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM in condition {cond_name}")
                gc.collect()
                torch.cuda.empty_cache()
                result[cond_name] = {"error": "OOM"}
            except Exception as e:
                print(f"    Error in condition {cond_name}: {e}")
                traceback.print_exc()
                result[cond_name] = {"error": str(e)}

        all_results.append(result)
        elapsed = time.time() - prob_start
        print(f"  Problem done in {elapsed:.1f}s")

        # Save intermediate results
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=_convert)

    # ── Aggregate Analysis ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    print(f"Baseline accuracy: {n_correct_baseline}/{n_attempted} ({n_correct_baseline/max(1,n_attempted):.0%})")

    valid = [r for r in all_results if not r.get("skipped", False)]
    print(f"Valid problems (baseline correct, no OOM): {len(valid)}/{len(all_results)}")

    if not valid:
        print("No valid results to analyze!")
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump({"error": "no_valid_results", "all_results": all_results,
                       "baseline_accuracy": f"{n_correct_baseline}/{n_attempted}"}, f, indent=2, default=_convert)
        return

    # Compute accuracy per condition
    conditions_list = ["no_pruning", "prune_answer_coupled", "prune_text_coupled", "prune_random"]
    summary = {}
    for cond in conditions_list:
        correct = sum(1 for r in valid if r.get(cond, {}).get("correct", False))
        error = sum(1 for r in valid if "error" in r.get(cond, {}))
        n_valid_cond = len(valid) - error
        acc = correct / n_valid_cond if n_valid_cond > 0 else 0

        text_loss_changes = [
            r[cond]["text_loss_change"]
            for r in valid
            if cond in r and "text_loss_change" in r.get(cond, {})
        ]
        mean_text_loss_change = float(np.mean(text_loss_changes)) if text_loss_changes else 0
        std_text_loss_change = float(np.std(text_loss_changes)) if len(text_loss_changes) > 1 else 0

        summary[cond] = {
            "accuracy": acc,
            "correct": correct,
            "total": n_valid_cond,
            "errors": error,
            "mean_text_loss_change": mean_text_loss_change,
            "std_text_loss_change": std_text_loss_change,
            "text_loss_changes": text_loss_changes,
        }

        print(f"\n{cond}:")
        print(f"  Accuracy: {correct}/{n_valid_cond} ({acc:.1%})")
        print(f"  Mean text loss change: {mean_text_loss_change:+.4f} (std: {std_text_loss_change:.4f})")

    # ── Double dissociation check ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("DOUBLE DISSOCIATION TEST")
    print(f"{'='*60}")

    ac_acc = summary["prune_answer_coupled"]["accuracy"]
    tc_acc = summary["prune_text_coupled"]["accuracy"]
    rnd_acc = summary["prune_random"]["accuracy"]
    ctrl_acc = summary["no_pruning"]["accuracy"]

    ac_text = summary["prune_answer_coupled"]["mean_text_loss_change"]
    tc_text = summary["prune_text_coupled"]["mean_text_loss_change"]
    rnd_text = summary["prune_random"]["mean_text_loss_change"]
    ctrl_text = summary["no_pruning"]["mean_text_loss_change"]

    print(f"\nAccuracy: control={ctrl_acc:.1%}, AC-pruned={ac_acc:.1%}, TC-pruned={tc_acc:.1%}, random={rnd_acc:.1%}")
    print(f"Text loss change: control={ctrl_text:+.4f}, AC-pruned={ac_text:+.4f}, TC-pruned={tc_text:+.4f}, random={rnd_text:+.4f}")

    dissociation_accuracy = ac_acc < tc_acc
    dissociation_text = tc_text > ac_text
    double_dissociation = dissociation_accuracy and dissociation_text

    print(f"\nAccuracy dissociation (AC drops more): {dissociation_accuracy}")
    print(f"Text loss dissociation (TC increases more): {dissociation_text}")
    print(f"DOUBLE DISSOCIATION: {double_dissociation}")

    # Effect sizes
    acc_effect = tc_acc - ac_acc  # positive means AC hurts more
    text_effect = tc_text - ac_text  # positive means TC hurts more
    print(f"\nEffect sizes:")
    print(f"  Accuracy effect (TC - AC): {acc_effect:+.3f}")
    print(f"  Text loss effect (TC - AC): {text_effect:+.4f}")

    # Save full summary
    full_summary = {
        "experiment": "exp_002_double_dissociation",
        "model": MODEL_NAME,
        "num_problems": NUM_PROBLEMS,
        "num_attempted": n_attempted,
        "num_valid": len(valid),
        "baseline_accuracy": n_correct_baseline / max(1, n_attempted),
        "prune_fraction": PRUNE_FRACTION,
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "text_loss_changes"} for k, v in summary.items()},
        "double_dissociation": {
            "accuracy_dissociation": dissociation_accuracy,
            "text_dissociation": dissociation_text,
            "full_dissociation": double_dissociation,
            "accuracy_effect_size": acc_effect,
            "text_loss_effect_size": text_effect,
        },
        "elapsed_seconds": time.time() - start_time,
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(full_summary, f, indent=2, default=_convert)

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=_convert)

    # ── Generate Figures ────────────────────────────────────────────────
    generate_figures(summary, valid, conditions_list, RESULTS_DIR)

    total_time = time.time() - start_time
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")

    print(f"\n{'='*60}")
    print("SUMMARY FOR LOG")
    print(f"{'='*60}")
    print(f"Double dissociation: {'YES' if double_dissociation else 'NO'}")
    print(f"Baseline accuracy: {n_correct_baseline}/{n_attempted} ({n_correct_baseline/max(1,n_attempted):.0%})")
    print(f"Valid problems: {len(valid)}")
    print(f"Accuracy — control: {ctrl_acc:.0%}, AC-pruned: {ac_acc:.0%}, TC-pruned: {tc_acc:.0%}, random: {rnd_acc:.0%}")
    print(f"Text loss Δ — AC-pruned: {ac_text:+.4f}, TC-pruned: {tc_text:+.4f}, random: {rnd_text:+.4f}")


def generate_figures(summary, valid, conditions_list, results_dir):
    """Generate all experiment figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctrl_acc = summary["no_pruning"]["accuracy"]
    ac_acc = summary["prune_answer_coupled"]["accuracy"]
    tc_acc = summary["prune_text_coupled"]["accuracy"]
    rnd_acc = summary["prune_random"]["accuracy"]

    ctrl_text = summary["no_pruning"]["mean_text_loss_change"]
    ac_text = summary["prune_answer_coupled"]["mean_text_loss_change"]
    tc_text = summary["prune_text_coupled"]["mean_text_loss_change"]
    rnd_text = summary["prune_random"]["mean_text_loss_change"]

    # Figure 1: Double dissociation summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cond_labels = ["Control", "Answer-\ncoupled", "Text-\ncoupled", "Random"]
    accuracies = [ctrl_acc, ac_acc, tc_acc, rnd_acc]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6"]

    axes[0].bar(cond_labels, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Answer Accuracy")
    axes[0].set_title("Answer Accuracy by Pruning Condition")
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")

    text_changes = [ctrl_text, ac_text, tc_text, rnd_text]
    axes[1].bar(cond_labels, text_changes, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Text Loss Change (nats)")
    axes[1].set_title("Text Prediction Loss Change by Pruning Condition")
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    for i, v in enumerate(text_changes):
        offset = 0.002 if v >= 0 else -0.005
        axes[1].text(i, v + offset, f"{v:+.3f}", ha="center", fontsize=9)

    fig.suptitle(f"Exp 002: Double Dissociation (n={len(valid)} problems, Qwen3-4B-Base)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "double_dissociation_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: double_dissociation_summary.png")

    # Figure 2: Per-problem results
    if len(valid) >= 3:
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        x = np.arange(len(valid))
        width = 0.25

        for i, (cond, color, label) in enumerate([
            ("prune_answer_coupled", "#e74c3c", "Answer-coupled"),
            ("prune_text_coupled", "#3498db", "Text-coupled"),
            ("prune_random", "#95a5a6", "Random"),
        ]):
            acc_vals = [1 if r.get(cond, {}).get("correct", False) else 0 for r in valid]
            axes2[0].bar(x + (i - 1) * width, acc_vals, width, color=color, label=label, alpha=0.8)

            tlc_vals = [r.get(cond, {}).get("text_loss_change", 0) for r in valid]
            axes2[1].bar(x + (i - 1) * width, tlc_vals, width, color=color, label=label, alpha=0.8)

        axes2[0].set_ylabel("Correct (1/0)")
        axes2[0].set_title("Per-Problem Answer Accuracy")
        axes2[0].legend()
        axes2[0].set_yticks([0, 1])

        axes2[1].set_ylabel("Text Loss Change (nats)")
        axes2[1].set_title("Per-Problem Text Loss Change")
        axes2[1].set_xlabel("Problem index")
        axes2[1].axhline(y=0, color="black", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        fig2.savefig(os.path.join(results_dir, "per_problem_results.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Figure saved: per_problem_results.png")

    # Figure 3: Text loss change distributions
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    for cond, color, label in [
        ("prune_answer_coupled", "#e74c3c", "Answer-coupled"),
        ("prune_text_coupled", "#3498db", "Text-coupled"),
        ("prune_random", "#95a5a6", "Random"),
    ]:
        vals = summary[cond].get("text_loss_changes", [])
        if vals:
            ax3.hist(vals, bins=15, alpha=0.5, color=color, label=label, edgecolor="black", linewidth=0.5)
    ax3.set_xlabel("Text Loss Change (nats)")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of Text Prediction Loss Changes")
    ax3.legend()
    ax3.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
    fig3.savefig(os.path.join(results_dir, "text_loss_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: text_loss_distribution.png")


def _convert(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()
