#!/usr/bin/env python3
"""
Experiment 001: Double Dissociation via KV Position Pruning

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

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B"
NUM_PROBLEMS = 20
PRUNE_FRACTION = 0.25  # prune top quartile
ATTENTION_LAYERS = [-1, -2, -3, -4]  # last 4 layers for attention analysis
MAX_GEN_TOKENS = 512
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_001")

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
        "question": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load does it take to download the file?",
        "answer": "First find how long it takes to download 40% of the file: 200 GB * 0.4 / 2 GB/minute = <<200*0.4/2=40>>40 minutes\nThen find how long it takes to download the whole file once the restart is complete: 200 GB / 2 GB/minute = <<200/2=100>>100 minutes\nThen add the time to download 40% of the file, the restart time, and the time to download the whole file: 40 + 20 + 100 = <<40+20+100=160>>160 minutes\n#### 160"
    },
]


def build_prompt(question: str) -> str:
    """Build 8-shot CoT prompt for a GSM8K question."""
    prompt = ""
    for ex in GSM8K_EXEMPLARS:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    prompt += f"Q: {question}\nA: Let's think step by step.\n"
    return prompt


def extract_answer(text: str) -> str:
    """Extract the numeric answer after ####."""
    if "####" in text:
        ans = text.split("####")[-1].strip()
        # Remove commas and dollar signs
        ans = ans.replace(",", "").replace("$", "").strip()
        return ans
    return ""


def parse_gsm8k_answer(answer_text: str) -> str:
    """Parse ground truth answer from GSM8K."""
    if "####" in answer_text:
        ans = answer_text.split("####")[-1].strip()
        ans = ans.replace(",", "").replace("$", "").strip()
        return ans
    return ""


@torch.no_grad()
def generate_trace(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS):
    """Generate a CoT trace and return the full text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=1.0,
    )
    generated_ids = outputs[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, generated_ids


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

    # Forward pass with attention
    outputs = model(
        **inputs,
        output_attentions=True,
        use_cache=False,
    )

    # Extract attention from last 4 layers
    # Each attention tensor: (batch, num_heads, seq_len, seq_len)
    num_layers = len(outputs.attentions)
    layer_indices = [num_layers + i for i in ATTENTION_LAYERS]

    # Answer-coupling: attention from the LAST token to each reasoning position
    last_token_idx = seq_len - 1

    answer_coupling = torch.zeros(reasoning_len, device=model.device)
    text_coupling = torch.zeros(reasoning_len, device=model.device)

    for li in layer_indices:
        attn = outputs.attentions[li][0]  # (num_heads, seq_len, seq_len)

        # Answer coupling: sum of attention from last token to each reasoning pos
        # attn[:, last_token_idx, prompt_len:seq_len] → (num_heads, reasoning_len)
        answer_coupling += attn[:, last_token_idx, prompt_len:seq_len].sum(dim=0)

        # Text coupling: for each reasoning position, average attention FROM
        # later reasoning tokens TO this position (vectorized)
        # reasoning_attn[h, i, j] = attention from reasoning pos i to reasoning pos j
        reasoning_attn = attn[:, prompt_len:seq_len, prompt_len:seq_len]  # (H, R, R)
        # Sum over heads: (R, R) where [i,j] = total attn from pos i to pos j
        reasoning_attn_sum = reasoning_attn.sum(dim=0)  # (R, R)
        # For each column j (source position), sum attention from rows i > j
        # Use lower triangular mask (rows > cols)
        mask = torch.tril(torch.ones(reasoning_len, reasoning_len, device=model.device), diagonal=-1)
        # mask[i,j] = 1 if i > j (later tokens attending to earlier positions)
        weighted = reasoning_attn_sum * mask  # (R, R)
        col_sums = weighted.sum(dim=0)  # (R,) — total attention TO each position from later positions
        col_counts = mask.sum(dim=0).clamp(min=1)  # number of later tokens per position
        text_coupling += col_sums / col_counts

    num_layers_used = len(layer_indices)
    num_heads = attn.shape[0]
    answer_coupling = answer_coupling / (num_layers_used * num_heads)
    text_coupling = text_coupling / (num_layers_used * num_heads)

    # Free attention memory
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
def pruned_generation(model, tokenizer, prompt, trace_text, positions_to_prune, prompt_len):
    """
    Teacher-force prompt + trace with KV cache, then prune specific
    reasoning positions from the cache, then generate the answer.

    positions_to_prune: list of reasoning-relative positions (0-indexed from
    start of reasoning, will be offset by prompt_len)

    Returns: generated answer text, per-token losses for reasoning tokens
    """
    full_text = prompt + trace_text
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
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compute per-token cross-entropy loss for reasoning tokens
    # Loss at position t = -log P(token_t | tokens_<t)
    reasoning_losses = []
    for t in range(reasoning_len - 1):
        pos = prompt_len + t
        target_id = inputs.input_ids[0, pos + 1].item()
        log_probs = torch.log_softmax(logits[0, pos], dim=-1)
        loss = -log_probs[target_id].item()
        reasoning_losses.append(loss)

    # Prune KV cache at specified positions
    # KV cache structure: tuple of (key, value) per layer
    # Each key/value: (batch, num_heads, seq_len, head_dim)
    abs_positions = [prompt_len + p for p in positions_to_prune if prompt_len + p < seq_len]

    if abs_positions:
        pruned_kv = []
        for layer_kv in past_kv:
            key, value = layer_kv[0].clone(), layer_kv[1].clone()
            for pos in abs_positions:
                key[:, :, pos, :] = 0.0
                value[:, :, pos, :] = 0.0
            pruned_kv.append((key, value))
        past_kv = tuple(pruned_kv)

    # Now compute loss for reasoning tokens UNDER PRUNED cache
    # Re-run forward pass with pruned cache to get new logits
    # We need to process token-by-token to measure the effect of pruning
    # on text prediction. But that's expensive. Instead, we can do a single
    # forward pass with the pruned cache applied.
    # Actually, we need to re-run the full sequence through the pruned cache.
    # But the cache IS the result of processing the sequence - we can't just
    # swap the cache and get new logits. The logits came from the original
    # forward pass.
    #
    # The right approach: the pruned cache affects FUTURE token predictions.
    # So we measure the effect on the ANSWER token, not on reasoning tokens.
    #
    # For text prediction loss under pruning, we'd need to re-run teacher-forcing
    # with the pruned cache at each step. This is expensive but doable:
    # We re-process just the last few reasoning tokens + answer generation
    # using the pruned prefix cache.

    # Generate answer with pruned cache
    # Feed the last token again to kick off generation from the pruned cache
    last_token = inputs.input_ids[:, -1:]
    generated_ids = []

    # Re-compute logits for the last reasoning token using pruned cache
    # to also measure text prediction impact
    # We'll re-process the last 20 reasoning tokens to measure text loss change
    lookback = min(20, reasoning_len)
    lookback_start = seq_len - lookback
    lookback_tokens = inputs.input_ids[:, lookback_start:seq_len]

    # Build a truncated cache (everything before lookback_start)
    trunc_kv = []
    for layer_kv in past_kv:
        key = layer_kv[0][:, :, :lookback_start, :]
        value = layer_kv[1][:, :, :lookback_start, :]
        trunc_kv.append((key, value))
    trunc_kv = tuple(trunc_kv)

    # Re-process lookback tokens with truncated (pruned) prefix cache
    lookback_outputs = model(
        input_ids=lookback_tokens,
        past_key_values=trunc_kv,
        use_cache=True,
        output_attentions=False,
    )

    # Compute losses for these lookback tokens
    pruned_text_losses = []
    for t in range(lookback - 1):
        target_id = lookback_tokens[0, t + 1].item()
        log_probs = torch.log_softmax(lookback_outputs.logits[0, t], dim=-1)
        loss = -log_probs[target_id].item()
        pruned_text_losses.append(loss)

    # Now generate answer from the end of the trace
    gen_kv = lookback_outputs.past_key_values
    next_token_logits = lookback_outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_ids.append(next_token[0, 0].item())

    # Continue generating
    for _ in range(100):
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
        # Stop on EOS or ####
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "####" in decoded or token_id == tokenizer.eos_token_id:
            break

    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Also compute original (unpruned) text losses for the lookback window
    orig_text_losses = reasoning_losses[-(lookback - 1):]

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
    print(f"Experiment 001: Double Dissociation via KV Position Pruning")
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
    )
    model.eval()
    print(f"Model loaded in {time.time() - start_time:.1f}s")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GSM8K
    print("Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    # Select problems
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected = indices[:NUM_PROBLEMS]

    all_results = []

    for prob_idx, ds_idx in enumerate(selected):
        prob_start = time.time()
        problem = dataset[ds_idx]
        question = problem["question"]
        true_answer = parse_gsm8k_answer(problem["answer"])
        prompt = build_prompt(question)

        print(f"\n--- Problem {prob_idx + 1}/{NUM_PROBLEMS} (GSM8K #{ds_idx}) ---")
        print(f"True answer: {true_answer}")

        # Phase 1: Generate CoT trace
        print("  Generating trace...")
        trace_text, trace_ids = generate_trace(model, tokenizer, prompt)
        generated_answer = extract_answer(trace_text)
        correct_baseline = (generated_answer == true_answer)
        print(f"  Generated answer: {generated_answer} (correct: {correct_baseline})")

        if not correct_baseline:
            print("  SKIP: baseline incorrect, can't test pruning effect")
            all_results.append({
                "problem_idx": ds_idx,
                "true_answer": true_answer,
                "baseline_answer": generated_answer,
                "baseline_correct": False,
                "skipped": True,
            })
            continue

        # Phase 2: Extract attention patterns
        print("  Extracting attention patterns...")
        try:
            attn_info = teacher_force_with_attention(model, tokenizer, prompt, trace_text)
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

        # Phase 3: Classify positions
        n_prune = max(1, int(reasoning_len * PRUNE_FRACTION))

        # Answer-coupled: top positions by answer_coupling
        answer_coupled_positions = np.argsort(answer_coupling)[-n_prune:].tolist()
        # Text-coupled: top positions by text_coupling
        text_coupled_positions = np.argsort(text_coupling)[-n_prune:].tolist()
        # Random control
        all_positions = list(range(reasoning_len))
        random_positions = random.sample(all_positions, n_prune)

        # Phase 4: Pruning experiments
        result = {
            "problem_idx": ds_idx,
            "true_answer": true_answer,
            "baseline_answer": generated_answer,
            "baseline_correct": True,
            "skipped": False,
            "reasoning_len": reasoning_len,
            "n_pruned": n_prune,
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
                    model, tokenizer, prompt, trace_text,
                    positions, prompt_len
                )
                cond_answer = cond_result["answer"]
                cond_correct = (cond_answer == true_answer)
                print(f"    Answer: {cond_answer} (correct: {cond_correct})")
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
        print(f"  Problem done in {time.time() - prob_start:.1f}s")

        # Save intermediate results
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    # ── Aggregate Analysis ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    valid = [r for r in all_results if not r.get("skipped", False)]
    print(f"Valid problems: {len(valid)}/{len(all_results)}")

    if not valid:
        print("No valid results to analyze!")
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump({"error": "no_valid_results", "all_results": all_results}, f, indent=2)
        return

    # Compute accuracy per condition
    conditions = ["no_pruning", "prune_answer_coupled", "prune_text_coupled", "prune_random"]
    summary = {}
    for cond in conditions:
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

        summary[cond] = {
            "accuracy": acc,
            "correct": correct,
            "total": n_valid_cond,
            "errors": error,
            "mean_text_loss_change": mean_text_loss_change,
            "text_loss_changes": text_loss_changes,
        }

        print(f"\n{cond}:")
        print(f"  Accuracy: {correct}/{n_valid_cond} ({acc:.1%})")
        print(f"  Mean text loss change: {mean_text_loss_change:+.4f}")

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

    print(f"\nAccuracy: control={ctrl_acc:.1%}, answer-coupled={ac_acc:.1%}, text-coupled={tc_acc:.1%}, random={rnd_acc:.1%}")
    print(f"Text loss change: control={ctrl_text:+.4f}, answer-coupled={ac_text:+.4f}, text-coupled={tc_text:+.4f}, random={rnd_text:+.4f}")

    # Dissociation criteria:
    # 1. Pruning answer-coupled: accuracy drops MORE than text-coupled
    # 2. Pruning text-coupled: text loss increases MORE than answer-coupled
    dissociation_accuracy = ac_acc < tc_acc
    dissociation_text = tc_text > ac_text
    double_dissociation = dissociation_accuracy and dissociation_text

    print(f"\nAccuracy dissociation (AC drops more): {dissociation_accuracy}")
    print(f"Text loss dissociation (TC increases more): {dissociation_text}")
    print(f"DOUBLE DISSOCIATION: {double_dissociation}")

    # Save full summary
    full_summary = {
        "experiment": "exp_001_double_dissociation",
        "model": MODEL_NAME,
        "num_problems": NUM_PROBLEMS,
        "num_valid": len(valid),
        "prune_fraction": PRUNE_FRACTION,
        "conditions": summary,
        "double_dissociation": {
            "accuracy_dissociation": dissociation_accuracy,
            "text_dissociation": dissociation_text,
            "full_dissociation": double_dissociation,
        },
        "elapsed_seconds": time.time() - start_time,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(full_summary, f, indent=2, default=convert)

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    # ── Generate Figures ────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: Accuracy by condition
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cond_labels = ["Control", "Answer-\ncoupled", "Text-\ncoupled", "Random"]
    accuracies = [ctrl_acc, ac_acc, tc_acc, rnd_acc]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6"]

    axes[0].bar(cond_labels, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Answer Accuracy")
    axes[0].set_title("Answer Accuracy by Pruning Condition")
    axes[0].set_ylim(0, 1.05)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)

    # Figure 2: Text loss change by condition
    text_changes = [ctrl_text, ac_text, tc_text, rnd_text]
    axes[1].bar(cond_labels, text_changes, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Text Loss Change (nats)")
    axes[1].set_title("Text Prediction Loss Change by Pruning Condition")
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    for i, v in enumerate(text_changes):
        axes[1].text(i, v + 0.001 if v >= 0 else v - 0.003, f"{v:+.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "double_dissociation_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {os.path.join(RESULTS_DIR, 'double_dissociation_summary.png')}")

    # Figure 3: Per-problem accuracy scatter
    if len(valid) >= 3:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        for i, r in enumerate(valid):
            for cond, marker, color, label in [
                ("prune_answer_coupled", "v", "#e74c3c", "Answer-coupled"),
                ("prune_text_coupled", "^", "#3498db", "Text-coupled"),
                ("prune_random", "s", "#95a5a6", "Random"),
            ]:
                if cond in r and "correct" in r[cond]:
                    y = 1 if r[cond]["correct"] else 0
                    ax2.scatter(i, y, marker=marker, c=color, s=60,
                               label=label if i == 0 else None, alpha=0.7)

        ax2.set_xlabel("Problem index")
        ax2.set_ylabel("Correct (1) / Incorrect (0)")
        ax2.set_title("Per-Problem Accuracy Under Pruning")
        ax2.legend()
        ax2.set_yticks([0, 1])
        fig2.savefig(os.path.join(RESULTS_DIR, "per_problem_accuracy.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Figure 4: Distribution of text loss changes
    if any(summary[c]["text_loss_changes"] for c in conditions):
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
        for cond, color, label in [
            ("prune_answer_coupled", "#e74c3c", "Answer-coupled"),
            ("prune_text_coupled", "#3498db", "Text-coupled"),
            ("prune_random", "#95a5a6", "Random"),
        ]:
            vals = summary[cond]["text_loss_changes"]
            if vals:
                ax3.hist(vals, bins=15, alpha=0.5, color=color, label=label, edgecolor="black", linewidth=0.5)
        ax3.set_xlabel("Text Loss Change (nats)")
        ax3.set_ylabel("Count")
        ax3.set_title("Distribution of Text Prediction Loss Changes")
        ax3.legend()
        ax3.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
        fig3.savefig(os.path.join(RESULTS_DIR, "text_loss_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close()

    total_time = time.time() - start_time
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Print concise summary for experiment log
    print(f"\n{'='*60}")
    print("SUMMARY FOR LOG")
    print(f"{'='*60}")
    print(f"Double dissociation: {'YES' if double_dissociation else 'NO'}")
    print(f"Accuracy — control: {ctrl_acc:.0%}, AC-pruned: {ac_acc:.0%}, TC-pruned: {tc_acc:.0%}, random: {rnd_acc:.0%}")
    print(f"Text loss change — AC-pruned: {ac_text:+.4f}, TC-pruned: {tc_text:+.4f}, random: {rnd_text:+.4f}")


if __name__ == "__main__":
    main()
