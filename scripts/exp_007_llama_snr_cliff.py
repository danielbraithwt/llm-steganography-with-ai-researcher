#!/usr/bin/env python3
"""
Experiment 007: SNR Cliff Replication on Llama-3.1-8B (retry of exp_006)

Exp_006 and original exp_007 failed: 'DynamicCache' object is not subscriptable.
Root cause: transformers 5.3.0 changed DynamicCache API. key_cache/value_cache
attributes no longer exist. Access is via past_kv.layers[i].keys/.values.

Fixes in this version (cycle 8):
1. attn_implementation="eager" (required for KV cache manipulation)
2. DynamicCache API: past_kv.layers[i].keys/values instead of key_cache/value_cache
3. DynamicCache() + .update() for truncated cache instead of tuple-of-tuples
4. In-place KV cache modification with restore from cloned clean tensors
5. Smoke test before full run to catch bugs early

Tests whether the sharp KV cache SNR cliff (~14 dB on Qwen, 3 dB transition
window) replicates on Llama-3.1-8B.
"""

import os
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
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NUM_PROBLEMS = 30
# Fine-grained sampling around expected cliff (predict higher than Qwen's 14 dB)
SNR_LEVELS_DB = [0, 5, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25]
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 1536
LOOKBACK = 20  # tokens for re-processing under noised context
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_007")

os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 8-shot GSM8K exemplars ──────────────────────────────────────────────
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


CHUNK_SIZE = 128  # Process this many tokens at a time to avoid OOM with eager attention


@torch.no_grad()
def teacher_force_chunked(model, input_ids, chunk_size=CHUNK_SIZE):
    """Teacher-force a sequence in chunks to avoid OOM from large attention matrices."""
    seq_len = input_ids.shape[1]
    past_kv = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        if past_kv is not None:
            outputs = model(input_ids=chunk, past_key_values=past_kv, use_cache=True)
        else:
            outputs = model(input_ids=chunk, use_cache=True)
        past_kv = outputs.past_key_values
        del outputs
    return past_kv


@torch.no_grad()
def evaluate_at_snr(model, past_kv, num_layers, input_ids, prompt_len, seq_len,
                    true_answer, snr_db, tokenizer, clean_keys, clean_values):
    """
    Evaluate model at a given SNR level.

    Strategy:
    1. Restore clean KV cache from saved clones
    2. Add calibrated Gaussian noise to reasoning KV entries in-place
    3. Build truncated DynamicCache for lookback context
    4. Re-process lookback tokens with noised context -> token accuracy
    5. Continue generating freely -> answer accuracy
    """
    # Step 1: Restore clean cache (transformers 5.3.0 API: .layers[i].keys/.values)
    for layer_idx in range(num_layers):
        past_kv.layers[layer_idx].keys.copy_(clean_keys[layer_idx])
        past_kv.layers[layer_idx].values.copy_(clean_values[layer_idx])

    # Step 2: Add noise to reasoning positions (prompt_len to seq_len)
    noise_ratio = 10.0 ** (-snr_db / 20.0)

    for layer_idx in range(num_layers):
        k = past_kv.layers[layer_idx].keys
        v = past_kv.layers[layer_idx].values

        rk = k[:, :, prompt_len:seq_len, :]
        rv = v[:, :, prompt_len:seq_len, :]

        k_norm = rk.norm().item()
        v_norm = rv.norm().item()

        kn = torch.randn_like(rk)
        vn = torch.randn_like(rv)

        k_scale = k_norm * noise_ratio / (kn.norm().item() + 1e-8)
        v_scale = v_norm * noise_ratio / (vn.norm().item() + 1e-8)

        rk.add_(kn * k_scale)
        rv.add_(vn * v_scale)

    # Step 3: Build truncated DynamicCache
    lookback = min(LOOKBACK, seq_len - prompt_len)
    lookback_start = max(prompt_len, seq_len - lookback)

    trunc_kv = DynamicCache()
    for i in range(num_layers):
        trunc_kv.update(
            past_kv.layers[i].keys[:, :, :lookback_start, :].clone(),
            past_kv.layers[i].values[:, :, :lookback_start, :].clone(),
            i,
        )

    # Step 4: Re-process lookback tokens with noised context
    lookback_tokens = input_ids[:, lookback_start:seq_len]
    lookback_out = model(
        input_ids=lookback_tokens,
        past_key_values=trunc_kv,
        use_cache=True,
        output_attentions=False,
    )

    # Token accuracy
    token_correct = 0
    token_total = 0
    for t in range(lookback_tokens.shape[1] - 1):
        predicted = torch.argmax(lookback_out.logits[0, t], dim=-1).item()
        actual = lookback_tokens[0, t + 1].item()
        if predicted == actual:
            token_correct += 1
        token_total += 1
    token_accuracy = token_correct / token_total if token_total > 0 else 0.0

    # Step 5: Generate answer freely
    gen_kv = lookback_out.past_key_values
    next_logits = lookback_out.logits[:, -1, :]
    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
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
            if re.search(r'\d+', after):
                break
        if re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?\s*-?[\d,]+', decoded):
            break
        if "\nQ:" in decoded:
            break

    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    gen_answer = extract_answer(answer_text)
    gen_norm = normalize_answer(gen_answer) if gen_answer else ""
    answer_correct = (gen_norm == true_answer)

    del lookback_out, gen_kv
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "answer_correct": answer_correct,
        "token_accuracy": token_accuracy,
        "generated_answer": gen_answer,
    }


def main():
    start_time = time.time()
    print(f"{'='*70}")
    print(f"Experiment 007: SNR Cliff Replication on Llama-3.1-8B")
    print(f"(Retry of exp_006 with DynamicCache bug fix)")
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {NUM_PROBLEMS}")
    print(f"SNR levels (dB): {SNR_LEVELS_DB}")
    print(f"Lookback: {LOOKBACK} tokens")
    print(f"{'='*70}\n")

    # Load model with attn_implementation="eager" (critical for KV cache manipulation)
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Required for DynamicCache compatibility
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded in {time.time() - start_time:.1f}s")
    print(f"Layers: {num_layers}, "
          f"Heads: {model.config.num_attention_heads}, "
          f"KV heads: {getattr(model.config, 'num_key_value_heads', 'N/A')}")

    # Load GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected = indices[:NUM_PROBLEMS]

    # ── Phase 1: Generate traces and verify baseline correctness ──────
    print("\n=== Phase 1: Generating clean traces ===")
    problems_data = []

    for prob_idx, ds_idx in enumerate(selected):
        elapsed = time.time() - start_time
        if elapsed > 600:  # 10 min budget for phase 1
            print(f"Phase 1 time budget reached at problem {prob_idx}")
            break

        problem = dataset[ds_idx]
        question = problem["question"]
        true_answer = normalize_answer(
            problem["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
        )
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

        # Get reasoning portion
        if "####" in trace_text:
            reasoning_text = trace_text[:trace_text.index("####")]
        else:
            reasoning_text = trace_text

        # Compute lengths
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]

        full_text = prompt + reasoning_text
        full_inputs = tokenizer(full_text, return_tensors="pt")
        full_seq_len = full_inputs.input_ids.shape[1]
        reasoning_len = full_seq_len - prompt_len

        if reasoning_len < 10:
            print("  SKIP: reasoning too short")
            continue

        if full_seq_len > MAX_SEQ_LEN:
            # Truncate reasoning to fit
            max_reasoning = MAX_SEQ_LEN - prompt_len
            trace_tokens = tokenizer(reasoning_text, return_tensors="pt").input_ids[0][:max_reasoning]
            reasoning_text = tokenizer.decode(trace_tokens, skip_special_tokens=True)
            full_text = prompt + reasoning_text
            full_inputs = tokenizer(full_text, return_tensors="pt")
            full_seq_len = full_inputs.input_ids.shape[1]
            reasoning_len = full_seq_len - prompt_len

        problems_data.append({
            "ds_idx": ds_idx,
            "true_answer": true_answer,
            "prompt": prompt,
            "reasoning_text": reasoning_text,
            "prompt_len": prompt_len,
            "reasoning_len": reasoning_len,
            "seq_len": full_seq_len,
        })
        print(f"  Valid! reasoning_len={reasoning_len}, seq_len={full_seq_len}")

    n_valid = len(problems_data)
    print(f"\n=== Phase 1 complete: {n_valid} valid problems ===")
    print(f"Phase 1 elapsed: {time.time() - start_time:.0f}s")

    if n_valid < 3:
        print("ERROR: Too few valid problems!")
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump({"error": "too_few_valid", "n_valid": n_valid}, f, indent=2)
        return

    # ── Smoke test ─────────────────────────────────────────────────────
    print("\n=== Smoke test (1 problem, SNR=20 dB) ===")
    pd0 = problems_data[0]
    full_text = pd0["prompt"] + pd0["reasoning_text"]
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    past_kv = teacher_force_chunked(model, inputs.input_ids)
    print(f"  KV cache type: {type(past_kv).__name__}")
    print(f"  Num layers in cache: {len(past_kv.layers)}")
    print(f"  Key shape (layer 0): {past_kv.layers[0].keys.shape}")

    # Test DynamicCache truncation
    test_trunc = DynamicCache()
    for i in range(num_layers):
        test_trunc.update(
            past_kv.layers[i].keys[:, :, :pd0["prompt_len"], :].clone(),
            past_kv.layers[i].values[:, :, :pd0["prompt_len"], :].clone(),
            i,
        )
    test_tokens = inputs.input_ids[:, pd0["prompt_len"]:pd0["seq_len"]]
    test_out = model(input_ids=test_tokens, past_key_values=test_trunc, use_cache=True)
    print(f"  DynamicCache truncation: OK (logits shape: {test_out.logits.shape})")

    # Test full evaluate_at_snr
    clean_keys = [past_kv.layers[i].keys.clone() for i in range(num_layers)]
    clean_values = [past_kv.layers[i].values.clone() for i in range(num_layers)]
    test_result = evaluate_at_snr(
        model, past_kv, num_layers, inputs.input_ids,
        pd0["prompt_len"], pd0["seq_len"], pd0["true_answer"],
        20, tokenizer, clean_keys, clean_values
    )
    print(f"  SNR=20 dB: answer_correct={test_result['answer_correct']}, "
          f"token_acc={test_result['token_accuracy']:.1%}, "
          f"answer='{test_result['generated_answer']}'")
    print("  Smoke test PASSED!")

    del past_kv, test_trunc, test_out, clean_keys, clean_values
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: SNR sweep ────────────────────────────────────────────
    print(f"\n=== Phase 2: SNR sweep ({len(SNR_LEVELS_DB)} levels x {n_valid} problems) ===")
    snr_results = {}

    for pi, pd in enumerate(problems_data):
        elapsed = time.time() - start_time
        if elapsed > 1500:  # leave margin for analysis
            print(f"Time budget reached at problem {pi}")
            break

        full_text = pd["prompt"] + pd["reasoning_text"]
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

        # Teacher-force in chunks to build clean KV cache (avoids OOM)
        past_kv = teacher_force_chunked(model, inputs.input_ids)

        # Save clean cache tensors
        clean_keys = [past_kv.layers[i].keys.clone() for i in range(num_layers)]
        clean_values = [past_kv.layers[i].values.clone() for i in range(num_layers)]

        # Evaluate at each SNR level
        for snr_db in SNR_LEVELS_DB:
            snr_key = f"snr_{snr_db}"
            if snr_key not in snr_results:
                snr_results[snr_key] = []

            try:
                result = evaluate_at_snr(
                    model, past_kv, num_layers, inputs.input_ids,
                    pd["prompt_len"], pd["seq_len"], pd["true_answer"],
                    snr_db, tokenizer, clean_keys, clean_values
                )
                snr_results[snr_key].append({
                    "ds_idx": pd["ds_idx"],
                    "answer_correct": result["answer_correct"],
                    "token_accuracy": result["token_accuracy"],
                    "generated_answer": result["generated_answer"],
                })
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                snr_results[snr_key].append({
                    "ds_idx": pd["ds_idx"],
                    "error": "OOM",
                })
            except Exception as e:
                snr_results[snr_key].append({
                    "ds_idx": pd["ds_idx"],
                    "error": str(e),
                })

        # Clean up this problem
        del past_kv, clean_keys, clean_values
        gc.collect()
        torch.cuda.empty_cache()

        # Per-problem progress
        print(f"\nProblem {pi+1}/{n_valid} (#{pd['ds_idx']}) [{time.time()-start_time:.0f}s]")
        for snr_db in [25, 20, 15, 10, 5]:
            snr_key = f"snr_{snr_db}"
            if snr_key in snr_results:
                r = snr_results[snr_key][-1]
                if "error" not in r:
                    print(f"  SNR {snr_db:2d}: ans={r['answer_correct']}, tok={r['token_accuracy']:.0%}")
                else:
                    print(f"  SNR {snr_db:2d}: ERROR {r['error'][:40]}")

    # Add clean baseline (infinite SNR)
    n_evaluated = min(len(problems_data), len(snr_results.get("snr_0", [])))
    snr_results["snr_inf"] = [
        {
            "ds_idx": pd["ds_idx"],
            "answer_correct": True,
            "token_accuracy": 1.0,
            "generated_answer": pd["true_answer"],
        }
        for pd in problems_data[:n_evaluated]
    ]

    # ── Phase 3: Analysis ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SNR SWEEP RESULTS")
    print(f"{'='*70}")

    summary = {
        "experiment": "exp_007_llama_snr_cliff",
        "model": MODEL_NAME,
        "num_valid": n_valid,
        "num_evaluated": n_evaluated,
        "snr_levels_db": SNR_LEVELS_DB + [float('inf')],
        "lookback_tokens": LOOKBACK,
        "results": {},
    }

    all_snr_keys = sorted(
        snr_results.keys(),
        key=lambda k: float(k.split("_")[1]) if k != "snr_inf" else float('inf')
    )

    print(f"\n{'SNR (dB)':<12} {'Ans Acc':<14} {'Tok Acc':<14} {'n':<6} {'errors':<8}")
    print("-" * 54)

    for snr_key in all_snr_keys:
        valid = [r for r in snr_results[snr_key] if "error" not in r]
        n = len(valid)
        if n == 0:
            errors = len(snr_results[snr_key])
            snr_label = snr_key.replace("snr_", "")
            print(f"{snr_label:<12} {'N/A':<14} {'N/A':<14} {0:<6} {errors:<8}")
            continue

        ans_correct = sum(1 for r in valid if r["answer_correct"])
        ans_acc = ans_correct / n
        tok_accs = [r["token_accuracy"] for r in valid]
        mean_tok_acc = float(np.mean(tok_accs))
        std_tok_acc = float(np.std(tok_accs))
        errors = sum(1 for r in snr_results[snr_key] if "error" in r)

        snr_label = snr_key.replace("snr_", "")
        if snr_label == "inf":
            snr_label = "inf (clean)"

        print(f"{snr_label:<12} {ans_correct}/{n} ({ans_acc:.1%}){'':<2} "
              f"{mean_tok_acc:.1%} +/-{std_tok_acc:.1%}{'':<2} {n:<6} {errors:<8}")

        summary["results"][snr_key] = {
            "answer_accuracy": ans_acc,
            "answer_correct": ans_correct,
            "answer_total": n,
            "token_accuracy_mean": mean_tok_acc,
            "token_accuracy_std": std_tok_acc,
            "token_accuracies": [float(t) for t in tok_accs],
        }

    # ── Cliff analysis ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CLIFF ANALYSIS")
    print(f"{'='*70}")

    snr_vals = []
    acc_vals = []
    tok_vals = []
    for snr_key in all_snr_keys:
        if snr_key not in summary["results"]:
            continue
        snr_val = float('inf') if snr_key == "snr_inf" else float(snr_key.split("_")[1])
        snr_vals.append(snr_val)
        acc_vals.append(summary["results"][snr_key]["answer_accuracy"])
        tok_vals.append(summary["results"][snr_key]["token_accuracy_mean"])

    # Find cliff: largest accuracy drop between adjacent levels
    max_drop = 0
    cliff_snr_lower = None
    cliff_snr_upper = None
    for i in range(len(snr_vals) - 1):
        if snr_vals[i + 1] == float('inf'):
            continue
        drop = acc_vals[i + 1] - acc_vals[i]  # positive = accuracy increases with SNR
        if drop > max_drop:
            max_drop = drop
            cliff_snr_lower = snr_vals[i]
            cliff_snr_upper = snr_vals[i + 1]

    # Find 90% and 10% thresholds
    snr_90 = None  # lowest SNR with >= 90% accuracy
    snr_10 = None  # highest SNR with < 10% accuracy
    for i in range(len(snr_vals) - 1, -1, -1):
        if snr_vals[i] == float('inf'):
            continue
        if acc_vals[i] >= 0.9:
            snr_90 = snr_vals[i]
            break
    for i in range(len(snr_vals)):
        if snr_vals[i] == float('inf'):
            continue
        if acc_vals[i] < 0.1:
            snr_10 = snr_vals[i]

    cliff_width = None
    if snr_90 is not None and snr_10 is not None:
        cliff_width = snr_90 - snr_10

    print(f"\nLargest single-step accuracy drop: {max_drop:.1%}")
    if cliff_snr_lower is not None:
        print(f"  Between SNR {cliff_snr_lower} dB and {cliff_snr_upper} dB")
    if snr_90 is not None:
        print(f"Lowest SNR with >= 90% answer accuracy: {snr_90} dB")
    if snr_10 is not None:
        print(f"Highest SNR with < 10% answer accuracy: {snr_10} dB")
    if cliff_width is not None:
        print(f"Cliff width (90% -> <10%): {cliff_width} dB")
    else:
        print("Cliff width: could not determine (data may not span the full cliff)")

    # Comparison with Qwen
    print(f"\n--- Comparison with Qwen3-4B-Base (Exp 3) ---")
    print(f"Qwen cliff: ~14 dB center, 3 dB width (15 dB -> 12 dB)")
    print(f"Qwen: 100% at 15 dB, 93% at 14 dB, 41% at 13 dB, 4% at 12 dB, 0% at 10 dB")
    if snr_90 is not None:
        print(f"Llama 90% threshold: {snr_90} dB")
    if cliff_width is not None:
        print(f"Llama cliff width: {cliff_width} dB")

    # Token vs answer dissociation
    print(f"\n--- Token vs Answer Accuracy Dissociation ---")
    for i, snr_key in enumerate(all_snr_keys):
        if snr_key not in summary["results"]:
            continue
        r = summary["results"][snr_key]
        dissoc = r["token_accuracy_mean"] - r["answer_accuracy"]
        snr_label = snr_key.replace("snr_", "")
        if abs(dissoc) > 0.05:
            print(f"  SNR {snr_label:>5s} dB: token={r['token_accuracy_mean']:.1%}, "
                  f"answer={r['answer_accuracy']:.1%}, gap={dissoc:+.1%}")

    summary["cliff_analysis"] = {
        "max_drop": float(max_drop),
        "cliff_snr_lower": cliff_snr_lower,
        "cliff_snr_upper": cliff_snr_upper,
        "snr_90pct": snr_90,
        "snr_10pct": snr_10,
        "cliff_width_db": cliff_width,
    }

    summary["elapsed_seconds"] = time.time() - start_time

    # Save results
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_convert)

    with open(os.path.join(RESULTS_DIR, "snr_results.json"), "w") as f:
        json.dump(snr_results, f, indent=2, default=_convert)

    # ── Generate figures ──────────────────────────────────────────────
    generate_figures(summary, RESULTS_DIR)

    total_time = time.time() - start_time
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("DONE")


def generate_figures(summary, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = summary["results"]
    if not results:
        print("No data for figures")
        return

    n = summary.get("num_evaluated", summary["num_valid"])
    model_short = "Llama-3.1-8B"

    # Extract data for plotting (exclude inf)
    snr_vals = []
    ans_accs = []
    tok_accs = []
    tok_stds = []

    for snr_key in sorted(results.keys(),
                          key=lambda k: float(k.split("_")[1]) if k != "snr_inf" else 999):
        if snr_key == "snr_inf":
            continue
        snr_vals.append(float(snr_key.split("_")[1]))
        ans_accs.append(results[snr_key]["answer_accuracy"])
        tok_accs.append(results[snr_key]["token_accuracy_mean"])
        tok_stds.append(results[snr_key]["token_accuracy_std"])

    snr_vals = np.array(snr_vals)
    ans_accs = np.array(ans_accs)
    tok_accs = np.array(tok_accs)
    tok_stds = np.array(tok_stds)

    # ── Figure 1: SNR cliff (main result) ────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(snr_vals, ans_accs * 100, 'o-', color='#e74c3c', lw=2.5, ms=8,
             label='Answer Accuracy', zorder=3)
    ax1.plot(snr_vals, tok_accs * 100, 's-', color='#3498db', lw=2.5, ms=8,
             label='Token Accuracy', zorder=3)
    ax1.fill_between(snr_vals,
                     (tok_accs - tok_stds) * 100,
                     (tok_accs + tok_stds) * 100,
                     alpha=0.15, color='#3498db')

    # Qwen reference cliff region
    ax1.axvspan(12, 15, alpha=0.1, color='orange', label='Qwen cliff region (12-15 dB)')
    ax1.axvline(x=14, color='orange', linestyle='--', alpha=0.5, label='Qwen cliff center (14 dB)')

    ax1.set_xlabel('SNR (dB)', fontsize=13)
    ax1.set_ylabel('Accuracy (%)', fontsize=13)
    ax1.set_title(f'Exp 007: KV Cache SNR Cliff \u2014 {model_short}\n'
                  f'(n={n} problems, noise in reasoning KV positions)', fontsize=13)
    ax1.set_ylim(-5, 105)
    ax1.set_xlim(-1, 26)
    ax1.legend(fontsize=11, loc='center right')
    ax1.grid(True, alpha=0.3)

    # Mark Llama cliff
    cliff = summary.get("cliff_analysis", {})
    if cliff.get("snr_90pct") is not None:
        ax1.axvline(x=cliff["snr_90pct"], color='#e74c3c', linestyle=':', alpha=0.7)
        ax1.annotate(f'90% acc: {cliff["snr_90pct"]} dB',
                     xy=(cliff["snr_90pct"], 90), fontsize=9,
                     xytext=(cliff["snr_90pct"] + 1, 95),
                     arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    plt.tight_layout()
    fig1.savefig(os.path.join(results_dir, "snr_cliff.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: snr_cliff.png")

    # ── Figure 2: Cross-model comparison ─────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    # Qwen data from Exp 3 (research_spec)
    qwen_snr = [0, 10, 12, 13, 14, 15, 20]
    qwen_ans = [0, 0, 3.7, 40.7, 92.6, 100, 100]
    qwen_tok = [2.7, 14.3, 23.9, 55.3, 82.0, 89.8, 98.6]

    ax2.plot(qwen_snr, qwen_ans, 'o--', color='#e67e22', lw=2, ms=7,
             label='Qwen3-4B: Answer Acc', alpha=0.8)
    ax2.plot(qwen_snr, qwen_tok, 's--', color='#f39c12', lw=2, ms=7,
             label='Qwen3-4B: Token Acc', alpha=0.8)

    ax2.plot(snr_vals, ans_accs * 100, 'o-', color='#e74c3c', lw=2.5, ms=8,
             label=f'{model_short}: Answer Acc')
    ax2.plot(snr_vals, tok_accs * 100, 's-', color='#3498db', lw=2.5, ms=8,
             label=f'{model_short}: Token Acc')

    ax2.set_xlabel('SNR (dB)', fontsize=13)
    ax2.set_ylabel('Accuracy (%)', fontsize=13)
    ax2.set_title('Cross-Model SNR Cliff Comparison\n'
                  'KV Cache Noise Fragility: Qwen3-4B vs Llama-3.1-8B', fontsize=13)
    ax2.set_ylim(-5, 105)
    ax2.set_xlim(-1, 26)
    ax2.legend(fontsize=10, loc='center right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(os.path.join(results_dir, "cross_model_snr_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: cross_model_snr_comparison.png")

    # ── Figure 3: Token-Answer dissociation across SNR ───────────────
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    dissociation = tok_accs * 100 - ans_accs * 100

    colors = ['#2ecc71' if d > 5 else '#95a5a6' if d > -5 else '#e74c3c' for d in dissociation]
    ax3.bar(range(len(snr_vals)), dissociation, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_xticks(range(len(snr_vals)))
    ax3.set_xticklabels([str(int(s)) for s in snr_vals], fontsize=10)
    ax3.set_xlabel('SNR (dB)', fontsize=12)
    ax3.set_ylabel('Token Acc - Answer Acc (pp)', fontsize=12)
    ax3.set_title(f'Token vs Answer Accuracy Dissociation by SNR\n'
                  f'({model_short}, n={n})', fontsize=12)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    fig3.savefig(os.path.join(results_dir, "token_answer_dissociation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: token_answer_dissociation.png")


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
