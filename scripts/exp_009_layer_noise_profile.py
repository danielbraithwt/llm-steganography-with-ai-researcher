#!/usr/bin/env python3
"""
Experiment 009: Per-Layer Noise Sensitivity Profiling

Tests the digital/distributed encoding theory at the layer dimension.
For each model (Qwen3-4B-Base, Llama-3.1-8B-Instruct):
- Destroy one layer's KV cache entries at a time (full replacement with matched-norm noise)
- Measure accuracy for each single-layer ablation
- Compare layer sensitivity profiles between models

Prediction: Qwen shows concentrated sensitivity (specific critical layers),
            Llama shows distributed sensitivity (uniform across layers).
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
MODELS = [
    {"name": "Qwen/Qwen3-4B-Base", "short": "Qwen3-4B"},
    {"name": "meta-llama/Llama-3.1-8B-Instruct", "short": "Llama-3.1-8B"},
]
NUM_PROBLEMS = 20
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 1536
CHUNK_SIZE = 128
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_009")

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


@torch.no_grad()
def teacher_force_chunked(model, input_ids, chunk_size=CHUNK_SIZE):
    """Teacher-force a sequence in chunks to avoid OOM with eager attention."""
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
def evaluate_layer_ablation(model, clean_keys, clean_values, num_layers, input_ids,
                            prompt_len, seq_len, true_answer, layer_idx, tokenizer):
    """
    Evaluate model with one layer's reasoning KV entries replaced by noise.

    layer_idx: which layer to ablate.
        None = clean baseline (no noise)
        -1 = all layers ablated
        0..num_layers-1 = specific layer ablated
    """
    # Build truncated cache (positions 0 to seq_len-2) with noise at target layer
    cache = DynamicCache()
    trunc_end = seq_len - 1  # exclude last position so we can re-process it

    for i in range(num_layers):
        k = clean_keys[i][:, :, :trunc_end, :].clone()
        v = clean_values[i][:, :, :trunc_end, :].clone()

        should_noise = False
        if layer_idx == -1:
            should_noise = True
        elif layer_idx is not None and layer_idx == i:
            should_noise = True

        if should_noise and trunc_end > prompt_len:
            # Replace reasoning positions with matched-norm Gaussian noise
            rk = k[:, :, prompt_len:, :]
            rv = v[:, :, prompt_len:, :]
            k_norm = rk.norm().item()
            v_norm = rv.norm().item()
            kn = torch.randn_like(rk)
            vn = torch.randn_like(rv)
            if kn.norm().item() > 1e-8:
                rk.copy_(kn * (k_norm / kn.norm().item()))
            if vn.norm().item() > 1e-8:
                rv.copy_(vn * (v_norm / vn.norm().item()))

        cache.update(k, v, i)

    # Process last token to get logits for generation
    last_token = input_ids[:, seq_len - 1:seq_len]
    out = model(input_ids=last_token, past_key_values=cache, use_cache=True)
    gen_kv = out.past_key_values

    # Generate answer freely
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
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

    del cache, gen_kv, out
    gc.collect()
    torch.cuda.empty_cache()

    return {"answer_correct": answer_correct, "generated_answer": gen_answer}


def run_model(model_config, dataset, global_start_time, deadline):
    """Run per-layer ablation experiment for one model."""
    model_name = model_config["name"]
    model_short = model_config["short"]
    model_start = time.time()
    model_budget = deadline - model_start

    print(f"\n{'='*70}")
    print(f"Model: {model_name} (budget: {model_budget:.0f}s)")
    print(f"{'='*70}")

    # Load model
    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    print(f"Loaded in {time.time() - t0:.1f}s — {num_layers} layers, "
          f"heads={model.config.num_attention_heads}, "
          f"kv_heads={getattr(model.config, 'num_key_value_heads', 'N/A')}")

    # Select problems (use same seed-based selection as previous experiments)
    indices = list(range(len(dataset)))
    random.seed(SEED)
    random.shuffle(indices)
    selected = indices[:NUM_PROBLEMS]

    # Phase 1: Generate traces and verify baseline correctness
    print(f"\n--- Phase 1: Generating traces ({model_short}) ---")
    problems_data = []

    for prob_idx, ds_idx in enumerate(selected):
        if time.time() > model_start + model_budget * 0.35:
            print(f"Phase 1 time budget reached at problem {prob_idx}")
            break

        problem = dataset[ds_idx]
        question = problem["question"]
        true_answer = normalize_answer(
            problem["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
        )
        prompt = build_prompt(question)

        trace_text = generate_trace(model, tokenizer, prompt)
        gen_answer = extract_answer(trace_text)
        gen_norm = normalize_answer(gen_answer) if gen_answer else ""
        correct = (gen_norm == true_answer)

        if not correct:
            continue

        # Compute lengths
        if "####" in trace_text:
            reasoning_text = trace_text[:trace_text.index("####")]
        else:
            reasoning_text = trace_text

        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]
        full_text = prompt + reasoning_text
        full_inputs = tokenizer(full_text, return_tensors="pt")
        full_seq_len = full_inputs.input_ids.shape[1]
        reasoning_len = full_seq_len - prompt_len

        if reasoning_len < 10:
            continue

        if full_seq_len > MAX_SEQ_LEN:
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
            "full_text": full_text,
            "prompt_len": prompt_len,
            "reasoning_len": reasoning_len,
            "seq_len": full_seq_len,
        })
        print(f"  Problem {prob_idx+1}: #{ds_idx}, true={true_answer}, "
              f"reasoning_len={reasoning_len}, seq_len={full_seq_len}")

    n_valid = len(problems_data)
    print(f"\n{model_short}: {n_valid} valid problems")

    if n_valid < 3:
        print(f"ERROR: Too few valid problems for {model_short}!")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": "too_few_valid", "n_valid": n_valid, "model": model_name}

    # Smoke test: clean baseline on first problem
    print(f"\n--- Smoke test ({model_short}) ---")
    pd0 = problems_data[0]
    inputs = tokenizer(pd0["full_text"], return_tensors="pt").to(model.device)
    past_kv = teacher_force_chunked(model, inputs.input_ids)
    print(f"  KV cache: {type(past_kv).__name__}, layers={len(past_kv.layers)}")
    print(f"  Key shape: {past_kv.layers[0].keys.shape}")

    clean_keys = [past_kv.layers[i].keys.clone() for i in range(num_layers)]
    clean_values = [past_kv.layers[i].values.clone() for i in range(num_layers)]

    # Test clean baseline
    baseline_result = evaluate_layer_ablation(
        model, clean_keys, clean_values, num_layers, inputs.input_ids,
        pd0["prompt_len"], pd0["seq_len"], pd0["true_answer"], None, tokenizer
    )
    print(f"  Baseline: correct={baseline_result['answer_correct']}, "
          f"answer='{baseline_result['generated_answer']}'")

    # Test single-layer ablation (layer 0)
    layer0_result = evaluate_layer_ablation(
        model, clean_keys, clean_values, num_layers, inputs.input_ids,
        pd0["prompt_len"], pd0["seq_len"], pd0["true_answer"], 0, tokenizer
    )
    print(f"  Layer 0 ablated: correct={layer0_result['answer_correct']}, "
          f"answer='{layer0_result['generated_answer']}'")

    # Test last layer ablation
    last_layer = num_layers - 1
    last_result = evaluate_layer_ablation(
        model, clean_keys, clean_values, num_layers, inputs.input_ids,
        pd0["prompt_len"], pd0["seq_len"], pd0["true_answer"], last_layer, tokenizer
    )
    print(f"  Layer {last_layer} ablated: correct={last_result['answer_correct']}, "
          f"answer='{last_result['generated_answer']}'")

    print("  Smoke test PASSED!")
    del past_kv, clean_keys, clean_values
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: Per-layer ablation sweep
    print(f"\n--- Phase 2: Per-layer ablation ({model_short}) ---")
    # Conditions: None (baseline), -1 (all layers), 0..num_layers-1
    conditions = [None] + list(range(num_layers)) + [-1]
    condition_labels = ["baseline"] + [f"layer_{i}" for i in range(num_layers)] + ["all_layers"]

    layer_results = {label: [] for label in condition_labels}

    for pi, pd in enumerate(problems_data):
        if time.time() > model_start + model_budget * 0.85:
            print(f"Time budget reached at problem {pi}")
            break

        inputs = tokenizer(pd["full_text"], return_tensors="pt").to(model.device)
        past_kv = teacher_force_chunked(model, inputs.input_ids)

        clean_keys = [past_kv.layers[i].keys.clone() for i in range(num_layers)]
        clean_values = [past_kv.layers[i].values.clone() for i in range(num_layers)]

        for cond_idx, (cond, label) in enumerate(zip(conditions, condition_labels)):
            try:
                result = evaluate_layer_ablation(
                    model, clean_keys, clean_values, num_layers, inputs.input_ids,
                    pd["prompt_len"], pd["seq_len"], pd["true_answer"], cond, tokenizer
                )
                layer_results[label].append({
                    "ds_idx": pd["ds_idx"],
                    "answer_correct": result["answer_correct"],
                    "generated_answer": result["generated_answer"],
                })
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                layer_results[label].append({
                    "ds_idx": pd["ds_idx"],
                    "error": "OOM",
                })
            except Exception as e:
                layer_results[label].append({
                    "ds_idx": pd["ds_idx"],
                    "error": str(e)[:100],
                })

        del past_kv, clean_keys, clean_values
        gc.collect()
        torch.cuda.empty_cache()

        # Progress update
        baseline_n = len(layer_results["baseline"])
        baseline_acc = sum(1 for r in layer_results["baseline"] if r.get("answer_correct")) / max(baseline_n, 1)
        layer_accs_so_far = []
        for l in range(num_layers):
            lr = layer_results[f"layer_{l}"]
            if lr:
                la = sum(1 for r in lr if r.get("answer_correct")) / len(lr)
                layer_accs_so_far.append((l, la))
        if layer_accs_so_far:
            worst_l, worst_a = min(layer_accs_so_far, key=lambda x: x[1])
            worst_str = f"worst=L{worst_l}({worst_a:.0%})"
        else:
            worst_str = "no layer data yet"
        print(f"  Problem {pi+1}/{n_valid} (#{pd['ds_idx']}): "
              f"baseline={baseline_acc:.0%}, {worst_str} "
              f"[{time.time()-global_start_time:.0f}s]")

    # Clean up model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Aggregate results
    n_evaluated = len(layer_results["baseline"])
    summary = {
        "model": model_name,
        "model_short": model_short,
        "num_layers": num_layers,
        "n_valid": n_valid,
        "n_evaluated": n_evaluated,
        "per_layer_accuracy": {},
    }

    print(f"\n--- Results ({model_short}, n={n_evaluated}) ---")
    print(f"{'Condition':<15} {'Accuracy':>10} {'n_correct':>10} {'n_total':>10}")
    print("-" * 50)

    accs = []
    for label in condition_labels:
        valid = [r for r in layer_results[label] if "error" not in r]
        n_total = len(valid)
        if n_total == 0:
            summary["per_layer_accuracy"][label] = None
            continue
        n_correct = sum(1 for r in valid if r["answer_correct"])
        acc = n_correct / n_total
        summary["per_layer_accuracy"][label] = acc

        if label.startswith("layer_"):
            accs.append(acc)

        if label in ["baseline", "all_layers"] or (label.startswith("layer_") and int(label.split("_")[1]) % 6 == 0):
            print(f"{label:<15} {acc:>9.1%} {n_correct:>10} {n_total:>10}")

    # Layer-level statistics
    if accs:
        accs_arr = np.array(accs)
        summary["layer_stats"] = {
            "mean_accuracy": float(np.mean(accs_arr)),
            "std_accuracy": float(np.std(accs_arr)),
            "min_accuracy": float(np.min(accs_arr)),
            "max_accuracy": float(np.max(accs_arr)),
            "range_accuracy": float(np.max(accs_arr) - np.min(accs_arr)),
            "n_critical_layers": int(np.sum(accs_arr < 0.5)),
            "worst_layer": int(np.argmin(accs_arr)),
            "best_layer": int(np.argmax(accs_arr)),
        }
        print(f"\nLayer statistics:")
        print(f"  Mean accuracy: {np.mean(accs_arr):.1%}")
        print(f"  Std accuracy:  {np.std(accs_arr):.1%}")
        print(f"  Range:         {np.min(accs_arr):.1%} - {np.max(accs_arr):.1%}")
        print(f"  Critical layers (acc < 50%): {np.sum(accs_arr < 0.5)}")
        print(f"  Worst layer:   {np.argmin(accs_arr)} ({np.min(accs_arr):.1%})")
        print(f"  Best layer:    {np.argmax(accs_arr)} ({np.max(accs_arr):.1%})")

    summary["raw_results"] = layer_results
    return summary


def generate_figures(results, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_results = {r["model_short"]: r for r in results if "error" not in r}

    if not model_results:
        print("No results to plot")
        return

    # Figure 1: Per-layer accuracy profiles (one subplot per model)
    n_models = len(model_results)
    fig1, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6), squeeze=False)

    for idx, (model_short, result) in enumerate(model_results.items()):
        ax = axes[0, idx]
        num_layers = result["num_layers"]
        accs = []
        for layer in range(num_layers):
            acc = result["per_layer_accuracy"].get(f"layer_{layer}")
            accs.append(acc if acc is not None else 0.0)

        accs = np.array(accs)
        layers = np.arange(num_layers)

        # Color bars by severity
        colors = []
        for a in accs:
            if a >= 0.8:
                colors.append("#2ecc71")
            elif a >= 0.5:
                colors.append("#f39c12")
            else:
                colors.append("#e74c3c")

        ax.bar(layers, accs * 100, color=colors, edgecolor="black", linewidth=0.3)

        # Add baseline and all-layer lines
        baseline = result["per_layer_accuracy"].get("baseline", 1.0)
        all_layer = result["per_layer_accuracy"].get("all_layers", 0.0)
        if baseline is not None:
            ax.axhline(y=baseline * 100, color="blue", linestyle="--", alpha=0.7, label=f"Baseline ({baseline:.0%})")
        if all_layer is not None:
            ax.axhline(y=all_layer * 100, color="red", linestyle="--", alpha=0.7, label=f"All-layer ({all_layer:.0%})")

        # Stats annotation
        stats = result.get("layer_stats", {})
        if stats:
            stats_text = (f"Mean: {stats['mean_accuracy']:.1%}\n"
                          f"Std: {stats['std_accuracy']:.1%}\n"
                          f"Range: {stats['range_accuracy']:.1%}\n"
                          f"Critical: {stats['n_critical_layers']}")
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Answer Accuracy (%)", fontsize=12)
        ax.set_title(f"{model_short} (n={result['n_evaluated']})", fontsize=13)
        ax.set_ylim(-5, 105)
        ax.set_xlim(-1, num_layers)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.2, axis="y")

    fig1.suptitle("Exp 009: Per-Layer KV Cache Noise Sensitivity\n"
                  "(Full replacement of reasoning KV entries at each layer)",
                  fontsize=14, y=1.02)
    plt.tight_layout()
    fig1.savefig(os.path.join(results_dir, "layer_sensitivity_profiles.png"),
                 dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: layer_sensitivity_profiles.png")

    # Figure 2: Cross-model comparison (overlaid)
    if len(model_results) >= 2:
        fig2, ax2 = plt.subplots(figsize=(12, 7))

        colors_model = ["#e74c3c", "#3498db"]
        for idx, (model_short, result) in enumerate(model_results.items()):
            num_layers = result["num_layers"]
            accs = []
            for layer in range(num_layers):
                acc = result["per_layer_accuracy"].get(f"layer_{layer}")
                accs.append(acc if acc is not None else 0.0)

            # Normalize layer index to [0, 1] for comparison
            norm_layers = np.linspace(0, 1, num_layers)
            ax2.plot(norm_layers, np.array(accs) * 100, 'o-', color=colors_model[idx],
                     ms=4, lw=1.5, alpha=0.8, label=f"{model_short} ({num_layers} layers)")

            # Add mean line
            mean_acc = np.mean(accs) * 100
            ax2.axhline(y=mean_acc, color=colors_model[idx], linestyle=":",
                        alpha=0.4, label=f"{model_short} mean ({mean_acc:.0f}%)")

        ax2.set_xlabel("Normalized Layer Position (0=first, 1=last)", fontsize=12)
        ax2.set_ylabel("Answer Accuracy (%)", fontsize=12)
        ax2.set_title("Cross-Model Layer Sensitivity Comparison\n"
                      "(Single-layer KV replacement noise)", fontsize=13)
        ax2.set_ylim(-5, 105)
        ax2.legend(fontsize=10, loc="lower left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig2.savefig(os.path.join(results_dir, "cross_model_layer_comparison.png"),
                     dpi=150, bbox_inches="tight")
        plt.close()
        print("Figure saved: cross_model_layer_comparison.png")

    # Figure 3: Concentration analysis (std/range comparison)
    if len(model_results) >= 2:
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

        model_names = list(model_results.keys())
        stds = [model_results[m].get("layer_stats", {}).get("std_accuracy", 0) * 100
                for m in model_names]
        ranges = [model_results[m].get("layer_stats", {}).get("range_accuracy", 0) * 100
                  for m in model_names]
        n_critical = [model_results[m].get("layer_stats", {}).get("n_critical_layers", 0)
                      for m in model_names]

        ax3a.bar(model_names, stds, color=["#e74c3c", "#3498db"], edgecolor="black")
        ax3a.set_ylabel("Std of Per-Layer Accuracy (pp)", fontsize=11)
        ax3a.set_title("Layer Sensitivity Concentration\n(higher = more concentrated)", fontsize=12)
        for i, v in enumerate(stds):
            ax3a.text(i, v + 0.5, f"{v:.1f}pp", ha="center", fontsize=10)

        ax3b.bar(model_names, ranges, color=["#e74c3c", "#3498db"], edgecolor="black")
        ax3b.set_ylabel("Range of Per-Layer Accuracy (pp)", fontsize=11)
        ax3b.set_title("Layer Sensitivity Range\n(higher = more variable)", fontsize=12)
        for i, v in enumerate(ranges):
            ax3b.text(i, v + 0.5, f"{v:.1f}pp", ha="center", fontsize=10)

        fig3.suptitle("Exp 009: Concentration Metrics", fontsize=13, y=1.02)
        plt.tight_layout()
        fig3.savefig(os.path.join(results_dir, "concentration_comparison.png"),
                     dpi=150, bbox_inches="tight")
        plt.close()
        print("Figure saved: concentration_comparison.png")


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def main():
    start_time = time.time()
    total_budget = 1700  # seconds, leave margin for analysis/figures

    print(f"{'='*70}")
    print("Experiment 009: Per-Layer Noise Sensitivity Profiling")
    print(f"Models: {', '.join(m['short'] for m in MODELS)}")
    print(f"Problems per model: {NUM_PROBLEMS}")
    print(f"Time budget: {total_budget}s")
    print(f"{'='*70}\n")

    # Load GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    all_results = []

    for model_idx, model_config in enumerate(MODELS):
        elapsed = time.time() - start_time
        remaining = total_budget - elapsed
        per_model_budget = remaining / (len(MODELS) - model_idx)

        print(f"\n[{elapsed:.0f}s elapsed, {remaining:.0f}s remaining, "
              f"budget for this model: {per_model_budget:.0f}s]")

        if remaining < 120:
            print(f"Insufficient time for {model_config['short']}, skipping")
            all_results.append({"error": "time_budget", "model": model_config["name"],
                                "model_short": model_config["short"]})
            continue

        deadline = time.time() + per_model_budget
        result = run_model(model_config, dataset, start_time, deadline)
        all_results.append(result)

    # Save raw results (without full per-problem data to keep file small)
    summary_results = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != "raw_results"}
        summary_results.append(r_copy)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary_results, f, indent=2, default=_convert)

    # Save full raw results separately
    raw_results = {}
    for r in all_results:
        if "raw_results" in r:
            raw_results[r.get("model_short", "unknown")] = r["raw_results"]
    with open(os.path.join(RESULTS_DIR, "raw_results.json"), "w") as f:
        json.dump(raw_results, f, indent=2, default=_convert)

    # Generate figures
    print(f"\n{'='*70}")
    print("Generating figures...")
    print(f"{'='*70}")
    generate_figures(all_results, RESULTS_DIR)

    # Cross-model comparison
    valid_results = [r for r in all_results if "error" not in r and "layer_stats" in r]
    if len(valid_results) >= 2:
        print(f"\n{'='*70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*70}")
        for r in valid_results:
            stats = r["layer_stats"]
            print(f"\n{r['model_short']}:")
            print(f"  Mean per-layer accuracy: {stats['mean_accuracy']:.1%}")
            print(f"  Std per-layer accuracy:  {stats['std_accuracy']:.1%}")
            print(f"  Range:                   {stats['range_accuracy']:.1%}")
            print(f"  Critical layers:         {stats['n_critical_layers']}")
            print(f"  Worst layer:             {stats['worst_layer']} ({stats['min_accuracy']:.1%})")
            print(f"  Best layer:              {stats['best_layer']} ({stats['max_accuracy']:.1%})")

        # Test prediction: Qwen std > Llama std
        qwen = [r for r in valid_results if "Qwen" in r["model_short"]]
        llama = [r for r in valid_results if "Llama" in r["model_short"]]
        if qwen and llama:
            qwen_std = qwen[0]["layer_stats"]["std_accuracy"]
            llama_std = llama[0]["layer_stats"]["std_accuracy"]
            ratio = qwen_std / max(llama_std, 1e-8)
            print(f"\nPrediction check: std(Qwen)/std(Llama) = {ratio:.2f} (predicted >= 1.5)")
            if ratio >= 1.5:
                print("  CONFIRMED: Qwen shows more concentrated layer sensitivity")
            else:
                print("  NOT CONFIRMED: Layer sensitivity concentration is similar")

    total_time = time.time() - start_time
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("DONE")


if __name__ == "__main__":
    main()
