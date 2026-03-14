#!/usr/bin/env python3
"""
Experiment 011: H2O Heavy-Hitter vs Answer-Coupled Position Overlap

Tests whether practical KV cache compression methods (H2O) implicitly preserve
the hidden channel by retaining answer-coupled positions.

Computes three per-position importance scores:
1. H2O score: cumulative attention received from ALL subsequent tokens (all layers)
2. AC score: attention from last (answer) token (last 4 layers, as in exp_004/005)
3. TC score: average attention from later reasoning tokens (last 4 layers)

Then measures correlation and overlap between these scores to determine whether
H2O compression preserves AC positions preferentially.
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
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────
NUM_PROBLEMS = 25
ATTENTION_LAYERS_AC = [-1, -2, -3, -4]  # last 4 layers for AC/TC (as in exp_004/005)
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 1536
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_011")
RETENTION_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 8-shot GSM8K exemplars (same as exp_004/005) ─────────────────────────
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
def compute_importance_scores(model, tokenizer, prompt, trace_text):
    """
    Compute per-position importance scores using three methods:
    1. H2O: cumulative attention from ALL subsequent tokens (all layers)
    2. AC: attention from last token (last 4 layers)
    3. TC: average attention from later reasoning tokens (last 4 layers)
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
        trace_tokens = tokenizer(trace_text, return_tensors="pt").input_ids[0][:max_reasoning]
        trace_text = tokenizer.decode(trace_tokens, skip_special_tokens=True)
        full_text = prompt + trace_text
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]
        reasoning_len = seq_len - prompt_len

    # Forward pass with all attentions
    outputs = model(**inputs, output_attentions=True, use_cache=False)

    num_layers = len(outputs.attentions)
    last_token_idx = seq_len - 1
    ac_layer_indices = [num_layers + i for i in ATTENTION_LAYERS_AC]

    # Initialize scores
    h2o_score = torch.zeros(reasoning_len, device=model.device, dtype=torch.float32)
    ac_score = torch.zeros(reasoning_len, device=model.device, dtype=torch.float32)
    tc_score = torch.zeros(reasoning_len, device=model.device, dtype=torch.float32)

    num_heads = outputs.attentions[0].shape[1]

    for li in range(num_layers):
        attn = outputs.attentions[li][0].float()  # (num_heads, seq_len, seq_len)

        # H2O: cumulative attention received from ALL subsequent tokens
        # For reasoning positions: column sums within the reasoning block, excluding self-attention
        # Also include attention from ALL positions in the sequence (not just reasoning)
        # Column i (reasoning position) gets attention from rows prompt_len+i+1 to seq_len-1
        # In causal attention, rows < prompt_len+i are already zero for column prompt_len+i
        # So we can take full column sum and subtract the diagonal

        # Full column sums for reasoning positions (from ALL tokens)
        full_col_sums = attn[:, :, prompt_len:seq_len].sum(dim=1)  # (heads, reasoning_len)
        # Diagonal (self-attention at each reasoning position)
        reasoning_block = attn[:, prompt_len:seq_len, prompt_len:seq_len]
        diag = torch.diagonal(reasoning_block, dim1=1, dim2=2)  # (heads, reasoning_len)
        # H2O = total attention received minus self-attention
        h2o_layer = (full_col_sums - diag).sum(dim=0)  # (reasoning_len,)
        h2o_score += h2o_layer

        # AC and TC for specific layers
        if li in ac_layer_indices:
            # AC: attention from last token to each reasoning position
            ac_score += attn[:, last_token_idx, prompt_len:seq_len].sum(dim=0)

            # TC: average attention from later reasoning tokens to each position
            reasoning_attn_sum = reasoning_block.sum(dim=0)  # (reasoning_len, reasoning_len)
            mask = torch.tril(
                torch.ones(reasoning_len, reasoning_len, device=model.device),
                diagonal=-1
            )
            weighted = reasoning_attn_sum * mask
            col_sums = weighted.sum(dim=0)
            col_counts = mask.sum(dim=0).clamp(min=1)
            tc_score += col_sums / col_counts

        # Free this layer's attention
        del attn, reasoning_block
        if li in ac_layer_indices:
            del reasoning_attn_sum, mask, weighted

    # Normalize
    num_ac_layers = len(ac_layer_indices)
    h2o_score = h2o_score / (num_layers * num_heads)
    ac_score = ac_score / (num_ac_layers * num_heads)
    tc_score = tc_score / (num_ac_layers * num_heads)

    # Get token texts for position analysis
    token_ids = inputs.input_ids[0, prompt_len:seq_len].cpu().tolist()
    token_texts = [tokenizer.decode([tid]) for tid in token_ids]

    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "h2o_score": h2o_score.cpu().numpy(),
        "ac_score": ac_score.cpu().numpy(),
        "tc_score": tc_score.cpu().numpy(),
        "prompt_len": prompt_len,
        "reasoning_len": reasoning_len,
        "seq_len": seq_len,
        "token_texts": token_texts,
    }


def compute_selectivity(ac_score, tc_score):
    n = len(ac_score)
    if n <= 1:
        return np.zeros(n)
    ac_ranks = np.argsort(np.argsort(ac_score)).astype(float) / (n - 1)
    tc_ranks = np.argsort(np.argsort(tc_score)).astype(float) / (n - 1)
    return ac_ranks - tc_ranks


def compute_correlations(h2o, ac, tc, selectivity):
    """Compute Spearman and Pearson correlations between score vectors."""
    results = {}
    pairs = [
        ("h2o_vs_ac", h2o, ac),
        ("h2o_vs_tc", h2o, tc),
        ("h2o_vs_selectivity", h2o, selectivity),
        ("ac_vs_tc", ac, tc),
    ]
    for name, x, y in pairs:
        if len(x) < 5:
            continue
        sp_rho, sp_p = stats.spearmanr(x, y)
        pr_r, pr_p = stats.pearsonr(x, y)
        results[name] = {
            "spearman_rho": float(sp_rho),
            "spearman_p": float(sp_p),
            "pearson_r": float(pr_r),
            "pearson_p": float(pr_p),
        }
    return results


def compute_overlap_at_retention(h2o, ac, tc, selectivity, retention_rates):
    """
    At each retention rate k, compute overlap between top-k by H2O and top-k by other metrics.
    Uses Jaccard index: |A ∩ B| / |A ∪ B|
    """
    n = len(h2o)
    results = {}

    for rate in retention_rates:
        k = max(1, int(n * rate))
        h2o_top = set(np.argsort(h2o)[-k:])
        ac_top = set(np.argsort(ac)[-k:])
        tc_top = set(np.argsort(tc)[-k:])
        sel_top = set(np.argsort(selectivity)[-k:])  # most AC-selective

        results[f"{rate:.1f}"] = {
            "k": k,
            "n": n,
            "h2o_ac_jaccard": len(h2o_top & ac_top) / len(h2o_top | ac_top) if len(h2o_top | ac_top) > 0 else 0,
            "h2o_tc_jaccard": len(h2o_top & tc_top) / len(h2o_top | tc_top) if len(h2o_top | tc_top) > 0 else 0,
            "h2o_sel_jaccard": len(h2o_top & sel_top) / len(h2o_top | sel_top) if len(h2o_top | sel_top) > 0 else 0,
            "h2o_ac_overlap_frac": len(h2o_top & ac_top) / k,  # fraction of H2O top-k that are also AC top-k
            "h2o_tc_overlap_frac": len(h2o_top & tc_top) / k,
            "h2o_sel_overlap_frac": len(h2o_top & sel_top) / k,
            # What fraction of H2O's retained positions are AC-selective (selectivity > 0)?
            "h2o_retains_ac_selective": sum(1 for p in h2o_top if selectivity[p] > 0) / k,
            "h2o_retains_tc_selective": sum(1 for p in h2o_top if selectivity[p] < 0) / k,
        }

    return results


def run_model(model_name, dataset, selected_indices, start_time, time_budget):
    """Run the full analysis for one model."""
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}\n")

    print("Loading model...")
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
    print(f"Model loaded: {num_layers} layers, {num_heads} heads, {kv_heads} KV heads")

    # ── Phase 1: Generate traces and compute importance scores ──────────
    print("\n=== Phase 1: Generate traces + importance scores ===")
    problems_data = []

    for prob_idx, ds_idx in enumerate(selected_indices):
        elapsed = time.time() - start_time
        if elapsed > time_budget * 0.6:  # 60% of budget for phase 1
            print(f"Phase 1 time budget reached at problem {prob_idx}")
            break

        problem = dataset[ds_idx]
        question = problem["question"]
        true_answer = normalize_answer(
            problem["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
        )
        prompt = build_prompt(question)

        print(f"\nProblem {prob_idx+1}/{len(selected_indices)} (#{ds_idx}), true={true_answer}")

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
            scores = compute_importance_scores(model, tokenizer, prompt, reasoning_text)
        except torch.cuda.OutOfMemoryError:
            print("  SKIP: OOM during attention extraction")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        if scores is None:
            print("  SKIP: trace too short")
            continue

        # Compute selectivity
        selectivity = compute_selectivity(scores["ac_score"], scores["tc_score"])

        # Compute per-problem correlations
        corrs = compute_correlations(
            scores["h2o_score"], scores["ac_score"],
            scores["tc_score"], selectivity
        )

        # Compute per-problem overlap
        overlaps = compute_overlap_at_retention(
            scores["h2o_score"], scores["ac_score"],
            scores["tc_score"], selectivity,
            RETENTION_RATES
        )

        problems_data.append({
            "ds_idx": ds_idx,
            "true_answer": true_answer,
            "reasoning_len": scores["reasoning_len"],
            "h2o_score": scores["h2o_score"],
            "ac_score": scores["ac_score"],
            "tc_score": scores["tc_score"],
            "selectivity": selectivity,
            "correlations": corrs,
            "overlaps": overlaps,
        })

        h2o_ac = corrs.get("h2o_vs_ac", {}).get("spearman_rho", float("nan"))
        h2o_tc = corrs.get("h2o_vs_tc", {}).get("spearman_rho", float("nan"))
        h2o_sel = corrs.get("h2o_vs_selectivity", {}).get("spearman_rho", float("nan"))
        print(f"  Valid! R={scores['reasoning_len']}, "
              f"H2O~AC={h2o_ac:.3f}, H2O~TC={h2o_tc:.3f}, H2O~sel={h2o_sel:.3f}")

    print(f"\n=== Phase 1 complete: {len(problems_data)} valid problems ===")

    if len(problems_data) < 3:
        print("ERROR: Too few valid problems!")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": "too_few_valid", "n_valid": len(problems_data), "model": model_name}

    # ── Phase 2: Aggregate analysis ──────────────────────────────────────
    print("\n=== Phase 2: Aggregate analysis ===")

    # Pooled correlations (concatenate all positions across problems)
    all_h2o = np.concatenate([pd["h2o_score"] for pd in problems_data])
    all_ac = np.concatenate([pd["ac_score"] for pd in problems_data])
    all_tc = np.concatenate([pd["tc_score"] for pd in problems_data])
    all_sel = np.concatenate([pd["selectivity"] for pd in problems_data])

    pooled_corrs = compute_correlations(all_h2o, all_ac, all_tc, all_sel)
    n_positions = len(all_h2o)

    print(f"\nPooled correlations (n={n_positions} positions):")
    for name, vals in pooled_corrs.items():
        print(f"  {name}: Spearman={vals['spearman_rho']:.4f} (p={vals['spearman_p']:.2e}), "
              f"Pearson={vals['pearson_r']:.4f}")

    # Mean per-problem correlations
    per_problem_corrs = {}
    for key in ["h2o_vs_ac", "h2o_vs_tc", "h2o_vs_selectivity", "ac_vs_tc"]:
        rhos = [pd["correlations"].get(key, {}).get("spearman_rho", float("nan"))
                for pd in problems_data]
        rhos = [r for r in rhos if not np.isnan(r)]
        if rhos:
            per_problem_corrs[key] = {
                "mean_rho": float(np.mean(rhos)),
                "std_rho": float(np.std(rhos)),
                "median_rho": float(np.median(rhos)),
                "min_rho": float(np.min(rhos)),
                "max_rho": float(np.max(rhos)),
                "n": len(rhos),
            }
            print(f"  {key} (per-problem mean): {np.mean(rhos):.4f} +/- {np.std(rhos):.4f} "
                  f"[{np.min(rhos):.3f}, {np.max(rhos):.3f}]")

    # Mean per-problem overlap at each retention rate
    mean_overlaps = {}
    for rate in RETENTION_RATES:
        rate_key = f"{rate:.1f}"
        jaccards_ac = [pd["overlaps"][rate_key]["h2o_ac_jaccard"] for pd in problems_data]
        jaccards_tc = [pd["overlaps"][rate_key]["h2o_tc_jaccard"] for pd in problems_data]
        jaccards_sel = [pd["overlaps"][rate_key]["h2o_sel_jaccard"] for pd in problems_data]
        ac_sel_frac = [pd["overlaps"][rate_key]["h2o_retains_ac_selective"] for pd in problems_data]
        tc_sel_frac = [pd["overlaps"][rate_key]["h2o_retains_tc_selective"] for pd in problems_data]

        mean_overlaps[rate_key] = {
            "h2o_ac_jaccard_mean": float(np.mean(jaccards_ac)),
            "h2o_tc_jaccard_mean": float(np.mean(jaccards_tc)),
            "h2o_sel_jaccard_mean": float(np.mean(jaccards_sel)),
            "h2o_retains_ac_selective_mean": float(np.mean(ac_sel_frac)),
            "h2o_retains_tc_selective_mean": float(np.mean(tc_sel_frac)),
        }

    print("\nMean overlap (Jaccard) at key retention rates:")
    for rate_key in ["0.3", "0.5", "0.7"]:
        if rate_key in mean_overlaps:
            mo = mean_overlaps[rate_key]
            print(f"  {rate_key}: H2O~AC={mo['h2o_ac_jaccard_mean']:.3f}, "
                  f"H2O~TC={mo['h2o_tc_jaccard_mean']:.3f}, "
                  f"H2O~Sel={mo['h2o_sel_jaccard_mean']:.3f}")
            print(f"         H2O retains: {mo['h2o_retains_ac_selective_mean']:.1%} AC-selective, "
                  f"{mo['h2o_retains_tc_selective_mean']:.1%} TC-selective")

    # Quartile analysis: bin positions by H2O score and report mean AC/TC
    print("\nQuartile analysis (positions binned by H2O score):")
    h2o_quartiles = np.percentile(all_h2o, [25, 50, 75])
    quartile_labels = ["Q1 (lowest H2O)", "Q2", "Q3", "Q4 (highest H2O)"]
    boundaries = [-np.inf] + list(h2o_quartiles) + [np.inf]

    quartile_data = []
    for qi in range(4):
        mask = (all_h2o >= boundaries[qi]) & (all_h2o < boundaries[qi + 1])
        if qi == 3:
            mask = (all_h2o >= boundaries[qi])
        n_q = mask.sum()
        mean_ac = all_ac[mask].mean() if n_q > 0 else 0
        mean_tc = all_tc[mask].mean() if n_q > 0 else 0
        mean_sel = all_sel[mask].mean() if n_q > 0 else 0
        mean_h2o = all_h2o[mask].mean() if n_q > 0 else 0
        quartile_data.append({
            "quartile": quartile_labels[qi],
            "n": int(n_q),
            "mean_h2o": float(mean_h2o),
            "mean_ac": float(mean_ac),
            "mean_tc": float(mean_tc),
            "mean_selectivity": float(mean_sel),
        })
        print(f"  {quartile_labels[qi]}: n={n_q}, H2O={mean_h2o:.4f}, "
              f"AC={mean_ac:.4f}, TC={mean_tc:.4f}, sel={mean_sel:.3f}")

    # Clean up model before figures
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    model_short = model_name.split("/")[-1]
    return {
        "model": model_name,
        "model_short": model_short,
        "n_valid": len(problems_data),
        "n_positions": n_positions,
        "pooled_correlations": pooled_corrs,
        "per_problem_correlations": per_problem_corrs,
        "mean_overlaps": mean_overlaps,
        "quartile_data": quartile_data,
        "per_problem_summary": [
            {
                "ds_idx": pd["ds_idx"],
                "reasoning_len": pd["reasoning_len"],
                "correlations": pd["correlations"],
            }
            for pd in problems_data
        ],
    }


def generate_figures(results_list, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(results_list)

    # ── Figure 1: Per-problem correlation distributions ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    corr_pairs = ["h2o_vs_ac", "h2o_vs_tc", "h2o_vs_selectivity"]
    titles = ["H2O vs AC Score", "H2O vs TC Score", "H2O vs Selectivity (AC-TC)"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for ax, pair, title, color in zip(axes, corr_pairs, titles, colors):
        for ri, res in enumerate(results_list):
            rhos = [p["correlations"].get(pair, {}).get("spearman_rho", float("nan"))
                    for p in res["per_problem_summary"]]
            rhos = [r for r in rhos if not np.isnan(r)]
            if rhos:
                label = f"{res['model_short']} (n={len(rhos)})"
                offset = -0.2 + 0.4 * ri
                bp = ax.boxplot([rhos], positions=[ri], widths=0.35,
                                patch_artist=True, showmeans=True)
                bp["boxes"][0].set_facecolor(colors[ri % len(colors)])
                bp["boxes"][0].set_alpha(0.6)
                ax.text(ri, np.median(rhos) + 0.02, f"med={np.median(rhos):.3f}",
                        ha="center", fontsize=8)
        ax.set_title(f"Spearman rho: {title}")
        ax.set_ylabel("Spearman rho")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([r["model_short"] for r in results_list], fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("H2O Heavy-Hitter Correlations with AC/TC/Selectivity", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "correlation_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: correlation_distributions.png")

    # ── Figure 2: Overlap curves at different retention rates ────────────
    fig2, axes2 = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes2 = [axes2]

    for ax, res in zip(axes2, results_list):
        rates = sorted([float(k) for k in res["mean_overlaps"].keys()])
        ac_jaccards = [res["mean_overlaps"][f"{r:.1f}"]["h2o_ac_jaccard_mean"] for r in rates]
        tc_jaccards = [res["mean_overlaps"][f"{r:.1f}"]["h2o_tc_jaccard_mean"] for r in rates]
        sel_jaccards = [res["mean_overlaps"][f"{r:.1f}"]["h2o_sel_jaccard_mean"] for r in rates]

        rates_pct = [r * 100 for r in rates]
        ax.plot(rates_pct, ac_jaccards, "o-", color="#e74c3c", label="H2O ∩ AC", lw=2, ms=6)
        ax.plot(rates_pct, tc_jaccards, "s-", color="#3498db", label="H2O ∩ TC", lw=2, ms=6)
        ax.plot(rates_pct, sel_jaccards, "^-", color="#2ecc71", label="H2O ∩ Selective", lw=2, ms=6)

        # Add random baseline (Jaccard of two random sets of size k from n)
        # Expected Jaccard ≈ k/(2n-k) for random top-k selections
        # Simplified: at rate r, expected = r/(2-r)
        random_jaccards = [r / (2 - r) for r in rates]
        ax.plot(rates_pct, random_jaccards, "--", color="gray", label="Random baseline", lw=1)

        ax.set_xlabel("Retention Rate (%)")
        ax.set_ylabel("Jaccard Overlap")
        ax.set_title(f"{res['model_short']} (n={res['n_valid']})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.05, 1.05)

    fig2.suptitle("Overlap Between H2O Retained Positions and AC/TC Top Positions", fontsize=13)
    plt.tight_layout()
    fig2.savefig(os.path.join(results_dir, "overlap_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: overlap_curves.png")

    # ── Figure 3: H2O retention composition (AC-selective vs TC-selective) ──
    fig3, axes3 = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes3 = [axes3]

    for ax, res in zip(axes3, results_list):
        rates = sorted([float(k) for k in res["mean_overlaps"].keys()])
        rates_pct = [r * 100 for r in rates]
        ac_sel = [res["mean_overlaps"][f"{r:.1f}"]["h2o_retains_ac_selective_mean"] * 100 for r in rates]
        tc_sel = [res["mean_overlaps"][f"{r:.1f}"]["h2o_retains_tc_selective_mean"] * 100 for r in rates]

        ax.bar(rates_pct, ac_sel, width=6, color="#e74c3c", alpha=0.7, label="AC-selective")
        ax.bar(rates_pct, tc_sel, width=6, bottom=ac_sel, color="#3498db", alpha=0.7, label="TC-selective")
        ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.5, label="50% baseline")
        ax.set_xlabel("Retention Rate (%)")
        ax.set_ylabel("% of H2O-retained positions")
        ax.set_title(f"{res['model_short']}: Composition of H2O-Retained Positions")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig3.savefig(os.path.join(results_dir, "h2o_retention_composition.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: h2o_retention_composition.png")

    # ── Figure 4: Quartile analysis ──────────────────────────────────────
    fig4, axes4 = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes4 = [axes4]

    for ax, res in zip(axes4, results_list):
        qd = res["quartile_data"]
        labels = [q["quartile"].split("(")[0].strip() for q in qd]
        ac_vals = [q["mean_ac"] for q in qd]
        tc_vals = [q["mean_tc"] for q in qd]
        sel_vals = [q["mean_selectivity"] for q in qd]

        x = np.arange(len(labels))
        width = 0.25

        ax.bar(x - width, ac_vals, width, label="AC Score", color="#e74c3c", edgecolor="black", linewidth=0.5)
        ax.bar(x, tc_vals, width, label="TC Score", color="#3498db", edgecolor="black", linewidth=0.5)
        ax.bar(x + width, sel_vals, width, label="Selectivity", color="#2ecc71", edgecolor="black", linewidth=0.5)

        ax.set_xlabel("H2O Score Quartile")
        ax.set_ylabel("Mean Score")
        ax.set_title(f"{res['model_short']}: AC/TC/Selectivity by H2O Quartile")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    fig4.savefig(os.path.join(results_dir, "quartile_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: quartile_analysis.png")

    # ── Figure 5: Pooled correlation summary bar chart ────────────────────
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    corr_names = ["h2o_vs_ac", "h2o_vs_tc", "h2o_vs_selectivity"]
    display_names = ["H2O vs AC", "H2O vs TC", "H2O vs Selectivity"]

    x = np.arange(len(corr_names))
    width = 0.8 / n_models
    model_colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for mi, res in enumerate(results_list):
        rhos = []
        for cn in corr_names:
            rho = res["pooled_correlations"].get(cn, {}).get("spearman_rho", 0)
            rhos.append(rho)
        ax5.bar(x + mi * width - (n_models - 1) * width / 2, rhos, width,
                label=res["model_short"], color=model_colors[mi % len(model_colors)],
                edgecolor="black", linewidth=0.5)

    ax5.set_xlabel("Correlation Pair")
    ax5.set_ylabel("Spearman rho (pooled)")
    ax5.set_title("Pooled Spearman Correlations: H2O vs AC/TC/Selectivity")
    ax5.set_xticks(x)
    ax5.set_xticklabels(display_names)
    ax5.legend()
    ax5.grid(True, alpha=0.2)
    ax5.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    fig5.savefig(os.path.join(results_dir, "pooled_correlation_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: pooled_correlation_summary.png")


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    start_time = time.time()
    print(f"{'='*70}")
    print(f"Experiment 011: H2O Heavy-Hitter vs AC Position Overlap")
    print(f"Problems: {NUM_PROBLEMS}")
    print(f"Retention rates: {RETENTION_RATES}")
    print(f"{'='*70}\n")

    # Load GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(dataset)))
    random.seed(SEED)
    random.shuffle(indices)
    selected = indices[:NUM_PROBLEMS]

    # Run primary model (Qwen)
    model_name_qwen = "Qwen/Qwen3-4B-Base"
    qwen_results = run_model(model_name_qwen, dataset, selected, start_time, time_budget=900)

    elapsed = time.time() - start_time
    print(f"\nQwen elapsed: {elapsed:.0f}s")

    # Run replication model (Llama) if time permits
    results_list = [qwen_results] if "error" not in qwen_results else []
    llama_results = None

    if elapsed < 1200:  # 20 min remaining
        model_name_llama = "meta-llama/Llama-3.1-8B-Instruct"
        llama_results = run_model(model_name_llama, dataset, selected, start_time, time_budget=1600)
        if "error" not in llama_results:
            results_list.append(llama_results)
    else:
        print(f"\nSkipping Llama — {elapsed:.0f}s elapsed, insufficient time remaining")

    if not results_list:
        print("ERROR: No valid results from any model!")
        return

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    for res in results_list:
        print(f"\n{res['model_short']} (n={res['n_valid']} problems, {res['n_positions']} positions):")
        print("  Pooled Spearman correlations:")
        for name, vals in res["pooled_correlations"].items():
            print(f"    {name}: rho={vals['spearman_rho']:.4f} (p={vals['spearman_p']:.2e})")
        print("  Mean per-problem correlations:")
        for name, vals in res["per_problem_correlations"].items():
            print(f"    {name}: mean={vals['mean_rho']:.4f} +/- {vals['std_rho']:.4f}")
        print("  Overlap at 50% retention:")
        if "0.5" in res["mean_overlaps"]:
            mo = res["mean_overlaps"]["0.5"]
            print(f"    H2O~AC Jaccard: {mo['h2o_ac_jaccard_mean']:.3f}")
            print(f"    H2O~TC Jaccard: {mo['h2o_tc_jaccard_mean']:.3f}")
            print(f"    H2O retains: {mo['h2o_retains_ac_selective_mean']:.1%} AC-selective, "
                  f"{mo['h2o_retains_tc_selective_mean']:.1%} TC-selective")

    # Generate figures
    print("\n=== Generating figures ===")
    generate_figures(results_list, RESULTS_DIR)

    # Save summary
    summary = {
        "experiment": "exp_011_h2o_overlap",
        "elapsed_seconds": time.time() - start_time,
        "models": {res["model_short"]: res for res in results_list},
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_convert)
    print(f"\nResults saved to {RESULTS_DIR}/summary.json")

    total_time = time.time() - start_time
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
