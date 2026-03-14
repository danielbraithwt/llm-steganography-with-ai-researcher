#!/usr/bin/env python3
"""
Experiment 016: Positional Confound Analysis on Qwen3-4B-Instruct

Tests whether the reversed dissociation (-47pp) found in exp_015 is a genuine
instruction-tuning effect or a positional confound (as exp_013 showed on Llama).

Key insight from exp_014 positional data:
  - Position vs SEL rho = +0.57 (late → high selectivity = high AC relative to TC)
  - Position vs TC rho = -0.44 (early → high TC)
This means SelTC destruction targets EARLY positions and SelAC targets LATE positions.
If early positions are critical (as on Llama), the reversed dissociation is a confound.

Design:
1. Replicate exp_015 destruction at 1.0x scale (SelAC, SelTC, Random)
2. Add positional strategies (destroy earliest vs latest N%)
3. Position-controlled within-half analysis (SelAC vs SelTC within early/late halves)
4. Position-score correlations and mean noise position recording
5. Protection test replication (AC, H2O, TC, Random)

Uses corrected pipeline from exp_015:
  - Truncate trace at "####" (reasoning only)
  - Additive noise at 1.0x scale
  - Lookback re-computation (20 tokens)
  - Generate answer from end of reasoning
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
MODEL_NAME = "Qwen/Qwen3-4B"
NUM_PROBLEMS = 40
NOISE_SCALE = 1.0
NOISE_FRACTIONS = [0.01, 0.03, 0.05]
ATTENTION_LAYERS_AC = [-1, -2, -3, -4]
MAX_GEN_TOKENS = 768
MAX_SEQ_LEN = 2048
LOOKBACK = 20
SEED = 42
START_IDX = 65  # non-overlapping with exp_014 (0-29) and exp_015 (30-64)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_016")

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
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
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
def teacher_force_with_attention(model, tokenizer, prompt, reasoning_text):
    """Teacher-force prompt + reasoning_text and compute per-position attention scores."""
    full_text = prompt + reasoning_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    reasoning_len = seq_len - prompt_len

    if reasoning_len < 5:
        return None

    if seq_len > MAX_SEQ_LEN:
        max_reasoning = MAX_SEQ_LEN - prompt_len
        trace_tokens = tokenizer(reasoning_text, return_tensors="pt").input_ids[0][:max_reasoning]
        reasoning_text = tokenizer.decode(trace_tokens, skip_special_tokens=True)
        full_text = prompt + reasoning_text
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]
        reasoning_len = seq_len - prompt_len

    outputs = model(**inputs, output_attentions=True, use_cache=False)

    num_layers = len(outputs.attentions)
    layer_indices_ac = [num_layers + i for i in ATTENTION_LAYERS_AC]
    last_token_idx = seq_len - 1

    answer_coupling = torch.zeros(reasoning_len, device=model.device)
    text_coupling = torch.zeros(reasoning_len, device=model.device)
    h2o_score = torch.zeros(reasoning_len, device=model.device)

    mask_full = torch.tril(torch.ones(seq_len, seq_len, device=model.device, dtype=torch.bool), diagonal=-1)
    mask_reason = torch.tril(torch.ones(reasoning_len, reasoning_len, device=model.device, dtype=torch.bool), diagonal=-1)

    for li in range(num_layers):
        attn = outputs.attentions[li][0]
        attn_summed = attn.sum(dim=0)
        col_sums = (attn_summed * mask_full).sum(dim=0)
        h2o_score += col_sums[prompt_len:seq_len]

        if li in layer_indices_ac:
            answer_coupling += attn[:, last_token_idx, prompt_len:seq_len].sum(dim=0)
            reasoning_attn = attn[:, prompt_len:seq_len, prompt_len:seq_len]
            ra_summed = reasoning_attn.sum(dim=0)
            text_coupling += (ra_summed * mask_reason).sum(dim=0)

        del attn, attn_summed
    del mask_full, mask_reason

    answer_coupling = answer_coupling.cpu().numpy()
    text_coupling = text_coupling.cpu().numpy()
    h2o_score = h2o_score.cpu().numpy()

    ac_sum = answer_coupling.sum()
    tc_sum = text_coupling.sum()
    h2o_sum = h2o_score.sum()

    if ac_sum > 0:
        answer_coupling = answer_coupling / ac_sum
    if tc_sum > 0:
        text_coupling = text_coupling / tc_sum
    if h2o_sum > 0:
        h2o_score = h2o_score / h2o_sum

    ac_ranks = scipy_stats.rankdata(answer_coupling)
    tc_ranks = scipy_stats.rankdata(text_coupling)
    selectivity = ac_ranks - tc_ranks

    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "ac": answer_coupling,
        "tc": text_coupling,
        "h2o": h2o_score,
        "sel": selectivity,
        "prompt_len": prompt_len,
        "reasoning_len": reasoning_len,
        "seq_len": seq_len,
    }


@torch.no_grad()
def build_base_kv(model, tokenizer, prompt, reasoning_text):
    """Build the base KV cache by teacher-forcing prompt + reasoning_text."""
    full_text = prompt + reasoning_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    if seq_len > MAX_SEQ_LEN:
        max_reasoning = MAX_SEQ_LEN - prompt_len
        trace_tokens = tokenizer(reasoning_text, return_tensors="pt").input_ids[0][:max_reasoning]
        reasoning_text = tokenizer.decode(trace_tokens, skip_special_tokens=True)
        full_text = prompt + reasoning_text
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]

    outputs = model(**inputs, use_cache=True, output_attentions=False)
    kv_cache = outputs.past_key_values

    reasoning_len = seq_len - prompt_len
    lookback = min(LOOKBACK, reasoning_len)
    lookback_start = seq_len - lookback

    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    return kv_cache, inputs.input_ids, seq_len, prompt_len, lookback_start


def clone_kv_cache(kv_cache):
    """Deep copy of DynamicCache."""
    new_cache = DynamicCache()
    for layer_idx in range(len(kv_cache.layers)):
        layer = kv_cache.layers[layer_idx]
        new_cache.update(layer.keys.clone(), layer.values.clone(), layer_idx)
    return new_cache


@torch.no_grad()
def noised_answer_generation(model, tokenizer, base_kv, input_ids, seq_len,
                              prompt_len, lookback_start, num_layers,
                              positions_to_ablate, noise_scale=1.0):
    """Clone base KV, apply scaled additive noise, lookback, generate answer."""
    past_kv = clone_kv_cache(base_kv)

    abs_positions = [prompt_len + p for p in positions_to_ablate if prompt_len + p < seq_len]

    if abs_positions and noise_scale > 0:
        for layer_idx in range(num_layers):
            layer = past_kv.layers[layer_idx]
            for pos in abs_positions:
                k_vec = layer.keys[:, :, pos, :]
                k_noise = torch.randn_like(k_vec)
                k_noise = k_noise * (noise_scale * k_vec.norm() / (k_noise.norm() + 1e-8))
                layer.keys[:, :, pos, :] = k_vec + k_noise

                v_vec = layer.values[:, :, pos, :]
                v_noise = torch.randn_like(v_vec)
                v_noise = v_noise * (noise_scale * v_vec.norm() / (v_noise.norm() + 1e-8))
                layer.values[:, :, pos, :] = v_vec + v_noise

    lookback_tokens = input_ids[:, lookback_start:seq_len]

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
            if re.search(r'\d+\s*\n', after) or (re.search(r'\d+\s*$', after) and len(after.strip()) >= 2):
                break
        if "\nQ:" in decoded:
            break

    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    del lookback_outputs, past_kv, gen_kv, trunc_kv
    gc.collect()
    torch.cuda.empty_cache()

    return answer_text


def select_positions(strategy, scores, n_noise, n_pos):
    """Select positions to noise based on strategy.

    Returns (positions_list, mean_relative_position).
    """
    half = n_pos // 2

    if strategy == "selac":
        positions = np.argsort(scores["sel"])[-n_noise:].tolist()
    elif strategy == "seltc":
        positions = np.argsort(scores["sel"])[:n_noise].tolist()
    elif strategy == "random":
        positions = np.random.choice(n_pos, n_noise, replace=False).tolist()
    elif strategy == "pos_early":
        positions = list(range(min(n_noise, n_pos)))
    elif strategy == "pos_late":
        positions = list(range(max(0, n_pos - n_noise), n_pos))
    elif strategy == "early_half_selac":
        # Within early half, select most AC-selective
        early_sel = scores["sel"][:half]
        n_sel = min(n_noise, len(early_sel))
        local_idx = np.argsort(early_sel)[-n_sel:]
        positions = local_idx.tolist()
    elif strategy == "early_half_seltc":
        # Within early half, select most TC-selective
        early_sel = scores["sel"][:half]
        n_sel = min(n_noise, len(early_sel))
        local_idx = np.argsort(early_sel)[:n_sel]
        positions = local_idx.tolist()
    elif strategy == "late_half_selac":
        # Within late half, select most AC-selective
        late_sel = scores["sel"][half:]
        n_sel = min(n_noise, len(late_sel))
        local_idx = np.argsort(late_sel)[-n_sel:]
        positions = (local_idx + half).tolist()
    elif strategy == "late_half_seltc":
        # Within late half, select most TC-selective
        late_sel = scores["sel"][half:]
        n_sel = min(n_noise, len(late_sel))
        local_idx = np.argsort(late_sel)[:n_sel]
        positions = (local_idx + half).tolist()
    # Protection strategies (noise BOTTOM by metric)
    elif strategy == "ac_protect":
        positions = np.argsort(scores["ac"])[:n_noise].tolist()
    elif strategy == "h2o_protect":
        positions = np.argsort(scores["h2o"])[:n_noise].tolist()
    elif strategy == "tc_protect":
        positions = np.argsort(scores["tc"])[:n_noise].tolist()
    elif strategy == "random_protect":
        positions = np.random.choice(n_pos, n_noise, replace=False).tolist()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Compute mean relative position
    if positions:
        mean_rel_pos = np.mean(positions) / max(n_pos - 1, 1)
    else:
        mean_rel_pos = 0.5

    return positions, float(mean_rel_pos)


def make_figures(results, results_dir):
    """Generate analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_valid = results.get("n_valid", "?")

    # ── Figure 1: Strategy comparison bars at 3% noise ──
    fig, ax = plt.subplots(figsize=(14, 6))

    dest_strategies = ["selac", "seltc", "random", "pos_early", "pos_late",
                       "early_half_selac", "early_half_seltc", "late_half_selac", "late_half_seltc"]
    dest_labels = ["SelAC", "SelTC", "Random", "Pos-Early", "Pos-Late",
                   "Early+SelAC", "Early+SelTC", "Late+SelAC", "Late+SelTC"]
    colors = ["#e74c3c", "#27ae60", "#95a5a6", "#f39c12", "#8e44ad",
              "#e74c3c", "#27ae60", "#e74c3c", "#27ae60"]
    hatches = ["", "", "", "", "", "//", "//", "//", "//"]

    frac_key = "0.03"
    accs = []
    for s in dest_strategies:
        entry = results.get("destruction", {}).get(s, {}).get(frac_key, {})
        accs.append(entry.get("accuracy", 0))

    bars = ax.bar(range(len(dest_strategies)), accs, color=colors, edgecolor="black", linewidth=0.5)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    ax.set_xticks(range(len(dest_strategies)))
    ax.set_xticklabels(dest_labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Qwen3-4B-Instruct: Destruction at 3% Noise Fraction, 1.0x Scale (n={n_valid})\nRed=AC-selective, Green=TC-selective, Hatched=Within-half")
    ax.axhline(results.get("pipeline_baseline_accuracy", 70), color="blue", linestyle="--",
               linewidth=1, label=f"Pipeline baseline ({results.get('pipeline_baseline_accuracy', 70):.0f}%)")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(accs):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "strategy_comparison_3pct.png"), dpi=150)
    plt.close(fig)

    # ── Figure 2: Mean noise position vs accuracy ──
    fig, ax = plt.subplots(figsize=(10, 6))

    for fi, frac in enumerate(NOISE_FRACTIONS):
        frac_key_f = str(frac)
        positions_list = []
        accs_list = []
        labels_list = []
        for s, label in zip(dest_strategies, dest_labels):
            entry = results.get("destruction", {}).get(s, {}).get(frac_key_f, {})
            if "accuracy" in entry and "mean_noise_position" in entry:
                positions_list.append(entry["mean_noise_position"])
                accs_list.append(entry["accuracy"])
                labels_list.append(label)

        if positions_list:
            ax.scatter(positions_list, accs_list, s=80, alpha=0.7,
                       label=f"NF={frac*100:.0f}%", zorder=3)
            if fi == 1:  # label points at 3%
                for x, y, lbl in zip(positions_list, accs_list, labels_list):
                    ax.annotate(lbl, (x, y), textcoords="offset points",
                                xytext=(5, 5), fontsize=7, alpha=0.8)

    ax.set_xlabel("Mean Relative Position of Noised Tokens (0=start, 1=end)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Position vs Accuracy: Does Position Predict Destruction Impact?\n(n={n_valid})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_xlim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "position_vs_accuracy.png"), dpi=150)
    plt.close(fig)

    # ── Figure 3: Within-half dissociation ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for fi, frac in enumerate(NOISE_FRACTIONS):
        frac_key_f = str(frac)
        ax = axes[fi]

        # Unconstrained
        selac_acc = results["destruction"].get("selac", {}).get(frac_key_f, {}).get("accuracy", 0)
        seltc_acc = results["destruction"].get("seltc", {}).get(frac_key_f, {}).get("accuracy", 0)

        # Early half
        early_selac = results["destruction"].get("early_half_selac", {}).get(frac_key_f, {}).get("accuracy", 0)
        early_seltc = results["destruction"].get("early_half_seltc", {}).get(frac_key_f, {}).get("accuracy", 0)

        # Late half
        late_selac = results["destruction"].get("late_half_selac", {}).get(frac_key_f, {}).get("accuracy", 0)
        late_seltc = results["destruction"].get("late_half_seltc", {}).get(frac_key_f, {}).get("accuracy", 0)

        x = np.arange(3)
        width = 0.35
        bars1 = ax.bar(x - width/2, [selac_acc, early_selac, late_selac], width,
                        color="#e74c3c", alpha=0.8, label="SelAC destroy")
        bars2 = ax.bar(x + width/2, [seltc_acc, early_seltc, late_seltc], width,
                        color="#27ae60", alpha=0.8, label="SelTC destroy")

        ax.set_xticks(x)
        ax.set_xticklabels(["Unconstrained", "Early Half", "Late Half"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"NF={frac*100:.0f}%")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate gaps
        for i, (a, t) in enumerate([(selac_acc, seltc_acc), (early_selac, early_seltc), (late_selac, late_seltc)]):
            gap = t - a
            y_pos = max(a, t) + 3
            ax.text(i, y_pos, f"gap={gap:+.0f}pp", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle(f"Within-Half Dissociation Test (n={n_valid})\nIf gap persists within halves → genuine effect. If gap = 0 → positional confound.", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "within_half_dissociation.png"), dpi=150)
    plt.close(fig)

    # ── Figure 4: Position-score correlations ──
    pos_corr = results.get("position_correlations", {})
    if pos_corr:
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = list(pos_corr.keys())
        rhos = [pos_corr[m]["rho"] for m in metrics]
        colors_c = ["#e74c3c" if "ac" in m.lower() or "sel" in m.lower()
                     else "#27ae60" if "tc" in m.lower()
                     else "#3498db" if "h2o" in m.lower()
                     else "#95a5a6" for m in metrics]

        bars = ax.barh(range(len(metrics)), rhos, color=colors_c, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([m.upper() for m in metrics])
        ax.set_xlabel("Spearman rho with Position (0=early, 1=late)")
        ax.set_title(f"Position-Score Correlations (Qwen3-4B-Instruct, n={n_valid})")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="x")

        for i, v in enumerate(rhos):
            ax.text(v + 0.02 if v >= 0 else v - 0.02, i, f"{v:.3f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9)

        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "position_score_correlations.png"), dpi=150)
        plt.close(fig)

    # ── Figure 5: Protection comparison (replication) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    prot_strategies = ["ac_protect", "h2o_protect", "tc_protect", "random_protect"]
    prot_labels = ["AC", "H2O", "TC", "Random"]
    prot_colors = ["#e74c3c", "#3498db", "#27ae60", "#95a5a6"]

    x = np.arange(len(NOISE_FRACTIONS))
    width = 0.2

    for si, (s, label, color) in enumerate(zip(prot_strategies, prot_labels, prot_colors)):
        accs_p = []
        for frac in NOISE_FRACTIONS:
            frac_key_f = str(frac)
            entry = results.get("protection", {}).get(s, {}).get(frac_key_f, {})
            accs_p.append(entry.get("accuracy", 0))
        ax.bar(x + si * width, accs_p, width, color=color, alpha=0.8, label=f"{label} protect", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"{f*100:.0f}%" for f in NOISE_FRACTIONS])
    ax.set_xlabel("Noise Fraction")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Protection Test Replication (scale=1.0x, n={n_valid})")
    ax.axhline(results.get("pipeline_baseline_accuracy", 70), color="blue", linestyle="--",
               linewidth=1, label="Pipeline baseline")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "protection_replication.png"), dpi=150)
    plt.close(fig)

    print(f"Figures saved to {results_dir}/")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print(f"{'='*70}")
    print(f"Experiment 016: Positional Confound Analysis on Qwen3-4B-Instruct")
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {NUM_PROBLEMS} (indices {START_IDX}-{START_IDX + NUM_PROBLEMS - 1})")
    print(f"Noise scale: {NOISE_SCALE}x")
    print(f"Noise fractions: {NOISE_FRACTIONS}")
    print(f"Pipeline: truncate at ####, lookback={LOOKBACK}, additive noise")
    print(f"{'='*70}\n")

    # ── Load model ──
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"Loaded. Layers={num_layers}, device={model.device}")
    print(f"Load time: {time.time() - t_start:.1f}s\n")

    # ── Load GSM8K ──
    print("Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    problems = list(dataset.select(range(START_IDX, START_IDX + NUM_PROBLEMS)))
    print(f"Using {len(problems)} problems (indices {START_IDX}-{START_IDX + NUM_PROBLEMS - 1})\n")

    # ── Phase 1: Generate baseline traces ──
    print("Phase 1: Generating baseline traces...")
    valid_problems = []

    for i, prob in enumerate(problems):
        question = prob["question"]
        true_ans = normalize_answer(prob["answer"].split("####")[-1].strip())
        prompt = build_prompt(question)

        try:
            trace = generate_trace(model, tokenizer, prompt)
            pred_ans = normalize_answer(extract_answer(trace))
            correct = (pred_ans == true_ans)
            print(f"  Problem {i+1}/{len(problems)}: pred={pred_ans} true={true_ans} {'OK' if correct else 'WRONG'}")

            if correct:
                if "####" in trace:
                    reasoning_text = trace[:trace.index("####")]
                else:
                    reasoning_text = trace

                valid_problems.append({
                    "idx": START_IDX + i,
                    "question": question,
                    "true_answer": true_ans,
                    "prompt": prompt,
                    "trace": trace,
                    "reasoning_text": reasoning_text,
                })
        except Exception as e:
            print(f"  Problem {i+1}/{len(problems)}: ERROR - {e}")

        gc.collect()
        torch.cuda.empty_cache()

    n_valid = len(valid_problems)
    baseline_acc = 100 * n_valid / len(problems)
    print(f"\nBaseline: {n_valid}/{len(problems)} correct ({baseline_acc:.0f}%)")
    print(f"Phase 1 time: {time.time() - t_start:.1f}s\n")

    if n_valid < 5:
        print("ERROR: Too few valid problems. Aborting.")
        results = {"error": "too_few_valid", "n_valid": n_valid, "n_total": len(problems)}
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return

    # ── Phase 2: Compute attention scores ──
    print("Phase 2: Computing attention scores...")
    for pi, prob in enumerate(valid_problems):
        print(f"  Attention {pi+1}/{n_valid}...", end=" ")
        try:
            scores = teacher_force_with_attention(model, tokenizer, prob["prompt"], prob["reasoning_text"])
            if scores is None:
                print("SKIP (too short)")
                prob["scores"] = None
                continue
            prob["scores"] = scores
            print(f"OK (reasoning_len={scores['reasoning_len']})")
        except Exception as e:
            print(f"ERROR: {e}")
            prob["scores"] = None
        gc.collect()
        torch.cuda.empty_cache()

    valid_problems = [p for p in valid_problems if p.get("scores") is not None]
    n_valid = len(valid_problems)
    print(f"\n{n_valid} problems with valid attention scores")
    print(f"Phase 2 time: {time.time() - t_start:.1f}s\n")

    if n_valid < 5:
        print("ERROR: Too few valid problems after attention. Aborting.")
        results = {"error": "too_few_valid_attention", "n_valid": n_valid}
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return

    # ── Phase 2.5: Compute position-score correlations ──
    print("Phase 2.5: Position-score correlations...")
    all_positions = []
    all_ac = []
    all_tc = []
    all_h2o = []
    all_sel = []

    for prob in valid_problems:
        s = prob["scores"]
        n_pos = s["reasoning_len"]
        rel_pos = np.arange(n_pos) / max(n_pos - 1, 1)
        all_positions.extend(rel_pos)
        all_ac.extend(s["ac"])
        all_tc.extend(s["tc"])
        all_h2o.extend(s["h2o"])
        all_sel.extend(s["sel"])

    position_correlations = {}
    for name, values in [("ac", all_ac), ("tc", all_tc), ("h2o", all_h2o), ("sel", all_sel)]:
        rho, p = scipy_stats.spearmanr(all_positions, values)
        position_correlations[name] = {"rho": round(rho, 4), "p": round(p, 6)}
        print(f"  Position vs {name.upper()}: rho={rho:.4f}, p={p:.2e}")

    n_total_positions = len(all_positions)
    print(f"  Total positions: {n_total_positions}")
    print()

    # ── Phase 2.75: Control test ──
    print("Phase 2.75: Control test (clean pipeline)...")
    control_pass = 0
    for pi, prob in enumerate(valid_problems[:3]):
        scores = prob["scores"]
        base_kv, input_ids, seq_len, prompt_len_actual, lookback_start = \
            build_base_kv(model, tokenizer, prob["prompt"], prob["reasoning_text"])
        gen_text = noised_answer_generation(
            model, tokenizer, base_kv, input_ids, seq_len,
            prompt_len_actual, lookback_start, num_layers,
            positions_to_ablate=[], noise_scale=0
        )
        gen_ans = normalize_answer(extract_answer(gen_text))
        ok = (gen_ans == prob["true_answer"])
        control_pass += int(ok)
        print(f"  Control {pi+1}: pred={gen_ans} true={prob['true_answer']} {'OK' if ok else 'FAIL'}")
        if not ok:
            print(f"    Generated: {gen_text[:200]}")
        del base_kv
        gc.collect()
        torch.cuda.empty_cache()

    print(f"  Control: {control_pass}/3 correct")
    if control_pass == 0:
        print("ERROR: Control test failed. Aborting.")
        return
    print()

    # ── Phase 3: Destruction and protection tests ──
    destruction_strategies = ["selac", "seltc", "random", "pos_early", "pos_late",
                              "early_half_selac", "early_half_seltc",
                              "late_half_selac", "late_half_seltc"]
    protection_strategies = ["ac_protect", "h2o_protect", "tc_protect", "random_protect"]
    all_strategies = destruction_strategies + protection_strategies

    print(f"Phase 3: Running {len(all_strategies)} strategies x {len(NOISE_FRACTIONS)} fractions...")
    print(f"  Destruction: {destruction_strategies}")
    print(f"  Protection:  {protection_strategies}")
    print()

    # Initialize results
    dest_results = {s: {} for s in destruction_strategies}
    prot_results = {s: {} for s in protection_strategies}

    # Track pipeline baseline
    pipeline_correct = 0

    for pi, prob in enumerate(valid_problems):
        scores = prob["scores"]
        n_pos = scores["reasoning_len"]
        prompt_len = scores["prompt_len"]

        print(f"  Problem {pi+1}/{n_valid} (idx={prob['idx']}, n_pos={n_pos})...", end=" ", flush=True)

        # Build base KV cache ONCE per problem
        try:
            base_kv, input_ids, seq_len, prompt_len_actual, lookback_start = \
                build_base_kv(model, tokenizer, prob["prompt"], prob["reasoning_text"])
        except Exception as e:
            print(f"KV build error: {e}")
            continue

        # Check pipeline baseline (clean generation)
        try:
            clean_text = noised_answer_generation(
                model, tokenizer, base_kv, input_ids, seq_len,
                prompt_len_actual, lookback_start, num_layers,
                positions_to_ablate=[], noise_scale=0
            )
            clean_ans = normalize_answer(extract_answer(clean_text))
            pipeline_ok = (clean_ans == prob["true_answer"])
            pipeline_correct += int(pipeline_ok)
        except Exception:
            pipeline_ok = False

        for frac in NOISE_FRACTIONS:
            n_noise = max(1, int(n_pos * frac))
            frac_key = str(frac)

            # Destruction tests
            for strategy in destruction_strategies:
                if frac_key not in dest_results[strategy]:
                    dest_results[strategy][frac_key] = {"correct": 0, "total": 0, "positions": []}

                try:
                    positions, mean_pos = select_positions(strategy, scores, n_noise, n_pos)
                    gen_text = noised_answer_generation(
                        model, tokenizer, base_kv, input_ids, seq_len,
                        prompt_len_actual, lookback_start, num_layers,
                        positions, noise_scale=NOISE_SCALE
                    )
                    pred_ans = normalize_answer(extract_answer(gen_text))
                    is_correct = (pred_ans == prob["true_answer"])
                    dest_results[strategy][frac_key]["correct"] += int(is_correct)
                    dest_results[strategy][frac_key]["positions"].append(mean_pos)
                except Exception:
                    pass
                dest_results[strategy][frac_key]["total"] += 1

            # Protection tests
            for strategy in protection_strategies:
                if frac_key not in prot_results[strategy]:
                    prot_results[strategy][frac_key] = {"correct": 0, "total": 0, "positions": []}

                try:
                    positions, mean_pos = select_positions(strategy, scores, n_noise, n_pos)
                    gen_text = noised_answer_generation(
                        model, tokenizer, base_kv, input_ids, seq_len,
                        prompt_len_actual, lookback_start, num_layers,
                        positions, noise_scale=NOISE_SCALE
                    )
                    pred_ans = normalize_answer(extract_answer(gen_text))
                    is_correct = (pred_ans == prob["true_answer"])
                    prot_results[strategy][frac_key]["correct"] += int(is_correct)
                    prot_results[strategy][frac_key]["positions"].append(mean_pos)
                except Exception:
                    pass
                prot_results[strategy][frac_key]["total"] += 1

        del base_kv
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        print(f"done ({elapsed:.0f}s)", flush=True)

        if elapsed > 1500:
            print(f"\nWARNING: approaching time limit ({elapsed:.0f}s). Stopping after {pi+1} problems.")
            n_valid = pi + 1
            break

    pipeline_baseline_acc = 100 * pipeline_correct / n_valid if n_valid > 0 else 0
    print(f"\nPipeline baseline: {pipeline_correct}/{n_valid} = {pipeline_baseline_acc:.0f}%")
    print(f"Phase 3 time: {time.time() - t_start:.1f}s\n")

    # ── Compute accuracies and mean positions ──
    for strategy in destruction_strategies:
        for frac_key in dest_results[strategy]:
            d = dest_results[strategy][frac_key]
            if d["total"] > 0:
                d["accuracy"] = round(100 * d["correct"] / d["total"], 1)
                if d["positions"]:
                    d["mean_noise_position"] = round(np.mean(d["positions"]), 4)

    for strategy in protection_strategies:
        for frac_key in prot_results[strategy]:
            d = prot_results[strategy][frac_key]
            if d["total"] > 0:
                d["accuracy"] = round(100 * d["correct"] / d["total"], 1)
                if d["positions"]:
                    d["mean_noise_position"] = round(np.mean(d["positions"]), 4)

    # ── Print results ──
    print("=" * 70)
    print("DESTRUCTION TEST RESULTS")
    print("=" * 70)
    for strategy in destruction_strategies:
        print(f"\n  {strategy}:")
        for frac in NOISE_FRACTIONS:
            frac_key = str(frac)
            d = dest_results[strategy].get(frac_key, {})
            if d.get("total", 0) > 0:
                pos_str = f" (mean_pos={d.get('mean_noise_position', '?'):.3f})" if 'mean_noise_position' in d else ""
                print(f"    NF={frac*100:.0f}%: {d['correct']}/{d['total']} = {d['accuracy']:.1f}%{pos_str}")

    print(f"\n{'='*70}")
    print("PROTECTION TEST RESULTS")
    print("=" * 70)
    for strategy in protection_strategies:
        print(f"\n  {strategy}:")
        for frac in NOISE_FRACTIONS:
            frac_key = str(frac)
            d = prot_results[strategy].get(frac_key, {})
            if d.get("total", 0) > 0:
                pos_str = f" (mean_pos={d.get('mean_noise_position', '?'):.3f})" if 'mean_noise_position' in d else ""
                print(f"    NF={frac*100:.0f}%: {d['correct']}/{d['total']} = {d['accuracy']:.1f}%{pos_str}")

    # ── Dissociation analysis ──
    print(f"\n{'='*70}")
    print("DISSOCIATION ANALYSIS")
    print("=" * 70)

    dissociation = {}
    for frac in NOISE_FRACTIONS:
        frac_key = str(frac)
        selac = dest_results["selac"].get(frac_key, {}).get("accuracy", None)
        seltc = dest_results["seltc"].get(frac_key, {}).get("accuracy", None)
        if selac is not None and seltc is not None:
            gap = seltc - selac
            dissociation[f"unconstrained_{frac_key}"] = gap
            print(f"  Unconstrained NF={frac*100:.0f}%: SelTC={seltc:.1f}% - SelAC={selac:.1f}% = {gap:+.1f}pp")

        # Within-half gaps
        for half_label in ["early_half", "late_half"]:
            ha = dest_results[f"{half_label}_selac"].get(frac_key, {}).get("accuracy", None)
            ht = dest_results[f"{half_label}_seltc"].get(frac_key, {}).get("accuracy", None)
            if ha is not None and ht is not None:
                gap_h = ht - ha
                dissociation[f"{half_label}_{frac_key}"] = gap_h
                print(f"  {half_label} NF={frac*100:.0f}%: SelTC={ht:.1f}% - SelAC={ha:.1f}% = {gap_h:+.1f}pp")

        # Positional strategies
        pos_early = dest_results["pos_early"].get(frac_key, {}).get("accuracy", None)
        pos_late = dest_results["pos_late"].get(frac_key, {}).get("accuracy", None)
        if pos_early is not None and pos_late is not None:
            pos_gap = pos_late - pos_early
            dissociation[f"positional_{frac_key}"] = pos_gap
            print(f"  Positional NF={frac*100:.0f}%: Late={pos_late:.1f}% - Early={pos_early:.1f}% = {pos_gap:+.1f}pp")
        print()

    # ── Key summary ──
    print("=" * 70)
    print("KEY COMPARISON AT 3% NOISE FRACTION")
    print("=" * 70)

    frac_key = "0.03"
    for s in destruction_strategies:
        d = dest_results[s].get(frac_key, {})
        pos = d.get("mean_noise_position", "?")
        acc = d.get("accuracy", "?")
        pos_str = f"{pos:.3f}" if isinstance(pos, float) else pos
        acc_str = f"{acc:.1f}%" if isinstance(acc, (int, float)) else acc
        print(f"  {s:25s}: acc={acc_str:>7s}  mean_pos={pos_str}")

    print()
    for s in protection_strategies:
        d = prot_results[s].get(frac_key, {})
        pos = d.get("mean_noise_position", "?")
        acc = d.get("accuracy", "?")
        pos_str = f"{pos:.3f}" if isinstance(pos, float) else pos
        acc_str = f"{acc:.1f}%" if isinstance(acc, (int, float)) else acc
        print(f"  {s:25s}: acc={acc_str:>7s}  mean_pos={pos_str}")

    # ── Confound assessment ──
    print(f"\n{'='*70}")
    print("CONFOUND ASSESSMENT")
    print("=" * 70)

    frac_key = "0.03"
    seltc_pos = dest_results["seltc"].get(frac_key, {}).get("mean_noise_position", None)
    selac_pos = dest_results["selac"].get(frac_key, {}).get("mean_noise_position", None)
    pos_early_acc = dest_results["pos_early"].get(frac_key, {}).get("accuracy", None)
    pos_late_acc = dest_results["pos_late"].get(frac_key, {}).get("accuracy", None)
    seltc_acc = dest_results["seltc"].get(frac_key, {}).get("accuracy", None)
    selac_acc = dest_results["selac"].get(frac_key, {}).get("accuracy", None)

    if all(v is not None for v in [seltc_pos, selac_pos, pos_early_acc, pos_late_acc, seltc_acc, selac_acc]):
        print(f"  SelTC noises position {seltc_pos:.3f} → accuracy {seltc_acc:.1f}%")
        print(f"  SelAC noises position {selac_pos:.3f} → accuracy {selac_acc:.1f}%")
        print(f"  POS_EARLY accuracy: {pos_early_acc:.1f}%")
        print(f"  POS_LATE accuracy: {pos_late_acc:.1f}%")

        if pos_early_acc < pos_late_acc - 10:
            print(f"\n  FINDING: Early positions ARE more critical (by {pos_late_acc - pos_early_acc:.0f}pp)")
            if seltc_pos < selac_pos:
                print(f"  SelTC noises earlier ({seltc_pos:.3f}) than SelAC ({selac_pos:.3f})")
                print(f"  → CONSISTENT WITH POSITIONAL CONFOUND")
            else:
                print(f"  SelTC noises later ({seltc_pos:.3f}) than SelAC ({selac_pos:.3f})")
                print(f"  → INCONSISTENT WITH POSITIONAL CONFOUND (reversed dissociation is genuine)")
        elif pos_early_acc > pos_late_acc + 10:
            print(f"\n  FINDING: Late positions are more critical (by {pos_early_acc - pos_late_acc:.0f}pp)")
        else:
            print(f"\n  FINDING: Position has minimal effect (gap={pos_late_acc - pos_early_acc:.0f}pp)")

        # Within-half test
        early_gap = dissociation.get(f"early_half_{frac_key}", None)
        late_gap = dissociation.get(f"late_half_{frac_key}", None)
        unconstrained_gap = dissociation.get(f"unconstrained_{frac_key}", None)
        if all(v is not None for v in [early_gap, late_gap, unconstrained_gap]):
            print(f"\n  WITHIN-HALF TEST:")
            print(f"    Unconstrained gap: {unconstrained_gap:+.1f}pp")
            print(f"    Early half gap:    {early_gap:+.1f}pp")
            print(f"    Late half gap:     {late_gap:+.1f}pp")
            if abs(early_gap) < 10 and abs(late_gap) < 10 and abs(unconstrained_gap) > 15:
                print(f"  → POSITIONAL CONFOUND CONFIRMED: gap disappears within halves")
            elif abs(early_gap) > 10 or abs(late_gap) > 10:
                print(f"  → GENUINE EFFECT: gap persists within position-controlled halves")
            else:
                print(f"  → INCONCLUSIVE")

    # ── Save results ──
    # Remove position lists (too large for JSON) and replace with means
    for d in [dest_results, prot_results]:
        for strategy in d:
            for frac_key in d[strategy]:
                if "positions" in d[strategy][frac_key]:
                    del d[strategy][frac_key]["positions"]

    results = {
        "model": MODEL_NAME,
        "n_total": len(problems),
        "n_valid": n_valid,
        "baseline_accuracy_direct": baseline_acc,
        "pipeline_baseline_accuracy": pipeline_baseline_acc,
        "noise_scale": NOISE_SCALE,
        "noise_fractions": NOISE_FRACTIONS,
        "start_idx": START_IDX,
        "pipeline": "truncate_at_####_lookback_20_additive_noise",
        "destruction": dest_results,
        "protection": prot_results,
        "dissociation": dissociation,
        "position_correlations": position_correlations,
        "n_total_positions": n_total_positions,
        "elapsed_seconds": time.time() - t_start,
    }

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Generate figures ──
    try:
        make_figures(results, RESULTS_DIR)
    except Exception as e:
        print(f"Figure generation error: {e}")
        import traceback
        traceback.print_exc()

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"{'='*70}")
    print("EXPERIMENT 016 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
