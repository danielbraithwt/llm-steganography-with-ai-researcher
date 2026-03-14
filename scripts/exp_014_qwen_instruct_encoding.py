#!/usr/bin/env python3
"""
Experiment 014: Encoding Strategy on Qwen3-4B (Instruct)

Tests whether instruction tuning affects KV cache encoding properties.
Qwen3-4B is architecturally identical to Qwen3-4B-Base (36 layers, 8 KV heads,
32 query heads, 2560 hidden dim) but instruction-tuned.

Key question: Is concentrated encoding a Qwen ARCHITECTURE property
(should replicate on instruct) or a BASE MODEL property (might change
after instruction tuning)?

Tests both:
1. Noise DESTRUCTION (SelAC vs SelTC vs Random) - replicates exp_004
2. Noise PROTECTION (AC, H2O, TC, Random) - replicates exp_012/013
3. Positional analysis - replicates exp_013

If encoding is concentrated (like Qwen3-4B-Base):
  - SelAC destruction should hurt MORE than SelTC (>15pp gap)
  - AC protection should work BETTER than random
  - PGD-like concentration at answer-coupled positions

If encoding is distributed (like Llama-3.1-8B-Instruct):
  - SelAC ≈ SelTC for destruction
  - AC protection ≈ Random; TC/H2O >> AC
  - Position dominates over attention-based metrics
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
NUM_PROBLEMS = 30
NOISE_FRACTIONS = [0.01, 0.03, 0.05, 0.07]
DESTRUCTION_FRACTIONS = [0.01, 0.03, 0.05]
ATTENTION_LAYERS_AC = [-1, -2, -3, -4]  # last 4 layers for AC/TC
MAX_GEN_TOKENS = 768
MAX_SEQ_LEN = 2048
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_014")

os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 8-shot GSM8K exemplars (same as all prior experiments) ─────────────
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
    # Strip any <think>...</think> blocks (Qwen3 instruct may produce these)
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

        # Stop if we see #### followed by a number and newline (number must be complete)
        if "####" in current_text:
            after = current_text.split("####")[-1]
            if re.search(r'\d+\s*\n', after):
                break

        # Stop if model starts a new question
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
    """Teacher-force the trace and compute per-position attention scores."""
    full_text = prompt + trace_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    reasoning_len = seq_len - prompt_len

    if reasoning_len < 5:
        return None

    if seq_len > MAX_SEQ_LEN:
        max_reasoning = MAX_SEQ_LEN - prompt_len
        trace_tokens = tokenizer(trace_text, return_tensors="pt").input_ids[0][:max_reasoning]
        trace_text = tokenizer.decode(trace_tokens, skip_special_tokens=True)
        full_text = prompt + trace_text
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]
        reasoning_len = seq_len - prompt_len

    outputs = model(**inputs, output_attentions=True, use_cache=False)

    num_layers = len(outputs.attentions)
    layer_indices_ac = [num_layers + i for i in ATTENTION_LAYERS_AC]
    last_token_idx = seq_len - 1

    # ── Compute per-position scores ──
    answer_coupling = torch.zeros(reasoning_len, device=model.device)
    text_coupling = torch.zeros(reasoning_len, device=model.device)
    h2o_score = torch.zeros(reasoning_len, device=model.device)

    # Pre-compute lower-triangular masks for vectorized column sums
    mask_full = torch.tril(torch.ones(seq_len, seq_len, device=model.device, dtype=torch.bool), diagonal=-1)
    mask_reason = torch.tril(torch.ones(reasoning_len, reasoning_len, device=model.device, dtype=torch.bool), diagonal=-1)

    for li in range(num_layers):
        attn = outputs.attentions[li][0]  # (num_heads, seq_len, seq_len)

        # H2O: cumulative attention from all subsequent tokens to each reasoning position
        # Vectorized: for each column j, sum rows > j (lower triangle), across all heads
        attn_summed = attn.sum(dim=0)  # (seq_len, seq_len)
        col_sums = (attn_summed * mask_full).sum(dim=0)  # (seq_len,)
        h2o_score += col_sums[prompt_len:seq_len]

        # AC and TC only from last 4 layers
        if li in layer_indices_ac:
            # Answer coupling: attention from last token to each reasoning position
            answer_coupling += attn[:, last_token_idx, prompt_len:seq_len].sum(dim=0)

            # Text coupling: vectorized lower-triangle column sum on reasoning submatrix
            reasoning_attn = attn[:, prompt_len:seq_len, prompt_len:seq_len]
            ra_summed = reasoning_attn.sum(dim=0)  # (reasoning_len, reasoning_len)
            text_coupling += (ra_summed * mask_reason).sum(dim=0)

        del attn, attn_summed
    del mask_full, mask_reason

    # Normalize
    answer_coupling = answer_coupling.cpu().numpy()
    text_coupling = text_coupling.cpu().numpy()
    h2o_score = h2o_score.cpu().numpy()

    # Avoid division by zero
    ac_sum = answer_coupling.sum()
    tc_sum = text_coupling.sum()
    h2o_sum = h2o_score.sum()

    if ac_sum > 0:
        answer_coupling = answer_coupling / ac_sum
    if tc_sum > 0:
        text_coupling = text_coupling / tc_sum
    if h2o_sum > 0:
        h2o_score = h2o_score / h2o_sum

    # Selectivity = rank(AC) - rank(TC)
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
def build_kv_cache(model, tokenizer, prompt, trace_text):
    """Build the base KV cache via teacher-forcing."""
    full_text = prompt + trace_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]

    if seq_len > MAX_SEQ_LEN:
        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        max_reasoning = MAX_SEQ_LEN - prompt_len
        trace_tokens = tokenizer(trace_text, return_tensors="pt").input_ids[0][:max_reasoning]
        trace_text = tokenizer.decode(trace_tokens, skip_special_tokens=True)
        full_text = prompt + trace_text
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]

    outputs = model(**inputs, use_cache=True)
    kv_cache = outputs.past_key_values

    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    return kv_cache, seq_len


def clone_kv_cache(kv_cache):
    """Deep copy of DynamicCache (compatible with transformers >=5.3)."""
    new_cache = DynamicCache()
    for layer_idx in range(len(kv_cache.layers)):
        layer = kv_cache.layers[layer_idx]
        new_cache.update(layer.keys.clone(), layer.values.clone(), layer_idx)
    return new_cache


def apply_noise_to_positions(kv_cache, positions, prompt_len, num_layers):
    """Apply norm-matched Gaussian noise to specified reasoning positions."""
    for layer_idx in range(num_layers):
        layer = kv_cache.layers[layer_idx]
        k = layer.keys
        v = layer.values
        for pos in positions:
            abs_pos = prompt_len + pos
            if abs_pos >= k.shape[2]:
                continue
            # Key noise
            k_vec = k[0, :, abs_pos, :]
            k_noise = torch.randn_like(k_vec)
            k_noise = k_noise * (k_vec.norm() / (k_noise.norm() + 1e-8))
            k[0, :, abs_pos, :] = k_vec + k_noise
            # Value noise
            v_vec = v[0, :, abs_pos, :]
            v_noise = torch.randn_like(v_vec)
            v_noise = v_noise * (v_vec.norm() / (v_noise.norm() + 1e-8))
            v[0, :, abs_pos, :] = v_vec + v_noise


@torch.no_grad()
def run_noised_generation(model, tokenizer, kv_cache, seq_len, last_token_id, max_tokens=64):
    """Generate from a KV cache, starting from the last token position."""
    # Truncate to seq_len-1
    trunc_cache = DynamicCache()
    for layer_idx in range(len(kv_cache.layers)):
        layer = kv_cache.layers[layer_idx]
        trunc_cache.update(
            layer.keys[:, :, :seq_len-1, :].clone(),
            layer.values[:, :, :seq_len-1, :].clone(),
            layer_idx
        )

    # Feed last token through truncated cache
    last_token = torch.tensor([[last_token_id]], device=model.device)
    outputs = model(input_ids=last_token, past_key_values=trunc_cache, use_cache=True)
    past_kv = outputs.past_key_values

    generated_ids = []
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated_ids.append(next_token[0, 0].item())

    for step in range(max_tokens - 1):
        if next_token[0, 0].item() == tokenizer.eos_token_id:
            break

        outputs = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids.append(next_token[0, 0].item())

        text_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "####" in text_so_far:
            after = text_so_far.split("####")[-1]
            if re.search(r'\d+\s*\n', after) or re.search(r'\d+\s*$', after) and len(after.strip()) >= 2:
                break
        if "\nQ:" in text_so_far:
            break

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    del past_kv, outputs, trunc_cache
    gc.collect()
    torch.cuda.empty_cache()
    return text


def select_positions_by_strategy(scores, n_positions, n_select, strategy_type):
    """
    Select positions to NOISE.
    For 'destroy': noise the TOP positions (most important by metric)
    For 'protect': noise the BOTTOM positions (least important by metric)
    """
    indices = np.argsort(scores)
    if strategy_type == "destroy":
        # Noise the highest-scoring positions
        return indices[-n_select:]
    elif strategy_type == "protect":
        # Noise the lowest-scoring positions (protect the highest)
        return indices[:n_select]
    elif strategy_type == "random":
        return np.random.choice(n_positions, n_select, replace=False)
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")


def make_figures(results, results_dir):
    """Generate analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Figure 1: Destruction test accuracy sweep ──
    fig, ax = plt.subplots(figsize=(8, 5))
    destruction = results.get("destruction", {})
    for strategy in ["selac", "seltc", "random"]:
        fracs = sorted(destruction.get(strategy, {}).keys())
        if not fracs:
            continue
        accs = [destruction[strategy][f]["accuracy"] for f in fracs]
        label = {"selac": "SelAC (destroy)", "seltc": "SelTC (destroy)", "random": "Random"}[strategy]
        ax.plot([float(f) for f in fracs], accs, 'o-', label=label, linewidth=2)
    ax.set_xlabel("Noise Fraction")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Qwen3-4B (Instruct): Noise Destruction Test\n(n={results.get('n_valid', '?')})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "destruction_accuracy_sweep.png"), dpi=150)
    plt.close(fig)

    # ── Figure 2: Protection test accuracy sweep ──
    fig, ax = plt.subplots(figsize=(8, 5))
    protection = results.get("protection", {})
    colors = {"ac": "red", "h2o": "blue", "tc": "green", "random": "gray"}
    for strategy in ["ac", "h2o", "tc", "random"]:
        fracs = sorted(protection.get(strategy, {}).keys())
        if not fracs:
            continue
        accs = [protection[strategy][f]["accuracy"] for f in fracs]
        ax.plot([float(f) for f in fracs], accs, 'o-', label=f"{strategy.upper()} protect",
                linewidth=2, color=colors.get(strategy, None))
    ax.set_xlabel("Noise Fraction")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Qwen3-4B (Instruct): Noise Protection Test\n(n={results.get('n_valid', '?')})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "protection_accuracy_sweep.png"), dpi=150)
    plt.close(fig)

    # ── Figure 3: Positional analysis ──
    pos_data = results.get("positional_analysis", {})
    if pos_data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Position-score correlations
        scores_names = ["ac", "tc", "h2o", "sel"]
        rhos = [pos_data.get(f"position_{s}_rho", 0) for s in scores_names]
        colors_bar = ["red", "green", "blue", "purple"]
        axes[0].bar(range(len(scores_names)), rhos, color=colors_bar)
        axes[0].set_xticks(range(len(scores_names)))
        axes[0].set_xticklabels([s.upper() for s in scores_names])
        axes[0].set_ylabel("Spearman rho with Position")
        axes[0].set_title("Position-Score Correlations")
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[0].grid(True, alpha=0.3)

        # Panel B: Mean noise position by strategy
        all_strategies = ["selac_destroy", "seltc_destroy", "ac_protect", "h2o_protect", "tc_protect", "random"]
        mean_positions = []
        labels = []
        for s in all_strategies:
            mp = pos_data.get(f"mean_position_{s}", None)
            if mp is not None:
                mean_positions.append(mp)
                labels.append(s.replace("_", "\n"))
        if mean_positions:
            bar_colors = ["darkred", "darkgreen", "red", "blue", "green", "gray"][:len(mean_positions)]
            axes[1].bar(range(len(mean_positions)), mean_positions, color=bar_colors)
            axes[1].set_xticks(range(len(mean_positions)))
            axes[1].set_xticklabels(labels, fontsize=8)
            axes[1].set_ylabel("Mean Relative Position of Noised Positions")
            axes[1].set_title("Where Each Strategy Noises")
            axes[1].set_ylim(0, 1.1)
            axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Qwen3-4B (Instruct): Positional Analysis (n={results.get('n_valid', '?')})")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "positional_analysis.png"), dpi=150)
        plt.close(fig)

    # ── Figure 4: Cross-model comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Destruction comparison at 5%
    models = ["Qwen3-4B-Base\n(exp_004)", "Llama-3.1-8B\n(exp_005)", "Qwen3-4B-Inst\n(exp_014)"]
    # exp_004 values at 5%: SelAC=44.1%, SelTC=67.6%
    # exp_005 values at 5%: SelAC=19.0%, SelTC=42.9%
    selac_vals = [44.1, 19.0]
    seltc_vals = [67.6, 42.9]
    if "0.05" in destruction.get("selac", {}):
        selac_vals.append(destruction["selac"]["0.05"]["accuracy"])
        seltc_vals.append(destruction["seltc"]["0.05"]["accuracy"])
    elif "0.03" in destruction.get("selac", {}):
        # Use 3% if 5% not available
        models[2] = "Qwen3-4B-Inst\n(exp_014, 3%)"
        selac_vals = [None, None, destruction["selac"]["0.03"]["accuracy"]]
        seltc_vals = [None, None, destruction["seltc"]["0.03"]["accuracy"]]

    if len(selac_vals) == 3 and all(v is not None for v in selac_vals):
        x = np.arange(len(models))
        w = 0.35
        axes[0].bar(x - w/2, selac_vals, w, label="SelAC destroy", color="red", alpha=0.7)
        axes[0].bar(x + w/2, seltc_vals, w, label="SelTC destroy", color="green", alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Destruction Test at 5% Noise\n(lower = more destructive)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Protection comparison at 3%
    models_p = ["Llama-3.1-8B\n(exp_013)", "Qwen3-4B-Inst\n(exp_014)"]
    # exp_013 at 3%: AC=24%, H2O=68%, TC=72%, Random=36%
    if "0.03" in protection.get("ac", {}):
        exp013 = [24, 68, 72, 36]
        exp014 = [
            protection["ac"]["0.03"]["accuracy"],
            protection["h2o"]["0.03"]["accuracy"],
            protection["tc"]["0.03"]["accuracy"],
            protection["random"]["0.03"]["accuracy"],
        ]
        strats = ["AC", "H2O", "TC", "Random"]
        x = np.arange(len(strats))
        w = 0.35
        axes[1].bar(x - w/2, exp013, w, label="Llama-3.1-8B (exp_013)", alpha=0.7)
        axes[1].bar(x + w/2, exp014, w, label="Qwen3-4B-Inst (exp_014)", alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(strats)
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Protection Test at 3% Noise\n(higher = better protection)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle("Cross-Model Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "cross_model_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"Figures saved to {results_dir}/")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print(f"{'='*70}")
    print(f"Experiment 014: Qwen3-4B (Instruct) Encoding Strategy")
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {NUM_PROBLEMS}")
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
    problems = list(dataset.select(range(NUM_PROBLEMS)))
    print(f"Using {len(problems)} problems\n")

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
                valid_problems.append({
                    "idx": i,
                    "question": question,
                    "true_answer": true_ans,
                    "prompt": prompt,
                    "trace": trace,
                })
        except Exception as e:
            print(f"  Problem {i+1}/{len(problems)}: ERROR - {e}")

        gc.collect()
        torch.cuda.empty_cache()

    n_valid = len(valid_problems)
    print(f"\nBaseline: {n_valid}/{len(problems)} correct ({100*n_valid/len(problems):.0f}%)")
    print(f"Phase 1 time: {time.time() - t_start:.1f}s\n")

    if n_valid < 3:
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
            scores = teacher_force_with_attention(model, tokenizer, prob["prompt"], prob["trace"])
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

    if n_valid < 3:
        print("ERROR: Too few valid problems after attention. Aborting.")
        results = {"error": "too_few_valid_attention", "n_valid": n_valid}
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return

    # ── Phase 3: Destruction + Protection tests ──
    print("Phase 3: Running noise tests...")

    # Results storage
    destruction_results = {s: {} for s in ["selac", "seltc", "random"]}
    protection_results = {s: {} for s in ["ac", "h2o", "tc", "random"]}
    position_data = []  # For positional analysis

    for pi, prob in enumerate(valid_problems):
        scores = prob["scores"]
        n_pos = scores["reasoning_len"]
        prompt_len = scores["prompt_len"]

        # Get last token ID for generation
        full_text = prob["prompt"] + prob["trace"]
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
        last_token_id = full_ids[-1].item()

        # Build base KV cache
        try:
            base_kv, seq_len = build_kv_cache(model, tokenizer, prob["prompt"], prob["trace"])
        except Exception as e:
            print(f"  Problem {pi+1}: KV cache error: {e}")
            continue

        print(f"  Problem {pi+1}/{n_valid} (n_pos={n_pos})...")

        # Record position data
        relative_positions = np.arange(n_pos) / max(n_pos - 1, 1)
        position_data.append({
            "n_pos": n_pos,
            "ac": scores["ac"].tolist(),
            "tc": scores["tc"].tolist(),
            "h2o": scores["h2o"].tolist(),
            "sel": scores["sel"].tolist(),
            "positions": relative_positions.tolist(),
        })

        # ── Destruction tests ──
        for frac in DESTRUCTION_FRACTIONS:
            n_noise = max(1, int(n_pos * frac))
            frac_key = f"{frac}"

            for strategy_name, score_key in [("selac", "sel"), ("seltc", "sel"), ("random", None)]:
                if frac_key not in destruction_results[strategy_name]:
                    destruction_results[strategy_name][frac_key] = {"correct": 0, "total": 0, "positions": []}

                if strategy_name == "random":
                    positions = np.random.choice(n_pos, n_noise, replace=False)
                elif strategy_name == "selac":
                    # Destroy most AC-selective positions (highest selectivity)
                    positions = np.argsort(scores["sel"])[-n_noise:]
                elif strategy_name == "seltc":
                    # Destroy most TC-selective positions (lowest selectivity)
                    positions = np.argsort(scores["sel"])[:n_noise]

                # Record mean relative position of noised set
                mean_pos = np.mean(relative_positions[positions])
                destruction_results[strategy_name][frac_key]["positions"].append(mean_pos)

                # Apply noise and generate
                kv_copy = clone_kv_cache(base_kv)
                apply_noise_to_positions(kv_copy, positions, prompt_len, num_layers)
                try:
                    gen_text = run_noised_generation(model, tokenizer, kv_copy, seq_len, last_token_id)
                    pred_ans = normalize_answer(extract_answer(gen_text))
                    is_correct = (pred_ans == prob["true_answer"])
                    destruction_results[strategy_name][frac_key]["correct"] += int(is_correct)
                except Exception as e:
                    print(f"    destroy {strategy_name}@{frac}: ERROR {e}")
                destruction_results[strategy_name][frac_key]["total"] += 1

                del kv_copy
                gc.collect()
                torch.cuda.empty_cache()

        # ── Protection tests ──
        for frac in NOISE_FRACTIONS:
            n_noise = max(1, int(n_pos * frac))
            frac_key = f"{frac}"

            for strategy_name in ["ac", "h2o", "tc", "random"]:
                if frac_key not in protection_results[strategy_name]:
                    protection_results[strategy_name][frac_key] = {"correct": 0, "total": 0, "positions": []}

                if strategy_name == "random":
                    positions = np.random.choice(n_pos, n_noise, replace=False)
                else:
                    # Protection: noise positions with LOWEST score (protect highest)
                    positions = np.argsort(scores[strategy_name])[:n_noise]

                mean_pos = np.mean(relative_positions[positions])
                protection_results[strategy_name][frac_key]["positions"].append(mean_pos)

                kv_copy = clone_kv_cache(base_kv)
                apply_noise_to_positions(kv_copy, positions, prompt_len, num_layers)
                try:
                    gen_text = run_noised_generation(model, tokenizer, kv_copy, seq_len, last_token_id)
                    pred_ans = normalize_answer(extract_answer(gen_text))
                    is_correct = (pred_ans == prob["true_answer"])
                    protection_results[strategy_name][frac_key]["correct"] += int(is_correct)
                except Exception as e:
                    print(f"    protect {strategy_name}@{frac}: ERROR {e}")
                protection_results[strategy_name][frac_key]["total"] += 1

                del kv_copy
                gc.collect()
                torch.cuda.empty_cache()

        del base_kv
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        print(f"    done ({elapsed:.0f}s elapsed)")

        # Time check
        if elapsed > 1500:
            print(f"\nWARNING: approaching time limit ({elapsed:.0f}s). Stopping after {pi+1} problems.")
            break

    # ── Compute final accuracies ──
    print(f"\nPhase 3 complete. Time: {time.time() - t_start:.1f}s\n")

    # Destruction results
    print("=" * 70)
    print("DESTRUCTION TEST RESULTS (noise TOP positions)")
    print("=" * 70)
    for strategy in ["selac", "seltc", "random"]:
        print(f"\n  {strategy.upper()}:")
        for frac_key in sorted(destruction_results[strategy].keys()):
            d = destruction_results[strategy][frac_key]
            if d["total"] > 0:
                acc = 100 * d["correct"] / d["total"]
                mean_pos = np.mean(d["positions"]) if d["positions"] else 0
                d["accuracy"] = acc
                d["mean_position"] = mean_pos
                print(f"    {frac_key}: {d['correct']}/{d['total']} = {acc:.1f}% (mean pos: {mean_pos:.2f})")

    # Protection results
    print(f"\n{'='*70}")
    print("PROTECTION TEST RESULTS (noise BOTTOM positions)")
    print("=" * 70)
    for strategy in ["ac", "h2o", "tc", "random"]:
        print(f"\n  {strategy.upper()}:")
        for frac_key in sorted(protection_results[strategy].keys()):
            d = protection_results[strategy][frac_key]
            if d["total"] > 0:
                acc = 100 * d["correct"] / d["total"]
                mean_pos = np.mean(d["positions"]) if d["positions"] else 0
                d["accuracy"] = acc
                d["mean_position"] = mean_pos
                print(f"    {frac_key}: {d['correct']}/{d['total']} = {acc:.1f}% (mean pos: {mean_pos:.2f})")

    # ── Positional analysis ──
    print(f"\n{'='*70}")
    print("POSITIONAL ANALYSIS")
    print("=" * 70)

    positional_analysis = {}
    if position_data:
        all_ac = np.concatenate([np.array(p["ac"]) for p in position_data])
        all_tc = np.concatenate([np.array(p["tc"]) for p in position_data])
        all_h2o = np.concatenate([np.array(p["h2o"]) for p in position_data])
        all_sel = np.concatenate([np.array(p["sel"]) for p in position_data])
        all_pos = np.concatenate([np.array(p["positions"]) for p in position_data])

        for name, vals in [("ac", all_ac), ("tc", all_tc), ("h2o", all_h2o), ("sel", all_sel)]:
            rho, pval = scipy_stats.spearmanr(all_pos, vals)
            positional_analysis[f"position_{name}_rho"] = rho
            positional_analysis[f"position_{name}_pval"] = pval
            print(f"  Position vs {name.upper()}: rho={rho:.3f} (p={pval:.2e})")

        # Mean noise positions for each strategy
        for strategy in ["selac", "seltc", "random"]:
            for frac_key in destruction_results[strategy]:
                d = destruction_results[strategy][frac_key]
                if d.get("positions"):
                    key = f"mean_position_{strategy}_destroy"
                    positional_analysis[key] = np.mean(d["positions"])
        for strategy in ["ac", "h2o", "tc", "random"]:
            for frac_key in protection_results[strategy]:
                d = protection_results[strategy][frac_key]
                if d.get("positions"):
                    key = f"mean_position_{strategy}_protect"
                    positional_analysis[key] = np.mean(d["positions"])

        # Print mean positions at the middle noise fraction
        mid_frac = "0.03"
        print(f"\n  Mean noise position at {mid_frac} noise:")
        for s in ["selac", "seltc", "random"]:
            d = destruction_results[s].get(mid_frac, {})
            if d.get("positions"):
                print(f"    {s}_destroy: {np.mean(d['positions']):.3f}")
        for s in ["ac", "h2o", "tc", "random"]:
            d = protection_results[s].get(mid_frac, {})
            if d.get("positions"):
                print(f"    {s}_protect: {np.mean(d['positions']):.3f}")

    positional_analysis["n_positions_total"] = sum(p["n_pos"] for p in position_data)

    # ── Dissociation metric ──
    print(f"\n{'='*70}")
    print("DISSOCIATION ANALYSIS")
    print("=" * 70)

    dissociation = {}
    for frac_key in sorted(set(
        list(destruction_results["selac"].keys()) +
        list(protection_results["ac"].keys())
    )):
        # Destruction dissociation: SelTC accuracy - SelAC accuracy
        selac_d = destruction_results["selac"].get(frac_key, {})
        seltc_d = destruction_results["seltc"].get(frac_key, {})
        if selac_d.get("accuracy") is not None and seltc_d.get("accuracy") is not None:
            dest_gap = seltc_d["accuracy"] - selac_d["accuracy"]
            dissociation[f"destruction_gap_{frac_key}"] = dest_gap
            print(f"  Destruction @{frac_key}: SelTC({seltc_d['accuracy']:.1f}%) - SelAC({selac_d['accuracy']:.1f}%) = {dest_gap:+.1f}pp")

        # Protection: AC vs H2O
        ac_p = protection_results["ac"].get(frac_key, {})
        h2o_p = protection_results["h2o"].get(frac_key, {})
        if ac_p.get("accuracy") is not None and h2o_p.get("accuracy") is not None:
            prot_gap = ac_p["accuracy"] - h2o_p["accuracy"]
            dissociation[f"protection_ac_vs_h2o_{frac_key}"] = prot_gap
            print(f"  Protection @{frac_key}: AC({ac_p['accuracy']:.1f}%) - H2O({h2o_p['accuracy']:.1f}%) = {prot_gap:+.1f}pp")

    # ── Save results ──
    # Clean positions lists (not serializable as-is)
    for s in destruction_results:
        for fk in destruction_results[s]:
            destruction_results[s][fk]["positions"] = [float(x) for x in destruction_results[s][fk].get("positions", [])]
    for s in protection_results:
        for fk in protection_results[s]:
            protection_results[s][fk]["positions"] = [float(x) for x in protection_results[s][fk].get("positions", [])]

    results = {
        "model": MODEL_NAME,
        "n_total": len(problems),
        "n_valid": n_valid,
        "baseline_accuracy": 100 * n_valid / len(problems),
        "destruction": destruction_results,
        "protection": protection_results,
        "positional_analysis": positional_analysis,
        "dissociation": dissociation,
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

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"{'='*70}")
    print("EXPERIMENT 014 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
