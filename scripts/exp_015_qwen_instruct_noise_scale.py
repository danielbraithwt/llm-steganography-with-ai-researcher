#!/usr/bin/env python3
"""
Experiment 015: Noise Scale Sweep on Qwen3-4B-Instruct

Exp_014 showed Qwen3-4B-Instruct at 0% accuracy under noise, but used a
BROKEN pipeline (generated from AFTER the complete answer). This experiment
fixes the pipeline using the correct approach from exp_004/005:
  1. Truncate trace at "####" (reasoning only, no answer)
  2. Teacher-force prompt + reasoning_text
  3. Apply scaled additive noise to KV cache
  4. Lookback re-computation (last 20 tokens through ablated cache)
  5. Generate answer from end of reasoning

Key question: Does spatial structure (SelAC vs SelTC dissociation) emerge
at lower noise scales? Sweeps noise scale from 0.01x to 1.0x norm.
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
NUM_PROBLEMS = 35
NOISE_SCALES = [0.01, 0.03, 0.1, 0.3, 1.0]
NOISE_FRACTIONS = [0.01, 0.03, 0.05]
ATTENTION_LAYERS_AC = [-1, -2, -3, -4]
MAX_GEN_TOKENS = 768
MAX_SEQ_LEN = 2048
LOOKBACK = 20
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp_015")

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
    """Teacher-force prompt + reasoning_text and compute per-position attention scores.
    NOTE: reasoning_text should NOT include the #### answer line."""
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
    """Build the base KV cache by teacher-forcing prompt + reasoning_text.
    Returns (kv_cache, input_ids, seq_len, prompt_len, lookback_start)."""
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
    """
    Clone the base KV cache, apply scaled additive noise, lookback
    re-computation, then generate answer.
    """
    # Clone cache to avoid modifying the base
    past_kv = clone_kv_cache(base_kv)

    # Apply additive noise to specified positions
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

    # Lookback re-computation
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
            if re.search(r'\d+\s*\n', after) or (re.search(r'\d+\s*$', after) and len(after.strip()) >= 2):
                break
        if "\nQ:" in decoded:
            break

    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    del lookback_outputs, past_kv, gen_kv, trunc_kv
    gc.collect()
    torch.cuda.empty_cache()

    return answer_text


@torch.no_grad()
def ablated_answer_generation(model, tokenizer, prompt, reasoning_text,
                               positions_to_ablate, prompt_len, num_layers,
                               noise_scale=1.0):
    """Convenience wrapper: builds cache and runs noised generation."""
    base_kv, input_ids, seq_len, prompt_len_actual, lookback_start = \
        build_base_kv(model, tokenizer, prompt, reasoning_text)
    result = noised_answer_generation(
        model, tokenizer, base_kv, input_ids, seq_len,
        prompt_len_actual, lookback_start, num_layers,
        positions_to_ablate, noise_scale
    )
    del base_kv
    gc.collect()
    torch.cuda.empty_cache()
    return result


def make_figures(results, results_dir):
    """Generate analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise_scales = results.get("noise_scales", NOISE_SCALES)
    n_valid = results.get("n_valid", "?")

    # ── Figure 1: Accuracy vs Noise Scale (destruction, per fraction) ──
    fig, axes = plt.subplots(1, len(NOISE_FRACTIONS), figsize=(6 * len(NOISE_FRACTIONS), 5), sharey=True)
    if len(NOISE_FRACTIONS) == 1:
        axes = [axes]

    destruction = results.get("destruction", {})
    for fi, frac in enumerate(NOISE_FRACTIONS):
        frac_key = str(frac)
        ax = axes[fi]
        for strategy, color, label in [
            ("selac", "red", "SelAC destroy"),
            ("seltc", "green", "SelTC destroy"),
            ("random", "gray", "Random"),
        ]:
            accs = []
            scales_plot = []
            for scale in noise_scales:
                scale_key = str(scale)
                entry = destruction.get(strategy, {}).get(f"{frac_key}_{scale_key}", {})
                if "accuracy" in entry:
                    accs.append(entry["accuracy"])
                    scales_plot.append(scale)
            if accs:
                ax.plot(scales_plot, accs, 'o-', color=color, label=label, linewidth=2)

        ax.set_xlabel("Noise Scale (x position norm)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Noise Fraction = {frac*100:.0f}%")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    fig.suptitle(f"Qwen3-4B-Instruct: Destruction Test — Noise Scale Sweep (n={n_valid})")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "destruction_scale_sweep.png"), dpi=150)
    plt.close(fig)

    # ── Figure 2: Accuracy vs Noise Scale (protection, per fraction) ──
    fig, axes = plt.subplots(1, len(NOISE_FRACTIONS), figsize=(6 * len(NOISE_FRACTIONS), 5), sharey=True)
    if len(NOISE_FRACTIONS) == 1:
        axes = [axes]

    protection = results.get("protection", {})
    prot_colors = {"ac": "red", "h2o": "blue", "tc": "green", "random": "gray"}
    for fi, frac in enumerate(NOISE_FRACTIONS):
        frac_key = str(frac)
        ax = axes[fi]
        for strategy in ["ac", "h2o", "tc", "random"]:
            accs = []
            scales_plot = []
            for scale in noise_scales:
                scale_key = str(scale)
                entry = protection.get(strategy, {}).get(f"{frac_key}_{scale_key}", {})
                if "accuracy" in entry:
                    accs.append(entry["accuracy"])
                    scales_plot.append(scale)
            if accs:
                ax.plot(scales_plot, accs, 'o-', color=prot_colors.get(strategy, "black"),
                        label=f"{strategy.upper()} protect", linewidth=2)

        ax.set_xlabel("Noise Scale (x position norm)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Noise Fraction = {frac*100:.0f}%")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    fig.suptitle(f"Qwen3-4B-Instruct: Protection Test — Noise Scale Sweep (n={n_valid})")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "protection_scale_sweep.png"), dpi=150)
    plt.close(fig)

    # ── Figure 3: Dissociation gap vs noise scale ──
    fig, ax = plt.subplots(figsize=(8, 5))
    dissociation = results.get("dissociation", {})
    for fi, frac in enumerate(NOISE_FRACTIONS):
        frac_key = str(frac)
        gaps = []
        scales_plot = []
        for scale in noise_scales:
            scale_key = str(scale)
            gap_key = f"destruction_gap_{frac_key}_{scale_key}"
            if gap_key in dissociation:
                gaps.append(dissociation[gap_key])
                scales_plot.append(scale)
        if gaps:
            ax.plot(scales_plot, gaps, 'o-', linewidth=2, label=f"NF={frac*100:.0f}%")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axhline(23.5, color="red", linewidth=1, linestyle=":", label="Qwen-Base gap (exp_004)")
    ax.axhline(23.8, color="blue", linewidth=1, linestyle=":", label="Llama gap (exp_005)")
    ax.set_xlabel("Noise Scale (x position norm)")
    ax.set_ylabel("SelTC - SelAC Accuracy Gap (pp)")
    ax.set_title(f"Dissociation Gap vs Noise Scale\n(positive = spatial structure present)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "dissociation_vs_scale.png"), dpi=150)
    plt.close(fig)

    # ── Figure 4: Heatmap — accuracy by strategy × scale (at 3% fraction) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Destruction heatmap
    dest_strategies = ["selac", "seltc", "random"]
    frac_key = "0.03"
    matrix = []
    for strategy in dest_strategies:
        row = []
        for scale in noise_scales:
            entry = destruction.get(strategy, {}).get(f"{frac_key}_{scale}", {})
            row.append(entry.get("accuracy", np.nan))
        matrix.append(row)
    matrix = np.array(matrix)

    im = axes[0].imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    axes[0].set_xticks(range(len(noise_scales)))
    axes[0].set_xticklabels([f"{s}x" for s in noise_scales], fontsize=9)
    axes[0].set_yticks(range(len(dest_strategies)))
    axes[0].set_yticklabels(["SelAC", "SelTC", "Random"])
    axes[0].set_xlabel("Noise Scale")
    axes[0].set_title("Destruction at 3% Noise Fraction")
    for i in range(len(dest_strategies)):
        for j in range(len(noise_scales)):
            v = matrix[i, j]
            if not np.isnan(v):
                axes[0].text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                           color="white" if v < 40 else "black")
    plt.colorbar(im, ax=axes[0], label="Accuracy (%)")

    # Protection heatmap
    prot_strategies = ["ac", "h2o", "tc", "random"]
    matrix_p = []
    for strategy in prot_strategies:
        row = []
        for scale in noise_scales:
            entry = protection.get(strategy, {}).get(f"{frac_key}_{scale}", {})
            row.append(entry.get("accuracy", np.nan))
        matrix_p.append(row)
    matrix_p = np.array(matrix_p)

    im2 = axes[1].imshow(matrix_p, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    axes[1].set_xticks(range(len(noise_scales)))
    axes[1].set_xticklabels([f"{s}x" for s in noise_scales], fontsize=9)
    axes[1].set_yticks(range(len(prot_strategies)))
    axes[1].set_yticklabels(["AC", "H2O", "TC", "Random"])
    axes[1].set_xlabel("Noise Scale")
    axes[1].set_title("Protection at 3% Noise Fraction")
    for i in range(len(prot_strategies)):
        for j in range(len(noise_scales)):
            v = matrix_p[i, j]
            if not np.isnan(v):
                axes[1].text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                           color="white" if v < 40 else "black")
    plt.colorbar(im2, ax=axes[1], label="Accuracy (%)")

    fig.suptitle(f"Qwen3-4B-Instruct: Strategy x Scale Heatmap (n={n_valid})")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "strategy_scale_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"Figures saved to {results_dir}/")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print(f"{'='*70}")
    print(f"Experiment 015: Qwen3-4B-Instruct Noise Scale Sweep")
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {NUM_PROBLEMS}")
    print(f"Noise scales: {NOISE_SCALES}")
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
    start_idx = 30
    problems = list(dataset.select(range(start_idx, start_idx + NUM_PROBLEMS)))
    print(f"Using {len(problems)} problems (indices {start_idx}-{start_idx + NUM_PROBLEMS - 1})\n")

    # ── Phase 1: Generate baseline traces and extract reasoning ──
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
                # Truncate trace at "####" — keep only reasoning
                if "####" in trace:
                    reasoning_text = trace[:trace.index("####")]
                else:
                    reasoning_text = trace

                valid_problems.append({
                    "idx": start_idx + i,
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

    if n_valid < 3:
        print("ERROR: Too few valid problems after attention. Aborting.")
        results = {"error": "too_few_valid_attention", "n_valid": n_valid}
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return

    # ── Phase 2.5: Control test — verify clean generation works ──
    print("Phase 2.5: Control test (clean KV cache → answer generation)...")
    control_pass = 0
    for pi, prob in enumerate(valid_problems[:3]):  # test first 3
        scores = prob["scores"]
        gen_text = ablated_answer_generation(
            model, tokenizer, prob["prompt"], prob["reasoning_text"],
            positions_to_ablate=[], prompt_len=scores["prompt_len"],
            num_layers=num_layers, noise_scale=0
        )
        gen_ans = normalize_answer(extract_answer(gen_text))
        ok = (gen_ans == prob["true_answer"])
        control_pass += int(ok)
        print(f"  Control {pi+1}: pred={gen_ans} true={prob['true_answer']} {'OK' if ok else 'FAIL'}")
        if not ok:
            print(f"    Generated: {gen_text[:200]}")
    print(f"  Control: {control_pass}/3 correct")
    if control_pass == 0:
        print("ERROR: Control test failed — pipeline issue. Aborting.")
        results = {"error": "control_failed", "n_valid": n_valid}
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return
    print()

    # ── Phase 3: Noise scale sweep ──
    print("Phase 3: Running noise scale sweep...")
    print(f"  Scales: {NOISE_SCALES}")
    print(f"  Fractions: {NOISE_FRACTIONS}")

    destruction_results = {s: {} for s in ["selac", "seltc", "random"]}
    protection_results = {s: {} for s in ["ac", "h2o", "tc", "random"]}

    for pi, prob in enumerate(valid_problems):
        scores = prob["scores"]
        n_pos = scores["reasoning_len"]
        prompt_len = scores["prompt_len"]

        print(f"  Problem {pi+1}/{n_valid} (n_pos={n_pos})...", end=" ")

        # Build base KV cache ONCE per problem
        try:
            base_kv, input_ids, seq_len, prompt_len_actual, lookback_start = \
                build_base_kv(model, tokenizer, prob["prompt"], prob["reasoning_text"])
        except Exception as e:
            print(f"KV build error: {e}")
            continue

        for scale in NOISE_SCALES:
            for frac in NOISE_FRACTIONS:
                n_noise = max(1, int(n_pos * frac))
                condition_key = f"{frac}_{scale}"

                # ── Destruction tests ──
                for strategy_name in ["selac", "seltc", "random"]:
                    if condition_key not in destruction_results[strategy_name]:
                        destruction_results[strategy_name][condition_key] = {"correct": 0, "total": 0}

                    if strategy_name == "random":
                        positions = np.random.choice(n_pos, n_noise, replace=False).tolist()
                    elif strategy_name == "selac":
                        positions = np.argsort(scores["sel"])[-n_noise:].tolist()
                    elif strategy_name == "seltc":
                        positions = np.argsort(scores["sel"])[:n_noise].tolist()

                    try:
                        gen_text = noised_answer_generation(
                            model, tokenizer, base_kv, input_ids, seq_len,
                            prompt_len_actual, lookback_start, num_layers,
                            positions, noise_scale=scale
                        )
                        pred_ans = normalize_answer(extract_answer(gen_text))
                        is_correct = (pred_ans == prob["true_answer"])
                        destruction_results[strategy_name][condition_key]["correct"] += int(is_correct)
                    except Exception as e:
                        pass
                    destruction_results[strategy_name][condition_key]["total"] += 1

                # ── Protection tests ──
                for strategy_name in ["ac", "h2o", "tc", "random"]:
                    if condition_key not in protection_results[strategy_name]:
                        protection_results[strategy_name][condition_key] = {"correct": 0, "total": 0}

                    if strategy_name == "random":
                        positions = np.random.choice(n_pos, n_noise, replace=False).tolist()
                    else:
                        positions = np.argsort(scores[strategy_name])[:n_noise].tolist()

                    try:
                        gen_text = noised_answer_generation(
                            model, tokenizer, base_kv, input_ids, seq_len,
                            prompt_len_actual, lookback_start, num_layers,
                            positions, noise_scale=scale
                        )
                        pred_ans = normalize_answer(extract_answer(gen_text))
                        is_correct = (pred_ans == prob["true_answer"])
                        protection_results[strategy_name][condition_key]["correct"] += int(is_correct)
                    except Exception as e:
                        pass
                    protection_results[strategy_name][condition_key]["total"] += 1

        del base_kv
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        print(f"done ({elapsed:.0f}s elapsed)")

        if elapsed > 1500:
            print(f"\nWARNING: approaching time limit ({elapsed:.0f}s). Stopping after {pi+1} problems.")
            break

    # ── Compute final accuracies ──
    print(f"\nPhase 3 complete. Time: {time.time() - t_start:.1f}s\n")

    print("=" * 70)
    print("DESTRUCTION TEST RESULTS (noise TOP positions)")
    print("=" * 70)
    for strategy in ["selac", "seltc", "random"]:
        print(f"\n  {strategy.upper()}:")
        for scale in NOISE_SCALES:
            for frac in NOISE_FRACTIONS:
                key = f"{frac}_{scale}"
                d = destruction_results[strategy].get(key, {})
                if d.get("total", 0) > 0:
                    acc = 100 * d["correct"] / d["total"]
                    d["accuracy"] = acc
                    print(f"    NF={frac*100:.0f}% scale={scale}x: {d['correct']}/{d['total']} = {acc:.1f}%")

    print(f"\n{'='*70}")
    print("PROTECTION TEST RESULTS (noise BOTTOM positions)")
    print("=" * 70)
    for strategy in ["ac", "h2o", "tc", "random"]:
        print(f"\n  {strategy.upper()}:")
        for scale in NOISE_SCALES:
            for frac in NOISE_FRACTIONS:
                key = f"{frac}_{scale}"
                d = protection_results[strategy].get(key, {})
                if d.get("total", 0) > 0:
                    acc = 100 * d["correct"] / d["total"]
                    d["accuracy"] = acc
                    print(f"    NF={frac*100:.0f}% scale={scale}x: {d['correct']}/{d['total']} = {acc:.1f}%")

    # ── Dissociation analysis ──
    print(f"\n{'='*70}")
    print("DISSOCIATION ANALYSIS (SelTC - SelAC gap)")
    print("=" * 70)

    dissociation = {}
    for scale in NOISE_SCALES:
        for frac in NOISE_FRACTIONS:
            key = f"{frac}_{scale}"
            selac_d = destruction_results["selac"].get(key, {})
            seltc_d = destruction_results["seltc"].get(key, {})
            if selac_d.get("accuracy") is not None and seltc_d.get("accuracy") is not None:
                gap = seltc_d["accuracy"] - selac_d["accuracy"]
                dissociation[f"destruction_gap_{key}"] = gap
                print(f"  NF={frac*100:.0f}% scale={scale}x: SelTC={seltc_d['accuracy']:.1f}% - SelAC={selac_d['accuracy']:.1f}% = {gap:+.1f}pp")

    # ── Key summary ──
    print(f"\n{'='*70}")
    print("SCALE SENSITIVITY SUMMARY (at 3% noise fraction)")
    print("=" * 70)

    for scale in NOISE_SCALES:
        key = f"0.03_{scale}"
        accs = {}
        for s in ["selac", "seltc", "random"]:
            d = destruction_results[s].get(key, {})
            accs[s] = d.get("accuracy", None)
        for s in ["ac", "h2o", "tc"]:
            d = protection_results[s].get(key, {})
            accs[s + "_p"] = d.get("accuracy", None)

        parts = []
        for label, k in [("SelAC", "selac"), ("SelTC", "seltc"), ("Rand", "random"),
                         ("AC_p", "ac_p"), ("H2O_p", "h2o_p"), ("TC_p", "tc_p")]:
            v = accs.get(k)
            if v is not None:
                parts.append(f"{label}={v:.0f}%")
        gap = dissociation.get(f"destruction_gap_0.03_{scale}", None)
        gap_str = f" | gap={gap:+.1f}pp" if gap is not None else ""
        print(f"  scale={scale}x: {', '.join(parts)}{gap_str}")

    # ── Save results ──
    results = {
        "model": MODEL_NAME,
        "n_total": len(problems),
        "n_valid": n_valid,
        "baseline_accuracy": 100 * n_valid / len(problems),
        "noise_scales": NOISE_SCALES,
        "noise_fractions": NOISE_FRACTIONS,
        "pipeline": "truncate_at_####_lookback_20_additive_noise",
        "destruction": destruction_results,
        "protection": protection_results,
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
        import traceback
        traceback.print_exc()

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"{'='*70}")
    print("EXPERIMENT 015 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
