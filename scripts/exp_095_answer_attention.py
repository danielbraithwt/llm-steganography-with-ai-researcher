#!/usr/bin/env python3
"""
Experiment 095: Answer-Step Attention Routing Analysis

Core question: When the model generates the final answer, does its attention
pattern differ from intermediate reasoning steps? Does the answer head (H5)
show distinctive retrieval patterns?

This is MECHANISTIC evidence for the hidden channel: probing (exps 083-094)
showed information IS present; this experiment tests whether the model's
attention mechanism RETRIEVES that information at the answer step.

Method:
1. Generate 8-shot CoT on GSM8K (200 problems) with Qwen3-4B-Base
2. Re-encode correct problems with output_attentions=True (eager attention)
3. Extract attention at the answer position and at mid-chain control positions
4. Bin attention by relative chain position (20 bins) + prompt fraction
5. Compare answer-step vs control-step attention distributions
6. Analyze answer head H5 vs other heads
"""

import os
import json
import time
import gc
import re
import sys
import random
import warnings

import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

T0 = time.time()
TIME_BUDGET = 6600  # 110 min
MAX_GEN = 512
MODEL_NAME = 'Qwen/Qwen3-4B-Base'
N_PROBLEMS = 200
N_BINS = 20
N_CONTROL_STEPS = 5
# 9 layers spanning full depth: ramp (0,5,9), transition (14), plateau (18,23,27,31,35)
LAYERS_PROBE = [0, 5, 9, 14, 18, 23, 27, 31, 35]

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_095"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plain-text 8-shot exemplars (same as exp_091) ──
EXEMPLARS = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
     "a": "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\nShe makes 9 * 2 = $18 every day at the farmer's market.\n#### 18"},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "a": "It takes 2/2 = 1 bolt of white fiber.\nSo the total bolts needed is 2 + 1 = 3.\n#### 3"},
    {"q": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
     "a": "The cost of the house and repairs came out to 80,000 + 50,000 = $130,000.\nHe increased the value of the house by 80,000 * 1.5 = $120,000.\nSo the new value of the house is 120,000 + 80,000 = $200,000.\nSo he made a profit of 200,000 - 130,000 = $70,000.\n#### 70000"},
    {"q": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "a": "He writes each friend 3 * 2 = 6 pages a week.\nSo he writes 6 * 2 = 12 pages every week.\nThat means he writes 12 * 52 = 624 pages a year.\n#### 624"},
    {"q": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
     "a": "If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3 * 20 = 60 cups of feed per day.\nIf she feeds the flock 15 cups in the morning, and 25 cups in the afternoon, then the carry-over to the final meal would be 60 - 15 - 25 = 20 cups.\n#### 20"},
    {"q": "Kylar went to the store to get water and some apples. The store sold apples for $1 each and water for $3 per bottle. Kylar wanted to buy one bag of apples and 2 bottles of water. How much would Kylar spend if each bag has 6 apples?",
     "a": "A bag has 6 apples and each apple costs $1, so a bag costs 6 * 1 = $6.\nKylar wants 2 bottles of water so that would cost 2 * 3 = $6.\nAltogether, Kylar would spend 6 + 6 = $12.\n#### 12"},
    {"q": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
     "a": "If Seattle has 20 sheep, Charleston has 4 * 20 = 80 sheep.\nToulouse has 2 * 80 = 160 sheep.\nTogether, they have 20 + 80 + 160 = 260 sheep.\n#### 260"},
    {"q": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
     "a": "First find how long it takes to download 40% of the file: 200 * 0.4 / 2 = 40 minutes.\nThen find how long it takes to download the whole file once the restart is complete: 200 / 2 = 100 minutes.\nThen add the time to download 40% of the file, the restart time, and the time to download the whole file: 40 + 20 + 100 = 160 minutes.\n#### 160"},
]


def build_prompt(question):
    prompt = ""
    for ex in EXEMPLARS:
        prompt += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '')
    m = re.search(r'answer\s+is\s+\$?(-?[\d,]+(?:\.\d+)?)', text, re.I)
    if m:
        return m.group(1).replace(',', '')
    return None


def extract_gold(answer_text):
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', answer_text)
    if m:
        return m.group(1).replace(',', '')
    return None


def load_gsm8k():
    from datasets import load_dataset
    return load_dataset("openai/gsm8k", "main", split="test")


def find_answer_position(full_ids, prompt_len, tokenizer):
    """Find position of the first answer digit token after '####' in the generated text."""
    gen_ids = full_ids[prompt_len:]
    text_so_far = ""
    hash_found = False
    for i, tid in enumerate(gen_ids):
        tok = tokenizer.decode([tid], skip_special_tokens=False)
        text_so_far += tok
        if not hash_found:
            if "####" in text_so_far:
                hash_found = True
            continue
        if hash_found:
            stripped = tok.strip()
            if stripped and (stripped[0].isdigit() or stripped[0] == '-'):
                return prompt_len + i  # absolute position in full sequence
    return None


def compute_entropy(attn_weights):
    """Compute entropy (bits) of attention distribution."""
    p = np.clip(attn_weights, 1e-12, 1.0)
    p = p / p.sum()  # renormalize
    return float(-np.sum(p * np.log2(p)))


def bin_chain_attention(attn_row, prompt_len, chain_end, n_bins=20):
    """
    Bin attention weights into: [n_bins chain bins, prompt_frac].

    attn_row: 1D array, length = number of positions this token can attend to
    prompt_len: number of prompt tokens
    chain_end: end of the visible chain (exclusive)

    Returns: array of length n_bins + 1
      [0..n_bins-1] = fraction of attention to each chain bin
      [n_bins] = fraction of attention to prompt
    """
    total = float(attn_row.sum())
    if total < 1e-10:
        return np.zeros(n_bins + 1, dtype=np.float32)

    prompt_frac = float(attn_row[:prompt_len].sum()) / total

    chain_len = chain_end - prompt_len
    if chain_len <= 0:
        result = np.zeros(n_bins + 1, dtype=np.float32)
        result[-1] = prompt_frac
        return result

    chain_attn = attn_row[prompt_len:chain_end].astype(np.float64)
    binned = np.zeros(n_bins, dtype=np.float32)
    bin_size = chain_len / n_bins
    for b in range(n_bins):
        b_start = int(b * bin_size)
        b_end = int((b + 1) * bin_size)
        if b == n_bins - 1:
            b_end = chain_len  # ensure last bin captures everything
        if b_end > b_start:
            binned[b] = float(chain_attn[b_start:b_end].sum()) / total

    result = np.zeros(n_bins + 1, dtype=np.float32)
    result[:n_bins] = binned
    result[-1] = prompt_frac
    return result


def main():
    t0 = time.time()

    print("=" * 70)
    print("Experiment 095: Answer-Step Attention Routing Analysis")
    print("Does the model attend differently when generating the answer?")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use eager attention to get attention weights
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager'
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_q_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, 'num_key_value_heads', n_q_heads)
    group_size = n_q_heads // n_kv_heads

    # Filter LAYERS_PROBE to valid range
    layers_probe = [l for l in LAYERS_PROBE if l < n_layers]

    print(f"\nModel: {MODEL_NAME}, {n_layers} layers")
    print(f"Q heads: {n_q_heads}, KV heads: {n_kv_heads}, group_size: {group_size}")
    print(f"H5 Q-head group: {5*group_size}-{(5+1)*group_size-1}")
    print(f"H0 Q-head group: {0}-{group_size-1}")
    print(f"Probing {len(layers_probe)} layers: {layers_probe}")

    ds = load_gsm8k()
    print(f"GSM8K: {len(ds)} problems")

    # ═══════════════════════════════════════════════════
    # PHASE 1: Generate CoT traces
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 1: Generating CoT traces")
    print(f"{'='*70}")

    generations = []
    for i in range(min(N_PROBLEMS, len(ds))):
        if time.time() - t0 > TIME_BUDGET * 0.20:
            print(f"  Time budget (20%) reached at problem {i}")
            break

        question = ds[i]['question']
        gold = extract_gold(ds[i]['answer'])
        prompt = build_prompt(question)

        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        prompt_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_GEN, do_sample=False,
                temperature=1.0, use_cache=True
            )

        gen_ids = output[0][prompt_len:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_answer(gen_text)
        correct = pred is not None and gold is not None and str(pred).strip() == str(gold).strip()

        generations.append({
            'idx': i, 'gold': gold, 'pred': pred, 'correct': correct,
            'gen_ids': gen_ids, 'prompt_len': prompt_len,
            'full_ids': output[0].tolist(), 'gen_text': gen_text,
        })

        if (i + 1) % 50 == 0:
            nc = sum(g['correct'] for g in generations)
            print(f"  {i+1} problems: {nc}/{i+1} correct ({100*nc/(i+1):.1f}%)")

    correct_gens = [g for g in generations if g['correct']]
    total_gen = len(generations)
    print(f"\nGenerated: {total_gen}, correct: {len(correct_gens)} ({100*len(correct_gens)/max(1,total_gen):.1f}%)")

    torch.cuda.empty_cache()
    gc.collect()

    # ═══════════════════════════════════════════════════
    # PHASE 2: Extract attention at answer + control positions
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 2: Extracting attention patterns")
    print(f"{'='*70}")

    # Storage: per problem, per position type (answer/control), per layer, per head
    # Stored as binned attention (n_bins + 1 values per head)
    all_answer_binned = []      # [problem][layer_idx][q_head] -> array(n_bins+1)
    all_answer_entropy = []     # [problem][layer_idx][q_head] -> float
    all_control_binned = []     # [problem][ctrl_idx][layer_idx][q_head] -> array(n_bins+1)
    all_control_entropy = []    # [problem][ctrl_idx][layer_idx][q_head] -> float
    all_prompt_fracs_answer = []  # [problem][layer_idx][q_head] -> float
    all_prompt_fracs_ctrl = []   # [problem][layer_idx][q_head] -> float
    metadata = []               # per-problem metadata

    problems_processed = 0

    for gi, g in enumerate(correct_gens):
        if time.time() - t0 > TIME_BUDGET * 0.75:
            print(f"  Time budget (75%) reached after {problems_processed} problems")
            break

        full_ids = g['full_ids']
        prompt_len = g['prompt_len']
        seq_len = len(full_ids)

        # Find answer position
        answer_pos = find_answer_position(full_ids, prompt_len, tokenizer)
        if answer_pos is None or answer_pos <= prompt_len + 20:
            continue

        chain_len = answer_pos - prompt_len

        # Control positions: random 20-80% of chain
        ctrl_start = prompt_len + int(0.2 * chain_len)
        ctrl_end = prompt_len + int(0.8 * chain_len)
        available = list(range(ctrl_start, ctrl_end))
        if len(available) < N_CONTROL_STEPS:
            continue
        ctrl_positions = sorted(random.sample(available, N_CONTROL_STEPS))

        # Forward pass with attention
        input_ids = torch.tensor([full_ids], device=model.device)
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True, use_cache=False)

        # ── Extract answer-step attention ──
        answer_target = answer_pos - 1  # position that predicts the first answer digit

        prob_answer_binned = {}
        prob_answer_entropy = {}
        prob_prompt_frac_answer = {}

        for layer_idx in layers_probe:
            attn = outputs.attentions[layer_idx][0]  # [n_q_heads, seq_len, seq_len]
            attn_row = attn[:, answer_target, :answer_target+1].cpu().float().numpy()
            # shape: [n_q_heads, answer_target+1]

            layer_binned = {}
            layer_entropy = {}
            layer_prompt_frac = {}
            for qh in range(n_q_heads):
                binned = bin_chain_attention(attn_row[qh], prompt_len, answer_pos, N_BINS)
                layer_binned[qh] = binned
                layer_entropy[qh] = compute_entropy(attn_row[qh])
                layer_prompt_frac[qh] = float(binned[-1])

            prob_answer_binned[layer_idx] = layer_binned
            prob_answer_entropy[layer_idx] = layer_entropy
            prob_prompt_frac_answer[layer_idx] = layer_prompt_frac

        all_answer_binned.append(prob_answer_binned)
        all_answer_entropy.append(prob_answer_entropy)
        all_prompt_fracs_answer.append(prob_prompt_frac_answer)

        # ── Extract control-step attention ──
        prob_ctrl_binned = []
        prob_ctrl_entropy = []
        prob_ctrl_prompt_frac = []

        for ctrl_pos in ctrl_positions:
            ctrl_b = {}
            ctrl_e = {}
            ctrl_pf = {}
            for layer_idx in layers_probe:
                attn = outputs.attentions[layer_idx][0]
                attn_row = attn[:, ctrl_pos, :ctrl_pos+1].cpu().float().numpy()

                layer_b = {}
                layer_e = {}
                layer_pf = {}
                for qh in range(n_q_heads):
                    binned = bin_chain_attention(attn_row[qh], prompt_len, ctrl_pos + 1, N_BINS)
                    layer_b[qh] = binned
                    layer_e[qh] = compute_entropy(attn_row[qh])
                    layer_pf[qh] = float(binned[-1])

                ctrl_b[layer_idx] = layer_b
                ctrl_e[layer_idx] = layer_e
                ctrl_pf[layer_idx] = layer_pf

            prob_ctrl_binned.append(ctrl_b)
            prob_ctrl_entropy.append(ctrl_e)
            prob_ctrl_prompt_frac.append(ctrl_pf)

        all_control_binned.append(prob_ctrl_binned)
        all_control_entropy.append(prob_ctrl_entropy)
        all_prompt_fracs_ctrl.append(prob_ctrl_prompt_frac)

        metadata.append({
            'idx': g['idx'], 'chain_len': chain_len,
            'prompt_len': prompt_len, 'answer_pos': answer_pos,
            'ctrl_positions': ctrl_positions,
        })

        problems_processed += 1

        del outputs
        torch.cuda.empty_cache()

        if problems_processed % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {problems_processed} problems, {elapsed:.0f}s elapsed")

    print(f"\nTotal: {problems_processed} problems with attention data")

    if problems_processed < 10:
        print("ERROR: Too few problems. Aborting.")
        sys.exit(1)

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ═══════════════════════════════════════════════════
    # PHASE 3: Analysis
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 3: Analysis")
    print(f"{'='*70}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_problems = problems_processed

    # ── 3A: Average attention distribution (answer vs control) ──
    print("\n--- 3A: Average attention distribution ---")

    # Aggregate: for each layer, average binned attention across all heads and problems
    for layer_idx in layers_probe:
        # Answer distribution: average across problems and heads
        ans_bins = []
        ctrl_bins = []
        for pi in range(n_problems):
            for qh in range(n_q_heads):
                ans_bins.append(all_answer_binned[pi][layer_idx][qh][:N_BINS])
            for ci in range(len(all_control_binned[pi])):
                for qh in range(n_q_heads):
                    ctrl_bins.append(all_control_binned[pi][ci][layer_idx][qh][:N_BINS])

        ans_mean = np.mean(ans_bins, axis=0)
        ctrl_mean = np.mean(ctrl_bins, axis=0)

        # Late vs mid vs early attention fraction
        early_frac_ans = float(ans_mean[:7].sum())
        mid_frac_ans = float(ans_mean[7:14].sum())
        late_frac_ans = float(ans_mean[14:].sum())
        early_frac_ctrl = float(ctrl_mean[:7].sum())
        mid_frac_ctrl = float(ctrl_mean[7:14].sum())
        late_frac_ctrl = float(ctrl_mean[14:].sum())

        print(f"  L{layer_idx:2d}: Answer [early={early_frac_ans:.3f} mid={mid_frac_ans:.3f} late={late_frac_ans:.3f}]"
              f"  Control [early={early_frac_ctrl:.3f} mid={mid_frac_ctrl:.3f} late={late_frac_ctrl:.3f}]")

    # ── 3B: Prompt fraction comparison ──
    print("\n--- 3B: Prompt attention fraction ---")
    for layer_idx in layers_probe:
        ans_pf = []
        ctrl_pf = []
        for pi in range(n_problems):
            for qh in range(n_q_heads):
                ans_pf.append(all_prompt_fracs_answer[pi][layer_idx][qh])
            for ci in range(len(all_prompt_fracs_ctrl[pi])):
                for qh in range(n_q_heads):
                    ctrl_pf.append(all_prompt_fracs_ctrl[pi][ci][layer_idx][qh])

        print(f"  L{layer_idx:2d}: Answer prompt_frac={np.mean(ans_pf):.4f}  Control prompt_frac={np.mean(ctrl_pf):.4f}")

    # ── 3C: Per-head entropy comparison ──
    print("\n--- 3C: Per-head entropy (answer vs control) ---")

    # Compute average entropy per head across problems
    head_entropy_answer = {layer_idx: np.zeros(n_q_heads) for layer_idx in layers_probe}
    head_entropy_control = {layer_idx: np.zeros(n_q_heads) for layer_idx in layers_probe}

    for layer_idx in layers_probe:
        for qh in range(n_q_heads):
            ans_e = [all_answer_entropy[pi][layer_idx][qh] for pi in range(n_problems)]
            head_entropy_answer[layer_idx][qh] = np.mean(ans_e)

            ctrl_e = []
            for pi in range(n_problems):
                for ci in range(len(all_control_entropy[pi])):
                    ctrl_e.append(all_control_entropy[pi][ci][layer_idx][qh])
            head_entropy_control[layer_idx][qh] = np.mean(ctrl_e)

    # Report H5 group vs others for key layers
    h5_q_start = 5 * group_size
    h5_q_end = (5 + 1) * group_size
    h0_q_start = 0
    h0_q_end = group_size

    for layer_idx in [9, 18, 27, 35]:
        if layer_idx not in head_entropy_answer:
            continue
        h5_ans = head_entropy_answer[layer_idx][h5_q_start:h5_q_end].mean()
        h5_ctrl = head_entropy_control[layer_idx][h5_q_start:h5_q_end].mean()
        others_ans = np.delete(head_entropy_answer[layer_idx], range(h5_q_start, h5_q_end)).mean()
        others_ctrl = np.delete(head_entropy_control[layer_idx], range(h5_q_start, h5_q_end)).mean()

        h0_ans = head_entropy_answer[layer_idx][h0_q_start:h0_q_end].mean()
        h0_ctrl = head_entropy_control[layer_idx][h0_q_start:h0_q_end].mean()

        print(f"  L{layer_idx:2d}: H5 ans={h5_ans:.2f} ctrl={h5_ctrl:.2f} delta={h5_ans-h5_ctrl:.2f}"
              f"  | H0 ans={h0_ans:.2f} ctrl={h0_ctrl:.2f} delta={h0_ans-h0_ctrl:.2f}"
              f"  | Others ans={others_ans:.2f} ctrl={others_ctrl:.2f} delta={others_ans-others_ctrl:.2f}")

    # ── 3D: KV head group attention profiles at key layer ──
    print("\n--- 3D: KV head group attention profiles (answer step) ---")
    key_layer = 18  # mid-plateau
    if key_layer in layers_probe[0:] or True:
        for kv_h in range(n_kv_heads):
            qh_start = kv_h * group_size
            qh_end = (kv_h + 1) * group_size
            bins = []
            for pi in range(n_problems):
                for qh in range(qh_start, qh_end):
                    bins.append(all_answer_binned[pi][key_layer][qh][:N_BINS])
            mean_profile = np.mean(bins, axis=0)
            late_frac = float(mean_profile[14:].sum())
            early_frac = float(mean_profile[:7].sum())
            marker = " <-- H5" if kv_h == 5 else (" <-- H0" if kv_h == 0 else "")
            print(f"  KV H{kv_h}: early={early_frac:.4f} late={late_frac:.4f}{marker}")

    # ── 3E: Answer-step attention difference (answer - control) per head group ──
    print("\n--- 3E: Answer vs Control difference per KV head ---")
    for kv_h in range(n_kv_heads):
        qh_start = kv_h * group_size
        qh_end = (kv_h + 1) * group_size

        ans_late_fracs = []
        ctrl_late_fracs = []
        for pi in range(n_problems):
            for qh in range(qh_start, qh_end):
                ans_late_fracs.append(float(all_answer_binned[pi][key_layer][qh][14:N_BINS].sum()))
            for ci in range(len(all_control_binned[pi])):
                for qh in range(qh_start, qh_end):
                    ctrl_late_fracs.append(float(all_control_binned[pi][ci][key_layer][qh][14:N_BINS].sum()))

        diff = np.mean(ans_late_fracs) - np.mean(ctrl_late_fracs)
        marker = " <-- H5" if kv_h == 5 else (" <-- H0" if kv_h == 0 else "")
        print(f"  KV H{kv_h}: late_frac diff (ans-ctrl) = {diff:+.4f}{marker}")

    # ═══════════════════════════════════════════════════
    # PHASE 4: Figures
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 4: Generating figures")
    print(f"{'='*70}")

    # ── Figure 1: Attention distribution (answer vs control) at 4 key layers ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Attention Distribution: Answer Step vs Control Steps\n(Qwen3-4B-Base, all heads averaged)', fontsize=14)

    key_layers_fig = [9, 18, 27, 35]
    bin_labels = [f"{5*b}-{5*(b+1)}%" for b in range(N_BINS)]
    x = np.arange(N_BINS)

    for ax_idx, layer_idx in enumerate(key_layers_fig):
        if layer_idx not in [l for l in layers_probe]:
            continue
        ax = axes[ax_idx // 2][ax_idx % 2]

        # Compute mean distributions
        ans_bins_list = []
        ctrl_bins_list = []
        for pi in range(n_problems):
            for qh in range(n_q_heads):
                ans_bins_list.append(all_answer_binned[pi][layer_idx][qh][:N_BINS])
            for ci in range(len(all_control_binned[pi])):
                for qh in range(n_q_heads):
                    ctrl_bins_list.append(all_control_binned[pi][ci][layer_idx][qh][:N_BINS])

        ans_mean = np.mean(ans_bins_list, axis=0)
        ctrl_mean = np.mean(ctrl_bins_list, axis=0)
        ans_sem = np.std(ans_bins_list, axis=0) / np.sqrt(len(ans_bins_list))
        ctrl_sem = np.std(ctrl_bins_list, axis=0) / np.sqrt(len(ctrl_bins_list))

        ax.bar(x - 0.15, ans_mean, 0.3, label='Answer step', color='#e74c3c', alpha=0.8)
        ax.bar(x + 0.15, ctrl_mean, 0.3, label='Control steps', color='#3498db', alpha=0.8)
        ax.set_title(f'Layer {layer_idx} ({100*layer_idx/(n_layers-1):.0f}% depth)', fontsize=12)
        ax.set_xlabel('Chain position (relative %)')
        ax.set_ylabel('Attention fraction')
        ax.set_xticks(x[::4])
        ax.set_xticklabels([bin_labels[i] for i in range(0, N_BINS, 4)], rotation=45, fontsize=8)
        if ax_idx == 0:
            ax.legend(fontsize=10)

    plt.tight_layout()
    fig1_path = str(RESULTS_DIR / "attention_distribution_answer_vs_control.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure 1 saved: {fig1_path}")

    # ── Figure 2: Entropy scatter (answer vs control) per head ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Attention Entropy: Answer Step vs Control Steps per Q-Head\n(each dot = one Q-head, averaged across problems)', fontsize=14)

    for ax_idx, layer_idx in enumerate(key_layers_fig):
        if layer_idx not in head_entropy_answer:
            continue
        ax = axes[ax_idx // 2][ax_idx % 2]

        ans_ent = head_entropy_answer[layer_idx]
        ctrl_ent = head_entropy_control[layer_idx]

        # Color by KV head group
        colors = []
        for qh in range(n_q_heads):
            kv_h = qh // group_size
            if kv_h == 5:
                colors.append('#e74c3c')  # red for H5
            elif kv_h == 0:
                colors.append('#2ecc71')  # green for H0
            else:
                colors.append('#3498db')  # blue for others

        ax.scatter(ctrl_ent, ans_ent, c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Diagonal line (equal entropy)
        min_e = min(ans_ent.min(), ctrl_ent.min())
        max_e = max(ans_ent.max(), ctrl_ent.max())
        ax.plot([min_e, max_e], [min_e, max_e], 'k--', alpha=0.3, label='Equal')

        ax.set_xlabel('Control step entropy (bits)')
        ax.set_ylabel('Answer step entropy (bits)')
        ax.set_title(f'Layer {layer_idx} ({100*layer_idx/(n_layers-1):.0f}% depth)')

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='H5 (answer head)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='H0'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Others'),
        ]
        if ax_idx == 0:
            ax.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    fig2_path = str(RESULTS_DIR / "entropy_answer_vs_control.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure 2 saved: {fig2_path}")

    # ── Figure 3: H5 attention profile at answer step across layers ──
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('H5 (Answer Head) Attention at Answer Step vs Control — All Probed Layers\n(Q-heads 20-23 averaged)', fontsize=14)

    for ax_idx, layer_idx in enumerate(layers_probe):
        ax = axes[ax_idx // 3][ax_idx % 3]

        # H5 answer distribution
        h5_ans = []
        h5_ctrl = []
        other_ans = []

        for pi in range(n_problems):
            for qh in range(h5_q_start, h5_q_end):
                h5_ans.append(all_answer_binned[pi][layer_idx][qh][:N_BINS])
            for ci in range(len(all_control_binned[pi])):
                for qh in range(h5_q_start, h5_q_end):
                    h5_ctrl.append(all_control_binned[pi][ci][layer_idx][qh][:N_BINS])
            # All other heads for comparison
            for qh in range(n_q_heads):
                if qh < h5_q_start or qh >= h5_q_end:
                    other_ans.append(all_answer_binned[pi][layer_idx][qh][:N_BINS])

        h5_ans_mean = np.mean(h5_ans, axis=0)
        h5_ctrl_mean = np.mean(h5_ctrl, axis=0)
        other_ans_mean = np.mean(other_ans, axis=0)

        ax.plot(x, h5_ans_mean, 'r-o', markersize=3, label='H5 answer', linewidth=2)
        ax.plot(x, h5_ctrl_mean, 'r--', alpha=0.5, label='H5 control', linewidth=1.5)
        ax.plot(x, other_ans_mean, 'b-', alpha=0.4, label='Others answer', linewidth=1.5)

        ax.set_title(f'Layer {layer_idx} ({100*layer_idx/(n_layers-1):.0f}%)', fontsize=11)
        ax.set_xlabel('Chain position (%)')
        ax.set_ylabel('Attention fraction')
        ax.set_xticks(x[::4])
        ax.set_xticklabels([f"{5*i}%" for i in range(0, N_BINS, 4)], fontsize=8)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig3_path = str(RESULTS_DIR / "h5_attention_profile_all_layers.png")
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure 3 saved: {fig3_path}")

    # ── Figure 4: KV head comparison heatmap at answer step ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Attention Profile at Answer Step — Per KV Head Group\n(Layer 18, mid-plateau)', fontsize=14)

    # Left: absolute attention at answer step
    kv_profiles_ans = np.zeros((n_kv_heads, N_BINS))
    kv_profiles_ctrl = np.zeros((n_kv_heads, N_BINS))

    for kv_h in range(n_kv_heads):
        qh_start = kv_h * group_size
        qh_end = (kv_h + 1) * group_size
        ans_b = []
        ctrl_b = []
        for pi in range(n_problems):
            for qh in range(qh_start, qh_end):
                ans_b.append(all_answer_binned[pi][key_layer][qh][:N_BINS])
            for ci in range(len(all_control_binned[pi])):
                for qh in range(qh_start, qh_end):
                    ctrl_b.append(all_control_binned[pi][ci][key_layer][qh][:N_BINS])
        kv_profiles_ans[kv_h] = np.mean(ans_b, axis=0)
        kv_profiles_ctrl[kv_h] = np.mean(ctrl_b, axis=0)

    im1 = axes[0].imshow(kv_profiles_ans, aspect='auto', cmap='Reds', interpolation='nearest')
    axes[0].set_title('Answer Step Attention')
    axes[0].set_xlabel('Chain position bin')
    axes[0].set_ylabel('KV Head')
    axes[0].set_yticks(range(n_kv_heads))
    axes[0].set_yticklabels([f'H{h}{"*" if h in [0,5] else ""}' for h in range(n_kv_heads)])
    plt.colorbar(im1, ax=axes[0], label='Attention fraction')

    # Right: difference (answer - control)
    diff_profiles = kv_profiles_ans - kv_profiles_ctrl
    vmax = max(abs(diff_profiles.min()), abs(diff_profiles.max()))
    im2 = axes[1].imshow(diff_profiles, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                         vmin=-vmax, vmax=vmax)
    axes[1].set_title('Difference (Answer - Control)')
    axes[1].set_xlabel('Chain position bin')
    axes[1].set_ylabel('KV Head')
    axes[1].set_yticks(range(n_kv_heads))
    axes[1].set_yticklabels([f'H{h}{"*" if h in [0,5] else ""}' for h in range(n_kv_heads)])
    plt.colorbar(im2, ax=axes[1], label='Δ Attention')

    plt.tight_layout()
    fig4_path = str(RESULTS_DIR / "kv_head_attention_heatmap.png")
    plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure 4 saved: {fig4_path}")

    # ── Figure 5: Late-chain attention at answer step — statistical test ──
    # For each KV head, compute late-chain attention at answer vs control
    # Bootstrap test for significant difference
    print("\n--- Statistical test: late-chain attention shift ---")

    fig, ax = plt.subplots(figsize=(10, 6))

    kv_late_ans_means = []
    kv_late_ctrl_means = []
    kv_diffs = []
    kv_pvals = []

    for kv_h in range(n_kv_heads):
        qh_start = kv_h * group_size
        qh_end = (kv_h + 1) * group_size

        # Per-problem late fraction at answer step
        ans_per_prob = []
        ctrl_per_prob = []

        for pi in range(n_problems):
            a_vals = []
            for qh in range(qh_start, qh_end):
                a_vals.append(float(all_answer_binned[pi][key_layer][qh][14:N_BINS].sum()))
            ans_per_prob.append(np.mean(a_vals))

            c_vals = []
            for ci in range(len(all_control_binned[pi])):
                for qh in range(qh_start, qh_end):
                    c_vals.append(float(all_control_binned[pi][ci][key_layer][qh][14:N_BINS].sum()))
            ctrl_per_prob.append(np.mean(c_vals))

        # Paired difference test (per problem)
        diffs = [a - c for a, c in zip(ans_per_prob, ctrl_per_prob)]
        mean_diff = np.mean(diffs)
        # Bootstrap p-value
        n_boot = 2000
        boot_means = []
        for _ in range(n_boot):
            sample = np.random.choice(diffs, size=len(diffs), replace=True)
            boot_means.append(np.mean(sample))
        boot_means = np.array(boot_means)
        if mean_diff > 0:
            p_val = float((boot_means <= 0).sum()) / n_boot
        else:
            p_val = float((boot_means >= 0).sum()) / n_boot

        kv_late_ans_means.append(np.mean(ans_per_prob))
        kv_late_ctrl_means.append(np.mean(ctrl_per_prob))
        kv_diffs.append(mean_diff)
        kv_pvals.append(p_val)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        marker = " <-- H5" if kv_h == 5 else (" <-- H0" if kv_h == 0 else "")
        print(f"  KV H{kv_h}: late_ans={np.mean(ans_per_prob):.4f} late_ctrl={np.mean(ctrl_per_prob):.4f}"
              f"  diff={mean_diff:+.4f} p={p_val:.4f} {sig}{marker}")

    # Bar chart of differences
    colors_bar = ['#2ecc71' if h == 0 else '#e74c3c' if h == 5 else '#3498db' for h in range(n_kv_heads)]
    bars = ax.bar(range(n_kv_heads), kv_diffs, color=colors_bar, alpha=0.8, edgecolor='black')

    # Add significance stars
    for h, (diff, pv) in enumerate(zip(kv_diffs, kv_pvals)):
        sig_str = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        if sig_str:
            y_pos = diff + 0.001 if diff > 0 else diff - 0.003
            ax.text(h, y_pos, sig_str, ha='center', fontsize=12, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('KV Head')
    ax.set_ylabel('Late-chain attention shift (answer - control)')
    ax.set_title(f'Late-Chain (70-100%) Attention Shift at Answer Step\nLayer {key_layer}, Qwen3-4B-Base')
    ax.set_xticks(range(n_kv_heads))
    ax.set_xticklabels([f'H{h}{"*" if h in [0,5] else ""}' for h in range(n_kv_heads)])

    plt.tight_layout()
    fig5_path = str(RESULTS_DIR / "late_chain_attention_shift.png")
    plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure 5 saved: {fig5_path}")

    # ── Figure 6: Multi-layer late-chain shift for H5 vs others ──
    fig, ax = plt.subplots(figsize=(10, 6))

    h5_shifts = []
    h0_shifts = []
    other_shifts = []
    layer_pcts = []

    for layer_idx in layers_probe:
        layer_pcts.append(100 * layer_idx / (n_layers - 1))

        for kv_h, shift_list in [(5, h5_shifts), (0, h0_shifts)]:
            qh_start = kv_h * group_size
            qh_end = (kv_h + 1) * group_size
            diffs_l = []
            for pi in range(n_problems):
                a = np.mean([float(all_answer_binned[pi][layer_idx][qh][14:N_BINS].sum())
                             for qh in range(qh_start, qh_end)])
                c_vals = []
                for ci in range(len(all_control_binned[pi])):
                    for qh in range(qh_start, qh_end):
                        c_vals.append(float(all_control_binned[pi][ci][layer_idx][qh][14:N_BINS].sum()))
                c = np.mean(c_vals) if c_vals else 0
                diffs_l.append(a - c)
            shift_list.append(np.mean(diffs_l))

        # Others: average across non-H5, non-H0 heads
        other_diffs_l = []
        for pi in range(n_problems):
            a_all = []
            c_all = []
            for kv_h in range(n_kv_heads):
                if kv_h in [0, 5]:
                    continue
                qh_start = kv_h * group_size
                qh_end = (kv_h + 1) * group_size
                for qh in range(qh_start, qh_end):
                    a_all.append(float(all_answer_binned[pi][layer_idx][qh][14:N_BINS].sum()))
                for ci in range(len(all_control_binned[pi])):
                    for qh in range(qh_start, qh_end):
                        c_all.append(float(all_control_binned[pi][ci][layer_idx][qh][14:N_BINS].sum()))
            if a_all and c_all:
                other_diffs_l.append(np.mean(a_all) - np.mean(c_all))
        other_shifts.append(np.mean(other_diffs_l))

    ax.plot(layer_pcts, h5_shifts, 'r-o', label='H5 (answer head)', linewidth=2, markersize=6)
    ax.plot(layer_pcts, h0_shifts, 'g-s', label='H0', linewidth=2, markersize=6)
    ax.plot(layer_pcts, other_shifts, 'b-^', label='Others (avg)', linewidth=2, markersize=6)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Layer depth (%)')
    ax.set_ylabel('Late-chain attention shift (answer - control)')
    ax.set_title('Late-Chain Attention Shift Across Layers\nH5 vs H0 vs Other Heads')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig6_path = str(RESULTS_DIR / "late_chain_shift_by_layer.png")
    plt.savefig(fig6_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure 6 saved: {fig6_path}")

    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Saving results")
    print(f"{'='*70}")

    # Save compact results (not full attention data — too large)
    results = {
        'model': MODEL_NAME,
        'n_problems': n_problems,
        'n_q_heads': n_q_heads,
        'n_kv_heads': n_kv_heads,
        'group_size': group_size,
        'n_layers': n_layers,
        'layers_probed': layers_probe,
        'n_bins': N_BINS,
        'n_control_steps': N_CONTROL_STEPS,
        'kv_late_shift': {
            'layer': key_layer,
            'heads': {str(h): {'ans': kv_late_ans_means[h], 'ctrl': kv_late_ctrl_means[h],
                                'diff': kv_diffs[h], 'p': kv_pvals[h]}
                      for h in range(n_kv_heads)},
        },
        'entropy_summary': {},
        'attention_profiles': {},
    }

    # Save entropy per layer per head
    for layer_idx in layers_probe:
        results['entropy_summary'][str(layer_idx)] = {
            'answer_mean': float(head_entropy_answer[layer_idx].mean()),
            'control_mean': float(head_entropy_control[layer_idx].mean()),
            'h5_answer': float(head_entropy_answer[layer_idx][h5_q_start:h5_q_end].mean()),
            'h5_control': float(head_entropy_control[layer_idx][h5_q_start:h5_q_end].mean()),
            'h0_answer': float(head_entropy_answer[layer_idx][h0_q_start:h0_q_end].mean()),
            'h0_control': float(head_entropy_control[layer_idx][h0_q_start:h0_q_end].mean()),
        }

    # Save average attention profiles at key layer
    for kv_h in range(n_kv_heads):
        qh_start = kv_h * group_size
        qh_end = (kv_h + 1) * group_size
        ans_b = []
        ctrl_b = []
        for pi in range(n_problems):
            for qh in range(qh_start, qh_end):
                ans_b.append(all_answer_binned[pi][key_layer][qh][:N_BINS].tolist()
                             if isinstance(all_answer_binned[pi][key_layer][qh], np.ndarray)
                             else all_answer_binned[pi][key_layer][qh][:N_BINS])
            for ci in range(len(all_control_binned[pi])):
                for qh in range(qh_start, qh_end):
                    ctrl_b.append(all_control_binned[pi][ci][key_layer][qh][:N_BINS].tolist()
                                  if isinstance(all_control_binned[pi][ci][key_layer][qh], np.ndarray)
                                  else all_control_binned[pi][ci][key_layer][qh][:N_BINS])
        results['attention_profiles'][f'H{kv_h}_answer'] = np.mean(ans_b, axis=0).tolist()
        results['attention_profiles'][f'H{kv_h}_control'] = np.mean(ctrl_b, axis=0).tolist()

    results['h5_layer_shifts'] = {str(l): s for l, s in zip(layers_probe, h5_shifts)}
    results['h0_layer_shifts'] = {str(l): s for l, s in zip(layers_probe, h0_shifts)}
    results['other_layer_shifts'] = {str(l): s for l, s in zip(layers_probe, other_shifts)}

    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {RESULTS_DIR / 'results.json'}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("DONE")


if __name__ == '__main__':
    main()
