#!/usr/bin/env python3
"""
Experiment 008: PGD Adversarial Null Space Replication on Llama-3.1-8B

Tests whether PGD-optimized KV cache perturbations can change the model's
final answer while preserving intermediate token predictions. Replicates
the null space finding from Exp 4 (Qwen3-4B-Base, 377x signal norm) on
a second architecture.

Approach: Prompt-only attacks — perturb only the prompt KV entries,
re-process reasoning tokens with perturbed context. Text preservation
is measured by token-level prediction match. Answer change is measured
by generating freely from the perturbed cache.

Key adaptations for Llama vs Qwen:
- attn_implementation="eager" (required for DynamicCache manipulation)
- DynamicCache API: .layers[i].keys/.values (transformers 5.3.0)
- Chunked teacher-forcing for memory efficiency
- GQA: 8 KV heads shared across 32 query heads
- float32 perturbation tensors for optimizer stability, cast to float16 for forward
"""

import os
import json
import time
import random
import gc
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NUM_PROBLEMS = 25
MAX_GEN_TOKENS = 512
MAX_REASONING_TOKENS = 250  # Cap reasoning length for PGD memory
MAX_SEQ_LEN = 1536
PGD_STEPS = 50
PGD_LR = 0.05
LAMBDA_ANS = 5.0          # Weight for answer change loss
MAX_PERT_RATIO = 500.0     # Max perturbation L2 norm / signal L2 norm
ANSWER_REGION_N = 10       # Last N tokens as answer region
CHUNK_SIZE = 128
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_008")

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
    """Generate a CoT trace with greedy decoding."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = []
    past_kv = None
    current_input = inputs.input_ids

    for step in range(max_tokens):
        if past_kv is not None:
            outputs = model(input_ids=current_input, past_key_values=past_kv,
                          use_cache=True)
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
        if "\nQ:" in current_text:
            idx = current_text.find("\nQ:")
            if idx > 0:
                generated_ids = tokenizer.encode(
                    current_text[:idx], add_special_tokens=False)
            break

        current_input = next_token

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    del past_kv, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return generated_text


@torch.no_grad()
def teacher_force_chunked(model, input_ids, chunk_size=CHUNK_SIZE):
    """Teacher-force a sequence in chunks, returning the KV cache."""
    seq_len = input_ids.shape[1]
    past_kv = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        if past_kv is not None:
            outputs = model(input_ids=chunk, past_key_values=past_kv,
                          use_cache=True)
        else:
            outputs = model(input_ids=chunk, use_cache=True)
        past_kv = outputs.past_key_values
        del outputs
    return past_kv


def find_answer_boundary(reasoning_len):
    """
    Position within reasoning tokens where the answer region starts.
    The reasoning tokens are everything BEFORE '####', so there's no
    explicit answer marker. We use the last ANSWER_REGION_N tokens as
    the 'answer region' — these include the final calculation result
    whose prediction leads to the answer.
    """
    return max(10, reasoning_len - ANSWER_REGION_N)


def pgd_attack(model, clean_prompt_keys, clean_prompt_values,
               reasoning_token_ids, clean_logits, num_layers,
               answer_start_pos, pgd_steps=PGD_STEPS, lr=PGD_LR):
    """
    PGD prompt-only attack: perturb prompt KV to change answer while
    preserving reasoning text predictions.

    Args:
        clean_prompt_keys: list of tensors [1, n_kv_heads, prompt_len, head_dim]
        clean_prompt_values: same shape
        reasoning_token_ids: [1, reasoning_len] tensor
        clean_logits: [1, reasoning_len, vocab_size] tensor (detached)
        answer_start_pos: index into reasoning tokens where answer region starts

    Returns:
        delta_k, delta_v: perturbation tensors (float32)
        step_metrics: list of per-step dicts
    """
    device = clean_prompt_keys[0].device
    dtype_model = clean_prompt_keys[0].dtype  # float16

    # Initialize perturbation in float32 for optimizer stability
    delta_k = []
    delta_v = []
    for l in range(num_layers):
        dk = torch.zeros(clean_prompt_keys[l].shape,
                        device=device, dtype=torch.float32)
        dk.requires_grad_(True)
        delta_k.append(dk)
        dv = torch.zeros(clean_prompt_values[l].shape,
                        device=device, dtype=torch.float32)
        dv.requires_grad_(True)
        delta_v.append(dv)

    optimizer = torch.optim.Adam(delta_k + delta_v, lr=lr)
    step_metrics = []

    for step in range(pgd_steps):
        optimizer.zero_grad()

        # Build perturbed prompt-only cache
        perturbed_cache = DynamicCache()
        for l in range(num_layers):
            pk = clean_prompt_keys[l] + delta_k[l].to(dtype_model)
            pv = clean_prompt_values[l] + delta_v[l].to(dtype_model)
            perturbed_cache.update(pk, pv, l)

        # Forward pass: re-process reasoning tokens with perturbed prompt KV
        outputs = model(input_ids=reasoning_token_ids,
                       past_key_values=perturbed_cache,
                       use_cache=False)
        perturbed_logits = outputs.logits  # [1, reasoning_len, vocab_size]

        # Text preservation loss: KL(perturbed || clean) on text region
        # Use float32 for numerical stability in KL computation
        text_log_p = F.log_softmax(perturbed_logits[:, :answer_start_pos, :].float(),
                                   dim=-1)
        text_q = F.softmax(clean_logits[:, :answer_start_pos, :].float(),
                          dim=-1).detach()
        text_kl = F.kl_div(text_log_p, text_q, reduction='batchmean')

        # Answer change loss: maximize KL(perturbed || clean) on answer region
        ans_log_p = F.log_softmax(perturbed_logits[:, answer_start_pos:, :].float(),
                                  dim=-1)
        ans_q = F.softmax(clean_logits[:, answer_start_pos:, :].float(),
                         dim=-1).detach()
        answer_kl = F.kl_div(ans_log_p, ans_q, reduction='batchmean')

        # Total loss: minimize text_kl, maximize answer_kl
        loss = text_kl - LAMBDA_ANS * answer_kl
        loss.backward()
        optimizer.step()

        # Project perturbation onto L2 ball (per layer)
        with torch.no_grad():
            for l in range(num_layers):
                sig_k = clean_prompt_keys[l].float().norm().item()
                max_k = MAX_PERT_RATIO * sig_k
                cur_k = delta_k[l].norm().item()
                if cur_k > max_k:
                    delta_k[l].mul_(max_k / cur_k)

                sig_v = clean_prompt_values[l].float().norm().item()
                max_v = MAX_PERT_RATIO * sig_v
                cur_v = delta_v[l].norm().item()
                if cur_v > max_v:
                    delta_v[l].mul_(max_v / cur_v)

        sm = {
            'step': step,
            'loss': loss.item(),
            'text_kl': text_kl.item(),
            'answer_kl': answer_kl.item(),
        }
        step_metrics.append(sm)

        if step % 10 == 0:
            # Compute text match rate for monitoring
            with torch.no_grad():
                clean_preds = clean_logits[:, :answer_start_pos, :].argmax(dim=-1)
                pert_preds = perturbed_logits[:, :answer_start_pos, :].argmax(dim=-1)
                text_match = (clean_preds == pert_preds).float().mean().item()
            print(f"    Step {step:3d}: loss={sm['loss']:.4f}, "
                  f"text_KL={sm['text_kl']:.4f}, answer_KL={sm['answer_kl']:.4f}, "
                  f"text_match={text_match:.1%}")

        # Free per-step memory
        del perturbed_cache, outputs, perturbed_logits, loss
        del text_log_p, text_q, ans_log_p, ans_q
        # Don't clear cache every step — too slow

    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()

    return delta_k, delta_v, step_metrics


@torch.no_grad()
def evaluate_attack(model, tokenizer, clean_prompt_keys, clean_prompt_values,
                    delta_k, delta_v, reasoning_token_ids, clean_logits,
                    true_answer, clean_answer, num_layers, answer_start_pos):
    """
    Evaluate PGD attack: measure text preservation and answer change.

    Returns dict with:
    - text_match_rate: fraction of text-region tokens with same argmax
    - all_match_rate: fraction of ALL tokens with same argmax
    - answer_changed: bool
    - attack_success: answer changed AND text preserved (≥90% match)
    - perturbation norms
    """
    dtype_model = clean_prompt_keys[0].dtype

    # Build perturbed prompt-only cache
    perturbed_cache = DynamicCache()
    for l in range(num_layers):
        pk = clean_prompt_keys[l] + delta_k[l].to(dtype_model)
        pv = clean_prompt_values[l] + delta_v[l].to(dtype_model)
        perturbed_cache.update(pk, pv, l)

    # Re-process reasoning tokens → get logits + updated cache
    eval_out = model(input_ids=reasoning_token_ids,
                    past_key_values=perturbed_cache, use_cache=True)
    perturbed_logits = eval_out.logits

    # Text preservation: argmax match
    clean_preds = clean_logits[:, :answer_start_pos, :].argmax(dim=-1)
    pert_preds = perturbed_logits[:, :answer_start_pos, :].argmax(dim=-1)
    text_match_rate = (clean_preds == pert_preds).float().mean().item()

    # All-token match
    all_clean = clean_logits.argmax(dim=-1)
    all_pert = perturbed_logits.argmax(dim=-1)
    all_match_rate = (all_clean == all_pert).float().mean().item()

    # Answer-region match
    ans_clean = clean_logits[:, answer_start_pos:, :].argmax(dim=-1)
    ans_pert = perturbed_logits[:, answer_start_pos:, :].argmax(dim=-1)
    answer_match_rate = (ans_clean == ans_pert).float().mean().item()

    # Generate answer from perturbed cache
    gen_cache = eval_out.past_key_values
    next_token = perturbed_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token[0, 0].item()]

    for _ in range(150):
        gen_out = model(input_ids=next_token, past_key_values=gen_cache,
                       use_cache=True)
        gen_cache = gen_out.past_key_values
        next_token = gen_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = next_token[0, 0].item()
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if "####" in decoded:
            after = decoded.split("####")[-1]
            if re.search(r'\d+', after):
                break
        if re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?\s*-?[\d,]+', decoded):
            break
        if "\nQ:" in decoded:
            break

    perturbed_answer_text = tokenizer.decode(generated, skip_special_tokens=True)
    perturbed_answer = extract_answer(perturbed_answer_text)
    pert_norm_ans = normalize_answer(perturbed_answer) if perturbed_answer else ""

    # Perturbation norms (average across layers)
    k_pert_norms = [dk.norm().item() for dk in delta_k]
    v_pert_norms = [dv.norm().item() for dv in delta_v]
    k_signal_norms = [ck.float().norm().item() for ck in clean_prompt_keys]
    v_signal_norms = [cv.float().norm().item() for cv in clean_prompt_values]

    k_ratios = [p / (s + 1e-8) for p, s in zip(k_pert_norms, k_signal_norms)]
    v_ratios = [p / (s + 1e-8) for p, s in zip(v_pert_norms, v_signal_norms)]

    mean_k_ratio = float(np.mean(k_ratios))
    mean_v_ratio = float(np.mean(v_ratios))
    overall_ratio = float(np.mean(k_ratios + v_ratios))

    clean_norm_ans = normalize_answer(clean_answer) if clean_answer else ""
    answer_changed = (pert_norm_ans != clean_norm_ans) and pert_norm_ans != ""
    attack_success = answer_changed and (text_match_rate >= 0.90)

    del eval_out, gen_cache
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'text_match_rate': text_match_rate,
        'all_match_rate': all_match_rate,
        'answer_match_rate': answer_match_rate,
        'perturbed_answer': perturbed_answer,
        'perturbed_answer_text': perturbed_answer_text[:200],
        'answer_changed': answer_changed,
        'attack_success': attack_success,
        'mean_k_ratio': mean_k_ratio,
        'mean_v_ratio': mean_v_ratio,
        'overall_pert_ratio': overall_ratio,
        'per_layer_k_ratios': k_ratios,
        'per_layer_v_ratios': v_ratios,
    }


def generate_figures(all_results, results_dir):
    """Generate analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not all_results:
        print("No results for figures")
        return

    # ── Figure 1: Attack success and text preservation ────────────────
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    problems = list(range(len(all_results)))
    text_matches = [r['text_match_rate'] * 100 for r in all_results]
    answer_changed = [r['answer_changed'] for r in all_results]
    attack_success = [r['attack_success'] for r in all_results]

    colors = ['#2ecc71' if s else '#e74c3c' if a else '#95a5a6'
              for s, a in zip(attack_success, answer_changed)]
    ax1.bar(problems, text_matches, color=colors, edgecolor='black', lw=0.5)
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
    ax1.set_xlabel('Problem', fontsize=12)
    ax1.set_ylabel('Text Prediction Match (%)', fontsize=12)
    ax1.set_title('Text Preservation by Problem\n'
                  'Green=success, Red=answer changed but text degraded, Gray=no change',
                  fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.legend()

    # Perturbation norms
    pert_ratios = [r['overall_pert_ratio'] for r in all_results]
    ax2.bar(problems, pert_ratios, color=colors, edgecolor='black', lw=0.5)
    ax2.set_xlabel('Problem', fontsize=12)
    ax2.set_ylabel('Perturbation / Signal Norm', fontsize=12)
    ax2.set_title('Perturbation Magnitude (relative to signal)',
                  fontsize=11)

    plt.tight_layout()
    fig1.savefig(os.path.join(results_dir, "attack_results.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: attack_results.png")

    # ── Figure 2: PGD optimization curves (first 3 problems) ─────────
    fig2, axes = plt.subplots(1, min(3, len(all_results)), figsize=(16, 5))
    if len(all_results) == 1:
        axes = [axes]

    for i, (ax, r) in enumerate(zip(axes, all_results[:3])):
        if 'step_metrics' not in r:
            continue
        steps = [m['step'] for m in r['step_metrics']]
        text_kls = [m['text_kl'] for m in r['step_metrics']]
        ans_kls = [m['answer_kl'] for m in r['step_metrics']]

        ax.plot(steps, text_kls, 'b-', label='Text KL (minimize)', lw=1.5)
        ax.plot(steps, ans_kls, 'r-', label='Answer KL (maximize)', lw=1.5)
        ax.set_xlabel('PGD Step')
        ax.set_ylabel('KL Divergence')
        ax.set_title(f'Problem {i}: {"SUCCESS" if r["attack_success"] else "FAIL"}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(os.path.join(results_dir, "pgd_optimization_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: pgd_optimization_curves.png")

    # ── Figure 3: Per-layer perturbation distribution ─────────────────
    # Average per-layer ratios across successful attacks
    successes = [r for r in all_results if r.get('attack_success')]
    if successes:
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        n_layers = len(successes[0].get('per_layer_k_ratios', []))
        if n_layers > 0:
            avg_k = np.mean([r['per_layer_k_ratios'] for r in successes], axis=0)
            avg_v = np.mean([r['per_layer_v_ratios'] for r in successes], axis=0)
            layers = np.arange(n_layers)
            width = 0.35
            ax3.bar(layers - width/2, avg_k, width, label='Key perturbation',
                   color='#3498db', alpha=0.8)
            ax3.bar(layers + width/2, avg_v, width, label='Value perturbation',
                   color='#e74c3c', alpha=0.8)
            ax3.set_xlabel('Layer', fontsize=12)
            ax3.set_ylabel('Perturbation / Signal Norm', fontsize=12)
            ax3.set_title('Per-Layer Perturbation Distribution (successful attacks)',
                         fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.2)

            plt.tight_layout()
            fig3.savefig(os.path.join(results_dir, "per_layer_perturbation.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("Figure saved: per_layer_perturbation.png")

    # ── Figure 4: Cross-model comparison with Qwen (Exp 4) ───────────
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    n_total = len(all_results)
    n_success = sum(1 for r in all_results if r['attack_success'])
    n_changed = sum(1 for r in all_results if r['answer_changed'])
    success_rate = n_success / n_total if n_total > 0 else 0
    changed_rate = n_changed / n_total if n_total > 0 else 0

    # Qwen results from Exp 4
    categories = ['Attack Success\n(answer changed,\ntext preserved)',
                  'Answer Changed\n(any text quality)',
                  'Mean Perturbation\nNorm Ratio']
    qwen_vals = [100, 100, 377]  # Exp 4: 100% success on prompt attacks
    llama_vals = [success_rate * 100, changed_rate * 100,
                  float(np.mean([r['overall_pert_ratio'] for r in all_results])) if all_results else 0]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax4.bar(x - width/2, qwen_vals, width, label='Qwen3-4B (Exp 4)',
                   color='#e67e22', alpha=0.8)
    bars2 = ax4.bar(x + width/2, llama_vals, width, label='Llama-3.1-8B (Exp 008)',
                   color='#3498db', alpha=0.8)

    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('PGD Null Space Attack: Cross-Model Comparison\n'
                  'Prompt-Only Attacks', fontsize=13)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.legend(fontsize=11)

    # Add value labels
    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig4.savefig(os.path.join(results_dir, "cross_model_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure saved: cross_model_comparison.png")


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool):
        return bool(obj)
    return obj


def main():
    start_time = time.time()
    print(f"{'='*70}")
    print(f"Experiment 008: PGD Null Space Replication on Llama-3.1-8B")
    print(f"Model: {MODEL_NAME}")
    print(f"PGD: {PGD_STEPS} steps, lr={PGD_LR}, lambda_ans={LAMBDA_ANS}")
    print(f"Max pert ratio: {MAX_PERT_RATIO}x signal norm")
    print(f"{'='*70}\n")

    # ── Load model ────────────────────────────────────────────────────
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
    model.requires_grad_(False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded in {time.time() - start_time:.1f}s")
    print(f"Layers: {num_layers}, KV heads: "
          f"{getattr(model.config, 'num_key_value_heads', 'N/A')}")

    # ── Load GSM8K ────────────────────────────────────────────────────
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected = indices[:NUM_PROBLEMS]

    # ── Phase 1: Generate traces ──────────────────────────────────────
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
        trace_text = generate_trace(model, tokenizer, prompt)
        gen_answer = extract_answer(trace_text)
        gen_norm = normalize_answer(gen_answer) if gen_answer else ""
        correct = (gen_norm == true_answer)

        print(f"  Problem {prob_idx+1}/{NUM_PROBLEMS} (#{ds_idx}): "
              f"'{gen_answer}' (correct: {correct})")

        if not correct:
            continue

        # Split at #### to separate reasoning from answer
        if "####" in trace_text:
            reasoning_text = trace_text[:trace_text.index("####")]
        else:
            reasoning_text = trace_text

        # Tokenize
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]

        full_text = prompt + trace_text
        full_inputs = tokenizer(full_text, return_tensors="pt")
        full_len = full_inputs.input_ids.shape[1]

        reasoning_only_text = prompt + reasoning_text
        reasoning_inputs = tokenizer(reasoning_only_text, return_tensors="pt")
        reasoning_len = reasoning_inputs.input_ids.shape[1] - prompt_len

        if reasoning_len < 10 or reasoning_len > MAX_REASONING_TOKENS:
            print(f"    SKIP: reasoning_len={reasoning_len} (min 10, max {MAX_REASONING_TOKENS})")
            continue
        if full_len > MAX_SEQ_LEN:
            print(f"    SKIP: seq too long ({full_len} > {MAX_SEQ_LEN})")
            continue

        problems_data.append({
            "ds_idx": ds_idx,
            "true_answer": true_answer,
            "clean_answer": gen_answer,
            "prompt": prompt,
            "trace_text": trace_text,
            "reasoning_text": reasoning_text,
            "prompt_len": prompt_len,
            "reasoning_len": reasoning_len,
            "full_len": full_len,
        })
        print(f"    Valid! prompt_len={prompt_len}, reasoning_len={reasoning_len}")

    n_valid = len(problems_data)
    print(f"\n=== Phase 1 complete: {n_valid} valid problems ===\n")

    if n_valid < 2:
        print("ERROR: Too few valid problems!")
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump({"error": "too_few_valid", "n_valid": n_valid}, f, indent=2)
        return

    # ── Phase 2: Smoke test (gradient flow) ───────────────────────────
    print("=== Smoke test: verifying gradient flow through DynamicCache ===")
    pd0 = problems_data[0]

    # Teacher-force prompt only
    prompt_inputs = tokenizer(pd0["prompt"], return_tensors="pt").to(model.device)
    prompt_kv = teacher_force_chunked(model, prompt_inputs.input_ids)
    clean_pk = [prompt_kv.layers[l].keys.clone() for l in range(num_layers)]
    clean_pv = [prompt_kv.layers[l].values.clone() for l in range(num_layers)]
    del prompt_kv
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Prompt KV shape: {clean_pk[0].shape}")
    print(f"  Prompt KV dtype: {clean_pk[0].dtype}")

    # Get reasoning token IDs
    reasoning_text_full = pd0["prompt"] + pd0["reasoning_text"]
    reasoning_full_ids = tokenizer(reasoning_text_full, return_tensors="pt").to(model.device)
    reasoning_token_ids = reasoning_full_ids.input_ids[:, pd0["prompt_len"]:]
    print(f"  Reasoning tokens: {reasoning_token_ids.shape}")

    # Get clean logits
    clean_cache = DynamicCache()
    for l in range(num_layers):
        clean_cache.update(clean_pk[l].clone(), clean_pv[l].clone(), l)
    with torch.no_grad():
        clean_out = model(input_ids=reasoning_token_ids,
                        past_key_values=clean_cache, use_cache=False)
        clean_logits = clean_out.logits.clone()
    del clean_cache, clean_out
    gc.collect()
    torch.cuda.empty_cache()

    # Create small perturbation and test gradient flow
    test_dk = torch.zeros_like(clean_pk[0], dtype=torch.float32)
    test_dk.requires_grad_(True)
    test_dv = torch.zeros_like(clean_pv[0], dtype=torch.float32)
    test_dv.requires_grad_(True)

    test_cache = DynamicCache()
    test_cache.update(clean_pk[0] + test_dk.half(), clean_pv[0] + test_dv.half(), 0)
    for l in range(1, num_layers):
        test_cache.update(clean_pk[l].clone(), clean_pv[l].clone(), l)

    test_out = model(input_ids=reasoning_token_ids[:, :5],  # Just 5 tokens
                    past_key_values=test_cache, use_cache=False)
    test_loss = test_out.logits.sum()
    test_loss.backward()

    grad_ok = (test_dk.grad is not None and test_dk.grad.abs().sum() > 0)
    print(f"  Gradient flow: {'OK' if grad_ok else 'FAILED!'}")
    if test_dk.grad is not None:
        print(f"  Gradient norm: {test_dk.grad.norm().item():.6f}")
    else:
        print("  ERROR: No gradient on perturbation tensor!")
        print("  Cannot run PGD without gradient flow. Aborting.")
        return

    del test_dk, test_dv, test_cache, test_out, test_loss
    del clean_pk, clean_pv, clean_logits, reasoning_token_ids
    gc.collect()
    torch.cuda.empty_cache()
    print("  Smoke test PASSED!\n")

    # ── Phase 3: Full PGD experiment ──────────────────────────────────
    print(f"=== Phase 3: PGD attacks ({n_valid} problems) ===")
    all_results = []

    for pi, pd in enumerate(problems_data):
        elapsed = time.time() - start_time
        if elapsed > 1500:  # Leave 5 min for analysis
            print(f"Time budget reached at problem {pi}")
            break

        print(f"\nProblem {pi+1}/{n_valid} (#{pd['ds_idx']}), "
              f"true_answer={pd['true_answer']} [{elapsed:.0f}s elapsed]")

        try:
            # Teacher-force prompt
            prompt_inputs = tokenizer(pd["prompt"], return_tensors="pt").to(model.device)
            prompt_kv = teacher_force_chunked(model, prompt_inputs.input_ids)
            clean_pk = [prompt_kv.layers[l].keys.clone() for l in range(num_layers)]
            clean_pv = [prompt_kv.layers[l].values.clone() for l in range(num_layers)]
            del prompt_kv
            gc.collect()
            torch.cuda.empty_cache()

            # Get reasoning token IDs
            reasoning_text_full = pd["prompt"] + pd["reasoning_text"]
            reasoning_full_ids = tokenizer(reasoning_text_full,
                                          return_tensors="pt").to(model.device)
            reasoning_token_ids = reasoning_full_ids.input_ids[:, pd["prompt_len"]:]

            # Find answer boundary in reasoning tokens
            reasoning_len_actual = reasoning_token_ids.shape[1]
            answer_start = find_answer_boundary(reasoning_len_actual)
            print(f"  Reasoning tokens: {reasoning_token_ids.shape[1]}, "
                  f"answer_start: {answer_start}")

            # Get clean logits
            clean_cache = DynamicCache()
            for l in range(num_layers):
                clean_cache.update(clean_pk[l].clone(), clean_pv[l].clone(), l)
            with torch.no_grad():
                clean_out = model(input_ids=reasoning_token_ids,
                                past_key_values=clean_cache, use_cache=False)
                clean_logits = clean_out.logits.clone()
            del clean_cache, clean_out
            gc.collect()
            torch.cuda.empty_cache()

            # PGD attack
            print(f"  Running PGD ({PGD_STEPS} steps)...")
            delta_k, delta_v, step_metrics = pgd_attack(
                model, clean_pk, clean_pv,
                reasoning_token_ids, clean_logits,
                num_layers, answer_start,
            )

            # Evaluate
            print(f"  Evaluating attack...")
            result = evaluate_attack(
                model, tokenizer, clean_pk, clean_pv,
                delta_k, delta_v, reasoning_token_ids, clean_logits,
                pd["true_answer"], pd["clean_answer"],
                num_layers, answer_start,
            )
            result['step_metrics'] = step_metrics
            result['ds_idx'] = pd['ds_idx']
            result['true_answer'] = pd['true_answer']
            result['clean_answer'] = pd['clean_answer']
            result['prompt_len'] = pd['prompt_len']
            result['reasoning_len'] = pd['reasoning_len']
            result['answer_start'] = answer_start

            print(f"  RESULT: text_match={result['text_match_rate']:.1%}, "
                  f"answer_changed={result['answer_changed']}, "
                  f"success={result['attack_success']}, "
                  f"pert_ratio={result['overall_pert_ratio']:.1f}x")
            print(f"  Clean answer: {pd['clean_answer']}, "
                  f"Perturbed answer: {result['perturbed_answer']}")

            all_results.append(result)

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM! Skipping this problem.")
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    # ── Phase 4: Analysis ─────────────────────────────────────────────
    n_evaluated = len(all_results)
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Problems evaluated: {n_evaluated}")

    if n_evaluated == 0:
        print("No problems evaluated successfully!")
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump({"error": "no_results", "n_evaluated": 0}, f, indent=2)
        return

    n_success = sum(1 for r in all_results if r['attack_success'])
    n_changed = sum(1 for r in all_results if r['answer_changed'])
    n_text_preserved = sum(1 for r in all_results if r['text_match_rate'] >= 0.90)

    success_rate = n_success / n_evaluated
    changed_rate = n_changed / n_evaluated
    text_preserved_rate = n_text_preserved / n_evaluated

    mean_text_match = np.mean([r['text_match_rate'] for r in all_results])
    mean_pert_ratio = np.mean([r['overall_pert_ratio'] for r in all_results])
    mean_all_match = np.mean([r['all_match_rate'] for r in all_results])

    print(f"\nAttack success (answer changed + text preserved ≥90%): "
          f"{n_success}/{n_evaluated} ({success_rate:.1%})")
    print(f"Answer changed (any text quality): "
          f"{n_changed}/{n_evaluated} ({changed_rate:.1%})")
    print(f"Text preserved (≥90% match): "
          f"{n_text_preserved}/{n_evaluated} ({text_preserved_rate:.1%})")
    print(f"\nMean text match rate: {mean_text_match:.1%}")
    print(f"Mean all-token match rate: {mean_all_match:.1%}")
    print(f"Mean perturbation/signal ratio: {mean_pert_ratio:.1f}x")

    # Per-problem details
    print(f"\n{'Problem':<10} {'Text%':<8} {'All%':<8} {'AnsChg':<8} "
          f"{'Success':<8} {'PertRat':<10} {'CleanAns':<10} {'PertAns':<10}")
    print("-" * 72)
    for r in all_results:
        print(f"#{r['ds_idx']:<8} {r['text_match_rate']:.1%}   "
              f"{r['all_match_rate']:.1%}   "
              f"{'YES' if r['answer_changed'] else 'no':<8} "
              f"{'YES' if r['attack_success'] else 'no':<8} "
              f"{r['overall_pert_ratio']:.1f}x{'':<5} "
              f"{r['clean_answer']:<10} {r['perturbed_answer']:<10}")

    # Compare with Qwen Exp 4
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL COMPARISON (Exp 4 Qwen vs Exp 008 Llama)")
    print(f"{'='*70}")
    print(f"{'Metric':<35} {'Qwen3-4B (Exp 4)':<20} {'Llama-8B (Exp 008)':<20}")
    print("-" * 75)
    print(f"{'Prompt-only success rate':<35} {'100% (6/6)':<20} "
          f"{f'{success_rate:.0%} ({n_success}/{n_evaluated})':<20}")
    print(f"{'Mean perturbation/signal ratio':<35} {'377x':<20} "
          f"{f'{mean_pert_ratio:.1f}x':<20}")
    print(f"{'Mean text preservation':<35} {'~100%':<20} "
          f"{f'{mean_text_match:.1%}':<20}")

    # ── Save results ──────────────────────────────────────────────────
    summary = {
        "experiment": "exp_008_llama_pgd_null_space",
        "model": MODEL_NAME,
        "n_valid": n_valid,
        "n_evaluated": n_evaluated,
        "pgd_steps": PGD_STEPS,
        "pgd_lr": PGD_LR,
        "lambda_ans": LAMBDA_ANS,
        "max_pert_ratio": MAX_PERT_RATIO,
        "attack_success_rate": success_rate,
        "answer_changed_rate": changed_rate,
        "text_preserved_rate": text_preserved_rate,
        "mean_text_match": float(mean_text_match),
        "mean_all_match": float(mean_all_match),
        "mean_pert_ratio": float(mean_pert_ratio),
        "elapsed_seconds": time.time() - start_time,
        "per_problem_results": [
            {k: v for k, v in r.items() if k != 'step_metrics'}
            for r in all_results
        ],
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_convert)

    # Save step metrics separately (can be large)
    step_data = {f"problem_{r['ds_idx']}": r.get('step_metrics', [])
                 for r in all_results}
    with open(os.path.join(RESULTS_DIR, "pgd_step_metrics.json"), "w") as f:
        json.dump(step_data, f, indent=2, default=_convert)

    # ── Generate figures ──────────────────────────────────────────────
    generate_figures(all_results, RESULTS_DIR)

    total_time = time.time() - start_time
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("DONE")


if __name__ == "__main__":
    main()
