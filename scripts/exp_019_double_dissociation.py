#!/usr/bin/env python3
"""
Experiment 019: Position-Controlled Double Dissociation

Tests functional separability of AC vs TC positions by measuring BOTH
answer accuracy AND text prediction quality after selective position noise.

Within narrow position bands (quartiles Q3, Q4), compare:
  - Noise AC-selective positions → measure answer accuracy + text quality
  - Noise TC-selective positions → measure answer accuracy + text quality

Double dissociation = AC-noise hurts accuracy more, TC-noise hurts text more.

Text quality: next-token prediction accuracy through the noised cache
(step-by-step forward, noise injected causally).
"""

import os
import json
import time
import random
import gc
import re

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B-Base"
NUM_PROBLEMS = 40
MAX_GEN_TOKENS = 512
MAX_SEQ_LEN = 1536
NOISE_FRACTIONS = [0.05, 0.10]
ATTENTION_LAYERS = [-1, -2, -3, -4]
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_019")

EXEMPLARS = [
    {"q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "a": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = <<21-15=6>>6 trees planted.\n#### 6"},
    {"q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "a": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = <<3+2=5>>5 cars are now in the parking lot.\n#### 5"},
    {"q": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "a": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = <<32+42=74>>74. After eating 35, they had 74 - 35 = <<74-35=39>>39 pieces left.\n#### 39"},
    {"q": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?", "a": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = <<20-12=8>>8 lollipops.\n#### 8"},
    {"q": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "a": "Shawn started with 5 toys. He got 2 toys each from mom and dad. So he got 2 + 2 = <<2+2=4>>4 more toys. Now he has 5 + 4 = <<5+4=9>>9 toys.\n#### 9"},
    {"q": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?", "a": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 4 * 5 = <<4*5=20>>20 computers were added. 9 + 20 = <<9+20=29>>29 computers now.\n#### 29"},
    {"q": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?", "a": "Michael started with 58 golf balls. After losing 23, he had 58 - 23 = <<58-23=35>>35. After losing 2 more, he had 35 - 2 = <<35-2=33>>33 golf balls.\n#### 33"},
    {"q": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "a": "Olivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = <<5*3=15>>15 dollars. She has 23 - 15 = <<23-15=8>>8 dollars left.\n#### 8"},
]


def build_prompt(question):
    prompt = ""
    for ex in EXEMPLARS:
        prompt += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
    prompt += f"Q: {question}\nA: Let's think step by step.\n"
    return prompt


def extract_answer(text):
    if "####" in text:
        after = text.split("####")[-1].strip()
        m = re.search(r'[\d,]+\.?\d*', after)
        return m.group(0).replace(',', '') if m else ""
    nums = re.findall(r'[\d,]+\.?\d*', text)
    return nums[-1].replace(',', '') if nums else ""


def normalize_answer(ans):
    try:
        val = float(ans)
        return str(int(val)) if val == int(val) else str(val)
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
                generated_ids = tokenizer.encode(current_text[:idx], add_special_tokens=False)
            break
        current_input = next_token

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    del past_kv, outputs
    gc.collect(); torch.cuda.empty_cache()
    return text


@torch.no_grad()
def build_prompt_cache(model, prompt_ids, num_layers):
    """Build clean KV cache for prompt tokens only."""
    outputs = model(input_ids=prompt_ids, use_cache=True)
    prompt_cache = outputs.past_key_values
    del outputs
    gc.collect(); torch.cuda.empty_cache()
    return prompt_cache


@torch.no_grad()
def compute_ac_scores(model, full_ids, prompt_len, reasoning_len, num_layers):
    """Compute AC scores by getting attention from last reasoning position."""
    # Forward just the last reasoning token with full cache (minus last token)
    # to get its attention to all previous positions
    prefix_ids = full_ids[:, :-1]
    last_token = full_ids[:, -1:]

    # Build cache for prefix
    prefix_out = model(input_ids=prefix_ids, use_cache=True)
    prefix_cache = prefix_out.past_key_values
    del prefix_out

    # Forward last token with output_attentions
    last_out = model(input_ids=last_token, past_key_values=prefix_cache,
                     use_cache=False, output_attentions=True)

    # Collect attention from last 4 layers
    n_layers_total = len(last_out.attentions)
    ac_scores = np.zeros(reasoning_len)
    for layer_idx in ATTENTION_LAYERS:
        actual_idx = layer_idx if layer_idx >= 0 else n_layers_total + layer_idx
        # attn shape: [1, n_heads, 1, total_seq_len-1]
        attn = last_out.attentions[actual_idx].float().squeeze(0).squeeze(1)  # [n_heads, total_len-1]
        # Average over heads
        attn_avg = attn.mean(dim=0)  # [total_len-1]
        # Extract reasoning region
        ac_scores += attn_avg[prompt_len:prompt_len + reasoning_len].cpu().numpy()

    ac_scores /= len(ATTENTION_LAYERS)

    del last_out, prefix_cache
    gc.collect(); torch.cuda.empty_cache()
    return ac_scores


@torch.no_grad()
def compute_tc_scores_fast(model, full_ids, prompt_len, reasoning_len, num_layers):
    """Compute TC scores using a single forward pass with output_attentions on a chunk."""
    # Use only the last 4 layers' attention from reasoning tokens
    reasoning_ids = full_ids[:, prompt_len:prompt_len + reasoning_len]

    # Build prompt cache
    prompt_ids = full_ids[:, :prompt_len]
    prompt_out = model(input_ids=prompt_ids, use_cache=True)
    prompt_cache = prompt_out.past_key_values
    del prompt_out

    # Forward reasoning with attention
    reasoning_out = model(input_ids=reasoning_ids, past_key_values=prompt_cache,
                          use_cache=False, output_attentions=True)

    n_layers_total = len(reasoning_out.attentions)
    tc_scores = np.zeros(reasoning_len)
    for layer_idx in ATTENTION_LAYERS:
        actual_idx = layer_idx if layer_idx >= 0 else n_layers_total + layer_idx
        # attn shape: [1, n_heads, reasoning_len, prompt_len+reasoning_len]
        attn = reasoning_out.attentions[actual_idx].float()
        attn = attn.mean(dim=(0, 1))  # [reasoning_len, prompt_len+reasoning_len]

        # TC[i] = mean attention from positions i+1..R-2 to position i
        for i in range(reasoning_len - 2):
            kv_idx = prompt_len + i
            later = attn[i + 1:reasoning_len - 1, kv_idx]
            tc_scores[i] += later.mean().item() if len(later) > 0 else 0

    tc_scores /= len(ATTENTION_LAYERS)

    del reasoning_out, prompt_cache
    gc.collect(); torch.cuda.empty_cache()
    return tc_scores


@torch.no_grad()
def evaluate_condition(model, tokenizer, prompt_cache, reasoning_tokens,
                       positions_to_noise, prompt_len, num_layers, true_answer):
    """
    Step through reasoning tokens one at a time, noising selected positions.
    Returns answer accuracy and text prediction accuracy.
    """
    reasoning_len = reasoning_tokens.shape[1]
    noise_set = set(positions_to_noise)

    # Clone prompt cache
    cache = DynamicCache()
    for l in range(num_layers):
        k = prompt_cache.layers[l].keys.clone()
        v = prompt_cache.layers[l].values.clone()
        cache.update(k, v, l)

    text_correct = 0
    text_total = 0

    for i in range(reasoning_len):
        token = reasoning_tokens[:, i:i+1]
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

        # Noise this position if selected
        if i in noise_set:
            for l in range(num_layers):
                pos = prompt_len + i
                k = cache.layers[l].keys
                v = cache.layers[l].values
                k_s = k[:, :, pos:pos+1, :]
                k_n = torch.randn_like(k_s)
                k_n = k_n * (k_s.norm() / (k_n.norm() + 1e-8))
                k[:, :, pos:pos+1, :] = k_n
                v_s = v[:, :, pos:pos+1, :]
                v_n = torch.randn_like(v_s)
                v_n = v_n * (v_s.norm() / (v_n.norm() + 1e-8))
                v[:, :, pos:pos+1, :] = v_n

        # Check prediction for next token
        if i < reasoning_len - 1:
            predicted = out.logits[:, -1, :].argmax(dim=-1).item()
            actual = reasoning_tokens[:, i + 1].item()
            if predicted == actual:
                text_correct += 1
            text_total += 1

        del out

    text_accuracy = text_correct / text_total if text_total > 0 else 0.0

    # Generate answer
    last_tok = reasoning_tokens[:, -1:]
    gen_out = model(input_ids=last_tok, past_key_values=cache, use_cache=True)
    gen_cache = gen_out.past_key_values
    next_tok = gen_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_tok[0, 0].item()]
    del gen_out

    for _ in range(80):
        g = model(input_ids=next_tok, past_key_values=gen_cache, use_cache=True)
        gen_cache = g.past_key_values
        next_tok = g.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tid = next_tok[0, 0].item()
        generated.append(tid)
        if tid == tokenizer.eos_token_id:
            break
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if "####" in decoded and re.search(r'\d+', decoded.split("####")[-1]):
            break
        if "\nQ:" in decoded:
            break
        del g

    answer_text = tokenizer.decode(generated, skip_special_tokens=True)
    answer = normalize_answer(extract_answer(answer_text)) if extract_answer(answer_text) else ""
    correct = (answer == true_answer)

    del cache, gen_cache
    gc.collect(); torch.cuda.empty_cache()

    return {
        'correct': correct,
        'answer': answer,
        'text_accuracy': text_accuracy,
        'answer_text': answer_text[:150],
    }


def select_positions(ac_scores, tc_scores, reasoning_len, noise_frac, strategy):
    """Select positions to noise based on strategy."""
    n = max(1, int(reasoning_len * noise_frac))

    if strategy == 'random':
        return sorted(random.sample(range(reasoning_len), min(n, reasoning_len)))
    if strategy == 'pos_early':
        return list(range(min(n, reasoning_len)))
    if strategy == 'pos_late':
        return list(range(max(0, reasoning_len - n), reasoning_len))

    # Selectivity scores
    ac_ranks = stats.rankdata(ac_scores)
    tc_ranks = stats.rankdata(tc_scores)
    sel = ac_ranks - tc_ranks

    positions = np.arange(reasoning_len) / max(reasoning_len - 1, 1)
    q_bounds = np.percentile(positions, [25, 50, 75])
    q_bins = np.digitize(positions, q_bounds)  # 0,1,2,3

    if strategy == 'selac':
        return sorted(np.argsort(-sel)[:n].tolist())
    elif strategy == 'seltc':
        return sorted(np.argsort(sel)[:n].tolist())

    # Quartile-controlled strategies
    parts = strategy.split('_')
    if len(parts) == 2 and parts[0].startswith('q'):
        q = int(parts[0][1]) - 1
        direction = parts[1]  # selac or seltc
        mask = (q_bins == q)
        q_idx = np.where(mask)[0]
        if len(q_idx) == 0:
            return []
        q_sel = sel[q_idx]
        n_q = min(n, len(q_idx))
        if direction == 'selac':
            top = np.argsort(-q_sel)[:n_q]
        else:
            top = np.argsort(q_sel)[:n_q]
        return sorted(q_idx[top].tolist())

    return []


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_figures(results, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = ['selac', 'seltc', 'random', 'q3_selac', 'q3_seltc', 'q4_selac', 'q4_seltc']
    labels = ['SelAC', 'SelTC', 'Random', 'Q3+AC', 'Q3+TC', 'Q4+AC', 'Q4+TC']
    colors_acc = '#e74c3c'
    colors_txt = '#3498db'

    for nf in NOISE_FRACTIONS:
        nfk = f"{int(nf*100)}pct"
        acc_vals, txt_vals, ns = [], [], []
        for strat in strategies:
            key = f"{strat}_{nfk}"
            accs = [1 if p['evaluations'][key]['correct'] else 0
                    for p in results if key in p.get('evaluations', {})]
            txts = [p['evaluations'][key]['text_accuracy']
                    for p in results if key in p.get('evaluations', {})]
            acc_vals.append(np.mean(accs) * 100 if accs else 0)
            txt_vals.append(np.mean(txts) * 100 if txts else 0)
            ns.append(len(accs))

        if not any(ns):
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(strategies))
        w = 0.4

        bars1 = ax1.bar(x, acc_vals, w, color=colors_acc, alpha=0.8)
        ax1.set_ylabel('Answer Accuracy (%)')
        ax1.set_title(f'Answer Accuracy at {int(nf*100)}% Noise (n={max(ns)})')
        ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylim(0, 105); ax1.grid(True, alpha=0.2, axis='y')
        for b, v in zip(bars1, acc_vals):
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.0f}%',
                    ha='center', va='bottom', fontsize=9)

        bars2 = ax2.bar(x, txt_vals, w, color=colors_txt, alpha=0.8)
        ax2.set_ylabel('Text Prediction Accuracy (%)')
        ax2.set_title(f'Text Quality at {int(nf*100)}% Noise')
        ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylim(0, 105); ax2.grid(True, alpha=0.2, axis='y')
        for b, v in zip(bars2, txt_vals):
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.0f}%',
                    ha='center', va='bottom', fontsize=9)

        plt.suptitle(f'Double Dissociation at {int(nf*100)}% Noise')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'double_dissociation_{int(nf*100)}pct.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    # Dissociation score figure
    fig, ax = plt.subplots(figsize=(10, 6))
    for ni, nf in enumerate(NOISE_FRACTIONS):
        nfk = f"{int(nf*100)}pct"
        dissoc = []
        for strat in strategies:
            key = f"{strat}_{nfk}"
            accs = [1 if p['evaluations'][key]['correct'] else 0
                    for p in results if key in p.get('evaluations', {})]
            txts = [p['evaluations'][key]['text_accuracy']
                    for p in results if key in p.get('evaluations', {})]
            clean_txts = [p['clean_text_acc'] for p in results if key in p.get('evaluations', {})]
            if accs:
                acc_drop = 1.0 - np.mean(accs)
                txt_drop = np.mean(clean_txts) - np.mean(txts)
                dissoc.append(acc_drop - txt_drop)
            else:
                dissoc.append(0)

        offset = (ni - 0.5) * 0.3
        ax.bar(np.arange(len(strategies)) + offset, [d*100 for d in dissoc], 0.25,
              label=f'{int(nf*100)}%', alpha=0.8)

    ax.set_ylabel('Dissociation Score (%)\n(+)=accuracy hurt more  (-)=text hurt more')
    ax.set_title('Functional Dissociation Score')
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.legend(); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dissociation_scores.png'),
               dpi=150, bbox_inches='tight')
    plt.close()


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers")

    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:NUM_PROBLEMS]

    strategies = ['selac', 'seltc', 'random', 'q3_selac', 'q3_seltc', 'q4_selac', 'q4_seltc']
    results = []
    n_valid = 0

    for pi, ds_idx in enumerate(indices):
        elapsed = time.time() - t0
        if elapsed > 1400:
            print(f"\n⏰ Time limit at problem {pi}/{len(indices)}")
            break

        item = ds[ds_idx]
        true_answer = normalize_answer(extract_answer(item['answer']))
        prompt = build_prompt(item['question'])

        print(f"\nProblem {pi+1}/{len(indices)} (idx={ds_idx}), "
              f"true={true_answer}, t={elapsed:.0f}s")

        trace = generate_trace(model, tokenizer, prompt)
        gen_answer = normalize_answer(extract_answer(trace)) if extract_answer(trace) else ""
        if gen_answer != true_answer:
            print(f"  WRONG: {gen_answer}")
            continue
        n_valid += 1

        full_text = prompt + trace
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_len = prompt_ids.shape[1]
        reasoning_len = full_ids.shape[1] - prompt_len

        if reasoning_len < 20 or full_ids.shape[1] > MAX_SEQ_LEN:
            print(f"  Skip: R={reasoning_len}, total={full_ids.shape[1]}")
            continue

        # Compute AC and TC scores
        print(f"  R={reasoning_len} tokens. Computing scores...")
        ac_scores = compute_ac_scores(model, full_ids, prompt_len, reasoning_len, num_layers)
        tc_scores = compute_tc_scores_fast(model, full_ids, prompt_len, reasoning_len, num_layers)

        # Build prompt cache (shared across conditions)
        prompt_cache = build_prompt_cache(model, prompt_ids, num_layers)

        # Clean text accuracy baseline
        clean_eval = evaluate_condition(
            model, tokenizer, prompt_cache, full_ids[:, prompt_len:prompt_len+reasoning_len],
            [], prompt_len, num_layers, true_answer)
        clean_text_acc = clean_eval['text_accuracy']
        print(f"  Clean: acc={'✓' if clean_eval['correct'] else '✗'}, text={clean_text_acc:.1%}")

        # Rebuild prompt cache (was consumed)
        prompt_cache = build_prompt_cache(model, prompt_ids, num_layers)

        problem_result = {
            'ds_idx': ds_idx,
            'true_answer': true_answer,
            'reasoning_len': reasoning_len,
            'clean_text_acc': clean_text_acc,
            'clean_correct': clean_eval['correct'],
            'evaluations': {},
        }

        reasoning_tokens = full_ids[:, prompt_len:prompt_len + reasoning_len]

        for nf in NOISE_FRACTIONS:
            for strat in strategies:
                key = f"{strat}_{int(nf*100)}pct"
                positions = select_positions(ac_scores, tc_scores, reasoning_len, nf, strat)
                if not positions:
                    continue

                mean_pos = np.mean([p / max(reasoning_len-1, 1) for p in positions])

                # Rebuild prompt cache for each condition
                pc = build_prompt_cache(model, prompt_ids, num_layers)
                ev = evaluate_condition(
                    model, tokenizer, pc, reasoning_tokens,
                    positions, prompt_len, num_layers, true_answer)

                print(f"    {key}: acc={'✓' if ev['correct'] else '✗'}, "
                      f"text={ev['text_accuracy']:.1%}, pos={mean_pos:.3f}")

                problem_result['evaluations'][key] = {
                    'correct': ev['correct'],
                    'answer': ev['answer'],
                    'text_accuracy': ev['text_accuracy'],
                    'mean_pos': float(mean_pos),
                    'n_noised': len(positions),
                }

        results.append(problem_result)
        del full_ids, prompt_ids, reasoning_tokens, prompt_cache
        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_valid} valid, {len(results)} processed")

    summary = {
        'experiment': 'exp_019_double_dissociation',
        'model': MODEL_NAME,
        'n_valid': n_valid,
        'n_processed': len(results),
        'noise_fractions': NOISE_FRACTIONS,
        'strategies': strategies,
    }

    agg = {}
    for nf in NOISE_FRACTIONS:
        nfk = f"{int(nf*100)}pct"
        print(f"\n── {int(nf*100)}% Noise ──")
        print(f"  {'Strategy':<15} {'Acc%':>6} {'Text%':>6} {'AccDrop':>8} {'TxtDrop':>8} {'Dissoc':>8} {'MeanPos':>8} {'n':>3}")
        for strat in strategies:
            key = f"{strat}_{nfk}"
            accs = [1 if p['evaluations'][key]['correct'] else 0
                    for p in results if key in p.get('evaluations', {})]
            txts = [p['evaluations'][key]['text_accuracy']
                    for p in results if key in p.get('evaluations', {})]
            pos = [p['evaluations'][key]['mean_pos']
                   for p in results if key in p.get('evaluations', {})]
            ctxts = [p['clean_text_acc'] for p in results if key in p.get('evaluations', {})]
            if accs:
                a_mean = np.mean(accs)
                t_mean = np.mean(txts)
                ct_mean = np.mean(ctxts)
                a_drop = 1.0 - a_mean
                t_drop = ct_mean - t_mean
                dissoc = a_drop - t_drop
                agg[key] = {
                    'n': len(accs), 'accuracy': float(a_mean),
                    'text_accuracy': float(t_mean),
                    'clean_text_accuracy': float(ct_mean),
                    'accuracy_drop': float(a_drop), 'text_drop': float(t_drop),
                    'dissociation': float(dissoc), 'mean_position': float(np.mean(pos)),
                }
                print(f"  {strat:<15} {a_mean*100:>5.1f}% {t_mean*100:>5.1f}% "
                      f"{a_drop*100:>7.1f}% {t_drop*100:>7.1f}% "
                      f"{dissoc*100:>7.1f}% {np.mean(pos):>7.3f} {len(accs):>3}")

    summary['aggregated'] = agg

    # Key dissociation tests
    print("\n── DOUBLE DISSOCIATION TESTS ──")
    for nf in NOISE_FRACTIONS:
        nfk = f"{int(nf*100)}pct"
        for q_label, qa, qt in [('Q3', f'q3_selac_{nfk}', f'q3_seltc_{nfk}'),
                                 ('Q4', f'q4_selac_{nfk}', f'q4_seltc_{nfk}'),
                                 ('All', f'selac_{nfk}', f'seltc_{nfk}')]:
            if qa in agg and qt in agg:
                a_ac, t_ac = agg[qa]['accuracy_drop'], agg[qa]['text_drop']
                a_tc, t_tc = agg[qt]['accuracy_drop'], agg[qt]['text_drop']
                print(f"\n  {q_label} at {int(nf*100)}%:")
                print(f"    AC-noise: acc_drop={a_ac*100:.1f}%, text_drop={t_ac*100:.1f}%, dissoc={agg[qa]['dissociation']*100:.1f}%")
                print(f"    TC-noise: acc_drop={a_tc*100:.1f}%, text_drop={t_tc*100:.1f}%, dissoc={agg[qt]['dissociation']*100:.1f}%")
                # Double dissociation: AC-noise has higher acc_drop AND lower text_drop than TC-noise
                dd = (a_ac > a_tc) and (t_ac < t_tc)
                print(f"    Double dissociation: {'YES' if dd else 'NO'} "
                      f"(acc_drop: AC{'>' if a_ac>a_tc else '<'}TC, "
                      f"text_drop: AC{'<' if t_ac<t_tc else '>'}TC)")

    summary['elapsed_seconds'] = time.time() - t0

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=_convert)
    with open(os.path.join(RESULTS_DIR, 'per_problem.json'), 'w') as f:
        json.dump(results, f, indent=2, default=_convert)

    print("\nGenerating figures...")
    generate_figures(results, RESULTS_DIR)
    print(f"\nDone. Total: {summary['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
