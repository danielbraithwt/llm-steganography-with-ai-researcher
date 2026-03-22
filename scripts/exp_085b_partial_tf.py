#!/usr/bin/env python3
"""
Experiment 085b: Partial Teacher-Forcing Paraphrase Disruption

Follow-up to exp_085. Full teacher-forcing trivially achieved 100% across all
conditions because the model copies the last number in the CoT. This test
teacher-forces only the FIRST HALF of the CoT and lets the model GENERATE
the remaining computation + answer.

If non-number token K-routing carries computation state, paraphrasing the
prefix should disrupt the model's ability to generate correct continuations.
If the model reads only numbers, paraphrased prefix should work fine.
"""

import torch
import json
import os
import re
import random
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'scripts')
from exp_085_paraphrase_disruption import (
    paraphrase_cot, extract_gold_answer, extract_cot_and_answer,
    parse_answer, is_number_or_math
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "Qwen/Qwen3-4B-Base"
N_PROBLEMS = 200
DEVICE = "cuda"
RESULTS_DIR = "results/exp_085"


def find_line_split(text, target_frac=0.5):
    """Find the newline boundary closest to target_frac of text length."""
    target = int(len(text) * target_frac)
    newlines = [i for i, c in enumerate(text) if c == '\n']
    if not newlines:
        return target
    closest = min(newlines, key=lambda x: abs(x - target))
    return closest + 1  # split after newline


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXP 085b: Partial Teacher-Forcing Paraphrase Disruption")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GSM8K
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    test_data = ds['test']
    train_data = ds['train']

    # 8-shot prompt
    exemplars = [
        f"Question: {train_data[i]['question']}\nAnswer: {train_data[i]['answer']}"
        for i in range(8)
    ]
    prompt_prefix = "\n\n".join(exemplars) + "\n\n"

    # Phase 1: Generate CoT (deterministic, same as exp_085)
    print(f"\nPhase 1: Generate CoT for {N_PROBLEMS} problems")
    correct = []
    for idx in range(N_PROBLEMS):
        q = test_data[idx]['question']
        gold = extract_gold_answer(test_data[idx]['answer'])

        prompt = prompt_prefix + f"Question: {q}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=512,
                do_sample=False, pad_token_id=tokenizer.pad_token_id
            )

        gen_text = tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True)
        cot_body, pred = extract_cot_and_answer(gen_text)

        if pred == gold and cot_body and len(cot_body) > 10:
            correct.append({
                'idx': idx, 'question': q, 'gold': gold, 'cot_body': cot_body
            })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{N_PROBLEMS}] {len(correct)} correct")

    print(f"Generated: {len(correct)}/{N_PROBLEMS} correct")

    # Phase 2: Prepare prefixes
    print(f"\nPhase 2: Prepare prefixes for {len(correct)} problems")

    # Filter: need at least 2 lines in CoT for meaningful split
    valid = []
    for p in correct:
        lines = p['cot_body'].strip().split('\n')
        if len(lines) >= 2:
            valid.append(p)
    print(f"  {len(valid)} problems have ≥2 lines (out of {len(correct)})")

    for p in valid:
        split = find_line_split(p['cot_body'], 0.5)
        prefix = p['cot_body'][:split]
        suffix = p['cot_body'][split:]
        p['prefix'] = prefix
        p['suffix'] = suffix
        p['split_frac'] = split / len(p['cot_body']) if len(p['cot_body']) > 0 else 0

        # Count how many computation lines are in prefix vs suffix
        prefix_lines = [l for l in prefix.strip().split('\n') if l.strip()]
        suffix_lines = [l for l in suffix.strip().split('\n') if l.strip()]
        p['n_prefix_lines'] = len(prefix_lines)
        p['n_suffix_lines'] = len(suffix_lines)

        # Paraphrase prefix only
        rng_s = random.Random(SEED + p['idx'])
        syn_prefix, n_s, n_e = paraphrase_cot(
            prefix, rate=0.5, use_random=False, rng=rng_s)
        rng_r = random.Random(SEED + p['idx'] + 100000)
        rand_prefix, n_r, _ = paraphrase_cot(
            prefix, rate=0.5, use_random=True, rng=rng_r)

        p['syn_prefix'] = syn_prefix
        p['rand_prefix'] = rand_prefix
        p['n_syn'] = n_s
        p['n_rand'] = n_r
        p['n_eligible'] = n_e

    avg_split = np.mean([p['split_frac'] for p in valid])
    avg_pfx_lines = np.mean([p['n_prefix_lines'] for p in valid])
    avg_sfx_lines = np.mean([p['n_suffix_lines'] for p in valid])
    print(f"  Avg split: {avg_split:.1%} of CoT")
    print(f"  Avg prefix lines: {avg_pfx_lines:.1f}, suffix lines: {avg_sfx_lines:.1f}")

    # Show examples
    for i in range(min(2, len(valid))):
        p = valid[i]
        print(f"\n  --- Problem {p['idx']} (gold={p['gold']}, "
              f"split={p['split_frac']:.0%}) ---")
        print(f"  PREFIX: {p['prefix'][:150].replace(chr(10), ' | ')}...")
        print(f"  SUFFIX: {p['suffix'][:150].replace(chr(10), ' | ')}...")
        print(f"  SYN_PFX: {p['syn_prefix'][:150].replace(chr(10), ' | ')}...")

    # Phase 3: Generate from prefix (3 conditions)
    print(f"\n{'='*60}")
    print(f"Phase 3: Partial TF generation ({len(valid)} problems x 3 conditions)")
    print(f"{'='*60}")

    for cond, pkey in [
        ('original', 'prefix'),
        ('synonym', 'syn_prefix'),
        ('random', 'rand_prefix')
    ]:
        print(f"\n  Condition: {cond}")
        t_cond = time.time()

        for pi, p in enumerate(valid):
            pfx = p[pkey]
            tf_prompt = prompt_prefix + f"Question: {p['question']}\nAnswer: {pfx}"
            input_ids = tokenizer.encode(tf_prompt, return_tensors='pt').to(DEVICE)

            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=400,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id
                )

            gen = tokenizer.decode(
                out[0][input_ids.shape[1]:], skip_special_tokens=True)

            # Parse: look for #### in generated text
            _, pred = extract_cot_and_answer(gen)
            p[f'{cond}_gen'] = gen[:300]
            p[f'{cond}_ans'] = pred
            p[f'{cond}_ok'] = (pred == p['gold'])

            if (pi + 1) % 50 == 0:
                acc = sum(p2[f'{cond}_ok'] for p2 in valid[:pi+1]) / (pi + 1)
                print(f"    [{pi+1}/{len(valid)}] acc={acc:.1%}")

        acc = sum(p[f'{cond}_ok'] for p in valid) / len(valid)
        n_ok = sum(p[f'{cond}_ok'] for p in valid)
        print(f"  → {cond}: {acc:.1%} ({n_ok}/{len(valid)}) "
              f"[{time.time()-t_cond:.0f}s]")

    # Phase 4: Analysis
    print(f"\n{'='*60}")
    print("Phase 4: Analysis")
    print(f"{'='*60}")

    n = len(valid)
    acc_o = sum(p['original_ok'] for p in valid) / n
    acc_s = sum(p['synonym_ok'] for p in valid) / n
    acc_r = sum(p['random_ok'] for p in valid) / n

    print(f"\n  n = {n}")
    print(f"  Original prefix:  {acc_o:.1%} ({sum(p['original_ok'] for p in valid)}/{n})")
    print(f"  Synonym prefix:   {acc_s:.1%} (drop: {acc_o-acc_s:+.1%})")
    print(f"  Random prefix:    {acc_r:.1%} (drop: {acc_o-acc_r:+.1%})")

    # McNemar tests
    from scipy.stats import binomtest

    for label, cond in [('Synonym', 'synonym'), ('Random', 'random')]:
        list_a = [p['original_ok'] for p in valid]
        list_b = [p[f'{cond}_ok'] for p in valid]
        a_only = sum(a and not b for a, b in zip(list_a, list_b))
        b_only = sum(not a and b for a, b in zip(list_a, list_b))
        both = sum(a and b for a, b in zip(list_a, list_b))
        neither = sum(not a and not b for a, b in zip(list_a, list_b))
        disc = a_only + b_only
        pv = binomtest(a_only, disc, 0.5).pvalue if disc > 0 else 1.0
        print(f"\n  McNemar Original vs {label}:")
        print(f"    Both correct: {both}, Orig-only: {a_only}, "
              f"{label}-only: {b_only}, Both wrong: {neither}")
        print(f"    Discordant: {disc}, p = {pv:.4f}")

    # Bootstrap CIs
    n_boot = 2000
    np.random.seed(SEED)
    boot_s, boot_r = [], []
    for _ in range(n_boot):
        bi = np.random.choice(n, n, replace=True)
        bo = np.mean([valid[i]['original_ok'] for i in bi])
        bs = np.mean([valid[i]['synonym_ok'] for i in bi])
        br = np.mean([valid[i]['random_ok'] for i in bi])
        boot_s.append(bo - bs)
        boot_r.append(bo - br)

    ci_s = np.percentile(boot_s, [2.5, 97.5])
    ci_r = np.percentile(boot_r, [2.5, 97.5])
    print(f"\n  Bootstrap 95% CI:")
    print(f"    Synonym drop: [{ci_s[0]:+.1%}, {ci_s[1]:+.1%}]")
    print(f"    Random drop:  [{ci_r[0]:+.1%}, {ci_r[1]:+.1%}]")

    # Error analysis
    orig_wrong = [p for p in valid if not p['original_ok']]
    syn_wrong = [p for p in valid if p['original_ok'] and not p['synonym_ok']]
    rand_wrong = [p for p in valid if p['original_ok'] and not p['random_ok']]
    syn_gained = [p for p in valid if not p['original_ok'] and p['synonym_ok']]
    rand_gained = [p for p in valid if not p['original_ok'] and p['random_ok']]

    print(f"\n  Original wrong (baseline): {len(orig_wrong)}")
    print(f"  Synonym: lost {len(syn_wrong)}, gained {len(syn_gained)}")
    print(f"  Random:  lost {len(rand_wrong)}, gained {len(rand_gained)}")

    if syn_wrong:
        print(f"\n  Examples of synonym-broken problems:")
        for p in syn_wrong[:3]:
            print(f"    Problem {p['idx']}: gold={p['gold']}, "
                  f"orig_ans={p['original_ans']}, syn_ans={p['synonym_ans']}")
            print(f"      Orig gen: {p['original_gen'][:100]}...")
            print(f"      Syn gen:  {p['synonym_gen'][:100]}...")

    if rand_wrong and len(rand_wrong) > len(syn_wrong):
        print(f"\n  Examples of random-broken (not synonym-broken) problems:")
        extra = [p for p in rand_wrong if p['synonym_ok']]
        for p in extra[:3]:
            print(f"    Problem {p['idx']}: gold={p['gold']}, "
                  f"rand_ans={p['random_ans']}")
            print(f"      Rand gen: {p['random_gen'][:100]}...")

    # Figure: Accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Original\nprefix', 'Synonym\nprefix', 'Random\nprefix']
    accs_pct = [acc_o * 100, acc_s * 100, acc_r * 100]
    colors = ['#2196F3', '#FF9800', '#F44336']
    bars = ax.bar(labels, accs_pct, color=colors, width=0.5,
                  edgecolor='black', linewidth=0.5)
    for bar, a in zip(bars, accs_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{a:.1f}%', ha='center', va='bottom', fontsize=12,
                fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Exp 085b: Partial TF Paraphrase Disruption\n'
                 'Model generates remaining ~50% of CoT + answer',
                 fontsize=12)
    ax.set_ylim(0, max(accs_pct) * 1.2)
    ax.axhline(y=acc_o * 100, color='gray', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/partial_tf_accuracy.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"\n  Saved partial_tf_accuracy.png")

    # Save results
    results = {
        'n': n, 'avg_split_frac': float(avg_split),
        'avg_prefix_lines': float(avg_pfx_lines),
        'avg_suffix_lines': float(avg_sfx_lines),
        'acc_original': acc_o, 'acc_synonym': acc_s, 'acc_random': acc_r,
        'drop_synonym': acc_o - acc_s, 'drop_random': acc_o - acc_r,
        'ci_synonym': [float(ci_s[0]), float(ci_s[1])],
        'ci_random': [float(ci_r[0]), float(ci_r[1])],
        'n_orig_wrong': len(orig_wrong),
        'n_syn_broken': len(syn_wrong),
        'n_rand_broken': len(rand_wrong),
        'n_syn_gained': len(syn_gained),
        'n_rand_gained': len(rand_gained),
        'runtime_seconds': time.time() - t0,
    }
    with open(f'{RESULTS_DIR}/partial_tf_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"EXP 085b COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"  Original prefix: {acc_o:.1%}")
    print(f"  Synonym prefix:  {acc_s:.1%} (drop: {acc_o-acc_s:+.1%})")
    print(f"  Random prefix:   {acc_r:.1%} (drop: {acc_o-acc_r:+.1%})")


if __name__ == '__main__':
    main()
