#!/usr/bin/env python3
"""
Experiment 085: Paraphrase Disruption (Experiment B)

Tests whether specific token choices in CoT carry hidden information
beyond their semantic content.

Method:
- Generate CoT for GSM8K problems with Qwen3-4B-Base
- Create paraphrased versions (synonym replacement, preserving all numbers)
- Teacher-force original and paraphrased text, generate final answer
- Compare accuracy: original vs synonym vs random-replacement control

Three conditions:
1. ORIGINAL: Teacher-force model's own CoT text → baseline accuracy
2. SYNONYM: ~50% non-number words replaced with synonyms → test condition
3. RANDOM: ~50% non-number words replaced with random nouns → positive control
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

# ====================================================================
# Configuration
# ====================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "Qwen/Qwen3-4B-Base"
N_PROBLEMS = 200
MAX_GEN_TOKENS = 512
MAX_ANSWER_TOKENS = 30
REPLACEMENT_RATE = 0.5
DEVICE = "cuda"
RESULTS_DIR = "results/exp_085"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ====================================================================
# Synonym dictionary — single-word, grammatically safe replacements
# ====================================================================
SYNONYMS = {
    # Present tense verbs
    "has": "possesses", "have": "possess",
    "gets": "obtains", "get": "obtain",
    "buys": "purchases", "buy": "purchase",
    "sells": "offers", "sell": "offer",
    "gives": "provides", "give": "provide",
    "makes": "creates", "make": "create",
    "earns": "gains", "earn": "gain",
    "needs": "requires", "need": "require",
    "uses": "utilizes", "use": "utilize",
    "eats": "consumes", "eat": "consume",
    "saves": "reserves", "save": "reserve",
    "spends": "expends", "spend": "expend",
    "keeps": "retains", "keep": "retain",
    "wants": "desires", "want": "desire",
    "finds": "discovers", "find": "discover",
    "puts": "places", "put": "place",
    "works": "labors", "work": "labor",
    "grows": "increases", "grow": "increase",
    "picks": "selects", "pick": "select",
    "collects": "gathers", "collect": "gather",
    "receives": "obtains", "receive": "obtain",
    "builds": "constructs", "build": "construct",
    "starts": "begins", "start": "begin",
    "finishes": "completes", "finish": "complete",
    "adds": "combines", "add": "combine",
    "carries": "transports", "carry": "transport",
    "contains": "holds", "contain": "hold",
    "costs": "charges", "cost": "charge",
    "pays": "remits", "pay": "remit",
    "loses": "forfeits", "lose": "forfeit",
    "brings": "fetches", "bring": "fetch",
    "sends": "dispatches", "send": "dispatch",
    "helps": "assists", "help": "assist",
    "takes": "demands", "take": "demand",
    "covers": "wraps", "shares": "distributes",
    "fills": "loads", "cuts": "divides",
    "opens": "unlocks", "closes": "shuts",
    "counts": "tallies", "turns": "rotates",
    "raises": "elevates", "orders": "requests",
    "cleans": "scrubs", "fixes": "repairs",
    "mixes": "blends", "stores": "stocks",
    "splits": "separates", "joins": "combines",
    "plants": "sows", "paints": "coats",
    # Past tense
    "had": "possessed", "got": "obtained",
    "bought": "purchased", "sold": "offered",
    "gave": "provided", "made": "created",
    "earned": "gained", "needed": "required",
    "used": "utilized", "ate": "consumed",
    "saved": "reserved", "spent": "expended",
    "kept": "retained", "wanted": "desired",
    "found": "discovered", "worked": "labored",
    "started": "began", "finished": "completed",
    "added": "combined", "lost": "forfeited",
    "brought": "fetched", "sent": "dispatched",
    "built": "constructed", "helped": "assisted",
    "paid": "remitted", "collected": "gathered",
    "picked": "selected", "carried": "transported",
    "took": "demanded", "opened": "unlocked",
    "closed": "shut", "ordered": "requested",
    "cleaned": "scrubbed", "fixed": "repaired",
    "mixed": "blended", "stored": "stocked",
    "raised": "elevated", "turned": "rotated",
    "planted": "sowed", "painted": "coated",
    "covered": "wrapped", "filled": "loaded",
    "shared": "distributed", "joined": "combined",
    # Gerunds
    "making": "creating", "earning": "gaining",
    "buying": "purchasing", "selling": "offering",
    "using": "utilizing", "working": "laboring",
    "building": "constructing", "getting": "obtaining",
    "giving": "providing", "spending": "expending",
    "saving": "reserving", "keeping": "retaining",
    "adding": "combining", "collecting": "gathering",
    "carrying": "transporting", "counting": "tallying",
    # Connectives & adverbs
    "so": "therefore", "then": "next",
    "thus": "consequently", "also": "additionally",
    "already": "previously", "now": "currently",
    "still": "yet", "finally": "ultimately",
    "usually": "typically", "often": "frequently",
    "always": "consistently", "however": "nevertheless",
    "because": "since", "although": "though",
    # Adjectives & determiners
    "each": "every", "every": "each",
    "total": "combined", "remaining": "leftover",
    "new": "fresh", "big": "large",
    "small": "little", "first": "initial",
    "last": "final", "next": "following",
    "many": "numerous", "few": "several",
    "whole": "entire", "same": "identical",
    "extra": "additional", "full": "complete",
    "different": "distinct", "enough": "sufficient",
    "various": "multiple", "only": "solely",
    # Nouns (safe generic replacements)
    "money": "funds", "price": "cost",
    "amount": "quantity", "pieces": "units",
    "people": "individuals", "children": "kids",
    "students": "pupils", "box": "container",
    "bag": "sack", "store": "shop",
    "trip": "journey", "game": "match",
    "group": "collection", "rest": "remainder",
    "things": "items", "thing": "item",
    "place": "location", "part": "portion",
    "kind": "type", "job": "task",
}

# Random words for the control condition (clearly non-mathematical)
RANDOM_POOL = [
    "quantum", "glacier", "compass", "melody", "crystal", "phantom",
    "ember", "summit", "prism", "zenith", "anchor", "beacon",
    "comet", "dagger", "falcon", "goblet", "herald", "ivory",
    "jewel", "karma", "lantern", "mosaic", "nebula", "orchid",
    "plume", "quartz", "raven", "scepter", "thorn", "vortex",
    "willow", "aurora", "bronze", "chalice", "delta", "eclipse",
    "fossil", "granite", "haven", "iris", "jasper", "kite",
    "lotus", "magnet", "nimbus", "oasis", "pearl", "ridge",
]


# ====================================================================
# Helper functions
# ====================================================================
def is_number_or_math(word):
    """Check if word is a number, operator, or math symbol."""
    clean = word.strip(".,;:!?()[]{}\"'$")
    if not clean:
        return True  # pure punctuation — preserve
    if any(c.isdigit() for c in clean):
        return True
    if clean in {'+', '-', '*', '/', '=', '%', '$', '####', '>>', '<<', '**'}:
        return True
    return False


def paraphrase_cot(text, rate=0.5, use_random=False, rng=None):
    """
    Paraphrase a CoT body by replacing non-number words.
    Returns: (paraphrased_text, n_replaced, n_eligible)
    """
    if rng is None:
        rng = random.Random(SEED)

    lines = text.split("\n")
    new_lines = []
    total_replaced = 0
    total_eligible = 0

    for line in lines:
        words = line.split()
        result = []

        for word in words:
            if is_number_or_math(word):
                result.append(word)
                continue

            # Strip leading/trailing punctuation to get core word
            leading = ""
            trailing = ""
            core = word

            while core and core[0] in ".,;:!?()[]{}\"'":
                leading += core[0]
                core = core[1:]

            while core and core[-1] in ".,;:!?()[]{}\"'":
                trailing = core[-1] + trailing
                core = core[:-1]

            if not core or len(core) <= 1:
                result.append(word)
                continue

            total_eligible += 1

            # Probabilistic replacement
            if rng.random() > rate:
                result.append(word)
                continue

            core_lower = core.lower()

            if use_random:
                replacement = rng.choice(RANDOM_POOL)
            else:
                replacement = SYNONYMS.get(core_lower, None)
                if replacement is None:
                    result.append(word)
                    continue

            # Preserve capitalization
            if core[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]

            result.append(leading + replacement + trailing)
            total_replaced += 1

        new_lines.append(" ".join(result))

    return "\n".join(new_lines), total_replaced, total_eligible


def extract_gold_answer(answer_text):
    """Extract numeric answer from GSM8K answer field."""
    match = re.search(r'####\s*([\-]?[\d,]+)', answer_text)
    if match:
        return int(match.group(1).replace(',', ''))
    return None


def extract_cot_and_answer(generated_text):
    """Split generated text into CoT body and answer."""
    idx = generated_text.find("####")
    if idx >= 0:
        cot_body = generated_text[:idx].rstrip()
        answer_str = generated_text[idx + 4:].strip()
        match = re.search(r'([\-]?\d[\d,]*)', answer_str)
        if match:
            try:
                return cot_body, int(match.group(1).replace(',', ''))
            except ValueError:
                return cot_body, None
    return generated_text.rstrip(), None


def parse_answer(text):
    """Parse numeric answer from generated text."""
    text = text.strip()
    match = re.match(r'\s*([\-]?\d[\d,]*)', text)
    if match:
        try:
            return int(match.group(1).replace(',', ''))
        except ValueError:
            return None
    return None


def verify_numbers(original, modified):
    """Check if all numbers in original appear in modified in same order."""
    orig_nums = re.findall(r'\d+', original)
    mod_nums = re.findall(r'\d+', modified)
    return orig_nums == mod_nums


def token_change_frac(orig, modified, tokenizer):
    """Compute fraction of tokens that differ between original and modified."""
    t1 = tokenizer.encode(orig)
    t2 = tokenizer.encode(modified)
    if not t1:
        return 0.0
    min_len = min(len(t1), len(t2))
    diff = sum(1 for a, b in zip(t1[:min_len], t2[:min_len]) if a != b)
    diff += abs(len(t1) - len(t2))
    return diff / max(len(t1), len(t2))


# ====================================================================
# Main experiment
# ====================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("EXPERIMENT 085: Paraphrase Disruption (Experiment B)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"N problems: {N_PROBLEMS}")
    print(f"Replacement rate: {REPLACEMENT_RATE}")
    print(f"Synonym dictionary: {len(SYNONYMS)} entries")

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on {next(model.parameters()).device}")

    # ----------------------------------------------------------------
    # Load GSM8K
    # ----------------------------------------------------------------
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    test_data = ds['test']
    train_data = ds['train']

    # Build 8-shot prompt from training examples
    exemplars = []
    for i in range(8):
        exemplars.append(
            f"Question: {train_data[i]['question']}\nAnswer: {train_data[i]['answer']}"
        )
    prompt_prefix = "\n\n".join(exemplars) + "\n\n"
    prefix_len = len(tokenizer.encode(prompt_prefix))
    print(f"8-shot prompt: {prefix_len} tokens")

    # ----------------------------------------------------------------
    # PHASE 1: Generate CoT
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 1: Generate CoT for {N_PROBLEMS} problems")
    print(f"{'='*60}")

    correct = []
    n_total = 0

    for idx in range(N_PROBLEMS):
        q = test_data[idx]['question']
        gold = extract_gold_answer(test_data[idx]['answer'])

        prompt = prompt_prefix + f"Question: {q}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=MAX_GEN_TOKENS,
                do_sample=False, pad_token_id=tokenizer.pad_token_id
            )

        gen_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        cot_body, pred = extract_cot_and_answer(gen_text)
        is_correct = pred is not None and gold is not None and pred == gold
        n_total += 1

        if is_correct and cot_body and len(cot_body) > 10:
            correct.append({
                'idx': idx, 'question': q, 'gold': gold,
                'cot_body': cot_body, 'generated': gen_text,
            })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{N_PROBLEMS}] {len(correct)}/{n_total} correct "
                  f"({len(correct)/n_total:.0%})")

    gen_acc = len(correct) / n_total
    print(f"\nGeneration complete: {len(correct)}/{n_total} ({gen_acc:.0%})")

    # ----------------------------------------------------------------
    # PHASE 2: Create paraphrases
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 2: Create paraphrases for {len(correct)} problems")
    print(f"{'='*60}")

    for p in correct:
        rng_syn = random.Random(SEED + p['idx'])
        rng_rand = random.Random(SEED + p['idx'] + 100000)

        syn_text, n_syn, n_elig = paraphrase_cot(
            p['cot_body'], rate=REPLACEMENT_RATE, use_random=False, rng=rng_syn)
        rand_text, n_rand, _ = paraphrase_cot(
            p['cot_body'], rate=REPLACEMENT_RATE, use_random=True, rng=rng_rand)

        p['synonym_cot'] = syn_text
        p['random_cot'] = rand_text
        p['n_syn'] = n_syn
        p['n_rand'] = n_rand
        p['n_eligible'] = n_elig
        p['syn_frac'] = n_syn / n_elig if n_elig > 0 else 0
        p['rand_frac'] = n_rand / n_elig if n_elig > 0 else 0
        p['nums_ok_syn'] = verify_numbers(p['cot_body'], syn_text)
        p['nums_ok_rand'] = verify_numbers(p['cot_body'], rand_text)

    syn_ok = sum(p['nums_ok_syn'] for p in correct)
    rand_ok = sum(p['nums_ok_rand'] for p in correct)
    avg_syn_frac = np.mean([p['syn_frac'] for p in correct])
    avg_rand_frac = np.mean([p['rand_frac'] for p in correct])
    avg_elig = np.mean([p['n_eligible'] for p in correct])
    avg_n_syn = np.mean([p['n_syn'] for p in correct])
    avg_n_rand = np.mean([p['n_rand'] for p in correct])

    print(f"Numbers preserved: synonym={syn_ok}/{len(correct)}, random={rand_ok}/{len(correct)}")
    print(f"Avg eligible words: {avg_elig:.1f}")
    print(f"Avg replacements: synonym={avg_n_syn:.1f} ({avg_syn_frac:.1%}), "
          f"random={avg_n_rand:.1f} ({avg_rand_frac:.1%})")

    # Print example paraphrases
    for i in range(min(3, len(correct))):
        p = correct[i]
        print(f"\n--- Problem {p['idx']} (gold={p['gold']}) ---")
        orig_preview = p['cot_body'][:200].replace('\n', ' | ')
        syn_preview = p['synonym_cot'][:200].replace('\n', ' | ')
        rand_preview = p['random_cot'][:200].replace('\n', ' | ')
        print(f"  ORIG: {orig_preview}...")
        print(f"  SYN:  {syn_preview}...")
        print(f"  RAND: {rand_preview}...")
        print(f"  syn_replaced={p['n_syn']}/{p['n_eligible']}, "
              f"rand_replaced={p['n_rand']}/{p['n_eligible']}, "
              f"nums_ok={p['nums_ok_syn']},{p['nums_ok_rand']}")

    # ----------------------------------------------------------------
    # PHASE 3: Teacher-force and generate answers
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 3: Teacher-force + generate ({len(correct)} problems x 3 conditions)")
    print(f"{'='*60}")

    conditions = [
        ('original', 'cot_body'),
        ('synonym', 'synonym_cot'),
        ('random', 'random_cot'),
    ]

    for cond_name, cot_key in conditions:
        print(f"\n  Condition: {cond_name}")
        t_cond = time.time()

        for pi, p in enumerate(correct):
            cot_text = p[cot_key]

            # Build teacher-forced input
            tf_prompt = (prompt_prefix
                         + f"Question: {p['question']}\nAnswer: {cot_text}\n#### ")
            input_ids = tokenizer.encode(tf_prompt, return_tensors='pt').to(DEVICE)

            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=MAX_ANSWER_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id
                )

            ans_text = tokenizer.decode(
                out[0][input_ids.shape[1]:], skip_special_tokens=True)
            pred = parse_answer(ans_text)

            p[f'{cond_name}_ans'] = pred
            p[f'{cond_name}_ans_text'] = ans_text
            p[f'{cond_name}_ok'] = (pred == p['gold'])

            if (pi + 1) % 50 == 0:
                acc = sum(p2[f'{cond_name}_ok'] for p2 in correct[:pi+1]) / (pi + 1)
                print(f"    [{pi+1}/{len(correct)}] acc={acc:.1%}")

        cond_acc = sum(p[f'{cond_name}_ok'] for p in correct) / len(correct)
        print(f"  → {cond_name}: {cond_acc:.1%} "
              f"({sum(p[f'{cond_name}_ok'] for p in correct)}/{len(correct)}) "
              f"[{time.time()-t_cond:.0f}s]")

    # ----------------------------------------------------------------
    # PHASE 4: Analysis
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 4: Analysis")
    print(f"{'='*60}")

    n = len(correct)
    acc_o = sum(p['original_ok'] for p in correct) / n
    acc_s = sum(p['synonym_ok'] for p in correct) / n
    acc_r = sum(p['random_ok'] for p in correct) / n
    drop_s = acc_o - acc_s
    drop_r = acc_o - acc_r

    print(f"\n  n = {n} problems")
    print(f"  Original:  {acc_o:.1%} ({sum(p['original_ok'] for p in correct)}/{n})")
    print(f"  Synonym:   {acc_s:.1%} (drop: {drop_s:+.1%})")
    print(f"  Random:    {acc_r:.1%} (drop: {drop_r:+.1%})")

    # McNemar tests
    from scipy import stats as sp_stats

    def mcnemar_test(list_a, list_b, label):
        both = sum(a and b for a, b in zip(list_a, list_b))
        a_only = sum(a and not b for a, b in zip(list_a, list_b))
        b_only = sum(not a and b for a, b in zip(list_a, list_b))
        neither = sum(not a and not b for a, b in zip(list_a, list_b))
        disc = a_only + b_only
        if disc > 0:
            p_val = sp_stats.binomtest(a_only, disc, 0.5).pvalue
        else:
            p_val = 1.0
        print(f"\n  McNemar ({label}):")
        print(f"    Both correct: {both}, A-only: {a_only}, "
              f"B-only: {b_only}, Neither: {neither}")
        print(f"    Discordant: {disc}, p = {p_val:.4f}")
        return {
            'both': both, 'a_only': a_only, 'b_only': b_only,
            'neither': neither, 'p': float(p_val)
        }

    orig_list = [p['original_ok'] for p in correct]
    syn_list = [p['synonym_ok'] for p in correct]
    rand_list = [p['random_ok'] for p in correct]

    mc_os = mcnemar_test(orig_list, syn_list, "Original vs Synonym")
    mc_or = mcnemar_test(orig_list, rand_list, "Original vs Random")
    mc_sr = mcnemar_test(syn_list, rand_list, "Synonym vs Random")

    # Bootstrap CIs
    n_boot = 2000
    np.random.seed(SEED)
    boot_drops_s = []
    boot_drops_r = []
    for _ in range(n_boot):
        bi = np.random.choice(n, n, replace=True)
        bo = np.mean([correct[i]['original_ok'] for i in bi])
        bs = np.mean([correct[i]['synonym_ok'] for i in bi])
        br = np.mean([correct[i]['random_ok'] for i in bi])
        boot_drops_s.append(bo - bs)
        boot_drops_r.append(bo - br)

    ci_s = np.percentile(boot_drops_s, [2.5, 97.5])
    ci_r = np.percentile(boot_drops_r, [2.5, 97.5])
    p_boot_s = np.mean([d <= 0 for d in boot_drops_s])
    p_boot_r = np.mean([d <= 0 for d in boot_drops_r])

    print(f"\n  Bootstrap 95% CI (n={n_boot}):")
    print(f"    Synonym drop: [{ci_s[0]:+.1%}, {ci_s[1]:+.1%}], P(drop≤0)={p_boot_s:.3f}")
    print(f"    Random drop:  [{ci_r[0]:+.1%}, {ci_r[1]:+.1%}], P(drop≤0)={p_boot_r:.3f}")

    # Dose-response: per-problem replacement fraction vs correctness
    syn_fracs = np.array([p['syn_frac'] for p in correct])
    syn_ok_arr = np.array([int(p['synonym_ok']) for p in correct])
    if np.std(syn_fracs) > 0:
        r_dose, p_dose = sp_stats.pointbiserialr(syn_ok_arr, syn_fracs)
        print(f"\n  Dose-response (synonym): r_pb = {r_dose:.3f}, p = {p_dose:.4f}")
    else:
        r_dose, p_dose = 0.0, 1.0
        print(f"\n  Dose-response: no variance in replacement fraction")

    # Text similarity metrics
    from difflib import SequenceMatcher
    syn_sims = [SequenceMatcher(None, p['cot_body'], p['synonym_cot']).ratio()
                for p in correct]
    rand_sims = [SequenceMatcher(None, p['cot_body'], p['random_cot']).ratio()
                 for p in correct]

    print(f"\n  Text similarity (SequenceMatcher):")
    print(f"    Synonym: {np.mean(syn_sims):.3f} ± {np.std(syn_sims):.3f}")
    print(f"    Random:  {np.mean(rand_sims):.3f} ± {np.std(rand_sims):.3f}")

    # Token-level change
    syn_tok_changes = [token_change_frac(p['cot_body'], p['synonym_cot'], tokenizer)
                       for p in correct]
    rand_tok_changes = [token_change_frac(p['cot_body'], p['random_cot'], tokenizer)
                        for p in correct]

    print(f"\n  Token change fraction:")
    print(f"    Synonym: {np.mean(syn_tok_changes):.3f} ± {np.std(syn_tok_changes):.3f}")
    print(f"    Random:  {np.mean(rand_tok_changes):.3f} ± {np.std(rand_tok_changes):.3f}")

    # Error analysis: what answers does synonym produce when wrong?
    syn_wrong = [p for p in correct if p['original_ok'] and not p['synonym_ok']]
    if syn_wrong:
        print(f"\n  Synonym errors (original correct, synonym wrong): {len(syn_wrong)}")
        for p in syn_wrong[:5]:
            print(f"    Problem {p['idx']}: gold={p['gold']}, "
                  f"synonym_ans={p['synonym_ans']}, "
                  f"ans_text='{p['synonym_ans_text'][:40]}'")

    # Check: does original TF faithfully reproduce generation?
    orig_mismatch = sum(1 for p in correct if not p['original_ok'])
    print(f"\n  Original TF mismatches (sanity check): {orig_mismatch}/{n} "
          f"({orig_mismatch/n:.1%})")

    # ----------------------------------------------------------------
    # PHASE 5: Figures
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 5: Figures")
    print(f"{'='*60}")

    # Figure 1: Accuracy bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Original\n(teacher-forced)', 'Synonym\nparaphrase', 'Random\nreplacement']
    accs_pct = [acc_o * 100, acc_s * 100, acc_r * 100]
    colors = ['#2196F3', '#FF9800', '#F44336']
    bars = ax.bar(labels, accs_pct, color=colors, width=0.5,
                  edgecolor='black', linewidth=0.5)
    for bar, a in zip(bars, accs_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{a:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Exp 085: Paraphrase Disruption\n'
                 f'Qwen3-4B-Base, {REPLACEMENT_RATE:.0%} word replacement rate',
                 fontsize=12)
    ax.set_ylim(0, max(accs_pct) * 1.2)
    ax.axhline(y=acc_o * 100, color='gray', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved accuracy_comparison.png")

    # Figure 2: McNemar contingency tables
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mc, title, cA, cB in [
        (axes[0], mc_os, 'Original vs Synonym', 'Orig-only', 'Syn-only'),
        (axes[1], mc_or, 'Original vs Random', 'Orig-only', 'Rand-only'),
    ]:
        cats = ['Both\ncorrect', cA, cB, 'Both\nwrong']
        vals = [mc['both'], mc['a_only'], mc['b_only'], mc['neither']]
        cs = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
        ax.bar(cats, vals, color=cs, edgecolor='black', linewidth=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.5, str(v), ha='center', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\n(McNemar p = {mc["p"]:.4f})', fontsize=11)
        ax.set_ylabel('Problems')
    plt.suptitle('Per-Problem Agreement', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/mcnemar_contingency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved mcnemar_contingency.png")

    # Figure 3: Bootstrap distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(np.array(boot_drops_s) * 100, bins=40, color='#FF9800',
            alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=drop_s * 100, color='red', linewidth=2,
               label=f'Observed: {drop_s:+.1%}')
    ax.axvline(x=10, color='green', linestyle=':', linewidth=1.5,
               label='Threshold: 10%')
    ax.set_xlabel('Accuracy Drop (%)')
    ax.set_title('Synonym Drop')
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(np.array(boot_drops_r) * 100, bins=40, color='#F44336',
            alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=drop_r * 100, color='red', linewidth=2,
               label=f'Observed: {drop_r:+.1%}')
    ax.set_xlabel('Accuracy Drop (%)')
    ax.set_title('Random Drop')
    ax.legend(fontsize=9)

    plt.suptitle(f'Bootstrap Accuracy Drop Distributions (n={n_boot})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/bootstrap_drops.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved bootstrap_drops.png")

    # Figure 4: Dose-response scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    # Bin by replacement fraction and compute accuracy per bin
    bins_edges = np.linspace(0, max(syn_fracs) + 0.01, 8)
    bin_centers = []
    bin_accs = []
    bin_ns = []
    for i in range(len(bins_edges) - 1):
        mask = (syn_fracs >= bins_edges[i]) & (syn_fracs < bins_edges[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins_edges[i] + bins_edges[i+1]) / 2)
            bin_accs.append(syn_ok_arr[mask].mean())
            bin_ns.append(mask.sum())

    if bin_centers:
        ax.bar(bin_centers, [a * 100 for a in bin_accs], width=0.03,
               color='#FF9800', alpha=0.7, edgecolor='black', linewidth=0.5)
        for x, a, nn in zip(bin_centers, bin_accs, bin_ns):
            ax.text(x, a * 100 + 1, f'n={nn}', ha='center', fontsize=8)
        ax.axhline(y=acc_o * 100, color='blue', linestyle='--',
                   label=f'Original: {acc_o:.1%}')
        ax.set_xlabel('Synonym Replacement Fraction', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'Dose-Response: Replacement Rate vs Accuracy\n'
                     f'r_pb = {r_dose:.3f}, p = {p_dose:.4f}', fontsize=12)
        ax.legend()
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/dose_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved dose_response.png")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    results_data = {
        'config': {
            'model': MODEL_NAME, 'n_problems': N_PROBLEMS,
            'replacement_rate': REPLACEMENT_RATE, 'seed': SEED,
            'synonym_dict_size': len(SYNONYMS),
        },
        'generation': {
            'accuracy': gen_acc, 'n_correct': len(correct), 'n_total': n_total,
        },
        'accuracy': {
            'original': acc_o, 'synonym': acc_s, 'random': acc_r,
            'drop_synonym': drop_s, 'drop_random': drop_r,
        },
        'mcnemar': {
            'orig_vs_syn': mc_os, 'orig_vs_rand': mc_or, 'syn_vs_rand': mc_sr,
        },
        'bootstrap': {
            'ci_synonym': [float(ci_s[0]), float(ci_s[1])],
            'ci_random': [float(ci_r[0]), float(ci_r[1])],
            'p_boot_syn': float(p_boot_s),
            'p_boot_rand': float(p_boot_r),
            'n_boot': n_boot,
        },
        'dose_response': {
            'r_pb': float(r_dose), 'p': float(p_dose),
        },
        'text_stats': {
            'avg_eligible_words': float(avg_elig),
            'avg_n_syn': float(avg_n_syn),
            'avg_n_rand': float(avg_n_rand),
            'avg_syn_frac': float(avg_syn_frac),
            'avg_rand_frac': float(avg_rand_frac),
            'avg_similarity_synonym': float(np.mean(syn_sims)),
            'avg_similarity_random': float(np.mean(rand_sims)),
            'avg_token_change_synonym': float(np.mean(syn_tok_changes)),
            'avg_token_change_random': float(np.mean(rand_tok_changes)),
            'nums_preserved_synonym': syn_ok,
            'nums_preserved_random': rand_ok,
        },
        'sanity': {
            'original_tf_mismatches': orig_mismatch,
            'original_tf_mismatch_rate': orig_mismatch / n,
        },
        'runtime_seconds': time.time() - t0,
    }

    with open(f'{RESULTS_DIR}/results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    # Per-problem details
    per_prob = [{
        'idx': p['idx'], 'gold': p['gold'],
        'original_ok': p['original_ok'],
        'synonym_ok': p['synonym_ok'],
        'random_ok': p['random_ok'],
        'original_ans': p.get('original_ans'),
        'synonym_ans': p.get('synonym_ans'),
        'random_ans': p.get('random_ans'),
        'n_syn': p['n_syn'], 'n_rand': p['n_rand'],
        'n_eligible': p['n_eligible'],
        'syn_frac': p['syn_frac'],
        'nums_ok_syn': p['nums_ok_syn'],
        'nums_ok_rand': p['nums_ok_rand'],
    } for p in correct]

    with open(f'{RESULTS_DIR}/per_problem.json', 'w') as f:
        json.dump(per_prob, f, indent=2)

    # Save examples for manual inspection
    examples = [{
        'idx': p['idx'], 'question': p['question'][:200],
        'gold': p['gold'],
        'original': p['cot_body'][:400],
        'synonym': p['synonym_cot'][:400],
        'random': p['random_cot'][:400],
        'original_ans_text': p.get('original_ans_text', '')[:50],
        'synonym_ans_text': p.get('synonym_ans_text', '')[:50],
        'random_ans_text': p.get('random_ans_text', '')[:50],
    } for p in correct[:10]]

    with open(f'{RESULTS_DIR}/examples.json', 'w') as f:
        json.dump(examples, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 085 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"  Generation:   {gen_acc:.0%} ({len(correct)}/{n_total})")
    print(f"  Original TF:  {acc_o:.1%}")
    print(f"  Synonym:      {acc_s:.1%} (drop: {drop_s:+.1%}, "
          f"McNemar p={mc_os['p']:.4f})")
    print(f"  Random:       {acc_r:.1%} (drop: {drop_r:+.1%}, "
          f"McNemar p={mc_or['p']:.4f})")
    print(f"  Bootstrap CI: syn=[{ci_s[0]:+.1%},{ci_s[1]:+.1%}], "
          f"rand=[{ci_r[0]:+.1%},{ci_r[1]:+.1%}]")
    print(f"  Dose-response: r_pb={r_dose:.3f}, p={p_dose:.4f}")
    print(f"  Token change: syn={np.mean(syn_tok_changes):.1%}, "
          f"rand={np.mean(rand_tok_changes):.1%}")
    print(f"  Results: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
