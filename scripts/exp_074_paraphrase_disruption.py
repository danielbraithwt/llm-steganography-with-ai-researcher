#!/usr/bin/env python3
"""
Experiment 074: Paraphrase Disruption — Token Replacement Test

Tests whether specific token choices in CoT carry hidden information
by replacing non-numeric tokens and measuring accuracy impact.

Three conditions (all teacher-forced):
1. Original: unmodified CoT tokens (sanity check — should match generation accuracy)
2. Synonym: non-numeric tokens replaced with closest embedding neighbor
3. Random: non-numeric tokens replaced with random common words

Core prediction:
- If hidden channel hypothesis is true: synonym replacement drops accuracy ≥10pp
  because different tokens produce different K-routing patterns
- If false: synonym replacement has minimal impact (<5pp) because model reads
  semantic content, not exact tokens

Method:
1. Generate 8-shot CoT on GSM8K (greedy, ~200 problems)
2. For correctly-solved problems, classify each token as numeric/structural vs replaceable
3. Create synonym version: replace replaceable tokens with closest embedding neighbor
4. Create random version: replace replaceable tokens with random common words
5. Teacher-force each condition up to "####", let model generate answer
6. Compare accuracy across conditions
"""

import os
import json
import time
import gc
import re
import sys
import random

import numpy as np
import torch
from pathlib import Path

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

TIME_BUDGET = 6000  # 100 min
MAX_GEN = 512
MODEL_NAME = 'Qwen/Qwen3-4B-Base'
N_PROBLEMS_MAX = 250
N_ANSWER_TOKENS = 40  # tokens to generate for answer

RESULTS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "exp_074"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 8-shot exemplars (same as exp_071-073) ──
EXEMPLARS = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
     "a": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "a": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total bolts needed is 2+1=<<2+1=3>>3\n#### 3"},
    {"q": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
     "a": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*150%=$<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"},
    {"q": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "a": "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624"},
    {"q": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
     "a": "If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.\nIf she feeds the flock 15 cups in the morning, and 25 cups in the afternoon, then the carry-over to the final meal would be 60-15-25=<<60-15-25=20>>20 cups.\n#### 20"},
    {"q": "Kylar went to the store to get water and some apples. The store sold apples for $1 each and water for $3 per bottle. Kylar wanted to buy one bag of apples and 2 bottles of water. How much would Kylar spend if each bag has 6 apples?",
     "a": "A bag has 6 apples and each apple costs $1, so a bag costs 6*1=$<<6*1=6>>6\nKylar wants 2 bottles of water so that would cost 2*3=$<<2*3=6>>6\nAltogether, Kylar would spend 6+6=$<<6+6=12>>12\n#### 12"},
    {"q": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
     "a": "If Seattle has 20 sheep, Charleston has 4 * 20 = <<4*20=80>>80 sheep\nToulouse has 2 * 80 = <<2*80=160>>160 sheep\nTogether, they have 20 + 80 + 160 = <<20+80+160=260>>260 sheep\n#### 260"},
    {"q": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
     "a": "First find how long it takes to download 40% of the file: 200 GB * 0.4 / 2 GB/minute = <<200*0.4/2=40>>40 minutes\nThen find how long it takes to download the whole file once the restart is complete: 200 GB / 2 GB/minute = <<200/2=100>>100 minutes\nThen add the time to download 40% of the file, the restart time, and the time to download the whole file: 40 + 20 + 100 = <<40+20+100=160>>160 minutes\n#### 160"},
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


def should_replace(token_text):
    """Determine if a token should be replaced (non-numeric, non-structural)."""
    stripped = token_text.strip()
    if not stripped:
        return False  # Whitespace-only → structural
    if '\n' in token_text:
        return False  # Newlines → structural
    # Contains any digit → preserve
    if re.search(r'\d', stripped):
        return False
    # Math operators and structural markers
    if stripped in ['+', '-', '*', '/', '=', '(', ')', '<', '>', '<<', '>>',
                    '####', '$', '%', '.', ',', ':', ';', '!', '?', "'", '"',
                    '#', '&', '@', '|', '\\', '~', '^', '_', '{', '}', '[', ']']:
        return False
    # Very short non-alpha tokens
    if len(stripped) <= 1 and not stripped.isalpha():
        return False
    return True


def precompute_synonyms(unique_token_ids, emb_matrix, emb_norms, tokenizer, top_k=10):
    """
    Precompute nearest-neighbor synonyms for a set of token IDs.
    Returns dict: token_id → list of candidate replacement IDs (sorted by similarity).
    """
    if len(unique_token_ids) == 0:
        return {}

    # Get embeddings for query tokens
    query_ids = list(unique_token_ids)
    query_embs = emb_matrix[query_ids]  # (n_queries, hidden_dim)
    query_norms = emb_norms[query_ids]  # (n_queries,)

    # Compute cosine similarities: (n_queries, vocab_size)
    # Do in chunks to avoid OOM
    chunk_size = 200
    synonym_map = {}

    for start in range(0, len(query_ids), chunk_size):
        end = min(start + chunk_size, len(query_ids))
        chunk_embs = query_embs[start:end]  # (chunk, hidden_dim)
        chunk_norms = query_norms[start:end]  # (chunk,)

        # (chunk, vocab_size)
        sims = torch.mm(chunk_embs, emb_matrix.T)
        sims = sims / (chunk_norms.unsqueeze(1) * emb_norms.unsqueeze(0) + 1e-8)

        # Get top-K+5 for each (extra for filtering)
        topk_vals, topk_ids = torch.topk(sims, top_k + 5, dim=1)

        for i in range(end - start):
            orig_id = query_ids[start + i]
            orig_text = tokenizer.decode([orig_id])
            candidates = []
            for j in range(topk_ids.shape[1]):
                cand_id = topk_ids[i, j].item()
                if cand_id == orig_id:
                    continue
                cand_text = tokenizer.decode([cand_id])
                # Filter: must be a "word-like" token, not numeric
                cand_stripped = cand_text.strip()
                if not cand_stripped:
                    continue
                if re.search(r'\d', cand_stripped):
                    continue
                if len(cand_stripped) <= 1 and not cand_stripped.isalpha():
                    continue
                # Must be different from original (case-insensitive)
                if orig_text.strip().lower() == cand_stripped.lower():
                    continue
                candidates.append(cand_id)
                if len(candidates) >= top_k:
                    break
            synonym_map[orig_id] = candidates

    return synonym_map


def build_random_pool(tokenizer, vocab_size, pool_size=5000):
    """Build a pool of common word tokens for random replacement."""
    pool = []
    # Scan through vocabulary, picking word-like tokens
    for idx in range(100, min(vocab_size, 50000)):
        text = tokenizer.decode([idx])
        stripped = text.strip()
        if not stripped:
            continue
        if re.search(r'\d', stripped):
            continue
        if len(stripped) <= 1 and not stripped.isalpha():
            continue
        # Must be alphabetic (word-like)
        if re.match(r'^[a-zA-Z]', stripped):
            pool.append(idx)
        if len(pool) >= pool_size:
            break
    return pool


def create_modified_cot(cot_ids, tokenizer, synonym_map, random_pool, mode, rng):
    """
    Create modified version of CoT token IDs.
    mode: 'synonym' or 'random'
    Returns: (modified_ids, n_replaced, n_replaceable)
    """
    modified = []
    n_replaced = 0
    n_replaceable = 0

    for tid in cot_ids:
        token_text = tokenizer.decode([tid])
        if should_replace(token_text):
            n_replaceable += 1
            if mode == 'synonym':
                candidates = synonym_map.get(tid, [])
                if candidates:
                    # Pick from top-3 (closest neighbors)
                    new_tid = rng.choice(candidates[:3])
                    modified.append(new_tid)
                    n_replaced += 1
                else:
                    modified.append(tid)  # No synonym found
            elif mode == 'random':
                new_tid = rng.choice(random_pool)
                modified.append(new_tid)
                n_replaced += 1
            else:
                modified.append(tid)
        else:
            modified.append(tid)

    return modified, n_replaced, n_replaceable


def find_hash_split(gen_ids, tokenizer):
    """
    Find the token position where '####' starts in generated tokens.
    Returns (cot_ids, hash_trigger_ids) where hash_trigger_ids contains
    ONLY the #### marker (and trailing space), NOT the answer number.
    Returns None if #### not found.
    """
    # Decode cumulatively to find "####"
    cum_text = ""
    hash_start_tok = None
    for i, tid in enumerate(gen_ids):
        tok_text = tokenizer.decode([tid], skip_special_tokens=False)
        cum_text += tok_text
        if "####" in cum_text and hash_start_tok is None:
            # Find the exact token where #### starts
            prefix = cum_text[:cum_text.index("####")]
            char_count = 0
            for j, t in enumerate(gen_ids):
                t_text = tokenizer.decode([t], skip_special_tokens=False)
                char_count += len(t_text)
                if char_count > len(prefix):
                    hash_start_tok = j
                    break
            if hash_start_tok is None:
                hash_start_tok = i
            break

    if hash_start_tok is None:
        return None

    cot_ids = gen_ids[:hash_start_tok]

    # Extract ONLY the #### marker tokens (not the answer number)
    # Walk forward from hash_start_tok, collecting tokens until we hit a digit
    hash_trigger = []
    for k in range(hash_start_tok, len(gen_ids)):
        tid = gen_ids[k]
        tok_text = tokenizer.decode([tid], skip_special_tokens=False)
        stripped = tok_text.strip()
        # Stop before the first token containing a digit (the answer number)
        if stripped and re.search(r'\d', stripped):
            break
        hash_trigger.append(tid)

    # If we couldn't find any non-digit hash tokens (edge case), encode "#### " directly
    if not hash_trigger:
        hash_trigger = tokenizer.encode("#### ", add_special_tokens=False)

    return cot_ids, hash_trigger


def main():
    t0 = time.time()

    print("=" * 60)
    print("Experiment 074: Paraphrase Disruption — Token Replacement")
    print("=" * 60)

    # ── Load model ──
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
    )
    model.eval()

    vocab_size = model.config.vocab_size
    print(f"Model: {MODEL_NAME}")
    print(f"Vocab size: {vocab_size}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Layers: {model.config.num_hidden_layers}")

    # ── Get embedding matrix for synonym computation ──
    print("Extracting embedding matrix...")
    emb_weight = model.get_input_embeddings().weight.data.float().cpu()  # (vocab, hidden)
    emb_norms = emb_weight.norm(dim=1)  # (vocab,)
    print(f"Embedding shape: {emb_weight.shape}")

    # ── Build random replacement pool ──
    print("Building random replacement pool...")
    random_pool = build_random_pool(tokenizer, vocab_size)
    print(f"Random pool size: {len(random_pool)}")

    # ── Load GSM8K ──
    print("\nLoading GSM8K...")
    ds = load_gsm8k()
    print(f"GSM8K test set: {len(ds)} problems")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Generate CoT traces
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: Generating CoT traces")
    print("=" * 60)

    generations = []
    for i in range(min(N_PROBLEMS_MAX, len(ds))):
        if time.time() - t0 > TIME_BUDGET * 0.30:
            print(f"  Time budget (30%) reached at problem {i}")
            break

        question = ds[i]['question']
        gold = extract_gold(ds[i]['answer'])
        prompt = build_prompt(question)

        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        prompt_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_GEN, do_sample=False,
                temperature=1.0, return_dict_in_generate=True, use_cache=True
            )

        gen_ids = output.sequences[0][prompt_len:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_answer(gen_text)
        correct = pred is not None and str(pred).strip() == str(gold).strip()

        generations.append({
            'idx': i,
            'question': question,
            'gold': gold,
            'gen_text': gen_text,
            'gen_ids': gen_ids,
            'pred': pred,
            'correct': correct,
            'prompt_len': prompt_len,
        })

        if (i + 1) % 50 == 0:
            n_corr = sum(g['correct'] for g in generations)
            print(f"  Generated {i+1} problems, {n_corr}/{len(generations)} correct "
                  f"({100*n_corr/len(generations):.1f}%)")

    n_total = len(generations)
    n_correct = sum(g['correct'] for g in generations)
    print(f"\nPhase 1 complete: {n_total} problems generated")
    print(f"Accuracy: {n_correct}/{n_total} = {100*n_correct/n_total:.1f}%")

    # Filter to correctly-solved problems with #### marker
    correct_gens = []
    for g in generations:
        if not g['correct']:
            continue
        split = find_hash_split(g['gen_ids'], tokenizer)
        if split is None:
            continue
        cot_ids, hash_trigger = split
        if len(cot_ids) < 5:
            continue
        g['cot_ids'] = cot_ids
        g['hash_trigger'] = hash_trigger
        correct_gens.append(g)

    print(f"Usable correctly-solved problems: {len(correct_gens)}")

    # ── Collect unique replaceable tokens and precompute synonyms ──
    print("\nCollecting unique replaceable tokens...")
    unique_replaceable = set()
    for g in correct_gens:
        for tid in g['cot_ids']:
            token_text = tokenizer.decode([tid])
            if should_replace(token_text):
                unique_replaceable.add(tid)
    print(f"Unique replaceable tokens: {len(unique_replaceable)}")

    # Show some examples
    print("\nSample replaceable tokens:")
    sample_ids = list(unique_replaceable)[:20]
    for tid in sample_ids:
        print(f"  {tid}: '{tokenizer.decode([tid])}'")

    print("\nPrecomputing synonyms (GPU)...")
    t_syn = time.time()
    # Move embeddings to GPU for fast computation
    emb_gpu = emb_weight.to(model.device)
    norms_gpu = emb_norms.to(model.device)
    synonym_map = precompute_synonyms(unique_replaceable, emb_gpu, norms_gpu, tokenizer)
    del emb_gpu, norms_gpu
    torch.cuda.empty_cache()
    print(f"Synonyms computed in {time.time()-t_syn:.1f}s")

    # Show sample synonyms
    print("\nSample synonym replacements:")
    for tid in sample_ids[:10]:
        orig = tokenizer.decode([tid])
        syns = synonym_map.get(tid, [])
        syn_texts = [tokenizer.decode([s]) for s in syns[:5]]
        print(f"  '{orig}' → {syn_texts}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Teacher-force and generate answers
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: Paraphrase disruption test")
    print("=" * 60)

    conditions = ['original', 'synonym', 'random']
    results = {c: {'correct': 0, 'total': 0, 'answers': [], 'n_replaced': [], 'n_replaceable': []}
               for c in conditions}

    # For logging individual examples
    example_log = []

    rng_syn = random.Random(SEED)
    rng_rand = random.Random(SEED + 1)

    for gi, gen in enumerate(correct_gens):
        if time.time() - t0 > TIME_BUDGET * 0.90:
            print(f"\n  Time budget (90%) reached at problem {gi}")
            break

        cot_ids = gen['cot_ids']
        hash_trigger = gen['hash_trigger']
        gold = gen['gold']

        # Build prompt token IDs
        prompt = build_prompt(gen['question'])
        prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'][0].tolist()

        example_entry = {
            'idx': gen['idx'],
            'gold': gold,
            'cot_len': len(cot_ids),
        }

        for condition in conditions:
            if condition == 'original':
                mod_cot = list(cot_ids)
                n_replaced = 0
                n_replaceable = sum(1 for t in cot_ids if should_replace(tokenizer.decode([t])))
            elif condition == 'synonym':
                mod_cot, n_replaced, n_replaceable = create_modified_cot(
                    cot_ids, tokenizer, synonym_map, random_pool, 'synonym', rng_syn)
            elif condition == 'random':
                mod_cot, n_replaced, n_replaceable = create_modified_cot(
                    cot_ids, tokenizer, synonym_map, random_pool, 'random', rng_rand)

            # Teacher-force: prompt + modified_cot + hash_trigger (#### marker only, NOT answer)
            full_ids = prompt_ids + mod_cot + hash_trigger
            input_tensor = torch.tensor([full_ids], device=model.device)

            # Check length
            if input_tensor.shape[1] > 2048 - N_ANSWER_TOKENS:
                # Too long, skip
                continue

            with torch.no_grad():
                output = model.generate(
                    input_tensor, max_new_tokens=N_ANSWER_TOKENS,
                    do_sample=False, temperature=1.0, use_cache=True
                )

            # Extract continuation
            cont_ids = output[0][len(full_ids):].tolist()
            cont_text = tokenizer.decode(cont_ids, skip_special_tokens=True)

            # The hash_trigger contains "####" so the model should generate the number
            # Try to extract answer from continuation directly
            # First try: continuation might just be a number
            number_match = re.match(r'\s*(-?[\d,]+(?:\.\d+)?)', cont_text)
            if number_match:
                pred = number_match.group(1).replace(',', '')
            else:
                # Try extracting from hash_trigger + continuation as "#### <number>"
                full_gen = tokenizer.decode(hash_trigger + cont_ids, skip_special_tokens=True)
                pred = extract_answer(full_gen)

            is_correct = pred is not None and str(pred).strip() == str(gold).strip()

            results[condition]['correct'] += int(is_correct)
            results[condition]['total'] += 1
            results[condition]['answers'].append({
                'idx': gen['idx'],
                'pred': pred,
                'gold': gold,
                'correct': is_correct,
            })
            results[condition]['n_replaced'].append(n_replaced)
            results[condition]['n_replaceable'].append(n_replaceable)

            example_entry[f'{condition}_pred'] = pred
            example_entry[f'{condition}_correct'] = is_correct
            example_entry[f'{condition}_n_replaced'] = n_replaced

        example_log.append(example_entry)

        if (gi + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"\n  Processed {gi+1}/{len(correct_gens)} problems ({elapsed:.0f}s)")
            for cond in conditions:
                if results[cond]['total'] > 0:
                    acc = 100 * results[cond]['correct'] / results[cond]['total']
                    mean_rep = np.mean(results[cond]['n_replaced']) if results[cond]['n_replaced'] else 0
                    print(f"    {cond:>10}: {results[cond]['correct']}/{results[cond]['total']} "
                          f"({acc:.1f}%)  mean_replaced={mean_rep:.1f}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Analysis and Results
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    summary = {}
    for cond in conditions:
        total = results[cond]['total']
        correct = results[cond]['correct']
        acc = 100 * correct / total if total > 0 else 0
        mean_rep = np.mean(results[cond]['n_replaced']) if results[cond]['n_replaced'] else 0
        mean_replaceable = np.mean(results[cond]['n_replaceable']) if results[cond]['n_replaceable'] else 0
        rep_rate = mean_rep / mean_replaceable if mean_replaceable > 0 else 0

        summary[cond] = {
            'total': total,
            'correct': correct,
            'accuracy': acc,
            'mean_tokens_replaced': float(mean_rep),
            'mean_tokens_replaceable': float(mean_replaceable),
            'replacement_rate': float(rep_rate),
        }
        print(f"\n{cond.upper()}:")
        print(f"  Accuracy: {correct}/{total} = {acc:.1f}%")
        print(f"  Mean tokens replaced: {mean_rep:.1f} / {mean_replaceable:.1f} "
              f"({100*rep_rate:.1f}%)")

    # Accuracy drops
    orig_acc = summary['original']['accuracy']
    syn_acc = summary['synonym']['accuracy']
    rand_acc = summary['random']['accuracy']
    syn_drop = orig_acc - syn_acc
    rand_drop = orig_acc - rand_acc

    print(f"\n{'='*40}")
    print(f"KEY COMPARISONS:")
    print(f"  Original → Synonym:  {orig_acc:.1f}% → {syn_acc:.1f}% (drop: {syn_drop:+.1f}pp)")
    print(f"  Original → Random:   {orig_acc:.1f}% → {rand_acc:.1f}% (drop: {rand_drop:+.1f}pp)")
    print(f"  Synonym → Random:    {syn_acc:.1f}% → {rand_acc:.1f}% (drop: {syn_acc-rand_acc:+.1f}pp)")

    # ── Statistical test: McNemar's test for paired proportions ──
    print(f"\n{'='*40}")
    print("STATISTICAL TESTS (McNemar's)")

    # Build paired data
    orig_answers = results['original']['answers']
    syn_answers = results['synonym']['answers']
    rand_answers = results['random']['answers']

    # Ensure same problems in each condition
    n_paired = min(len(orig_answers), len(syn_answers), len(rand_answers))

    # McNemar for original vs synonym
    if n_paired > 0:
        a = sum(1 for i in range(n_paired) if orig_answers[i]['correct'] and syn_answers[i]['correct'])
        b = sum(1 for i in range(n_paired) if orig_answers[i]['correct'] and not syn_answers[i]['correct'])
        c = sum(1 for i in range(n_paired) if not orig_answers[i]['correct'] and syn_answers[i]['correct'])
        d = sum(1 for i in range(n_paired) if not orig_answers[i]['correct'] and not syn_answers[i]['correct'])

        print(f"\nOriginal vs Synonym (n={n_paired}):")
        print(f"  Both correct: {a}, Orig only: {b}, Syn only: {c}, Both wrong: {d}")
        # McNemar statistic
        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
            from scipy.stats import chi2
            mcnemar_p = 1 - chi2.cdf(mcnemar_stat, 1) if mcnemar_stat > 0 else 1.0
            print(f"  McNemar chi2={mcnemar_stat:.2f}, p={mcnemar_p:.4f}")
            summary['mcnemar_orig_vs_syn'] = {'chi2': float(mcnemar_stat), 'p': float(mcnemar_p),
                                               'b': b, 'c': c}
        else:
            print(f"  McNemar: no discordant pairs")

        # McNemar for original vs random
        a2 = sum(1 for i in range(n_paired) if orig_answers[i]['correct'] and rand_answers[i]['correct'])
        b2 = sum(1 for i in range(n_paired) if orig_answers[i]['correct'] and not rand_answers[i]['correct'])
        c2 = sum(1 for i in range(n_paired) if not orig_answers[i]['correct'] and rand_answers[i]['correct'])
        d2 = sum(1 for i in range(n_paired) if not orig_answers[i]['correct'] and not rand_answers[i]['correct'])

        print(f"\nOriginal vs Random (n={n_paired}):")
        print(f"  Both correct: {a2}, Orig only: {b2}, Rand only: {c2}, Both wrong: {d2}")
        if b2 + c2 > 0:
            mcnemar_stat2 = (abs(b2 - c2) - 1) ** 2 / (b2 + c2) if (b2 + c2) > 0 else 0
            mcnemar_p2 = 1 - chi2.cdf(mcnemar_stat2, 1) if mcnemar_stat2 > 0 else 1.0
            print(f"  McNemar chi2={mcnemar_stat2:.2f}, p={mcnemar_p2:.4f}")
            summary['mcnemar_orig_vs_rand'] = {'chi2': float(mcnemar_stat2), 'p': float(mcnemar_p2),
                                                'b': b2, 'c': c2}
        else:
            print(f"  McNemar: no discordant pairs")

    # ── Per-problem agreement analysis ──
    print(f"\n{'='*40}")
    print("PER-PROBLEM AGREEMENT:")

    if n_paired > 0:
        orig_syn_agree = sum(1 for i in range(n_paired) if orig_answers[i]['correct'] == syn_answers[i]['correct'])
        orig_rand_agree = sum(1 for i in range(n_paired) if orig_answers[i]['correct'] == rand_answers[i]['correct'])
        syn_rand_agree = sum(1 for i in range(n_paired) if syn_answers[i]['correct'] == rand_answers[i]['correct'])
        print(f"  Original-Synonym agreement: {orig_syn_agree}/{n_paired} ({100*orig_syn_agree/n_paired:.1f}%)")
        print(f"  Original-Random agreement:  {orig_rand_agree}/{n_paired} ({100*orig_rand_agree/n_paired:.1f}%)")
        print(f"  Synonym-Random agreement:   {syn_rand_agree}/{n_paired} ({100*syn_rand_agree/n_paired:.1f}%)")

        summary['agreement'] = {
            'orig_syn': orig_syn_agree / n_paired,
            'orig_rand': orig_rand_agree / n_paired,
            'syn_rand': syn_rand_agree / n_paired,
        }

    # ── Analyze relationship between replacement count and accuracy ──
    print(f"\n{'='*40}")
    print("REPLACEMENT COUNT vs ACCURACY:")

    for cond in ['synonym', 'random']:
        answers = results[cond]['answers']
        n_rep = results[cond]['n_replaced']
        if len(answers) < 10:
            continue
        # Split into tertiles by replacement count
        rep_arr = np.array(n_rep[:len(answers)])
        correct_arr = np.array([a['correct'] for a in answers])
        tertile_bounds = np.percentile(rep_arr, [33, 67])

        low_mask = rep_arr <= tertile_bounds[0]
        mid_mask = (rep_arr > tertile_bounds[0]) & (rep_arr <= tertile_bounds[1])
        high_mask = rep_arr > tertile_bounds[1]

        for label, mask in [('Low', low_mask), ('Mid', mid_mask), ('High', high_mask)]:
            if mask.sum() > 0:
                acc = 100 * correct_arr[mask].mean()
                mean_r = rep_arr[mask].mean()
                print(f"  {cond} {label} ({mask.sum()} problems, mean_rep={mean_r:.0f}): {acc:.1f}%")

    # ── Log sample modified texts ──
    print(f"\n{'='*40}")
    print("SAMPLE MODIFICATIONS (first 3 problems):")

    for gi in range(min(3, len(correct_gens))):
        gen = correct_gens[gi]
        cot_ids = gen['cot_ids']
        orig_text = tokenizer.decode(cot_ids, skip_special_tokens=True)

        # Recreate synonym and random versions
        rng_s = random.Random(SEED + gi * 100)
        rng_r = random.Random(SEED + gi * 100 + 1)
        syn_ids, _, _ = create_modified_cot(cot_ids, tokenizer, synonym_map, random_pool, 'synonym', rng_s)
        rand_ids, _, _ = create_modified_cot(cot_ids, tokenizer, synonym_map, random_pool, 'random', rng_r)

        syn_text = tokenizer.decode(syn_ids, skip_special_tokens=True)
        rand_text = tokenizer.decode(rand_ids, skip_special_tokens=True)

        print(f"\n  Problem {gen['idx']} (gold={gen['gold']}):")
        print(f"  ORIGINAL: {orig_text[:200]}...")
        print(f"  SYNONYM:  {syn_text[:200]}...")
        print(f"  RANDOM:   {rand_text[:200]}...")

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Figures
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Generating figures...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: Accuracy by condition
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    conds = ['original', 'synonym', 'random']
    accs = [summary[c]['accuracy'] for c in conds]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(conds, accs, color=colors, edgecolor='black', linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Paraphrase Disruption: Accuracy by Condition\n'
                 f'Qwen3-4B-Base, n={summary["original"]["total"]} problems', fontsize=13)
    ax.set_ylim(0, 105)
    ax.axhline(y=orig_acc, color='gray', linestyle='--', alpha=0.5, label=f'Original ({orig_acc:.1f}%)')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'accuracy_by_condition.png', dpi=150)
    plt.close()

    # Figure 2: Accuracy drop bar chart
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    drops = [syn_drop, rand_drop]
    labels = [f'Synonym\n({syn_drop:+.1f}pp)', f'Random\n({rand_drop:+.1f}pp)']
    bar_colors = ['#3498db', '#e74c3c']
    bars = ax.bar(labels, drops, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.3, label='10pp threshold (hypothesis)')
    ax.set_ylabel('Accuracy Drop from Original (pp)', fontsize=12)
    ax.set_title('Accuracy Drop by Replacement Type', fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'accuracy_drop.png', dpi=150)
    plt.close()

    # Figure 3: Paired outcome matrix (heatmap)
    if n_paired > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original vs Synonym
        matrix1 = np.array([[a, b], [c, d]])
        im1 = axes[0].imshow(matrix1, cmap='YlOrRd', vmin=0)
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        axes[0].set_xticklabels(['Syn Correct', 'Syn Wrong'])
        axes[0].set_yticklabels(['Orig Correct', 'Orig Wrong'])
        for ii in range(2):
            for jj in range(2):
                axes[0].text(jj, ii, str(matrix1[ii, jj]), ha='center', va='center',
                           fontsize=16, fontweight='bold')
        axes[0].set_title('Original vs Synonym')

        # Original vs Random
        matrix2 = np.array([[a2, b2], [c2, d2]])
        im2 = axes[1].imshow(matrix2, cmap='YlOrRd', vmin=0)
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(['Rand Correct', 'Rand Wrong'])
        axes[1].set_yticklabels(['Orig Correct', 'Orig Wrong'])
        for ii in range(2):
            for jj in range(2):
                axes[1].text(jj, ii, str(matrix2[ii, jj]), ha='center', va='center',
                           fontsize=16, fontweight='bold')
        axes[1].set_title('Original vs Random')

        plt.suptitle(f'Paired Outcome Matrices (n={n_paired})', fontsize=14)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'paired_outcomes.png', dpi=150)
        plt.close()

    # ── Save results ──
    summary['n_generated'] = n_total
    summary['n_correct_generated'] = n_correct
    summary['generation_accuracy'] = 100 * n_correct / n_total if n_total > 0 else 0
    summary['n_paired'] = n_paired
    summary['drops'] = {
        'synonym_drop_pp': float(syn_drop),
        'random_drop_pp': float(rand_drop),
    }
    summary['elapsed_seconds'] = time.time() - t0

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save example log
    with open(RESULTS_DIR / 'example_log.json', 'w') as f:
        json.dump(example_log[:50], f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE. Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
