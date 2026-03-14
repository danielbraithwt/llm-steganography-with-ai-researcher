#!/usr/bin/env python3
"""Smoke test for exp_027 — run 1 valid problem with first 4 conditions."""
import os, time, random, gc
import numpy as np
import torch

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import scripts.exp_027_snr_controlled_kv as exp

from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
random.seed(42); np.random.seed(42); torch.manual_seed(42)

print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(exp.MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    exp.MODEL_NAME, dtype=torch.bfloat16, device_map='auto',
    trust_remote_code=True, attn_implementation='eager')
model.eval()
num_layers = model.config.num_hidden_layers
print(f'Model loaded: {num_layers} layers, t={time.time()-t0:.0f}s')

test_input = tokenizer('test', return_tensors='pt').to(model.device)
test_out = model(**test_input, use_cache=True)
k_shape = test_out.past_key_values.layers[0].keys.shape
print(f'Cache: layers[0].keys.shape = {k_shape}')
del test_input, test_out; gc.collect(); torch.cuda.empty_cache()

from datasets import load_dataset
ds = load_dataset('openai/gsm8k', 'main', split='test')
indices = list(range(len(ds)))
random.shuffle(indices)

for pi, ds_idx in enumerate(indices[:20]):
    item = ds[ds_idx]
    true_answer = exp.normalize_answer(exp.extract_answer(item['answer']))
    prompt = exp.build_prompt(item['question'])

    print(f'\nProblem {pi+1} (idx={ds_idx}), true={true_answer}')
    trace, prompt_ids, reasoning_ids = exp.generate_trace(model, tokenizer, prompt)
    gen_answer = exp.normalize_answer(exp.extract_answer(trace)) if exp.extract_answer(trace) else ''
    if gen_answer != true_answer:
        print(f'  WRONG: got {gen_answer}')
        del prompt_ids, reasoning_ids; gc.collect(); torch.cuda.empty_cache()
        continue

    trunc_pos = exp.find_truncation_point(reasoning_ids, tokenizer)
    if trunc_pos is None or trunc_pos < 10:
        print(f'  Skip: no truncation')
        del prompt_ids, reasoning_ids; gc.collect(); torch.cuda.empty_cache()
        continue

    reasoning_ids_truncated = reasoning_ids[:, :trunc_pos]
    prompt_len = prompt_ids.shape[1]
    reasoning_len = reasoning_ids_truncated.shape[1]
    print(f'  R={reasoning_len}, prompt={prompt_len}')

    positions = exp.select_late_positions(reasoning_len)
    prompt_cache = exp.build_prompt_cache(model, prompt_ids, num_layers)

    clean = exp.evaluate_condition(model, tokenizer, prompt_cache, reasoning_ids_truncated,
                                    [], prompt_len, num_layers, true_answer,
                                    component='k', perturb_type='snr', snr_db=100)
    print(f'  Clean: acc={clean["correct"]}, text={clean["text_accuracy"]:.1%}')

    if not clean['correct']:
        print('  Skip: clean wrong')
        del prompt_ids, reasoning_ids, reasoning_ids_truncated, prompt_cache
        gc.collect(); torch.cuda.empty_cache()
        continue

    # Test first 4 conditions + last 2 (direction replacement)
    test_conds = exp.CONDITIONS[:4] + exp.CONDITIONS[-2:]
    for cond_name, component, perturb_type, snr_db in test_conds:
        pc = exp.build_prompt_cache(model, prompt_ids, num_layers)
        ev = exp.evaluate_condition(model, tokenizer, pc, reasoning_ids_truncated,
                                     positions, prompt_len, num_layers, true_answer,
                                     component=component, perturb_type=perturb_type,
                                     snr_db=snr_db if snr_db is not None else 0)
        print(f'    {cond_name}: acc={ev["correct"]}, text={ev["text_accuracy"]:.1%}, '
              f'k_rms={ev["k_rms"]:.1f}, v_rms={ev["v_rms"]:.1f}, '
              f'k_snr={ev["k_actual_snr_db"]:.1f}dB, v_snr={ev["v_actual_snr_db"]:.1f}dB')

    del prompt_ids, reasoning_ids, reasoning_ids_truncated, prompt_cache
    gc.collect(); torch.cuda.empty_cache()
    print(f'\nSmoke test PASSED. Total: {time.time()-t0:.0f}s')
    break

print(f'Done. t={time.time()-t0:.0f}s')
