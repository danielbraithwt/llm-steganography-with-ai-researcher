#!/usr/bin/env python3
"""Quick smoke test for exp_012 — test core functions with 2 problems."""
import sys
sys.path.insert(0, '/workspace/llm-steganography-with-ai-researcher/scripts')
from exp_012_ac_vs_h2o_protection import *

# Override config for smoke test
import exp_012_ac_vs_h2o_protection as exp
exp.NUM_PROBLEMS = 5
exp.NOISE_FRACTIONS = [0.05]
exp.MAX_SEQ_LEN = 1536

print("=== SMOKE TEST: exp_012 ===")
print("Testing with 5 problems, 1 noise fraction, Llama only\n")

start = time.time()
dataset = load_dataset("openai/gsm8k", "main", split="test")
indices = list(range(len(dataset)))
random.seed(42)
random.shuffle(indices)
selected = indices[:5]

# Only Llama
model_name = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager"
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

num_layers = model.config.num_hidden_layers
print(f"Loaded: {num_layers} layers")

# Find first correct problem
for ds_idx in selected:
    problem = dataset[ds_idx]
    question = problem["question"]
    true_answer = normalize_answer(
        problem["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
    )
    prompt = build_prompt(question)
    print(f"\nProblem #{ds_idx}, true={true_answer}")

    trace_text = generate_trace(model, tokenizer, prompt)
    gen_answer = extract_answer(trace_text)
    gen_norm = normalize_answer(gen_answer) if gen_answer else ""
    correct = (gen_norm == true_answer)
    print(f"Generated: '{gen_answer}' (correct: {correct})")

    if not correct:
        print("Skip — baseline incorrect")
        continue

    # Extract reasoning
    if "####" in trace_text:
        reasoning_text = trace_text[:trace_text.index("####")]
    else:
        reasoning_text = trace_text

    # Test compute_importance_scores
    print("Computing importance scores...")
    scores = compute_importance_scores(model, tokenizer, prompt, reasoning_text)
    if scores is None:
        print("Skip — trace too short")
        continue

    reasoning_text = scores["truncated_trace"]
    print(f"  reasoning_len={scores['reasoning_len']}, seq_len={scores['seq_len']}")
    print(f"  AC scores range: [{scores['ac_score'].min():.6f}, {scores['ac_score'].max():.6f}]")
    print(f"  H2O scores range: [{scores['h2o_score'].min():.4f}, {scores['h2o_score'].max():.4f}]")
    print(f"  AC-H2O correlation: {np.corrcoef(scores['ac_score'], scores['h2o_score'])[0,1]:.3f}")

    # Test build_kv_cache
    print("Building KV cache...")
    base_kv, last_token_id, seq_len = build_kv_cache(model, tokenizer, prompt, reasoning_text)
    print(f"  KV cache: {num_layers} layers, seq_len={seq_len}")
    k0, v0 = get_kv(base_kv, 0)
    print(f"  key_cache[0] shape: {k0.shape}")

    # Test baseline (no noise)
    print("Testing baseline (no noise)...")
    _, baseline_answer = noise_and_generate(
        model, tokenizer, base_kv, [],
        scores["prompt_len"], seq_len, num_layers, last_token_id
    )
    baseline_norm = normalize_answer(baseline_answer) if baseline_answer else ""
    print(f"  Baseline: '{baseline_answer}' (correct: {baseline_norm == true_answer})")

    # Test full noise
    print("Testing full noise...")
    all_pos = list(range(scores["reasoning_len"]))
    _, full_answer = noise_and_generate(
        model, tokenizer, base_kv, all_pos,
        scores["prompt_len"], seq_len, num_layers, last_token_id
    )
    full_norm = normalize_answer(full_answer) if full_answer else ""
    print(f"  Full noise: '{full_answer}' (correct: {full_norm == true_answer})")

    # Test each strategy at 5% noise
    print("Testing strategies at 5% noise:")
    for strat in STRATEGIES:
        positions_to_noise = select_positions_to_noise(scores, scores["reasoning_len"], 0.05, strat)
        print(f"  {strat}: noising {len(positions_to_noise)} positions "
              f"(mean AC={scores['ac_score'][positions_to_noise].mean():.6f})")
        _, answer = noise_and_generate(
            model, tokenizer, base_kv, positions_to_noise,
            scores["prompt_len"], seq_len, num_layers, last_token_id
        )
        ans_norm = normalize_answer(answer) if answer else ""
        print(f"    Answer: '{answer}' (correct: {ans_norm == true_answer})")

    del base_kv
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n=== SMOKE TEST PASSED in {time.time()-start:.1f}s ===")
    break

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
