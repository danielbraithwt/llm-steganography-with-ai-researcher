#!/usr/bin/env python3
"""Quick smoke test for exp_014: verify Qwen3-4B instruct model loads and generates."""
import torch, gc, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen3-4B"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager"
)
model.eval()
print(f"Loaded. Layers={model.config.num_hidden_layers}")
print(f"Num KV heads: {model.config.num_key_value_heads}")
print(f"Num attention heads: {model.config.num_attention_heads}")

ds = load_dataset("openai/gsm8k", "main", split="test")
q = ds[0]["question"]
true_ans = ds[0]["answer"].split("####")[-1].strip()

prompt = """Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?
A: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
#### 18

Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: It takes 2/2=<<2/2=1>>1 bolt of white fiber
So the total bolts needed is 2+1=<<2+1=3>>3
#### 3

Q: """ + q + "\nA:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Prompt tokens: {inputs.input_ids.shape[1]}")

with torch.no_grad():
    out = model.generate(inputs.input_ids, max_new_tokens=256, do_sample=False, temperature=1.0)
gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"Generated: {gen[:300]}...")
print(f"True answer: {true_ans}")

if "<think>" in tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False):
    print("NOTE: Model produces <think> blocks (instruct behavior)")

# Extract answer
text = re.sub(r'<think>.*?</think>', '', gen, flags=re.DOTALL)
if "####" in text:
    ans = text.split("####")[-1].strip().replace(",", "").replace("$", "")
    m = re.match(r'^-?[\d.]+', ans)
    if m:
        print(f"Extracted answer: {m.group(0)}")
        print(f"Correct: {m.group(0) == true_ans.replace(',','')}")

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
print("Smoke test passed!")
