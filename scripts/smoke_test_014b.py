#!/usr/bin/env python3
"""Quick smoke test for exp_014 with fixed dtype."""
import torch, gc, re, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen3-4B"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()
print(f"Loaded: {model.config.num_hidden_layers} layers, GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Same exemplars as exp_014
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
    return "?"

dataset = load_dataset("openai/gsm8k", "main", split="test")

# Test 3 problems
for i in range(3):
    prob = dataset[i]
    true_ans = prob["answer"].split("####")[-1].strip().replace(",", "")
    prompt = build_prompt(prob["question"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    print(f"\nProblem {i+1}: prompt_len={prompt_len}")

    generated_ids = []
    past_kv = None
    current_input = inputs.input_ids
    t0 = time.time()

    for step in range(512):
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
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "####" in text:
            after = text.split("####")[-1]
            if re.search(r'\d+\s*$', after):
                break
        if "\nQ:" in text:
            break
        current_input = next_token

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    t1 = time.time()
    pred = extract_answer(text)

    print(f"  Generated {len(generated_ids)} tokens in {t1-t0:.1f}s")
    print(f"  Text (first 200 chars): {text[:200]}")
    print(f"  Pred={pred} True={true_ans} Match={pred==true_ans}")
    print(f"  GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    del past_kv, outputs
    gc.collect()
    torch.cuda.empty_cache()

# Also test attention computation on problem 0 if it was correct
print("\n--- Testing attention computation ---")
prob = dataset[0]
prompt = build_prompt(prob["question"])
trace = "Let's solve this step by step.\n#### 18"  # Dummy short trace for testing
full_text = prompt + trace
inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
seq_len = inputs.input_ids.shape[1]
prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
reasoning_len = seq_len - prompt_len
print(f"Attention test: seq_len={seq_len}, prompt_len={prompt_len}, reasoning_len={reasoning_len}")

t0 = time.time()
att_outputs = model(**inputs, output_attentions=True, use_cache=False)
n_layers = len(att_outputs.attentions)
print(f"Attention computed: {n_layers} layers, shape={att_outputs.attentions[0].shape}")
print(f"Time: {time.time()-t0:.1f}s, GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Test vectorized H2O
h2o = torch.zeros(reasoning_len, device=model.device)
mask = torch.tril(torch.ones(seq_len, seq_len, device=model.device, dtype=torch.bool), diagonal=-1)
t0 = time.time()
for li in range(n_layers):
    attn = att_outputs.attentions[li][0]
    attn_summed = attn.sum(dim=0)
    col_sums = (attn_summed * mask).sum(dim=0)
    h2o += col_sums[prompt_len:seq_len]
print(f"H2O vectorized: {time.time()-t0:.2f}s, shape={h2o.shape}")
print(f"H2O: min={h2o.min().item():.4f}, max={h2o.max().item():.4f}")

del att_outputs
gc.collect()
torch.cuda.empty_cache()

print("\nSmoke test PASSED!")
