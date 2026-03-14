#!/usr/bin/env python3
"""Smoke test for exp_014 with torch.no_grad()."""
import torch, gc, re, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()
print(f"Model loaded: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Short 4-shot prompt to reduce memory
GSM8K_EXEMPLARS = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
     "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total bolts needed is 2+1=<<2+1=3>>3\n#### 3"},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
     "answer": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*150%=$<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"},
    {"question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "answer": "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624"},
]

def build_prompt(question):
    prompt = ""
    for ex in GSM8K_EXEMPLARS:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt

dataset = load_dataset("openai/gsm8k", "main", split="test")

with torch.no_grad():
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

        for step in range(256):
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
        text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        if "####" in text_clean:
            ans = text_clean.split("####")[-1].strip().replace(",", "").replace("$", "")
            m = re.match(r'^-?[\d.]+', ans)
            pred = m.group(0) if m else "?"
        else:
            pred = "?"
        print(f"  {len(generated_ids)} tokens in {t1-t0:.1f}s")
        print(f"  Pred={pred} True={true_ans} Match={pred==true_ans}")
        print(f"  GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        del past_kv, outputs
        gc.collect()
        torch.cuda.empty_cache()

print("\nSmoke test passed!")
