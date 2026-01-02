import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained("camembert-base")

tokenizer = AutoTokenizer.from_pretrained(
    "Houzeric/mini-gpt-french",
    trust_remote_code=True,
    use_fast=False
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "Houzeric/mini-gpt-french",
    trust_remote_code=True
)
model.eval()

prompt = "Où est Paris?"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        temperature=0.6,
        top_p=0.85,
        top_k=30,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
