import os
import json
import torch
from transformers import CamembertTokenizer
from dotenv import load_dotenv

from model.model import MiniGPT


load_dotenv()

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DATASET_NAME = os.getenv("DATASET_NAME", "iproskurina/TinyStories-French")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "camembert-base")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "checkpoints/best_miniGPT.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")


tokenizer = CamembertTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = len(tokenizer)
block_size = config["model"]["block_size"]
embed_dim = config["model"]["embed_dim"]
depth = config["model"]["depth"]
heads = config["model"]["heads"]

model = MiniGPT(vocab_size, block_size, embed_dim=embed_dim, depth=depth, heads=heads).to(device)

if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {MODEL_SAVE_PATH}")
else:
    raise FileNotFoundError(f"❌ No model checkpoint found at {MODEL_SAVE_PATH}")

@torch.no_grad()
def generate_text(prompt, max_new_tokens=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)[0]
    text = tokenizer.decode(output.tolist(), skip_special_tokens=True)
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text using MiniGPT")
    parser.add_argument("--prompt", type=str, required=False, default="Il était une fois",
                        help="Texte de départ pour la génération")
    parser.add_argument("--tokens", type=int, default=100,
                        help="Nombre maximal de nouveaux tokens à générer")

    args = parser.parse_args()
    prompt = args.prompt

    print(f"\n📝 Prompt: {prompt}\n")
    generated = generate_text(prompt, max_new_tokens=args.tokens)
    print("✨ Texte généré :\n")
    print(generated)
