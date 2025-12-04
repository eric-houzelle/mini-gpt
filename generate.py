import os
import json
import torch
from transformers import AutoTokenizer
from dotenv import load_dotenv
from model.model import MiniGPT

# === Chargement config + env ===
load_dotenv()

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "camembert-base")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "checkpoints/best_miniGPT.pt")
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE")
STOP_SEQUENCE = os.getenv("STOP_SEQUENCE")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"✅ Device: {device} (dtype={dtype})")

# === Tokenizer ===
def load_tokenizer(tokenizer_name):
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.sep_token is not None:
            tok.pad_token = tok.sep_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok

tokenizer = load_tokenizer(TOKENIZER_NAME)

# === Hyperparams ===
vocab_size = len(tokenizer)
block_size = config["model"]["block_size"]
embed_dim = config["model"]["embed_dim"]
depth = config["model"]["depth"]
heads = config["model"]["heads"]
dropout = config["model"]["dropout"]
hidden_dim = config["model"]["hidden_dim"]

# === Modèle ===
model = MiniGPT(
    len(tokenizer),
    block_size,
    embed_dim=embed_dim,
    depth=depth,
    heads=heads,
    dropout=dropout,
    hidden_dim=hidden_dim
).to(device)

# === Chargement du checkpoint ===
if os.path.exists(MODEL_SAVE_PATH):
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"✅ Model loaded from {MODEL_SAVE_PATH}")
else:
    raise FileNotFoundError(f"❌ No model checkpoint found at {MODEL_SAVE_PATH}")

model.eval()

# === Compilation (si dispo) ===
try:
    model = torch.compile(model)
    print("⚙️ Model compiled for optimized inference")
except Exception:
    print("⚠️ torch.compile not supported here — running normally")

# === Génération ===
class _SafeDict(dict):
    def __missing__(self, key):
        return ""


@torch.no_grad()
def _format_prompt(prompt, caption=None, instructions=None):
    if PROMPT_TEMPLATE:
        fmt = _SafeDict({
            "prompt": prompt,
            "caption": caption or prompt,
            "instructions": instructions or prompt
        })
        return PROMPT_TEMPLATE.format_map(fmt)
    return prompt


def generate_text(prompt, max_new_tokens=100, caption=None, instructions=None):
    formatted_prompt = _format_prompt(prompt, caption=caption, instructions=instructions)
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    with torch.cuda.amp.autocast(enabled=(dtype == torch.float16)):
        output = model.generate(input_ids, max_new_tokens=max_new_tokens)[0]

    # Retirer le prompt: generer uniquement les nouveaux tokens
    gen_tokens = output[input_ids.shape[-1]:]
    text = tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)

    if STOP_SEQUENCE:
        stop_index = text.find(STOP_SEQUENCE)
        if stop_index != -1:
            text = text[:stop_index]
    return text.strip()

# === Script principal ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate text using MiniGPT")
    parser.add_argument("--prompt", type=str, default="Il était une fois", help="Texte de départ ou champ libre")
    parser.add_argument("--caption", type=str, default=None, help="Caption structurée (si DATASET_TEMPLATE attend ce champ)")
    parser.add_argument("--instructions", type=str, default=None, help="Instructions structurées (si nécessaires)")
    parser.add_argument("--tokens", type=int, default=100, help="Nombre de tokens à générer")
    args = parser.parse_args()

    print(f"\n📝 Prompt: {args.prompt}")
    if args.caption or args.instructions:
        print("📋 Champs structurés:", end=" ")
        parts = []
        if args.caption:
            parts.append("caption")
        if args.instructions:
            parts.append("instructions")
        print(", ".join(parts))
    print()
    generated = generate_text(
        args.prompt,
        max_new_tokens=args.tokens,
        caption=args.caption,
        instructions=args.instructions
    )
    print("✨ Texte généré :\n")
    print(generated)
