import os
import json
import torch
import contextlib
from transformers import AutoTokenizer
from dotenv import load_dotenv
from model.model import MiniGPT

# --- Environment & config ----------------------------------------------------
load_dotenv()

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "camembert-base")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "checkpoints/best_miniGPT.pt")
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE")
STOP_SEQUENCE = os.getenv("STOP_SEQUENCE")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

# Recommended TF32 API (avoid deprecated allow_tf32 flags)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.conv.fp32_precision = "high"


# --- Tokenizer ---------------------------------------------------------------
def load_tokenizer(tokenizer_name: str):
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


# --- Model -------------------------------------------------------------------
model_cfg = config["model"]
model = MiniGPT(
    len(tokenizer),
    model_cfg["block_size"],
    embed_dim=model_cfg["embed_dim"],
    depth=model_cfg["depth"],
    heads=model_cfg["heads"],
    dropout=model_cfg["dropout"],
    hidden_dim=model_cfg["hidden_dim"],
    weight_sharing=model_cfg.get("weight_sharing", "none"),
    use_rope=model_cfg.get("use_rope", True),
    use_gradient_checkpointing=False,  # inference: no checkpointing
).to(device)

if os.path.exists(MODEL_SAVE_PATH):
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    print(f"✅ Model loaded from {MODEL_SAVE_PATH}")
else:
    raise FileNotFoundError(f"❌ No model checkpoint found at {MODEL_SAVE_PATH}")

model.eval()

try:
    model = torch.compile(model, mode="reduce-overhead")
    print("⚙️ Model compiled for optimized inference")
except Exception:
    print("⚠️ torch.compile not supported here — running normally")


# --- Prompt helpers ----------------------------------------------------------
class _SafeDict(dict):
    def __missing__(self, key):
        return ""


def format_prompt(prompt: str, caption: str | None, instructions: str | None) -> str:
    if PROMPT_TEMPLATE:
        fmt = _SafeDict(
            {"prompt": prompt, "caption": caption or prompt, "instructions": instructions or prompt}
        )
        return PROMPT_TEMPLATE.format_map(fmt)
    return prompt


# --- Generation --------------------------------------------------------------
@torch.inference_mode()
def generate_text(
    prompt: str,
    max_new_tokens: int = 100,
    caption: str | None = None,
    instructions: str | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_new_tokens: int = 0,
):
    formatted = format_prompt(prompt, caption=caption, instructions=instructions)
    input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)

    # autocast for GPU inference
    if torch.cuda.is_available():
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    with autocast_ctx:
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_new_tokens=min_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

    gen_tokens = output[input_ids.shape[-1] :]
    text = tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)

    if STOP_SEQUENCE:
        cut = text.find(STOP_SEQUENCE)
        if cut != -1:
            text = text[:cut]
    return text.strip()


# --- CLI ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text using MiniGPT")
    parser.add_argument("--prompt", type=str, default="Il était une fois", help="Texte de départ")
    parser.add_argument("--caption", type=str, default=None, help="Champ structuré optionnel")
    parser.add_argument("--instructions", type=str, default=None, help="Champ structuré optionnel")
    parser.add_argument("--tokens", type=int, default=100, help="Nombre de tokens à générer")
    parser.add_argument("--temperature", type=float, default=1.0, help="Température d'échantillonnage")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p / nucleus sampling")
    parser.add_argument("--min_new_tokens", type=int, default=0, help="Génère au moins ce nombre de tokens")
    args = parser.parse_args()

    print(f"\n📝 Prompt: {args.prompt}")
    generated = generate_text(
        args.prompt,
        max_new_tokens=args.tokens,
        caption=args.caption,
        instructions=args.instructions,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_new_tokens=args.min_new_tokens,
    )
    print("\n✨ Texte généré :\n")
    print(generated)
