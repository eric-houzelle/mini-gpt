import os
import json
import torch
import string
from transformers import AutoTokenizer
from dotenv import load_dotenv
from model.configuration import MiniGPTConfig
from model.modeling_minigpt import MiniGPTForCausalLM

# -----------------------------------------------------------------------------
# Environment & config
# -----------------------------------------------------------------------------
load_dotenv()

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "camembert-base")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "checkpoints/best_miniGPT.pt")
DATASET_TEMPLATE = os.getenv("DATASET_TEMPLATE")  # DOIT être identique à train.py

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------
def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok

tokenizer = load_tokenizer(TOKENIZER_NAME)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model_cfg = config["model"]
model_config = MiniGPTConfig(
    vocab_size=len(tokenizer),
    block_size=model_cfg["block_size"],
    embed_dim=model_cfg["embed_dim"],
    depth=model_cfg["depth"],
    heads=model_cfg["heads"],
    dropout=model_cfg["dropout"],
    hidden_dim=model_cfg["hidden_dim"],
    weight_sharing=model_cfg.get("weight_sharing", "none"),
    use_rope=model_cfg.get("use_rope", True),
    use_gradient_checkpointing=False,  # inference only
)

model = MiniGPTForCausalLM(model_config).to(device)

if not os.path.exists(MODEL_SAVE_PATH):
    raise FileNotFoundError(f"Checkpoint introuvable: {MODEL_SAVE_PATH}")

checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
state = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state, strict=False)
model.eval()

print(f"✅ Model loaded from {MODEL_SAVE_PATH}")

# -----------------------------------------------------------------------------
# Generation (ISO avec train.py)
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_example(
    model,
    tokenizer,
    block_size,
    device,
    dataset_template,
    eval_prompt,
    max_new_tokens=200,
    min_new_tokens=20,
    temperature=0.8,
):
    model.eval()

    # --- Construction STRICTEMENT identique à train.py ---
    if dataset_template:
        class _SafeDict(dict):
            def __missing__(self, key):
                return eval_prompt

        formatter = string.Formatter()
        field_names = {
            field_name
            for _, field_name, _, _ in formatter.parse(dataset_template)
            if field_name
        }
        fmt = _SafeDict({name: eval_prompt for name in field_names})
        example_prompt = dataset_template.format_map(fmt)
        prompt_ids = tokenizer.encode(example_prompt, return_tensors="pt").to(device)
    else:
        example_prompt = eval_prompt
        prompt_ids = tokenizer.encode(eval_prompt, return_tensors="pt").to(device)

    eos_id = tokenizer.eos_token_id
    idx = prompt_ids

    for step in range(max_new_tokens):
        # ⚠️ PAS DE LEFT-PADDING
        idx_cond = idx[:, -block_size:]

        logits = model(idx_cond).logits[:, -1, :]
        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # éviter EOS trop tôt
        if eos_id is not None and step < min_new_tokens:
            while next_token.item() == eos_id:
                next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

        if eos_id is not None and step >= min_new_tokens:
            if next_token.item() == eos_id:
                break

    # --- Décodage ISO ---
    sample = idx[0]
    prompt_len = prompt_ids.shape[-1]
    gen_tokens = sample[prompt_len:].tolist()

    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    gen_text_raw = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()

    return example_prompt, gen_text, gen_text_raw, gen_tokens


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MiniGPT inference (ISO train.py)")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Prompt: Quelle est la capitale de la France ? Answer: ",
    )
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--min_new_tokens", type=int, default=20)
    args = parser.parse_args()

    print("\n📝 Prompt:")
    print(args.prompt)

    example_prompt, gen_text, _, _ = generate_example(
        model=model,
        tokenizer=tokenizer,
        block_size=model_config.block_size,
        device=device,
        dataset_template=DATASET_TEMPLATE,
        eval_prompt=args.prompt,
        max_new_tokens=args.tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
    )

    print("\n📌 Prompt réel envoyé au modèle:\n")
    print(example_prompt)

    print("\n✨ Texte généré:\n")
    print(gen_text)
