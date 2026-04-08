"""
Interactive chat with the SFT-finetuned MiniGPT model.

Usage:
    python chat.py
    python chat.py --checkpoint checkpoints/best_miniGPT_sft.pt
    python chat.py --temperature 0.5 --max_tokens 200
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from dotenv import load_dotenv

from model.configuration import MiniGPTConfig
from model.modeling_minigpt import MiniGPTForCausalLM
from dataset.chat_dataset import add_chat_tokens, format_chat, SPECIAL_TOKENS

load_dotenv()

CONFIG_PATH = os.getenv("SFT_CONFIG_PATH", "config_sft.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

CHECKPOINT_PATH = os.getenv("SFT_SAVE_PATH", "checkpoints/best_miniGPT_sft.pt")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "camembert-base")

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


tokenizer = load_tokenizer(TOKENIZER_NAME)
add_chat_tokens(tokenizer)

all_ids = list(tokenizer.get_vocab().values())
vocab_size = max(len(tokenizer), max(all_ids) + 1) if all_ids else len(tokenizer)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model_cfg = config["model"]
model_config = MiniGPTConfig(
    vocab_size=vocab_size,
    block_size=model_cfg["block_size"],
    embed_dim=model_cfg["embed_dim"],
    depth=model_cfg["depth"],
    heads=model_cfg["heads"],
    num_kv_heads=model_cfg.get("num_kv_heads", model_cfg["heads"]),
    dropout=0.0,
    hidden_dim=model_cfg["hidden_dim"],
    weight_sharing=model_cfg.get("weight_sharing", "none"),
    use_rope=model_cfg.get("use_rope", True),
    use_gradient_checkpointing=False,
)

model = MiniGPTForCausalLM(model_config)

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint introuvable: {CHECKPOINT_PATH}")

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
state = checkpoint.get("model_state_dict", checkpoint)
state = {k: v for k, v in state.items() if "rope.cos_cached" not in k and "rope.sin_cached" not in k}
model.load_state_dict(state, strict=False)
model.lm_head.weight = model.model.token_emb.weight

model = model.to(device)
model.eval()

val_loss = checkpoint.get("val_loss", "N/A")
step = checkpoint.get("global_step", "N/A")
print(f"Modèle chargé depuis {CHECKPOINT_PATH} (val_loss={val_loss}, step={step})")
print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {device}\n")

end_tag = SPECIAL_TOKENS["end"]
_end_ids = tokenizer.encode(end_tag, add_special_tokens=False)
end_token_id = _end_ids[0] if _end_ids else tokenizer.eos_token_id

print(f"End token: '{end_tag}' → id={end_token_id}")


# ---------------------------------------------------------------------------
# Prompt building — construct the inference prompt directly
# ---------------------------------------------------------------------------
def build_prompt(user_msg, system_msg=None):
    """Build the ChatML prompt for inference, ending right after <|assistant|>\\n
    so the model generates the response content."""
    sys = system_msg or "Tu es un assistant utile et concis. Réponds en français."
    S, U, A, E = SPECIAL_TOKENS["system"], SPECIAL_TOKENS["user"], SPECIAL_TOKENS["assistant"], SPECIAL_TOKENS["end"]
    return f"{S}\n{sys}{E}\n{U}\n{user_msg}{E}\n{A}\n"


# ---------------------------------------------------------------------------
# Generation — manual loop, stops on any special token
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_response(user_msg, temperature=0.7, top_p=0.9, max_new_tokens=150, stream=False, debug=False):
    prompt = build_prompt(user_msg)
    idx = tokenizer.encode(prompt, return_tensors="pt").to(device)
    block_size = model_config.block_size
    gen_tokens = []
    printed_so_far = ""

    if debug:
        print(f"\n[DEBUG] Prompt ({idx.shape[-1]} tokens): {repr(prompt[:200])}")
        print(f"[DEBUG] Last 10 token IDs: {idx[0, -10:].tolist()}")

    for step in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(idx_cond).logits[:, -1, :]

        if temperature != 1.0:
            logits = logits / temperature

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            indices_to_remove = mask.scatter(1, sorted_indices, mask)
            logits[indices_to_remove] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()

        if debug and step < 5:
            top5 = torch.topk(probs, 5)
            top5_tokens = [(tokenizer.decode([tid], skip_special_tokens=False), f"{p:.3f}") for tid, p in zip(top5.indices[0].tolist(), top5.values[0].tolist())]
            print(f"[DEBUG] Step {step}: generated id={token_id} ({repr(tokenizer.decode([token_id], skip_special_tokens=False))}) | top5={top5_tokens}")

        if token_id == end_token_id and step >= 3:
            break
        if token_id == end_token_id:
            # Too early to stop — pick the next best non-special token
            logits[0, end_token_id] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

        idx = torch.cat((idx, next_token), dim=1)
        gen_tokens.append(token_id)

        if stream:
            decoded_so_far = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            new_text = decoded_so_far[len(printed_so_far):]
            if new_text:
                print(new_text, end="", flush=True)
                printed_so_far = decoded_so_far

    response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return response


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chat with MiniGPT SFT")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--debug", action="store_true", help="Show debug info for first 5 tokens")
    args = parser.parse_args()

    print("=" * 60)
    print("  MiniGPT Chat — tapez 'quit' pour quitter")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        print("MiniGPT > ", end="", flush=True)
        response = generate_response(
            user_input,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
            stream=True,
            debug=args.debug,
        )
        print("\n")
