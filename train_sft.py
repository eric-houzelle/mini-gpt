"""
Supervised Fine-Tuning (SFT) for MiniGPT.

Loads a pretrained checkpoint and fine-tunes it on conversational data
using ChatML formatting. Only the assistant response tokens contribute
to the loss (prompt tokens are masked with -100).

Usage:
    cp .env_sft .env
    python train_sft.py
"""

import os
import json
import math
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
import trackio

from model.configuration import MiniGPTConfig
from model.modeling_minigpt import MiniGPTForCausalLM
from dataset.chat_dataset import ChatDataset, add_chat_tokens, format_chat, SPECIAL_TOKENS

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = os.getenv("SFT_CONFIG_PATH", "config_sft.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

PRETRAINED_PATH = os.getenv("PRETRAINED_PATH", "checkpoints/best_miniGPT.pt")
SFT_SAVE_PATH = os.getenv("SFT_SAVE_PATH", "checkpoints/best_miniGPT_sft.pt")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "camembert-base")
DATASET_NAME = os.getenv("SFT_DATASET_NAME", "BAAI/Infinity-Instruct")
DATASET_SUBSET = os.getenv("SFT_DATASET_SUBSET")
USER_KEY = os.getenv("SFT_USER_KEY", "user")
ASSISTANT_KEY = os.getenv("SFT_ASSISTANT_KEY", "assistant")
CONVERSATIONS_KEY = os.getenv("SFT_CONVERSATIONS_KEY")
LANGUAGE_FILTER = os.getenv("SFT_LANGUAGE_FILTER")
EVAL_EVERY_STEPS = int(os.getenv("EVAL_EVERY_STEPS", "500"))

num_epochs = config["training"]["num_epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
warmup = config["training"]["warmup"]
grad_accum_steps = config["training"].get("gradient_accumulation_steps", 1)
max_grad_norm = config["training"].get("max_grad_norm", 1.0)

embed_dim = config["model"]["embed_dim"]
depth = config["model"]["depth"]
heads = config["model"]["heads"]
num_kv_heads = config["model"].get("num_kv_heads", heads)
block_size = config["model"]["block_size"]
dropout = config["model"]["dropout"]
hidden_dim = config["model"]["hidden_dim"]
weight_sharing = config["model"].get("weight_sharing", "none")
use_rope = config["model"].get("use_rope", True)
use_gradient_checkpointing = config["model"].get("use_gradient_checkpointing", False)

max_texts = config["data"]["max_texts"]
train_split_ratio = config["data"]["train_split_ratio"]

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Tokenizer + chat tokens
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
original_vocab_size = len(tokenizer)
added_tokens = add_chat_tokens(tokenizer)
new_vocab_size = len(tokenizer)
print(f"[{now()}] Tokenizer: {TOKENIZER_NAME}")
print(f"   Vocab: {original_vocab_size} → {new_vocab_size} (+{new_vocab_size - original_vocab_size} chat tokens)")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_conversations():
    """Load and normalize conversations from the configured dataset.

    Supports two common HF dataset layouts:
      1. Flat: each row has a 'user' column and an 'assistant' column.
      2. Nested: each row has a 'conversations' list of {role, content} dicts
         (OpenAI / ShareGPT style).
    """
    print(f"[{now()}] Loading SFT dataset: {DATASET_NAME}")

    if DATASET_SUBSET:
        ds = load_dataset(DATASET_NAME, DATASET_SUBSET, split="train")
    else:
        ds = load_dataset(DATASET_NAME, split="train")

    if LANGUAGE_FILTER:
        lang_col = "language" if "language" in ds.column_names else "lang"
        if lang_col in ds.column_names:
            ds = ds.filter(lambda x: x[lang_col] == LANGUAGE_FILTER)
            print(f"   Filtered to language='{LANGUAGE_FILTER}': {len(ds)} rows")

    conversations = []
    count = min(max_texts, len(ds))

    if CONVERSATIONS_KEY and CONVERSATIONS_KEY in ds.column_names:
        for i in range(count):
            turns = ds[i][CONVERSATIONS_KEY]
            if not turns or not isinstance(turns, list):
                continue
            user_msg, assistant_msg = None, None
            for turn in turns:
                role = turn.get("role", turn.get("from", ""))
                content = turn.get("content", turn.get("value", ""))
                if role in ("user", "human") and user_msg is None:
                    user_msg = content
                elif role in ("assistant", "gpt") and assistant_msg is None:
                    assistant_msg = content
                if user_msg and assistant_msg:
                    break
            if user_msg and assistant_msg:
                conversations.append({"user": user_msg, "assistant": assistant_msg})
    else:
        for i in range(count):
            row = ds[i]
            u = row.get(USER_KEY, "")
            a = row.get(ASSISTANT_KEY, "")
            if u and a:
                conversations.append({"user": str(u), "assistant": str(a)})

    print(f"   Loaded {len(conversations)} conversation pairs")
    return conversations


conversations = load_conversations()

split = int(train_split_ratio * len(conversations))
train_convs = conversations[:split]
val_convs = conversations[split:]

train_ds = ChatDataset(train_convs, tokenizer, block_size)
val_ds = ChatDataset(val_convs, tokenizer, block_size)

print(f"[{now()}] Train: {len(train_ds)} | Val: {len(val_ds)}")


# ---------------------------------------------------------------------------
# Collate — pad to longest in batch
# ---------------------------------------------------------------------------
def collate_fn(batch):
    xs, ys = zip(*batch)
    pad_id = tokenizer.pad_token_id
    xs = [x[:block_size - 1] for x in xs]
    ys = [y[:block_size - 1] for y in ys]
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=-100)
    return xs_padded, ys_padded


train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# ---------------------------------------------------------------------------
# Model — load pretrained then resize embeddings
# ---------------------------------------------------------------------------
model_config = MiniGPTConfig(
    vocab_size=original_vocab_size,
    block_size=block_size,
    embed_dim=embed_dim,
    depth=depth,
    heads=heads,
    num_kv_heads=num_kv_heads,
    dropout=dropout,
    hidden_dim=hidden_dim,
    weight_sharing=weight_sharing,
    use_rope=use_rope,
    use_gradient_checkpointing=use_gradient_checkpointing,
)

model = MiniGPTForCausalLM(model_config)

if os.path.exists(PRETRAINED_PATH):
    checkpoint = torch.load(PRETRAINED_PATH, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {k: v for k, v in state.items() if "rope.cos_cached" not in k and "rope.sin_cached" not in k}
    model.load_state_dict(state, strict=False)
    print(f"[{now()}] Pretrained checkpoint loaded from {PRETRAINED_PATH}")
    pretrained_loss = checkpoint.get("val_loss", "N/A")
    pretrained_step = checkpoint.get("global_step", "N/A")
    print(f"   Pretrained val_loss={pretrained_loss} at step {pretrained_step}")
else:
    print(f"⚠️  No pretrained checkpoint at {PRETRAINED_PATH} — training from scratch!")

# Resize embeddings for the new chat tokens.
# Manual resize to guarantee weight tying is preserved.
if new_vocab_size != original_vocab_size:
    old_emb_weight = model.model.token_emb.weight.data
    mean_emb = old_emb_weight.mean(dim=0)

    new_emb = nn.Embedding(new_vocab_size, embed_dim)
    with torch.no_grad():
        new_emb.weight[:original_vocab_size] = old_emb_weight
        new_emb.weight[original_vocab_size:] = mean_emb

    model.model.token_emb = new_emb
    model.lm_head = nn.Linear(embed_dim, new_vocab_size, bias=False)
    model.lm_head.weight = model.model.token_emb.weight  # re-tie
    model.config.vocab_size = new_vocab_size
    print(f"[{now()}] Embeddings resized: {original_vocab_size} → {new_vocab_size}")

# Sanity check: verify token IDs are within vocab range
sample_ids = train_ds[0][0]
max_id = sample_ids.max().item()
assert max_id < new_vocab_size, f"Token ID {max_id} >= vocab_size {new_vocab_size}!"
print(f"[{now()}] Token ID sanity check passed (max_id={max_id}, vocab={new_vocab_size})")

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n{'='*70}")
print(f"SFT — MiniGPT Fine-Tuning")
print(f"{'='*70}")
print(f"   Parameters: {trainable_params:,} trainable / {total_params:,} total")
print(f"   LR: {learning_rate} | Warmup: {warmup} steps")
print(f"   Batch: {batch_size} x {grad_accum_steps} accum = {batch_size * grad_accum_steps} effective")
print(f"   Block size: {block_size}")
print(f"   Epochs: {num_epochs}")
print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------------------------
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.ndim >= 2:
        decay_params.append(param)
    else:
        no_decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=learning_rate,
    betas=(0.9, 0.95),
    eps=1e-8,
    fused=torch.cuda.is_available(),
)

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

total_steps = num_epochs * (len(train_loader) // grad_accum_steps)


def warmup_then_cosine(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


scheduler = warmup_then_cosine(optimizer, warmup_steps=warmup, total_steps=total_steps)

# Resume SFT checkpoint if it exists
start_epoch = 0
best_loss = float("inf")
global_step = 0

if os.path.exists(SFT_SAVE_PATH):
    sft_ckpt = torch.load(SFT_SAVE_PATH, map_location=device)
    model.load_state_dict(sft_ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(sft_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in sft_ckpt:
        scheduler.load_state_dict(sft_ckpt["scheduler_state_dict"])
    start_epoch = sft_ckpt.get("epoch", 0) + 1
    best_loss = sft_ckpt.get("val_loss", float("inf"))
    global_step = sft_ckpt.get("global_step", 0)
    print(f"[{now()}] Resumed SFT from epoch {start_epoch}, step {global_step}, best_loss={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_val_loss():
    model.eval()
    total, count = 0.0, 0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(xb).logits
            B, T, C = logits.shape
            loss = loss_fn(logits.view(B * T, C), yb.view(B * T))
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Generation sample (chat format)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_sample(prompt="Quelle est la capitale de la France ?"):
    model.eval()
    text = format_chat(prompt, "", None)
    # Remove the trailing <|end|> so the model continues generating the response
    end_tag = SPECIAL_TOKENS["end"]
    if text.endswith(end_tag):
        text = text[: -len(end_tag)]

    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    with torch.amp.autocast("cuda"):
        output = model.generate(
            input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.convert_tokens_to_ids(end_tag),
            min_new_tokens=5,
        )

    gen_tokens = output[0][input_ids.shape[-1]:]
    response = tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True).strip()
    model.train()
    return response


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
trackio.init(
    project="mini-gpt-sft",
    name=f"sft_{embed_dim}d_{depth}L",
    config=config,
    resume="allow",
)

scaler = torch.amp.GradScaler("cuda")

for epoch in range(start_epoch, num_epochs):
    print(f"\n{'='*70}")
    print(f"[{now()}] SFT Epoch {epoch + 1}/{num_epochs}")
    print(f"{'='*70}")
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0

    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)

        with torch.amp.autocast("cuda"):
            logits = model(xb).logits
            B, T, C = logits.shape
            loss = loss_fn(logits.view(B * T, C), yb.view(B * T))
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        running_loss += loss.item()

        is_accum_step = (i + 1) % grad_accum_steps == 0
        is_last_batch = (i + 1) == len(train_loader)

        if is_accum_step or is_last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            if global_step % 50 == 0:
                trackio.log(
                    {"train/loss": running_loss, "lr": scheduler.get_last_lr()[0], "epoch": epoch + 1},
                    step=global_step,
                )

            if global_step % 100 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"[{now()}] [Epoch {epoch + 1} | Step {global_step}] loss={running_loss:.4f} | LR={lr:.2e}")

            running_loss = 0.0

            if global_step % EVAL_EVERY_STEPS == 0:
                val_loss = compute_val_loss()
                improvement = best_loss - val_loss
                print(f"[{now()}] [Step {global_step}] val_loss={val_loss:.4f} (best: {best_loss:.4f}, diff: {improvement:+.4f})")

                if val_loss < best_loss:
                    best_loss = val_loss
                    model_to_save = model
                    torch.save(
                        {
                            "model_state_dict": model_to_save.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "val_loss": val_loss,
                            "vocab_size": new_vocab_size,
                        },
                        SFT_SAVE_PATH,
                    )
                    print(f"[{now()}] ✅ New best SFT model saved! (val_loss: {val_loss:.4f})")
                else:
                    print(f"[{now()}] ⚠️  No improvement")

                trackio.log(
                    {"val/loss": val_loss, "best_val_loss": best_loss},
                    step=global_step,
                )

                response = generate_sample()
                print(f"[{now()}] 💬 Exemple: Q='Quelle est la capitale de la France ?'")
                print(f"   R: {response}")

                model.train()

trackio.finish()
print(f"\n[{now()}] SFT terminé. Meilleur modèle: {SFT_SAVE_PATH} (val_loss: {best_loss:.4f})")
