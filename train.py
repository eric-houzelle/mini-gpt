import os
import json
import math
import torch
import trackio
import torch.nn as nn
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import CamembertTokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence

from dataset.text_dataset import TextDataset
from model.model import MiniGPT

# =====================================================================
# INIT CONFIG
# =====================================================================

load_dotenv()

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DATASET_NAME = os.getenv("DATASET_NAME", "iproskurina/TinyStories-French")
DATASET_KEY  = os.getenv("DATASET_KEY", "french-tinystories")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "camembert-base")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "checkpoints/best_miniGPT.pt")

num_epochs     = config["training"]["num_epochs"]
batch_size     = config["training"]["batch_size"]
learning_rate  = config["training"]["learning_rate"]
warmup  = config["training"]["warmup"]

embed_dim  = config["model"]["embed_dim"]
depth      = config["model"]["depth"]
heads      = config["model"]["heads"]
block_size = config["model"]["block_size"]
dropout    = config["model"]["dropout"]
hidden_dim = config["model"]["hidden_dim"]


max_texts = config["data"]["max_texts"]
train_split_ratio = config["data"]["train_split_ratio"]

# =====================================================================
# DATASET
# =====================================================================

tokenizer = CamembertTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"LOAD DATASET: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME)
texts = dataset["train"][DATASET_KEY][:max_texts]

split = int(train_split_ratio * len(texts))
train_ds = TextDataset(texts[:split], tokenizer, block_size)
val_ds   = TextDataset(texts[split:], tokenizer, block_size)

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True, padding_value=tokenizer.pad_token_id)
    ys = pad_sequence(ys, batch_first=True, padding_value=tokenizer.pad_token_id)
    return xs, ys

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# =====================================================================
# MODEL
# =====================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MiniGPT(
    len(tokenizer),
    block_size,
    embed_dim=embed_dim,
    depth=depth,
    heads=heads,
    dropout=dropout,
    hidden_dim=hidden_dim
).to(device)

total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel:")
print(f" - total params     : {total_params:,}")
print(f" - trainable params : {train_params:,}")

# =====================================================================
# LOSS
# =====================================================================

loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id,
    label_smoothing=0.05
)

# =====================================================================
# OPTIMIZER (GPT-style weight decay)
# =====================================================================

decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.ndim >= 2:
        decay_params.append(param)   # weight matrices
    else:
        no_decay_params.append(param)  # bias, embeddings, RMSNorm weights

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=learning_rate,
    betas=(0.9, 0.95),
    eps=1e-8
)

# =====================================================================
# LR SCHEDULER : COSINE + WARMUP
# =====================================================================

total_steps = num_epochs * len(train_loader)

def cosine_with_warmup(step):
    warmup_steps = warmup
    if step < warmup_steps:
        return step / warmup_steps
    
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return max(0.1, cosine)

scheduler = LambdaLR(optimizer, lr_lambda=cosine_with_warmup)

# =====================================================================
# RESUME TRAINING
# =====================================================================

if os.path.exists(MODEL_SAVE_PATH):
    print("\n>>> RESUME TRAINING <<<")

    ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    best_loss = ckpt.get("loss", float("inf"))
    global_step = ckpt.get("global_step", start_epoch * len(train_loader))

else:
    start_epoch = 0
    best_loss = float("inf")
    global_step = 0

# =====================================================================
# TRACKING
# =====================================================================

trackio.init(
    project="mini-gpt-finetune",
    name=f"MiniGPT_{embed_dim}d_{depth}L",
    config=config,
)

# =====================================================================
# TRAIN LOOP
# =====================================================================

scaler = torch.amp.GradScaler("cuda")
model = torch.compile(model)

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for epoch in range(start_epoch, num_epochs):

    print(f"\n[{now()}] === Epoch {epoch+1}/{num_epochs} ===")

    model.train()
    for i, (xb, yb) in enumerate(train_loader):

        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(xb)
            B, T, C = logits.size()
            loss = loss_fn(logits.view(B*T, C), yb.view(B*T))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        global_step += 1

        if i % 100 == 0:
            print(f"[{now()}] step={i} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        # SAVE BEST
        if loss.item() < best_loss:
            best_loss = loss.item()
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "loss": best_loss
            }, MODEL_SAVE_PATH)
            print(f"[{now()}] ** Best model saved! loss={best_loss:.4f} **")

    # =================================================================
    # VALIDATION
    # =================================================================

    if epoch % 2 == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    B, T, C = logits.shape
                    val_loss += loss_fn(logits.view(B*T, C), yb.view(B*T)).item()

        val_loss /= len(val_loader)
        print(f"[{now()}] Validation loss: {val_loss:.4f}")

        trackio.log({"val/loss": val_loss})

        if not hasattr(model, 'best_val_loss'):
            model.best_val_loss = float('inf')
    
        if val_loss < model.best_val_loss:
            model.best_val_loss = val_loss
            
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            
            torch.save({
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": best_loss,
                "val_loss": val_loss
            }, MODEL_SAVE_PATH.replace('.pt', '_best_val.pt'))
            
            print(f"[{now()}] 💎 Best VAL model saved! val_loss={val_loss:.4f}")

    # =================================================================
    # Exemple génération
    # =================================================================

    model.eval()
    with torch.no_grad():
        context = torch.zeros((1,1), dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=80)[0].tolist()
        print(f"[{now()}] Exemple:", tokenizer.decode(out, skip_special_tokens=True))

trackio.finish()
