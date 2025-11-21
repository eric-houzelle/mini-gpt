import os
import json
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CamembertTokenizer
from datasets import load_dataset
from torch.optim.lr_scheduler import OneCycleLR
from dataset.text_dataset import TextDataset
from model.model import MiniGPT
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv
import trackio
import math

load_dotenv()

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DATASET_NAME = os.getenv("DATASET_NAME", "iproskurina/TinyStories-French")
DATASET_KEY = os.getenv("DATASET_KEY", "french-tinystories")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "camembert-base")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "checkpoints/best_miniGPT.pt")

num_epochs = config["training"]["num_epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
warmup = config["training"]["warmup"]

embed_dim = config["model"]["embed_dim"]
depth = config["model"]["depth"]
heads = config["model"]["heads"]
block_size = config["model"]["block_size"]
dropout = config["model"]["dropout"]
hidden_dim = config["model"]["hidden_dim"]
weight_sharing = config["model"].get("weight_sharing", "none")  # STLM: weight sharing entre blocs
use_rope = config["model"].get("use_rope", True)  # STLM: RoPE embeddings

max_texts = config["data"]["max_texts"]
train_split_ratio = config["data"]["train_split_ratio"]

tokenizer = CamembertTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"LOAD DATASET: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME)
texts = dataset["train"][DATASET_KEY][:max_texts]

split = int(train_split_ratio * len(texts))
train_ds = TextDataset(texts[:split], tokenizer, block_size)
val_ds   = TextDataset(texts[split:], tokenizer, block_size)

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True, padding_value=tokenizer.pad_token_id)
    ys = pad_sequence(ys, batch_first=True, padding_value=tokenizer.pad_token_id)
    return xs, ys

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(
    len(tokenizer), 
    block_size, 
    embed_dim=embed_dim, 
    depth=depth, 
    heads=heads, 
    dropout=dropout, 
    hidden_dim=hidden_dim,
    weight_sharing=weight_sharing,  # STLM: "none", "ffn" ou "full"
    use_rope=use_rope  # STLM: RoPE au lieu de learned pos embeddings
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Afficher les stats détaillées du modèle
model_stats = model.count_parameters()

def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def human_readable(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)

print(f"\n{'='*70}")
print(f"MODEL INITIALIZED - Super Tiny Language Model (STLM)")
print(f"{'='*70}")
print(f"\n📊 Architecture:")
print(f"   - Layers: {depth}")
print(f"   - Embed dim: {embed_dim}")
print(f"   - Heads: {heads}")
print(f"   - Hidden dim: {hidden_dim}")
print(f"   - Block size: {block_size}")
print(f"\n🔬 STLM Techniques:")
print(f"   - Weight sharing: {weight_sharing.upper()}")
print(f"   - Position encoding: {'RoPE' if use_rope else 'Learned'}")
print(f"   - FFN activation: SwiGLU")
print(f"\n📈 Parameters:")
print(f"   - Total: {human_readable(total_params)} ({total_params:,})")
print(f"   - Trainable: {human_readable(trainable_params)} ({trainable_params:,})")
print(f"   - Token embeddings: {human_readable(model_stats['token_emb'])}")
print(f"   - Pos embeddings: {human_readable(model_stats['pos_emb'])}")
print(f"   - Transformer blocks: {human_readable(model_stats['blocks'])}")
print(f"{'='*70}\n")


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

loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id,
    label_smoothing=0.05
)


if os.path.exists(MODEL_SAVE_PATH):
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('val_loss', checkpoint.get('loss', float("inf")))  # Priorité à val_loss
    scheduler_state_dict = checkpoint.get("scheduler_state_dict", None)
    global_step = checkpoint.get('global_step', start_epoch * len(train_loader))
    last_epoch = global_step - 1
    print(f"\n✅ Checkpoint loaded from epoch {checkpoint['epoch']}")
    print(f"   Best validation loss so far: {best_loss:.4f}")
else:
    start_epoch = 0
    best_loss = float("inf")
    global_step = 0
    last_epoch = -1
    scheduler_state_dict = None
    print(f"\n🆕 Starting fresh training")

total_steps = num_epochs * len(train_loader)
scheduler = cosine_with_warmup(
    optimizer,
    warmup_steps=warmup,
    total_steps=total_steps,
    min_lr_ratio=0.1
)

if scheduler_state_dict is not None:
    scheduler.load_state_dict(scheduler_state_dict)

trackio.init(
    project="mini-gpt-1511-v5",
    name=f"mini-gpt_{config['model']['embed_dim']}d_{config['model']['depth']}L",
    config=config,
    resume="allow"
)

scaler = torch.amp.GradScaler("cuda")
model = torch.compile(model, mode="reduce-overhead")

epochs_without_improvement = 0
patience = 10  # Nombre d'epochs sans amélioration avant warning

for epoch in range(start_epoch, num_epochs):
    print(f"\n{'='*70}")
    print(f"[{now()}] Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*70}")
    model.train()
    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        # logits = model(xb)
        # B, T, C = logits.shape
        # loss = loss_fn(logits.view(B*T, C), yb.view(B*T))
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(xb)
            B, T, C = logits.shape
            loss = loss_fn(logits.view(B*T, C), yb.view(B*T))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1 
        
        ## without amp scaler
        # loss.backward() 
        # optimizer.step()
        # scheduler.step()
        
        if i % 50 == 0:
            trackio.log({
                "train/loss": loss.item(),
                "epoch": epoch + 1,
                "step": i,
                "lr": scheduler.get_last_lr()[0]
            })  

        if i % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[{now()}] [Epoch {epoch+1} | Step {i}] train_loss={loss.item():.4f} | LR={current_lr:.2e}")
    
    # Validation à chaque epoch
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            B, T, C = logits.shape
            val_loss += loss_fn(logits.view(B*T, C), yb.view(B*T)).item()
    val_loss /= len(val_loader)
    
    # Afficher et comparer avec le meilleur
    improvement = best_loss - val_loss
    if best_loss != float("inf"):
        print(f"[{now()}] Validation loss: {val_loss:.4f} (best: {best_loss:.4f}, diff: {improvement:+.4f})")
    else:
        print(f"[{now()}] Validation loss: {val_loss:.4f}")
    
    # Sauvegarder si la validation loss s'améliore
    if val_loss < best_loss:
        if hasattr(model, '_orig_mod'):
            model_to_save = model._orig_mod
        else:
            model_to_save = model
        
        best_loss = val_loss
        epochs_without_improvement = 0
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'val_loss': val_loss
        }, MODEL_SAVE_PATH)
        print(f"[{now()}] ✅ New best model saved! (val_loss: {val_loss:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"[{now()}] ⚠️  No improvement for {epochs_without_improvement} epoch(s)")
        
        if epochs_without_improvement >= patience:
            print(f"[{now()}] 💡 Consider reducing learning rate or stopping soon (no improvement for {epochs_without_improvement} epochs)")
    
    # Log sur trackio
    trackio.log({
        "val/loss": val_loss,
        "best_val_loss": best_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "epoch": epoch + 1
    })


    
    model.eval()
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=50, temperature=0.8, top_p=0.9)[0].tolist()
    print(f"[{now()}] Exemple génération:", tokenizer.decode(out, skip_special_tokens=True))
trackio.finish()
