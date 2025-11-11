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
scheduler_max_lr = config["training"]["scheduler_max_lr"]

embed_dim = config["model"]["embed_dim"]
depth = config["model"]["depth"]
heads = config["model"]["heads"]
block_size = config["model"]["block_size"]
dropout = config["model"]["dropout"]
hidden_dim = config["model"]["hidden_dim"]

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
model = MiniGPT(len(tokenizer), block_size, embed_dim=embed_dim, depth=depth, heads=heads, dropout=dropout, hidden_dim= hidden_dim).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

def human_readable(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)

print(f"\nModel initialized with:")
print(f" - {human_readable(total_params)} total parameters")
print(f" - {human_readable(trainable_params)} trainable parameters")



optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


if os.path.exists(MODEL_SAVE_PATH):
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('loss', float("inf"))
    last_epoch = start_epoch * len(train_loader)
else:
    start_epoch = 0
    best_loss = float("inf")
    last_epoch = -1

scheduler = OneCycleLR(
    optimizer,
    max_lr=scheduler_max_lr,
    total_steps=num_epochs * len(train_loader),
    last_epoch=last_epoch
)

trackio.init(
    project="mini-gpt",
    name=f"mini-gpt_{config['model']['embed_dim']}d_{config['model']['depth']}L",
    config=config, 
)

scaler = torch.amp.GradScaler("cuda")
model = torch.compile(model)
for epoch in range(num_epochs):
    print(f"\n[{now()}] === Epoch {epoch+1}/{num_epochs} ===")
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
            print(f"[{now()}] [Epoch {epoch+1} | Step {i}] loss={loss.item():.4f}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss.item()
                }, MODEL_SAVE_PATH)
                print(f"[{now()}] New best model saved!")

    if epoch % 20 == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                B, T, C = logits.shape
                val_loss += loss_fn(logits.view(B*T, C), yb.view(B*T)).item()
        val_loss /= len(val_loader)
        print(f"[{now()}] Validation loss: {val_loss:.4f}")
        
    trackio.log({
        "val/loss": val_loss,
        "epoch": epoch + 1
        })


    
    model.eval()
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=50)[0].tolist()
    print(f"[{now()}] Exemple génération:", tokenizer.decode(out, skip_special_tokens=True))
trackio.finish()