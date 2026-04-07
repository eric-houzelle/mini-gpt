import os
import json
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import string
from datasets import load_dataset
from torch.optim.lr_scheduler import OneCycleLR
from dataset.text_dataset import TextDataset, pretokenize
from model.configuration import MiniGPTConfig
from model.modeling_minigpt import MiniGPTForCausalLM
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv
import torch.nn.functional as F
import trackio
import math
load_dotenv()

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DATASET_NAME = os.getenv("DATASET_NAME", "iproskurina/TinyStories-French")
DATASET_KEY = os.getenv("DATASET_KEY", "french-tinystories")
DATASET_TEMPLATE = os.getenv("DATASET_TEMPLATE")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "camembert-base")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "checkpoints/best_miniGPT.pt")
EVAL_PROMPT = os.getenv("EVAL_PROMPT", "Prompt: En quelle année est sortie la chanson 'Baby' de Justin Bieber ? Answer: ")
EVAL_EVERY_STEPS = int(os.getenv("EVAL_EVERY_STEPS", "500"))
DATASET_FILTER_FIELD = os.getenv("DATASET_FILTER_FIELD")
DATASET_FILTER_VALUE = os.getenv("DATASET_FILTER_VALUE")
num_epochs = config["training"]["num_epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
warmup = config["training"]["warmup"]
override_lr = os.getenv("OVERRIDE_LR") or config["training"].get("override_lr")
override_best_loss = os.getenv("OVERRIDE_BEST_LOSS") or config["training"].get("override_best_loss")

embed_dim = config["model"]["embed_dim"]
depth = config["model"]["depth"]
heads = config["model"]["heads"]
num_kv_heads = config["model"].get("num_kv_heads", heads)
block_size = config["model"]["block_size"]
dropout = config["model"]["dropout"]
hidden_dim = config["model"]["hidden_dim"]
weight_sharing = config["model"].get("weight_sharing", "none")  # STLM: weight sharing entre blocs
use_rope = config["model"].get("use_rope", True)  # STLM: RoPE embeddings
use_gradient_checkpointing = config["model"].get("use_gradient_checkpointing", True)

max_texts = config["data"]["max_texts"]
train_split_ratio = config["data"]["train_split_ratio"]


def load_tokenizer(tokenizer_name):
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
#    special_tokens = {
#        "additional_special_tokens": [
#            "<|user|>",
#            "<|assistant|>"
#        ]
#    }

#    tok.add_special_tokens(special_tokens)

    return tok

tokenizer = load_tokenizer(TOKENIZER_NAME)

print(f"LOAD DATASET: {DATASET_NAME}")
# Support pour les fichiers CSV locaux
csv_path = None
if DATASET_NAME.endswith('.csv'):
    # Chemin relatif ou absolu vers un fichier CSV
    if os.path.exists(DATASET_NAME):
        csv_path = DATASET_NAME
    elif os.path.exists(os.path.abspath(DATASET_NAME)):
        csv_path = os.path.abspath(DATASET_NAME)

if csv_path:
    print(f"📁 Loading local CSV file: {csv_path}")
    dataset = load_dataset("csv", data_files=csv_path)
    train_split_ds = dataset["train"]
    print(f"✅ CSV file loaded successfully")
else:
    dataset = load_dataset(DATASET_NAME)
    train_split_ds = dataset["train"]

if DATASET_FILTER_FIELD and DATASET_FILTER_VALUE:
    if DATASET_FILTER_FIELD not in train_split_ds.column_names:
        print(f"⚠️ Warning: Filter field '{DATASET_FILTER_FIELD}' not found. Available columns: {train_split_ds.column_names}")
    else:
        # Support pour plusieurs termes séparés par des virgules (Logique OR)
        filter_values = [v.strip().lower() for v in DATASET_FILTER_VALUE.split(",")]
        print(f"🔍 Filtering dataset: rows where '{DATASET_FILTER_FIELD}' contains ANY of {filter_values}")
        train_split_ds = train_split_ds.filter(
            lambda x: any(val in str(x[DATASET_FILTER_FIELD]).lower() for val in filter_values)
        )
        print(f"✅ Filtered dataset: {len(train_split_ds)} rows remaining")


if DATASET_TEMPLATE:
    class _SafeDict(dict):
        def __missing__(self, key):
            return ""

    max_items = min(max_texts, len(train_split_ds))
    subset = train_split_ds.select(range(max_items))
    texts = [DATASET_TEMPLATE.format_map(_SafeDict(example)) for example in subset]
    print("Using DATASET_TEMPLATE to format structured samples")
else:
    if DATASET_KEY not in train_split_ds.column_names:
        raise ValueError(
            f"Column '{DATASET_KEY}' not found in dataset. Available columns: {train_split_ds.column_names}"
        )
    texts = train_split_ds[DATASET_KEY][:max_texts]

print(f"⏳ Pre-tokenizing {len(texts)} texts (one-time cost)...")
all_token_ids = pretokenize(texts, tokenizer, block_size)
print(f"✅ Pre-tokenized: {len(all_token_ids)} sequences kept (>= 2 tokens)")

split = int(train_split_ratio * len(all_token_ids))
train_ds = TextDataset(all_token_ids[:split])
val_ds   = TextDataset(all_token_ids[split:])

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def collate_fn(batch):
    xs, ys = zip(*batch)

    max_len = block_size - 1 
    pad_id = tokenizer.pad_token_id

    # Tronquer et pad pour garantir une longueur fixe (évite les formes dynamiques CUDAGraph)
    xs = [x[:max_len] if len(x) > max_len else x for x in xs]
    ys = [y[:max_len] if len(y) > max_len else y for y in ys]

    xs = [torch.nn.functional.pad(x, (0, max_len - len(x)), value=pad_id) for x in xs]
    ys = [torch.nn.functional.pad(y, (0, max_len - len(y)), value=pad_id) for y in ys]

    return torch.stack(xs), torch.stack(ys)

def compute_validation_loss(model, val_loader, loss_fn, device):
    """Calcule la loss de validation moyenne sur l'ensemble du loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb_val, yb_val in val_loader:
            xb_val, yb_val = xb_val.to(device), yb_val.to(device)
            logits = model(xb_val).logits
            B, T, C = logits.shape
            total_loss += loss_fn(logits.view(B * T, C), yb_val.view(B * T)).item()
    return total_loss / len(val_loader)

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("inf")):
    """
    logits: tenseur de forme (1, vocab_size)
    Retourne un tenseur de même forme avec certains logits mis à -inf.
    """
    # On clone pour ne pas modifier in-place sans le vouloir
    logits = logits.clone()

    # On travaille en 1D (batch = 1)
    assert logits.dim() == 2 and logits.size(0) == 1, "Cette implémentation suppose batch_size=1"
    logits_1d = logits[0]  # shape: (vocab_size,)

    # --- Top-K ---
    if top_k > 0:
        top_k = min(top_k, logits_1d.size(-1))  # sécurité
        # seuil = plus petit logit parmi les top_k plus grands
        kth_values = torch.topk(logits_1d, top_k)[0][-1]
        indices_to_remove = logits_1d < kth_values
        logits_1d[indices_to_remove] = filter_value

    # --- Top-P (nucleus) ---
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits_1d, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        # On récupère les ids de tokens à masquer
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits_1d[indices_to_remove] = filter_value

    # On remet en (1, vocab_size)
    logits[0] = logits_1d
    return logits


def generate_example_v2(
    model,
    tokenizer,
    block_size,
    device,
    dataset_template,
    eval_prompt,
    max_new_tokens=128,
    min_new_tokens=20,
    temperature=0.8,
    top_k=0,
    top_p=0.9,
):
    model.eval()

    # Utiliser le modèle non compilé pour la génération
    gen_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # --- Prompt ---
    if dataset_template:
        class _SafeDict(dict):
            def __missing__(self, key):
                return eval_prompt

        formatter = string.Formatter()
        fields = {f for _, f, _, _ in formatter.parse(dataset_template) if f}
        fmt = _SafeDict({f: eval_prompt for f in fields})
        example_prompt = dataset_template.format_map(fmt)
        prompt_ids = tokenizer.encode(example_prompt, return_tensors="pt").to(device)
    else:
        example_prompt = eval_prompt
        prompt_ids = tokenizer.encode(eval_prompt, return_tensors="pt").to(device)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    idx = prompt_ids

    with torch.no_grad():
        # 🔒 Désactiver AMP uniquement ici
        with torch.cuda.amp.autocast(enabled=False):
            for step in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]

                logits = gen_model(idx_cond).logits[:, -1, :].float()

                if temperature != 1.0:
                    logits = logits / temperature

                # Empêcher EOS trop tôt (AVANT softmax)
                if eos_id is not None and step < min_new_tokens:
                    logits[:, eos_id] = -float("inf")

                logits = top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p
                )

                probs = torch.softmax(logits, dim=-1)

                # --- Sécurité numérique ---
                if not torch.isfinite(probs).all():
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

                probs_sum = probs.sum(dim=-1, keepdim=True)

                if (probs_sum <= 0).any():
                    # fallback neutre (ultra rare)
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = probs / probs_sum
                    next_token = torch.multinomial(probs, 1)

                idx = torch.cat([idx, next_token], dim=1)

                if eos_id is not None and step >= min_new_tokens:
                    if next_token.item() == eos_id:
                        break

    sample = idx[0]
    gen_tokens = sample[prompt_ids.shape[-1]:].tolist()

    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    gen_text_raw = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()

    return example_prompt, gen_text, gen_text_raw, gen_tokens



train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


device = "cuda" if torch.cuda.is_available() else "cpu"

# Créer la configuration Hugging Face
model_config = MiniGPTConfig(
    vocab_size=len(tokenizer),
    block_size=block_size,
    embed_dim=embed_dim,
    depth=depth,
    heads=heads,
    num_kv_heads=num_kv_heads,
    dropout=dropout,
    hidden_dim=hidden_dim,
    weight_sharing=weight_sharing,
    use_rope=use_rope,
    use_gradient_checkpointing=use_gradient_checkpointing
)

model = MiniGPTForCausalLM(model_config).to(device)
#model.resize_token_embeddings(len(tokenizer))
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Afficher les stats détaillées du modèle
model_stats = model.count_parameters()

def warmup_then_constant(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0  # LR = learning_rate
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
gqa_label = f"GQA {heads}Q/{num_kv_heads}KV" if num_kv_heads < heads else f"MHA {heads}"
print(f"\n📊 Architecture:")
print(f"   - Layers: {depth}")
print(f"   - Embed dim: {embed_dim}")
print(f"   - Attention: {gqa_label}")
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
    if override_best_loss is not None:
        best_loss = float(override_best_loss)
    else:
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

scheduler = warmup_then_constant(optimizer, warmup_steps=warmup)

if scheduler_state_dict is not None:
    scheduler.load_state_dict(scheduler_state_dict)

# Option d'override du LR lors d'une reprise : on garde l'état optimizer/scheduler
# mais on force la nouvelle base de LR pour tous les param_groups et le scheduler.
if override_lr is not None:
    override_lr = float(override_lr)
    for pg in optimizer.param_groups:
        pg["lr"] = override_lr
    scheduler.base_lrs = [override_lr] * len(optimizer.param_groups)
    print(f"🔧 Override LR actif -> nouveau LR de base: {override_lr}")

trackio.init(
    project="mini-gpt-1511-v5",
    name=f"mini-gpt_{config['model']['embed_dim']}d_{config['model']['depth']}L",
    config=config,
    resume="allow"
)

scaler = torch.amp.GradScaler("cuda")
model = torch.compile(model, mode="reduce-overhead")

epochs_without_improvement = 0
patience = 10

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
            logits = model(xb).logits
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
            trackio.log(
                {
                    "train/loss": loss.item(),
                    "epoch": epoch + 1,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=i,
            )

        if i % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[{now()}] [Epoch {epoch+1} | Step {i}] train_loss={loss.item():.4f} | LR={current_lr:.2e}")

        # Validation et génération périodiques
        if global_step % EVAL_EVERY_STEPS == 0:
            val_loss = compute_validation_loss(model, val_loader, loss_fn, device)
            improvement = best_loss - val_loss
            if best_loss != float("inf"):
                print(f"[{now()}] [step {global_step}] Validation loss: {val_loss:.4f} (best: {best_loss:.4f}, diff: {improvement:+.4f})")
            else:
                print(f"[{now()}] [step {global_step}] Validation loss: {val_loss:.4f}")

            if val_loss < best_loss:
                model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
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
                print(f"[{now()}] ⚠️  No improvement for {epochs_without_improvement} eval(s)")

                if epochs_without_improvement >= patience:
                    print(f"[{now()}] 💡 Consider reducing learning rate or stopping soon (no improvement for {epochs_without_improvement} evals)")

            trackio.log(
                {
                    "val/loss": val_loss,
                    "best_val_loss": best_loss,
                    "epochs_without_improvement": epochs_without_improvement,
                    "epoch": epoch + 1,
                },
                step=global_step,
            )
                
            _, gen_text, gen_text_raw, gen_tokens = generate_example_v2(
                model,
                tokenizer,
                block_size,
                device,
                DATASET_TEMPLATE,
                EVAL_PROMPT,
                temperature=0.7, # Température légèrement réduite
                top_k=0,
                top_p=0.9
            )
            print(f"[{now()}] Exemple génération v2 (suite de l'invite):\n{gen_text}")
            if not gen_text:
                print(f"[DEBUG] gen_tokens v2 (len={len(gen_tokens)}): {gen_tokens}")
                print(f"[DEBUG] gen_text_raw v2: {gen_text_raw}")

            model.train()
trackio.finish()
