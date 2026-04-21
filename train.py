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
from dataset.text_dataset import TextDataset, LazyTextDataset, pretokenize_cached
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
EVAL_EVERY_STEPS = int(os.getenv("EVAL_EVERY_STEPS", "1000"))
DATASET_FILTER_FIELD = os.getenv("DATASET_FILTER_FIELD")
DATASET_FILTER_VALUE = os.getenv("DATASET_FILTER_VALUE")
num_epochs = config["training"]["num_epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
warmup = config["training"]["warmup"]
override_lr = os.getenv("OVERRIDE_LR") or config["training"].get("override_lr")
override_best_loss = os.getenv("OVERRIDE_BEST_LOSS") or config["training"].get("override_best_loss")
grad_accum_steps = config["training"].get("gradient_accumulation_steps", 1)

embed_dim = config["model"]["embed_dim"]
depth = config["model"]["depth"]
heads = config["model"]["heads"]
num_kv_heads = config["model"].get("num_kv_heads", heads)
block_size = config["model"]["block_size"]
dropout = config["model"]["dropout"]
hidden_dim = config["model"]["hidden_dim"]
weight_sharing = config["model"].get("weight_sharing", "none")
use_rope = config["model"].get("use_rope", True)
use_gradient_checkpointing = config["model"].get("use_gradient_checkpointing", True)

# Recurrent-Depth Transformer params
rdt_config = config["model"].get("recurrent_depth", {})
num_prelude_layers = rdt_config.get("num_prelude_layers", 2)
num_coda_layers = rdt_config.get("num_coda_layers", 2)
num_recurrent_steps = rdt_config.get("num_recurrent_steps", 8)
use_lti_injection = rdt_config.get("use_lti_injection", True)
use_act_halting = rdt_config.get("use_act_halting", True)
act_halt_threshold = rdt_config.get("act_halt_threshold", 0.99)
depth_lora_rank = rdt_config.get("depth_lora_rank", 8)
act_loss_weight = rdt_config.get("act_loss_weight", 0.01)

max_texts = config["data"].get("max_texts", config["data"].get("max_texts_per_dataset", 500000))
max_texts_per_ds = config["data"].get("max_texts_per_dataset", max_texts)
train_split_ratio = config["data"]["train_split_ratio"]
use_pretokenize = config["data"].get("pretokenize", True)
progressive_schedule = config["data"].get("progressive_block_sizes")
dataset_configs = config.get("datasets", [])


def load_tokenizer(tokenizer_name):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
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

def _load_single_pretrain_dataset(ds_cfg):
    """Load one pretraining dataset and return a list of text strings.

    Supports "data_files" to download only specific parquet files instead of
    the entire dataset (critical for huge datasets like fineweb-2).
    """
    name = ds_cfg["name"]
    text_key = ds_cfg.get("text_key", "text")
    subset = ds_cfg.get("subset")
    max_n = ds_cfg.get("max_texts", max_texts_per_ds)
    template = ds_cfg.get("template")
    data_files = ds_cfg.get("data_files")

    label = f"  → Loading {name}"
    if subset:
        label += f" [{subset}]"
    if data_files:
        label += f" (data_files={data_files})"
    print(label)

    if name.endswith('.csv') and os.path.exists(name):
        ds = load_dataset("csv", data_files=name, split="train")
    elif data_files:
        if subset:
            ds = load_dataset(name, subset, data_files=data_files, split="train")
        else:
            ds = load_dataset(name, data_files=data_files, split="train")
    elif subset:
        ds = load_dataset(name, subset, split="train")
    else:
        ds = load_dataset(name, split="train")

    filter_col = ds_cfg.get("filter_column")
    include_values = ds_cfg.get("include_values")
    exclude_values = ds_cfg.get("exclude_values")
    if filter_col and filter_col in ds.column_names:
        if include_values:
            allowed = {v.lower() for v in include_values}
            ds = ds.filter(lambda x: str(x[filter_col]).lower() in allowed)
            print(f"    Include filter on '{filter_col}': {len(ds)} rows")
        elif exclude_values:
            blocked = {v.lower() for v in exclude_values}
            ds = ds.filter(lambda x: str(x[filter_col]).lower() not in blocked)
            print(f"    Exclude filter on '{filter_col}': {len(ds)} rows")

    count = min(max_n, len(ds))
    print(f"    Dataset loaded: {len(ds)} rows, taking {count}")

    if template:
        class _SafeDict(dict):
            def __missing__(self, key):
                return ""
        subset_ds = ds.select(range(count))
        result = [template.format_map(_SafeDict(row)) for row in subset_ds]
    else:
        if text_key not in ds.column_names:
            raise ValueError(
                f"Column '{text_key}' not found in {name}. Available: {ds.column_names}"
            )
        result = ds[text_key][:count]

    result = [t for t in result if t and len(t.strip()) > 10]
    print(f"    → {len(result)} texts extracted")
    return result


if dataset_configs:
    print(f"Loading {len(dataset_configs)} pretraining dataset(s)...")
    texts = []
    for ds_cfg in dataset_configs:
        texts.extend(_load_single_pretrain_dataset(ds_cfg))
    import random
    random.shuffle(texts)
    print(f"Total: {len(texts)} texts (shuffled)")
else:
    print(f"LOAD DATASET: {DATASET_NAME}")
    csv_path = None
    if DATASET_NAME.endswith('.csv'):
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

if use_pretokenize:
    print(f"⏳ Pre-tokenizing {len(texts)} texts...")
    all_token_ids = pretokenize_cached(texts, tokenizer, block_size)
    print(f"✅ Pre-tokenized: {len(all_token_ids)} sequences kept (>= 2 tokens)")
    split = int(train_split_ratio * len(all_token_ids))
    train_ds = TextDataset(all_token_ids[:split])
    val_ds   = TextDataset(all_token_ids[split:])
else:
    print(f"⚡ Lazy tokenization mode (no pre-tokenization)")
    split = int(train_split_ratio * len(texts))
    train_ds = LazyTextDataset(texts[:split], tokenizer, block_size)
    val_ds   = LazyTextDataset(texts[split:], tokenizer, block_size)

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def make_collate_fn(current_block_size):
    """Return a collate_fn that pads/truncates to current_block_size."""
    max_len = current_block_size - 1
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        xs, ys = zip(*batch)
        xs = [x[:max_len] if len(x) > max_len else x for x in xs]
        ys = [y[:max_len] if len(y) > max_len else y for y in ys]
        xs = [torch.nn.functional.pad(x, (0, max_len - len(x)), value=pad_id) for x in xs]
        ys = [torch.nn.functional.pad(y, (0, max_len - len(y)), value=pad_id) for y in ys]
        return torch.stack(xs), torch.stack(ys)

    return collate_fn


def get_current_block_size(step, schedule, max_block_size):
    """Return the block_size for a given optimizer step based on the progressive schedule."""
    if not schedule:
        return max_block_size
    current = schedule[0][1]
    for start_step, size in schedule:
        if step >= start_step:
            current = size
    return min(current, max_block_size)


def compute_validation_loss(model, val_loader, loss_fn, device, max_batches=50):
    """Calcule la loss de validation sur un sous-ensemble du loader pour aller plus vite."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for xb_val, yb_val in val_loader:
            if num_batches >= max_batches:
                break
            xb_val, yb_val = xb_val.to(device), yb_val.to(device)
            logits = model(xb_val).logits
            B, T, C = logits.shape
            total_loss += loss_fn(logits.view(B * T, C), yb_val.view(B * T)).item()
            num_batches += 1
    return total_loss / num_batches

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
    max_new_tokens=64,
    min_new_tokens=10,
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
        with torch.cuda.amp.autocast(enabled=False):
            past_key_values = None
            for step in range(max_new_tokens):
                if past_key_values is None:
                    input_ids = idx[:, -block_size:]
                else:
                    input_ids = idx[:, -1:]

                outputs = gen_model(input_ids, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :].float()
                past_key_values = outputs.past_key_values

                if temperature != 1.0:
                    logits = logits / temperature

                if eos_id is not None and step < min_new_tokens:
                    logits[:, eos_id] = -float("inf")

                logits = top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p
                )

                probs = torch.softmax(logits, dim=-1)

                if not torch.isfinite(probs).all():
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

                probs_sum = probs.sum(dim=-1, keepdim=True)

                if (probs_sum <= 0).any():
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



initial_block_size = get_current_block_size(0, progressive_schedule, block_size)
current_block_size = initial_block_size
collate_fn = make_collate_fn(current_block_size)
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
    use_gradient_checkpointing=use_gradient_checkpointing,
    num_prelude_layers=num_prelude_layers,
    num_coda_layers=num_coda_layers,
    num_recurrent_steps=num_recurrent_steps,
    use_lti_injection=use_lti_injection,
    use_act_halting=use_act_halting,
    act_halt_threshold=act_halt_threshold,
    depth_lora_rank=depth_lora_rank,
)

model = MiniGPTForCausalLM(model_config).to(device)
#model.resize_token_embeddings(len(tokenizer))
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Afficher les stats détaillées du modèle
model_stats = model.count_parameters()

def warmup_then_cosine(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
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
if progressive_schedule:
    stages = " → ".join(f"{s}@step{s_}" for s_, s in progressive_schedule)
    print(f"   - Progressive resizing: {stages}")
effective_batch = batch_size * grad_accum_steps
print(f"\n🔬 STLM Techniques:")
print(f"   - Weight sharing: {weight_sharing.upper()}")
print(f"   - Position encoding: {'RoPE' if use_rope else 'Learned'}")
print(f"   - FFN activation: SwiGLU")
if weight_sharing == "recurrent_depth":
    print(f"\n🔄 Recurrent-Depth Transformer:")
    print(f"   - Prelude layers: {num_prelude_layers}")
    print(f"   - Recurrent steps (T): {num_recurrent_steps}")
    print(f"   - Coda layers: {num_coda_layers}")
    print(f"   - LTI injection: {'ON' if use_lti_injection else 'OFF'}")
    print(f"   - ACT halting: {'ON (threshold={act_halt_threshold})' if use_act_halting else 'OFF'}")
    print(f"   - ACT loss weight: {act_loss_weight}")
    print(f"   - Depth LoRA rank: {depth_lora_rank}")
    rdt_extra = model_stats.get('rdt_extra', 0)
    if rdt_extra:
        print(f"   - RDT extra params (LTI+LoRA+ACT): {human_readable(rdt_extra)}")
print(f"\n⚡ Training:")
print(f"   - Micro-batch: {batch_size}")
print(f"   - Gradient accumulation: {grad_accum_steps} steps")
print(f"   - Effective batch: {effective_batch}")
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
    eps=1e-8,
    fused=True
)

loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id,
    label_smoothing=0.02
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

scheduler = warmup_then_cosine(optimizer, warmup_steps=warmup, total_steps=total_steps)

if override_lr is not None:
    # When overriding LR, rebuild the scheduler so the cosine decay runs
    # from global_step to total_steps with the new base LR.
    override_lr = float(override_lr)
    for pg in optimizer.param_groups:
        pg["lr"] = override_lr
        pg["initial_lr"] = override_lr
    remaining_steps = total_steps - global_step
    scheduler = warmup_then_cosine(optimizer, warmup_steps=0, total_steps=remaining_steps)
    print(f"🔧 Override LR actif -> LR={override_lr}, cosine sur {remaining_steps} steps restants")
elif scheduler_state_dict is not None:
    scheduler.load_state_dict(scheduler_state_dict)

trackio.init(
    project="mini-gpt-1511-v5",
    name=f"mini-gpt_{config['model']['embed_dim']}d_{config['model']['depth']}L",
    config=config,
    resume="allow"
)

scaler = torch.amp.GradScaler("cuda")
model = torch.compile(model)

epochs_without_improvement = 0
patience = 10

for epoch in range(start_epoch, num_epochs):
    print(f"\n{'='*70}")
    print(f"[{now()}] Epoch {epoch+1}/{num_epochs}  (block_size={current_block_size})")
    print(f"{'='*70}")
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0

    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)

        with torch.amp.autocast("cuda"):
            logits = model(xb).logits
            B, T, C = logits.shape
            loss = loss_fn(logits.view(B*T, C), yb.view(B*T))
            if weight_sharing == "recurrent_depth" and use_act_halting:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                loss = loss + act_loss_weight * raw_model.act_loss
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        running_loss += loss.item()

        is_accum_step = (i + 1) % grad_accum_steps == 0
        is_last_batch = (i + 1) == len(train_loader)

        if is_accum_step or is_last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            if global_step % 50 == 0:
                trackio.log(
                    {
                        "train/loss": running_loss,
                        "epoch": epoch + 1,
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            if global_step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"[{now()}] [Epoch {epoch+1} | Step {global_step}] train_loss={running_loss:.4f} | LR={current_lr:.2e}")

            running_loss = 0.0

            new_bs = get_current_block_size(global_step, progressive_schedule, block_size)
            if new_bs != current_block_size:
                current_block_size = new_bs
                collate_fn = make_collate_fn(current_block_size)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                print(f"📐 Progressive resizing @ step {global_step}: block_size → {current_block_size}")
                break

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
                    temperature=0.7,
                    top_k=0,
                    top_p=0.9
                )
                print(f"[{now()}] Exemple génération v2 (suite de l'invite):\n{gen_text}")
                if not gen_text:
                    print(f"[DEBUG] gen_tokens v2 (len={len(gen_tokens)}): {gen_tokens}")
                    print(f"[DEBUG] gen_text_raw v2: {gen_text_raw}")

                model.train()
trackio.finish()
