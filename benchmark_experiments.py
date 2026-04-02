import os
import json
import torch
import csv
import time
from datetime import datetime
from model.configuration import MiniGPTConfig
from model.modeling_minigpt import MiniGPTForCausalLM
from dataset.text_dataset import TextDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

# --- Experiment Settings ---
EXPERIMENTS = [
    {"name": "Baseline (Vanilla)", "weight_sharing": "none"},
    {"name": "STLM (Shared FFN)", "weight_sharing": "ffn"},
    {"name": "STLM (Universal)", "weight_sharing": "full"},
]

COMMON_CONFIG = {
    "embed_dim": 128,
    "depth": 4,
    "heads": 4,
    "block_size": 256,
    "dropout": 0.1,
    "hidden_dim": 1024,
    "use_rope": True,
    "use_gradient_checkpointing": False
}

TRAIN_EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
MAX_TEXTS = 1000
EVAL_EVERY_STEPS = 50
RESULT_FILE = "experiments/results.csv"

# --- Helper Functions ---

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total

def run_experiment(exp_name, weight_sharing):
    print(f"\n🚀 Running Experiment: {exp_name}")
    print(f"   Weight Sharing: {weight_sharing}")
    
    # 1. Setup Model
    config = MiniGPTConfig(
        vocab_size=32000, # Approximate, adjusted later
        weight_sharing=weight_sharing,
        **COMMON_CONFIG
    )
    
    # 2. Setup Data (simplified)
    # Using a small mock dataset or real one if tokenized
    # For this script we assume the real environment setup
    TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "camembert-base")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except:
        print("Tokenizer not found, using dummy.")
        return None

    config.vocab_size = len(tokenizer)
    model = MiniGPTForCausalLM(config).cuda()
    
    params = count_params(model)
    print(f"   Parameters: {params:,}")
    
    # 3. Setup Loop (Minimal)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Load Real Data Subset
    from datasets import load_dataset
    dataset_name = os.getenv("DATASET_NAME", "iproskurina/TinyStories-French")
    try:
        ds = load_dataset(dataset_name, split=f"train[:{MAX_TEXTS}]")
        texts = ds[os.getenv("DATASET_KEY", "text")]
    except:
         print("   ⚠️ Dataset unavailable. Using random data for throughput test.")
         texts = ["Exemple de texte pour le benchmark." * 20] * 100
    
    # Run Training
    # Create collate function with tokenizer access
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    def collate_fn(batch):
        from torch.nn.utils.rnn import pad_sequence
        xs, ys = zip(*batch)
        # Pad to longest in batch
        xs_pad = pad_sequence(xs, batch_first=True, padding_value=pad_id)
        ys_pad = pad_sequence(ys, batch_first=True, padding_value=pad_id)
        return xs_pad, ys_pad

    # Simple Dataset Wrapper for resizing
    # Split into Train/Val
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    train_ds = TextDataset(train_texts, tokenizer, COMMON_CONFIG["block_size"])
    val_ds = TextDataset(val_texts, tokenizer, COMMON_CONFIG["block_size"])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model.train()
    start_time = time.time()
    total_steps = 0
    
    import subprocess
    
    def get_gpu_metrics():
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,nounits,noheader"],
                encoding="utf-8"
            )
            util, mem = result.strip().split(",")
            return float(util), float(mem)
        except:
            return 0.0, 0.0

    # ... loop ...
    gpu_utils = []
    gpu_mems = []

    # Run Training
    for epoch in range(TRAIN_EPOCHS):
        for xb, yb in train_loader:
            xb, yb = xb.cuda(), yb.cuda()
            
            # Ensure shape compatibility
            if xb.shape[1] > COMMON_CONFIG["block_size"]:
                xb = xb[:, :COMMON_CONFIG["block_size"]]
                yb = yb[:, :COMMON_CONFIG["block_size"]]

            optimizer.zero_grad()
            outputs = model(xb, labels=yb)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_steps += 1
            
            # Monitor GPU every 10 steps to avoid slowing down too much
            if total_steps % 10 == 0:
                u, m = get_gpu_metrics()
                gpu_utils.append(u)
                gpu_mems.append(m)
                print(f"     Step {total_steps}: Train Loss {loss.item():.4f} | GPU: {u}% Mem: {m}MB", end="\r")

    total_time = time.time() - start_time
    throughput = (total_steps * BATCH_SIZE) / total_time
    
    avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
    avg_gpu_mem = sum(gpu_mems) / len(gpu_mems) if gpu_mems else 0
    
    # Validation Loop
    print(f"\n   Running Validation...")
    model.eval()
    val_loss_accum = 0.0
    val_steps = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.cuda(), yb.cuda()
            if xb.shape[1] > COMMON_CONFIG["block_size"]:
                xb = xb[:, :COMMON_CONFIG["block_size"]]
                yb = yb[:, :COMMON_CONFIG["block_size"]]
            
            outputs = model(xb, labels=yb)
            val_loss_accum += outputs.loss.item()
            val_steps += 1
            
            
    final_val_loss = val_loss_accum / val_steps if val_steps > 0 else float('inf')
    
    # Peak Memory (keep as separate metric)
    max_memory = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    
    # Generation Sample
    print(f"   generating sample...")
    prompt = "Il était une fois"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        # Generate 50 tokens
        output_ids = model.generate(
            inputs.input_ids, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"   📝 Generated: {generated_text[:100]}...") # Print first 100 chars
    
    print(f"   ✅ Done. Val Loss: {final_val_loss:.4f}, Speed: {throughput:.2f} samples/s")
    
    return {
        "Experiment": exp_name,
        "Weight Sharing": weight_sharing,
        "Parameters": params,
        "Val Loss": final_val_loss,
        "Samples/Sec": throughput,
        "Peak Memory (MB)": max_memory,
        "Sample": generated_text,
        "Epochs": TRAIN_EPOCHS,
        "Batch Size": BATCH_SIZE,
        "LR": LEARNING_RATE,
        "Embed Dim": COMMON_CONFIG["embed_dim"],
        "Depth": COMMON_CONFIG["depth"],
        "Heads": COMMON_CONFIG["heads"],
        "Block Size": COMMON_CONFIG["block_size"],
        "Dropout": COMMON_CONFIG["dropout"],
        "Hidden Dim": COMMON_CONFIG["hidden_dim"],
        "Use RoPE": COMMON_CONFIG["use_rope"],
        "Gradient Checkpointing": COMMON_CONFIG["use_gradient_checkpointing"],
        "Avg GPU Util (%)": f"{avg_gpu_util:.2f}",
        "Avg GPU Mem (MB)": f"{avg_gpu_mem:.2f}"
    }




# --- Main ---
if __name__ == "__main__":
    os.makedirs("experiments", exist_ok=True)
    
    results = []
    
    for exp in EXPERIMENTS:
        res = run_experiment(exp["name"], exp["weight_sharing"])
        if res:
            results.append(res)
            
    # Save Results
    # Save Results
    keys = results[0].keys()
    
    file_exists = os.path.isfile(RESULT_FILE)
    
    with open(RESULT_FILE, "a", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            dict_writer.writeheader()
        dict_writer.writerows(results)
        
    print(f"\n📄 Results saved to {RESULT_FILE}")
