import os
import csv
import time
import subprocess
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from model.configuration import MiniGPTConfig
from model.modeling_minigpt import MiniGPTForCausalLM
from dataset.text_dataset import TextDataset

# =========================================================
# ENV & GLOBAL CONFIG
# =========================================================
load_dotenv()

DEVICE = "cuda"
RESULT_FILE = "experiments/results_v2.csv"

# Early stopping / convergence
EVAL_EVERY_STEPS = 50
EARLY_STOP_PATIENCE = 10     # nb d'évals sans amélioration
MIN_DELTA = 1e-4
MAX_STEPS = 200_000          # garde-fou

# Training
TRAIN_EPOCHS_MAX = 10_000    # virtuel (on stop avant)
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_TEXTS = 1000

# =========================================================
# EXPERIMENTS
# =========================================================
EXPERIMENTS = [
    {"name": "Baseline", "weight_sharing": "none"},
    {"name": "STLM-FFN", "weight_sharing": "ffn"},
    {"name": "STLM-Universal", "weight_sharing": "full"},
]

COMMON_CONFIG = {
    "embed_dim": 32,
    "depth": 2,
    "heads": 2,
    "hidden_dim": 256,
    "block_size": 256,
    "dropout": 0.1,
    "use_rope": True,
    "use_gradient_checkpointing": False,
}

# =========================================================
# UTILS
# =========================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_gpu_metrics():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        util, mem = out.strip().split(",")
        return float(util), float(mem)
    except:
        return 0.0, 0.0

# =========================================================
# MAIN EXPERIMENT
# =========================================================
def run_experiment(exp_name, weight_sharing):
    print(f"\n🚀 Running {exp_name} | weight_sharing={weight_sharing}")

    # -------------------------
    # Model
    # -------------------------
    config = MiniGPTConfig(
        vocab_size=32000,
        weight_sharing=weight_sharing,
        **COMMON_CONFIG
    )

    tokenizer = AutoTokenizer.from_pretrained(
        os.getenv("TOKENIZER_NAME", "camembert-base")
    )
    config.vocab_size = len(tokenizer)

    model = MiniGPTForCausalLM(config).to(DEVICE)
    params = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Dataset
    # -------------------------
    dataset_name = os.getenv("DATASET_NAME", "iproskurina/TinyStories-French")
    ds = load_dataset(dataset_name, split=f"train[:{MAX_TEXTS}]")
    texts = ds[os.getenv("DATASET_KEY", "text")]

    split = int(0.9 * len(texts))
    train_ds = TextDataset(texts[:split], tokenizer, COMMON_CONFIG["block_size"])
    val_ds = TextDataset(texts[split:], tokenizer, COMMON_CONFIG["block_size"])

    pad_id = tokenizer.pad_token_id or 0

    def collate_fn(batch):
        from torch.nn.utils.rnn import pad_sequence
        xs, ys = zip(*batch)
        return (
            pad_sequence(xs, batch_first=True, padding_value=pad_id),
            pad_sequence(ys, batch_first=True, padding_value=pad_id),
        )

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # -------------------------
    # Tracking
    # -------------------------
    total_steps = 0
    best_val_loss = float("inf")
    best_step = 0
    best_time = 0.0
    best_epoch = 0

    no_improve = 0
    early_stopped = False

    gpu_utils, gpu_mems = [], []

    start_time = time.time()
    model.train()

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(TRAIN_EPOCHS_MAX):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(xb, labels=yb)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_steps += 1

            if total_steps % 10 == 0:
                u, m = get_gpu_metrics()
                gpu_utils.append(u)
                gpu_mems.append(m)

            # -------- Validation --------
            if total_steps % EVAL_EVERY_STEPS == 0:
                model.eval()
                val_loss_sum, n = 0.0, 0

                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                        out = model(vx, labels=vy)
                        val_loss_sum += out.loss.item()
                        n += 1

                val_loss = val_loss_sum / n
                elapsed = time.time() - start_time

                print(
                    f"Step {total_steps:6d} | "
                    f"Val {val_loss:.4f} | "
                    f"Best {best_val_loss:.4f}"
                )

                if val_loss < best_val_loss - MIN_DELTA:
                    best_val_loss = val_loss
                    best_step = total_steps
                    best_time = elapsed
                    best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1

                model.train()

                if no_improve >= EARLY_STOP_PATIENCE:
                    early_stopped = True
                    print("⏹️ Early stopping triggered")
                    break

            if total_steps >= MAX_STEPS:
                break

        if early_stopped or total_steps >= MAX_STEPS:
            break

    # -------------------------
    # Metrics
    # -------------------------
    total_time = time.time() - start_time
    throughput = (total_steps * BATCH_SIZE) / total_time

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()

    avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
    avg_gpu_mem = sum(gpu_mems) / len(gpu_mems) if gpu_mems else 0

    print(f"✅ Done | Best Val {best_val_loss:.4f} @ step {best_step}")

    return {
        "Experiment": exp_name,
        "Weight Sharing": weight_sharing,
        "Parameters": params,

        "Best Val Loss": best_val_loss,
        "Best Step": best_step,
        "Best Time (sec)": round(best_time, 2),
        "Best Epoch": best_epoch,

        "Total Steps": total_steps,
        "Total Time (sec)": round(total_time, 2),
        "Early Stopped": early_stopped,

        "Samples/Sec": round(throughput, 2),
        "Peak Memory (MB)": round(peak_mem, 2),
        "Avg GPU Util (%)": round(avg_gpu_util, 2),

        "Depth": COMMON_CONFIG["depth"],
        "Embed Dim": COMMON_CONFIG["embed_dim"],
        "Heads": COMMON_CONFIG["heads"],
        "Hidden Dim": COMMON_CONFIG["hidden_dim"],
        "Block Size": COMMON_CONFIG["block_size"],
        "Use RoPE": COMMON_CONFIG["use_rope"],
    }

# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    os.makedirs("experiments", exist_ok=True)

    results = []
    for exp in EXPERIMENTS:
        results.append(run_experiment(exp["name"], exp["weight_sharing"]))

    file_exists = os.path.isfile(RESULT_FILE)

    with open(RESULT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())

        # Écrit l'en-tête UNE SEULE FOIS
        if not file_exists:
            writer.writeheader()

        writer.writerows(results)

    print(f"\n📄 Results appended to {RESULT_FILE}")
