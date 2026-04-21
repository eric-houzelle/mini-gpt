from torch.utils.data import Dataset
import torch
import os
import json
import hashlib
import multiprocessing
from functools import partial
from tqdm import tqdm


class TextDataset(Dataset):
    """LM dataset from pre-tokenized sequences."""

    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        tokens = self.token_ids[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class LazyTextDataset(Dataset):
    """LM dataset that tokenizes on-the-fly (no pre-tokenization needed)."""

    def __init__(self, texts, tokenizer, block_size):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=self.block_size,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        if len(ids) < 2:
            ids = [0, 0]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def _tokenize_worker(args):
    """Worker function for parallel tokenization.

    Each worker gets a chunk of texts, tokenizes them, splits into
    block_size windows, and writes to its own temp file.
    Returns (temp_file_path, num_sequences).
    """
    worker_id, texts_chunk, tokenizer_name, block_size, batch_size, cache_dir = args
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    tmp_path = os.path.join(cache_dir, f"_worker_{worker_id}.jsonl")
    count = 0

    with open(tmp_path, "w") as f:
        for i in range(0, len(texts_chunk), batch_size):
            encoded = tok(
                texts_chunk[i : i + batch_size],
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            for ids in encoded:
                for start in range(0, len(ids), block_size):
                    chunk = ids[start : start + block_size]
                    if len(chunk) >= 2:
                        f.write(json.dumps(chunk) + "\n")
                        count += 1

    return tmp_path, count


def pretokenize_cached(texts, tokenizer, block_size, cache_dir="cache", batch_size=10_000):
    """Tokenize in parallel across all CPUs and stream to a JSONL cache file.

    On subsequent calls, loads from cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha256(
        f"{len(texts)}_{block_size}_{tokenizer.name_or_path}".encode()
    ).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"tokens_{key}.jsonl")

    if os.path.exists(cache_path):
        print(f"♻️  Loading cached tokens from {cache_path}")
        all_ids = []
        with open(cache_path, "r") as f:
            for line in f:
                all_ids.append(json.loads(line))
        return all_ids

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    chunk_size = (len(texts) + num_workers - 1) // num_workers

    print(f"⚡ Parallel tokenization: {num_workers} workers, "
          f"{len(texts):,} texts, batch_size={batch_size}")

    worker_args = []
    for w in range(num_workers):
        start = w * chunk_size
        end = min(start + chunk_size, len(texts))
        if start >= len(texts):
            break
        worker_args.append((
            w, texts[start:end], tokenizer.name_or_path,
            block_size, batch_size, cache_dir,
        ))

    with multiprocessing.Pool(len(worker_args)) as pool:
        results = list(tqdm(
            pool.imap(_tokenize_worker, worker_args),
            total=len(worker_args),
            desc="Tokenizing (workers)",
            unit="worker",
        ))

    # Merge worker outputs into final cache (preserves original order)
    tmp_path = cache_path + ".tmp"
    total_count = 0
    all_ids = []

    print("📎 Merging worker outputs...")
    with open(tmp_path, "w") as out:
        for worker_file, count in results:
            total_count += count
            with open(worker_file, "r") as inp:
                for line in inp:
                    out.write(line)
                    all_ids.append(json.loads(line))
            os.remove(worker_file)

    os.replace(tmp_path, cache_path)
    print(f"💾 Cached {total_count:,} sequences → {cache_path}")
    return all_ids
