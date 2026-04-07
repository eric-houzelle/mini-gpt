from torch.utils.data import Dataset
import torch
import os
import json
import hashlib
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


def pretokenize_cached(texts, tokenizer, block_size, cache_dir="cache", batch_size=10_000):
    """Tokenize in batches and stream to a JSONL cache file.

    Uses incremental writes to avoid RAM spikes from torch.save.
    On subsequent calls, loads from cache (~2s).
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

    tmp_path = cache_path + ".tmp"
    count = 0
    all_ids = []

    with open(tmp_path, "w") as f:
        for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", unit="batch"):
            encoded = tokenizer(
                texts[i : i + batch_size],
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            for ids in encoded:
                for start in range(0, len(ids), block_size):
                    chunk = ids[start:start + block_size]
                    if len(chunk) >= 2:
                        f.write(json.dumps(chunk) + "\n")
                        all_ids.append(chunk)
                        count += 1

    os.replace(tmp_path, cache_path)
    print(f"💾 Cached {count} sequences → {cache_path}")
    return all_ids
