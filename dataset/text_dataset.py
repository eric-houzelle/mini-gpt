from torch.utils.data import Dataset
import torch
import os
import hashlib
from tqdm import tqdm


class TextDataset(Dataset):
    """LM dataset from pre-tokenized sequences.

    Args:
        token_ids: list of lists of int (already tokenized, truncated to block_size).
    """
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        tokens = self.token_ids[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def pretokenize(texts, tokenizer, block_size, batch_size=10_000):
    """Tokenize texts in batches with a progress bar.

    Sequences shorter than 2 tokens are discarded (can't form an x/y pair).
    """
    all_ids = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", unit="batch"):
        encoded = tokenizer(
            texts[i : i + batch_size],
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        all_ids.extend(ids for ids in encoded if len(ids) >= 2)
    return all_ids


def pretokenize_cached(texts, tokenizer, block_size, cache_dir="cache"):
    """Tokenize once and cache to disk. Subsequent calls load from cache."""
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha256(
        f"{len(texts)}_{block_size}_{tokenizer.name_or_path}".encode()
    ).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"tokens_{key}.pt")

    if os.path.exists(cache_path):
        print(f"♻️  Loading cached tokens from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    all_ids = pretokenize(texts, tokenizer, block_size)
    torch.save(all_ids, cache_path)
    print(f"💾 Cached {len(all_ids)} sequences → {cache_path}")
    return all_ids
