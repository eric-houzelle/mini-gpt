from torch.utils.data import Dataset
import torch
import os
import json
import hashlib
import struct
import array as _array
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


def _write_chunk_binary(out_file, chunk):
    """Write a single chunk as [uint32 length][uint32[] token IDs]."""
    out_file.write(struct.pack('<I', len(chunk)))
    out_file.write(_array.array('I', chunk).tobytes())


def _read_binary_chunks(path):
    """Read all token chunks from a binary cache file."""
    chunks = []
    header_size = struct.calcsize('<I')
    with open(path, "rb") as f:
        while True:
            header = f.read(header_size)
            if len(header) < header_size:
                break
            n = struct.unpack('<I', header)[0]
            data = f.read(n * 4)
            a = _array.array('I')
            a.frombytes(data)
            chunks.append(a.tolist())
    return chunks


def _tokenize_worker(args):
    """Worker function for parallel tokenization.

    Reads texts from a shard file on disk (no RAM duplication),
    tokenizes them, and writes token sequences to a binary output file.
    """
    worker_id, shard_path, output_path, tokenizer_name, block_size, batch_size = args
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    count = 0
    batch = []

    with open(output_path, "wb") as out, open(shard_path, "r") as inp:
        for line in inp:
            text = line.rstrip("\n")
            if not text:
                continue
            batch.append(text)

            if len(batch) >= batch_size:
                count += _tokenize_batch(tok, batch, block_size, out)
                batch = []

        if batch:
            count += _tokenize_batch(tok, batch, block_size, out)

    os.remove(shard_path)
    return output_path, count


def _tokenize_batch(tok, batch, block_size, out_file):
    """Tokenize a batch of texts, chunk into block_size, write binary to file."""
    encoded = tok(
        batch,
        add_special_tokens=True,
        truncation=True,
        max_length=block_size * 4,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    count = 0
    for ids in encoded:
        for start in range(0, len(ids), block_size):
            chunk = ids[start : start + block_size]
            if len(chunk) >= 2:
                _write_chunk_binary(out_file, chunk)
                count += 1
    return count


def pretokenize_cached(texts, tokenizer, block_size, cache_dir="cache", batch_size=10_000):
    """Tokenize in parallel across CPUs and stream to a binary cache file.

    Writes text shards to disk first so workers don't duplicate RAM.
    On subsequent calls, loads from cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha256(
        f"{len(texts)}_{block_size}_{tokenizer.name_or_path}".encode()
    ).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"tokens_{key}.bin")
    legacy_path = os.path.join(cache_dir, f"tokens_{key}.jsonl")

    if os.path.exists(cache_path):
        print(f"♻️  Loading cached tokens from {cache_path}")
        return _read_binary_chunks(cache_path)

    if os.path.exists(legacy_path):
        print(f"♻️  Loading cached tokens from {legacy_path} (legacy jsonl)")
        all_ids = []
        with open(legacy_path, "r") as f:
            for line in f:
                all_ids.append(json.loads(line))
        return all_ids

    num_workers = min(max(1, multiprocessing.cpu_count() - 1), 12)
    chunk_size = (len(texts) + num_workers - 1) // num_workers

    print(f"⚡ Writing {num_workers} text shards to disk (freeing RAM)...")
    shard_paths = []
    for w in range(num_workers):
        start = w * chunk_size
        end = min(start + chunk_size, len(texts))
        if start >= len(texts):
            break
        shard_path = os.path.join(cache_dir, f"_shard_{w}.txt")
        max_chars = block_size * 8
        with open(shard_path, "w") as f:
            for t in texts[start:end]:
                f.write(t.replace("\n", " ")[:max_chars] + "\n")
        shard_paths.append(shard_path)

    actual_workers = len(shard_paths)
    del texts  # free the big list before forking

    print(f"⚡ Parallel tokenization: {actual_workers} workers, batch_size={batch_size}")

    worker_args = [
        (w, shard_paths[w],
         os.path.join(cache_dir, f"_worker_{w}.bin"),
         tokenizer.name_or_path, block_size, batch_size)
        for w in range(actual_workers)
    ]

    with multiprocessing.Pool(actual_workers) as pool:
        results = list(tqdm(
            pool.imap(_tokenize_worker, worker_args),
            total=actual_workers,
            desc="Tokenizing (workers)",
            unit="worker",
        ))

    tmp_path = cache_path + ".tmp"
    total_count = 0

    print("📎 Merging worker outputs...")
    with open(tmp_path, "wb") as out:
        for worker_file, count in results:
            total_count += count
            with open(worker_file, "rb") as inp:
                while True:
                    block = inp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            os.remove(worker_file)

    os.replace(tmp_path, cache_path)
    print(f"💾 Cached {total_count:,} sequences → {cache_path}")
    return _read_binary_chunks(cache_path)
