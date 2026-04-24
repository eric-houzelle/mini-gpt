"""
Compare tokenizer efficiency on a French corpus sample.

Examples:
    python tools/compare_tokenizers.py \
        --dataset CATIE-AQ/wikipedia_fr_2022 \
        --text-key text \
        --sample-size 10000 \
        --hf-tokenizers camembert-base \
        --train-spm

    python tools/compare_tokenizers.py \
        --text-file corpus_sample.txt \
        --hf-tokenizers camembert-base mistralai/Mistral-Nemo-Base-2407 \
        --train-spm \
        --spm-model-types unigram bpe \
        --spm-vocab-sizes 16000 32000 50000
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

SPECIAL_TOKENS = [
    "<pad>",
    "<s>",
    "</s>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]

WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Metrics:
    name: str
    total_tokens: int
    total_words: int
    total_chars: int
    docs: int
    avg_tokens_per_doc: float
    avg_tokens_per_word: float
    avg_chars_per_token: float
    p50_tokens_per_doc: float
    p95_tokens_per_doc: float
    max_tokens_per_doc: int


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.text_file:
        path = Path(args.text_file)
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    else:
        if not args.dataset:
            raise ValueError("Provide --dataset or --text-file")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("datasets is required for --dataset. Run: pip install -r requirements.txt") from exc
        load_kwargs = {}
        if args.subset:
            load_kwargs["name"] = args.subset
        if args.data_files:
            load_kwargs["data_files"] = args.data_files
        ds = load_dataset(args.dataset, split=args.split, **load_kwargs)
        if args.text_key not in ds.column_names:
            raise ValueError(f"Column '{args.text_key}' not found. Available: {ds.column_names}")
        if args.shuffle:
            ds = ds.shuffle(seed=args.seed)
        ds = ds.select(range(min(args.sample_size, len(ds))))
        texts = [str(x) for x in ds[args.text_key]]

    texts = [t.strip() for t in texts if t and len(t.strip()) >= args.min_chars]
    if args.sample_size and len(texts) > args.sample_size:
        rng = random.Random(args.seed)
        texts = rng.sample(texts, args.sample_size)
    return texts


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(values[lo])
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def compute_metrics(name: str, texts: list[str], encode_fn) -> Metrics:
    token_counts = []
    total_tokens = 0
    total_words = 0
    total_chars = 0

    for text in texts:
        ids = encode_fn(text)
        n_tokens = len(ids)
        n_words = count_words(text)
        n_chars = len(text)

        token_counts.append(n_tokens)
        total_tokens += n_tokens
        total_words += n_words
        total_chars += n_chars

    docs = len(texts)
    return Metrics(
        name=name,
        total_tokens=total_tokens,
        total_words=total_words,
        total_chars=total_chars,
        docs=docs,
        avg_tokens_per_doc=total_tokens / max(docs, 1),
        avg_tokens_per_word=total_tokens / max(total_words, 1),
        avg_chars_per_token=total_chars / max(total_tokens, 1),
        p50_tokens_per_doc=statistics.median(token_counts) if token_counts else 0.0,
        p95_tokens_per_doc=percentile(token_counts, 0.95),
        max_tokens_per_doc=max(token_counts) if token_counts else 0,
    )


def train_sentencepiece(
    texts: list[str],
    args: argparse.Namespace,
    model_type: str,
    vocab_size: int,
) -> Path:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required for --train-spm. Run: pip install -r requirements.txt") from exc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"fr_{model_type}_{vocab_size}"

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        input_path = Path(f.name)
        for text in texts:
            f.write(text.replace("\n", " ") + "\n")

    try:
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=args.character_coverage,
            normalization_rule_name="nfkc",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=",".join(SPECIAL_TOKENS[3:]),
            input_sentence_size=args.spm_input_sentence_size,
            shuffle_input_sentence=True,
            train_extremely_large_corpus=True,
        )
    finally:
        input_path.unlink(missing_ok=True)

    return prefix.with_suffix(".model")


def print_results(results: list[Metrics]) -> None:
    baseline = results[0].total_tokens if results else 0
    headers = [
        "tokenizer",
        "total_tokens",
        "vs_baseline",
        "tok/word",
        "chars/token",
        "avg/doc",
        "p50/doc",
        "p95/doc",
        "max/doc",
    ]
    rows = []
    for m in results:
        saved = 0.0 if baseline == 0 else (1 - (m.total_tokens / baseline)) * 100
        rows.append([
            m.name,
            f"{m.total_tokens:,}",
            f"{saved:+.2f}%",
            f"{m.avg_tokens_per_word:.3f}",
            f"{m.avg_chars_per_token:.3f}",
            f"{m.avg_tokens_per_doc:.1f}",
            f"{m.p50_tokens_per_doc:.0f}",
            f"{m.p95_tokens_per_doc:.0f}",
            f"{m.max_tokens_per_doc:,}",
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tokenizer efficiency on French text.")
    parser.add_argument("--dataset", help="Hugging Face dataset name")
    parser.add_argument("--subset", help="Optional dataset subset/config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--data-files", help="Optional HF data_files argument")
    parser.add_argument("--text-file", help="One text sample per line")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--hf-tokenizers", nargs="*", default=["camembert-base"])
    parser.add_argument("--train-spm", action="store_true")
    parser.add_argument("--spm-model", help="Existing SentencePiece .model to compare")
    parser.add_argument(
        "--spm-vocab-size",
        type=int,
        default=None,
        help="Backward-compatible alias for one SentencePiece vocab size",
    )
    parser.add_argument("--spm-vocab-sizes", nargs="*", type=int, default=[32000])
    parser.add_argument("--spm-model-types", nargs="*", default=["unigram"], choices=["unigram", "bpe"])
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--spm-input-sentence-size", type=int, default=1000000)
    parser.add_argument("--output-dir", default="tokenizers")
    args = parser.parse_args()

    texts = load_texts(args)
    if not texts:
        raise ValueError("No usable texts found")

    print(f"Loaded {len(texts):,} texts")
    print(f"Total chars: {sum(len(t) for t in texts):,}")
    print(f"Total words: {sum(count_words(t) for t in texts):,}\n")

    results: list[Metrics] = []

    for name in args.hf_tokenizers:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for --hf-tokenizers. Run: pip install -r requirements.txt") from exc
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        results.append(compute_metrics(name, texts, lambda text, tok=tok: tok.encode(text, add_special_tokens=False)))

    spm_models = []
    if args.train_spm:
        vocab_sizes = [args.spm_vocab_size] if args.spm_vocab_size else args.spm_vocab_sizes
        for model_type in args.spm_model_types:
            for vocab_size in vocab_sizes:
                model_path = train_sentencepiece(texts, args, model_type=model_type, vocab_size=vocab_size)
                spm_models.append(model_path)
                print(f"Trained SentencePiece model: {model_path}")
    if args.spm_model:
        spm_models.append(Path(args.spm_model))

    for model_path in spm_models:
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError("sentencepiece is required for --spm-model. Run: pip install -r requirements.txt") from exc

        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        results.append(compute_metrics(
            model_path.stem,
            texts,
            lambda text, processor=processor: processor.encode(text, out_type=int),
        ))

    print()
    print_results(results)


if __name__ == "__main__":
    main()
