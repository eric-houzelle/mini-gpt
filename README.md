# Mini-GPT — Transformer Language Model

**A complete, production-grade Transformer language model fully implemented in PyTorch.**

Mini-GPT is a research project by **Eric Houzelle**, demonstrating that a modern decoder-only Transformer with state-of-the-art architectural components can be designed, trained, and fine-tuned by a single engineer — on a single GPU, for under €50.

---

## Highlights

- Full Transformer architecture implemented independently — no pre-trained weights, no external model code
- Modern LLM building blocks: **RoPE**, **RMSNorm**, **SwiGLU**, **Grouped-Query Attention (GQA)**
- Experimental **Recurrent-Depth Transformer (RDT)** mode with LTI injection, Adaptive Computation Time (ACT), and per-step LoRA adapters
- Complete two-phase training pipeline: **pre-training** + **supervised fine-tuning (SFT)** with ChatML formatting
- KV-cache inference with streaming chat interface
- Hugging Face-compatible (`AutoModelForCausalLM`, `PreTrainedModel`, `GenerationMixin`)
- Trained end-to-end on a **single NVIDIA L40S** for a total compute cost of approximately **€50**

---

## Architecture

Mini-GPT implements a decoder-only Transformer with the following components, mirroring techniques found in LLaMA, Mistral, and Gemma:

| Component | Implementation |
|:--|:--|
| Self-Attention | Multi-Head Attention with optional **GQA** (configurable Q/KV head ratio) |
| Position Encoding | **Rotary Position Embeddings (RoPE)** — no learned positional embeddings |
| Normalization | **RMSNorm** (pre-norm) — faster than LayerNorm, no bias |
| FFN Activation | **SwiGLU** — gated linear unit with SiLU activation |
| Weight Tying | Input embeddings tied to output projection head |
| Regularization | Dropout, LayerDrop, label smoothing (0.02) |
| Weight Sharing | 4 modes: `none`, `ffn` (shared FFN), `full` (single shared block), `recurrent_depth` (RDT) |

### Recurrent-Depth Transformer (RDT)

An experimental architecture inspired by OpenMythos/Parcae (Prairie et al., 2026), where a single Transformer block is applied iteratively in a recurrent loop:

```
Input → [Prelude Layers] → [Recurrent Block × T steps] → [Coda Layers] → Output
```

RDT-specific features:
- **LTI Injection** — Linear Time-Invariant state coupling: `h(t+1) = A·h(t) + B·e + block(h(t))` with guaranteed spectral stability
- **Adaptive Computation Time (ACT)** — learned per-position halting mechanism for dynamic compute allocation
- **Depth LoRA** — low-rank adapters applied per recurrent step for step-wise behavioral specialization

---

## Model Configurations

### Pre-training (config.json)

| Parameter | Value |
|:--|:--|
| Embedding dim | 1024 |
| Transformer layers | 14 |
| Attention heads (Q / KV) | 16 / 4 (GQA) |
| FFN hidden dim | 4096 |
| Context window | 512 tokens (progressive: 256 → 512) |
| Weight sharing | Recurrent-Depth |
| RDT steps | 3 prelude + 8 recurrent + 3 coda |
| Tokenizer | CamemBERT (`camembert-base`) |

### Fine-Tuning SFT (config_sft.json)

| Parameter | Value |
|:--|:--|
| Embedding dim | 768 |
| Transformer layers | 14 |
| Attention heads (Q / KV) | 12 / 4 (GQA) |
| FFN hidden dim | 3072 |
| Context window | 256 tokens |
| Format | ChatML |
| Learning rate | 2e-5 |
| Epochs | 15 |

SFT training uses prompt masking — only assistant tokens contribute to the loss. Conversation data comes from 6 curated French datasets including custom-built corpora hosted on Hugging Face.

---

## Training Details

### Phase 1 — Pre-training

The model is pre-trained on a large French text corpus (~26M documents from Wikipedia FR and FineWeb-2) with randomly initialized weights.

| | |
|:--|:--|
| Data | ~26M French texts (Wikipedia FR + FineWeb-2 `fra_Latn`) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight decay 0.1) |
| Scheduler | Linear warmup (2000 steps) → cosine decay |
| Precision | Mixed precision (AMP + GradScaler) |
| Gradient accumulation | 4 steps (effective batch 128) |
| Compilation | `torch.compile` |
| Progressive training | Block size 256 → 512 at step 2000 |

### Phase 2 — Supervised Fine-Tuning (SFT)

The pre-trained model is fine-tuned on conversational data in ChatML format across 15 epochs.

| | |
|:--|:--|
| Datasets | 6 French conversational datasets (including custom Hugging Face repos) |
| Format | ChatML with `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|end\|>` tags |
| Loss masking | Only assistant response tokens |
| Epochs | 15 |
| Learning rate | 2e-5 |
| Hardware | Single NVIDIA L40S |
| Total compute cost | ~€50 |

---

## Project Structure

```
mini-gpt-v2/
├── train.py                    # Pre-training script
├── train_sft.py                # Supervised fine-tuning script
├── chat.py                     # Interactive chat interface (streaming)
├── generate.py                 # Text generation from prompt
├── generatev2.py               # Generation with KV-cache
├── config.json                 # Pre-training hyperparameters
├── config_sft.json             # SFT hyperparameters
├── .env                        # Environment variables
│
├── model/
│   ├── configuration.py        # MiniGPTConfig (HF-compatible)
│   ├── model.py                # Core components (RoPE, RMSNorm, SwiGLU, GQA, RDT, ACT)
│   ├── modeling_minigpt_core.py # MiniGPTModel — backbone without LM head
│   └── modeling_minigpt.py     # MiniGPTForCausalLM — full model with generation
│
├── dataset/
│   ├── text_dataset.py         # Pre-training dataset (with pre-tokenization cache)
│   └── chat_dataset.py         # SFT dataset (ChatML formatting + prompt masking)
│
├── export_to_hf.py             # Export to Hugging Face Hub
├── upload_to_hf.py             # Upload model to HF Hub
├── convert_to_hf.py            # Checkpoint conversion utility
├── benchmark_experiments.py    # Benchmarking suite
└── requirements.txt            # Dependencies
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/eric-houzelle/mini-gpt.git
cd mini-gpt
pip install -r requirements.txt
```

### Pre-training

```bash
python train.py
```

Training monitors progress via [Trackio](https://github.com/gradio-app/trackio), with periodic validation loss evaluation and sample generation every 1000 steps. Checkpoints are saved automatically when validation loss improves.

### Fine-Tuning (SFT)

```bash
python train_sft.py
```

### Interactive Chat

```bash
python chat.py
```

Launches a streaming chat session with the SFT-tuned model. Supports temperature, top-p sampling, and debug mode:

```bash
python chat.py --temperature 0.5 --top_p 0.9 --max_tokens 200 --debug
```

### Text Generation

```bash
python generate.py --prompt "La France est un pays" --tokens 100
```

---

## Hugging Face Integration

The model is fully compatible with the Hugging Face ecosystem:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Klovis-ai/Klovis-144M-french-130426", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Klovis-ai/Klovis-144M-french-130426", trust_remote_code=True)

inputs = tokenizer("La France est", return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=100, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Limitations

This is a **demonstration project** — showing that a single engineer can design, train, and deploy a modern Transformer on a single GPU. It is not intended for production use.

- Compact model (~144M parameters) — limited factual knowledge, frequent hallucinations
- Context window limited to 256–512 tokens
- French only
- No RLHF or alignment beyond SFT

---

## License

Apache 2.0 — free to use, modify, and distribute.

---

**Designed and trained by Eric Houzelle.**
