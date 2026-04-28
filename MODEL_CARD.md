---
language:
  - fr
license: apache-2.0
tags:
  - text-generation
  - french
  - gpt
  - causal-lm
  - sft
  - chatml
  - custom-architecture
  - transformer
  - rope
  - swiglu
  - rmsnorm
  - gqa
  - recurrent-depth
library_name: transformers
pipeline_tag: text-generation
model-index:
  - name: Klovis-144M-french
    results: []
---

# Klovis-144M — French Language Model

A **144M-parameter** French language model **fully designed, implemented, and trained** by [Eric Houzelle](https://huggingface.co/Houzeric). Every component — architecture, training pipeline, and inference engine — was written in PyTorch without relying on any pre-trained weights or third-party model code.

Klovis demonstrates that a single engineer can deliver a complete, modern Transformer with state-of-the-art architectural components, trained end-to-end on **a single NVIDIA L40S GPU** for a total compute budget of approximately **€50**.

---

## Key Facts

| | |
|:--|:--|
| **Parameters** | 144M |
| **Architecture** | Decoder-only Transformer |
| **Language** | French |
| **Tokenizer** | CamemBERT (`camembert-base`, 32k vocab) |
| **Context window** | 256 tokens |
| **Chat format** | ChatML |
| **Training hardware** | 1× NVIDIA L40S |
| **Total training cost** | ~€50 |
| **Pre-training data** | ~26M French texts (Wikipedia FR + FineWeb-2) |
| **SFT data** | 6 curated French conversational datasets |
| **SFT epochs** | 15 |
| **License** | Apache 2.0 |
| **Author** | Eric Houzelle |

---

## Quick Start

### Installation

```bash
pip install transformers torch safetensors sentencepiece
```

### Text Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Klovis-ai/Klovis-144M-french"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

prompt = "La France est un pays"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Conversational Mode (ChatML)

The model was fine-tuned with ChatML formatting for assistant-style interactions:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Klovis-ai/Klovis-144M-french"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

prompt = (
    "<|system|>\n"
    "Tu es un assistant utile et concis. Réponds en français.<|end|>\n"
    "<|user|>\n"
    "Quelle est la capitale de la France ?<|end|>\n"
    "<|assistant|>\n"
)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

### Special Tokens

| Token | Role |
|:------|:-----|
| `<\|system\|>` | Start of system message |
| `<\|user\|>` | Start of user message |
| `<\|assistant\|>` | Start of assistant response |
| `<\|end\|>` | End of turn |

---

## Architecture

Klovis implements a decoder-only Transformer using the same building blocks found in LLaMA, Mistral, and Gemma — scaled down to a compact 144M-parameter footprint:

| Component | Detail |
|:----------|:-------|
| Embedding dim | 768 |
| Transformer layers | 14 |
| Query heads | 12 |
| KV heads (GQA) | 4 |
| FFN hidden dim | 3072 |
| FFN activation | **SwiGLU** |
| Normalization | **RMSNorm** (pre-norm) |
| Position encoding | **RoPE** (Rotary Position Embedding) |
| Weight tying | Input embeddings ↔ output projection |

**Grouped-Query Attention (GQA)**: 12 query heads share 4 KV heads, reducing KV-cache memory by 3× while preserving attention capacity.

### Advanced Feature: Recurrent-Depth Transformer (RDT)

The codebase also implements an experimental Recurrent-Depth Transformer mode (inspired by OpenMythos/Parcae, Prairie et al. 2026), where a single Transformer block is applied iteratively:

```
Input → [Prelude Layers] → [Shared Block × T steps] → [Coda Layers] → Output
```

RDT components:
- **LTI Injection** — Linear Time-Invariant state coupling with guaranteed spectral stability
- **Adaptive Computation Time (ACT)** — learned per-position halting for dynamic compute allocation
- **Depth LoRA** — low-rank adapters per recurrent step for step-wise specialization

---

## Training

### Phase 1 — Pre-training

| | |
|:--|:--|
| **Data** | ~26M French texts |
| **Sources** | [CATIE-AQ/wikipedia_fr_2022](https://huggingface.co/datasets/CATIE-AQ/wikipedia_fr_2022), [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`fra_Latn`) |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.95, weight decay 0.1) |
| **Scheduler** | Linear warmup (2000 steps) → cosine decay |
| **Precision** | Mixed precision (AMP + GradScaler) |
| **Effective batch size** | 128 (32 × 4 gradient accumulation) |
| **Progressive training** | Block size 256 → 512 at step 2000 |
| **Compilation** | `torch.compile` |
| **Label smoothing** | 0.02 |

### Phase 2 — Supervised Fine-Tuning (SFT)

The pre-trained model was fine-tuned on 6 curated French conversational datasets across **15 epochs**, with prompt masking so that only assistant tokens contribute to the loss.

| | |
|:--|:--|
| **Epochs** | 15 |
| **Learning rate** | 2e-5 |
| **Effective batch size** | 128 (32 × 4 gradient accumulation) |
| **Max grad norm** | 1.0 |
| **Chat format** | ChatML |
| **Loss** | Cross-entropy on assistant tokens only |

#### SFT Datasets

| Dataset | Format | Details |
|:--------|:-------|:--------|
| [Houzeric/everyday-conversations](https://huggingface.co/datasets/Houzeric/everyday-conversations) | Flat (user/assistant) | Custom-built |
| [CATIE-AQ/facebook-community-alignment-dataset_french_conversation](https://huggingface.co/datasets/CATIE-AQ/facebook-community-alignment-dataset_french_conversation) | Multi-turn conversations | Community alignment |
| [angeluriot/french_instruct](https://huggingface.co/datasets/angeluriot/french_instruct) | Multi-turn conversations | 40k samples, filtered |
| [jpacifico/French-Alpaca-dataset-Instruct-55K](https://huggingface.co/datasets/jpacifico/French-Alpaca-dataset-Instruct-55K) | Alpaca format | 20k samples |
| [Houzeric/french-prompts-and-questions](https://huggingface.co/datasets/Houzeric/french-prompts-and-questions) | Flat (prompt/answer) | 15k samples, filtered |
| [Houzeric/physics-FR-qa-dataset](https://huggingface.co/datasets/Houzeric/physics-FR-qa-dataset) | Flat (question/answer) | 5k samples, scientific QA |

---

## What to Expect

Klovis is a **technical demonstration** — showing that a single engineer can design, train, and deploy a modern Transformer on a single GPU for under €50.

With 144M parameters, the model is capable of:

- Generating grammatically correct French text
- Following the ChatML conversational format
- Producing coherent responses on simple topics

Known limitations:

- Factual responses are frequently **incorrect or fabricated** (hallucinations)
- Logical reasoning is limited
- Responses can be repetitive or drift off-topic
- Context limited to 256 tokens
- French only

> **This model is a demonstration of what a single developer can achieve with a modern architecture at small scale.** It is not intended to replace larger models for production use.

---

## Technical Details

### Implementation Highlights

- **Custom implementation**: every component (attention, RoPE, RMSNorm, SwiGLU, GQA, training loop, generation) is implemented in PyTorch — no external model code
- **Hugging Face compatible**: inherits from `PreTrainedModel` and `GenerationMixin`, works with `AutoModelForCausalLM`
- **KV-cache inference**: supports incremental decoding for efficient generation
- **Multiple weight-sharing modes**: standard, shared FFN, full sharing, and Recurrent-Depth
- **Streaming chat**: interactive CLI with real-time token-by-token output
- **Monitoring**: integrated with [Trackio](https://github.com/gradio-app/trackio) for live training dashboards

### Source Code

The full source code is available at: [github.com/eric-houzelle/mini-gpt](https://github.com/eric-houzelle/mini-gpt)

---

## License

This model is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

---

**Designed and trained by Eric Houzelle.**
