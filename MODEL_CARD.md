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
  - from-scratch
  - transformer
library_name: transformers
pipeline_tag: text-generation
model-index:
  - name: Klovis-144M-french-130426
    results: []
---

# Klovis-144M-french-130426

Un modèle de langage **100% français** de **144M de paramètres**, entraîné entièrement from scratch — du pré-entraînement au fine-tuning conversationnel (SFT).

Klovis est un décodeur Transformer compact conçu pour la génération de texte en français. Il utilise les mêmes briques architecturales que les LLMs modernes (RoPE, RMSNorm, SwiGLU, GQA) dans un format léger et accessible.

## Caractéristiques principales

| | |
|---|---|
| **Paramètres** | 144M |
| **Architecture** | Transformer decoder-only |
| **Langue** | Français |
| **Tokenizer** | CamemBERT (`camembert-base`) |
| **Contexte** | 256 tokens |
| **Format de chat** | ChatML |
| **Licence** | Apache 2.0 |

## Démarrage rapide

### Installation

```bash
pip install transformers torch safetensors sentencepiece
```

### Génération de texte

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Klovis-ai/Klovis-144M-french-130426"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

prompt = "La France est un pays"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=100, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Mode conversationnel (ChatML)

Le modèle a été fine-tuné avec le format ChatML pour les interactions en mode assistant :

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Klovis-ai/Klovis-144M-french-130426"
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

### Tokens spéciaux

| Token | Rôle |
|:------|:-----|
| `<\|system\|>` | Début du message système |
| `<\|user\|>` | Début du message utilisateur |
| `<\|assistant\|>` | Début de la réponse de l'assistant |
| `<\|end\|>` | Fin d'un tour de parole |

---

## Architecture

Klovis reprend les composants des LLMs modernes dans un format compact :

| Composant | Détail |
|:----------|:-------|
| Embedding dim | 768 |
| Couches Transformer | 14 |
| Têtes d'attention (Q) | 12 |
| Têtes KV (GQA) | 4 |
| FFN hidden dim | 3072 |
| Activation FFN | SwiGLU |
| Normalisation | RMSNorm (pré-norm) |
| Position encoding | RoPE (Rotary Position Embedding) |
| Weight tying | Embeddings d'entrée ↔ tête de sortie |

**Grouped-Query Attention (GQA)** : 12 query heads partagent 4 KV heads, réduisant l'empreinte mémoire du KV cache tout en maintenant la capacité d'attention.

---

## Entraînement

### Phase 1 — Pré-entraînement

Le modèle a été pré-entraîné from scratch sur un large corpus de textes français.

| Hyperparamètre | Valeur |
|:---------------|:-------|
| Données | ~50M textes français |
| Epochs | 25 |
| Batch size | 128 |
| Learning rate | 1.2e-3 |
| Warmup | 5 000 steps |
| Optimiseur | AdamW (β₁=0.9, β₂=0.95, wd=0.1) |
| Scheduler | Warmup linéaire → cosine decay |
| Précision | Mixed precision (AMP) |
| Block size | Progressif : 128 → 256 tokens |
| Label smoothing | 0.02 |

### Phase 2 — Fine-tuning SFT

Le modèle pré-entraîné a ensuite été fine-tuné sur des données conversationnelles françaises avec masquage du prompt (seuls les tokens de l'assistant contribuent à la loss).

| Hyperparamètre | Valeur |
|:---------------|:-------|
| Dataset | [angeluriot/french_instruct](https://huggingface.co/datasets/angeluriot/french_instruct) |
| Conversations | ~275 000 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size effectif | 128 (32 × 4 gradient accumulation) |
| Max grad norm | 1.0 |
| Format | ChatML |

---

## Limitations

- **Modèle compact** (144M paramètres) — conçu comme un modèle de recherche et d'expérimentation, pas comme un remplacement des grands LLMs.
- **Contexte limité** à 256 tokens.
- **Français uniquement** — les performances sur d'autres langues ne sont pas garanties.
- **Peut générer du contenu incorrect, biaisé ou incohérent**, comme tout modèle de langage.
- Non conçu pour des applications critiques ou de production sans supervision humaine.

## Licence

Ce modèle est distribué sous licence [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
