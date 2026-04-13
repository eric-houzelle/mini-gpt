"""
Convertit le checkpoint SFT PyTorch en un dossier Hugging Face complet et autonome.

Le dossier de sortie contient tout le nécessaire pour que n'importe qui puisse
charger le modèle avec :

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("username/minigpt-fr-sft", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("username/minigpt-fr-sft")

Usage:
    python convert_to_hf.py
    python convert_to_hf.py --checkpoint checkpoints/best_miniGPT_sft.pt --output_dir ./minigpt-hf-sft
    python convert_to_hf.py --config config_sft.json --tokenizer camembert-base
"""
import argparse
import json
import os
import shutil
import torch
from pathlib import Path
from safetensors.torch import save_file
from transformers import AutoTokenizer

from model.configuration import MiniGPTConfig
from model.modeling_minigpt import MiniGPTForCausalLM
from dataset.chat_dataset import add_chat_tokens, SPECIAL_TOKENS


def main():
    parser = argparse.ArgumentParser(description="Convertir un checkpoint MiniGPT SFT → format Hugging Face")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_miniGPT_sft.pt",
                        help="Chemin vers le checkpoint SFT (.pt)")
    parser.add_argument("--output_dir", type=str, default="./minigpt-hf-sft",
                        help="Répertoire de sortie pour le modèle HF")
    parser.add_argument("--config", type=str, default="config_sft.json",
                        help="Chemin vers le fichier config JSON (section 'model')")
    parser.add_argument("--tokenizer", type=str, default="camembert-base",
                        help="Nom du tokenizer HF de base")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Charger la config du modèle ──────────────────────────────────
    print("📋 Chargement de la configuration...")
    with open(args.config, "r") as f:
        raw_config = json.load(f)
    model_cfg = raw_config["model"]

    # ── 2. Préparer le tokenizer avec les tokens de chat ────────────────
    print("📥 Préparation du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    add_chat_tokens(tokenizer)

    all_ids = list(tokenizer.get_vocab().values())
    vocab_size = max(len(tokenizer), max(all_ids) + 1) if all_ids else len(tokenizer)
    print(f"   Vocab size (avec tokens de chat): {vocab_size}")

    # ── 3. Construire le modèle ─────────────────────────────────────────
    print("🏗️  Construction du modèle...")
    hf_config = MiniGPTConfig(
        vocab_size=vocab_size,
        block_size=model_cfg["block_size"],
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        heads=model_cfg["heads"],
        num_kv_heads=model_cfg.get("num_kv_heads", model_cfg["heads"]),
        dropout=0.0,
        hidden_dim=model_cfg["hidden_dim"],
        weight_sharing=model_cfg.get("weight_sharing", "none"),
        use_rope=model_cfg.get("use_rope", True),
        use_gradient_checkpointing=False,
    )

    model = MiniGPTForCausalLM(hf_config)

    # ── 4. Charger les poids du checkpoint ──────────────────────────────
    print(f"📦 Chargement des poids depuis {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Filtrer les buffers RoPE (recalculés automatiquement, non-persistent)
    rope_keys = {"rope.cos_cached", "rope.sin_cached", "rope.inv_freq"}
    state_dict = {k: v for k, v in state_dict.items()
                  if not any(rk in k for rk in rope_keys)}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.tie_weights()

    # Les seules clés "manquantes" attendues sont les buffers RoPE non-persistent
    real_missing = [k for k in missing if not any(rk in k for rk in rope_keys)]
    if real_missing:
        print(f"   ⚠️  Clés manquantes: {real_missing[:5]}{'...' if len(real_missing) > 5 else ''}")
    if unexpected:
        print(f"   ⚠️  Clés inattendues: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres: {total_params:,}")

    # ── 5. Sauvegarder en safetensors ───────────────────────────────────
    print("💾 Sauvegarde du modèle (safetensors)...")
    model.save_pretrained(output_dir, safe_serialization=True)

    # ── 6. Sauvegarder le tokenizer ─────────────────────────────────────
    print("💾 Sauvegarde du tokenizer...")
    tokenizer.save_pretrained(output_dir)

    # ── 7. Copier les fichiers Python pour trust_remote_code ────────────
    print("📋 Copie des fichiers Python pour trust_remote_code...")
    project_root = Path(__file__).parent

    files_to_copy = {
        "configuration_minigpt.py": project_root / "model" / "configuration.py",
        "modeling_minigpt.py": None,  # sera généré (version autonome)
    }

    # Copier configuration.py
    shutil.copy2(files_to_copy["configuration_minigpt.py"], output_dir / "configuration_minigpt.py")
    print("   ✅ configuration_minigpt.py")

    # Générer un modeling_minigpt.py autonome (tout dans un seul fichier)
    _write_standalone_modeling(output_dir / "modeling_minigpt.py", model_cfg)
    print("   ✅ modeling_minigpt.py (autonome)")

    # ── 8. Mettre à jour config.json avec auto_map ──────────────────────
    print("🔧 Mise à jour de config.json avec auto_map...")
    config_path = output_dir / "config.json"
    with open(config_path, "r") as f:
        config_json = json.load(f)

    config_json["auto_map"] = {
        "AutoConfig": "configuration_minigpt.MiniGPTConfig",
        "AutoModel": "modeling_minigpt.MiniGPTForCausalLM",
        "AutoModelForCausalLM": "modeling_minigpt.MiniGPTForCausalLM",
    }
    config_json["model_type"] = "minigpt"
    config_json["architectures"] = ["MiniGPTForCausalLM"]
    config_json["tokenizer_class"] = "CamembertTokenizer"

    # Infos utiles pour la model card
    config_json["torch_dtype"] = "float32"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2, ensure_ascii=False)

    # ── 9. Écrire la model card ─────────────────────────────────────────
    print("📝 Génération de la model card (README.md)...")
    _write_model_card(output_dir, hf_config, total_params, args)

    # ── 10. Résumé ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ Export terminé dans : {output_dir.resolve()}")
    print(f"{'='*60}")
    print(f"\n📁 Fichiers générés :")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        unit = "KB" if size < 1_000_000 else "MB"
        val = size / 1024 if unit == "KB" else size / (1024 * 1024)
        print(f"   {f.name:40s} {val:8.1f} {unit}")

    val_loss = checkpoint.get("val_loss", "N/A")
    step = checkpoint.get("global_step", "N/A")
    print(f"\n📊 Checkpoint info: val_loss={val_loss}, global_step={step}")
    print(f"\n🚀 Prochaine étape :")
    print(f"   python upload_to_hf.py --model_dir {args.output_dir} --repo_id VOTRE_USERNAME/minigpt-fr-sft")


def _write_standalone_modeling(path: Path, model_cfg: dict):
    """Écrit un fichier modeling_minigpt.py autonome pour le Hub HF."""
    content = '''"""
MiniGPT — modèle de langage français léger.

Ce fichier est autonome : il contient toutes les classes nécessaires pour
charger le modèle depuis le Hugging Face Hub avec trust_remote_code=True.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from .configuration_minigpt import MiniGPTConfig


class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021)."""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._rebuild_cache(max_seq_len)

    def _rebuild_cache(self, max_seq_len):
        t = torch.arange(max_seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, offset=0):
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, offset:offset + seq_len, :]
        sin = self.sin_cached[:, :, offset:offset + seq_len, :]
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN: (Swish(xW) * xV) W2."""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.v = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w(x)) * self.v(x))


class SelfAttention(nn.Module):
    """Multi-Head Attention with optional Grouped-Query Attention (GQA)."""
    def __init__(self, embed_dim, heads, dropout, max_seq_len=2048, use_rope=True, num_kv_heads=None):
        super().__init__()
        self.num_heads = heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else heads
        self.head_dim = embed_dim // heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPEEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        B, T, C = x.size()
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            offset = past_kv[0].shape[2] if past_kv is not None else 0
            q, k = self.rope(q, k, offset=offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=(past_kv is None),
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out(attn)), new_kv


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1, hidden_dim=512,
                 shared_ff=None, max_seq_len=2048, use_rope=True, num_kv_heads=None):
        super().__init__()
        self.attn = SelfAttention(embed_dim, heads, dropout, max_seq_len=max_seq_len,
                                  use_rope=use_rope, num_kv_heads=num_kv_heads)
        self.ln1 = RMSNorm(embed_dim)
        self.ff = shared_ff if shared_ff is not None else SwiGLU(embed_dim, hidden_dim)
        self.ln2 = RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln1(x), mask, past_kv=past_kv, use_cache=use_cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return (x, new_kv) if use_cache else x


class MiniGPTModel(nn.Module):
    """Core transformer (no LM head)."""
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.use_rope = config.use_rope
        self.block_size = config.block_size
        self.depth = config.depth
        self.weight_sharing = config.weight_sharing

        if not config.use_rope:
            self.pos_emb = nn.Embedding(config.block_size, config.embed_dim)
        else:
            self.pos_emb = None

        if self.weight_sharing == "none":
            self.blocks = nn.ModuleList([
                TransformerBlock(config.embed_dim, config.heads, config.dropout,
                                 config.hidden_dim, max_seq_len=config.block_size,
                                 use_rope=config.use_rope, num_kv_heads=config.num_kv_heads)
                for _ in range(config.depth)
            ])
        elif self.weight_sharing == "ffn":
            shared_ff = SwiGLU(config.embed_dim, config.hidden_dim)
            self.blocks = nn.ModuleList([
                TransformerBlock(config.embed_dim, config.heads, config.dropout,
                                 config.hidden_dim, shared_ff=shared_ff,
                                 max_seq_len=config.block_size, use_rope=config.use_rope,
                                 num_kv_heads=config.num_kv_heads)
                for _ in range(config.depth)
            ])
        elif self.weight_sharing == "full":
            self.shared_block = TransformerBlock(
                config.embed_dim, config.heads, config.dropout, config.hidden_dim,
                max_seq_len=config.block_size, use_rope=config.use_rope,
                num_kv_heads=config.num_kv_heads)
            self.blocks = None

        self.ln_f = RMSNorm(config.embed_dim)

    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, value):
        self.token_emb = value

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else True
        use_cache = use_cache if use_cache is not None else False
        B, T = input_ids.shape
        x = self.token_emb(input_ids)

        if self.pos_emb is not None:
            past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            pos = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(pos)

        new_key_values = []
        if use_cache:
            if self.weight_sharing == "full":
                for layer_idx in range(self.depth):
                    past_kv = past_key_values[layer_idx] if past_key_values is not None else None
                    x, kv = self.shared_block(x, None, past_kv=past_kv, use_cache=True)
                    new_key_values.append(kv)
            else:
                for layer_idx, block in enumerate(self.blocks):
                    past_kv = past_key_values[layer_idx] if past_key_values is not None else None
                    x, kv = block(x, None, past_kv=past_kv, use_cache=True)
                    new_key_values.append(kv)
        else:
            if self.weight_sharing == "full":
                for _ in range(self.depth):
                    x = self.shared_block(x, None)
            else:
                for block in self.blocks:
                    x = block(x, None)

        hidden_states = self.ln_f(x)
        if not return_dict:
            return (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(new_key_values) if use_cache else None,
        )


class MiniGPTForCausalLM(PreTrainedModel, GenerationMixin):
    """MiniGPT with a causal language modeling head."""
    config_class = MiniGPTConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniGPTModel(config)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        self.lm_head.weight = self.model.token_emb.weight

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                past_key_values=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=outputs.past_key_values if return_dict else None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_model_card(output_dir: Path, config: MiniGPTConfig, total_params: int, args):
    """Génère une model card README.md complète."""
    params_m = total_params / 1_000_000
    card = f"""---
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
library_name: transformers
pipeline_tag: text-generation
model-index:
  - name: MiniGPT-FR-SFT
    results: []
---

# MiniGPT-FR-SFT

**Modèle de langage français léger** ({params_m:.0f}M paramètres), entraîné from scratch puis fine-tuné par SFT (Supervised Fine-Tuning) sur des données conversationnelles françaises.

## Caractéristiques

| Propriété | Valeur |
|-----------|--------|
| Paramètres | **{params_m:.0f}M** |
| Architecture | Transformer decoder-only |
| Embedding dim | {config.embed_dim} |
| Couches | {config.depth} |
| Têtes d'attention | {config.heads} (GQA: {config.num_kv_heads} KV heads) |
| FFN hidden dim | {config.hidden_dim} |
| Context length | {config.block_size} tokens |
| Position encoding | RoPE (Rotary Position Embedding) |
| Normalisation | RMSNorm |
| FFN | SwiGLU |
| Tokenizer | CamemBERT (camembert-base) |
| Langue | Français |

## Utilisation rapide

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "VOTRE_USERNAME/minigpt-fr-sft"  # À remplacer

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Génération simple
prompt = "Bonjour, comment ça va ?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Format de conversation (ChatML)

Le modèle a été fine-tuné avec le format ChatML. Pour de meilleurs résultats en mode conversationnel :

```python
prompt = \"\"\"<|system|>
Tu es un assistant utile et concis. Réponds en français.<|end|>
<|user|>
Quelle est la capitale de la France ?<|end|>
<|assistant|>
\"\"\"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=150)
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

### Tokens spéciaux

| Token | Rôle |
|-------|------|
| `<\\|system\\|>` | Début du message système |
| `<\\|user\\|>` | Début du message utilisateur |
| `<\\|assistant\\|>` | Début de la réponse de l'assistant |
| `<\\|end\\|>` | Fin d'un tour de parole |

## Entraînement

### Pré-entraînement
- Données : ~50M textes français
- Epochs : 25
- Batch size : 128
- Learning rate : 1.2e-3
- Warmup : 5000 steps
- Optimiseur : AdamW (betas=0.9/0.95, weight decay=0.1)
- Scheduler : warmup + cosine decay
- Précision mixte (AMP)
- Block size progressif : 128 → 256 tokens
- Pré-tokenisation activée

### Fine-tuning SFT
- Dataset : [angeluriot/french_instruct](https://huggingface.co/datasets/angeluriot/french_instruct) (~275K conversations)
- Learning rate : 2e-5
- Epochs : 3
- Batch size effectif : 32 × 4 = 128 (gradient accumulation)
- Max grad norm : 1.0
- Loss uniquement sur les tokens de l'assistant (prompt masqué)

## Architecture

Le modèle utilise une architecture Transformer decoder-only moderne avec :
- **RoPE** (Rotary Position Embeddings) pour l'encodage positionnel
- **RMSNorm** (pré-normalisation) au lieu de LayerNorm
- **SwiGLU** comme fonction d'activation dans le FFN
- **Grouped-Query Attention** (GQA) avec {config.num_kv_heads} KV heads pour {config.heads} query heads
- **Weight tying** entre les embeddings d'entrée et la tête de sortie

## Limitations

- Modèle de petite taille ({params_m:.0f}M params) — ne rivalisera pas avec les grands LLMs
- Contexte limité à {config.block_size} tokens
- Entraîné principalement sur du français
- Peut générer du contenu incorrect ou biaisé

## Licence

Apache 2.0
"""
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(card)


if __name__ == "__main__":
    main()
