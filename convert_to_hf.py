"""
Script pour convertir un checkpoint PyTorch en modèle Hugging Face.

Usage:
    python convert_to_hf.py --checkpoint checkpoints/best_miniGPT.pt --output_dir ./minigpt-hf --tokenizer camembert-base
"""
import argparse
import json
import os
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import MiniGPTForCausalLM, MiniGPTConfig


def load_config_from_checkpoint(checkpoint_path, config_path="config.json"):
    """Charge la configuration depuis le checkpoint ou le fichier config.json."""
    # Essayer de charger depuis config.json
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        model_config = config_dict.get("model", {})
        return {
            "vocab_size": None,  # Sera déterminé depuis le tokenizer
            "block_size": model_config.get("block_size", 256),
            "embed_dim": model_config.get("embed_dim", 256),
            "depth": model_config.get("depth", 8),
            "heads": model_config.get("heads", 8),
            "dropout": model_config.get("dropout", 0.1),
            "hidden_dim": model_config.get("hidden_dim", 512),
            "weight_sharing": model_config.get("weight_sharing", "none"),
            "use_rope": model_config.get("use_rope", True),
            "use_gradient_checkpointing": model_config.get("use_gradient_checkpointing", False),
        }
    
    # Si pas de config.json, utiliser des valeurs par défaut
    return {
        "vocab_size": None,
        "block_size": 256,
        "embed_dim": 256,
        "depth": 8,
        "heads": 8,
        "dropout": 0.1,
        "hidden_dim": 512,
        "weight_sharing": "none",
        "use_rope": True,
        "use_gradient_checkpointing": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Convertir un checkpoint MiniGPT vers le format Hugging Face")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Chemin vers le checkpoint PyTorch (.pt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Répertoire de sortie pour le modèle Hugging Face"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="camembert-base",
        help="Nom du tokenizer Hugging Face à utiliser"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Chemin vers le fichier config.json"
    )
    
    args = parser.parse_args()
    
    print(f"🔄 Conversion du checkpoint vers le format Hugging Face...")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Sortie: {args.output_dir}")
    print(f"   Tokenizer: {args.tokenizer}")
    
    # Charger le tokenizer pour obtenir vocab_size
    print(f"\n📥 Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    vocab_size = len(tokenizer)
    print(f"   Vocab size: {vocab_size}")
    
    # Charger la configuration
    print(f"\n📋 Chargement de la configuration...")
    config_dict = load_config_from_checkpoint(args.checkpoint, args.config)
    config_dict["vocab_size"] = vocab_size
    
    # Créer la configuration Hugging Face
    hf_config = MiniGPTConfig(**config_dict)
    print(f"   Configuration créée:")
    print(f"     - Embed dim: {hf_config.embed_dim}")
    print(f"     - Depth: {hf_config.depth}")
    print(f"     - Heads: {hf_config.heads}")
    print(f"     - Block size: {hf_config.block_size}")
    print(f"     - Weight sharing: {hf_config.weight_sharing}")
    print(f"     - Use RoPE: {hf_config.use_rope}")
    
    # Créer le modèle
    print(f"\n🏗️  Création du modèle...")
    model = MiniGPTForCausalLM(hf_config)
    
    # Charger les poids du checkpoint
    print(f"\n📦 Chargement des poids depuis le checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Gérer le cas où le modèle a été compilé avec torch.compile
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Charger les poids (strict=False pour gérer les différences potentielles)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"   ⚠️  Clés manquantes: {len(missing_keys)}")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"      - {key}")
        else:
            for key in missing_keys[:10]:
                print(f"      - {key}")
            print(f"      ... et {len(missing_keys) - 10} autres")
    
    if unexpected_keys:
        print(f"   ⚠️  Clés inattendues: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"      - {key}")
        else:
            for key in unexpected_keys[:10]:
                print(f"      - {key}")
            print(f"      ... et {len(unexpected_keys) - 10} autres")
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Sauvegarder le modèle et la configuration
    print(f"\n💾 Sauvegarde du modèle Hugging Face...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Copier les fichiers Python nécessaires pour que le modèle soit utilisable depuis Hugging Face Hub
    print(f"\n📋 Copie des fichiers Python nécessaires...")
    model_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(model_dir)
    
    # Créer le répertoire model dans output_dir
    output_model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(output_model_dir, exist_ok=True)
    
    # Copier les fichiers nécessaires (seulement ceux pour Hugging Face)
    files_to_copy = [
        ("model/configuration.py", "model/configuration.py"),
        ("model/modeling_minigpt_core.py", "model/modeling_minigpt_core.py"),
        ("model/modeling_minigpt.py", "model/modeling_minigpt.py"),
        ("model/model.py", "model/model.py"),  # Pour les dépendances (RoPEEmbedding, SwiGLU, etc.)
    ]
    
    for src_file, dest_file in files_to_copy:
        src_path = os.path.join(project_root, src_file)
        dest_path = os.path.join(args.output_dir, dest_file)
        
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)
            print(f"   ✅ Copié: {dest_file}")
        else:
            print(f"   ⚠️  Fichier non trouvé: {src_path}")
    
    # Créer un __init__.py minimal pour Hugging Face
    init_content = """# Hugging Face model package
from .configuration import MiniGPTConfig
from .modeling_minigpt_core import MiniGPTModel
from .modeling_minigpt import MiniGPTForCausalLM
from .model import RoPEEmbedding, SwiGLU, SelfAttention, TransformerBlock

__all__ = ["MiniGPTConfig", "MiniGPTModel", "MiniGPTForCausalLM"]
"""
    init_path = os.path.join(args.output_dir, "model", "__init__.py")
    with open(init_path, "w", encoding="utf-8") as f:
        f.write(init_content)
    print(f"   ✅ Créé: model/__init__.py")
    
    # Créer un README basique
    readme_content = f"""# MiniGPT

Modèle de langage MiniGPT converti au format Hugging Face.

## Configuration

- Embed dim: {hf_config.embed_dim}
- Depth: {hf_config.depth}
- Heads: {hf_config.heads}
- Block size: {hf_config.block_size}
- Hidden dim: {hf_config.hidden_dim}
- Weight sharing: {hf_config.weight_sharing}
- Use RoPE: {hf_config.use_rope}
- Dropout: {hf_config.dropout}

## Utilisation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{args.output_dir}")
model = AutoModelForCausalLM.from_pretrained("{args.output_dir}")

# Génération
prompt = "Il était une fois"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```
"""
    
    with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"\n✅ Modèle sauvegardé avec succès dans {args.output_dir}")
    print(f"\n📝 Prochaines étapes:")
    print(f"   1. Vérifier que le modèle fonctionne correctement")
    print(f"   2. Utiliser upload_to_hf.py pour uploader sur Hugging Face Hub")


if __name__ == "__main__":
    main()

