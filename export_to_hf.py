import json
import torch
from pathlib import Path

# =========================
# CONFIG À ADAPTER
# =========================

CHECKPOINT_PATH = "checkpoints/3.5220.pt"
OUTPUT_DIR = "hf_minigpt_french"

MODEL_CLASS_NAME = "MiniGPTForCausalLM"
MODEL_TYPE = "minigpt"

MODEL_CONFIG = {
    "model_type": MODEL_TYPE,
    "architectures": [MODEL_CLASS_NAME],

    # Tokenizer
    "tokenizer_class": "CamembertTokenizer",
    "tokenizer_name": "camembert-base",

    # Spécial tokens
    "vocab_size": 32005,
    "pad_token_id": 1,
    "bos_token_id": 0,
    "eos_token_id": 2,

    # ⚠️ NOMS ATTENDUS PAR TON MODÈLE
    "embed_dim": 640,
    "depth": 20,
    "heads": 10,
    "hidden_dim": 2560,
    "block_size": 256,

    # Divers
    "dropout": 0.10,
    "use_rope": True,
    "weight_sharing": "ffn",
    "tie_word_embeddings": False
}
# =========================
# IMPORT DE TON MODÈLE
# =========================

from model.modeling_minigpt import MiniGPTForCausalLM
from model.configuration import MiniGPTConfig


# =========================
# EXPORT
# =========================

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📦 Chargement du checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint


    print("🧠 Reconstruction du modèle...")
    config = MiniGPTConfig(**MODEL_CONFIG)
    model = MiniGPTForCausalLM(config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # -------------------------
    # Sauvegarde des poids
    # -------------------------
    print("💾 Sauvegarde de pytorch_model.bin...")
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")

    # -------------------------
    # Sauvegarde config.json
    # -------------------------
    print("🧾 Sauvegarde de config.json...")
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(MODEL_CONFIG, f, indent=2, ensure_ascii=False)

    print("\n✅ Export terminé")
    print("📁 Fichiers générés :")
    print(" - pytorch_model.bin")
    print(" - config.json")
    print("\n⚠️ N'oublie pas de copier manuellement :")
    print(" - modeling_minigpt.py")
    print(" - configuration_minigpt.py")


if __name__ == "__main__":
    main()
