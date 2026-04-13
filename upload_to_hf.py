"""
Upload un modèle MiniGPT converti sur le Hugging Face Hub.

Usage:
    python upload_to_hf.py --model_dir ./minigpt-hf-sft --repo_id username/minigpt-fr-sft
    python upload_to_hf.py --model_dir ./minigpt-hf-sft --repo_id username/minigpt-fr-sft --private
"""
import argparse
import os
from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    parser = argparse.ArgumentParser(description="Uploader un modèle MiniGPT sur Hugging Face Hub")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Répertoire contenant le modèle HF (sortie de convert_to_hf.py)")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="ID du repository (format: username/model-name)")
    parser.add_argument("--private", action="store_true",
                        help="Créer un repository privé")
    parser.add_argument("--token", type=str, default=None,
                        help="Token Hugging Face (ou variable HF_TOKEN)")
    parser.add_argument("--commit_message", type=str,
                        default="Upload MiniGPT-FR-SFT model",
                        help="Message de commit")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("❌ Erreur: Token Hugging Face requis.")
        print("   Fournissez --token ou définissez HF_TOKEN.")
        print("   Créez un token sur https://huggingface.co/settings/tokens")
        return

    if not os.path.exists(args.model_dir):
        print(f"❌ Le répertoire {args.model_dir} n'existe pas.")
        print(f"   Lancez d'abord : python convert_to_hf.py")
        return

    # Vérifier les fichiers essentiels
    has_weights = (
        os.path.exists(os.path.join(args.model_dir, "model.safetensors"))
        or os.path.exists(os.path.join(args.model_dir, "pytorch_model.bin"))
    )
    has_config = os.path.exists(os.path.join(args.model_dir, "config.json"))
    has_modeling = os.path.exists(os.path.join(args.model_dir, "modeling_minigpt.py"))

    if not has_weights or not has_config or not has_modeling:
        print("⚠️  Fichiers manquants dans le dossier :")
        if not has_weights:
            print("   - model.safetensors ou pytorch_model.bin")
        if not has_config:
            print("   - config.json")
        if not has_modeling:
            print("   - modeling_minigpt.py")
        print("   Lancez d'abord : python convert_to_hf.py")
        return

    print(f"🚀 Upload sur Hugging Face Hub")
    print(f"   Modèle : {args.model_dir}")
    print(f"   Repo   : {args.repo_id}")
    print(f"   Privé  : {args.private}")

    print(f"\n📦 Création du repository...")
    try:
        create_repo(
            repo_id=args.repo_id, token=token,
            private=args.private, exist_ok=True, repo_type="model",
        )
        print(f"   ✅ Repository prêt")
    except Exception as e:
        print(f"   ⚠️  {e}")

    print(f"\n📤 Upload des fichiers...")
    try:
        upload_folder(
            folder_path=args.model_dir,
            repo_id=args.repo_id,
            token=token,
            commit_message=args.commit_message,
            ignore_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store"],
        )
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return

    url = f"https://huggingface.co/{args.repo_id}"
    print(f"\n{'='*60}")
    print(f"✅ Modèle uploadé avec succès !")
    print(f"🔗 {url}")
    print(f"{'='*60}")
    print(f"\n💡 Pour utiliser le modèle :")
    print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{args.repo_id}', trust_remote_code=True)")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{args.repo_id}', trust_remote_code=True)")


if __name__ == "__main__":
    main()

