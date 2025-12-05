"""
Script pour uploader un modèle Hugging Face sur le Hugging Face Hub.

Usage:
    python upload_to_hf.py --model_dir ./minigpt-hf --repo_id username/minigpt-fr --private
"""
import argparse
import os
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Uploader un modèle MiniGPT sur Hugging Face Hub")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Répertoire contenant le modèle Hugging Face"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="ID du repository (format: username/model-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Créer un repository privé"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token Hugging Face (ou utiliser HF_TOKEN env var)"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload MiniGPT model",
        help="Message de commit"
    )
    
    args = parser.parse_args()
    
    # Obtenir le token
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("❌ Erreur: Token Hugging Face requis")
        print("   Fournissez --token ou définissez la variable d'environnement HF_TOKEN")
        return
    
    print(f"🚀 Upload du modèle sur Hugging Face Hub...")
    print(f"   Modèle: {args.model_dir}")
    print(f"   Repository: {args.repo_id}")
    print(f"   Privé: {args.private}")
    
    # Vérifier que le modèle existe
    if not os.path.exists(args.model_dir):
        print(f"❌ Erreur: Le répertoire {args.model_dir} n'existe pas")
        return
    
    # Vérifier que les fichiers nécessaires existent
    required_files = ["config.json", "pytorch_model.bin"]
    if not os.path.exists(os.path.join(args.model_dir, "pytorch_model.bin")):
        # Peut-être que c'est un modèle safetensors
        if not os.path.exists(os.path.join(args.model_dir, "model.safetensors")):
            print(f"⚠️  Avertissement: Fichier de modèle non trouvé dans {args.model_dir}")
            print(f"   Vérifiez que le modèle a été correctement converti avec convert_to_hf.py")
    
    # Créer le repository s'il n'existe pas
    print(f"\n📦 Création du repository...")
    try:
        create_repo(
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"   ✅ Repository créé/vérifié")
    except Exception as e:
        print(f"   ⚠️  Erreur lors de la création du repository: {e}")
        print(f"   (Le repository existe peut-être déjà, continuation...)")
    
    # Uploader les fichiers
    print(f"\n📤 Upload des fichiers...")
    try:
        upload_folder(
            folder_path=args.model_dir,
            repo_id=args.repo_id,
            token=token,
            commit_message=args.commit_message,
            ignore_patterns=[".git", "__pycache__", "*.pyc"]
        )
        print(f"   ✅ Upload réussi!")
    except Exception as e:
        print(f"   ❌ Erreur lors de l'upload: {e}")
        return
    
    print(f"\n✅ Modèle uploadé avec succès!")
    print(f"\n🔗 Lien: https://huggingface.co/{args.repo_id}")
    print(f"\n💡 Pour utiliser le modèle:")
    print(f"   from transformers import AutoModelForCausalLM")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{args.repo_id}')")


if __name__ == "__main__":
    main()

