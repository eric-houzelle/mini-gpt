"""
Script de génération amélioré avec contrôle de la qualité.
Utilise temperature, top-k et top-p pour générer du texte cohérent.
"""
import torch
import json
from transformers import CamembertTokenizer
from model.model import MiniGPT
import argparse

def generate_text(model, tokenizer, prompt, max_tokens, temperature, top_k, top_p, device):
    """Génère du texte avec les paramètres donnés."""
    model.eval()
    
    # Encoder le prompt
    if prompt:
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    else:
        tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Générer
    with torch.no_grad():
        output = model.generate(
            tokens, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Décoder
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

def main():
    parser = argparse.ArgumentParser(description='Génération de texte avec miniGPT')
    parser.add_argument('--prompt', type=str, default="", help='Prompt initial')
    parser.add_argument('--tokens', type=int, default=100, help='Nombre de tokens à générer')
    parser.add_argument('--temperature', type=float, default=0.8, help='Température (0.1=conservateur, 1.0=normal, 2.0=créatif)')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling (ex: 50)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p/nucleus sampling (ex: 0.9)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_miniGPT.pt', help='Chemin du checkpoint')
    parser.add_argument('--num_samples', type=int, default=1, help='Nombre de samples à générer')
    
    args = parser.parse_args()
    
    # Charger la config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    
    # Modèle
    model = MiniGPT(
        vocab_size=len(tokenizer),
        block_size=config['model']['block_size'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        dropout=config['model']['dropout'],
        hidden_dim=config['model']['hidden_dim'],
        weight_sharing=config['model']['weight_sharing'],
        use_rope=config['model']['use_rope']
    ).to(device)
    
    # Charger les poids
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n{'='*80}")
    print(f"Modèle chargé depuis: {args.checkpoint}")
    print(f"Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"{'='*80}\n")
    
    # Paramètres de génération
    print(f"Paramètres de génération:")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Top-k: {args.top_k}")
    print(f"  - Top-p: {args.top_p}")
    print(f"  - Tokens: {args.tokens}\n")
    
    # Générer plusieurs samples
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n{'='*80}")
            print(f"Sample {i+1}/{args.num_samples}")
            print(f"{'='*80}")
        
        text = generate_text(
            model, tokenizer, args.prompt, args.tokens,
            args.temperature, args.top_k, args.top_p, device
        )
        
        print(f"\n{text}\n")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

