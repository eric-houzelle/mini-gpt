#!/usr/bin/env python3
"""
Script de diagnostic pour vérifier la compatibilité d'un checkpoint
avec l'architecture actuelle du modèle.
"""
import torch
import os
import sys
import json

CONFIG_PATH = "config.json"
CHECKPOINT_PATH = "checkpoints/best_miniGPT.pt"

def main():
    print("="*70)
    print("DIAGNOSTIC CHECKPOINT")
    print("="*70)
    
    # Charger la config
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config non trouvée: {CONFIG_PATH}")
        return 1
    
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    num_experts = config["model"].get("num_experts", 1)
    depth = config["model"]["depth"]
    
    print(f"\n📋 Configuration actuelle:")
    print(f"   - Depth: {depth}")
    print(f"   - Num experts: {num_experts}")
    print(f"   - Weight sharing: {config['model'].get('weight_sharing', 'none')}")
    
    # Charger le checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n✅ Pas de checkpoint existant: {CHECKPOINT_PATH}")
        print(f"   → Entraînement from scratch recommandé")
        return 0
    
    print(f"\n📦 Chargement du checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Analyser les clés
    all_keys = set(state_dict.keys())
    
    # Chercher ff_backbone
    ff_backbone_keys = [k for k in all_keys if 'ff_backbone' in k]
    expert_keys = [k for k in all_keys if 'experts.' in k]
    
    print(f"\n🔍 Analyse du checkpoint:")
    print(f"   - Total de clés: {len(all_keys)}")
    print(f"   - ff_backbone: {len(ff_backbone_keys)} clés")
    print(f"   - experts: {len(expert_keys)} clés")
    
    # Détecter les problèmes
    problems = []
    
    if len(ff_backbone_keys) == 0:
        problems.append("❌ CRITIQUE: Aucune clé ff_backbone trouvée!")
        problems.append("   → Ce checkpoint utilise l'ancienne architecture")
        problems.append("   → Le chargement créera des poids ALÉATOIRES pour ff_backbone")
        problems.append("   → Cela CAUSERA une explosion de la loss")
    else:
        print(f"\n✅ ff_backbone présent dans le checkpoint")
        print(f"   Exemples de clés:")
        for key in ff_backbone_keys[:3]:
            print(f"     - {key}")
    
    # Vérifier le nombre d'experts
    if expert_keys:
        # Extraire les IDs d'experts
        expert_ids = set()
        for key in expert_keys:
            # Format: model.blocks.X.experts.Y.ff.w.weight
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'experts' and i + 1 < len(parts):
                    try:
                        expert_ids.add(int(parts[i + 1]))
                    except ValueError:
                        pass
        
        print(f"\n✅ Experts trouvés dans le checkpoint: {sorted(expert_ids)}")
        
        if len(expert_ids) != num_experts:
            problems.append(f"⚠️  WARNING: Checkpoint a {len(expert_ids)} experts, config demande {num_experts}")
            if len(expert_ids) < num_experts:
                problems.append(f"   → Les experts {list(range(len(expert_ids), num_experts))} seront initialisés aléatoirement")
    else:
        if num_experts > 0:
            problems.append(f"⚠️  WARNING: Config demande {num_experts} experts mais aucun dans le checkpoint")
    
    # Afficher les problèmes
    if problems:
        print(f"\n{'='*70}")
        print("PROBLÈMES DÉTECTÉS:")
        print('='*70)
        for problem in problems:
            print(problem)
        
        print(f"\n{'='*70}")
        print("SOLUTIONS:")
        print('='*70)
        print("\n1. RECOMMANDÉ: Recommencer from scratch")
        print(f"   mv {CHECKPOINT_PATH} {CHECKPOINT_PATH}.old")
        print(f"   python train.py")
        
        print("\n2. Si vous voulez garder le checkpoint:")
        print(f"   - Réduire num_experts à ce qui existe dans le checkpoint")
        print(f"   - Accepter que les nouveaux modules soient aléatoires")
        print(f"   - Utiliser un learning rate très faible (1e-5)")
        
        return 1
    else:
        print(f"\n{'='*70}")
        print("✅ CHECKPOINT COMPATIBLE")
        print('='*70)
        print("\n   Le checkpoint semble compatible avec la configuration actuelle.")
        print(f"   Vous pouvez continuer l'entraînement en toute sécurité.")
        
        # Afficher des infos supplémentaires
        if 'epoch' in checkpoint:
            print(f"\n   Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"   Val loss: {checkpoint['val_loss']:.4f}")
        if 'global_step' in checkpoint:
            print(f"   Global step: {checkpoint['global_step']}")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())

