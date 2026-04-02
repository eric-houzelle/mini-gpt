import torch

# Charger le checkpoint corrompu
checkpoint = torch.load('checkpoints/best_miniGPT.pt', map_location='cpu')

# Nettoyer le state_dict
cleaned_state_dict = {}
for key, value in checkpoint['model_state_dict'].items():
    # Retirer le préfixe _orig_mod.
    new_key = key.replace('_orig_mod.', '')
    cleaned_state_dict[new_key] = value

# Remplacer le state_dict
checkpoint['model_state_dict'] = cleaned_state_dict

# Sauvegarder le checkpoint nettoyé
torch.save(checkpoint, 'checkpoints/best_miniGPT_cleaned.pt')

print("✅ Checkpoint cleaned!")
print(f"Original keys: {len(checkpoint['model_state_dict'])} (first: {list(checkpoint['model_state_dict'].keys())[0]})")
