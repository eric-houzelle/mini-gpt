# Phase 3 — DPO (Direct Preference Optimization)

## Objectif

Après le pretraining (phase 1) et le SFT (phase 2), le modèle sait générer du texte
en français et répondre dans un format conversationnel. La phase 3 vise à **aligner**
le modèle avec les préférences humaines pour :

- Réduire les hallucinations
- Améliorer la pertinence et la précision des réponses
- Éviter les réponses toxiques ou hors-sujet
- Préférer les réponses concises aux réponses verbeuses

## Pourquoi DPO plutôt que RLHF ?

| Critère | RLHF | DPO |
|---|---|---|
| Complexité | Élevée (reward model + PPO) | Simple (une seule étape) |
| Stabilité | Instable, difficile à tuner | Stable, loss standard |
| Ressources | 2-3x plus de VRAM | Même coût que le SFT |
| Résultats | Référence historique | Comparable voire supérieur |
| Adapté petit modèle | Difficile | **Idéal** |

**DPO est le choix recommandé** pour MiniGPT : il transforme le problème de RL en une
simple classification binaire sur des paires (réponse préférée, réponse rejetée).

## Datasets recommandés

### 1. `argilla/distilabel-intel-orca-dpo-pairs` (recommandé pour commencer)
- ~12K paires de préférences
- Format : `chosen` / `rejected` (conversations complètes)
- Haute qualité, bien structuré
- Inconvénient : en anglais → nécessite traduction ou dataset FR

### 2. `jondurbin/truthy-dpo-v0.1`
- ~1.5K paires axées sur la véracité
- Spécialement conçu pour réduire les hallucinations
- En anglais

### 3. `OpenAssistant/oasst2` (adaptable)
- Contient des votes humains (rank) sur les réponses
- On peut construire des paires chosen/rejected à partir des rangs
- Multilingue (français inclus)

### 4. `MaziyarPanahi/french_instruct_dpo` ⭐
- Paires de préférences en français
- Directement utilisable sans traduction

### 5. Construction manuelle (meilleure qualité)
- Générer 2 réponses par question avec le modèle SFT
- Choisir manuellement (ou via un modèle juge) la meilleure
- 500-2000 paires suffisent pour un petit modèle

## Architecture du pipeline

```
┌──────────────────────────────────────────────────┐
│  Checkpoint SFT (best_miniGPT_sft.pt)            │
│         ↓                                         │
│  Modèle "policy" (π_θ)                           │
│         ↓                                         │
│  DPO Loss = -log σ(β · (log π_θ(y_w|x)          │
│                         - log π_ref(y_w|x)        │
│                         - log π_θ(y_l|x)          │
│                         + log π_ref(y_l|x)))      │
│         ↓                                         │
│  Checkpoint DPO (best_miniGPT_dpo.pt)            │
└──────────────────────────────────────────────────┘

π_θ   = modèle qu'on entraîne
π_ref = copie figée du modèle SFT (reference model)
y_w   = réponse préférée (chosen / winner)
y_l   = réponse rejetée (rejected / loser)
β     = température DPO (0.1 - 0.5, typiquement 0.1)
```

## Implémentation — Fichiers à créer

### 1. `config_dpo.json`

```json
{
  "training": {
    "num_epochs": 3,
    "batch_size": 8,
    "learning_rate": 5e-7,
    "warmup": 100,
    "gradient_accumulation_steps": 8,
    "max_grad_norm": 1.0,
    "beta": 0.1
  },
  "model": {
    "embed_dim": 768,
    "depth": 14,
    "heads": 12,
    "num_kv_heads": 4,
    "block_size": 256,
    "dropout": 0.0,
    "hidden_dim": 3072,
    "weight_sharing": "none",
    "use_rope": true,
    "use_gradient_checkpointing": false
  },
  "data": {
    "max_texts": 50000,
    "train_split_ratio": 0.95
  },
  "datasets": [
    {
      "name": "MaziyarPanahi/french_instruct_dpo",
      "format": "dpo_pairs"
    }
  ]
}
```

### 2. `dataset/dpo_dataset.py`

Le dataset doit fournir pour chaque exemple :
- `prompt` : la question/instruction formatée en ChatML
- `chosen` : tokens de la réponse préférée
- `rejected` : tokens de la réponse rejetée
- `chosen_labels` / `rejected_labels` : labels avec masquage du prompt (-100)

### 3. `train_dpo.py`

Le script d'entraînement suit cette logique :

```python
# 1. Charger le modèle SFT
policy_model = load_sft_checkpoint("best_miniGPT_sft.pt")
ref_model = load_sft_checkpoint("best_miniGPT_sft.pt")
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# 2. Pour chaque batch (prompt, chosen, rejected) :
def dpo_loss(policy, ref, chosen_ids, rejected_ids, labels_chosen, labels_rejected, beta):
    # Log-probs du policy model
    policy_chosen_logps = get_log_probs(policy, chosen_ids, labels_chosen)
    policy_rejected_logps = get_log_probs(policy, rejected_ids, labels_rejected)

    # Log-probs du reference model (pas de gradient)
    with torch.no_grad():
        ref_chosen_logps = get_log_probs(ref, chosen_ids, labels_chosen)
        ref_rejected_logps = get_log_probs(ref, rejected_ids, labels_rejected)

    # DPO loss
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss

def get_log_probs(model, input_ids, labels):
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    # Récupérer les log-probs des tokens cibles (ignorer -100)
    per_token = torch.gather(log_probs[:, :-1], 2, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = (labels[:, 1:] != -100).float()
    return (per_token * mask).sum(dim=-1) / mask.sum(dim=-1)
```

## Hyperparamètres recommandés pour MiniGPT

| Paramètre | Valeur | Justification |
|---|---|---|
| `learning_rate` | **5e-7** | 10-100x plus bas que le SFT |
| `beta` | **0.1** | Standard DPO, augmenter si trop agressif |
| `num_epochs` | **1-3** | Le DPO converge très vite |
| `batch_size` | **8** | Petit car chaque exemple = 2 forward passes |
| `gradient_accumulation` | **8** | Effective batch = 64 |
| `dropout` | **0.0** | Pas de dropout en DPO |
| `warmup` | **50-100 steps** | Court warmup suffisant |
| `max_grad_norm` | **1.0** | Clipping standard |

## Bibliothèque alternative : TRL

La bibliothèque `trl` de HuggingFace fournit un `DPOTrainer` prêt à l'emploi :

```python
from trl import DPOConfig, DPOTrainer

training_args = DPOConfig(
    output_dir="checkpoints/dpo",
    per_device_train_batch_size=8,
    learning_rate=5e-7,
    beta=0.1,
    num_train_epochs=3,
    warmup_steps=100,
    gradient_accumulation_steps=8,
    max_length=256,
    max_prompt_length=128,
)

trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

> **Note :** `trl` s'attend à un modèle HuggingFace standard. Si `MiniGPTForCausalLM`
> est bien enregistré avec `AutoModelForCausalLM`, ça fonctionnera directement.
> Sinon, on implémentera le DPO loop manuellement comme pour le SFT.

## VRAM et performances

Le DPO nécessite **2 copies du modèle** en mémoire (policy + reference) :

- MiniGPT (~100M params) × 2 ≈ **800 Mo** en fp16
- Avec les activations et gradients : **~3-4 Go** total
- Largement faisable sur un GPU 8 Go+

## Ordre des opérations

```
1. ✅ Terminer le pretraining (val_loss la plus basse possible)
2. ✅ Lancer le SFT multi-dataset (french_instruct + French-Alpaca)
3. 🔲 Évaluer le modèle SFT (chat.py) — vérifier le format et la qualité
4. 🔲 Préparer le dataset DPO (télécharger ou générer des paires)
5. 🔲 Créer train_dpo.py + dataset/dpo_dataset.py + config_dpo.json
6. 🔲 Lancer le DPO (1-3 epochs, surveiller la loss)
7. 🔲 Évaluer le modèle final (chat.py avec le checkpoint DPO)
8. 🔲 Publier sur HuggingFace
```

## Métriques à surveiller pendant le DPO

- **DPO loss** : doit descendre, typiquement de ~0.69 (random) vers ~0.3-0.4
- **Accuracy** : % de fois où le modèle préfère `chosen` à `rejected` → viser > 70%
- **Reward margin** : écart moyen entre chosen/rejected rewards → doit augmenter
- **KL divergence** : écart avec le ref model → ne doit pas exploser (< 5.0)

Si la KL explose, augmenter `beta` ou réduire le `learning_rate`.
