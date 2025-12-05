"""
Modèle MiniGPT pour Hugging Face Transformers.

Ce fichier contient MiniGPTForCausalLM qui est la classe standard
attendue par Hugging Face pour les modèles de génération de texte.

MiniGPTForCausalLM hérite de MiniGPTModel et ajoute uniquement la tête de langage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration import MiniGPTConfig
from .modeling_minigpt_core import MiniGPTModel



class MiniGPTForCausalLM(PreTrainedModel):
    """
    MiniGPT model avec une tête de langage pour la génération de texte.
    
    Cette classe est compatible avec l'écosystème Hugging Face et peut être
    utilisée avec AutoModelForCausalLM une fois enregistrée.
    
    Elle contient :
    - L'enrobage Hugging Face (méthodes standard)
    - La logique LM (tête de prédiction)
    - L'appel au modèle interne (MiniGPTModel)
    """
    config_class = MiniGPTConfig
    base_model_prefix = "model"
    
    def __init__(self, config):
        super().__init__(config)
        
        # Modèle core (architecture sans la tête)
        self.model = MiniGPTModel(config)
        
        # Tête de langage (prédiction de tokens)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Weight tying : partager les poids entre token_emb et lm_head
        self.lm_head.weight = self.model.token_emb.weight
        
        # Post-initialisation
        self.post_init()

    def get_input_embeddings(self):
        """Retourne les embeddings d'entrée."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Définit les embeddings d'entrée."""
        self.model.set_input_embeddings(value)
        # Mettre à jour le weight tying
        self.lm_head.weight = self.model.token_emb.weight

    def get_output_embeddings(self):
        """Retourne la tête de sortie."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Définit la tête de sortie."""
        self.lm_head = new_embeddings
        # Mettre à jour le weight tying
        self.lm_head.weight = self.model.token_emb.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("input_ids doit être fourni")

        # Appel au modèle core
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extraire les hidden states selon le format de retour
        if return_dict:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Appliquer la tête de langage
        logits = self.lm_head(hidden_states)

        # Calculer la loss si labels fournis
        loss = None
        if labels is not None:
            # Shift logits et labels pour l'alignement (prédire le token suivant)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Format de sortie selon return_dict
        if not return_dict:
            output = (logits,)
            if loss is not None:
                return (loss,) + output
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prépare les inputs pour la génération."""
        # Pour l'instant, on ne supporte pas le past_key_values
        # Mais on garde la structure pour compatibilité future
        return {"input_ids": input_ids}
    
    @torch.no_grad()
    def generate(self, input_ids=None, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None, 
                 min_new_tokens=0, eos_token_id=None, **kwargs):
        """
        Génération de texte avec contrôle de la diversité.
        
        Args:
            input_ids: Context initial [batch, seq_len]
            max_new_tokens: Nombre de tokens à générer
            temperature: Contrôle la diversité (0.1=conservateur, 1.0=normal, 2.0=créatif)
            top_k: Garde seulement les k tokens les plus probables
            top_p: Nucleus sampling, garde les tokens dont la somme des probas = p
            min_new_tokens: Génère au moins ce nombre de tokens avant d'autoriser l'arrêt sur eos_token_id
            eos_token_id: Id du token EOS pour stopper la génération (optionnel)
        """
        if input_ids is None:
            raise ValueError("input_ids doit être fourni")
        
        idx = input_ids
        block_size = self.config.block_size
        
        for step in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self.forward(idx_cond).logits
            logits = logits[:, -1, :]
            
            # Appliquer la température
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Retirer les tokens au-delà du seuil top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Garder au moins le premier token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter les valeurs -inf
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Échantillonner
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Éviter un EOS trop tôt
            if eos_token_id is not None and step < min_new_tokens:
                while next_token.item() == eos_token_id:
                    next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

            # Arrêt précoce si EOS après le minimum requis
            if eos_token_id is not None and step >= min_new_tokens and next_token.item() == eos_token_id:
                break
        
        return idx
    
    def count_parameters(self):
        """Délègue au modèle core MiniGPTModel."""
        return self.model.count_parameters()
