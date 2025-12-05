import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast

from .model import RoPEEmbedding, SwiGLU, SelfAttention, TransformerBlock

from .configuration import MiniGPTConfig


class MiniGPTModel(nn.Module):
    """
    Modèle core MiniGPT — sans tête LM, pure architecture Transformer.
    NE DOIT PAS hériter de PreTrainedModel.
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()

        vocab_size = config.vocab_size
        block_size = config.block_size
        embed_dim = config.embed_dim
        depth = config.depth
        heads = config.heads
        dropout = config.dropout
        hidden_dim = config.hidden_dim

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.use_rope = config.use_rope
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        self.block_size = block_size
        self.depth = depth
        self.weight_sharing = config.weight_sharing

        # Positional embeddings only if not using RoPE
        if not config.use_rope:
            self.pos_emb = nn.Embedding(block_size, embed_dim)
        else:
            self.pos_emb = None

        # Blocks
        if self.weight_sharing == "none":
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, heads, dropout, hidden_dim,
                                 max_seq_len=block_size, use_rope=config.use_rope)
                for _ in range(depth)
            ])

        elif self.weight_sharing == "ffn":
            shared_ff = SwiGLU(embed_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, heads, dropout, hidden_dim,
                                 shared_ff=shared_ff, max_seq_len=block_size,
                                 use_rope=config.use_rope)
                for _ in range(depth)
            ])

        elif self.weight_sharing == "full":
            self.shared_block = TransformerBlock(embed_dim, heads, dropout, hidden_dim,
                                                 max_seq_len=block_size, use_rope=config.use_rope)
            self.blocks = None

        self.ln_f = nn.LayerNorm(embed_dim)
        
    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, value):
        self.token_emb = value
        
    def get_output_embeddings(self):
        return None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        """
        Forward pass du modèle core.
        
        Args:
            input_ids: Tokens d'entrée [batch_size, seq_len]
            attention_mask: Masque d'attention (non utilisé pour l'instant)
            past_key_values: Cache KV pour génération (non supporté pour l'instant)
            use_cache: Si True, retourne past_key_values (non supporté pour l'instant)
            output_attentions: Si True, retourne les attentions (non supporté pour l'instant)
            output_hidden_states: Si True, retourne tous les hidden states (non supporté pour l'instant)
            return_dict: Si True, retourne un BaseModelOutputWithPast, sinon un tuple
        
        Returns:
            BaseModelOutputWithPast si return_dict=True, sinon tuple (hidden_states,)
        """
        return_dict = return_dict if return_dict is not None else True
        
        # Pour l'instant, on ignore ces paramètres (non supportés pour l'instant)
        # On les ignore silencieusement pour la compatibilité avec l'écosystème Hugging Face
        if past_key_values is not None:
            # TODO: Implémenter le support de past_key_values pour la génération efficace
            pass
        if output_attentions:
            # TODO: Implémenter le retour des attentions
            pass
        if output_hidden_states:
            # TODO: Implémenter le retour de tous les hidden states
            pass
        
        B, T = input_ids.shape
        x = self.token_emb(input_ids)

        if self.pos_emb is not None:  # not using RoPE
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(pos)

        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        if self.use_gradient_checkpointing and self.training:
            if self.weight_sharing == "full":
                for _ in range(self.depth):
                    x = checkpoint(self.shared_block.forward_checkpointed, x, mask, use_reentrant=False)
            else:
                for block in self.blocks:
                    x = checkpoint(block.forward_checkpointed, x, mask, use_reentrant=False)
        else:
            if self.weight_sharing == "full":
                for _ in range(self.depth):
                    x = self.shared_block(x, mask)
            else:
                for block in self.blocks:
                    x = block(x, mask)

        hidden_states = self.ln_f(x)
        
        if not return_dict:
            return (hidden_states,)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    def count_parameters(self):
        """Compte le nombre de paramètres selon le type de weight sharing et l’utilisation de RoPE."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        token_emb_params = self.token_emb.weight.numel()
        pos_emb_params = self.pos_emb.weight.numel() if self.pos_emb is not None else 0
        embedding_params = token_emb_params + pos_emb_params

        if self.weight_sharing == "full":
            block_params = sum(p.numel() for p in self.shared_block.parameters())
        else:
            block_params = sum(p.numel() for p in self.blocks.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "embedding": embedding_params,
            "token_emb": token_emb_params,
            "pos_emb": pos_emb_params,
            "blocks": block_params,
            "head": 0,
            "weight_sharing": self.weight_sharing,
            "use_rope": self.use_rope
        }

