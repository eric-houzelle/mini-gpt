import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast

from .model import RoPEEmbedding, RMSNorm, SwiGLU, SelfAttention, TransformerBlock

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
        num_kv_heads = config.num_kv_heads
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
                                 max_seq_len=block_size, use_rope=config.use_rope,
                                 num_kv_heads=num_kv_heads)
                for _ in range(depth)
            ])

        elif self.weight_sharing == "ffn":
            shared_ff = SwiGLU(embed_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, heads, dropout, hidden_dim,
                                 shared_ff=shared_ff, max_seq_len=block_size,
                                 use_rope=config.use_rope, num_kv_heads=num_kv_heads)
                for _ in range(depth)
            ])

        elif self.weight_sharing == "full":
            self.shared_block = TransformerBlock(embed_dim, heads, dropout, hidden_dim,
                                                 max_seq_len=block_size, use_rope=config.use_rope,
                                                 num_kv_heads=num_kv_heads)
            self.blocks = None

        self.ln_f = RMSNorm(embed_dim)
        
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
        return_dict = return_dict if return_dict is not None else True
        use_cache = use_cache if use_cache is not None else False

        B, T = input_ids.shape
        x = self.token_emb(input_ids)

        if self.pos_emb is not None:
            past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            pos = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(pos)

        mask = None

        new_key_values = []

        if self.use_gradient_checkpointing and self.training:
            if self.weight_sharing == "full":
                for _ in range(self.depth):
                    x = checkpoint(self.shared_block.forward_checkpointed, x, mask, use_reentrant=False)
            else:
                for block in self.blocks:
                    x = checkpoint(block.forward_checkpointed, x, mask, use_reentrant=False)
        else:
            if use_cache:
                if self.weight_sharing == "full":
                    for layer_idx in range(self.depth):
                        past_kv = past_key_values[layer_idx] if past_key_values is not None else None
                        x, kv = self.shared_block(x, mask, past_kv=past_kv, use_cache=True)
                        new_key_values.append(kv)
                else:
                    for layer_idx, block in enumerate(self.blocks):
                        past_kv = past_key_values[layer_idx] if past_key_values is not None else None
                        x, kv = block(x, mask, past_kv=past_kv, use_cache=True)
                        new_key_values.append(kv)
            else:
                if self.weight_sharing == "full":
                    for _ in range(self.depth):
                        x = self.shared_block(x, mask)
                else:
                    for block in self.blocks:
                        x = block(x, mask)

        hidden_states = self.ln_f(x)

        present_key_values = tuple(new_key_values) if use_cache else None

        if not return_dict:
            return (hidden_states,)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present_key_values,
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

