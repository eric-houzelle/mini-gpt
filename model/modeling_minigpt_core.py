import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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

    def forward(self, input_ids):
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

        return self.ln_f(x)
