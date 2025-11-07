import torch
import torch.nn as nn
import torch.nn.functional as F




class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = dropout 

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,                               # ← clé
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(attn)

# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim, heads):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.heads = heads
#         self.head_dim = embed_dim // heads
#         self.qkv = nn.Linear(embed_dim, embed_dim * 3)
#         self.out = nn.Linear(embed_dim, embed_dim)

#     def forward(self, x, mask=None):
#         B, T, C = x.size()
#         qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2,0,3,1,4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5) 
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#         weights = F.softmax(scores, dim=-1)
#         attn = weights @ v
#         attn = attn.transpose(1,2).contiguous().view(B,T,C)
#         return self.out(attn)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1, hidden_dim = 512):
        super().__init__()
        self.attn = SelfAttention(embed_dim, heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim=256, depth=8, heads=8, dropout = 0.1, hidden_dim = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads, dropout, hidden_dim) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
